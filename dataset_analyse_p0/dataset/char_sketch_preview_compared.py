import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.cm as cm
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

# 🌟 引入中轴变换与拓扑骨架提取的核心库
from skimage.morphology import medial_axis

# ==========================================
# ⚙️ 全局配置区
# ==========================================
TARGET_DIR = "alien_tensors_raw"  
NUM_SAMPLES = 5         # 为了保证计算性能与展示效果，抽取5个样本
RESOLUTION = 600        # 高精度光栅化的像素网格分辨率 (600x600)

# ==========================================
# 🖋️ 核心黑科技：全能路径截获笔
# ==========================================
class PathExtractorPen(BasePen):
    """
    不仅拦截每一段曲线以供变色渲染，
    还将整个闭合轮廓拼装成标准 Path，为后续的光栅化做准备。
    """
    def __init__(self, glyphSet):
        super().__init__(glyphSet)
        self.segments = []      # 用于分段彩色渲染
        self.all_pts = []       # 用于光栅化
        self.codes = []         # 用于光栅化
        self.current_pt = (0, 0)
        self.start_pt = (0, 0)

    def _moveTo(self, p):
        self.current_pt = p
        self.start_pt = p
        self.all_pts.append(p)
        self.codes.append(Path.MOVETO)

    def _lineTo(self, p):
        self.segments.append(('line', [self.current_pt, p]))
        self.all_pts.append(p)
        self.codes.append(Path.LINETO)
        self.current_pt = p

    def _curveToOne(self, p1, p2, p3):
        self.segments.append(('cubic', [self.current_pt, p1, p2, p3]))
        self.all_pts.extend([p1, p2, p3])
        self.codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
        self.current_pt = p3

    def _qCurveToOne(self, p1, p2):
        self.segments.append(('quadratic', [self.current_pt, p1, p2]))
        self.all_pts.extend([p1, p2])
        self.codes.extend([Path.CURVE3, Path.CURVE3])
        self.current_pt = p2

    def _closePath(self):
        if self.current_pt != self.start_pt:
            self.segments.append(('line', [self.current_pt, self.start_pt]))
            self.all_pts.append(self.start_pt)
            self.codes.append(Path.LINETO)
        # 强制闭合 Polygon，以便计算“内部”和“外部”
        self.all_pts.append(self.start_pt)
        self.codes.append(Path.CLOSEPOLY)
        self.current_pt = self.start_pt

# ==========================================
# 🚀 降维打击主流水线
# ==========================================
def extract_medial_axis_tensors():
    print(f"🛸 启动 MAT 拓扑降维实验室，扫描目录: {TARGET_DIR}")
    
    font_files = glob.glob(os.path.join(TARGET_DIR, "*.[ot]tf"))
    if not font_files:
        return print(f"❌ 未在 {TARGET_DIR} 找到任何 TTF 或 OTF 文件！")
        
    sample_files = random.sample(font_files, min(NUM_SAMPLES, len(font_files)))
    plt.style.use('dark_background')
    
    # 构建 5行2列 的对比视窗
    fig, axes = plt.subplots(NUM_SAMPLES, 2, figsize=(12, 3.5 * NUM_SAMPLES))
    fig.suptitle("Alien Tensors: Contour vs Weighted Skeleton (MAT)", fontsize=16, color='#00ff41', y=0.98)

    for i, file_path in enumerate(sample_files):
        ax_left = axes[i, 0]
        ax_right = axes[i, 1]
        font_name = os.path.basename(file_path)
        
        try:
            font = TTFont(file_path)
            cmap = font.getBestCmap()
            glyph_set = font.getGlyphSet()
            valid_chars = list(cmap.keys())
            random.shuffle(valid_chars)
            
            target_pen = None
            target_char = ""
            
            # 寻找有足够复杂度的字形
            for codepoint in valid_chars:
                glyph_name = cmap[codepoint]
                pen = PathExtractorPen(glyph_set)
                glyph_set[glyph_name].draw(pen)
                
                if len(pen.segments) > 8: # 至少8段以上才有骨架提取的价值
                    target_pen = pen
                    target_char = chr(codepoint)
                    break
                    
            if not target_pen:
                ax_left.set_title("No complex glyph found", color='red')
                continue

            # ----------------------------------------------------
            # 🎬 阶段 1：左侧视窗 - 多色贝塞尔轮廓渲染
            # ----------------------------------------------------
            colors = cm.hsv(np.linspace(0, 1, len(target_pen.segments)))
            all_x, all_y = [], []
            
            for (seg_type, pts), color in zip(target_pen.segments, colors):
                for p in pts:
                    all_x.append(p[0])
                    all_y.append(p[1])
                    
                codes = [Path.MOVETO, Path.LINETO] if seg_type == 'line' else \
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3] if seg_type == 'quadratic' else \
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                        
                patch = patches.PathPatch(Path(pts, codes), facecolor='none', edgecolor=color, lw=2, capstyle='round')
                ax_left.add_patch(patch)

            # 获取边界与留白
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            padding = max(xmax - xmin, ymax - ymin) * 0.15
            xlim = (xmin - padding, xmax + padding)
            ylim = (ymin - padding, ymax + padding)

            ax_left.set_xlim(xlim)
            ax_left.set_ylim(ylim)
            ax_left.set_aspect('equal')
            ax_left.axis('off')
            ax_left.set_title(f"Original Contour: {font_name[:15]}... ({target_char})", color='#c9d1d9')

            # ----------------------------------------------------
            # 🎬 阶段 2：高精度光栅化 (Rasterization)
            # ----------------------------------------------------
            full_path = Path(target_pen.all_pts, target_pen.codes)
            
            xx = np.linspace(xlim[0], xlim[1], RESOLUTION)
            yy = np.linspace(ylim[0], ylim[1], RESOLUTION)
            X_grid, Y_grid = np.meshgrid(xx, yy)
            grid_points = np.vstack((X_grid.flatten(), Y_grid.flatten())).T
            
            # 使用 contains_points 瞬间判断网格点是否在闭合轮廓内部！
            mask_flat = full_path.contains_points(grid_points)
            mask = mask_flat.reshape(RESOLUTION, RESOLUTION)

            # ----------------------------------------------------
            # 🎬 阶段 3：中轴变换与宽度提取 (MAT)
            # ----------------------------------------------------
            # 一行代码，同时算出 骨架(山脊) 与 距离变换(内切圆半径)！
            skeleton, distance = medial_axis(mask, return_distance=True)

            # ----------------------------------------------------
            # 🎬 阶段 4：右侧视窗 - [X, Y, W] 张量可视化
            # ----------------------------------------------------
            # 1. 提取离散化张量
            rows, cols = np.where(skeleton)
            skel_x = xx[cols]
            skel_y = yy[rows]
            skel_w = distance[skeleton] # 纯正的宽度权重！
            
            # 2. 绘制极度黯淡的原始轮廓底底图，用于对比
            bg_patch = patches.PathPatch(full_path, facecolor='#161b22', edgecolor='#30363d', lw=1, alpha=0.5)
            ax_right.add_patch(bg_patch)
            
            # 3. 将 [X,Y,W] 绘制为彩虹散点图，映射宽度 W
            # 使用 plasma 色带：黄色/红色代表宽（胖），蓝色/紫色代表窄（尖锐）
            scatter = ax_right.scatter(skel_x, skel_y, c=skel_w, cmap='plasma', s=2, alpha=0.8)

            ax_right.set_xlim(xlim)
            ax_right.set_ylim(ylim)
            ax_right.set_aspect('equal')
            ax_right.axis('off')
            ax_right.set_title("Medial Axis Tensor [X, Y, Width]", color='#00ff41')

        except Exception as e:
            ax_left.set_title(f"Error: {e}", color='red', fontsize=8)
            ax_right.set_title("Failed", color='red', fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    extract_medial_axis_tensors()