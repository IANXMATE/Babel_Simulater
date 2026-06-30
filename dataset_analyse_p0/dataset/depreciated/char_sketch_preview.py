import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
import matplotlib.cm as cm

# ==========================================
# ⚙️ 全局配置区
# ==========================================
# 你的字体文件夹路径（纯物理文件库）
TARGET_DIR = "alien_tensors_raw"  
NUM_SAMPLES = 10

# ==========================================
# 🖋️ 核心黑科技：自定义指令截获笔 (Digital Interceptor Pen)
# ==========================================
class SegmentExtractorPen(BasePen):
    """
    这支笔不会真正在画布上画图，而是拦截所有的底层指令，
    把轮廓拆解成一段段独立的数学曲线（Line, Quadratic, Cubic）。
    """
    def __init__(self, glyphSet):
        super().__init__(glyphSet)
        self.segments = []
        self.current_pt = (0, 0)
        self.start_pt = (0, 0)

    def _moveTo(self, p):
        self.current_pt = p
        self.start_pt = p

    def _lineTo(self, p):
        self.segments.append(('line', [self.current_pt, p]))
        self.current_pt = p

    def _curveToOne(self, p1, p2, p3):
        # OTF 专用的三次贝塞尔曲线 (Cubic Bezier: 2个控制点)
        self.segments.append(('cubic', [self.current_pt, p1, p2, p3]))
        self.current_pt = p3

    def _qCurveToOne(self, p1, p2):
        # TTF 专用的二次贝塞尔曲线 (Quadratic Bezier: 1个控制点)
        self.segments.append(('quadratic', [self.current_pt, p1, p2]))
        self.current_pt = p2

    def _closePath(self):
        # 强制闭合轮廓
        if self.current_pt != self.start_pt:
            self.segments.append(('line', [self.current_pt, self.start_pt]))
        self.current_pt = self.start_pt

# ==========================================
# 🚀 主渲染流水线
# ==========================================
def render_bezier_anatomy():
    print(f"🛸 启动底层几何解剖仪，扫描目录: {TARGET_DIR}")
    
    # 获取所有字体文件
    font_files = glob.glob(os.path.join(TARGET_DIR, "*.[ot]tf"))
    if not font_files:
        return print(f"❌ 未在 {TARGET_DIR} 找到任何 TTF 或 OTF 文件！")
        
    # 随机抽取 10 个（或全部，如果不足 10 个）
    sample_files = random.sample(font_files, min(NUM_SAMPLES, len(font_files)))
    
    # 开启赛博朋克暗黑主题
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    fig.suptitle("Alien Tensors: Bezier Curve Anatomy", fontsize=20, color='#00ff41', y=0.98)

    for i, file_path in enumerate(sample_files):
        ax = axes[i]
        font_name = os.path.basename(file_path)
        
        try:
            font = TTFont(file_path)
            cmap = font.getBestCmap()
            glyph_set = font.getGlyphSet()
            
            # 打乱字符集，寻找一个不是空白且含有足够曲线的复杂异星字符
            valid_chars = list(cmap.keys())
            random.shuffle(valid_chars)
            
            target_pen = None
            target_char = ""
            
            for codepoint in valid_chars:
                glyph_name = cmap[codepoint]
                pen = SegmentExtractorPen(glyph_set)
                glyph_set[glyph_name].draw(pen)
                
                # 如果这个字符由超过 3 段曲线/直线组成，判定为有效字形
                if len(pen.segments) > 3:
                    target_pen = pen
                    target_char = chr(codepoint)
                    break
                    
            if not target_pen:
                ax.set_title(f"{font_name}\n(No valid glyphs)", color='red', fontsize=8)
                ax.axis('off')
                continue

            # 开始绘制！使用 HSV 色轮为不同的曲线段分配高对比度的荧光色
            colors = cm.hsv(np.linspace(0, 1, len(target_pen.segments)))
            
            all_x, all_y = [], []
            
            for (seg_type, pts), color in zip(target_pen.segments, colors):
                # 记录所有坐标以便设置画布视野
                for p in pts:
                    all_x.append(p[0])
                    all_y.append(p[1])
                
                # 构建 Matplotlib 路径
                if seg_type == 'line':
                    codes = [Path.MOVETO, Path.LINETO]
                elif seg_type == 'quadratic':
                    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                elif seg_type == 'cubic':
                    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                    
                path = Path(pts, codes)
                patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=3, capstyle='round')
                ax.add_patch(patch)
                
                # 🔬 深度解剖：画出那些极其重要但不可见的“贝塞尔控制点”
                if seg_type in ['quadratic', 'cubic']:
                    ctrl_pts = pts[1:-1] # 掐头去尾，中间的都是控制点
                    c_x = [p[0] for p in ctrl_pts]
                    c_y = [p[1] for p in ctrl_pts]
                    
                    # 用白色的 X 标记控制点
                    ax.plot(c_x, c_y, 'x', color='white', markersize=6, alpha=0.8)
                    
                    # 画出控制点和曲线起止点之间的引力虚线
                    ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], ':', color=color, alpha=0.6)
                    ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], ':', color=color, alpha=0.6)

            # 自适应调整相机的视野范围
            if all_x and all_y:
                margin_x = (max(all_x) - min(all_x)) * 0.15
                margin_y = (max(all_y) - min(all_y)) * 0.15
                ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
                ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)

            # 隐藏坐标轴边框，只显示网格
            ax.set_aspect('equal')
            ax.grid(True, color='#30363d', linestyle='--', alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 设置标题：字体名 + 字符 + 它是由多少个张量切片拼成的
            title_text = f"{font_name[:15]}...\nChar: {target_char} | Segments: {len(target_pen.segments)}"
            ax.set_title(title_text, color='#c9d1d9', fontsize=10)

        except Exception as e:
            ax.set_title(f"Error reading\n{font_name}", color='red', fontsize=8)
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    render_bezier_anatomy()