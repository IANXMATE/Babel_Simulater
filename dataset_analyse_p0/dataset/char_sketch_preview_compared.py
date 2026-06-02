import os
import random
import glob
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from matplotlib.font_manager import FontProperties
from matplotlib.text import TextPath
import matplotlib.patches as patches

TARGET_DIR = "alien_tensors_raw"  
NUM_SAMPLES = 10
RENDER_SIZE = 800  # 工业级高精度光栅化分辨率

# ==========================================
# 🛠️ 核心 1：基于边遍历的 100% 无损图论拆解
# ==========================================
def extract_paths_from_skeleton(skeleton_img):
    """
    将单像素宽的二值骨架转化为连续的坐标序列。
    采用 Edge-based Traversal (边遍历)，确保 100% 覆盖率，绝不遗漏任何线条！
    """
    rows, cols = np.where(skeleton_img)
    points = set(zip(rows, cols))
    
    # 1. 构建 8-连通 无向图
    G = nx.Graph()
    for r, c in points:
        G.add_node((r, c))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                if (r+dr, c+dc) in points:
                    G.add_edge((r, c), (r+dr, c+dc))
                    
    # 2. 识别交叉路口 (Degree > 2)
    junctions = set(n for n, d in G.degree() if d > 2)
    
    paths = []
    visited_edges = set()
    
    # 3. 🚨 工业级核心：遍历所有“边”而不是“点”
    for u, v in G.edges():
        edge = tuple(sorted((u, v)))
        if edge in visited_edges: continue
        
        # 发现一条未访问的边，开始沿着它向两端疯狂生长，直到撞到路口或尽头！
        path = [u, v]
        visited_edges.add(edge)
        
        # ---> 向 v 方向生长
        curr = v
        prev = u
        while curr not in junctions and G.degree(curr) == 2:
            neighbors = list(G.neighbors(curr))
            nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
            nxt_edge = tuple(sorted((curr, nxt)))
            if nxt_edge in visited_edges: break
            
            path.append(nxt)
            visited_edges.add(nxt_edge)
            prev = curr
            curr = nxt
            
        # <--- 向 u 方向生长
        curr = u
        prev = v
        while curr not in junctions and G.degree(curr) == 2:
            neighbors = list(G.neighbors(curr))
            nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
            nxt_edge = tuple(sorted((curr, nxt)))
            if nxt_edge in visited_edges: break
            
            path.insert(0, nxt)
            visited_edges.add(nxt_edge)
            prev = curr
            curr = nxt
            
        # 过滤极小噪点 (保留有意义的线条)
        if len(path) > 3:
            paths.append(path)
            
    return paths, list(junctions)

# ==========================================
# 🚀 主管线：渲染 -> 细化 -> 提取
# ==========================================
def run_industrial_raster_to_graph_pipeline():
    print(f"🛸 启动 V13 工业级光栅拓扑引擎，扫描目录: {TARGET_DIR}")
    
    font_files = glob.glob(os.path.join(TARGET_DIR, "*.[ot]tf"))
    if not font_files: return print(f"❌ 未找到字体文件！")
        
    sample_files = random.sample(font_files, min(NUM_SAMPLES, len(font_files)))
    plt.style.use('dark_background')
    
    fig, axes = plt.subplots(NUM_SAMPLES, 2, figsize=(8, 1.5 * NUM_SAMPLES))
    fig.suptitle("Alien Tensors: FreeType Rasterization -> Medial Axis Graph", fontsize=18, color='#00ff41', y=0.98)

    for i, file_path in enumerate(sample_files):
        ax_left = axes[i, 0] if NUM_SAMPLES > 1 else axes[0]
        ax_right = axes[i, 1] if NUM_SAMPLES > 1 else axes[1]
        font_name = os.path.basename(file_path)
        
        try:
            # 使用 PIL (FreeType) 获取真实蒙版
            pil_font = ImageFont.truetype(file_path, RENDER_SIZE)
            
            # 使用 matplotlib (也是 FreeType) 获取用于左侧展示的精准矢量边界
            prop = FontProperties(fname=file_path)
            
            target_char = ""
            mask = None
            
            # 随机寻找一个非空字符
            chars_to_test = [chr(c) for c in range(33, 126)] + [chr(c) for c in range(0x4E00, 0x4E50)]
            random.shuffle(chars_to_test)
            
            for char in chars_to_test:
                try:
                    # 尝试渲染
                    bbox = pil_font.getbbox(char)
                    if not bbox: continue
                    left, top, right, bottom = bbox
                    w, h = right - left, bottom - top
                    
                    if w > 50 and h > 50:
                        # 生成完美包围盒的高精度画布
                        img = Image.new('L', (w + 100, h + 100), 0)
                        draw = ImageDraw.Draw(img)
                        # 将字符完美绘制在正中心
                        draw.text((50 - left, 50 - top), char, font=pil_font, fill=255)
                        
                        mask = np.array(img) > 128
                        target_char = char
                        break
                except Exception:
                    continue
                    
            if mask is None:
                ax_left.set_title("No valid glyph", color='red')
                ax_left.axis('off'); ax_right.axis('off'); continue

            # ========================== 左侧：原生矢量边缘展示 ==========================
            # 使用 TextPath 直接获取该字体的精准矢量轮廓
            text_path = TextPath((0, 0), target_char, prop=prop, size=RENDER_SIZE)
            patch = patches.PathPatch(text_path, facecolor='#00e5ff', edgecolor='white', lw=1, alpha=0.6)
            ax_left.add_patch(patch)
            
            # 自动调整视野
            vertices = text_path.vertices
            if len(vertices) > 0:
                ax_left.set_xlim(vertices[:, 0].min() - 50, vertices[:, 0].max() + 50)
                ax_left.set_ylim(vertices[:, 1].min() - 50, vertices[:, 1].max() + 50)
                
            ax_left.set_aspect('equal')
            ax_left.axis('off')
            ax_left.set_title(f"TrueType Vector Glyph: {font_name[:15]}... ({target_char})", color='#c9d1d9')

            # ========================== 右侧：工业级骨架提取 ==========================
            # 1. 拓扑细化 (Zhang-Suen 算法)
            skeleton = skeletonize(mask)
            
            # 2. 图论无损拆解序列
            stroke_segments, junctions = extract_paths_from_skeleton(skeleton)

            # 将图像坐标系(y, x)转换为与字体一致的直角坐标系(x, -y)
            # 画出极其黯淡的原始 Mask 底图作为参考
            mask_rows, mask_cols = np.where(mask)
            ax_right.scatter(mask_cols, -mask_rows, color='#161b22', s=1, alpha=0.1)

            colors_right = cm.spring(np.linspace(0, 1, max(1, len(stroke_segments))))
            
            # 绘制提取出的连续张量序列
            for seg_pts, color in zip(stroke_segments, colors_right):
                skel_y = [-p[0] for p in seg_pts] # 注意 Y 轴翻转
                skel_x = [p[1] for p in seg_pts]
                ax_right.plot(skel_x, skel_y, color=color, linewidth=2.5, alpha=1.0, solid_capstyle='round', zorder=3)

            # 标记交叉切断点
            if junctions:
                jx = [p[1] for p in junctions]
                jy = [-p[0] for p in junctions]
                ax_right.scatter(jx, jy, color='red', marker='P', s=60, zorder=5, label="Junctions")
                ax_right.legend(loc="upper right", facecolor='#161b22', fontsize=8, labelcolor='white')

            # 视觉对齐
            ax_right.set_aspect('equal')
            ax_right.axis('off')
            ax_right.set_title(f"Industrial Skeleton Paths | Segments: {len(stroke_segments)}", color='#00ff41')

        except Exception as e:
            ax_left.set_title(f"Error: {e}", color='red', fontsize=8)
            ax_right.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_industrial_raster_to_graph_pipeline()