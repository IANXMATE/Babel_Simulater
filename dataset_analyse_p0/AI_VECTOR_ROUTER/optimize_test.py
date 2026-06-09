import os
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import minimize
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 📂 路径配置 (自动适配跨目录)
# ==========================================
Script_DIR = os.path.dirname(os.path.abspath(__file__))
# 假设脚本在 XXX/dataset/eval_tools 下，那么它的父目录就是 dataset

ANNOTATIONS_DIR = os.path.join(Script_DIR, "annotations")
TENSORS_DIR = os.path.join(Script_DIR, "../dataset/alien_tensors_storage")


COLORS = [
    (255, 105, 180), (135, 206, 235), (144, 238, 144), (255, 165, 0),
    (218, 112, 214), (255, 215, 0),   (0, 206, 209),   (255, 99, 71)
]

# ==========================================
# 📐 数学核心：贝塞尔曲线求值
# ==========================================
def evaluate_cubic_bezier(p0, p1, p2, p3, num_points=50):
    """根据 4 个控制点，生成平滑的曲线离散点用于 OpenCV 渲染"""
    t = np.linspace(0, 1, num_points)
    # 三次贝塞尔参数方程
    curve_x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
    curve_y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
    return np.vstack((curve_x, curve_y)).T

# ==========================================
# 🔠 原图实时渲染引擎
# ==========================================
def render_char_from_font(font_path, hex_key, canvas_size=400, font_size=320):
    """支持 'U+05DF' 或 '05DF' 格式的提取"""
    raw_hex = hex_key.replace("U+", "")
    try:
        char_str = chr(int(raw_hex, 16))
    except ValueError:
        char_str = hex_key 
        
    img = Image.new('L', (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"❌ 无法加载字体: {font_path}")
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
    bbox = font.getbbox(char_str)
    if bbox is None: return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas_size - text_w) // 2 - bbox[0]
    y = (canvas_size - text_h) // 2 - bbox[1]
    
    draw.text((x, y), char_str, font=font, fill=255)
    return np.array(img)

# ==========================================
# 🛠️ 核心修补算法 (基于母线)
# ==========================================
def apply_topological_snapping(stroke_data_list, snap_threshold=4.0):
    """拓扑吸附：只检查并吸附 mother_bezier 的起点 (p0) 和终点 (p3)"""
    repaired_strokes = copy.deepcopy(stroke_data_list)
    n = len(repaired_strokes)
    
    for i in range(n):
        for j in range(i + 1, n):
            # idx: 0 是起点, 3 是终点
            for idx_i in [0, 3]: 
                for idx_j in [0, 3]: 
                    p_i = np.array(repaired_strokes[i]['mother_bezier'][idx_i])
                    p_j = np.array(repaired_strokes[j]['mother_bezier'][idx_j])
                    
                    if np.linalg.norm(p_i - p_j) < snap_threshold:
                        # 距离极近，强行把 j 的端点坐标等于 i 的端点坐标
                        repaired_strokes[j]['mother_bezier'][idx_j] = repaired_strokes[i]['mother_bezier'][idx_i].copy()
    return repaired_strokes

def optimize_stroke_width(mother_bezier, target_mask, default_width=4.0):
    """线宽寻优：拿着这条贝塞尔曲线去拟合真实 Mask 的粗细"""
    if target_mask is None or np.max(target_mask) == 0:
        return default_width
        
    # 将贝塞尔转为离散点用于渲染测试
    p0, p1, p2, p3 = mother_bezier
    curve_pts = evaluate_cubic_bezier(p0, p1, p2, p3, num_points=50)
    pts_int32 = np.array(curve_pts, np.int32).reshape((-1, 1, 2))
    
    def loss_fn(w):
        w_val = max(1.0, w[0])
        canvas = np.zeros_like(target_mask)
        
        # 测试渲染这条曲线
        cv2.polylines(canvas, [pts_int32], False, 255, int(w_val), cv2.LINE_AA)
        radius = int(w_val) // 2
        if radius > 0:
            cv2.circle(canvas, tuple(pts_int32[0][0]), radius, 255, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts_int32[-1][0]), radius, 255, -1, cv2.LINE_AA)
                
        intersection = np.logical_and(canvas > 0, target_mask > 0).sum()
        union = np.logical_or(canvas > 0, target_mask > 0).sum()
        return 1.0 - (intersection / (union + 1e-6))

    res = minimize(loss_fn, [default_width], method='Nelder-Mead', options={'maxiter': 15})
    return max(1.0, res.x[0])

# ==========================================
# 🎨 渲染引擎 (解析 mother_bezier)
# ==========================================
def render_bezier_strokes(stroke_data_list, shape, colored=True, use_round_cap=False, optimized_widths=None):
    canvas = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 230 
    if not colored:
        canvas = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 255 
        
    for i, stroke in enumerate(stroke_data_list):
        # 取出 4 个控制点
        mb = stroke['mother_bezier']
        if len(mb) != 4: continue
        
        curve_pts = evaluate_cubic_bezier(mb[0], mb[1], mb[2], mb[3], num_points=60)
        pts_int32 = np.array(curve_pts, np.int32).reshape((-1, 1, 2))
            
        color = COLORS[i % len(COLORS)] if colored else (0, 0, 0)
        
        # 优先使用优化后的线宽，否则使用 JSON 自带的线宽 (如果都没有则默认 4)
        if optimized_widths:
            width = int(optimized_widths[i])
        else:
            width = int(stroke.get('width', 4))
        
        cv2.polylines(canvas, [pts_int32], isClosed=False, color=color, thickness=width, lineType=cv2.LINE_AA)
        
        # 圆角填充 (只在首尾打圆点，因为贝塞尔本身很平滑，只需要补齐端点)
        if use_round_cap and width > 1:
            radius = width // 2
            cv2.circle(canvas, tuple(pts_int32[0][0]), radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts_int32[-1][0]), radius, color, -1, cv2.LINE_AA)
                
    return canvas

# ==========================================
# 🚀 主控逻辑
# ==========================================
def run_visualization():
    print(f"🔍 寻找贝塞尔标注文件... ({ANNOTATIONS_DIR})")
    json_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.json')]
    if not json_files:
        print("❌ 未找到标注 JSON 文件。")
        return
        
    random_file = random.choice(json_files)
    
    # 根据你的文件命名习惯修改，比如假设就叫 "alien_font_01.json"
    font_name = random_file.split(".json")[0] 
    
    with open(os.path.join(ANNOTATIONS_DIR, random_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    hex_key = random.choice(list(data.keys()))
    
    # 🌟 直接拿到属于这个字符的母线列表！
    bezier_strokes = data[hex_key]
    
    print(f"✅ 抽中字典: {random_file} | 字符 ID: {hex_key} | 贝塞尔母线数: {len(bezier_strokes)}")

    # 1. 寻找对应的 TTF/OTF 文件并实时渲染原图
    font_path = None
    for ext in ['.ttf', '.otf', '.TTF', '.OTF']:
        potential_path = os.path.join(TENSORS_DIR, f"{font_name}{ext}")
        if os.path.exists(potential_path):
            font_path = potential_path
            break
            
    if font_path is None:
        print(f"⚠️ 找不到对应的 TTF/OTF 字体文件，原图将以空白展示。")
        orig_img = np.zeros((400, 400), dtype=np.uint8)
    else:
        orig_img = render_char_from_font(font_path, hex_key, canvas_size=400, font_size=320)
    
    shape = orig_img.shape if len(orig_img.shape) == 2 else orig_img.shape[:2]

    # 2. 生成人工标注结果图 (未修补，平头)
    manual_color_img = render_bezier_strokes(bezier_strokes, shape, colored=True, use_round_cap=False)
    manual_bw_img = render_bezier_strokes(bezier_strokes, shape, colored=False, use_round_cap=False)

    # --- 执行修补程序 ---
    print("⚙️ 正在执行端点吸附...")
    repaired_strokes = apply_topological_snapping(bezier_strokes, snap_threshold=4.0)
    
    optimized_widths = []
    print("⚙️ 正在执行线宽 IoU 寻优...")
    for stroke in repaired_strokes:
        mb = stroke['mother_bezier']
        best_w = optimize_stroke_width(mb, orig_img, default_width=stroke.get('width', 4.0))
        optimized_widths.append(best_w)

    # 3. 生成修补后结果图 (修补过，圆头，自适应宽度)
    repaired_color_img = render_bezier_strokes(repaired_strokes, shape, colored=True, use_round_cap=True, optimized_widths=optimized_widths)
    repaired_bw_img = render_bezier_strokes(repaired_strokes, shape, colored=False, use_round_cap=True, optimized_widths=optimized_widths)

    # ==========================================
    # 📊 Matplotlib 对比展示
    # ==========================================
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.canvas.manager.set_window_title(f"Font: {font_name} | Char: {hex_key}")

    axes[0].imshow(orig_img, cmap='gray')
    axes[0].set_title("1. Original Image (From TTF)")

    axes[1].imshow(manual_color_img)
    axes[1].set_title("2. Manual (Colored, Butt Caps)")

    axes[2].imshow(manual_bw_img)
    axes[2].set_title("3. Manual (B&W)")

    axes[3].imshow(repaired_color_img)
    axes[3].set_title("4. Repaired (Snapping + Round Joins)")

    axes[4].imshow(repaired_bw_img)
    axes[4].set_title("5. Repaired (B&W + Width Optimized)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_visualization()