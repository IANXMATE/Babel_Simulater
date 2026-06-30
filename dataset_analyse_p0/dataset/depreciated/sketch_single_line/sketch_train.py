import os
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pydiffvg
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# ⚙️ 渲染与系统配置
# ==========================================
TARGET_DIR = "../alien_tensors_raw"  
CANVAS_SIZE = 256
NUM_STROKES = 10      # 初始化 10 根待优化的骨架
NUM_SAMPLES = 100     # 每根贝塞尔曲线采样 100 个点来生成多边形
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pydiffvg.set_device(DEVICE)

# ==========================================
# 🎯 数据准备：随机字体 -> Ground Truth 图像
# ==========================================
def get_random_glyph_target(font_dir, size=CANVAS_SIZE):
    font_files = glob.glob(os.path.join(font_dir, "*.[ot]tf"))
    if not font_files: raise ValueError("找不到字体文件！")
    
    font_path = random.choice(font_files)
    pil_font = ImageFont.truetype(font_path, int(size * 0.8))
    
    # 找一个有效的字符
    valid_chars = [chr(c) for c in range(33, 126)] + [chr(c) for c in range(0x4E00, 0x4E50)]
    random.shuffle(valid_chars)
    
    for char in valid_chars:
        try:
            bbox = pil_font.getbbox(char)
            if bbox and (bbox[2]-bbox[0]) > 20:
                img = Image.new('L', (size, size), 255) # 白底
                draw = ImageDraw.Draw(img)
                # 居中绘制黑字
                draw.text(((size-(bbox[2]-bbox[0]))/2, (size-(bbox[3]-bbox[1]))/2), 
                          char, font=pil_font, fill=0)
                
                # 转化为 PyTorch Tensor (H, W, 4) RGBA
                target = torch.from_numpy(np.array(img)).float() / 255.0
                target = target.unsqueeze(-1).repeat(1, 1, 4)
                target[:, :, 3] = 1.0 # Alpha = 1
                return target.to(DEVICE), font_path, char
        except: continue
    raise ValueError("未找到可渲染字符")

# ==========================================
# 🧠 核心架构：实现你的 "Width Trick" 的可微笔画
# ==========================================
class WidthTrickStroke(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 母线贝塞尔 P (4个控制点，范围在画布内)
        p_init = torch.rand(4, 2) * CANVAS_SIZE
        # 将起点和终点收拢在中心附近，方便梯度寻找方向
        p_init = (p_init + torch.tensor([CANVAS_SIZE/2, CANVAS_SIZE/2])) / 2.0
        self.P = nn.Parameter(p_init)
        
        # 2. 宽度贝塞尔 Q (4个控制点)
        # 初始化基线为 (0,0) 到 (20,0)，宽度鼓起约为 5
        q_init = torch.tensor([
            [0.0, 0.0],
            [6.0, 8.0],
            [14.0, 8.0],
            [20.0, 0.0]
        ])
        self.Q = nn.Parameter(q_init)
        
        # 3. 存在概率 (透明度 Alpha)，用于控制曲线数量的最少化
        # 初始化为较大值，鼓励先去覆盖面积
        self.alpha_logit = nn.Parameter(torch.tensor(1.0)) 

    def cubic_bezier(self, pts, t):
        # pts: [4, 2], t: [S, 1]
        t2, t3 = t**2, t**3
        mt = 1 - t
        mt2, mt3 = mt**2, mt**3
        return mt3*pts[0] + 3*mt2*t*pts[1] + 3*mt*t2*pts[2] + t3*pts[3]

    def cubic_bezier_deriv(self, pts, t):
        t2 = t**2
        mt = 1 - t
        mt2 = mt**2
        return 3*mt2*(pts[1]-pts[0]) + 6*mt*t*(pts[2]-pts[1]) + 3*t2*(pts[3]-pts[2])

    def forward(self, t):
        # 1. 计算母线中心轨迹 M(t)
        M_t = self.cubic_bezier(self.P, t)
        
        # 2. 计算母线法向量 N(t)
        dM_t = self.cubic_bezier_deriv(self.P, t)
        normal = torch.stack([-dM_t[:, 1], dM_t[:, 0]], dim=1)
        normal = normal / (torch.norm(normal, dim=1, keepdim=True) + 1e-6)
        
        # 3. 计算你的宽度 Trick！R(t) = W(t) 到基线的法向距离
        W_t = self.cubic_bezier(self.Q, t)
        baseline_vec = self.Q[3] - self.Q[0]
        v_norm = torch.norm(baseline_vec) + 1e-6
        base_normal = torch.stack([-baseline_vec[1], baseline_vec[0]]) / v_norm
        
        # 宽度的核心公式：距离基线的绝对高度
        R_t = torch.abs(torch.matmul(W_t - self.Q[0], base_normal)).unsqueeze(1)
        
        # 4. 生成左边缘和右边缘，缝合成一个完整的填充多边形！
        left_bound = M_t + normal * R_t
        right_bound = M_t - normal * R_t
        polygon_pts = torch.cat([left_bound, torch.flip(right_bound, dims=[0])], dim=0)
        
        alpha = torch.sigmoid(self.alpha_logit)
        return polygon_pts, alpha

# ==========================================
# 🚀 训练主循环 (反向传播与可微渲染)
# ==========================================
def train_diff_render():
    print("🛸 获取 Ground Truth 图像...")
    target_img, font_path, char = get_random_glyph_target(TARGET_DIR)
    
    # 实例化 N 根可微笔画
    strokes = nn.ModuleList([WidthTrickStroke().to(DEVICE) for _ in range(NUM_STROKES)])
    
    # 优化器
    optimizer = optim.Adam(strokes.parameters(), lr=2.0)
    
    # 采样参数 t
    t = torch.linspace(0, 1, NUM_SAMPLES, device=DEVICE).unsqueeze(1)
    
    print(f"🎯 正在优化字符: '{char}' from {os.path.basename(font_path)}")
    print("🔥 开始可微渲染与梯度反传...")

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for epoch in range(500):
        optimizer.zero_grad()
        
        shapes = []
        shape_groups = []
        
        sparsity_loss = 0.0
        
        # 前向生成多边形
        for i, stroke in enumerate(strokes):
            polygon_pts, alpha = stroke(t)
            
            # 构建 DiffVG 渲染对象
            polygon = pydiffvg.Polygon(points=polygon_pts, is_closed=True)
            shapes.append(polygon)
            
            # 设置颜色为黑色 (R=0,G=0,B=0)，透明度为 alpha
            color = torch.cat([torch.zeros(3, device=DEVICE), alpha.unsqueeze(0)])
            shape_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([i]), fill_color=color)
            shape_groups.append(shape_group)
            
            # 累加稀疏度惩罚 (L1)
            sparsity_loss += alpha
            
        # 场景序列化与渲染
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width=CANVAS_SIZE, canvas_height=CANVAS_SIZE, 
            shapes=shapes, shape_groups=shape_groups
        )
        render = pydiffvg.RenderFunction.apply
        img = render(CANVAS_SIZE, CANVAS_SIZE, 2, 2, 0, None, *scene_args)
        
        # 将 DiffVG 的透明底组合到白底上，匹配 Target
        pred_rgb = img[:, :, :3]
        pred_alpha = img[:, :, 3:4]
        pred_img = pred_alpha * pred_rgb + (1 - pred_alpha) * torch.ones_like(pred_rgb)
        
        # ================== 核心 Loss 函数 ==================
        # 1. Coverage Loss: 面积覆盖率 (L1 / MSE)
        loss_coverage = torch.nn.functional.mse_loss(pred_img, target_img[:, :, :3])
        
        # 2. Sparsity Loss: 惩罚无用的笔画，促使骨架数量最少化！
        # 随着训练进行，慢慢增大稀疏惩罚的权重
        lambda_sparse = 0.01 + (epoch / 500) * 0.1 
        loss_sparse = lambda_sparse * sparsity_loss
        
        loss = loss_coverage + loss_sparse
        loss.backward()
        optimizer.step()
        
        # ================== 可视化更新 ==================
        if epoch % 10 == 0:
            axes[0].clear(); axes[1].clear(); axes[2].clear()
            
            # 图1：Ground Truth
            axes[0].imshow(target_img.cpu().numpy()[:,:,0], cmap='gray')
            axes[0].set_title("Ground Truth (Target)")
            axes[0].axis('off')
            
            # 图2：当前可微渲染的重建结果
            axes[1].imshow(pred_img.detach().cpu().numpy())
            axes[1].set_title(f"Diff Render (Epoch {epoch})")
            axes[1].axis('off')
            
            # 图3：提取的纯净骨架！(只画出 alpha > 0.5 的母线)
            axes[2].set_xlim(0, CANVAS_SIZE); axes[2].set_ylim(CANVAS_SIZE, 0)
            axes[2].set_aspect('equal')
            axes[2].set_title("Optimized Mother Tensors")
            
            for stroke in strokes:
                if torch.sigmoid(stroke.alpha_logit) > 0.5:
                    m_pts = stroke.cubic_bezier(stroke.P, t).detach().cpu().numpy()
                    axes[2].plot(m_pts[:, 0], m_pts[:, 1], linewidth=2)
                    
            plt.pause(0.01)

    print("✅ 训练完成！多余的笔画已因为 Sparsity Loss 被消灭，留下的就是最少量的母线张量！")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train_diff_render()