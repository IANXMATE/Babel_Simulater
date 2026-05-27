import os
import math
import torch
import numpy as np
from svgpathtools import svg2paths

# ==========================================
# 核心物理计算逻辑
# ==========================================
def normalize_angle(angle):
    """
    【极其关键的修正】
    将任意角度强制映射到 [-pi, pi] 之间。
    防止模型在 179° 到 -179° 之间计算出 358° 的死亡误差角。
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

def process_svg_directory(input_dir, output_file, num_samples=15):
    """
    遍历目录下的SVG，提取并转化为极坐标残差序列
    :param input_dir: 存放 .svg 文件的目录
    :param output_file: 保存的 .pt 模型数据文件路径
    :param num_samples: 每条笔画采样的点数
    """
    if not os.path.exists(input_dir):
        print(f"错误: 找不到文件夹 '{input_dir}'")
        return

    all_sequences = []
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.svg')]
    total_files = len(file_list)
    
    print(f"共发现 {total_files} 个 SVG 文件，开始极坐标转换...\n")

    for i, filename in enumerate(file_list):
        filepath = os.path.join(input_dir, filename)
        
        # svg2paths 直接从文件中读取路径对象
        paths, _ = svg2paths(filepath)
        
        if not paths:
            continue

        sequence = []
        current_x, current_y = None, None
        
        # 初始参考系：假设画笔初始朝向正右方 (X轴正方向，0度)
        current_angle = 0.0 
        
        # [Stroke_ID, Distance, Delta_Theta, Delta_Width]
        # 起始符 BOS (编号为 0)
        sequence.append([0.0, 0.0, 0.0, 0.0])
        
        stroke_id = 1.0

        for path in paths:
            # 过滤掉退化的零长度笔画
            if path.length() < 1e-6:
                continue

            points = [path.point(t) for t in np.linspace(0, 1, num_samples)]

            for j, p in enumerate(points):
                x, y = p.real, p.imag

                # 第一笔的第一个点：初始化绝对位置，不产生移动
                if current_x is None:
                    current_x, current_y = x, y
                    # 由于我们抛弃了绝对坐标，这个初始坐标相当于被直接“忘掉”了，
                    # 它只作为后续计算距离和角度的原点。这是平移不变性的精髓。
                    continue

                # 计算绝对位移
                dx = x - current_x
                dy = y - current_y
                distance = math.hypot(dx, dy)

                # 避免极短移动导致 atan2 计算出不稳定的剧烈角度
                if distance < 1e-5:
                    continue 

                # 计算绝对角度
                abs_angle = math.atan2(dy, dx)
                
                # 计算相对角度变化量 (角速度) 并修正越界
                delta_theta = normalize_angle(abs_angle - current_angle)
                
                # 预留宽度变化量 (如果SVG没有自带宽度数据，默认为0)
                delta_width = 0.0

                # 记录 Token
                sequence.append([stroke_id, distance, delta_theta, delta_width])

                # 更新物理状态机
                current_x, current_y = x, y
                current_angle = abs_angle

            # 一笔画完，笔画编号 +1
            stroke_id += 1.0

        # 结束符 EOS (编号为 -1)
        sequence.append([-1.0, 0.0, 0.0, 0.0])
        
        # 转为 PyTorch Tensor
        seq_tensor = torch.tensor(sequence, dtype=torch.float32)
        
        all_sequences.append({
            "filename": filename,
            "sequence": seq_tensor
        })

        if (i + 1) % 100 == 0 or (i + 1) == total_files:
            print(f"已处理: {i + 1}/{total_files} ...")

    # ==========================================
    # 距离归一化 (防梯度爆炸)
    # ==========================================
    # 极坐标中，角度自然在 [-pi, pi] 之间，不需要缩放。
    # 但 distance 取决于 SVG 画布大小 (可能高达上千)。必须全局归一化。
    print("\n正在对移动距离 (Distance) 进行全局归一化...")
    all_distances = torch.cat([item["sequence"][:, 1] for item in all_sequences])
    # 取非零距离的平均值和标准差
    valid_distances = all_distances[all_distances > 0]
    dist_mean = valid_distances.mean()
    dist_std = valid_distances.std() + 1e-6

    for item in all_sequences:
        seq = item["sequence"]
        # 仅对 distance 所在列 (索引1) 进行标准化，跳过 BOS 和 EOS 的 0
        mask = seq[:, 1] > 0
        seq[mask, 1] = (seq[mask, 1] - dist_mean) / dist_std
        item["sequence"] = seq

    torch.save(all_sequences, output_file)
    print(f"✅ 大功告成！全量极坐标数据已保存至: {output_file}")
    
    # 打印一条数据的 Demo 供检查
    if all_sequences:
        demo_seq = all_sequences[0]["sequence"]
        print(f"\n[数据检验] 文件: {all_sequences[0]['filename']}")
        print(f"序列 Shape: {demo_seq.shape}")
        print("前 4 步数据 [ID, \u0394d, \u0394\u03b8, \u0394w]:")
        for step in demo_seq[:4]:
            print(f"  {step.tolist()}")

if __name__ == "__main__":
    # 设定你的 SVG 文件夹名称
    INPUT_DIR = "kanji"  
    OUTPUT_FILE = "multilingual_polar_trajectories.pt"
    
    # 如果没有文件夹，自动创建一个防止报错
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    process_svg_directory(INPUT_DIR, OUTPUT_FILE, num_samples=15)