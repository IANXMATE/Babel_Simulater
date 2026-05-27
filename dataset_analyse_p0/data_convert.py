import os
import math
import torch

# ==========================================
# 核心物理计算逻辑
# ==========================================
def normalize_angle(angle):
    """强制将角度映射到 [-pi, pi] 之间，防止旋转溢出"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def process_omniglot_directory(input_dir, output_file):
    """
    遍历 Omniglot 的 strokes_background 目录，转化为极坐标残差序列
    """
    if not os.path.exists(input_dir):
        print(f"[致命错误] 找不到文件夹 '{input_dir}'")
        return

    all_sequences = []
    
    print(f"正在扫描 '{input_dir}' 目录下的所有真实手写轨迹...\n")

    # 1. 递归遍历所有语言/字母/txt文件
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.endswith('.txt'):
                continue
                
            filepath = os.path.join(root, filename)
            
            # 2. 读取并解析文本文件
            strokes = []
            current_stroke = []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 遇到空行，代表当前笔画结束，抬笔
                    if not line:
                        if current_stroke:
                            strokes.append(current_stroke)
                            current_stroke = []
                    else:
                        # 解析坐标 (兼容逗号或空格分割)
                        parts = line.replace(',', ' ').split()
                        if len(parts) >= 2:
                            x, y = float(parts[0]), float(parts[1])
                            current_stroke.append((x, y))
            
            # 收尾最后一个笔画
            if current_stroke:
                strokes.append(current_stroke)

            if not strokes:
                continue

            # 3. 转化为极坐标残差序列
            sequence = []
            current_x, current_y = None, None
            current_angle = 0.0 
            
            # BOS (起始符)
            sequence.append([0.0, 0.0, 0.0, 0.0])
            stroke_id = 1.0

            for stroke in strokes:
                for x, y in stroke:
                    if current_x is None:
                        current_x, current_y = x, y
                        continue

                    # 计算绝对位移
                    dx = x - current_x
                    dy = y - current_y
                    distance = math.hypot(dx, dy)

                    # 过滤掉人类手抖导致的完全停滞点 (防止 atan2 出现除零错误)
                    if distance < 1e-4:
                        continue 

                    # 计算角度和变化量
                    abs_angle = math.atan2(dy, dx)
                    delta_theta = normalize_angle(abs_angle - current_angle)
                    
                    # Omniglot 没有压力数据，宽度变化保持为 0
                    delta_w = 0.0

                    sequence.append([stroke_id, distance, delta_theta, delta_w])

                    # 更新状态机
                    current_x, current_y = x, y
                    current_angle = abs_angle

                # 换笔
                stroke_id += 1.0

            # EOS (结束符)
            sequence.append([-1.0, 0.0, 0.0, 0.0])
            
            # 转为 Tensor 并保存
            seq_tensor = torch.tensor(sequence, dtype=torch.float32)
            
            # 记录类别信息 (利用 Omniglot 的目录结构: 语言_字母)
            # 例如: root = 'strokes_background/Latin/character01' -> label = 'Latin_character01'
            path_parts = os.path.normpath(root).split(os.sep)
            label = f"{path_parts[-2]}_{path_parts[-1]}" if len(path_parts) >= 2 else "Unknown"

            all_sequences.append({
                "label": label,
                "filename": filename,
                "sequence": seq_tensor
            })

    # ==========================================
    # 距离归一化 (核心防爆机制)
    # ==========================================
    if len(all_sequences) == 0:
        print("\n[错误] 没有提取到任何序列，请检查 txt 文件是否包含坐标数据。")
        return

    print(f"成功提取了 {len(all_sequences)} 个真实手写样本！")
    print("正在对移动距离 (Distance) 进行全局归一化...")
    
    all_distances = torch.cat([item["sequence"][:, 1] for item in all_sequences])
    valid_distances = all_distances[all_distances > 0]
    dist_mean = valid_distances.mean()
    dist_std = valid_distances.std() + 1e-6

    for item in all_sequences:
        seq = item["sequence"]
        mask = seq[:, 1] > 0
        seq[mask, 1] = (seq[mask, 1] - dist_mean) / dist_std
        item["sequence"] = seq

    torch.save(all_sequences, output_file)
    print(f"✅ 大功告成！全量 Omniglot 极坐标数据已保存至: {output_file}")
    
    # 打印一条数据核对
    demo_item = all_sequences[0]
    print(f"\n[数据检验] 样本分类: {demo_item['label']} | 文件: {demo_item['filename']}")
    print(f"序列 Shape: {demo_item['sequence'].shape}")
    print("前 4 步数据 [ID, \u0394d, \u0394\u03b8, \u0394w]:")
    for step in demo_item['sequence'][:4]:
        print(f"  {step.tolist()}")

if __name__ == "__main__":
    # 你的数据文件夹名称
    INPUT_DIR = "strokes_background"  
    OUTPUT_FILE = "multilingual_polar_trajectories.pt"
    
    process_omniglot_directory(INPUT_DIR, OUTPUT_FILE)