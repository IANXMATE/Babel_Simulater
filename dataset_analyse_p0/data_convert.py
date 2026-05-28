import os
import torch
import numpy as np

# ==========================================
# 1. 复杂度过滤器 (保留你的硬核筛选逻辑)
# ==========================================
def is_complex_enough(strokes, min_strokes=4, min_points=150):
    num_strokes = len(strokes)
    total_points = sum(len(s) for s in strokes)
    
    if num_strokes >= min_strokes:
        return True
    if total_points >= min_points:
        return True
    return False

# ==========================================
# 2. 核心处理主函数
# ==========================================
def process_omniglot_cartesian(input_dir, output_file, min_strokes=4, min_points=200):
    """
    将 Omniglot 转化为 SketchRNN 标准 5 维相对坐标 Token:
    [dx, dy, p1, p2, p3]
    - dx, dy: 相对上一个点的位移
    - p1: 1表示正在画线 (Pen Down)
    - p2: 1表示悬空跳跃寻找新起笔点 (Pen Up)
    - p3: 1表示整个字符结束 (End of Sequence)
    """
    if not os.path.exists(input_dir):
        print(f"[致命错误] 找不到文件夹 '{input_dir}'")
        return

    all_sequences = []
    total_files_scanned = 0
    
    print(f"🔍 正在扫描 '{input_dir}' 下的真实手写样本...")
    print("✨ 采用全新架构: 相对直角坐标系 [dx, dy, p1, p2, p3]\n")

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.endswith('.txt'):
                continue
                
            total_files_scanned += 1
            filepath = os.path.join(root, filename)
            
            strokes = []
            current_stroke = []
            
            # --- 文本解析逻辑 ---
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if not line or line == "START":
                        continue
                    elif line == "BREAK":
                        if current_stroke:
                            strokes.append(current_stroke)
                            current_stroke = []
                    else:
                        parts = line.replace(',', ' ').split()
                        if len(parts) >= 2:
                            try:
                                x, y = float(parts[0]), float(parts[1])
                                current_stroke.append((x, y))
                            except ValueError:
                                continue
            
            if current_stroke:
                strokes.append(current_stroke)

            if not strokes:
                continue

            # --- 执行复杂度筛选 ---
            if not is_complex_enough(strokes, min_strokes, min_points):
                continue

            # ==========================================
            # 🎯 转化为 5 维 Token 序列
            # ==========================================
            sequence = []
            current_x, current_y = None, None

            # 添加一个虚拟的 BOS (起始符)，随便用一个全 0 状态
            sequence.append([0.0, 0.0, 0.0, 0.0, 0.0])

            for stroke_idx, stroke in enumerate(strokes):
                for point_idx, (x, y) in enumerate(stroke):
                    # 全局第一个点，只初始化坐标，不产生位移 Token
                    if current_x is None:
                        current_x, current_y = x, y
                        continue
                    
                    # 计算相对位移
                    dx = x - current_x
                    dy = y - current_y

                    # 判定状态机
                    if point_idx == 0:
                        # 这是一个新笔画的第一个点！说明刚才跨越了半个屏幕悬空飞过来的
                        # 触发 p2 (Pen Up)
                        p1, p2, p3 = 0.0, 1.0, 0.0
                    else:
                        # 同一笔画内的移动，留下墨迹
                        # 触发 p1 (Pen Down)
                        p1, p2, p3 = 1.0, 0.0, 0.0

                    sequence.append([dx, dy, p1, p2, p3])

                    current_x, current_y = x, y

            # 最后一个 Token：宣告结束
            sequence.append([0.0, 0.0, 0.0, 0.0, 1.0]) # 触发 p3 (End)
            
            seq_tensor = torch.tensor(sequence, dtype=torch.float32)
            
            path_parts = os.path.normpath(root).split(os.sep)
            label = f"{path_parts[-2]}_{path_parts[-1]}" if len(path_parts) >= 2 else "Unknown"

            all_sequences.append({
                "label": label,
                "filename": filename,
                "sequence": seq_tensor
            })

    # ==========================================
    # 数据集统计与标准化 (极其关键)
    # ==========================================
    kept_files = len(all_sequences)
    print("-" * 45)
    print(f"📊 扫描总计: {total_files_scanned} 个字符")
    print(f"✅ 成功保留 (高复杂度): {kept_files} 个")
    print("-" * 45)

    if kept_files == 0:
        print("\n[错误] 条件太严苛，所有字符都被过滤掉了！")
        return

    # 💡 全局标准差缩放 (Standard Deviation Scaling)
    # 为什么不用 min-max？因为我们需要保持 0 依然是 0，且不破坏 dx 和 dy 的比例关系。
    print("正在计算全局坐标偏移量的标准差，进行防爆缩放...")
    
    # 收集所有的 dx 和 dy
    all_dx = torch.cat([item["sequence"][:, 0] for item in all_sequences])
    all_dy = torch.cat([item["sequence"][:, 1] for item in all_sequences])
    
    # 将它们合并在一起算全局标准差 (保持 x 和 y 缩放比例完全一致，字才不会被拉伸变扁)
    all_deltas = torch.cat([all_dx, all_dy])
    global_std = all_deltas.std().item() + 1e-6
    
    print(f"计算得出全局缩放因子 (Std): {global_std:.4f}")

    for item in all_sequences:
        seq = item["sequence"]
        # 直接让 dx 和 dy 均除以全局标准差
        seq[:, 0] = seq[:, 0] / global_std
        seq[:, 1] = seq[:, 1] / global_std
        item["sequence"] = seq

    torch.save(all_sequences, output_file)
    print(f"🎉 直角坐标 5 维数据已完美保存至: {output_file}")
    
    # 打印一条数据看看长什么样
    demo_seq = all_sequences[0]["sequence"]
    print(f"\n[数据检验] 前 5 步数据 [dx, dy, p1, p2, p3]:")
    for step in demo_seq[:5]:
        print(f"  [{step[0]:.4f}, {step[1]:.4f}, {step[2]:.0f}, {step[3]:.0f}, {step[4]:.0f}]")

if __name__ == "__main__":
    INPUT_DIR = "strokes_background"  
    OUTPUT_FILE = "omniglot_cartesian_5d.pt"
    
    process_omniglot_cartesian(
        input_dir=INPUT_DIR, 
        output_file=OUTPUT_FILE, 
        min_strokes=4,  
        min_points=200   
    )