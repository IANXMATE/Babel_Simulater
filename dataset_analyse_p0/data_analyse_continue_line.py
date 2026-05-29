import torch
import numpy as np
import math

def get_min_angle_diff(th1, th2):
    """计算两个弧度角之间的最小绝对夹角，并转换为角度(Degrees)"""
    diff = abs(th1 - th2) % (2 * math.pi)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return math.degrees(diff)

def analyze_connected_strokes(data_path="omniglot_parametric_7d.pt"):
    try:
        data = torch.load(data_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {data_path}，请确认数据是否已生成。")
        return

    # 存储特征的列表
    turning_angles = []
    combined_lengths = []
    
    total_pairs = 0
    connected_straight_pairs = 0
    
    # 判定阈值 (可根据实际归一化尺度微调)
    CONNECTION_TOLERANCE = 0.05  # 首尾相连的最大容差距离 (归一化尺度下 5%)
    MIN_ANGLE_DIFF = 15.0        # 最小转折角度 (度)，小于此值认为是同一条直线
    
    print(f"🚀 正在启动连笔雷达，寻找 [直线]+[直线] 构成的完美转折...")

    for item in data:
        seq = item["sequence"]
        
        # 滑动窗口遍历相邻的两笔 (i 和 i+1)
        for i in range(len(seq) - 1):
            token1 = seq[i]
            token2 = seq[i+1]
            
            # 解析特征
            # token = [dx, dy, L, theta, kappa, W, P]
            L1, th1, k1, P1 = token1[2].item(), token1[3].item(), token1[4].item(), token1[6].item()
            dx2, dy2, L2, th2, k2, P2 = token2[0].item(), token2[1].item(), token2[2].item(), token2[3].item(), token2[4].item(), token2[6].item()
            
            # 排除包含结束符的对子
            if P1 == 1.0 or P2 == 1.0:
                continue
                
            total_pairs += 1
            
            # 条件 1: 必须都是直线
            if abs(k1) > 1e-4 or abs(k2) > 1e-4:
                continue
                
            # 条件 2: 必须首尾相连 (第二笔的起点相对第一笔终点的距离极小)
            jump_distance = math.hypot(dx2, dy2)
            if jump_distance > CONNECTION_TOLERANCE:
                continue
                
            # 条件 3: 必须有一定角度 (产生真正的转折折角)
            angle_diff = get_min_angle_diff(th1, th2)
            if angle_diff < MIN_ANGLE_DIFF:
                continue
                
            # 🎯 命中目标！这完全符合你的连笔定义
            connected_straight_pairs += 1
            turning_angles.append(angle_diff)
            combined_lengths.append(L1 + L2)

    if not turning_angles:
        print("⚠️ 未发现符合条件的连笔特征。可能容差阈值过严，或数据集本身没有此类结构。")
        return

    angles_arr = np.array(turning_angles)
    lengths_arr = np.array(combined_lengths)
    
    quantiles = [0, 5, 25, 50, 75, 95, 100]
    angle_percentiles = np.percentile(angles_arr, quantiles)
    length_percentiles = np.percentile(lengths_arr, quantiles)

    # 打印极简报表
    print("\n" + "=" * 50)
    print(" ⚡ 连笔 (转折直线) 物理特征分位数报表")
    print("=" * 50)
    print(f"扫描相邻笔画对总数 : {total_pairs}")
    print(f"命中目标连笔数量   : {connected_straight_pairs} (占比 {(connected_straight_pairs/total_pairs)*100:.1f}%)")
    print("-" * 50)
    
    print("【特征 A: 转折角度 (Degrees)】 - 决定了连笔的锋利程度")
    for q, val in zip(quantiles, angle_percentiles):
        note = " (极小锐角，类似闪电)" if q == 0 else " (近乎直角)" if abs(val-90)<5 else " (极限折返)" if q == 100 else ""
        print(f" P{q:<4}:  {val:6.2f}° {note}")
        
    print("-" * 50)
    print("【特征 B: 连笔总长度 (L1+L2)】 - 决定了连笔的视觉张力")
    for q, val in zip(quantiles, length_percentiles):
        print(f" P{q:<4}:  {val:6.4f}")
    print("=" * 50)

if __name__ == "__main__":
    analyze_connected_strokes()