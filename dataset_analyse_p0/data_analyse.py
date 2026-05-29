import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_stroke_quantiles(data_path="omniglot_parametric_7d.pt"):
    print(f"📦 正在加载并解剖数据集: {data_path}...")
    try:
        data = torch.load(data_path)
    except Exception as e:
        print(f"❌ 加载失败，请检查路径: {e}")
        return

    # ==========================================
    # 1. 精准提取笔画数 (Math Equation Count)
    # ==========================================
    stroke_counts = []
    for sample in data:
        seq = sample["sequence"]
        
        # 在 7D 数据中，笔画数其实就是寻找 P==1.0 (结束符) 之前的有效方程组数量
        P_column = seq[:, 6]
        end_idx = (P_column == 1.0).nonzero(as_tuple=True)[0]
        
        if len(end_idx) > 0:
            # 比如 end_idx 是 5，说明有 0,1,2,3,4 这 5 个有效笔画
            strokes = end_idx[0].item() 
        else:
            # 容错：如果没有结束符，整个长度就是笔画数
            strokes = len(seq) 
            
        stroke_counts.append(strokes)

    stroke_counts = np.array(stroke_counts)
    total_samples = len(stroke_counts)
    max_strokes = stroke_counts.max()

    # ==========================================
    # 2. 终端统计播报
    # ==========================================
    print(f"\n📊 扫描完毕！总计字符样本: {total_samples}")
    print(f"➖ 极简字符: {stroke_counts.min()} 笔")
    print(f"➕ 终极狂草: {stroke_counts.max()} 笔")
    print(f"📉 平均笔画: {stroke_counts.mean():.2f} 笔")
    print(f"🎯 绝对中位数 (50% 的字符少于该笔画): {np.median(stroke_counts):.0f} 笔")

    # ==========================================
    # 3. 核心计算：以 1 为间隔的累积分布 (分位数)
    # ==========================================
    counter = Counter(stroke_counts)
    
    print("\n📈 笔画数阶梯分位数 (Cumulative Quantiles):")
    print("=" * 65)
    print(f"{'笔画数限制':<15} | {'覆盖字符数':<15} | {'数据集累积占比 (Quantile)':<20}")
    print("-" * 65)

    cumulative_count = 0
    x_ticks = []
    cumulative_percentages = []

    for i in range(1, max_strokes + 1):
        count = counter.get(i, 0)
        cumulative_count += count
        quantile = (cumulative_count / total_samples) * 100
        
        x_ticks.append(i)
        cumulative_percentages.append(quantile)
        
        # 过滤掉中间完全没有数据的空档期，只打印关键信息
        if count > 0 or i % 5 == 0 or i == max_strokes:
            print(f"<= {i:<12} | {cumulative_count:<12} | {quantile:>6.2f}%")

    print("=" * 65)

    # ==========================================
    # 4. 赛博风双轴数据可视化
    # ==========================================
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    bg_color = '#030613'
    fig.patch.set_facecolor(bg_color)
    ax1.set_facecolor(bg_color)
    ax1.grid(color='#0a192f', linestyle='--', linewidth=0.5, alpha=0.5)

    # 左轴：频次直方图 (PDF)
    bins = np.arange(1, max_strokes + 2) - 0.5
    ax1.hist(stroke_counts, bins=bins, color='#00F0FF', alpha=0.6, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Number of Strokes (Equations)', fontsize=12, color='#00F0FF')
    ax1.set_ylabel('Character Count (Frequency)', fontsize=12, color='#00F0FF')
    ax1.tick_params(axis='x', colors='#00F0FF')
    ax1.tick_params(axis='y', colors='#00F0FF')

    # 右轴：累积分布折线图 (CDF)
    ax2 = ax1.twinx()
    ax2.plot(x_ticks, cumulative_percentages, color='#FF003C', marker='o', markersize=4, linewidth=2, label='CDF (Quantile)')
    ax2.set_ylabel('Cumulative Quantile (%)', fontsize=12, color='#FF003C', fontweight='bold')
    ax2.tick_params(axis='y', colors='#FF003C')
    ax2.set_ylim(0, 105)

    # 标注几个关键的工业界常用拦截位 (90%, 95%, 99%)
    quantiles_to_mark = [90.0, 95.0, 99.0]
    for q in quantiles_to_mark:
        # 找到刚刚超过这个分位数的笔画数
        idx = next((i for i, v in enumerate(cumulative_percentages) if v >= q), None)
        if idx is not None:
            stroke_val = x_ticks[idx]
            ax2.axhline(y=q, color='#FFFFFF', linestyle=':', alpha=0.4)
            ax2.axvline(x=stroke_val, color='#FFFFFF', linestyle=':', alpha=0.4)
            ax2.text(stroke_val + 0.5, q - 3, f'{q}% -> {stroke_val} strokes', color='#FFFFFF', fontsize=9)

    plt.title('7D Parametric Dataset: Stroke Distribution & Quantiles', fontsize=16, fontweight='bold', color='white', pad=20)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_stroke_quantiles()