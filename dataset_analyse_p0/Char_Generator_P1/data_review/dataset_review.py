import os
import json
from collections import Counter

# ==========================================
# ⚙️ 配置路径
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(SCRIPT_DIR, "../fontgpt_dataset.json")

def analyze_dataset():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ 找不到文件: {DATASET_FILE}")
        return

    print(f"📖 正在读取数据集: {DATASET_FILE} ...")
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    shape_counter = Counter()
    junction_counter = Counter()
    
    total_strokes = 0
    total_junctions = 0
    total_sequences = len(dataset)

    # 遍历统计
    for item in dataset:
        sequence = item.get("sequence", [])
        for token in sequence:
            if token.get("token_type") == "STROKE":
                shape_counter[token.get("shape_code", -1)] += 1
                total_strokes += 1
            elif token.get("token_type") == "JUNCTION":
                junction_counter[token.get("j_type", "Unknown")] += 1
                total_junctions += 1

    # ==========================================
    # 📈 打印统计报告
    # ==========================================
    print("\n" + "="*50)
    print(f"🧩 汉字矢量大模型 (FontGPT) 数据集体检报告")
    print("="*50)
    print(f"总序列样本数 : {total_sequences}")
    
    print("\n" + "-"*50)
    print(f"📏 【STROKE】 Shape 聚类形态分布 (Top 15) | 总计: {total_strokes} 笔")
    print("-"*50)
    for shape, count in shape_counter.most_common(15):
        pct = (count / total_strokes) * 100
        # 用进度条可视化占比
        bar_len = int(pct / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"Shape {shape:<4} | {count:<6} 次 | {pct:>5.1f}% | {bar}")

    print("\n" + "-"*50)
    print(f"🔗 【JUNCTION】 拓扑连接类型分布 | 总计: {total_junctions} 个")
    print("-"*50)
    for j_type, count in junction_counter.most_common():
        pct = (count / total_junctions) * 100
        bar_len = int(pct / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"Type {j_type:<5} | {count:<6} 次 | {pct:>5.1f}% | {bar}")

if __name__ == "__main__":
    analyze_dataset()