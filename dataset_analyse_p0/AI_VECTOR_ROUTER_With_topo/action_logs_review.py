import os
import json
import glob
from collections import Counter

# 1. 配置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_LOGS_DIR = os.path.join(SCRIPT_DIR, "action_logs")

def analyze_action_logs():
    all_files = glob.glob(os.path.join(ACTION_LOGS_DIR, "*.json"))
    if not all_files:
        print("❌ No action logs found!")
        return

    total_chars = 0
    action_counts = Counter()
    trajectory_lengths = []
    action_bigrams = Counter() # 统计动作连招 (例如: Split -> Delete)

    print(f"🔍 Analyzing {len(all_files)} action log file(s)...")

    for file_path in all_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

            for hex_key, log_sequence in data.items():
                total_chars += 1
                # 过滤掉 Init (Raw) 状态，只看真实动作
                actual_actions = [step["action"] for step in log_sequence if step["action"] not in ("Init (Raw)", "Re-edit Init")]
                
                trajectory_lengths.append(len(actual_actions))
                action_counts.update(actual_actions)

                # 提取动作连招 (Bi-grams)
                for i in range(len(actual_actions) - 1):
                    # 简化动作名，去掉后面的 (M), (D) 等方便统计
                    a1 = actual_actions[i].split(" ")[0]
                    a2 = actual_actions[i+1].split(" ")[0]
                    action_bigrams[f"{a1} -> {a2}"] += 1

    if total_chars == 0:
        print("⚠️ Found files, but no valid character trajectories inside.")
        return

    # --- 打印统计报告 ---
    print("\n" + "="*40)
    print("📊 GRAPH EDITING DATASET STATISTICS")
    print("="*40)
    print(f"Total Characters Annotated: {total_chars}")
    print(f"Average Graph Edits per Char: {sum(trajectory_lengths)/total_chars:.2f} steps")
    print(f"Max Edits on a single Char: {max(trajectory_lengths)} steps")
    
    print("\n🛠️ Action Type Distribution:")
    total_actions = sum(action_counts.values())
    for act, count in action_counts.most_common():
        print(f"  - {act.ljust(15)}: {count} ({count/total_actions*100:.1f}%)")

    print("\n🔗 Top 5 Action Sequences (The 'Expert Combos'):")
    for combo, count in action_bigrams.most_common(5):
        print(f"  - {combo.ljust(20)}: {count} times")
    print("="*40)

if __name__ == "__main__":
    analyze_action_logs()