import os
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from stage1_train import FontGraphGenerator, NUM_EDGE_TYPES

# ==========================================
# ⚙️ 路径与配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "fontgpt_canonical_graph_latest.pth")
DATASET_FILE = os.path.join(SCRIPT_DIR, "fontgpt_dataset_graph.json")

CANVAS_SIZE = 400.0
GRID_BINS = 32.0

# 可选：固定随机种子方便复现；设为 None 就每次随机
RANDOM_SEED = None

# 可选：指定样本 index；设为 None 就随机抽样
SAMPLE_INDEX = None

# 可选：过滤节点数量，避免随机到太复杂或太简单的样本
MIN_NODES = 2
MAX_NODES_FILTER = 12

# 右图是否使用物理 Solver 后结果
# 如果你想看“纯模型推理能力”，建议 False
# 如果你想看“模型 + 拓扑吸附后的最终效果”，设 True
APPLY_SOLVER_FOR_RIGHT = False


# ==========================================
# 📦 从真实 dataset 随机读取一个样本
# ==========================================
def load_random_dataset_sample(dataset_file, sample_index=None):
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"未找到数据集文件: {dataset_file}")

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise RuntimeError("数据集为空。")

    candidates = [
        (idx, item) for idx, item in enumerate(data)
        if MIN_NODES <= item.get("num_nodes", 0) <= MAX_NODES_FILTER
    ]

    if not candidates:
        raise RuntimeError(
            f"没有满足节点数过滤条件的样本: MIN_NODES={MIN_NODES}, MAX_NODES_FILTER={MAX_NODES_FILTER}"
        )

    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(data):
            raise IndexError(f"SAMPLE_INDEX 越界: {sample_index}, dataset size={len(data)}")
        item = data[sample_index]
        idx = sample_index
    else:
        idx, item = random.choice(candidates)

    return idx, item


# ==========================================
# 🧱 将 JSON 样本转成模型输入 + GT 可视化数据
# ==========================================
def build_sample_tensors(item):
    """
    返回：
    - edge_types: [N, N]
    - edge_ts:    [N, N, 6]
    - gt_shapes:  [N]
    - gt_widths:  [N]
    - gt_coords_px: [N, 4]，原始像素坐标，用于左图
    - gt_center_norm: [2]，训练时减掉的中心，用于把预测中心化坐标还原回像素空间
    """
    num_nodes = item["num_nodes"]

    gt_shapes = np.zeros(num_nodes, dtype=np.int64)
    gt_widths = np.zeros(num_nodes, dtype=np.int64)
    gt_coords_norm = np.zeros((num_nodes, 4), dtype=np.float32)

    # 节点 GT
    for n in item["nodes"]:
        i = int(n["node_id"])

        gt_shapes[i] = int(n["shape_code"])
        gt_widths[i] = int(n["width_token"])

        p0_x = (n["p0_cell"][0] + n["p0_offset"][0]) / GRID_BINS
        p0_y = (n["p0_cell"][1] + n["p0_offset"][1]) / GRID_BINS
        p3_x = (n["p3_cell"][0] + n["p3_offset"][0]) / GRID_BINS
        p3_y = (n["p3_cell"][1] + n["p3_offset"][1]) / GRID_BINS

        gt_coords_norm[i] = np.array([p0_x, p0_y, p3_x, p3_y], dtype=np.float32)

    # 训练时使用的 glyph center：所有端点均值
    all_pts = gt_coords_norm.reshape(-1, 2)
    gt_center_norm = all_pts.mean(axis=0).astype(np.float32)

    # 左图原始像素坐标：不中心化，直接画原样
    gt_coords_px = gt_coords_norm * CANVAS_SIZE

    # 边输入
    edge_types = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    edge_ts = torch.zeros((num_nodes, num_nodes, 6), dtype=torch.float32)

    for e in item["edges"]:
        u = int(e["u"])
        v = int(e["v"])

        edge_types[u, v] = int(e["j_type_idx"])
        edge_ts[u, v] = torch.tensor(
            [
                float(e["t_u"]),
                float(e["t_v"]),
                float(e.get("t_diff", abs(float(e["t_u"]) - float(e["t_v"])))),
                float(e.get("t_prod", float(e["t_u"]) * float(e["t_v"]))),
                float(e.get("angle_sin", 0.0)),
                float(e.get("angle_cos", 1.0)),
            ],
            dtype=torch.float32,
        )

    return (
        edge_types,
        edge_ts,
        gt_shapes,
        gt_widths,
        gt_coords_px,
        gt_center_norm,
    )


# ==========================================
# 🔍 检查双向边 t 是否互换
# ==========================================
def print_reverse_edge_check(edge_types, edge_ts):
    print("\n🔍 双向边 t 检查:")

    N = edge_types.size(0)
    found = False

    for i in range(N):
        for j in range(i + 1, N):
            if edge_types[i, j].item() > 0 or edge_types[j, i].item() > 0:
                found = True

                et_ij = int(edge_types[i, j].item())
                et_ji = int(edge_types[j, i].item())

                tu_ij = float(edge_ts[i, j, 0].item())
                tv_ij = float(edge_ts[i, j, 1].item())
                tu_ji = float(edge_ts[j, i, 0].item())
                tv_ji = float(edge_ts[j, i, 1].item())

                ok = (
                    et_ij == et_ji
                    and abs(tu_ij - tv_ji) < 1e-4
                    and abs(tv_ij - tu_ji) < 1e-4
                )

                status = "✅" if ok else "❌"
                print(
                    f"  {status} ({i},{j}) type {et_ij}/{et_ji} | "
                    f"{i}->{j}: [{tu_ij:.3f}, {tv_ij:.3f}] | "
                    f"{j}->{i}: [{tu_ji:.3f}, {tv_ji:.3f}]"
                )

    if not found:
        print("  没有正边。")


# ==========================================
# 🚀 推理：用真实样本拓扑作为输入
# ==========================================
@torch.no_grad()
def infer_from_topology(model, edge_types, edge_ts, gt_center_norm, device):
    model.eval()

    N = edge_types.size(0)

    b_edge_types = edge_types.unsqueeze(0).to(device)
    b_edge_ts = edge_ts.unsqueeze(0).to(device)
    b_mask = torch.zeros(1, N, dtype=torch.bool, device=device)

    shape_logits, width_logits, coords_pred, edge_preds = model(
        b_edge_types,
        b_edge_ts,
        b_mask,
    )

    pred_shapes = torch.argmax(shape_logits[0], dim=-1).cpu().numpy()
    pred_widths = torch.argmax(width_logits[0], dim=-1).cpu().numpy()

    # 关键：
    # 训练时 gt_coords = raw_norm - center
    # 所以模型输出 coords_pred 也是中心化坐标
    # 可视化时要加回该样本自己的 center，而不是 (x+1)/2
    pred_centered = coords_pred[0].cpu().numpy()  # expected around [-0.5, 0.5]
    pred_norm = pred_centered.copy()

    pred_norm[:, 0] += gt_center_norm[0]  # P0_x
    pred_norm[:, 1] += gt_center_norm[1]  # P0_y
    pred_norm[:, 2] += gt_center_norm[0]  # P3_x
    pred_norm[:, 3] += gt_center_norm[1]  # P3_y

    pred_coords_px = pred_norm * CANVAS_SIZE

    pred_edge_t = edge_preds[0, :, :, NUM_EDGE_TYPES:NUM_EDGE_TYPES + 2].cpu().numpy()

    return pred_shapes, pred_widths, pred_coords_px, pred_edge_t


# ==========================================
# 🧲 可选物理 Solver：使用输入拓扑 t，而不是预测 t
# ==========================================
def apply_physical_solver(coords_px, edge_types, edge_ts):
    refined = coords_px.copy()
    N = refined.shape[0]
    snap_log = []

    THRESH = 0.05

    edge_types_np = edge_types.cpu().numpy()
    edge_ts_np = edge_ts.cpu().numpy()

    for i in range(N):
        for j in range(i + 1, N):
            et = int(edge_types_np[i, j])
            if et <= 0:
                continue

            tu = float(edge_ts_np[i, j, 0])
            tv = float(edge_ts_np[i, j, 1])

            pt_i = refined[i, :2] * (1.0 - tu) + refined[i, 2:] * tu
            pt_j = refined[j, :2] * (1.0 - tv) + refined[j, 2:] * tv

            i_is_end = (tu < THRESH) or (tu > 1.0 - THRESH)
            j_is_end = (tv < THRESH) or (tv > 1.0 - THRESH)

            if i_is_end and j_is_end:
                junc = (pt_i + pt_j) / 2.0

                if tu < 0.5:
                    refined[i, :2] = junc
                else:
                    refined[i, 2:] = junc

                if tv < 0.5:
                    refined[j, :2] = junc
                else:
                    refined[j, 2:] = junc

                snap_log.append(
                    f"E2E-like: N{i}@{tu:.2f} + N{j}@{tv:.2f} -> {junc.round(1)}"
                )

            elif i_is_end:
                junc = pt_j

                if tu < 0.5:
                    refined[i, :2] = junc
                else:
                    refined[i, 2:] = junc

                snap_log.append(
                    f"T-like: N{i}.end@{tu:.2f} -> N{j}@{tv:.2f} {junc.round(1)}"
                )

            elif j_is_end:
                junc = pt_i

                if tv < 0.5:
                    refined[j, :2] = junc
                else:
                    refined[j, 2:] = junc

                snap_log.append(
                    f"T-like: N{j}.end@{tv:.2f} -> N{i}@{tu:.2f} {junc.round(1)}"
                )

            else:
                # X 型内部交叉：这里只记录，不移动端点
                junc = (pt_i + pt_j) / 2.0
                snap_log.append(
                    f"X-like: N{i}@{tu:.2f} + N{j}@{tv:.2f} -> {junc.round(1)}"
                )

    return refined, snap_log


# ==========================================
# 📊 打印推理诊断
# ==========================================
def print_diagnostics(
    edge_types,
    edge_ts,
    gt_shapes,
    gt_widths,
    gt_coords_px,
    pred_shapes,
    pred_widths,
    pred_coords_px,
    pred_edge_t,
):
    N = len(gt_shapes)

    print("\n📊 Node 诊断:")
    for i in range(N):
        gt_p0 = gt_coords_px[i, :2]
        gt_p3 = gt_coords_px[i, 2:]
        pr_p0 = pred_coords_px[i, :2]
        pr_p3 = pred_coords_px[i, 2:]

        gt_len = np.linalg.norm(gt_p3 - gt_p0)
        pr_len = np.linalg.norm(pr_p3 - pr_p0)

        print(
            f"  Node {i}: "
            f"GT len={gt_len:7.1f}px shape={gt_shapes[i]:4d} w={gt_widths[i]} | "
            f"Pred len={pr_len:7.1f}px shape={pred_shapes[i]:4d} w={pred_widths[i]} | "
            f"P0_pred={pr_p0.round(1)} P3_pred={pr_p3.round(1)}"
        )

    print("\n📊 Edge t 诊断:")
    for i in range(N):
        for j in range(N):
            if edge_types[i, j].item() > 0:
                tu_in = float(edge_ts[i, j, 0].item())
                tv_in = float(edge_ts[i, j, 1].item())
                tu_pred = float(pred_edge_t[i, j, 0])
                tv_pred = float(pred_edge_t[i, j, 1])

                print(
                    f"  Edge ({i}->{j}): "
                    f"t_u(input)={tu_in:.3f}, pred={tu_pred:.3f} | "
                    f"t_v(input)={tv_in:.3f}, pred={tv_pred:.3f}"
                )


# ==========================================
# 🎨 渲染 GT vs Pred
# ==========================================
def render_gt_vs_pred(
    sample_idx,
    item,
    edge_types,
    edge_ts,
    gt_shapes,
    gt_widths,
    gt_coords_px,
    pred_shapes,
    pred_widths,
    pred_coords_px,
):
    if APPLY_SOLVER_FOR_RIGHT:
        right_coords, snap_log = apply_physical_solver(pred_coords_px, edge_types, edge_ts)
        right_title = "Model Inference + Physical Solver"
    else:
        right_coords = pred_coords_px
        snap_log = []
        right_title = "Raw Model Inference"

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
        "#00ACC1", "#6D4C41", "#C0CA33", "#3949AB", "#D81B60"
    ]

    j_names = {1: "E2E", 2: "X", 3: "T"}
    N = len(gt_shapes)

    def setup_ax(ax, title):
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, CANVAS_SIZE)
        ax.set_ylim(CANVAS_SIZE, 0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    def draw_graph(ax, coords_px, shapes, widths, prefix):
        # draw strokes
        for i in range(N):
            p0 = coords_px[i, :2]
            p3 = coords_px[i, 2:]
            c = colors[i % len(colors)]
            lw = max(1.5, 2.0 + int(widths[i]) * 1.5)

            ax.plot(
                [p0[0], p3[0]],
                [p0[1], p3[1]],
                color=c,
                linewidth=lw,
                solid_capstyle="round",
                alpha=0.95,
            )

            ax.scatter(p0[0], p0[1], color="green", s=50, zorder=5)
            ax.scatter(p3[0], p3[1], color="red", s=50, zorder=5)

            mid = (p0 + p3) / 2.0
            length = np.linalg.norm(p3 - p0)

            ax.text(
                mid[0] + 4,
                mid[1] - 4,
                f"N{i}\nS{int(shapes[i])}, W{int(widths[i])}\nL{length:.0f}",
                color=c,
                fontsize=8,
                fontweight="bold",
            )

        # draw topology edges
        edge_types_np = edge_types.cpu().numpy()
        edge_ts_np = edge_ts.cpu().numpy()

        for i in range(N):
            for j in range(i + 1, N):
                et = int(edge_types_np[i, j])
                if et <= 0:
                    continue

                tu = edge_ts_np[i, j, 0]
                tv = edge_ts_np[i, j, 1]

                pt_i = coords_px[i, :2] * (1.0 - tu) + coords_px[i, 2:] * tu
                pt_j = coords_px[j, :2] * (1.0 - tv) + coords_px[j, 2:] * tv
                junc = (pt_i + pt_j) / 2.0

                ax.plot(
                    [pt_i[0], pt_j[0]],
                    [pt_i[1], pt_j[1]],
                    "--",
                    color="#777",
                    linewidth=1,
                    alpha=0.55,
                )

                ax.scatter(junc[0], junc[1], color="#222", s=25, zorder=6)

                ax.text(
                    junc[0] + 3,
                    junc[1] + 3,
                    f"{j_names.get(et, '?')}\n{tu:.2f},{tv:.2f}",
                    fontsize=7,
                    color="#333",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1),
                )

    char = item.get("char", "")
    hex_key = item.get("hex_key", "")
    derivation = item.get("derivation", "")

    setup_ax(
        axes[0],
        f"GT Original Graph | idx={sample_idx}, char={char}, hex={hex_key}"
    )
    draw_graph(axes[0], gt_coords_px, gt_shapes, gt_widths, "GT")

    setup_ax(
        axes[1],
        f"{right_title} | deriv={derivation}"
    )
    draw_graph(axes[1], right_coords, pred_shapes, pred_widths, "Pred")

    if snap_log:
        snap_text = "Snaps:\n" + "\n".join(snap_log[:12])
        if len(snap_log) > 12:
            snap_text += f"\n... +{len(snap_log) - 12} more"
        axes[1].text(
            5,
            CANVAS_SIZE - 10,
            snap_text,
            fontsize=8,
            color="#444",
            verticalalignment="bottom",
            bbox=dict(facecolor="#FFFDE7", alpha=0.9, edgecolor="#FBC02D", pad=4),
        )

    plt.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, f"infer_dataset_sample_{sample_idx}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n💾 结果已保存至: {out_path}")

    plt.show()


# ==========================================
# 🧠 加载模型
# ==========================================
def load_model(device):
    model = FontGraphGenerator().to(device)

    if not os.path.exists(MODEL_PATH):
        print("⚠️ 未找到权重文件，使用随机初始化")
        return model

    state = torch.load(MODEL_PATH, map_location=device)
    model_state = model.state_dict()

    compatible = {
        k: v for k, v in state.items()
        if k in model_state and model_state[k].shape == v.shape
    }

    skipped = [k for k in state.keys() if k not in compatible]

    model_state.update(compatible)
    model.load_state_dict(model_state)

    print(f"✅ 权重加载：{len(compatible)}/{len(state)} 层兼容")
    if skipped:
        print(f"⚠️ 跳过层数量: {len(skipped)}")
        for k in skipped[:10]:
            print(f"  - {k}")
        if len(skipped) > 10:
            print(f"  ... +{len(skipped) - 10} more")

    return model


# ==========================================
# 🎬 主函数
# ==========================================
def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    sample_idx, item = load_random_dataset_sample(DATASET_FILE, SAMPLE_INDEX)

    print("\n🎲 抽到真实样本:")
    print(f"  dataset_idx: {sample_idx}")
    print(f"  source_file: {item.get('source_file', '')}")
    print(f"  char: {item.get('char', '')}")
    print(f"  hex_key: {item.get('hex_key', '')}")
    print(f"  derivation: {item.get('derivation', '')}")
    print(f"  num_nodes: {item.get('num_nodes')}")
    print(f"  num_edges: {item.get('num_edges')}")

    (
        edge_types,
        edge_ts,
        gt_shapes,
        gt_widths,
        gt_coords_px,
        gt_center_norm,
    ) = build_sample_tensors(item)

    print_reverse_edge_check(edge_types, edge_ts)

    model = load_model(device)

    pred_shapes, pred_widths, pred_coords_px, pred_edge_t = infer_from_topology(
        model=model,
        edge_types=edge_types,
        edge_ts=edge_ts,
        gt_center_norm=gt_center_norm,
        device=device,
    )

    print_diagnostics(
        edge_types=edge_types,
        edge_ts=edge_ts,
        gt_shapes=gt_shapes,
        gt_widths=gt_widths,
        gt_coords_px=gt_coords_px,
        pred_shapes=pred_shapes,
        pred_widths=pred_widths,
        pred_coords_px=pred_coords_px,
        pred_edge_t=pred_edge_t,
    )

    render_gt_vs_pred(
        sample_idx=sample_idx,
        item=item,
        edge_types=edge_types,
        edge_ts=edge_ts,
        gt_shapes=gt_shapes,
        gt_widths=gt_widths,
        gt_coords_px=gt_coords_px,
        pred_shapes=pred_shapes,
        pred_widths=pred_widths,
        pred_coords_px=pred_coords_px,
    )


if __name__ == "__main__":
    main()