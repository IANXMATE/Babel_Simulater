import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from stage1_train import FontGraphGenerator, D_MODEL, D_EDGE, N_HEADS, N_LAYERS, NUM_EDGE_TYPES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "fontgpt_canonical_graph_latest.pth")

CANVAS_SIZE = 400.0


# ==========================================
# 🎲 Stage 1: 手工拓扑骨架（测试用）
# ==========================================
def get_random_graph_topology(num_nodes=3):
    """
    手工构造一个简单拓扑骨架。
    0:NONE, 1:E2E, 2:X, 3:T
    """
    print("\n🎲 拓扑结构 (Stage 1):")
    edges = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    ts = torch.zeros((num_nodes, num_nodes, 6), dtype=torch.float)  # [t_u, t_v, t_diff, t_prod, angle_sin, angle_cos]

    # 笔画 0 → 笔画 1：E2E（0的末端接1的起端）
    # E2E 夹角定义为 180° （切线方向相反）→ sin(180°)=0, cos(180°)=-1
    edges[0, 1] = edges[1, 0] = 1
    ts[0, 1] = ts[1, 0] = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    print("  (0,1) E2E: t_u=1.0 → P3端, t_v=0.0 → P0端, angle=180°")

    # 笔画 1 → 笔画 2：T型（2的起端挂在1的中间）
    # T型典型夹角 90° → sin(90°)=1, cos(90°)=0
    edges[1, 2] = edges[2, 1] = 3
    ts[1, 2] = ts[2, 1] = torch.tensor([0.5, 0.0, 0.5, 0.0, 1.0, 0.0])
    print("  (1,2) T: t_u=0.5 → 1的中点, t_v=0.0 → P0端, angle=90°")

    return edges, ts


# ==========================================
# 🚀 推理引擎（完全拓扑驱动，与训练对齐）
# ==========================================
@torch.no_grad()
def infer_graph(model, edge_types, edge_ts, device):
    model.eval()
    B = 1
    N = edge_types.size(0)

    b_edge_types = edge_types.unsqueeze(0).to(device)
    b_edge_ts = edge_ts.unsqueeze(0).to(device)
    b_mask = torch.zeros(B, N, dtype=torch.bool).to(device)

    # 🚀 一次前向传播（只需拓扑，无任何 GT 节点特征输入）
    shape_logits, width_logits, coords_pred, edge_preds = model(
        b_edge_types, b_edge_ts, b_mask
    )

    # decode_coords 输出 Tanh ∈ [-1,1]，表示相对字形中心的偏移
    # 可视化时加上画布中心 (0.5, 0.5) 再 * CANVAS_SIZE
    shapes = torch.argmax(shape_logits[0], dim=-1).cpu().numpy()
    widths = torch.argmax(width_logits[0], dim=-1).cpu().numpy()
    coords_raw = coords_pred[0].cpu().numpy()              # [-1,1]
    coords = (coords_raw + 1.0) / 2.0 * CANVAS_SIZE       # 还原到像素空间，中心≈200px
    # Solver 使用输入的 t 值（拓扑描述里的权威 t），不用模型预测的 t（仅辅助监督）
    # edge_ts[:,:,0]=t_u, edge_ts[:,:,1]=t_v
    topo_ts = edge_ts.numpy()  # [N, N, 6]，直接用输入 t_u/t_v

    # 仅用于诊断打印：模型预测的 t 值
    pred_ts_debug = edge_preds[0, :, :, NUM_EDGE_TYPES:NUM_EDGE_TYPES+2].cpu().numpy()

    print(f"\n📊 推理诊断:")
    for i in range(N):
        p0, p3 = coords[i, :2], coords[i, 2:]
        dist = np.linalg.norm(p3 - p0)
        print(f"  Node {i}: P0={p0.round(1)}, P3={p3.round(1)}, len={dist:.1f}px  shape={shapes[i]}, w={widths[i]}")
    for i in range(N):
        for j in range(N):
            if edge_types[i, j] > 0:
                tu_in = topo_ts[i, j, 0]
                tv_in = topo_ts[i, j, 1]
                tu_pred = pred_ts_debug[i, j, 0]
                tv_pred = pred_ts_debug[i, j, 1]
                print(f"  Edge ({i}→{j}): t_u(input)={tu_in:.3f} pred={tu_pred:.3f},  t_v(input)={tv_in:.3f} pred={tv_pred:.3f}")

    return shapes, widths, coords, topo_ts  # ← 返回输入 t（权威），不是预测 t


# ==========================================
# 🎨 渲染引擎（左:原始预测  右:Solver吸附后）
# ==========================================
def render_graph(edge_types, shapes, widths, coords, predicted_ts):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    N = len(shapes)
    j_names = {1: 'E2E', 2: 'X', 3: 'T'}

    def _draw_nodes(ax, pts, label=""):
        for i in range(N):
            p0, p3 = pts[i, :2], pts[i, 2:]
            lw = max(1, 2 + widths[i] * 2)
            c = colors[i % len(colors)]
            ax.plot([p0[0], p3[0]], [p0[1], p3[1]], color=c, linewidth=lw)
            ax.scatter(*p0, color='green', s=60, zorder=5)
            ax.scatter(*p3, color='red', s=60, zorder=5)
            mid = (p0 + p3) / 2
            ax.text(mid[0] + 5, mid[1] - 5, f"N{i}", color=c, fontsize=12, fontweight='bold')
        for i in range(N):
            for j in range(N):
                if edge_types[i, j] > 0 and i < j:
                    ci = (pts[i, :2] + pts[i, 2:]) / 2
                    cj = (pts[j, :2] + pts[j, 2:]) / 2
                    ax.plot([ci[0], cj[0]], [ci[1], cj[1]], '--', color='#999', lw=1, alpha=0.5)
                    ax.text((ci[0]+cj[0])/2, (ci[1]+cj[1])/2,
                            j_names.get(int(edge_types[i, j]), '?'),
                            fontsize=10, color='#555', ha='center')

    # ── 左图：原始输出
    ax1 = axes[0]
    ax1.set_title("Raw Model Output (No Solver)", fontsize=12)
    ax1.set_xlim(0, CANVAS_SIZE); ax1.set_ylim(CANVAS_SIZE, 0)
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
    _draw_nodes(ax1, coords)

    # ── 右图：Solver 吸附后
    ax2 = axes[1]
    ax2.set_title("After Physical Solver (Snapping)", fontsize=12)
    ax2.set_xlim(0, CANVAS_SIZE); ax2.set_ylim(CANVAS_SIZE, 0)
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)

    refined = coords.copy()
    snap_log = []

    # ── Solver：正确实现
    # 语义：predicted_ts[i,j,0]=t_u（交点在 Node_i 的 stroke 上的参数位置）
    #        predicted_ts[i,j,1]=t_v（交点在 Node_j 的 stroke 上的参数位置）
    # 规则：
    #   (1) 端点 = t < THRESH 或 t > 1-THRESH
    #   (2) 两者都是端点 → 吸附到两者的均值
    #   (3) i 是端点，j 是内部点 → i 的端点吸附到 j 的内部位置（j 是权威）
    #   (4) j 是端点，i 是内部点 → j 的端点吸附到 i 的内部位置（i 是权威）
    #   (5) 两者都是内部点（X型）→ 两者都移动到均值
    THRESH = 0.05  # 判断是否为端点的阈值
    for i in range(N):
        for j in range(i + 1, N):
            et = int(edge_types[i, j])
            if et <= 0:
                continue

            tu = predicted_ts[i, j, 0]   # t on Node i
            tv = predicted_ts[i, j, 1]   # t on Node j

            pt_i = refined[i, :2] * (1 - tu) + refined[i, 2:] * tu  # 交点在 i 上的坐标
            pt_j = refined[j, :2] * (1 - tv) + refined[j, 2:] * tv  # 交点在 j 上的坐标

            i_is_end = (tu < THRESH) or (tu > 1 - THRESH)
            j_is_end = (tv < THRESH) or (tv > 1 - THRESH)

            if i_is_end and j_is_end:
                # 两端点对接（E2E 典型场景）→ 均值对齐
                junc = (pt_i + pt_j) / 2.0
                if tu < 0.5: refined[i, :2] = junc
                else:        refined[i, 2:] = junc
                if tv < 0.5: refined[j, :2] = junc
                else:        refined[j, 2:] = junc
                snap_log.append(f"N{i}@t={tu:.2f} + N{j}@t={tv:.2f} → avg={junc.round(1)}")
            elif i_is_end:
                # i 的端点吸附到 j 的内部点（j 是权威）
                junc = pt_j
                if tu < 0.5: refined[i, :2] = junc
                else:        refined[i, 2:] = junc
                snap_log.append(f"N{i}.end(t={tu:.2f}) → N{j}@t={tv:.2f}={junc.round(1)}")
            elif j_is_end:
                # j 的端点吸附到 i 的内部点（i 是权威）
                junc = pt_i
                if tv < 0.5: refined[j, :2] = junc
                else:        refined[j, 2:] = junc
                snap_log.append(f"N{j}.end(t={tv:.2f}) → N{i}@t={tu:.2f}={junc.round(1)}")
            else:
                # X 型：两个内部点 → 均值
                junc = (pt_i + pt_j) / 2.0
                # 找最近端点并移动
                snap_log.append(f"X: N{i}@t={tu:.2f}+N{j}@t={tv:.2f} → avg={junc.round(1)}")

    _draw_nodes(ax2, refined)
    snap_text = "Snaps:\n" + ("\n".join(snap_log) if snap_log else "None")
    ax2.text(5, CANVAS_SIZE - 10, snap_text, fontsize=9, color='#444',
             verticalalignment='bottom',
             bbox=dict(facecolor='#FFFDE7', alpha=0.9, edgecolor='#FBC02D', pad=4))

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "infer_result.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\n💾 结果已保存至 {out_path}")
    plt.show()


# ==========================================
# 🎬 主函数
# ==========================================
def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    model = FontGraphGenerator().to(device)

    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        model_state = model.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in state if k not in compatible]
        model_state.update(compatible)
        model.load_state_dict(model_state)
        print(f"✅ 权重加载：{len(compatible)}/{len(state)} 层兼容")
        if skipped:
            print(f"⚠️  跳过层: {skipped}")
    else:
        print("⚠️  未找到权重文件，使用随机初始化")

    edge_types, edge_ts = get_random_graph_topology()
    shapes, widths, coords, predicted_ts = infer_graph(model, edge_types, edge_ts, device)
    render_graph(edge_types, shapes, widths, coords, predicted_ts)


if __name__ == "__main__":
    main()
