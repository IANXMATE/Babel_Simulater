import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

# ==========================================
# ⚙️ 导入图模型基建 (确保指向的是新的 stage2_graph_train)
# ==========================================
from stage1_train import FontGraphGenerator, D_MODEL, D_EDGE, N_HEADS, N_LAYERS, NUM_EDGE_TYPES
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "fontgpt_canonical_graph_latest.pth")
CLUSTER_FILE = os.path.join(SCRIPT_DIR, "clustered_results.json")

CANVAS_SIZE = 400.0

# 加载形状密码本
shape_codebook = {}
try:
    with open(CLUSTER_FILE, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
        for item in cluster_data:
            cid = int(item["cluster_id"])
            if cid != -1 and cid not in shape_codebook:
                shape_codebook[cid] = item["mother_bezier"]
    print(f"📚 成功加载 VQ-VAE 形状密码本！")
except: print("⚠️ 警告：无法加载密码本。")

# ==========================================
# 🎲 Stage 1: 拓扑生成器 (Topology Prior)
# ==========================================
def get_random_graph_topology(num_nodes=3):
    """
    手动定义一个拓扑骨架。
    0:NONE, 1:E2E, 2:X, 3:T
    """
    print("\n🎲 正在生成随机拓扑结构 (Stage 1)...")
    
    # 构造一个 num_nodes x num_nodes 的全零矩阵
    edges = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    ts = torch.zeros((num_nodes, num_nodes, 4), dtype=torch.float)
    
    # 手动定义一些拓扑关系，你可以根据需要在这里修改逻辑
    # 示例：一个简单的 3 笔画“工”字型骨架
    edges[0, 1] = edges[1, 0] = 1 # 笔画 0 和 1 端点相连 (E2E)
    ts[0, 1] = ts[1, 0] = torch.tensor([1.0, 0.0, 1.0, 0.0]) # ta=1.0, tb=0.0
    
    edges[1, 2] = edges[2, 1] = 3 # 笔画 2 挂在笔画 1 的中间 (T型)
    ts[1, 2] = ts[2, 1] = torch.tensor([0.5, 0.0, 0.5, 0.0]) # ta=0.5, tb=0.0
    
    return edges, ts

# ==========================================
# 📐 贝塞尔曲线求值 (渲染用)
# ==========================================
def get_bezier_point_single(pts, t):
    mt = 1.0 - t
    return (mt**3)*pts[0] + 3*(mt**2)*t*pts[1] + 3*mt*(t**2)*pts[2] + (t**3)*pts[3]

# ==========================================
# 🚀 推理引擎
# ==========================================
@torch.no_grad()
def infer_graph(model, edge_types, edge_ts, device):
    model.eval()
    B = 1
    N = edge_types.size(0)

    # 构造空白的节点特征：推理时无先验 shape/width/coords，全部由模型从拓扑中求解
    b_shapes = torch.zeros(B, N, dtype=torch.long).to(device)
    b_widths = torch.zeros(B, N, dtype=torch.long).to(device)
    b_coords = torch.full((B, N, 4), 0.5, dtype=torch.float).to(device)  # 初始化为画布中心

    b_edge_types = edge_types.unsqueeze(0).to(device)
    b_edge_ts = edge_ts.unsqueeze(0).to(device)
    b_mask = torch.zeros(B, N, dtype=torch.bool).to(device)

    # 🚀 一次前向传播，解算所有笔画几何
    shape_logits, width_logits, coords_pred, edge_preds = model(
        b_shapes, b_widths, b_coords, b_edge_types, b_edge_ts, b_mask
    )

    shapes = torch.argmax(shape_logits[0], dim=-1).cpu().numpy()
    widths = torch.argmax(width_logits[0], dim=-1).cpu().numpy()
    coords = coords_pred[0].cpu().numpy() * CANVAS_SIZE

    # 提取网络回归出的 t_u, t_v（用于物理吸附 Solver）
    predicted_ts = edge_preds[0, :, :, -2:].cpu().numpy()  # [N, N, 2]

    return shapes, widths, coords, predicted_ts

# ==========================================
# 🎨 渲染引擎 (含物理吸附 Solver)
# ==========================================
def render_graph(edge_types, shapes, widths, coords, predicted_ts):
    plt.figure(figsize=(8, 8))
    plt.title("FontGPT Graph - Physical Solver Inference")
    plt.xlim(0, CANVAS_SIZE); plt.ylim(CANVAS_SIZE, 0)
    
    refined_coords = coords.copy()
    N = len(shapes)
    drawn_pts = [] 

    # 物理吸附 Solver 范式转移：由网络提供先验和几何特征，直接由 solver 一次性解析拓扑。
    # 这里我们利用网络预测的 t_u, t_v 来作为节点对齐的参考。
    # 根据网络生成的边类型以及边上回归出的精确 t_a, t_b 值，微调生成笔画。
    for i in range(N):
        for j in range(N):
            if edge_types[i, j] > 0:
                tu, tv = predicted_ts[i, j, 0], predicted_ts[i, j, 1]
                # 计算出对方应该相交的点
                pt_j = refined_coords[j, :2] * (1-tv) + refined_coords[j, 2:] * tv
                # 把本笔画的关键点拉过去吸附
                if tu < 0.1: refined_coords[i, :2] = pt_j
                elif tu > 0.9: refined_coords[i, 2:] = pt_j
    
    # 绘制
    for i in range(N):
        p0, p3 = refined_coords[i, :2], refined_coords[i, 2:]
        lw = 2 + widths[i] * 2
        plt.plot([p0[0], p3[0]], [p0[1], p3[1]], 'b-', linewidth=lw)
        drawn_pts.append(np.array([p0, p0+(p3-p0)/3, p0+(p3-p0)*2/3, p3]))
        
    plt.show()

# ==========================================
# 🎬 主函数
# ==========================================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FontGraphGenerator().to(device)
    # 架构升级后兼容性加载：过滤掉形状不匹配的权重
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        model_state = model.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = [k for k in state if k not in compatible]
        model_state.update(compatible)
        model.load_state_dict(model_state)
        print(f"✅ 权重加载：{len(compatible)}/{len(state)} 层兼容")
        if skipped:
            print(f"⚠️  跳过不兼容层（架构已升级，需重新训练获得完整权重）: {skipped[:4]}")
    else:
        print(f"⚠️  未找到权重文件，使用随机初始化")
    
    # 获取纯粹的拓扑结构骨架 (只包含连接关系和粗略的位置暗示)
    edge_types, edge_ts = get_random_graph_topology()
    
    # 彻底前向抛给网络：一键算出所有笔画几何坐标、宽度，并预测节点吸附参数(预测的 t)
    shapes, widths, coords, predicted_ts = infer_graph(model, edge_types, edge_ts, device)
    
    # Solver 利用预测网络提供的物理参量进行精确的拓扑微调吸附
    render_graph(edge_types, shapes, widths, coords, predicted_ts)

if __name__ == "__main__":
    main()