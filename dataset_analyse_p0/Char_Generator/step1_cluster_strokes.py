import os
import json
import glob
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

# 配置数据路径 (动态获取绝对路径)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../AI_VECTOR_ROUTER_With_topo/annotations_topo"))

MAX_STROKES = 2000 # 为防止距离矩阵计算过久，建议先截取部分笔画测试

distance_threshold = 0.03
# ==========================================
# 📐 1. 基础几何与函数化重采样
# ==========================================
def get_bezier_point(pts, t):
    mt = 1 - t
    return (mt**3)*pts[0] + 3*(mt**2)*t*pts[1] + 3*mt*(t**2)*pts[2] + (t**3)*pts[3]

def normalize_and_sample_function(mother_bezier, N=500):
    """
    归一化，并强制进行“等 X 轴重采样”，使其成为严格的离散函数 y = f(x)。
    """
    pts = np.array(mother_bezier)
    if np.isnan(pts).any() or np.isinf(pts).any(): return None

    p0 = pts[0]
    pts_t = pts - p0
    
    dists_from_p0 = np.linalg.norm(pts_t, axis=1)
    max_idx = np.argmax(dists_from_p0)
    vec = pts_t[max_idx]
    L = dists_from_p0[max_idx]
    
    if L < 1e-5: return None
        
    cos_theta, sin_theta = vec[0]/L, vec[1]/L
    R = np.array([[cos_theta, sin_theta],
                  [-sin_theta, cos_theta]])
    pts_r = pts_t @ R.T
    
    pts_norm = pts_r / L
    
    ts_dense = np.linspace(0, 1, 300)[:, None]
    curve_dense = get_bezier_point(pts_norm, ts_dense)
    
    # 强制 x 轴单调递增化处理 (确保它是函数)
    x_dense = np.maximum.accumulate(curve_dense[:, 0])
    y_dense = curve_dense[:, 1]
    
    if x_dense[-1] < 1e-5: return None
    x_dense = x_dense / x_dense[-1] 
    
    target_x = np.linspace(0, 1.0, N)
    y_eq = np.interp(target_x, x_dense, y_dense)
    
    return y_eq

def get_4_isomorphisms_y_only(Y):
    Y0 = Y.copy()
    Y1 = -Y[::-1] # 起终点倒转
    Y2 = -Y       # X 轴翻转
    Y3 = Y[::-1]  # 中心对称
    return [Y0, Y1, Y2, Y3]

# ==========================================
# 📏 2. 微积分距离：函数绝对面积差
# ==========================================
def get_integral_distance(Y_A, Y_B):
    """
    计算 ∫ |y_A(x) - y_B(x)| dx (黎曼面积差)
    """
    vars_B = get_4_isomorphisms_y_only(Y_B)
    min_dist = float('inf')
    
    for Y_B_var in vars_B:
        area_diff = np.mean(np.abs(Y_A - Y_B_var))
        if area_diff < min_dist: 
            min_dist = area_diff
            
    return min_dist

# ==========================================
# 🚀 3. 主程序
# ==========================================
def main():
    json_files = glob.glob(os.path.join(DATA_DIR, "*_topo.json"))
    all_strokes = []
    
    print(f"📁 数据目录: {DATA_DIR}")
    for fp in json_files:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for hex_key, char_data in data.items():
                for s in char_data.get("strokes", []):
                    all_strokes.append({
                        "hex_key": hex_key,
                        "bezier_id": s["bezier_id"],
                        "mother_bezier": s["mother_bezier"]
                    })

    if len(all_strokes) == 0: return
    if len(all_strokes) > MAX_STROKES: all_strokes = all_strokes[:MAX_STROKES]

    print(f"提取了 {len(all_strokes)} 条笔画，开始 X 轴定积分重采样...")
    valid_strokes = []
    sampled_y_arrays = []
    for s in all_strokes:
        Y = normalize_and_sample_function(s["mother_bezier"], N=50)
        if Y is not None:
            valid_strokes.append(s)
            sampled_y_arrays.append(Y)

    N = len(valid_strokes)
    if N == 0: return

    dist_matrix = np.zeros((N, N))
    print(f"⚡ 开始计算积分面积差距离矩阵...")
    
    for i in tqdm(range(N)):
        for j in range(i + 1, N):
            d = get_integral_distance(sampled_y_arrays[i], sampled_y_arrays[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    np.fill_diagonal(dist_matrix, 0.0)

    print("🤖 正在运行 层次聚类 (Agglomerative Clustering)...")
    try:
        # 🌟 核心破局点：使用层次聚类的 average linkage 强行斩断连续渐变！
        # distance_threshold 决定了你的 Token 颗粒度。
        # 0.02 表示：只有平均面积偏差小于 2% 的线条，才能被放进同一个桶里。
        clusterer = AgglomerativeClustering(
            n_clusters=None, 
            metric='precomputed', 
            linkage='average',
            distance_threshold=distance_threshold # 🔥 调节这个值：越小桶越多，越大桶越少
        )
        labels = clusterer.fit_predict(dist_matrix)
    except Exception as e:
        print(f"❌ 聚类异常: {str(e)}")
        return

    out_data = []
    for i, s in enumerate(valid_strokes):
        out_data.append({
            "hex_key": s["hex_key"],
            "bezier_id": s["bezier_id"],
            "mother_bezier": s["mother_bezier"],
            "cluster_id": int(labels[i])
        })

    out_file = os.path.join(SCRIPT_DIR, "clustered_results.json")
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
        
    n_clusters = len(set(labels))
    print(f"🎉 聚类完成！共切分出 {n_clusters} 个 Shape Token 桶。")

if __name__ == "__main__":
    main()