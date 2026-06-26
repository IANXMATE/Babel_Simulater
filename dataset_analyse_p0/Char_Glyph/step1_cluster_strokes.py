import os
import json
import glob
import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

# ==========================================
# ⚙️ 配置数据路径
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../AI_VECTOR_ROUTER_With_topo/annotations_topo")
)

OUT_FILE = os.path.join(SCRIPT_DIR, "clustered_results.json")

MAX_STROKES = 20000
RANDOM_SAMPLE_WHEN_TOO_MANY = True
RANDOM_SEED = 42

distance_threshold = 0.03


# ==========================================
# 🆔 UID 工具
# ==========================================
def make_glyph_uid(source_file, hex_key):
    return f"{source_file}::{hex_key}"


def make_stroke_uid(source_file, hex_key, bezier_id):
    return f"{source_file}::{hex_key}::{bezier_id}"


# ==========================================
# 📐 1. 基础几何与函数化重采样
# ==========================================
def get_bezier_point(pts, t):
    mt = 1 - t
    return (
        (mt ** 3) * pts[0]
        + 3 * (mt ** 2) * t * pts[1]
        + 3 * mt * (t ** 2) * pts[2]
        + (t ** 3) * pts[3]
    )


def normalize_and_sample_function(mother_bezier, N=500):
    """
    归一化，并强制进行“等 X 轴重采样”，使其成为离散函数 y = f(x)。

    注意：
    这个表示适合把 stroke shape 分桶；
    不适合作为最终几何，因为它会丢掉绝对位置、旋转、尺度。
    """
    pts = np.array(mother_bezier, dtype=float)

    if np.isnan(pts).any() or np.isinf(pts).any():
        return None

    p0 = pts[0]
    pts_t = pts - p0

    dists_from_p0 = np.linalg.norm(pts_t, axis=1)
    max_idx = np.argmax(dists_from_p0)

    vec = pts_t[max_idx]
    L = dists_from_p0[max_idx]

    if L < 1e-5:
        return None

    cos_theta = vec[0] / L
    sin_theta = vec[1] / L

    R = np.array([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ])

    pts_r = pts_t @ R.T
    pts_norm = pts_r / L

    ts_dense = np.linspace(0, 1, 300)[:, None]
    curve_dense = get_bezier_point(pts_norm, ts_dense)

    # 强制 x 单调，避免局部回折导致插值异常
    x_dense = np.maximum.accumulate(curve_dense[:, 0])
    y_dense = curve_dense[:, 1]

    if x_dense[-1] < 1e-5:
        return None

    x_dense = x_dense / x_dense[-1]

    target_x = np.linspace(0, 1.0, N)
    y_eq = np.interp(target_x, x_dense, y_dense)

    return y_eq


def get_4_isomorphisms_y_only(Y):
    """
    同一 stroke 形状的 4 种等价形态。

    Y0: 原方向
    Y1: 起终点倒转 + 上下翻转
    Y2: X 轴翻转
    Y3: 中心对称 / 倒序
    """
    Y0 = Y.copy()
    Y1 = -Y[::-1]
    Y2 = -Y
    Y3 = Y[::-1]
    return [Y0, Y1, Y2, Y3]


# ==========================================
# 📏 2. 距离：函数绝对面积差
# ==========================================
def get_integral_distance(Y_A, Y_B):
    """
    计算 min over 4 isomorphisms:
        mean |Y_A - transform(Y_B)|

    这会把同一 stroke 的翻转 / 倒转形态放进同一个桶。
    """
    vars_B = get_4_isomorphisms_y_only(Y_B)
    min_dist = float("inf")

    for Y_B_var in vars_B:
        area_diff = np.mean(np.abs(Y_A - Y_B_var))
        if area_diff < min_dist:
            min_dist = area_diff

    return min_dist


# ==========================================
# 🚀 3. 主程序
# ==========================================
def main():
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_topo.json")))

    all_strokes = []

    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📄 topo 文件数: {len(json_files)}")

    for fp in json_files:
        source_file = os.path.basename(fp)

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        for hex_key, char_data in data.items():
            glyph_uid = make_glyph_uid(source_file, hex_key)
            char = char_data.get("glyph_info", {}).get("char", "")

            for s in char_data.get("strokes", []):
                bezier_id = s["bezier_id"]
                stroke_uid = make_stroke_uid(source_file, hex_key, bezier_id)

                all_strokes.append({
                    "source_file": source_file,
                    "glyph_uid": glyph_uid,
                    "stroke_uid": stroke_uid,

                    "hex_key": hex_key,
                    "char": char,
                    "bezier_id": bezier_id,

                    "mother_bezier": s["mother_bezier"],

                    # 可选：保存宽度，后面 primitive library 也能复用
                    "width_mean": s.get("width_mean", None),
                })

    if len(all_strokes) == 0:
        print("❌ 没有提取到任何 stroke")
        return

    print(f"📦 原始提取 stroke 数: {len(all_strokes)}")

    # 避免总是截断到 alphabetically 排前的几个字体
    if len(all_strokes) > MAX_STROKES:
        if RANDOM_SAMPLE_WHEN_TOO_MANY:
            random.seed(RANDOM_SEED)
            all_strokes = random.sample(all_strokes, MAX_STROKES)
            print(f"⚠️ stroke 数超过 MAX_STROKES，随机采样 {MAX_STROKES} 条")
        else:
            all_strokes = all_strokes[:MAX_STROKES]
            print(f"⚠️ stroke 数超过 MAX_STROKES，截取前 {MAX_STROKES} 条")

    print(f"提取了 {len(all_strokes)} 条笔画，开始 X 轴定积分重采样...")

    valid_strokes = []
    sampled_y_arrays = []

    for s in tqdm(all_strokes, desc="Normalize strokes"):
        Y = normalize_and_sample_function(s["mother_bezier"], N=50)
        if Y is not None:
            valid_strokes.append(s)
            sampled_y_arrays.append(Y)

    N = len(valid_strokes)

    if N == 0:
        print("❌ 没有有效 stroke")
        return

    print(f"✅ 有效 stroke 数: {N}")
    print("⚡ 开始计算积分面积差距离矩阵...")

    dist_matrix = np.zeros((N, N), dtype=np.float32)

    for i in tqdm(range(N), desc="Distance matrix"):
        for j in range(i + 1, N):
            d = get_integral_distance(sampled_y_arrays[i], sampled_y_arrays[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    np.fill_diagonal(dist_matrix, 0.0)

    print("🤖 正在运行层次聚类 AgglomerativeClustering...")

    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clusterer.fit_predict(dist_matrix)

    except TypeError:
        # 兼容旧 sklearn：旧版本参数名叫 affinity
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clusterer.fit_predict(dist_matrix)

    except Exception as e:
        print(f"❌ 聚类异常: {str(e)}")
        return

    out_data = []

    for i, s in enumerate(valid_strokes):
        out_data.append({
            "source_file": s["source_file"],
            "glyph_uid": s["glyph_uid"],
            "stroke_uid": s["stroke_uid"],

            "hex_key": s["hex_key"],
            "char": s.get("char", ""),
            "bezier_id": s["bezier_id"],

            "mother_bezier": s["mother_bezier"],
            "width_mean": s.get("width_mean", None),

            "cluster_id": int(labels[i]),
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    n_clusters = len(set(labels))

    print("\n🎉 聚类完成！")
    print(f"📦 有效 stroke 数: {N}")
    print(f"🧩 Shape Token 桶数: {n_clusters}")
    print(f"📏 distance_threshold: {distance_threshold}")
    print(f"💾 已保存至: {OUT_FILE}")

    # 简单检查是否还有 UID 冲突
    stroke_uids = [x["stroke_uid"] for x in out_data]
    if len(stroke_uids) != len(set(stroke_uids)):
        print("⚠️ 警告：stroke_uid 存在重复，请检查 bezier_id 是否在同一个 glyph 内重复")
    else:
        print("✅ stroke_uid 无重复")


if __name__ == "__main__":
    main()