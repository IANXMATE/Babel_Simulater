import os
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np


# ==========================================
# ⚙️ 全局配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CORPUS_FILE = os.path.join(SCRIPT_DIR, "alien_glyph_pcg_corpus.json")
TOPOLOGY_FILE = os.path.join(SCRIPT_DIR, "graph_grammar_samples.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_with_primitives.json")

RANDOM_SEED = 42

# 第一版不限制数量；如果调试慢，可以设成 20 / 50
MAX_CANDIDATES = None

# 采样配置
SAMPLER_CONFIG = {
    "shape_role_weight": 3.0,
    "shape_degree_weight": 2.0,
    "shape_global_weight": 0.8,

    "shape_temperature": 1.6,
    "width_temperature": 1.10,

    "variant_policy": "uniform_4",

    "length_noise_scale": 0.25,

    "min_length_prior_norm": 0.08,
    "max_length_prior_norm": 0.70,

    "allow_global_fallback": True,

    "diversity_penalty_same_shape": 0.45,

    "print_candidate_preview": True,
    "preview_count": 8,
}


# ==========================================
# 🧮 基础工具
# ==========================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def as_counter(d):
    c = Counter()
    if not d:
        return c
    for k, v in d.items():
        try:
            c[str(k)] += int(v)
        except Exception:
            pass
    return c


def weighted_choice_from_counter(counter, rng, temperature=1.0, used_shape_counter=None, diversity_penalty=1.0):
    """
    从 Counter 中按权重采样。

    used_shape_counter:
        当前 glyph 内已经用过的 shape_code，用于轻微降低重复 shape 的概率。
    """
    if not counter:
        return None

    keys = list(counter.keys())
    weights = []

    temp = max(float(temperature), 1e-6)

    for k in keys:
        w = float(counter[k])

        # 温度平滑
        w = w ** (1.0 / temp)

        # 同 glyph 内 shape 重复惩罚
        if used_shape_counter is not None and used_shape_counter.get(str(k), 0) > 0:
            w *= float(diversity_penalty) ** used_shape_counter[str(k)]

        weights.append(w)

    weights = np.asarray(weights, dtype=float)

    if np.isnan(weights).any() or np.isinf(weights).any() or weights.sum() <= 0:
        weights = np.ones(len(keys), dtype=float)

    probs = weights / weights.sum()
    idx = rng.choice(len(keys), p=probs)

    return keys[int(idx)]


def merge_counters_weighted(counter_items):
    """
    counter_items:
        [(counter, weight), ...]
    """
    out = Counter()

    for c, w in counter_items:
        if not c:
            continue

        for k, v in c.items():
            out[str(k)] += float(v) * float(w)

    return out


def get_primitive_y(primitive_item):
    """
    支持两种字段：
    1. prototype_y_function_norm: [y0, y1, ...]
    2. prototype_polyline_norm: [[x,y], ...]
    """
    if "prototype_y_function_norm" in primitive_item:
        y = np.asarray(primitive_item["prototype_y_function_norm"], dtype=float)
        return y

    if "prototype_polyline_norm" in primitive_item:
        arr = np.asarray(primitive_item["prototype_polyline_norm"], dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1]

    return None


def make_polyline_from_y(y, variant_id=0):
    """
    根据 shape primitive 的 y=f(x) 生成局部 polyline。

    variant 定义沿用你的聚类逻辑：
    0: Y
    1: -Y[::-1]
    2: -Y
    3: Y[::-1]
    """
    y = np.asarray(y, dtype=float)

    if variant_id == 0:
        yy = y.copy()
    elif variant_id == 1:
        yy = -y[::-1]
    elif variant_id == 2:
        yy = -y
    elif variant_id == 3:
        yy = y[::-1]
    else:
        yy = y.copy()

    x = np.linspace(0.0, 1.0, len(yy))

    # 以中心为原点，方便 solver 后续 scale/rotate/translate
    # x 从 [0,1] 改到 [-0.5,0.5]
    x = x - 0.5

    # y 也减均值，避免 primitive 自带偏移
    yy = yy - np.mean(yy)

    return np.stack([x, yy], axis=1)


def polyline_to_list(polyline, ndigits=6):
    return np.round(np.asarray(polyline, dtype=float), ndigits).tolist()


def sample_variant(rng, policy="uniform_4"):
    if policy == "uniform_4":
        return int(rng.integers(0, 4))
    return 0


def sample_length_prior(primitive_item, rng, cfg):
    mu = float(primitive_item.get("length_norm_mean", 0.25))
    sd = float(primitive_item.get("length_norm_std", 0.05))

    if sd <= 1e-6:
        sd = max(0.03, 0.15 * mu)

    noise_scale = float(cfg.get("length_noise_scale", 0.35))

    val = rng.normal(mu, sd * noise_scale)

    val = clamp(
        val,
        float(cfg.get("min_length_prior_norm", 0.08)),
        float(cfg.get("max_length_prior_norm", 0.85)),
    )

    return round(float(val), 6)


# ==========================================
# 📖 读取数据
# ==========================================
def load_json(path, required_key=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if required_key is not None and required_key not in obj:
        raise ValueError(f"{path} 中缺少字段: {required_key}")

    return obj


# ==========================================
# 📊 统计读取
# ==========================================
class PrimitiveStats:
    def __init__(self, corpus):
        self.corpus = corpus

        self.primitive_library = corpus.get("stroke_primitive_library", {})
        self.grammar_stats = corpus.get("grammar_stats", {})

        self.shape_code_hist = as_counter(
            self.grammar_stats.get("shape_code_hist", {})
        )

        self.width_token_hist = as_counter(
            self.grammar_stats.get("width_token_hist", {})
        )

        self.shape_by_degree = {
            str(k): as_counter(v)
            for k, v in self.grammar_stats.get("shape_by_degree", {}).items()
        }

        self.shape_by_role = {
            str(k): as_counter(v)
            for k, v in self.grammar_stats.get("shape_by_role", {}).items()
        }

        self.width_by_shape = {
            str(k): as_counter(v)
            for k, v in self.grammar_stats.get("width_by_shape", {}).items()
        }

    def has_shape(self, shape_code):
        return str(shape_code) in self.primitive_library

    def get_primitive(self, shape_code):
        return self.primitive_library.get(str(shape_code), None)

    def get_shape_counter_for_node(self, node, cfg):
        role_weight = float(cfg.get("shape_role_weight", 3.0))
        degree_weight = float(cfg.get("shape_degree_weight", 2.0))
        global_weight = float(cfg.get("shape_global_weight", 0.5))

        grammar_role = node.get("grammar_role", {})
        degree = str(grammar_role.get("degree", 0))
        roles = grammar_role.get("roles", [])

        counter_items = []

        # role-based
        for r in roles:
            r = str(r)
            if r in self.shape_by_role:
                counter_items.append((self.shape_by_role[r], role_weight))

        # degree-based
        if degree in self.shape_by_degree:
            counter_items.append((self.shape_by_degree[degree], degree_weight))

        # global fallback
        if cfg.get("allow_global_fallback", True):
            counter_items.append((self.shape_code_hist, global_weight))

        merged = merge_counters_weighted(counter_items)

        # 过滤掉 primitive_library 中不存在的 shape
        filtered = Counter()
        for k, v in merged.items():
            if self.has_shape(k):
                filtered[str(k)] += float(v)

        return filtered

    def get_width_counter_for_shape(self, shape_code, cfg):
        shape_key = str(shape_code)

        if shape_key in self.width_by_shape and self.width_by_shape[shape_key]:
            return self.width_by_shape[shape_key]

        if cfg.get("allow_global_fallback", True):
            return self.width_token_hist

        return Counter()


# ==========================================
# 🧩 分配 Primitive
# ==========================================
def assign_primitive_to_node(node, pstats, rng, cfg, used_shape_counter):
    shape_counter = pstats.get_shape_counter_for_node(node, cfg)

    shape_code = weighted_choice_from_counter(
        shape_counter,
        rng=rng,
        temperature=cfg.get("shape_temperature", 1.25),
        used_shape_counter=used_shape_counter,
        diversity_penalty=cfg.get("diversity_penalty_same_shape", 0.75),
    )

    source = "role_degree_mixed"

    if shape_code is None:
        # 最后兜底：直接从 primitive_library 里均匀抽
        all_shapes = list(pstats.primitive_library.keys())
        if not all_shapes:
            raise RuntimeError("primitive_library 为空，无法采样 shape_code")

        shape_code = str(all_shapes[int(rng.integers(0, len(all_shapes)))])
        source = "uniform_primitive_fallback"

    primitive = pstats.get_primitive(shape_code)

    if primitive is None:
        raise RuntimeError(f"shape_code={shape_code} 不在 primitive_library 中")

    width_counter = pstats.get_width_counter_for_shape(shape_code, cfg)

    width_token = weighted_choice_from_counter(
        width_counter,
        rng=rng,
        temperature=cfg.get("width_temperature", 1.10),
    )

    if width_token is None:
        width_token = "0"

    variant_id = sample_variant(
        rng,
        policy=cfg.get("variant_policy", "uniform_4"),
    )

    y = get_primitive_y(primitive)

    if y is None:
        primitive_polyline = None
        prototype_available = False
    else:
        primitive_polyline = make_polyline_from_y(y, variant_id=variant_id)
        prototype_available = True

    length_prior_norm = sample_length_prior(primitive, rng, cfg)

    new_node = dict(node)

    new_node["shape_code"] = int(shape_code)
    new_node["variant_id"] = int(variant_id)
    new_node["width_token"] = int(width_token)

    new_node["primitive_assignment"] = {
        "assignment_source": source,
        "shape_counter_size": int(len(shape_counter)),
        "width_counter_size": int(len(width_counter)),
    }

    new_node["primitive_ref"] = {
        "shape_code": int(shape_code),
        "prototype_available": bool(prototype_available),
        "variant_id": int(variant_id),
    }

    if prototype_available:
        new_node["primitive_ref"]["primitive_polyline_local_norm"] = polyline_to_list(
            primitive_polyline
        )

    new_node["solver_priors"] = {
        # solver 后面优化 center / rotation / scale
        # 这里只给先验
        "center_norm": None,
        "rotation_rad": None,
        "length_prior_norm": length_prior_norm,

        # 之后 solver 可以用这个作为 scale 初始值
        "scale_init_norm": length_prior_norm,

        # 后续可以扩展：role-based rotation prior
        "rotation_init_policy": "solver_random_or_layout_based",
    }

    used_shape_counter[str(shape_code)] += 1

    return new_node


def assign_primitives_to_candidate(topo_sample, pstats, rng, cfg, candidate_idx):
    candidate = dict(topo_sample)

    generated_glyph_id = f"glyph_candidate_{candidate_idx:05d}"

    nodes = topo_sample.get("nodes", [])
    assigned_nodes = []

    used_shape_counter = Counter()

    for node in nodes:
        assigned = assign_primitive_to_node(
            node=node,
            pstats=pstats,
            rng=rng,
            cfg=cfg,
            used_shape_counter=used_shape_counter,
        )
        assigned_nodes.append(assigned)

    candidate["generated_glyph_id"] = generated_glyph_id
    candidate["generation_stage"] = "topology_plus_primitives"

    candidate["nodes"] = assigned_nodes

    candidate["primitive_assignment_summary"] = {
        "num_nodes": len(assigned_nodes),
        "unique_shape_count": len(set(n["shape_code"] for n in assigned_nodes)),
        "shape_code_hist": {
            str(k): int(v)
            for k, v in Counter(n["shape_code"] for n in assigned_nodes).items()
        },
        "width_token_hist": {
            str(k): int(v)
            for k, v in Counter(n["width_token"] for n in assigned_nodes).items()
        },
        "variant_id_hist": {
            str(k): int(v)
            for k, v in Counter(n["variant_id"] for n in assigned_nodes).items()
        },
    }

    return candidate


# ==========================================
# 🔍 诊断
# ==========================================
def diagnose_inputs(corpus, topology_data, pstats):
    sampled_topologies = topology_data.get("sampled_topologies", [])

    primitive_count = len(pstats.primitive_library)
    shape_hist_count = len(pstats.shape_code_hist)
    role_keys = sorted(pstats.shape_by_role.keys())
    degree_keys = sorted(pstats.shape_by_degree.keys())

    print("\n" + "=" * 80)
    print("📊 Stroke Primitive Sampler Input Diagnostics")
    print("=" * 80)

    print("\n[输入文件]")
    print(f"  corpus:   {CORPUS_FILE}")
    print(f"  topology: {TOPOLOGY_FILE}")

    print("\n[Corpus / Primitive]")
    print(f"  stroke_primitive_library size: {primitive_count}")
    print(f"  shape_code_hist size:          {shape_hist_count}")
    print(f"  shape_by_role keys:            {role_keys}")
    print(f"  shape_by_degree keys:          {degree_keys}")
    print(f"  width_by_shape size:           {len(pstats.width_by_shape)}")

    print("\n[Graph Grammar Samples]")
    print(f"  sampled_topologies: {len(sampled_topologies)}")

    if sampled_topologies:
        n_hist = Counter()
        role_hist = Counter()
        degree_hist = Counter()

        for s in sampled_topologies:
            for n in s.get("nodes", []):
                gr = n.get("grammar_role", {})
                degree_hist[str(gr.get("degree", 0))] += 1
                for r in gr.get("roles", []):
                    role_hist[str(r)] += 1
            n_hist[s["topology"]["num_nodes"]] += 1

        print("\n[Topology Node Role 分布]")
        print(f"  stroke_count_hist: {dict(n_hist)}")
        print(f"  node_degree_hist:  {dict(degree_hist)}")
        print(f"  node_role_hist:    {dict(role_hist)}")

    print("\n[初步可行性判断]")
    if primitive_count < 20:
        print("  ⚠️ primitive_count < 20，生成视觉多样性可能偏弱。")
    else:
        print("  ✅ primitive library 初步够用。")

    if not role_keys:
        print("  ⚠️ 没有 shape_by_role，采样会主要依赖 degree/global。")
    else:
        print("  ✅ shape_by_role 可用，可以按 guest/host/stroke 角色采样。")

    if not degree_keys:
        print("  ⚠️ 没有 shape_by_degree，采样会主要依赖 role/global。")
    else:
        print("  ✅ shape_by_degree 可用，可以按节点度数采样。")

    if len(sampled_topologies) == 0:
        print("  ❌ 没有 sampled_topologies，无法继续。")
    else:
        print("  ✅ topology samples 存在，可以进行 primitive assignment。")

    print("=" * 80 + "\n")


def summarize_output(candidates):
    shape_hist = Counter()
    width_hist = Counter()
    variant_hist = Counter()
    unique_shape_per_glyph = []
    assignment_source_hist = Counter()

    for c in candidates:
        shapes = []

        for n in c.get("nodes", []):
            shape_hist[str(n.get("shape_code"))] += 1
            width_hist[str(n.get("width_token"))] += 1
            variant_hist[str(n.get("variant_id"))] += 1
            shapes.append(n.get("shape_code"))

            src = n.get("primitive_assignment", {}).get("assignment_source", "unknown")
            assignment_source_hist[src] += 1

        unique_shape_per_glyph.append(len(set(shapes)))

    avg_unique_shape = (
        float(np.mean(unique_shape_per_glyph))
        if unique_shape_per_glyph
        else 0.0
    )

    summary = {
        "candidate_count": len(candidates),
        "shape_code_hist": dict(shape_hist),
        "width_token_hist": dict(width_hist),
        "variant_id_hist": dict(variant_hist),
        "assignment_source_hist": dict(assignment_source_hist),
        "avg_unique_shape_per_glyph": round(avg_unique_shape, 4),
        "unique_shape_codes_used": len(shape_hist),
    }

    print("\n" + "=" * 80)
    print("📦 Primitive Assignment Output Summary")
    print("=" * 80)
    print(f"  candidate_count: {summary['candidate_count']}")
    print(f"  unique_shape_codes_used: {summary['unique_shape_codes_used']}")
    print(f"  avg_unique_shape_per_glyph: {summary['avg_unique_shape_per_glyph']:.3f}")

    print("\nTop shape_code:")
    for k, v in shape_hist.most_common(12):
        print(f"  {k}: {v}")

    print("\nWidth token:")
    for k, v in width_hist.most_common():
        print(f"  {k}: {v}")

    print("\nVariant id:")
    for k, v in variant_hist.most_common():
        print(f"  {k}: {v}")

    print("\nAssignment source:")
    for k, v in assignment_source_hist.most_common():
        print(f"  {k}: {v}")

    print("=" * 80 + "\n")

    return summary


def print_candidate_preview(candidates, max_count=5):
    print("\n" + "=" * 80)
    print("🔍 Candidate Preview")
    print("=" * 80)

    for c in candidates[:max_count]:
        print(f"\n{c['generated_glyph_id']} | {c.get('grammar_sample_id', '')}")
        print(f"  topology: N={c['topology']['num_nodes']} E={c['topology']['num_undirected_edges']}")

        for n in c["nodes"]:
            gr = n.get("grammar_role", {})
            print(
                f"  node {n['node_id']}: "
                f"deg={gr.get('degree')} roles={gr.get('roles')} "
                f"shape={n['shape_code']} var={n['variant_id']} "
                f"width={n['width_token']} "
                f"len_prior={n['solver_priors']['length_prior_norm']}"
            )

    print("=" * 80 + "\n")


# ==========================================
# 🚀 主程序
# ==========================================
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    corpus = load_json(CORPUS_FILE, required_key="stroke_primitive_library")
    topology_data = load_json(TOPOLOGY_FILE, required_key="sampled_topologies")

    pstats = PrimitiveStats(corpus)

    diagnose_inputs(corpus, topology_data, pstats)

    sampled_topologies = topology_data["sampled_topologies"]

    if MAX_CANDIDATES is not None:
        sampled_topologies = sampled_topologies[:MAX_CANDIDATES]

    candidates = []

    print("🎲 正在为 sampled topologies 分配 stroke primitives...")

    for i, topo_sample in enumerate(sampled_topologies):
        candidate = assign_primitives_to_candidate(
            topo_sample=topo_sample,
            pstats=pstats,
            rng=rng,
            cfg=SAMPLER_CONFIG,
            candidate_idx=i,
        )
        candidates.append(candidate)

    output_summary = summarize_output(candidates)

    if SAMPLER_CONFIG.get("print_candidate_preview", True):
        print_candidate_preview(
            candidates,
            max_count=int(SAMPLER_CONFIG.get("preview_count", 8)),
        )

    output = {
        "schema_version": "stroke_primitive_sampler_v1",

        "description": (
            "Assigns source-aware stroke primitive codes to graph grammar "
            "topology samples. The output is intended for constraint_solver.py, "
            "which will solve node center, rotation, and scale."
        ),

        "source_corpus_file": CORPUS_FILE,
        "source_topology_file": TOPOLOGY_FILE,
        "output_file": OUTPUT_FILE,

        "random_seed": RANDOM_SEED,
        "sampler_config": SAMPLER_CONFIG,

        "input_summary": {
            "primitive_library_size": len(pstats.primitive_library),
            "sampled_topology_count": len(topology_data["sampled_topologies"]),
            "used_topology_count": len(sampled_topologies),
        },

        "output_summary": output_summary,

        "glyph_candidates": candidates,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"💾 已保存至: {OUTPUT_FILE}")

    print("\n📌 下一步：")
    print("  1. 检查 glyph_candidates_with_primitives.json")
    print("  2. 看每个 node 是否都有 shape_code / width_token / primitive_polyline_local_norm")
    print("  3. 下一步写 constraint_solver.py，用 topology edges 约束 center/rotation/scale")
    print("  4. solver 输出后再接 aesthetic_scorer.py")


if __name__ == "__main__":
    main()