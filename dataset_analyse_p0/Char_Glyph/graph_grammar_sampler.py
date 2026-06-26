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
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "graph_grammar_samples.json")

RANDOM_SEED = 42
NUM_SAMPLES = 100

# 采样风格配置：后续 LLM 就是生成这个 JSON
STYLE_CONFIG = {
    # 推荐第一版先控制 stroke_count，不要太宽
    "stroke_count_range": [3, 8],

    # 是否偏好连通图
    "prefer_connected": True,

    # 复杂度目标：0~1
    # 越高越偏向多边、多交叉、多分支
    "complexity": 0.55,

    # motif 偏好
    "motif_weights": {
        "has_E2E": 0.6,
        "has_T": 1.2,
        "has_X": 0.8,
        "cycle": 0.3,
        "branch_or_hub": 1.0,
        "connected": 1.0,
    },

    # edge 类型偏好
    "edge_type_weights": {
        "E2E": 1.0,
        "T": 1.2,
        "X": 0.8,
    },

    # 是否随机重编号节点
    # 推荐 True：避免后续系统依赖原 node_id 顺序
    "relabel_nodes": True,

    # 对 t / angle 做轻微扰动，提高生成变化
    "jitter_t_std": 0.025,
    "jitter_angle_std_deg": 5.0,

    # 第一版建议保持 0，不要破坏真实拓扑模板
    "mutate_drop_edge_prob": 0.00,
    "mutate_add_edge_prob": 0.00,

    # 生成时每个原 glyph 的派生样本会被均衡降权
    # 防止某个 glyph 因为派生多而支配 grammar
    "balance_by_glyph_uid": True,
}

J_TYPE_TO_IDX = {
    "NONE": 0,
    "E2E": 1,
    "X": 2,
    "T": 3,
}

IDX_TO_J_TYPE = {
    0: "NONE",
    1: "E2E",
    2: "X",
    3: "T",
}

T_BINS = 32
ANGLE_BINS = 24


# ==========================================
# 🧮 基础工具
# ==========================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def quantize_t(t_val, bins=T_BINS):
    t_val = clamp(float(t_val), 0.0, 1.0)
    return int(round(t_val * bins))


def quantize_angle_deg(angle_deg, bins=ANGLE_BINS):
    angle = float(angle_deg) % 180.0
    return int(math.floor(angle / 180.0 * bins))


def angle_to_sincos(angle_deg):
    rad = math.radians(float(angle_deg))
    return abs(math.sin(rad)), math.cos(rad)


def safe_counter_get(d, key, default=0):
    try:
        return int(d.get(key, default))
    except Exception:
        return default


def weighted_choice(items, weights, rng):
    weights = np.asarray(weights, dtype=float)

    if len(items) == 0:
        raise ValueError("weighted_choice got empty items")

    if np.isnan(weights).any() or np.isinf(weights).any() or weights.sum() <= 0:
        weights = np.ones(len(items), dtype=float)

    probs = weights / weights.sum()
    idx = rng.choice(len(items), p=probs)
    return items[int(idx)]


def connected_components(num_nodes, undirected_edges):
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in undirected_edges:
        union(int(e["u"]), int(e["v"]))

    comps = defaultdict(list)
    for i in range(num_nodes):
        comps[find(i)].append(i)

    return list(comps.values())


def calc_cycle_rank(num_nodes, undirected_edges):
    comps = connected_components(num_nodes, undirected_edges)
    return max(0, len(undirected_edges) - num_nodes + len(comps))


def summarize_topology(num_nodes, undirected_edges):
    edge_type_counts = Counter(e["j_type"] for e in undirected_edges)

    degree = Counter()
    for e in undirected_edges:
        degree[int(e["u"])] += 1
        degree[int(e["v"])] += 1

    degrees = [int(degree[i]) for i in range(num_nodes)]

    comps = connected_components(num_nodes, undirected_edges)
    cycle_rank = calc_cycle_rank(num_nodes, undirected_edges)

    return {
        "num_nodes": int(num_nodes),
        "num_undirected_edges": int(len(undirected_edges)),
        "num_directed_edges": int(len(undirected_edges) * 2),
        "edge_type_counts": dict(edge_type_counts),
        "degrees": degrees,
        "max_degree": int(max(degrees) if degrees else 0),
        "num_components": int(len(comps)),
        "component_sizes": [int(len(c)) for c in comps],
        "cycle_rank": int(cycle_rank),
        "has_cycle": bool(cycle_rank > 0),
        "connected": bool(len(comps) == 1 if num_nodes > 0 else False),
    }


def topology_signature(num_nodes, undirected_edges):
    """
    简单签名，不做严格图同构 canonical。
    用于估计模板多样性。
    """
    edge_sigs = []

    for e in undirected_edges:
        u = int(e["u"])
        v = int(e["v"])
        a, b = sorted([u, v])
        jt = e.get("j_type", "NONE")
        edge_sigs.append(f"{a}-{b}:{jt}")

    edge_sigs = sorted(edge_sigs)
    return f"N{num_nodes}|" + "|".join(edge_sigs)


def motif_flags(summary):
    edge_counts = summary.get("edge_type_counts", {})

    return {
        "has_E2E": edge_counts.get("E2E", 0) > 0,
        "has_T": edge_counts.get("T", 0) > 0,
        "has_X": edge_counts.get("X", 0) > 0,
        "cycle": bool(summary.get("has_cycle", False)),
        "branch_or_hub": int(summary.get("max_degree", 0)) >= 3,
        "connected": bool(summary.get("connected", False)),
    }


# ==========================================
# 📖 读取 Corpus
# ==========================================
def load_corpus(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 corpus 文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    if "glyph_samples" not in corpus:
        raise ValueError("corpus 中没有 glyph_samples，请先运行 build_pcg_glyph_corpus.py")

    return corpus


# ==========================================
# 🔍 数据质量诊断
# ==========================================
def build_templates(corpus):
    samples = corpus["glyph_samples"]
    templates = []

    for s in samples:
        topo = s.get("topology", {})
        edges = topo.get("positive_edges_undirected", [])
        nodes = s.get("nodes", [])

        num_nodes = int(topo.get("num_nodes", len(nodes)))

        if num_nodes <= 0:
            continue

        # 第一版 graph grammar 需要至少一个连接边
        if len(edges) == 0:
            continue

        summary = summarize_topology(num_nodes, edges)
        sig = topology_signature(num_nodes, edges)

        glyph_uid = s.get("glyph_uid", s.get("char_key", "UNKNOWN_GLYPH"))
        source_file = s.get("source_file", "UNKNOWN_SOURCE")

        templates.append({
            "template_id": len(templates),
            "source_sample_id": s.get("sample_id", f"sample_{len(templates)}"),
            "source_file": source_file,
            "glyph_uid": glyph_uid,
            "hex_key": s.get("hex_key", ""),
            "char": s.get("char", ""),
            "derivation": s.get("derivation", {}),

            "num_nodes": num_nodes,
            "edges": edges,
            "summary": summary,
            "signature": sig,
            "motifs": motif_flags(summary),
        })

    return templates


def print_top_counter(title, counter, top_k=12):
    print(f"\n{title}")

    if not counter:
        print("  <empty>")
        return

    total = sum(counter.values())

    for k, v in counter.most_common(top_k):
        pct = 100.0 * v / max(total, 1)
        print(f"  {k}: {v} ({pct:.1f}%)")


def diagnose_corpus(corpus, templates):
    samples = corpus["glyph_samples"]
    primitive_lib = corpus.get("stroke_primitive_library", {})
    build_report = corpus.get("build_report", {})
    grammar_stats = corpus.get("grammar_stats", {})

    source_files = Counter()
    glyph_uids = Counter()
    stroke_count_hist = Counter()
    edge_count_hist = Counter()
    edge_type_hist = Counter()
    degree_hist = Counter()
    motif_hist = Counter()
    signature_hist = Counter()

    connected_count = 0
    cycle_count = 0
    branch_count = 0

    for t in templates:
        source_files[t["source_file"]] += 1
        glyph_uids[t["glyph_uid"]] += 1

        summary = t["summary"]
        stroke_count_hist[summary["num_nodes"]] += 1
        edge_count_hist[summary["num_undirected_edges"]] += 1

        for k, v in summary["edge_type_counts"].items():
            edge_type_hist[k] += int(v)

        for d in summary["degrees"]:
            degree_hist[d] += 1

        if summary["connected"]:
            connected_count += 1
        if summary["has_cycle"]:
            cycle_count += 1
        if summary["max_degree"] >= 3:
            branch_count += 1

        for m, yes in t["motifs"].items():
            if yes:
                motif_hist[m] += 1

        signature_hist[t["signature"]] += 1

    num_samples = len(samples)
    num_templates = len(templates)
    unique_sources = len(source_files)
    unique_glyphs = len(glyph_uids)
    unique_signatures = len(signature_hist)
    primitive_count = len(primitive_lib)

    avg_deriv_per_glyph = num_templates / max(unique_glyphs, 1)

    connected_ratio = connected_count / max(num_templates, 1)
    cycle_ratio = cycle_count / max(num_templates, 1)
    branch_ratio = branch_count / max(num_templates, 1)

    print("\n" + "=" * 80)
    print("📊 Graph Grammar Corpus Diagnostics")
    print("=" * 80)

    print("\n[Corpus 基本信息]")
    print(f"  schema_version: {corpus.get('schema_version', '<unknown>')}")
    print(f"  glyph_samples 总数: {num_samples}")
    print(f"  可用 topology templates: {num_templates}")
    print(f"  unique source_file: {unique_sources}")
    print(f"  unique glyph_uid: {unique_glyphs}")
    print(f"  unique topology signatures: {unique_signatures}")
    print(f"  stroke primitive count: {primitive_count}")

    print("\n[Build Report]")
    for k, v in build_report.items():
        print(f"  {k}: {v}")

    print("\n[派生膨胀情况]")
    print(f"  avg topology templates per glyph_uid: {avg_deriv_per_glyph:.2f}")
    print("  说明：如果这个值很高，采样时需要 balance_by_glyph_uid=True，避免派生样本支配 grammar。")

    print("\n[拓扑可行性指标]")
    print(f"  connected ratio: {connected_ratio:.3f}")
    print(f"  cycle ratio:     {cycle_ratio:.3f}")
    print(f"  branch ratio:    {branch_ratio:.3f}")
    print(f"  unique signature ratio: {unique_signatures / max(num_templates, 1):.3f}")

    print_top_counter("Top stroke_count 分布", stroke_count_hist)
    print_top_counter("Top edge_count 分布", edge_count_hist)
    print_top_counter("Edge type 总量", edge_type_hist)
    print_top_counter("Degree 分布", degree_hist)
    print_top_counter("Motif 分布", motif_hist)
    print_top_counter("Source file 样本分布", source_files, top_k=8)

    print("\n[初步可行性判断]")
    notes = []

    if num_templates < 50:
        notes.append("⚠️ topology template 少于 50，graph grammar 多样性可能不足。")
    else:
        notes.append("✅ topology template 数量初步够用。")

    if unique_glyphs < 30:
        notes.append("⚠️ unique glyph_uid 少于 30，可能更像 demo，不太够支撑强统计。")
    else:
        notes.append("✅ unique glyph_uid 数量初步够用。")

    if primitive_count < 20:
        notes.append("⚠️ stroke primitive 少于 20，笔画形状库偏少。")
    else:
        notes.append("✅ stroke primitive library 有一定规模。")

    if connected_ratio < 0.6:
        notes.append("⚠️ 连通 topology 比例偏低，生成结果可能容易散。建议 solver/scorer 中强罚 disconnected。")
    else:
        notes.append("✅ 大部分 topology 连通，适合作为字符骨架。")

    if edge_type_hist.get("T", 0) == 0:
        notes.append("⚠️ 没有 T-junction，外星字符结构复杂度可能不足。")
    else:
        notes.append("✅ 存在 T-junction，可支持分支型字符。")

    if edge_type_hist.get("X", 0) == 0:
        notes.append("⚠️ 没有 X-cross，交叉型 glyph 较弱。")
    else:
        notes.append("✅ 存在 X-cross，可支持交叉型字符。")

    if unique_signatures < 15:
        notes.append("⚠️ unique topology signature 偏少，grammar 可能只是模板重采样。")
    else:
        notes.append("✅ topology signature 有一定多样性。")

    for line in notes:
        print(" ", line)

    print("=" * 80 + "\n")

    return {
        "num_glyph_samples": num_samples,
        "num_templates": num_templates,
        "unique_source_files": unique_sources,
        "unique_glyph_uids": unique_glyphs,
        "unique_topology_signatures": unique_signatures,
        "stroke_primitive_count": primitive_count,
        "avg_templates_per_glyph_uid": round(avg_deriv_per_glyph, 4),
        "connected_ratio": round(connected_ratio, 4),
        "cycle_ratio": round(cycle_ratio, 4),
        "branch_ratio": round(branch_ratio, 4),
        "stroke_count_hist": {str(k): int(v) for k, v in stroke_count_hist.items()},
        "edge_count_hist": {str(k): int(v) for k, v in edge_count_hist.items()},
        "edge_type_hist": {str(k): int(v) for k, v in edge_type_hist.items()},
        "degree_hist": {str(k): int(v) for k, v in degree_hist.items()},
        "motif_hist": {str(k): int(v) for k, v in motif_hist.items()},
        "feasibility_notes": notes,
    }


# ==========================================
# 🧬 Grammar Sampling
# ==========================================
def template_score(template, style_config, glyph_template_count):
    summary = template["summary"]
    motifs = template["motifs"]

    n = summary["num_nodes"]
    m = summary["num_undirected_edges"]

    lo, hi = style_config.get("stroke_count_range", [1, 999])

    # 超出 stroke_count 范围，直接极低权重
    if n < lo or n > hi:
        return 1e-8

    score = 1.0

    # 每个原 glyph 平衡权重，避免派生过多的 glyph 过拟合 grammar
    if style_config.get("balance_by_glyph_uid", True):
        score *= 1.0 / max(glyph_template_count.get(template["glyph_uid"], 1), 1)

    # 连通偏好
    if style_config.get("prefer_connected", True):
        if summary.get("connected", False):
            score *= 2.0
        else:
            score *= 0.25

    # motif 偏好
    motif_weights = style_config.get("motif_weights", {})
    for motif_name, w in motif_weights.items():
        if motifs.get(motif_name, False):
            score *= (1.0 + float(w))

    # edge 类型偏好
    edge_type_weights = style_config.get("edge_type_weights", {})
    edge_counts = summary.get("edge_type_counts", {})

    for jt, cnt in edge_counts.items():
        if jt in edge_type_weights:
            score *= (float(edge_type_weights[jt]) ** max(int(cnt), 0))

    # 复杂度偏好
    complexity = float(style_config.get("complexity", 0.5))
    max_possible_edges = max(1.0, n * (n - 1) / 2)
    density = m / max_possible_edges

    # 把 complexity 映射到目标 density，避免复杂度过强
    target_density = 0.15 + 0.45 * clamp(complexity, 0.0, 1.0)
    density_penalty = abs(density - target_density)

    score *= math.exp(-3.0 * density_penalty)

    return max(score, 1e-8)


def jitter_edge_params(edge, style_config, rng):
    e = dict(edge)

    jt = e.get("j_type", "E2E")

    jitter_t_std = float(style_config.get("jitter_t_std", 0.0))
    jitter_angle_std = float(style_config.get("jitter_angle_std_deg", 0.0))

    t_u = float(e.get("t_u", 0.0))
    t_v = float(e.get("t_v", 0.0))

    # E2E 的端点 t 不建议 jitter，否则容易破坏端点吸附语义
    if jt == "E2E":
        t_u_new = t_u
        t_v_new = t_v

    else:
        t_u_new = clamp(t_u + rng.normal(0.0, jitter_t_std), 0.0, 1.0)
        t_v_new = clamp(t_v + rng.normal(0.0, jitter_t_std), 0.0, 1.0)

    angle_deg = float(e.get("angle_deg", 90.0))

    if jt == "E2E":
        angle_new = angle_deg
    else:
        angle_new = clamp(
            angle_deg + rng.normal(0.0, jitter_angle_std),
            0.0,
            180.0,
        )

    angle_sin, angle_cos = angle_to_sincos(angle_new)

    e["t_u"] = round(float(t_u_new), 6)
    e["t_v"] = round(float(t_v_new), 6)
    e["t_u_bin"] = quantize_t(t_u_new)
    e["t_v_bin"] = quantize_t(t_v_new)

    e["t_diff"] = round(abs(t_u_new - t_v_new), 6)
    e["t_prod"] = round(t_u_new * t_v_new, 6)

    e["angle_deg"] = round(float(angle_new), 6)
    e["angle_bin"] = quantize_angle_deg(angle_new)
    e["angle_sin"] = round(float(angle_sin), 6)
    e["angle_cos"] = round(float(angle_cos), 6)

    return e


def make_directed_edges_from_undirected(undirected_edges, grammar_sample_id):
    directed = []

    for e in undirected_edges:
        base = dict(e)

        directed.append({
            **base,
            "directed": True,
            "direction": "forward",
        })

        rev = dict(base)
        rev["directed"] = True
        rev["direction"] = "reverse"

        rev["u"] = int(base["v"])
        rev["v"] = int(base["u"])

        rev["role_u"] = base.get("role_v", "stroke")
        rev["role_v"] = base.get("role_u", "stroke")

        rev["t_u"] = base["t_v"]
        rev["t_v"] = base["t_u"]
        rev["t_u_bin"] = base["t_v_bin"]
        rev["t_v_bin"] = base["t_u_bin"]

        # bid / stroke_uid 这里是生成拓扑，不绑定具体原 stroke
        rev.pop("bid_u", None)
        rev.pop("bid_v", None)
        rev.pop("stroke_uid_u", None)
        rev.pop("stroke_uid_v", None)

        directed.append(rev)

    return directed


def relabel_template_edges(edges, num_nodes, rng):
    perm = list(range(num_nodes))
    rng.shuffle(perm)

    mapping = {
        old: new
        for old, new in zip(range(num_nodes), perm)
    }

    new_edges = []

    for e in edges:
        ne = dict(e)
        ne["u"] = int(mapping[int(e["u"])])
        ne["v"] = int(mapping[int(e["v"])])
        new_edges.append(ne)

    return new_edges, mapping


def sample_new_edge_params(u, v, existing_edges, style_config, rng):
    """
    可选 mutation: 添加新边。
    第一版默认不会用。
    """
    edge_type_weights = style_config.get("edge_type_weights", {"E2E": 1.0, "T": 1.0, "X": 1.0})
    types = list(edge_type_weights.keys())
    weights = [edge_type_weights[t] for t in types]

    jt = weighted_choice(types, weights, rng)
    jidx = J_TYPE_TO_IDX.get(jt, 1)

    if jt == "E2E":
        t_u = float(rng.choice([0.0, 1.0]))
        t_v = float(rng.choice([0.0, 1.0]))
        role_u = "stroke"
        role_v = "stroke"
        angle = 180.0

    elif jt == "T":
        # u 作为 guest，v 作为 host
        t_u = float(rng.choice([0.0, 1.0]))
        t_v = float(rng.uniform(0.15, 0.85))
        role_u = "guest"
        role_v = "host"
        angle = 90.0

    else:
        jt = "X"
        jidx = J_TYPE_TO_IDX["X"]
        t_u = float(rng.uniform(0.15, 0.85))
        t_v = float(rng.uniform(0.15, 0.85))
        role_u = "stroke"
        role_v = "stroke"
        angle = 90.0

    angle_sin, angle_cos = angle_to_sincos(angle)

    return {
        "j_type": jt,
        "j_type_idx": int(jidx),
        "u": int(u),
        "v": int(v),
        "role_u": role_u,
        "role_v": role_v,
        "t_u": round(t_u, 6),
        "t_v": round(t_v, 6),
        "t_u_bin": quantize_t(t_u),
        "t_v_bin": quantize_t(t_v),
        "t_diff": round(abs(t_u - t_v), 6),
        "t_prod": round(t_u * t_v, 6),
        "angle_deg": round(angle, 6),
        "angle_bin": quantize_angle_deg(angle),
        "angle_sin": round(float(angle_sin), 6),
        "angle_cos": round(float(angle_cos), 6),
    }


def maybe_mutate_edges(edges, num_nodes, style_config, rng):
    new_edges = [dict(e) for e in edges]

    # drop mutation
    drop_prob = float(style_config.get("mutate_drop_edge_prob", 0.0))
    if drop_prob > 0 and len(new_edges) > 1 and rng.random() < drop_prob:
        idx = int(rng.integers(0, len(new_edges)))
        new_edges.pop(idx)

    # add mutation
    add_prob = float(style_config.get("mutate_add_edge_prob", 0.0))
    if add_prob > 0 and rng.random() < add_prob:
        used_pairs = set()
        for e in new_edges:
            a, b = sorted([int(e["u"]), int(e["v"])])
            used_pairs.add((a, b))

        candidates = []
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if (u, v) not in used_pairs:
                    candidates.append((u, v))

        if candidates:
            u, v = candidates[int(rng.integers(0, len(candidates)))]
            new_edges.append(sample_new_edge_params(u, v, new_edges, style_config, rng))

    return new_edges


def build_generated_nodes(num_nodes, undirected_edges):
    degree = Counter()
    incident_types = defaultdict(list)
    roles = defaultdict(list)

    for e in undirected_edges:
        u = int(e["u"])
        v = int(e["v"])

        degree[u] += 1
        degree[v] += 1

        incident_types[u].append(e["j_type"])
        incident_types[v].append(e["j_type"])

        roles[u].append(e.get("role_u", "stroke"))
        roles[v].append(e.get("role_v", "stroke"))

    nodes = []

    for i in range(num_nodes):
        nodes.append({
            "node_id": int(i),

            # 此时还没有 shape_code，后续 stroke_primitive_sampler.py 再填
            "shape_code": None,
            "width_token": None,

            "grammar_role": {
                "degree": int(degree[i]),
                "incident_edge_types": sorted(list(set(incident_types[i]))),
                "roles": sorted(list(set(roles[i]))),
            },

            "solver_variable_slot": {
                "center_norm": None,
                "scale": None,
                "rotation_rad": None,
            },
        })

    return nodes


def sample_topology(templates, style_config, rng, sample_idx):
    glyph_template_count = Counter(t["glyph_uid"] for t in templates)

    weights = [
        template_score(t, style_config, glyph_template_count)
        for t in templates
    ]

    template = weighted_choice(templates, weights, rng)

    num_nodes = int(template["num_nodes"])
    edges = [dict(e) for e in template["edges"]]

    # 清理模板绑定的源 stroke 信息，生成拓扑阶段不绑定具体 stroke
    cleaned_edges = []
    for e in edges:
        ce = {
            "j_type": e.get("j_type", "E2E"),
            "j_type_idx": int(e.get("j_type_idx", J_TYPE_TO_IDX.get(e.get("j_type", "E2E"), 1))),
            "u": int(e["u"]),
            "v": int(e["v"]),
            "role_u": e.get("role_u", "stroke"),
            "role_v": e.get("role_v", "stroke"),
            "t_u": float(e.get("t_u", 0.0)),
            "t_v": float(e.get("t_v", 0.0)),
            "t_u_bin": int(e.get("t_u_bin", quantize_t(e.get("t_u", 0.0)))),
            "t_v_bin": int(e.get("t_v_bin", quantize_t(e.get("t_v", 0.0)))),
            "t_diff": float(e.get("t_diff", abs(float(e.get("t_u", 0.0)) - float(e.get("t_v", 0.0))))),
            "t_prod": float(e.get("t_prod", float(e.get("t_u", 0.0)) * float(e.get("t_v", 0.0)))),
            "angle_deg": float(e.get("angle_deg", 90.0)),
            "angle_bin": int(e.get("angle_bin", quantize_angle_deg(e.get("angle_deg", 90.0)))),
            "angle_sin": float(e.get("angle_sin", angle_to_sincos(e.get("angle_deg", 90.0))[0])),
            "angle_cos": float(e.get("angle_cos", angle_to_sincos(e.get("angle_deg", 90.0))[1])),
        }
        cleaned_edges.append(ce)

    if style_config.get("relabel_nodes", True):
        cleaned_edges, relabel_mapping = relabel_template_edges(cleaned_edges, num_nodes, rng)
    else:
        relabel_mapping = {i: i for i in range(num_nodes)}

    # jitter t / angle
    jittered_edges = [
        jitter_edge_params(e, style_config, rng)
        for e in cleaned_edges
    ]

    # optional mutate
    mutated_edges = maybe_mutate_edges(jittered_edges, num_nodes, style_config, rng)

    # 重新编号 edge_id / edge_uid
    grammar_sample_id = f"grammar_sample_{sample_idx:05d}"

    final_edges = []
    for edge_id, e in enumerate(mutated_edges):
        fe = dict(e)
        fe["edge_id"] = int(edge_id)
        fe["edge_uid"] = f"{grammar_sample_id}::edge_{edge_id}"
        final_edges.append(fe)

    summary = summarize_topology(num_nodes, final_edges)
    directed_edges = make_directed_edges_from_undirected(final_edges, grammar_sample_id)
    nodes = build_generated_nodes(num_nodes, final_edges)

    return {
        "grammar_sample_id": grammar_sample_id,

        "generation_type": "empirical_graph_grammar",

        "template_ref": {
            "template_id": int(template["template_id"]),
            "source_sample_id": template["source_sample_id"],

            # 注意：这些只是模板来源，不是新生成字符的 ID
            "source_file": template["source_file"],
            "glyph_uid": template["glyph_uid"],
            "hex_key": template["hex_key"],
            "char": template.get("char", ""),
            "derivation": template.get("derivation", {}),
        },

        "relabel_mapping": {
            str(k): int(v)
            for k, v in relabel_mapping.items()
        },

        "nodes": nodes,

        "topology": {
            **summary,
            "positive_edges_undirected": final_edges,
            "positive_edges_directed": directed_edges,
        },

        "style_config": style_config,
    }


# ==========================================
# 🚀 主程序
# ==========================================
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print(f"📖 正在读取 PCG corpus: {CORPUS_FILE}")
    corpus = load_corpus(CORPUS_FILE)

    print("🧩 正在抽取 topology templates...")
    templates = build_templates(corpus)

    if not templates:
        raise RuntimeError("没有可用 topology templates，请检查 alien_glyph_pcg_corpus.json")

    diagnostics = diagnose_corpus(corpus, templates)

    print("🎲 正在采样 graph grammar topologies...")

    sampled = []

    for i in range(NUM_SAMPLES):
        item = sample_topology(
            templates=templates,
            style_config=STYLE_CONFIG,
            rng=rng,
            sample_idx=i,
        )
        sampled.append(item)

    # 采样结果统计
    sampled_stroke_hist = Counter()
    sampled_edge_hist = Counter()
    sampled_edge_type_hist = Counter()
    sampled_motif_hist = Counter()
    sampled_connected = 0

    for s in sampled:
        topo = s["topology"]

        sampled_stroke_hist[topo["num_nodes"]] += 1
        sampled_edge_hist[topo["num_undirected_edges"]] += 1

        for k, v in topo["edge_type_counts"].items():
            sampled_edge_type_hist[k] += int(v)

        if topo["connected"]:
            sampled_connected += 1

        flags = motif_flags(topo)
        for k, yes in flags.items():
            if yes:
                sampled_motif_hist[k] += 1

    print("\n" + "=" * 80)
    print("📦 Sampled Topology Summary")
    print("=" * 80)
    print(f"  sampled count: {len(sampled)}")
    print(f"  connected ratio: {sampled_connected / max(len(sampled), 1):.3f}")

    print_top_counter("Sampled stroke_count 分布", sampled_stroke_hist)
    print_top_counter("Sampled edge_count 分布", sampled_edge_hist)
    print_top_counter("Sampled edge type 总量", sampled_edge_type_hist)
    print_top_counter("Sampled motif 分布", sampled_motif_hist)
    print("=" * 80 + "\n")

    output = {
        "schema_version": "graph_grammar_sampler_v1",

        "description": (
            "Empirical graph grammar sampler for alien glyph topology. "
            "Templates are sampled from source-aware expert annotations, "
            "using glyph_uid = source_file::hex_key to avoid cross-font collisions."
        ),

        "source_corpus_file": CORPUS_FILE,
        "output_file": OUTPUT_FILE,

        "random_seed": RANDOM_SEED,
        "num_samples": NUM_SAMPLES,

        "style_config": STYLE_CONFIG,

        "diagnostics": diagnostics,

        "sampled_summary": {
            "sampled_count": len(sampled),
            "connected_ratio": round(sampled_connected / max(len(sampled), 1), 4),
            "stroke_count_hist": {str(k): int(v) for k, v in sampled_stroke_hist.items()},
            "edge_count_hist": {str(k): int(v) for k, v in sampled_edge_hist.items()},
            "edge_type_hist": {str(k): int(v) for k, v in sampled_edge_type_hist.items()},
            "motif_hist": {str(k): int(v) for k, v in sampled_motif_hist.items()},
        },

        "sampled_topologies": sampled,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"💾 已保存 graph grammar samples 至: {OUTPUT_FILE}")
    print("\n📌 下一步：")
    print("  1. 打开 graph_grammar_samples.json 检查 sampled_topologies")
    print("  2. 如果 connected ratio 低，调高 prefer_connected 或在 scorer 里强罚 disconnected")
    print("  3. 如果 stroke_count 过小/过大，调整 STYLE_CONFIG['stroke_count_range']")
    print("  4. 下一步写 stroke_primitive_sampler.py，把 shape_code / width_token 填进 nodes")


if __name__ == "__main__":
    main()