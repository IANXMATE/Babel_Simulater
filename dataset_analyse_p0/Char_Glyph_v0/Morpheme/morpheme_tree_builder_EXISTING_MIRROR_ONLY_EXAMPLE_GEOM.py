# -*- coding: utf-8 -*-
r"""
morpheme_tree_builder.py

拓扑语素广义树 / 图式语素库构建脚本。

放置：
    dataset_analyse_p0/Char_Glyph_v0/Morpheme/morpheme_tree_builder.py

核心语义：
1. 语素只针对 topology / 母线骨架，不使用笔画样式、宽度、曲率细节。
   每条 stroke 在本脚本里被退化为 P0 -> P3 的直线段。
2. 单个语素整体旋转、平移、统一缩放，不改变语素分桶。
   对复合语素，子语素之间的相对角度、距离、连接方式会进入分桶。
3. 语素库是广义二叉树 / 图：
      morpheme_node <- composition_alternatives(children + topology_way)
   一个 parent morpheme 可以有多个 composition alternative。
4. 最底层永远是 LINE。
5. 两笔连接会按角度簇分类：acute / right / obtuse / collinear。
6. 镜像语素是特殊组合：single morpheme -> mirror/rotate copy -> new morpheme proposal。
"""

from __future__ import annotations

import os, json, math, time, copy, argparse, hashlib, itertools, traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_GLYPH_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) if os.path.basename(SCRIPT_DIR).lower() == "morpheme" else SCRIPT_DIR
DATASET_ANALYSE_DIR = os.path.abspath(os.path.join(CHAR_GLYPH_DIR, ".."))

DEFAULT_GOOD_DIR = os.path.join(CHAR_GLYPH_DIR, "annotation_tool", "pcg_filebacked_stage2_schema", "good")
DEFAULT_CLEANED_DIR = os.path.join(CHAR_GLYPH_DIR, "annotation_tool", "pcg_filebacked_stage2_schema", "cleaned")
DEFAULT_ANNOTATIONS_DIR = os.path.join(DATASET_ANALYSE_DIR, "AI_VECTOR_ROUTER_With_topo", "annotations_topo")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "output_tree")

DEFAULT_ENDPOINT_TOL = 4.0
DEFAULT_MAX_ENUM_STROKES = 8
DEFAULT_MAX_COMPOSITION_STROKES = 8
DEFAULT_CLUSTER_TH = 0.14
DEFAULT_STABLE_MIN_SUPPORT = 2
DEFAULT_MAX_JSON_MB = 80
DEFAULT_RIGHT_TOL_DEG = 12.0
DEFAULT_COLLINEAR_TOL_DEG = 15.0
DEFAULT_SYMMETRY_TH = 0.08
DEFAULT_MAX_RAW_SUBSETS_PER_GLYPH = 2500


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def norm_path(path):
    return os.path.abspath(path).replace("\\", "/")


def stable_json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha1_obj(obj, n=24):
    return hashlib.sha1(stable_json(obj).encode("utf-8")).hexdigest()[:n]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path, indent=2):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=True)
    os.replace(tmp, path)


def json_size_bytes(obj):
    return len(stable_json(obj).encode("utf-8"))


def save_shards(rows, out_dir, prefix, max_mb=80, extra=None):
    ensure_dir(out_dir)
    for fn in os.listdir(out_dir):
        if fn.startswith(prefix + "_shard_") and fn.endswith(".json"):
            try:
                os.remove(os.path.join(out_dir, fn))
            except Exception:
                pass

    max_bytes = int(max_mb * 1024 * 1024)
    shards, cur, cur_size = [], [], 0
    for r in rows:
        sz = json_size_bytes(r) + 4
        if cur and cur_size + sz > max_bytes:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(r)
        cur_size += sz
    if cur:
        shards.append(cur)

    files = []
    for i, sh in enumerate(shards):
        fn = f"{prefix}_shard_{i:04d}.json"
        p = os.path.join(out_dir, fn)
        save_json({"prefix": prefix, "shard_index": i, "row_count": len(sh), "rows": sh}, p, indent=None)
        files.append({"file": fn, "row_count": len(sh), "size_bytes": os.path.getsize(p)})

    manifest = {"prefix": prefix, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "row_count": len(rows), "max_mb": int(max_mb), "shards": files}
    if extra:
        manifest.update(extra)
    save_json(manifest, os.path.join(out_dir, f"{prefix}_manifest.json"), indent=2)
    return manifest


def iter_json_files(root):
    if not root or not os.path.exists(root):
        return
    if os.path.isfile(root) and root.lower().endswith(".json"):
        yield os.path.abspath(root)
        return
    for dp, _, fns in os.walk(root):
        for fn in fns:
            low = fn.lower()
            if low.endswith(".json") and not low.endswith(".tmp"):
                yield os.path.abspath(os.path.join(dp, fn))


def resolve_workers(workers):
    if str(workers).lower() == "auto":
        return max(1, (os.cpu_count() or 1) - 1)
    try:
        return max(1, int(workers))
    except Exception:
        return 1


def flatten_records(obj):
    if obj is None:
        return
    if isinstance(obj, list):
        for i, x in enumerate(obj):
            if isinstance(x, dict):
                yield str(i), x
        return
    if not isinstance(obj, dict):
        return

    if isinstance(obj.get("strokes"), list) or isinstance(obj.get("solved_nodes"), list) or isinstance(obj.get("nodes"), list):
        gi = obj.get("glyph_info", {}) if isinstance(obj.get("glyph_info"), dict) else {}
        yield str(obj.get("candidate_id") or gi.get("candidate_id") or gi.get("hex_key") or "direct"), obj
        return

    for key in ["items", "records", "candidates", "bundles", "good", "cleaned", "data"]:
        v = obj.get(key)
        if isinstance(v, list):
            for i, x in enumerate(v):
                if isinstance(x, dict):
                    yield f"{key}:{i}", x

    for k, v in obj.items():
        if isinstance(v, dict) and (isinstance(v.get("strokes"), list) or isinstance(v.get("solved_nodes"), list) or isinstance(v.get("nodes"), list)):
            yield str(k), v


def glyph_key(rec, fallback):
    gi = rec.get("glyph_info", {}) if isinstance(rec.get("glyph_info"), dict) else {}
    for k in ["hex_key", "unicode", "codepoint", "char"]:
        if gi.get(k) is not None:
            return str(gi[k])
    for k in ["hex_key", "unicode", "codepoint", "char"]:
        if rec.get(k) is not None:
            return str(rec[k])
    return str(fallback)


def record_id(rec, fallback):
    gi = rec.get("glyph_info", {}) if isinstance(rec.get("glyph_info"), dict) else {}
    for k in ["candidate_id", "generated_glyph_id", "source_candidate_id", "glyph_candidate_id", "sample_id", "id"]:
        if rec.get(k):
            return str(rec[k])
    for k in ["candidate_id", "hex_key", "char"]:
        if gi.get(k):
            return str(gi[k])
    return str(fallback)


def node_list(rec):
    for k in ["strokes", "solved_nodes", "nodes", "solved_segments"]:
        v = rec.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def as_bezier(v):
    try:
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape == (4, 2):
            return arr
    except Exception:
        pass
    return None


def get_bezier(nd):
    for k in ["mother_bezier", "bezier", "curve", "solved_bezier", "control_points", "path"]:
        P = as_bezier(nd.get(k))
        if P is not None:
            return P
    return None


@dataclass
class Segment:
    sid: int
    p0: List[float]
    p1: List[float]

    def p0_np(self):
        return np.asarray(self.p0, dtype=np.float32)

    def p1_np(self):
        return np.asarray(self.p1, dtype=np.float32)

    def vec(self):
        return self.p1_np() - self.p0_np()

    def length(self):
        return float(np.linalg.norm(self.vec()))


@dataclass
class Glyph:
    glyph_key: str
    record_id: str
    source_label: str
    source_priority: int
    source_file: str
    segments: List[Segment]
    topology_events: List[Dict[str, Any]]
    cycles: List[Dict[str, Any]]

    def light(self):
        return {"glyph_key": self.glyph_key, "record_id": self.record_id, "source_label": self.source_label, "source_file": self.source_file,
                "stroke_count": len(self.segments), "topology_event_count": len(self.topology_events), "cycle_count": len(self.cycles)}


def parse_glyph(rec, gkey, rid, label, pri, path):
    segs = []
    for i, nd in enumerate(node_list(rec)):
        P = get_bezier(nd)
        if P is None:
            continue
        raw = nd.get("bezier_id", nd.get("id", nd.get("node_id", i + 1)))
        try:
            sid = int(raw)
        except Exception:
            sid = i + 1
        # topology-only: P0 -> P3 line only
        segs.append(Segment(sid=sid, p0=P[0].astype(float).tolist(), p1=P[3].astype(float).tolist()))
    if not segs:
        return None
    ev = rec.get("topology_events", [])
    cy = rec.get("cycles", [])
    return Glyph(str(gkey), str(rid), label, int(pri), norm_path(path), segs, ev if isinstance(ev, list) else [], cy if isinstance(cy, list) else [])


def load_glyphs(good_dir, cleaned_dir, annotations_dir):
    specs = [("good", good_dir, 1), ("cleaned", cleaned_dir, 2), ("annotations_topo", annotations_dir, 3)]
    chosen = {}
    total, parsed = 0, 0
    for label, root, pri in specs:
        if not root or not os.path.exists(root):
            print(f"[Load] missing {label}: {root}")
            continue
        files = list(iter_json_files(root) or [])
        print(f"[Load] {label}: files={len(files)}")
        for path in files:
            try:
                obj = load_json(path)
            except Exception as e:
                print(f"[Load][WARN] failed {path}: {repr(e)}")
                continue
            mtime = os.path.getmtime(path)
            for local_id, rec in flatten_records(obj) or []:
                if not isinstance(rec, dict):
                    continue
                total += 1
                fb = f"{norm_path(path)}:{local_id}"
                g = parse_glyph(rec, glyph_key(rec, fb), record_id(rec, fb), label, pri, path)
                if g is None:
                    continue
                parsed += 1
                old = chosen.get(g.glyph_key)
                if old is None or pri > old[0] or (pri == old[0] and mtime >= old[1]):
                    chosen[g.glyph_key] = (pri, mtime, g)
    glyphs = [x[2] for x in chosen.values()]
    glyphs.sort(key=lambda x: (x.source_priority, x.glyph_key, x.record_id))
    print(f"[Load] total={total} parsed={parsed} deduped={len(glyphs)} sources={dict(Counter(g.source_label for g in glyphs))}")
    return glyphs


def cluster_endpoint_points(points, tol):
    labels, clusters = [], []
    for p in points:
        assigned = -1
        for ci, pts in enumerate(clusters):
            c = np.mean(np.stack(pts, axis=0), axis=0)
            if float(np.linalg.norm(p - c)) <= tol:
                assigned = ci
                break
        if assigned < 0:
            clusters.append([p])
            labels.append(len(clusters) - 1)
        else:
            clusters[assigned].append(p)
            labels.append(assigned)
    return labels


def angle_between(v1, v2):
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def undirected_line_angle(v1, v2):
    a = angle_between(v1, v2)
    return min(a, 180.0 - a)


def classify_angle(angle, right_tol=12.0, collinear_tol=15.0):
    a = float(angle)
    if a <= collinear_tol or a >= 180.0 - collinear_tol:
        return "collinear"
    if abs(a - 90.0) <= right_tol:
        return "right"
    if a < 90.0:
        return "acute"
    return "obtuse"


def endpoint_connection_angle(s1, s2):
    endpoints1 = [(s1.p0_np(), s1.p1_np()), (s1.p1_np(), s1.p0_np())]
    endpoints2 = [(s2.p0_np(), s2.p1_np()), (s2.p1_np(), s2.p0_np())]
    best = None
    for j1, o1 in endpoints1:
        for j2, o2 in endpoints2:
            d = float(np.linalg.norm(j1 - j2))
            if best is None or d < best[0]:
                best = (d, j1, o1, j2, o2)
    _, j1, o1, j2, o2 = best
    junction = (j1 + j2) * 0.5
    return angle_between(o1 - junction, o2 - junction)


def relation_angle(s1, s2, rel_type):
    typ = str(rel_type).upper()
    if typ.startswith("E2E"):
        return endpoint_connection_angle(s1, s2)
    return undirected_line_angle(s1.vec(), s2.vec())


def build_graph(glyph, endpoint_tol=4.0, right_tol=12.0, collinear_tol=15.0):
    n = len(glyph.segments)
    nodes = list(range(n))
    sid2idx = {int(s.sid): i for i, s in enumerate(glyph.segments)}
    edge_map = {}

    def add_edge(i, j, typ):
        if i == j:
            return
        a, b = sorted((int(i), int(j)))
        ang = relation_angle(glyph.segments[a], glyph.segments[b], typ)
        ac = classify_angle(ang, right_tol=right_tol, collinear_tol=collinear_tol)
        key = (a, b, str(typ).upper(), ac)
        edge_map[key] = {"a": a, "b": b, "type": str(typ).upper(), "angle": round(float(ang), 4), "angle_class": ac}

    for ev in glyph.topology_events:
        if not isinstance(ev, dict):
            continue
        typ = str(ev.get("type", "")).upper()
        if typ == "E2E":
            x, y = ev.get("stroke_a"), ev.get("stroke_b")
        elif typ == "T":
            x, y = ev.get("guest"), ev.get("host")
        elif typ == "X":
            x, y = ev.get("stroke_a"), ev.get("stroke_b")
        else:
            continue
        try:
            add_edge(sid2idx[int(x)], sid2idx[int(y)], typ)
        except Exception:
            pass

    pts, meta = [], []
    for i, s in enumerate(glyph.segments):
        pts += [s.p0_np(), s.p1_np()]
        meta += [(i, 0), (i, 1)]
    if pts:
        labels = cluster_endpoint_points(pts, endpoint_tol)
        by_lab = defaultdict(list)
        for lab, (si, ep) in zip(labels, meta):
            by_lab[lab].append(si)
        for ss in by_lab.values():
            uniq = sorted(set(ss))
            if len(uniq) >= 2:
                for a, b in itertools.combinations(uniq, 2):
                    add_edge(a, b, "E2E_GEOM")
    return nodes, list(edge_map.values())


def connected_components(nodes, edges):
    adj = defaultdict(list)
    for e in edges:
        adj[int(e["a"])].append(int(e["b"]))
        adj[int(e["b"])].append(int(e["a"]))
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(sorted(comp))
    return comps


def is_connected_subset(sub, edges):
    ss = set(sub)
    if len(ss) <= 1:
        return True
    adj = defaultdict(list)
    for e in edges:
        a, b = int(e["a"]), int(e["b"])
        if a in ss and b in ss:
            adj[a].append(b)
            adj[b].append(a)
    start = sub[0]
    seen, stack = {start}, [start]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return seen == ss


def enum_connected_subsets(nodes, edges, max_enum_strokes, max_raw_subsets):
    out, seen = [], set()
    def add(s):
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            out.append(key)

    for comp in connected_components(nodes, edges):
        for x in comp:
            add((x,))
        if len(comp) >= 2:
            max_k = min(len(comp), max_enum_strokes)
            for k in range(2, max_k + 1):
                for cmb in itertools.combinations(comp, k):
                    if is_connected_subset(cmb, edges):
                        add(cmb)
                        if len(out) >= max_raw_subsets:
                            return out
            add(tuple(comp))
    return out[:max_raw_subsets]


def normalize_points(X, pca=True):
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    scale = float(np.sqrt((X ** 2).sum(axis=1).max()))
    if scale < 1e-6:
        scale = 1.0
    X = X / scale
    if pca and len(X) >= 3:
        try:
            cov = X.T @ X
            vals, vecs = np.linalg.eigh(cov)
            axis = vecs[:, int(np.argmax(vals))]
            theta = math.atan2(float(axis[1]), float(axis[0]))
            c, s = math.cos(-theta), math.sin(-theta)
            R = np.asarray([[c, -s], [s, c]], dtype=np.float32)
            X = X @ R.T
            if np.sum(X[:, 0]) < -1e-6:
                X[:, 0] *= -1
            if np.sum(X[:, 1]) < -1e-6:
                X[:, 1] *= -1
        except Exception:
            pass
    return X.astype(np.float32)


def subset_edges(sub, edges):
    ss = set(sub)
    return [e for e in edges if int(e["a"]) in ss and int(e["b"]) in ss]


def degree_hist(sub, edges):
    deg = Counter({i: 0 for i in sub})
    for e in subset_edges(sub, edges):
        deg[int(e["a"])] += 1
        deg[int(e["b"])] += 1
    return tuple(sorted(deg.values()))


def cycle_rank(sub, edges):
    ie = {(min(int(e["a"]), int(e["b"])), max(int(e["a"]), int(e["b"]))) for e in subset_edges(sub, edges)}
    return max(0, len(ie) - len(sub) + 1)


def relation_hist(sub, edges):
    c = Counter()
    for e in subset_edges(sub, edges):
        c[str(e["type"])] += 1
    return dict(sorted(c.items()))


def angle_class_hist(sub, edges):
    c = Counter()
    for e in subset_edges(sub, edges):
        c[str(e["angle_class"])] += 1
    return dict(sorted(c.items()))


def relation_angles(sub, edges):
    return [float(e["angle"]) for e in subset_edges(sub, edges)]


def wl_hash(sub, edges, rounds=3):
    ss = set(sub)
    adj = defaultdict(list)
    for e in edges:
        a, b = int(e["a"]), int(e["b"])
        if a in ss and b in ss:
            lab = f"{e['type']}:{e['angle_class']}"
            adj[a].append((b, lab))
            adj[b].append((a, lab))
    colors = {n: f"N:deg{len(adj[n])}" for n in sub}
    hist = []
    for _ in range(rounds):
        new = {}
        for n in sub:
            neigh = sorted(f"{lab}:{colors.get(nb, 'UNK')}" for nb, lab in adj[n])
            new[n] = sha1_obj([colors[n], neigh], 16)
        colors = new
        hist.append(sorted(colors.values()))
    return sha1_obj(hist, 24)


def subset_descriptor(glyph, sub, edges):
    if len(sub) == 1:
        return [1.0]

    endpoints, lengths = [], []
    for i in sub:
        s = glyph.segments[i]
        endpoints += [s.p0_np(), s.p1_np()]
        lengths.append(s.length())

    E = normalize_points(np.stack(endpoints, axis=0), pca=True)

    pair_ds = []
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            pair_ds.append(float(np.linalg.norm(E[i] - E[j])))
    pair_ds = sorted(pair_ds)

    med = float(np.median(lengths)) if lengths else 1.0
    if med < 1e-6:
        med = 1.0
    norm_lengths = sorted([float(x / med) for x in lengths])

    line_angles = []
    for k, _i in enumerate(sub):
        a, b = E[2*k], E[2*k+1]
        v = b - a
        line_angles.append(float((math.atan2(float(v[1]), float(v[0])) % math.pi) / math.pi))
    line_angles = sorted(line_angles)

    rel_ang = sorted([float(x) / 180.0 for x in relation_angles(sub, edges)])

    xmin, ymin = np.min(E, axis=0)
    xmax, ymax = np.max(E, axis=0)
    ratio = float((xmax - xmin) / max(float(ymax - ymin), 1e-6))

    return [float(x) for x in pair_ds + norm_lengths + line_angles + rel_ang + [min(ratio, 10.0), float(len(sub))]]


def vec_l2(a, b):
    n = max(len(a), len(b))
    aa = np.asarray(a + [0.0]*(n-len(a)), dtype=np.float32)
    bb = np.asarray(b + [0.0]*(n-len(b)), dtype=np.float32)
    return float(np.linalg.norm(aa-bb) / math.sqrt(max(1,n)))


def family_key_for_subset(glyph, sub, edges):
    if len(sub) == 1:
        return "LINE"
    obj = {"stroke_count": len(sub), "degree_hist": degree_hist(sub, edges), "cycle_rank": cycle_rank(sub, edges),
           "relation_hist": relation_hist(sub, edges), "angle_class_hist": angle_class_hist(sub, edges), "wl": wl_hash(sub, edges, 3)}
    return sha1_obj(obj, 24)


def local_segments_json(glyph, sub):
    rows = []
    for local_i, si in enumerate(sub, 1):
        s = glyph.segments[si]
        rows.append({"local_id": int(local_i), "source_sid": int(s.sid),
                     "p0": [round(float(s.p0[0]),4), round(float(s.p0[1]),4)],
                     "p1": [round(float(s.p1[0]),4), round(float(s.p1[1]),4)]})
    return rows


def example_segments_json(glyph, sub, group_by_index=None):
    """
    保存某个 composition example 的真实几何。
    group_by_index:
        {stroke_index_in_glyph: "child_a" / "child_b" / ...}

    这个函数用于 preview 的完美分割：
      parent_example_segments 中每一条边都带 group 字段，
      因此 parent 上不会因为 cluster prototype 与 example 不一致而出现大量灰线。
    """
    group_by_index = group_by_index or {}
    rows = []
    for local_i, si in enumerate(sub, 1):
        s = glyph.segments[si]
        rows.append({
            "local_id": int(local_i),
            "source_sid": int(s.sid),
            "glyph_stroke_index": int(si),
            "group": str(group_by_index.get(si, "other")),
            "p0": [round(float(s.p0[0]), 4), round(float(s.p0[1]), 4)],
            "p1": [round(float(s.p1[0]), 4), round(float(s.p1[1]), 4)],
        })
    return rows


@dataclass
class RawInstance:
    instance_id: str
    glyph_key: str
    record_id: str
    source_label: str
    source_file: str
    subset: Tuple[int, ...]
    source_sids: Tuple[int, ...]
    stroke_count: int
    family_key: str
    descriptor: List[float]
    relation_hist: Dict[str, int]
    angle_class_hist: Dict[str, int]
    relation_angles: List[float]
    degree_hist: Tuple[int, ...]
    cycle_rank: int
    prototype_segments: List[Dict[str, Any]]


def make_raw_instance(glyph, sub, edges):
    sids = tuple(int(glyph.segments[i].sid) for i in sub)
    fam = family_key_for_subset(glyph, sub, edges)
    iid = "I_" + sha1_obj({"glyph": glyph.glyph_key, "sids": sids, "fam": fam}, 24)
    return RawInstance(iid, glyph.glyph_key, glyph.record_id, glyph.source_label, glyph.source_file, tuple(map(int, sub)),
                       sids, len(sub), fam, subset_descriptor(glyph, sub, edges), relation_hist(sub, edges),
                       angle_class_hist(sub, edges), relation_angles(sub, edges), degree_hist(sub, edges),
                       cycle_rank(sub, edges), local_segments_json(glyph, sub))


def extract_instances_from_glyph_payload(payload):
    try:
        glyph = payload["glyph"]
        nodes, edges = build_graph(glyph, endpoint_tol=payload["endpoint_tol"], right_tol=payload["right_tol"], collinear_tol=payload["collinear_tol"])
        subsets = enum_connected_subsets(nodes, edges, payload["max_enum_strokes"], payload["max_raw_subsets"])
        rows = [asdict(make_raw_instance(glyph, sub, edges)) for sub in subsets]
        roots = []
        for comp in connected_components(nodes, edges):
            roots.append({"subset": list(map(int, comp)), "source_sids": [int(glyph.segments[i].sid) for i in comp]})
        return {"ok": True, "glyph_key": glyph.glyph_key, "instances": rows, "roots": roots, "error": None}
    except Exception as e:
        return {"ok": False, "glyph_key": getattr(payload.get("glyph"), "glyph_key", "unknown"),
                "instances": [], "roots": [], "error": repr(e), "traceback": traceback.format_exc()[:2000]}


def extract_all_instances(glyphs, args):
    w = resolve_workers(args.workers)
    print(f"[Extract] glyphs={len(glyphs)} workers={w}")
    t0 = time.time()
    payloads = [{"glyph": g, "endpoint_tol": args.endpoint_tol, "right_tol": args.right_tol, "collinear_tol": args.collinear_tol,
                 "max_enum_strokes": args.max_enum_strokes, "max_raw_subsets": args.max_raw_subsets_per_glyph} for g in glyphs]
    rows, roots_by_glyph, errors = [], {}, []
    if w <= 1:
        for i,p in enumerate(payloads, 1):
            res = extract_instances_from_glyph_payload(p)
            if res["ok"]:
                rows.extend(res["instances"]); roots_by_glyph[res["glyph_key"]] = res["roots"]
            else:
                errors.append(res)
            if i % 50 == 0 or i == len(payloads):
                print(f"[Extract] {i}/{len(payloads)} raw_instances={len(rows)} elapsed={time.time()-t0:.1f}s")
    else:
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = [ex.submit(extract_instances_from_glyph_payload, p) for p in payloads]
            for i, f in enumerate(as_completed(futs), 1):
                res = f.result()
                if res["ok"]:
                    rows.extend(res["instances"]); roots_by_glyph[res["glyph_key"]] = res["roots"]
                else:
                    errors.append(res)
                if i % 50 == 0 or i == len(futs):
                    print(f"[Extract] {i}/{len(futs)} raw_instances={len(rows)} elapsed={time.time()-t0:.1f}s")
    insts = []
    for r in rows:
        r["subset"] = tuple(r["subset"])
        r["source_sids"] = tuple(r["source_sids"])
        r["degree_hist"] = tuple(r["degree_hist"])
        insts.append(RawInstance(**r))
    return insts, roots_by_glyph, errors


def mean_std(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {"mean": round(float(np.mean(arr)),6), "std": round(float(np.std(arr)),6), "min": round(float(np.min(arr)),6), "max": round(float(np.max(arr)),6)}


def cluster_instances(instances, cluster_th, stable_min_support):
    print(f"[Cluster] start raw_instances={len(instances)} cluster_th={cluster_th} stable_min_support={stable_min_support}", flush=True)
    t0_cluster = time.time()

    groups = defaultdict(list)
    for inst in instances:
        groups[(inst.stroke_count, inst.family_key)].append(inst)

    print(f"[Cluster] groups={len(groups)}", flush=True)

    instance_to_cluster = {}
    clusters_out = []

    line_insts = [x for x in instances if x.stroke_count == 1]
    if line_insts:
        for inst in line_insts:
            instance_to_cluster[inst.instance_id] = "M_LINE"
        clusters_out.append({"morpheme_id": "M_LINE", "kind": "base_line", "stroke_count": 1, "family_key": "LINE",
                             "support_count": len(line_insts), "source_glyph_count": len(set(x.glyph_key for x in line_insts)),
                             "source_glyphs": sorted(set(x.glyph_key for x in line_insts))[:50], "is_stable": True,
                             "descriptor_mean": [1.0], "descriptor_std": [0.0], "relation_hist": {}, "angle_class_hist": {},
                             "angle_stats": {}, "degree_hist_mode": [0], "cycle_rank_mode": 0,
                             "prototype_segments": line_insts[0].prototype_segments, "composition_count": 0, "symmetry": {}})

    next_id = 0
    sorted_groups = sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    for gi, ((stroke_count, fam), insts) in enumerate(sorted_groups, 1):
        if stroke_count == 1:
            continue
        if gi % 100 == 0 or len(insts) > 200:
            print(f"[Cluster] group {gi}/{len(sorted_groups)} stroke_count={stroke_count} insts={len(insts)} clusters_so_far={len(clusters_out)} elapsed={time.time()-t0_cluster:.1f}s", flush=True)
        online_clusters = []
        for inst in insts:
            best_i, best_d = None, float("inf")
            for ci, c in enumerate(online_clusters):
                d = vec_l2(inst.descriptor, c["mean"])
                if d < best_d:
                    best_i, best_d = ci, d
            if best_i is not None and best_d <= cluster_th:
                c = online_clusters[best_i]
                c["instances"].append(inst)
                c["mean"] = np.mean(np.asarray([x.descriptor for x in c["instances"]], dtype=np.float32), axis=0).astype(float).tolist()
            else:
                online_clusters.append({"mean": list(inst.descriptor), "instances": [inst]})

        for c in online_clusters:
            inst_list = c["instances"]
            cid = f"M{next_id:06d}_{stroke_count}s"; next_id += 1
            for inst in inst_list:
                instance_to_cluster[inst.instance_id] = cid

            desc_arr = np.asarray([x.descriptor for x in inst_list], dtype=np.float32)
            rel_c, angle_c, deg_c, cyc_c = Counter(), Counter(), Counter(), Counter()
            angles = []
            for x in inst_list:
                rel_c.update(x.relation_hist); angle_c.update(x.angle_class_hist); deg_c[tuple(x.degree_hist)] += 1; cyc_c[int(x.cycle_rank)] += 1
                angles.extend([float(a) for a in x.relation_angles])
            clusters_out.append({"morpheme_id": cid, "kind": "topology_morpheme", "stroke_count": int(stroke_count), "family_key": fam,
                                 "support_count": len(inst_list), "source_glyph_count": len(set(x.glyph_key for x in inst_list)),
                                 "source_glyphs": sorted(set(x.glyph_key for x in inst_list))[:100],
                                 "is_stable": len(set(x.glyph_key for x in inst_list)) >= stable_min_support,
                                 "descriptor_mean": [round(float(v),6) for v in np.mean(desc_arr, axis=0).tolist()],
                                 "descriptor_std": [round(float(v),6) for v in np.std(desc_arr, axis=0).tolist()],
                                 "relation_hist": dict(sorted(rel_c.items())), "angle_class_hist": dict(sorted(angle_c.items())),
                                 "angle_stats": mean_std(angles),
                                 "degree_hist_mode": list(max(deg_c.items(), key=lambda kv: kv[1])[0]) if deg_c else [],
                                 "cycle_rank_mode": int(max(cyc_c.items(), key=lambda kv: kv[1])[0]) if cyc_c else 0,
                                 "prototype_segments": inst_list[0].prototype_segments, "composition_count": 0, "symmetry": {}})
    print(f"[Cluster] done clusters={len(clusters_out)} elapsed={time.time()-t0_cluster:.1f}s", flush=True)
    return clusters_out, instance_to_cluster, {c["morpheme_id"]: c for c in clusters_out}


def inter_edges_between(A, B, edges):
    out = []
    for e in edges:
        a, b = int(e["a"]), int(e["b"])
        if (a in A and b in B) or (a in B and b in A):
            out.append(e)
    return out


def composition_signature(inter_edges):
    rel, ang, angles = Counter(), Counter(), []
    for e in inter_edges:
        rel[str(e["type"])] += 1
        ang[str(e["angle_class"])] += 1
        angles.append(float(e["angle"]))
    return {"relation_hist": dict(sorted(rel.items())), "angle_class_hist": dict(sorted(ang.items())),
            "edge_count": len(inter_edges), "angle_stats": mean_std(angles)}


def split_partitions(sub):
    arr = tuple(sorted(sub))
    if len(arr) <= 1:
        return
    first, rest = arr[0], arr[1:]
    for r in range(0, len(rest)):
        for comb in itertools.combinations(rest, r):
            A = tuple(sorted((first,) + comb))
            if len(A) == len(arr):
                continue
            B = tuple(x for x in arr if x not in set(A))
            if B:
                yield A, tuple(sorted(B))


def build_compositions(glyphs, raw_instances, instance_to_cluster, args):
    if getattr(args, "skip_composition", False):
        print("[Composition] skipped by --skip-composition", flush=True)
        return []

    if args.max_composition_raw_instances > 0 and len(raw_instances) > args.max_composition_raw_instances:
        print(f"[Composition] raw_instances={len(raw_instances)} capped to {args.max_composition_raw_instances}", flush=True)
        # Prefer larger morphemes first; they carry more useful construction alternatives.
        raw_instances = sorted(raw_instances, key=lambda x: (-x.stroke_count, x.glyph_key, x.instance_id))[:args.max_composition_raw_instances]

    print("[Composition] building alternatives...", flush=True)
    t0 = time.time()
    glyph_by_key = {g.glyph_key: g for g in glyphs}
    inst_lookup = {(x.glyph_key, tuple(x.subset)): x for x in raw_instances}
    comp_map = {}
    for idx, inst in enumerate(raw_instances, 1):
        if inst.stroke_count <= 1 or inst.stroke_count > args.max_composition_strokes:
            continue
        glyph = glyph_by_key.get(inst.glyph_key)
        if glyph is None:
            continue
        _, edges = build_graph(glyph, endpoint_tol=args.endpoint_tol, right_tol=args.right_tol, collinear_tol=args.collinear_tol)
        parent_id = instance_to_cluster.get(inst.instance_id)
        if not parent_id:
            continue
        for A, B in split_partitions(inst.subset):
            if not is_connected_subset(A, edges) or not is_connected_subset(B, edges):
                continue
            inter = inter_edges_between(set(A), set(B), edges)
            if not inter:
                continue
            ia, ib = inst_lookup.get((inst.glyph_key, tuple(A))), inst_lookup.get((inst.glyph_key, tuple(B)))
            if ia is None or ib is None:
                continue
            ca, cb = instance_to_cluster.get(ia.instance_id), instance_to_cluster.get(ib.instance_id)
            if ca is None or cb is None:
                continue
            topo = composition_signature(inter)
            topo_hash = sha1_obj(topo, 20)
            child_ids = tuple(sorted([ca, cb]))
            key = (parent_id, child_ids, topo_hash)
            if key not in comp_map:
                comp_map[key] = {"composition_id": "C_" + sha1_obj({"p": parent_id, "c": child_ids, "t": topo_hash}, 24),
                                 "parent_morpheme_id": parent_id, "child_morpheme_ids": list(child_ids),
                                 "topology_way_hash": topo_hash, "topology_way": topo,
                                 "support_count": 0, "source_glyphs": set(), "examples": []}
            row = comp_map[key]
            row["support_count"] += 1
            row["source_glyphs"].add(inst.glyph_key)
            if len(row["examples"]) < 10:
                group_by_index = {}
                for _si in A:
                    group_by_index[int(_si)] = "child_a"
                for _si in B:
                    group_by_index[int(_si)] = "child_b"

                row["examples"].append({
                    "glyph_key": inst.glyph_key,
                    "parent_morpheme_id": parent_id,
                    "child_a_morpheme_id": ca,
                    "child_b_morpheme_id": cb,

                    # source_sids 便于人读；但真正 preview 用下面的 *_example_segments。
                    "parent_source_sids": list(inst.source_sids),
                    "child_a_source_sids": list(ia.source_sids),
                    "child_b_source_sids": list(ib.source_sids),

                    # 真实 example-level 几何，保证 parent 被 child_a/child_b 完美分割。
                    "parent_example_segments": example_segments_json(glyph, inst.subset, group_by_index=group_by_index),
                    "child_a_example_segments": example_segments_json(glyph, A, group_by_index={int(x): "child_a" for x in A}),
                    "child_b_example_segments": example_segments_json(glyph, B, group_by_index={int(x): "child_b" for x in B}),
                })
        if idx % 5000 == 0:
            print(f"[Composition] processed {idx}/{len(raw_instances)} alternatives={len(comp_map)} elapsed={time.time()-t0:.1f}s")
    rows = []
    for row in comp_map.values():
        r = dict(row)
        r["source_glyph_count"] = len(row["source_glyphs"])
        r["source_glyphs"] = sorted(row["source_glyphs"])[:100]
        rows.append(r)
    rows.sort(key=lambda x: (-x["support_count"], x["parent_morpheme_id"], x["topology_way_hash"]))
    print(f"[Composition] alternatives={len(rows)} elapsed={time.time()-t0:.1f}s")
    return rows


def normalize_segments(segments):
    pairs = []
    for s in segments:
        p0 = np.asarray(s.get("p0"), dtype=np.float32)
        p1 = np.asarray(s.get("p1"), dtype=np.float32)
        if p0.shape == (2,) and p1.shape == (2,):
            pairs.append([p0, p1])
    if not pairs:
        return np.zeros((0, 2, 2), dtype=np.float32)
    arr = np.asarray(pairs, dtype=np.float32)
    pts = normalize_points(arr.reshape(-1, 2), pca=True)
    return pts.reshape(arr.shape).astype(np.float32)


def transform_segments(seg_arr, typ):
    X = np.asarray(seg_arr, dtype=np.float32).copy()
    if typ == "mirror_x":
        X[:, :, 0] *= -1
    elif typ == "mirror_y":
        X[:, :, 1] *= -1
    elif typ == "rot2":
        X *= -1
    elif typ == "rot3":
        theta = 2 * math.pi / 3
        c, s = math.cos(theta), math.sin(theta)
        X = X @ np.asarray([[c, -s], [s, c]], dtype=np.float32).T
    elif typ == "rot4":
        theta = math.pi / 2
        c, s = math.cos(theta), math.sin(theta)
        X = X @ np.asarray([[c, -s], [s, c]], dtype=np.float32).T
    return X


def segment_cost(a, b):
    d1 = float(np.linalg.norm(a[0] - b[0]) + np.linalg.norm(a[1] - b[1]))
    d2 = float(np.linalg.norm(a[0] - b[1]) + np.linalg.norm(a[1] - b[0]))
    return min(d1, d2) * 0.5


def match_segment_sets(A, B):
    """
    Fast stroke-set matching for symmetry.

    旧版对 n<=7 使用全排列精确匹配，会产生 n! 复杂度；
    当 morpheme cluster 很多时，这一步会看起来像“卡住”。

    这里改成贪心双向近似：
      score = 0.5 * (A->B greedy + B->A greedy)
    对 symmetry report 足够用，速度从阶乘降为 O(n^2)。
    """
    n = len(A)
    if n != len(B) or n == 0:
        return float("inf")

    def greedy_cost(X, Y):
        rem = set(range(len(Y)))
        total = 0.0
        for i in range(len(X)):
            best_j, best_c = None, float("inf")
            for j in rem:
                c = segment_cost(X[i], Y[j])
                if c < best_c:
                    best_j, best_c = j, c
            total += best_c
            rem.remove(best_j)
        return float(total / max(1, len(X)))

    return 0.5 * (greedy_cost(A, B) + greedy_cost(B, A))


def analyze_cluster_symmetry(cluster, th):
    segs = normalize_segments(cluster.get("prototype_segments", []))
    scores = {}
    for typ in ["mirror_x", "mirror_y", "rot2", "rot3", "rot4"]:
        scores[typ] = round(match_segment_sets(segs, transform_segments(segs, typ)), 6)
    best_type, best_score = min(scores.items(), key=lambda kv: kv[1])
    return {"scores": scores, "best_symmetry_type": best_type, "best_symmetry_score": float(best_score),
            "strict_stroke_symmetry": bool(best_score <= th), "symmetry_threshold": float(th)}


def pair_mirror_score_for_clusters(a, b, op):
    """
    判断两个已经存在的 child morpheme 是否像 mirror/rot 关系。
    这里只做选择/标记，不生成任何新语素。
    """
    A = normalize_segments(a.get("prototype_segments", []))
    B = normalize_segments(b.get("prototype_segments", []))
    if len(A) != len(B) or len(A) == 0:
        return float("inf")
    return float(match_segment_sets(transform_segments(A, op), B))


def attach_symmetry_and_select_existing(clusters, args):
    """
    只分析/挑出现有语素中的镜像候选，不生成 existing mirror candidate。

    输出 existing_mirror_candidates:
      1. self_strict_symmetry:
         某个现有 morpheme 自身就是严格镜像/旋转对称。
      2. mirror_pair_child:
         后面结合 composition_alternatives 后再补充：
         parent 由两个已存在 child 组成，且 child_a 与 child_b 近似 mirror/rot 关系。
    """
    print(f"[Symmetry] start clusters={len(clusters)} symmetry_th={args.symmetry_th}", flush=True)
    t0 = time.time()
    out, existing = [], []

    for i, c in enumerate(clusters, 1):
        cc = dict(c)
        cc["symmetry"] = analyze_cluster_symmetry(c, args.symmetry_th)
        out.append(cc)

        sym = cc.get("symmetry") or {}
        if cc["morpheme_id"] != "M_LINE" and sym.get("strict_stroke_symmetry"):
            existing.append({
                "candidate_id": "MIR_EXIST_" + sha1_obj({"kind": "self", "mid": cc["morpheme_id"]}, 20),
                "kind": "self_strict_symmetry",
                "morpheme_id": cc["morpheme_id"],
                "operation": sym.get("best_symmetry_type"),
                "score": sym.get("best_symmetry_score"),
                "stroke_count": cc.get("stroke_count"),
                "support_count": cc.get("support_count"),
                "source_glyph_count": cc.get("source_glyph_count"),
                "note": "现有语素自身可被视为由镜像/旋转规则产生；这里只挑选，不生成新语素。",
            })

        if i % 1000 == 0 or i == len(clusters):
            print(f"[Symmetry] {i}/{len(clusters)} existing_self_mirror={len(existing)} elapsed={time.time()-t0:.1f}s", flush=True)

    print(f"[Symmetry] done elapsed={time.time()-t0:.1f}s", flush=True)
    return out, existing


def select_existing_mirror_from_compositions(clusters, compositions, args):
    """
    从已有 composition_alternatives 中挑出“parent 可以被视作镜像组合”的现有语素。
    不新增任何 morpheme。

    条件：
      - composition 只有两个 child。
      - child_a / child_b 的 prototype_segments 在 mirror_x / mirror_y / rot2 下接近。
      - 记录 parent + children + topology_way。
    """
    print("[MirrorSelect] selecting existing mirror-composition candidates...", flush=True)
    t0 = time.time()
    id_to_cluster = {c["morpheme_id"]: c for c in clusters}
    rows = []

    for i, comp in enumerate(compositions, 1):
        child_ids = list(comp.get("child_morpheme_ids", []))
        if len(child_ids) != 2:
            continue

        a = id_to_cluster.get(child_ids[0])
        b = id_to_cluster.get(child_ids[1])
        parent = id_to_cluster.get(comp.get("parent_morpheme_id"))
        if a is None or b is None or parent is None:
            continue

        # M_LINE + M_LINE 也可以构成 V/Λ/十字等，但这里先保留；后续 preview 肉眼看。
        scores = {}
        for op in ["mirror_x", "mirror_y", "rot2"]:
            scores[op] = pair_mirror_score_for_clusters(a, b, op)
        best_op, best_score = min(scores.items(), key=lambda kv: kv[1])

        if best_score <= args.mirror_pair_th:
            rows.append({
                "candidate_id": "MIR_COMP_" + sha1_obj({
                    "p": comp.get("parent_morpheme_id"),
                    "children": child_ids,
                    "op": best_op,
                    "topo": comp.get("topology_way_hash"),
                }, 20),
                "kind": "existing_composition_mirror_pair",
                "parent_morpheme_id": comp.get("parent_morpheme_id"),
                "child_morpheme_ids": child_ids,
                "operation": best_op,
                "score": round(float(best_score), 6),
                "all_scores": {k: round(float(v), 6) for k, v in scores.items()},
                "composition_id": comp.get("composition_id"),
                "topology_way_hash": comp.get("topology_way_hash"),
                "topology_way": comp.get("topology_way"),
                "support_count": comp.get("support_count"),
                "source_glyph_count": comp.get("source_glyph_count"),
                "examples": comp.get("examples", [])[:5],
                "note": "现有 parent composition 可被视为两个 child 的镜像/旋转组合；这里只挑选，不生成新语素。",
            })

        if i % 20000 == 0:
            print(f"[MirrorSelect] {i}/{len(compositions)} selected={len(rows)} elapsed={time.time()-t0:.1f}s", flush=True)

    rows.sort(key=lambda r: (r["score"], -int(r.get("support_count") or 0)))
    print(f"[MirrorSelect] selected={len(rows)} elapsed={time.time()-t0:.1f}s", flush=True)
    return rows


def build_glyph_roots(glyphs, raw_instances, instance_to_cluster, roots_by_glyph):
    inst_lookup = {(x.glyph_key, tuple(x.subset)): x for x in raw_instances}
    out = []
    for g in glyphs:
        roots = []
        for r in roots_by_glyph.get(g.glyph_key, []):
            sub = tuple(r["subset"])
            inst = inst_lookup.get((g.glyph_key, sub))
            cid = instance_to_cluster.get(inst.instance_id) if inst else None
            roots.append({"root_morpheme_id": cid, "subset": list(sub), "source_sids": r["source_sids"]})
        out.append({"glyph": g.light(), "roots": roots})
    return out


def build_greedy(glyphs, raw_instances, instance_to_cluster, id_to_cluster, args):
    print("[Greedy] building decomposition forest...")
    inst_lookup = {(x.glyph_key, tuple(x.subset)): x for x in raw_instances}

    def support(cid):
        c = id_to_cluster.get(cid, {})
        return int(c.get("source_glyph_count", c.get("support_count", 0)))

    def parse(glyph, subset, edges):
        inst = inst_lookup.get((glyph.glyph_key, tuple(sorted(subset))))
        cid = instance_to_cluster.get(inst.instance_id) if inst else None
        node = {"morpheme_id": cid, "stroke_indices": list(map(int, subset)),
                "source_sids": [int(glyph.segments[i].sid) for i in subset],
                "stroke_count": len(subset), "children": [], "topology_way": None, "is_leaf": False}
        if len(subset) <= 1:
            node["morpheme_id"] = cid or "M_LINE"; node["is_leaf"] = True; return node
        best = None
        for A, B in split_partitions(tuple(sorted(subset))):
            if not is_connected_subset(A, edges) or not is_connected_subset(B, edges):
                continue
            inter = inter_edges_between(set(A), set(B), edges)
            if not inter:
                continue
            ia, ib = inst_lookup.get((glyph.glyph_key, tuple(A))), inst_lookup.get((glyph.glyph_key, tuple(B)))
            if ia is None or ib is None:
                continue
            ca, cb = instance_to_cluster.get(ia.instance_id), instance_to_cluster.get(ib.instance_id)
            if ca is None or cb is None:
                continue
            score = (1 if support(ca) >= args.stable_min_support and support(cb) >= args.stable_min_support else 0,
                     min(len(A), len(B)), max(len(A), len(B)), support(ca) + support(cb), -abs(len(A)-len(B)))
            if best is None or score > best[0]:
                best = (score, A, B, composition_signature(inter))
        if best is None:
            node["is_leaf"] = True; return node
        _, A, B, topo = best
        node["children"] = [parse(glyph, A, edges), parse(glyph, B, edges)]
        node["topology_way"] = topo
        return node

    rows = []
    for i,g in enumerate(glyphs, 1):
        nodes, edges = build_graph(g, endpoint_tol=args.endpoint_tol, right_tol=args.right_tol, collinear_tol=args.collinear_tol)
        rows.append({"glyph": g.light(), "roots": [parse(g, tuple(comp), edges) for comp in connected_components(nodes, edges)]})
        if i % 100 == 0 or i == len(glyphs):
            print(f"[Greedy] {i}/{len(glyphs)}")
    return rows


def build_pipeline(args):
    ensure_dir(args.out_dir)
    t_all = time.time()

    glyphs = load_glyphs(args.good_dir, args.cleaned_dir, args.annotations_dir)
    save_json({"glyph_count": len(glyphs), "priority": "annotations_topo > cleaned > good", "glyphs": [g.light() for g in glyphs]},
              os.path.join(args.out_dir, "loaded_glyphs_manifest.json"))

    raw_instances, roots_by_glyph, errors = extract_all_instances(glyphs, args)
    print(f"[Extract] raw_instances={len(raw_instances)} errors={len(errors)}")

    clusters, instance_to_cluster, id_to_cluster = cluster_instances(raw_instances, args.cluster_th, args.stable_min_support)
    clusters, existing_mirror = attach_symmetry_and_select_existing(clusters, args)
    id_to_cluster = {c["morpheme_id"]: c for c in clusters}

    compositions = build_compositions(glyphs, raw_instances, instance_to_cluster, args)
    existing_mirror.extend(select_existing_mirror_from_compositions(clusters, compositions, args))
    comp_count = Counter(x["parent_morpheme_id"] for x in compositions)
    for c in clusters:
        c["composition_count"] = int(comp_count.get(c["morpheme_id"], 0))

    glyph_roots = build_glyph_roots(glyphs, raw_instances, instance_to_cluster, roots_by_glyph)
    greedy = build_greedy(glyphs, raw_instances, instance_to_cluster, id_to_cluster, args)

    stable_count = sum(1 for c in clusters if c.get("is_stable"))
    strict_sym_count = sum(1 for c in clusters if (c.get("symmetry") or {}).get("strict_stroke_symmetry"))

    sym_report = {"symmetry_threshold": args.symmetry_th, "cluster_count": len(clusters), "strict_symmetric_count": strict_sym_count,
                  "by_best_type": dict(Counter((c.get("symmetry") or {}).get("best_symmetry_type", "none") for c in clusters)),
                  "top_strict_symmetric": [{"morpheme_id": c["morpheme_id"], "stroke_count": c["stroke_count"], "support_count": c["support_count"],
                                            "source_glyph_count": c["source_glyph_count"], "best_symmetry_type": c["symmetry"]["best_symmetry_type"],
                                            "best_symmetry_score": c["symmetry"]["best_symmetry_score"]}
                                           for c in sorted(clusters, key=lambda x: (x.get("symmetry") or {}).get("best_symmetry_score", 999.0))[:100]]}

    save_shards(clusters, args.out_dir, "morpheme_nodes", args.max_json_mb,
                {"description": "topology-only morpheme clusters; LINE is bottom primitive", "cluster_threshold": args.cluster_th,
                 "stable_min_support": args.stable_min_support})
    save_shards(compositions, args.out_dir, "composition_alternatives", args.max_json_mb,
                {"description": "generalized binary-tree/graph alternatives: parent <- children + topology_way"})
    save_json({"glyph_count": len(glyph_roots), "roots": glyph_roots}, os.path.join(args.out_dir, "glyph_roots.json"))
    save_shards(greedy, args.out_dir, "greedy_decomposition", args.max_json_mb,
                {"description": "large-block-first recursive decomposition forest"})
    save_json(sym_report, os.path.join(args.out_dir, "symmetry_report.json"))
    save_json({"count": len(existing_mirror), "candidates": existing_mirror}, os.path.join(args.out_dir, "existing_mirror_candidates.json"))
    if errors:
        save_json({"error_count": len(errors), "errors": errors[:500]}, os.path.join(args.out_dir, "extract_errors.json"))

    summary = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "sources": {"good": norm_path(args.good_dir), "cleaned": norm_path(args.cleaned_dir),
                           "annotations_topo": norm_path(args.annotations_dir), "priority": "annotations_topo > cleaned > good"},
               "params": {"endpoint_tol": args.endpoint_tol, "right_tol": args.right_tol, "collinear_tol": args.collinear_tol,
                          "max_enum_strokes": args.max_enum_strokes, "max_composition_strokes": args.max_composition_strokes,
                          "cluster_th": args.cluster_th, "stable_min_support": args.stable_min_support,
                          "symmetry_th": args.symmetry_th, "workers": args.workers},
               "counts": {"glyph_count": len(glyphs), "raw_instance_count": len(raw_instances),
                          "morpheme_node_count": len(clusters), "stable_morpheme_count": stable_count,
                          "composition_alternative_count": len(compositions),
                          "strict_symmetric_morpheme_count": strict_sym_count,
                          "existing_mirror_candidate_count": len(existing_mirror), "extract_errors": len(errors)},
               "outputs": ["loaded_glyphs_manifest.json", "morpheme_nodes_manifest.json", "composition_alternatives_manifest.json",
                           "glyph_roots.json", "greedy_decomposition_manifest.json", "symmetry_report.json",
                           "existing_mirror_candidates.json", "summary.json"],
               "semantics": ["Morpheme is topology-only. Width/style/curve interior P1/P2 are ignored.",
                             "Each stroke is reduced to straight motherline P0->P3.",
                             "Global rotate/translate/uniform scale does not change morpheme bucket.",
                             "Composite morpheme bucket depends on child relative topology way, angle class, distance/length descriptor.",
                             "Bottom primitive is always M_LINE.",
                             "A parent can have multiple composition alternatives.",
                             "Mirror candidates are selected from existing morphemes/compositions only; no new mirror morphemes are generated."],
               "total_seconds": round(time.time() - t_all, 3)}
    save_json(summary, os.path.join(args.out_dir, "summary.json"))
    print("[Summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser():
    p = argparse.ArgumentParser(description="Build topology-only generalized morpheme tree/graph library.")
    p.add_argument("--good-dir", default=DEFAULT_GOOD_DIR)
    p.add_argument("--cleaned-dir", default=DEFAULT_CLEANED_DIR)
    p.add_argument("--annotations-dir", default=DEFAULT_ANNOTATIONS_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--endpoint-tol", type=float, default=DEFAULT_ENDPOINT_TOL)
    p.add_argument("--right-tol", type=float, default=DEFAULT_RIGHT_TOL_DEG)
    p.add_argument("--collinear-tol", type=float, default=DEFAULT_COLLINEAR_TOL_DEG)
    p.add_argument("--max-enum-strokes", type=int, default=DEFAULT_MAX_ENUM_STROKES)
    p.add_argument("--max-composition-strokes", type=int, default=DEFAULT_MAX_COMPOSITION_STROKES)
    p.add_argument("--max-raw-subsets-per-glyph", type=int, default=DEFAULT_MAX_RAW_SUBSETS_PER_GLYPH)
    p.add_argument("--cluster-th", type=float, default=DEFAULT_CLUSTER_TH)
    p.add_argument("--stable-min-support", type=int, default=DEFAULT_STABLE_MIN_SUPPORT)
    p.add_argument("--symmetry-th", type=float, default=DEFAULT_SYMMETRY_TH)
    p.add_argument("--max-mirror-source-strokes", type=int, default=4)  # retained for compatibility; no synthetic generation in this version
    p.add_argument("--mirror-pair-th", type=float, default=0.10)
    p.add_argument("--max-json-mb", type=int, default=DEFAULT_MAX_JSON_MB)
    p.add_argument("--workers", default="auto")

    # Speed controls.
    # --skip-composition 可以先只构建 morpheme_nodes/symmetry/glyph_roots，跳过最慢的组合枚举。
    # --max-composition-raw-instances 用于限制组合枚举输入规模；0 表示不限制。
    p.add_argument("--skip-composition", action="store_true")
    p.add_argument("--max-composition-raw-instances", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    print("=" * 100)
    print("[MorphemeTreeBuilder] start")
    print(f"SCRIPT_DIR={SCRIPT_DIR}")
    print(f"CHAR_GLYPH_DIR={CHAR_GLYPH_DIR}")
    print(f"DATASET_ANALYSE_DIR={DATASET_ANALYSE_DIR}")
    print(f"good={args.good_dir} exists={os.path.exists(args.good_dir)}")
    print(f"cleaned={args.cleaned_dir} exists={os.path.exists(args.cleaned_dir)}")
    print(f"annotations={args.annotations_dir} exists={os.path.exists(args.annotations_dir)}")
    print(f"out={args.out_dir}")
    print("=" * 100)
    build_pipeline(args)


if __name__ == "__main__":
    main()
