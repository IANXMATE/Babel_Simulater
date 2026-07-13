# -*- coding: utf-8 -*-
r'''
morpheme_independent_rule_miner_v4_strict_contextual.py

放置位置建议：
    dataset_analyse_p0/Char_Glyph_v0/Morpheme_Demo/morpheme_independent_rule_miner_v4_strict_contextual.py

运行：
    cd dataset_analyse_p0/Char_Glyph_v0/Morpheme_Demo
    python morpheme_independent_rule_miner_v4_strict_contextual.py

目标：
    读取当前 PCG good / bad / cleaned 样本，以及人工 annotations_topo 样本；
    同时读取 Morpheme/output_tree 与 Morpheme/new_rule_cache 中的旧规则/派生规则信息；
    打印 Good/Bad/Manual 的拓扑、几何、motif、规则候选、旧规则正交性诊断。

设计原则：
    1. 新规则发现时允许读取旧规则，但只用于“去重/正交性检测”，不把旧 rule_id 当特征输入。
    2. Bad 样本不是“没有结构”，而是“整体质量低/结构组合失败”；所以报告会打印：
       - Good-enriched motifs
       - Bad-enriched motifs
       - Shared motifs: Good/Bad 都高频，说明它本身不是充分规则，需要上下文条件。
    3. 不做最终训练，只做审计日志，便于你把输出贴回来继续设计模型。

输出：
    - 控制台日志
    - Morpheme_Demo/rule_miner_v4_strict_contextual_outputs/independent_rule_miner_v4_strict_contextual_report.txt
    - Morpheme_Demo/rule_miner_v4_strict_contextual_outputs/independent_rule_miner_v4_strict_contextual_summary.json

依赖：
    仅使用 Python 标准库。可选使用 numpy；没有 numpy 时仍能运行，但几何统计会更弱。
'''

import os
import re
import sys
import json
import math
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean, median

try:
    import numpy as np
except Exception:
    np = None

# =============================================================================
# 0. Path resolution
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
# Expected: dataset_analyse_p0/Char_Glyph_v0/Morpheme_Demo
if SCRIPT_DIR.name.lower() == 'morpheme_demo':
    CHAR_GLYPH_DIR = SCRIPT_DIR.parent
else:
    # 允许你临时从其它目录运行；向上找 Char_Glyph_v0
    cur = SCRIPT_DIR
    found = None
    for p in [cur] + list(cur.parents):
        if p.name == 'Char_Glyph_v0':
            found = p
            break
        if (p / 'Char_Glyph_v0').is_dir():
            found = p / 'Char_Glyph_v0'
            break
    CHAR_GLYPH_DIR = found if found else SCRIPT_DIR.parent

DATASET_ANALYSE_DIR = CHAR_GLYPH_DIR.parent
ANNOTATION_TOOL_DIR = CHAR_GLYPH_DIR / 'annotation_tool'
PCG_POOL_DIR = ANNOTATION_TOOL_DIR / 'pcg_filebacked_stage2_schema'
PCG_GOOD_DIR = PCG_POOL_DIR / 'good'
PCG_BAD_DIR = PCG_POOL_DIR / 'bad'
PCG_CLEANED_DIR = PCG_POOL_DIR / 'cleaned'
MANUAL_ANNOTATIONS_DIR = DATASET_ANALYSE_DIR / 'AI_VECTOR_ROUTER_With_topo' / 'annotations_topo'
MORPHEME_DIR = CHAR_GLYPH_DIR / 'Morpheme'
MORPHEME_OUTPUT_TREE_DIR = MORPHEME_DIR / 'output_tree'
MORPHEME_NEW_RULE_CACHE_DIR = MORPHEME_DIR / 'new_rule_cache'
OUTPUT_DIR = SCRIPT_DIR / 'rule_miner_v4_strict_contextual_outputs'

# =============================================================================
# 1. Generic JSON loading
# =============================================================================

def safe_read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
        except Exception:
            return None
    except Exception:
        return None


def iter_json_files(root, max_files=None):
    root = Path(root)
    if not root.exists():
        return
    count = 0
    for p in sorted(root.rglob('*.json')):
        if max_files is not None and count >= max_files:
            break
        count += 1
        yield p


def looks_like_glyph_record(d):
    if not isinstance(d, dict):
        return False
    keys = set(d.keys())
    strong = {'strokes', 'solved_nodes', 'nodes', 'topology_events', 'cycles', 'glyph_info'}
    if keys & strong:
        if any(k in d for k in ['strokes', 'solved_nodes', 'nodes']) or any(k in d for k in ['topology_events', 'cycles']):
            return True
    if 'glyph_info' in d and isinstance(d.get('glyph_info'), dict):
        return True
    return False


def extract_records_from_json(obj):
    """Robustly extract glyph-like records from many possible pool formats."""
    records = []
    if obj is None:
        return records
    if isinstance(obj, list):
        for x in obj:
            records.extend(extract_records_from_json(x))
        return records
    if not isinstance(obj, dict):
        return records
    if looks_like_glyph_record(obj):
        return [obj]
    # common container keys
    for key in [
        'records', 'items', 'data', 'glyphs', 'samples', 'bundles', 'chars',
        'good', 'bad', 'cleaned', 'annotations', 'entries', 'pool', 'values'
    ]:
        v = obj.get(key)
        if isinstance(v, (list, dict)):
            records.extend(extract_records_from_json(v))
    # Some split files store id -> bundle dicts.
    if not records:
        dict_values = list(obj.values())
        if dict_values and all(isinstance(v, dict) for v in dict_values[: min(10, len(dict_values))]):
            hit = 0
            for v in dict_values:
                sub = extract_records_from_json(v)
                if sub:
                    hit += len(sub)
                    records.extend(sub)
            if hit:
                return records
    return records


def load_records_from_dir(root, label, max_files=None, max_records=None):
    out = []
    file_count = 0
    bad_json = 0
    for path in iter_json_files(root, max_files=max_files):
        file_count += 1
        obj = safe_read_json(path)
        if obj is None:
            bad_json += 1
            continue
        records = extract_records_from_json(obj)
        for i, rec in enumerate(records):
            if max_records is not None and len(out) >= max_records:
                return out, {'files': file_count, 'bad_json': bad_json, 'truncated': True}
            # shallow copy; add audit metadata without mutating original nested content too much
            if isinstance(rec, dict):
                r = dict(rec)
                r['_audit_label'] = label
                r['_audit_source_file'] = str(path)
                r['_audit_record_index'] = i
                out.append(r)
    return out, {'files': file_count, 'bad_json': bad_json, 'truncated': False}

# =============================================================================
# 2. Stroke / topology extraction
# =============================================================================

def get_record_id(rec, fallback='unknown'):
    for k in ['generated_glyph_id', 'source_candidate_id', 'candidate_id', 'glyph_candidate_id', 'sample_id', 'grammar_sample_id', 'id', 'char_id']:
        if rec.get(k):
            return str(rec.get(k))
    gi = rec.get('glyph_info') if isinstance(rec.get('glyph_info'), dict) else {}
    for k in ['char_id', 'unicode', 'char', 'glyph_id', 'font_name']:
        if gi.get(k):
            return str(gi.get(k))
    return fallback


def get_stroke_list(rec):
    for k in ['strokes', 'solved_nodes', 'nodes', 'solved_segments', 'edges']:
        v = rec.get(k)
        if isinstance(v, list):
            return v
    return []


def parse_point_pair_list(v):
    if isinstance(v, list) and len(v) == 4:
        ok = True
        pts = []
        for p in v:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    pts.append([float(p[0]), float(p[1])])
                except Exception:
                    ok = False
                    break
            else:
                ok = False
                break
        if ok:
            return pts
    return None


def get_bezier_from_stroke(st):
    if not isinstance(st, dict):
        return None
    for k in ['mother_bezier', 'bezier', 'curve', 'solved_bezier', 'control_points', 'path']:
        pts = parse_point_pair_list(st.get(k))
        if pts is not None:
            return pts
    return None


def get_stroke_id(st, idx):
    if isinstance(st, dict):
        for k in ['id', 'stroke_id', 'edge_id', 'node_id', 'bezier_id']:
            if k in st:
                try:
                    return int(st.get(k))
                except Exception:
                    return str(st.get(k))
    return idx


def cubic_sample(P, n=32):
    if P is None:
        return []
    if np is not None:
        pts = np.asarray(P, dtype=float)
        ts = np.linspace(0.0, 1.0, n)[:, None]
        mt = 1.0 - ts
        out = (mt ** 3) * pts[0] + 3 * (mt ** 2) * ts * pts[1] + 3 * mt * (ts ** 2) * pts[2] + (ts ** 3) * pts[3]
        return out.tolist()
    out = []
    for i in range(n):
        t = i / max(1, n - 1)
        mt = 1.0 - t
        x = (mt ** 3) * P[0][0] + 3 * (mt ** 2) * t * P[1][0] + 3 * mt * (t ** 2) * P[2][0] + (t ** 3) * P[3][0]
        y = (mt ** 3) * P[0][1] + 3 * (mt ** 2) * t * P[1][1] + 3 * mt * (t ** 2) * P[2][1] + (t ** 3) * P[3][1]
        out.append([x, y])
    return out


def dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def polyline_length(points):
    if not points or len(points) < 2:
        return 0.0
    return sum(dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def angle_deg_from_p0p3(P):
    if not P:
        return None
    dx = float(P[3][0]) - float(P[0][0])
    dy = float(P[3][1]) - float(P[0][1])
    if abs(dx) + abs(dy) < 1e-9:
        return None
    a = math.degrees(math.atan2(dy, dx))
    a = a % 180.0
    return a


def axis_class(angle_deg, tol=10.0):
    if angle_deg is None:
        return 'unknown'
    hdist = min(abs(angle_deg - 0.0), abs(angle_deg - 180.0))
    vdist = abs(angle_deg - 90.0)
    if hdist <= tol:
        return 'horizontal'
    if vdist <= tol:
        return 'vertical'
    return 'diagonal'


def curvature_score(P):
    if P is None:
        return 0.0
    p0, p1, p2, p3 = P
    base = dist(p0, p3)
    if base < 1e-6:
        return 0.0
    def point_line_distance(p, a, b):
        ax, ay = a; bx, by = b; px, py = p
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        area = abs(vx * wy - vy * wx)
        return area / max(1e-6, math.hypot(vx, vy))
    return max(point_line_distance(p1, p0, p3), point_line_distance(p2, p0, p3)) / base


def normalize_event_type(ev):
    if not isinstance(ev, dict):
        return 'unknown'
    raw = ''
    for k in ['type', 'event_type', 'relation', 'relation_type', 'topology_type', 'kind']:
        if ev.get(k) is not None:
            raw += ' ' + str(ev.get(k))
    raw_u = raw.upper()
    if 'E2E' in raw_u or 'END_TO_END' in raw_u or 'END-TO-END' in raw_u:
        return 'E2E'
    # Put X before T because some strings might include text with t.
    if re.search(r'(^|[^A-Z])X([^A-Z]|$)', raw_u) or 'CROSS' in raw_u or 'INTERSECT' in raw_u:
        return 'X'
    if re.search(r'(^|[^A-Z])T([^A-Z]|$)', raw_u) or 'T_JUNCTION' in raw_u or 'TJUNCTION' in raw_u:
        return 'T'
    return 'unknown'


def get_topology_events(rec):
    v = rec.get('topology_events')
    if isinstance(v, list):
        return v
    # Some schemas store topology under nested key.
    topo = rec.get('topology') if isinstance(rec.get('topology'), dict) else None
    if topo and isinstance(topo.get('events'), list):
        return topo.get('events')
    return []


def get_cycles(rec):
    v = rec.get('cycles')
    if isinstance(v, list):
        return v
    topo = rec.get('topology') if isinstance(rec.get('topology'), dict) else None
    if topo and isinstance(topo.get('cycles'), list):
        return topo.get('cycles')
    return []


def event_stroke_ids(ev):
    """Extract stroke ids conservatively from known id-like keys."""
    ids = []
    if not isinstance(ev, dict):
        return ids
    key_patterns = ['stroke', 'edge', 'line', 'host', 'guest', 'source', 'target']
    for k, v in ev.items():
        kl = str(k).lower()
        if not any(p in kl for p in key_patterns):
            continue
        if 'point' in kl or 'pos' in kl or 'coord' in kl or 't_' in kl or kl.endswith('_t'):
            continue
        if isinstance(v, int):
            ids.append(v)
        elif isinstance(v, str) and re.fullmatch(r'-?\d+', v.strip()):
            ids.append(int(v.strip()))
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, int):
                    ids.append(x)
                elif isinstance(x, str) and re.fullmatch(r'-?\d+', x.strip()):
                    ids.append(int(x.strip()))
    # common explicit alternatives
    for pair in [('stroke_a', 'stroke_b'), ('edge_a', 'edge_b'), ('a', 'b')]:
        if pair[0] in ev and pair[1] in ev:
            try:
                ids.extend([int(ev[pair[0]]), int(ev[pair[1]])])
            except Exception:
                pass
    # de-dup preserve order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x); out.append(x)
    return out[:4]


def connected_components_from_events(stroke_ids, events):
    parent = {sid: sid for sid in stroke_ids}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for ev in events:
        ids = event_stroke_ids(ev)
        if len(ids) >= 2:
            union(ids[0], ids[1])
    if not parent:
        return 0, Counter()
    comps = Counter(find(x) for x in parent)
    return len(comps), comps


def bucket_value(v, bins):
    for name, lo, hi in bins:
        if v >= lo and v < hi:
            return name
    return bins[-1][0]


def extract_features(rec):
    strokes = get_stroke_list(rec)
    stroke_ids = [get_stroke_id(st, i) for i, st in enumerate(strokes)]
    events = get_topology_events(rec)
    cycles = get_cycles(rec)
    rel_counts = Counter(normalize_event_type(ev) for ev in events)

    bezier_list = [get_bezier_from_stroke(st) for st in strokes]
    valid_beziers = [P for P in bezier_list if P is not None]
    lengths = []
    curvatures = []
    angle_classes = Counter()
    angles = []
    all_points = []
    for P in valid_beziers:
        pts = cubic_sample(P, n=24)
        all_points.extend(pts)
        lengths.append(polyline_length(pts))
        curvatures.append(curvature_score(P))
        a = angle_deg_from_p0p3(P)
        if a is not None:
            angles.append(a)
        angle_classes[axis_class(a)] += 1

    stroke_count = len(strokes)
    event_count = len(events)
    cycle_count = len(cycles)
    cc_count, comp_sizes = connected_components_from_events(stroke_ids, events)
    if cc_count == 0 and stroke_count > 0:
        cc_count = stroke_count
        comp_sizes = Counter({sid: 1 for sid in stroke_ids})

    # degree approximation from topology events
    deg = Counter()
    for ev in events:
        ids = event_stroke_ids(ev)
        if len(ids) >= 2:
            deg[ids[0]] += 1
            deg[ids[1]] += 1
    for sid in stroke_ids:
        deg.setdefault(sid, 0)
    max_degree = max(deg.values()) if deg else 0
    leaf_count = sum(1 for v in deg.values() if v <= 1)
    branch_count = sum(1 for v in deg.values() if v >= 3)

    # bbox and symmetry approximations
    if all_points:
        xs = [p[0] for p in all_points]; ys = [p[1] for p in all_points]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        bbox_w, bbox_h = maxx - minx, maxy - miny
        aspect = bbox_w / max(1e-6, bbox_h)
        bbox_area = bbox_w * bbox_h
    else:
        minx = maxx = miny = maxy = bbox_w = bbox_h = aspect = bbox_area = 0.0

    n_valid = max(1, len(valid_beziers))
    axis_total = angle_classes['horizontal'] + angle_classes['vertical']
    horizontal_ratio = angle_classes['horizontal'] / n_valid
    vertical_ratio = angle_classes['vertical'] / n_valid
    axis_ratio = axis_total / n_valid
    diagonal_ratio = angle_classes['diagonal'] / n_valid
    curve_ratio = sum(1 for c in curvatures if c >= 0.08) / max(1, len(curvatures))

    total_rel_known = rel_counts['E2E'] + rel_counts['T'] + rel_counts['X']
    rel_denom = max(1, total_rel_known)
    e2e_ratio = rel_counts['E2E'] / rel_denom
    t_ratio = rel_counts['T'] / rel_denom
    x_ratio = rel_counts['X'] / rel_denom

    # High-level structural proxy features, intentionally not using old rule ids.
    port_count_proxy = rel_counts['T'] + 2 * rel_counts['E2E'] + branch_count
    spine_score = min(1.0, vertical_ratio * 0.65 + (max_degree / max(1, stroke_count)) * 0.35)
    enclosure_score = min(1.0, cycle_count / max(1, stroke_count / 3.0))
    interlock_score = min(1.0, (rel_counts['X'] + rel_counts['T']) / max(1, stroke_count))
    orbit_score = min(1.0, (cycle_count * 0.45 + branch_count * 0.25 + rel_counts['X'] * 0.15) / max(1, stroke_count / 2.0))
    fragmentation_risk = min(1.0, (rel_counts['X'] * 0.5 + rel_counts['T'] * 0.25) / max(1, stroke_count))

    features = {
        'id': get_record_id(rec),
        'label': rec.get('_audit_label', 'unknown'),
        'source_file': rec.get('_audit_source_file', ''),
        'stroke_count': float(stroke_count),
        'valid_bezier_count': float(len(valid_beziers)),
        'topology_event_count': float(event_count),
        'cycle_count': float(cycle_count),
        'connected_components': float(cc_count),
        'component_max_size': float(max(comp_sizes.values()) if comp_sizes else 0),
        'E2E_count': float(rel_counts['E2E']),
        'T_count': float(rel_counts['T']),
        'X_count': float(rel_counts['X']),
        'unknown_event_count': float(rel_counts['unknown']),
        'E2E_ratio': float(e2e_ratio),
        'T_ratio': float(t_ratio),
        'X_ratio': float(x_ratio),
        'max_degree': float(max_degree),
        'leaf_count': float(leaf_count),
        'leaf_ratio': float(leaf_count / max(1, stroke_count)),
        'branch_count': float(branch_count),
        'branch_ratio': float(branch_count / max(1, stroke_count)),
        'horizontal_ratio': float(horizontal_ratio),
        'vertical_ratio': float(vertical_ratio),
        'axis_ratio': float(axis_ratio),
        'diagonal_ratio': float(diagonal_ratio),
        'curve_ratio': float(curve_ratio),
        'mean_length': float(mean(lengths) if lengths else 0.0),
        'median_length': float(median(lengths) if lengths else 0.0),
        'min_length': float(min(lengths) if lengths else 0.0),
        'max_length': float(max(lengths) if lengths else 0.0),
        'mean_curvature': float(mean(curvatures) if curvatures else 0.0),
        'bbox_w': float(bbox_w),
        'bbox_h': float(bbox_h),
        'bbox_area': float(bbox_area),
        'aspect': float(aspect),
        'port_count_proxy': float(port_count_proxy),
        'spine_score': float(spine_score),
        'enclosure_score': float(enclosure_score),
        'interlock_score': float(interlock_score),
        'orbit_score': float(orbit_score),
        'fragmentation_risk': float(fragmentation_risk),
    }
    features['_rel_counts'] = dict(rel_counts)
    features['_degree_sequence'] = sorted([int(v) for v in deg.values()], reverse=True)
    features['_motifs'] = build_motif_signatures(features)
    return features


def bucket_num(v, cuts):
    # cuts: list of thresholds, returns b0/b1/...
    for i, c in enumerate(cuts):
        if v <= c:
            return f'b{i}'
    return f'b{len(cuts)}'


def build_motif_signatures(f):
    motifs = []
    sc = int(f.get('stroke_count', 0))
    cyc = int(f.get('cycle_count', 0))
    e2e = int(f.get('E2E_count', 0))
    t = int(f.get('T_count', 0))
    x = int(f.get('X_count', 0))
    cc = int(f.get('connected_components', 0))
    maxd = int(f.get('max_degree', 0))
    branch = int(f.get('branch_count', 0))
    axis_b = bucket_num(f.get('axis_ratio', 0.0), [0.25, 0.50, 0.75])
    diag_b = bucket_num(f.get('diagonal_ratio', 0.0), [0.25, 0.50, 0.75])
    curve_b = bucket_num(f.get('curve_ratio', 0.0), [0.20, 0.45, 0.70])
    port_b = bucket_num(f.get('port_count_proxy', 0.0), [2, 5, 8, 12])
    spine_b = bucket_num(f.get('spine_score', 0.0), [0.25, 0.45, 0.65])
    encl_b = bucket_num(f.get('enclosure_score', 0.0), [0.15, 0.40, 0.70])
    inter_b = bucket_num(f.get('interlock_score', 0.0), [0.15, 0.35, 0.60])

    motifs.append(f'relhist:E{min(e2e,4)}_T{min(t,4)}_X{min(x,4)}_C{min(cyc,4)}')
    motifs.append(f'capacity:S{bucket_num(sc,[4,7,10,14])}_CC{min(cc,3)}_C{min(cyc,4)}')
    motifs.append(f'role:maxD{min(maxd,5)}_B{min(branch,4)}_port{port_b}')
    motifs.append(f'orient:axis{axis_b}_diag{diag_b}_curve{curve_b}')
    motifs.append(f'field:spine{spine_b}_encl{encl_b}_inter{inter_b}')

    # Human-readable named proxy motifs. These are not old rule ids; they are raw-feature composites.
    if cyc >= 1 and f.get('spine_score', 0) >= 0.45 and f.get('port_count_proxy', 0) >= 4:
        motifs.append('proxy:ring_core_spine_ports')
    if cyc >= 2:
        motifs.append('proxy:nested_or_multi_cycle')
    if f.get('vertical_ratio', 0) >= 0.25 and branch >= 1:
        motifs.append('proxy:vertical_spine_branching')
    if f.get('axis_ratio', 0) >= 0.55 and f.get('curve_ratio', 0) >= 0.25:
        motifs.append('proxy:axis_skeleton_with_curve_residual')
    if f.get('interlock_score', 0) >= 0.45 and f.get('port_count_proxy', 0) >= 4:
        motifs.append('proxy:interlocked_ports')
    if f.get('orbit_score', 0) >= 0.45:
        motifs.append('proxy:orbit_or_seal_field')
    if f.get('fragmentation_risk', 0) >= 0.60:
        motifs.append('proxy:high_fragmentation_risk')
    if cc > 1:
        motifs.append('proxy:multi_component_or_floating')
    return motifs

# =============================================================================
# 3. Old-rule loading / orthogonality vocabulary
# =============================================================================

def collect_rule_dicts(obj, path_hint='', out=None):
    if out is None:
        out = []
    if isinstance(obj, dict):
        if any(k in obj for k in ['rule_id', 'rules', 'terms', 'conditions', 'pattern_rules']):
            # If this object itself looks like a rule.
            if obj.get('rule_id') or obj.get('id') or obj.get('name'):
                out.append((obj, path_hint))
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                collect_rule_dicts(v, path_hint, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_rule_dicts(v, path_hint, out)
    return out


def load_old_rules(max_files=None):
    roots = [MORPHEME_OUTPUT_TREE_DIR, MORPHEME_NEW_RULE_CACHE_DIR]
    rules = []
    file_count = 0
    for root in roots:
        if not root.exists():
            continue
        for p in iter_json_files(root, max_files=max_files):
            file_count += 1
            obj = safe_read_json(p)
            if obj is None:
                continue
            rules.extend(collect_rule_dicts(obj, str(p)))
    parsed = []
    seen = set()
    for d, src in rules:
        rid = str(d.get('rule_id') or d.get('id') or d.get('name') or f'rule_{len(parsed)}')
        key = (rid, src)
        if key in seen:
            continue
        seen.add(key)
        features = set()
        tokens = set()
        def walk(x):
            if isinstance(x, dict):
                if 'feature' in x:
                    features.add(str(x.get('feature')))
                for kk, vv in x.items():
                    if kk in ['rule_id', 'id', 'name', 'description'] and isinstance(vv, str):
                        for tok in re.split(r'[^A-Za-z0-9_]+', vv):
                            if tok:
                                tokens.add(tok)
                    walk(vv)
            elif isinstance(x, list):
                for vv in x:
                    walk(vv)
            elif isinstance(x, str):
                # collect feature-looking strings conservatively
                for tok in re.split(r'[^A-Za-z0-9_]+', x):
                    if tok and any(s in tok.lower() for s in ['count','ratio','score','degree','cycle','branch','density','e2e','axis','grid','cross','fork','ladder']):
                        tokens.add(tok)
        walk(d)
        parsed.append({
            'rule_id': rid,
            'source_file': src,
            'features': sorted(features),
            'tokens': sorted(tokens),
            'raw_key_count': len(d.keys()) if isinstance(d, dict) else 0,
        })
    return parsed, file_count

# =============================================================================
# 4. Aggregation, motif stats, candidate rule mining
# =============================================================================

NUMERIC_FEATURES = [
    'stroke_count', 'topology_event_count', 'cycle_count', 'connected_components',
    'E2E_count', 'T_count', 'X_count', 'E2E_ratio', 'T_ratio', 'X_ratio',
    'max_degree', 'leaf_ratio', 'branch_ratio', 'horizontal_ratio', 'vertical_ratio',
    'axis_ratio', 'diagonal_ratio', 'curve_ratio', 'mean_length', 'min_length',
    'mean_curvature', 'bbox_area', 'aspect', 'port_count_proxy', 'spine_score',
    'enclosure_score', 'interlock_score', 'orbit_score', 'fragmentation_risk'
]


def summarize_numeric(rows, label):
    vals_by_f = {}
    for feat in NUMERIC_FEATURES:
        vals = [float(r.get(feat, 0.0)) for r in rows if r.get(feat) is not None and math.isfinite(float(r.get(feat, 0.0)))]
        if vals:
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            def q(p):
                if n == 1:
                    return vals_sorted[0]
                idx = min(n - 1, max(0, int(round(p * (n - 1)))))
                return vals_sorted[idx]
            vals_by_f[feat] = {
                'n': n,
                'mean': sum(vals_sorted) / n,
                'median': q(0.50),
                'p25': q(0.25),
                'p75': q(0.75),
                'min': vals_sorted[0],
                'max': vals_sorted[-1],
            }
    return vals_by_f


def fmt_float(x, nd=4):
    try:
        return f'{float(x):.{nd}f}'
    except Exception:
        return str(x)


def motif_support(rows):
    c = Counter()
    for r in rows:
        for m in r.get('_motifs', []):
            c[m] += 1
    n = max(1, len(rows))
    return {m: {'count': k, 'support': k / n} for m, k in c.items()}


def compare_motifs(good_rows, bad_rows, manual_rows, top_k=30):
    gs = motif_support(good_rows)
    bs = motif_support(bad_rows)
    ms = motif_support(manual_rows)
    all_m = set(gs) | set(bs) | set(ms)
    rows = []
    for m in all_m:
        g = gs.get(m, {'count':0,'support':0.0})
        b = bs.get(m, {'count':0,'support':0.0})
        ma = ms.get(m, {'count':0,'support':0.0})
        rows.append({
            'motif': m,
            'good_count': g['count'], 'good_support': g['support'],
            'bad_count': b['count'], 'bad_support': b['support'],
            'manual_count': ma['count'], 'manual_support': ma['support'],
            'good_minus_bad': g['support'] - b['support'],
            'bad_minus_good': b['support'] - g['support'],
            'shared_strength': min(g['support'], b['support']),
            'manual_alignment': ma['support'],
        })
    good_enriched = sorted(rows, key=lambda x: (x['good_minus_bad'], x['good_support']), reverse=True)[:top_k]
    bad_enriched = sorted(rows, key=lambda x: (x['bad_minus_good'], x['bad_support']), reverse=True)[:top_k]
    shared = sorted(rows, key=lambda x: (x['shared_strength'], x['good_support'] + x['bad_support']), reverse=True)[:top_k]
    manual_aligned = sorted(rows, key=lambda x: (x['manual_alignment'], x['good_support']), reverse=True)[:top_k]
    return {
        'good_enriched': good_enriched,
        'bad_enriched': bad_enriched,
        'shared': shared,
        'manual_aligned': manual_aligned,
    }

class Predicate:
    __slots__ = ('name','feature','op','value','func')
    def __init__(self, name, feature, op, value, func):
        self.name = name
        self.feature = feature
        self.op = op
        self.value = value
        self.func = func
    def __call__(self, row):
        try:
            return bool(self.func(row))
        except Exception:
            return False


def build_atomic_predicates(rows):
    preds = []
    for feat in NUMERIC_FEATURES:
        vals = sorted([float(r.get(feat, 0.0)) for r in rows if r.get(feat) is not None and math.isfinite(float(r.get(feat, 0.0)))])
        if len(vals) < 8:
            continue
        # Avoid constant features.
        if abs(vals[-1] - vals[0]) < 1e-9:
            continue
        qs = []
        for p in [0.25, 0.50, 0.75]:
            idx = min(len(vals)-1, max(0, int(round(p * (len(vals)-1)))))
            qs.append(vals[idx])
        # de-dup thresholds
        for qv in sorted(set(round(q, 6) for q in qs)):
            preds.append(Predicate(f'{feat}>={qv:g}', feat, '>=', qv, lambda r, f=feat, q=qv: float(r.get(f, 0.0)) >= q))
            preds.append(Predicate(f'{feat}<={qv:g}', feat, '<=', qv, lambda r, f=feat, q=qv: float(r.get(f, 0.0)) <= q))
        if qs[0] < qs[2]:
            lo, hi = qs[0], qs[2]
            preds.append(Predicate(f'{feat}_between[{lo:g},{hi:g}]', feat, 'between', [lo, hi], lambda r, f=feat, a=lo, b=hi: a <= float(r.get(f, 0.0)) <= b))
    # Add motif predicates.
    all_motifs = Counter()
    for r in rows:
        all_motifs.update(r.get('_motifs', []))
    for m, cnt in all_motifs.items():
        if cnt >= max(3, int(0.02 * len(rows))):
            preds.append(Predicate(f'motif:{m}', f'motif:{m}', 'has', True, lambda r, mm=m: mm in set(r.get('_motifs', []))))
    return preds


def eval_predicate_set(preds, good_rows, bad_rows, manual_rows=None):
    def hit(row):
        return all(p(row) for p in preds)
    gh = [r for r in good_rows if hit(r)]
    bh = [r for r in bad_rows if hit(r)]
    mh = [r for r in (manual_rows or []) if hit(r)]
    gsup = len(gh) / max(1, len(good_rows))
    bsup = len(bh) / max(1, len(bad_rows))
    msup = len(mh) / max(1, len(manual_rows or [])) if manual_rows is not None else 0.0
    precision = len(gh) / max(1, len(gh) + len(bh))
    lift = (gsup + 1e-9) / (bsup + 1e-9)
    return {
        'good_hits': len(gh), 'bad_hits': len(bh), 'manual_hits': len(mh),
        'good_support': gsup, 'bad_support': bsup, 'manual_support': msup,
        'precision_vs_bad': precision,
        'lift_good_vs_bad': lift,
    }


def old_rule_orthogonality(preds, old_rules):
    cand_features = set(p.feature for p in preds)
    # motif predicates are allowed, compare their proxy tokens too.
    cand_tokens = set()
    for p in preds:
        for tok in re.split(r'[^A-Za-z0-9_]+', p.name):
            if tok:
                cand_tokens.add(tok)
    max_j = 0.0
    nearest = None
    for rule in old_rules:
        old_set = set(rule.get('features', [])) | set(rule.get('tokens', []))
        if not old_set:
            continue
        inter = len((cand_features | cand_tokens) & old_set)
        union = len((cand_features | cand_tokens) | old_set)
        j = inter / max(1, union)
        if j > max_j:
            max_j = j
            nearest = rule.get('rule_id')
    return {
        'orthogonality': 1.0 - max_j,
        'nearest_old_rule': nearest,
        'max_old_jaccard': max_j,
        'candidate_features': sorted(cand_features),
    }


def diversity_bonus(preds):
    feats = [p.feature.split(':')[0] for p in preds]
    cats = set()
    for f in feats:
        if 'motif' in f:
            cats.add('motif')
        elif any(x in f for x in ['cycle','E2E','T_','X_','event','degree','branch','leaf','component']):
            cats.add('topology')
        elif any(x in f for x in ['axis','diagonal','horizontal','vertical','curve','length','curvature','bbox','aspect']):
            cats.add('geometry')
        elif any(x in f for x in ['spine','enclosure','interlock','orbit','port','fragmentation']):
            cats.add('semantic_proxy')
        else:
            cats.add('other')
    return min(1.0, len(cats) / 3.0)


def generality_score(gsup):
    # Prefer rules that cover a meaningful region, not a one-off and not almost everything.
    # Peak around 0.25~0.45.
    if gsup <= 0:
        return 0.0
    if gsup < 0.05:
        return gsup / 0.05 * 0.25
    if 0.05 <= gsup <= 0.45:
        return 0.65 + 0.35 * min(1.0, (gsup - 0.05) / 0.40)
    if gsup <= 0.75:
        return 1.0 - 0.50 * ((gsup - 0.45) / 0.30)
    return 0.25


def mine_candidate_rules(good_rows, bad_rows, manual_rows, old_rules, min_good_hits=5, top_k=40):
    all_rows = good_rows + bad_rows + manual_rows
    atoms = build_atomic_predicates(all_rows)
    # Score atoms first; keep moderately useful atoms, then compose.
    atom_scored = []
    for p in atoms:
        ev = eval_predicate_set([p], good_rows, bad_rows, manual_rows)
        if ev['good_hits'] >= max(2, min_good_hits // 2):
            discr = max(0.0, ev['good_support'] - ev['bad_support'])
            atom_scored.append((discr + 0.15 * ev['precision_vs_bad'] + 0.05 * min(5.0, ev['lift_good_vs_bad']), p, ev))
    atom_scored.sort(key=lambda x: x[0], reverse=True)
    seed_atoms = [x[1] for x in atom_scored[:80]]

    candidates = []
    # Use pairs and triples. Single-atom rules are printed only for debugging and marked as trivial.
    combos = []
    for k in [2, 3]:
        for combo in combinations(seed_atoms[:50 if k == 2 else 32], k):
            # Avoid redundant thresholds on exactly same feature too often.
            feat_counts = Counter(p.feature for p in combo)
            if any(v > 1 for v in feat_counts.values()):
                continue
            combos.append(combo)
    # Add motif + non-motif combos from top seeds.
    for combo in combos:
        ev = eval_predicate_set(combo, good_rows, bad_rows, manual_rows)
        if ev['good_hits'] < min_good_hits:
            continue
        ortho = old_rule_orthogonality(combo, old_rules)
        div = diversity_bonus(combo)
        discr = max(0.0, ev['good_support'] - ev['bad_support'])
        # Bad support is not forced to zero; if it exists, report it as context.
        # Strong rules can still have bad support if combined with missing quality constraints.
        gen = generality_score(ev['good_support'])
        precision = ev['precision_vs_bad']
        manual_align = ev.get('manual_support', 0.0)
        nontrivial = 1.0 if len(combo) >= 2 and div >= 0.66 else 0.55
        final = (
            0.32 * discr +
            0.20 * precision +
            0.18 * gen +
            0.18 * ortho['orthogonality'] +
            0.07 * div +
            0.05 * manual_align
        ) * nontrivial
        candidates.append({
            'rule_name': ' AND '.join(p.name for p in combo),
            'predicates': [{'name': p.name, 'feature': p.feature, 'op': p.op, 'value': p.value} for p in combo],
            **ev,
            **ortho,
            'diversity_bonus': div,
            'generality_score': gen,
            'score': final,
            'note': 'candidate independent rule; old rules used only for orthogonality/de-dup check',
        })
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_k], atom_scored[:top_k]



# =============================================================================
# 4.5 Bad subtype decomposition / contextual role helpers
# =============================================================================

BAD_SUBTYPE_DEFS = [
    (
        'bad_floating_or_multicomponent',
        lambda r: r.get('connected_components', 0) >= 2 or ('proxy:multi_component_or_floating' in set(r.get('_motifs', []))),
        '多连通块 / 漂浮部件：通常表示符文线没有通过 port/interlock 接回主结构。'
    ),
    (
        'bad_trivial_e2e_chain',
        lambda r: r.get('cycle_count', 0) <= 0 and r.get('X_count', 0) <= 0 and r.get('T_count', 0) <= 0 and r.get('E2E_ratio', 0) >= 0.90,
        '纯 E2E 链式结构：有拓扑，但构成过浅，通常缺少 seal/orbit/interlock 上下文。'
    ),
    (
        'bad_short_or_fragmented',
        lambda r: (0 < r.get('min_length', 0) < 64) or r.get('fragmentation_risk', 0) >= 0.60,
        '短线/碎片风险：适合作为 quality gate，不适合作为正向拓扑规则。'
    ),
    (
        'bad_low_context_cycle',
        lambda r: r.get('cycle_count', 0) >= 1 and r.get('spine_score', 0) < 0.25 and r.get('interlock_score', 0) < 0.18 and r.get('orbit_score', 0) < 0.30,
        '低上下文 cycle：有环，但没有 spine/port/interlock 支撑，像孤立框或无意义闭合。'
    ),
    (
        'bad_overdense_cross_fragment',
        lambda r: r.get('X_count', 0) >= 2 and r.get('fragmentation_risk', 0) >= 0.35,
        '高交叉且碎片风险高：说明 interlock/weave 可能不是从安全端口发生。'
    ),
    (
        'bad_unbalanced_multicomponent',
        lambda r: r.get('connected_components', 0) >= 2 and r.get('component_max_size', 0) <= max(2, r.get('stroke_count', 0) * 0.55),
        '多部件不平衡：可能需要提高 port coupling 或使用 component-level 组合约束。'
    ),
    (
        'bad_no_structural_field',
        lambda r: r.get('spine_score', 0) < 0.20 and r.get('enclosure_score', 0) < 0.20 and r.get('interlock_score', 0) < 0.15 and r.get('orbit_score', 0) < 0.25,
        '缺少结构场：不是坏在某个 motif，而是没有形成可解释构造中心。'
    ),
]


def bad_subtype_rows(good_rows, bad_rows):
    rows = []
    for name, fn, desc in BAD_SUBTYPE_DEFS:
        bh = [r for r in bad_rows if fn(r)]
        gh = [r for r in good_rows if fn(r)]
        rows.append({
            'bad_subtype': name,
            'bad_hits': len(bh),
            'bad_support': len(bh) / max(1, len(bad_rows)),
            'good_hits': len(gh),
            'good_support': len(gh) / max(1, len(good_rows)),
            'bad_minus_good': len(bh) / max(1, len(bad_rows)) - len(gh) / max(1, len(good_rows)),
            'description': desc,
        })
    rows.sort(key=lambda x: (x['bad_minus_good'], x['bad_support']), reverse=True)
    return rows


def predicate_is_quality_gate(p):
    n = p.name.lower()
    quality_keys = [
        'connected_components', 'min_length', 'mean_length', 'max_length',
        'fragmentation', 'stroke_count', 'bbox_area', 'aspect'
    ]
    return any(k.lower() in n for k in quality_keys)


def predicate_is_positive_ingredient(p):
    n = p.name.lower()
    if n.startswith('motif:'):
        return True
    ingredient_keys = ['spine', 'enclosure', 'interlock', 'orbit', 'cycle', 'x_count', 't_count', 'branch', 'port']
    return any(k in n for k in ingredient_keys)


# =============================================================================
# 4.6 V3 contextual miner: quality gates / ingredients / failure modes / rules
# =============================================================================

QUALITY_CONTEXT_FEATURES = {
    'connected_components', 'min_length', 'mean_length', 'fragmentation_risk',
    'stroke_count', 'bbox_area', 'aspect', 'E2E_ratio', 'leaf_ratio',
}

STRONG_TOPOLOGY_INGREDIENT_HINTS = [
    'proxy:orbit_or_seal_field',
    'proxy:nested_or_multi_cycle',
    'proxy:vertical_spine_branching',
    'proxy:interlocked_ports',
    'proxy:ring_core_spine_ports',
    'field:spineb',
    'enclb',
    'interb',
    'relhist:',
    'role:maxd',
    'orbit_score',
    'interlock_score',
    'enclosure_score',
    'spine_score',
    'port_count_proxy',
]

WEAK_COMPLEXITY_HINTS = [
    'x_count', 'x_ratio', 't_count', 't_ratio', 'cycle_count', 'branch_ratio',
    'max_degree', 'curve_ratio', 'mean_curvature'
]

BAD_MOTIF_HINTS = [
    'multi_component_or_floating',
    'spineb0_enclb0_interb0',
    'curveb0',
    'cc3',
    'relhist:e1_t0_x0',
    'relhist:e2_t0_x0',
]


def predicate_kind_v3(p):
    n = p.name.lower()
    f = p.feature.lower()
    if n.startswith('not_bad_'):
        return 'bad_avoid_context'
    if any(k in n for k in BAD_MOTIF_HINTS):
        return 'bad_failure_ingredient'
    if any(k in f for k in QUALITY_CONTEXT_FEATURES) or any(k in n for k in QUALITY_CONTEXT_FEATURES):
        return 'quality_context'
    if any(k in n for k in STRONG_TOPOLOGY_INGREDIENT_HINTS):
        return 'positive_ingredient'
    if any(k in n for k in WEAK_COMPLEXITY_HINTS):
        return 'structural_context'
    if n.startswith('motif:'):
        return 'positive_ingredient'
    return 'other'


def is_quality_context_v3(p):
    return predicate_kind_v3(p) == 'quality_context'


def is_positive_ingredient_v3(p):
    return predicate_kind_v3(p) == 'positive_ingredient'


def is_weak_structural_context_v3(p):
    return predicate_kind_v3(p) == 'structural_context'


def is_bad_failure_ingredient_v3(p):
    return predicate_kind_v3(p) == 'bad_failure_ingredient'


def make_bad_avoid_predicates():
    preds = []
    for name, fn, desc in BAD_SUBTYPE_DEFS:
        preds.append(Predicate(
            'NOT_' + name,
            'bad_subtype:' + name,
            'not',
            True,
            lambda r, f=fn: not f(r),
        ))
    return preds


def eval_rule_v3(preds, good_rows, bad_rows, cleaned_rows=None, manual_rows=None):
    cleaned_rows = cleaned_rows or []
    manual_rows = manual_rows or []
    def hit(row):
        return all(p(row) for p in preds)
    gh = [r for r in good_rows if hit(r)]
    bh = [r for r in bad_rows if hit(r)]
    ch = [r for r in cleaned_rows if hit(r)]
    mh = [r for r in manual_rows if hit(r)]
    gsup = len(gh) / max(1, len(good_rows))
    bsup = len(bh) / max(1, len(bad_rows))
    csup = len(ch) / max(1, len(cleaned_rows)) if cleaned_rows else 0.0
    msup = len(mh) / max(1, len(manual_rows)) if manual_rows else 0.0
    # Good/Bad precision intentionally excludes cleaned/manual, so it measures the actual flywheel contrast.
    precision = len(gh) / max(1, len(gh) + len(bh))
    lift = (gsup + 1e-9) / (bsup + 1e-9)
    return {
        'good_hits': len(gh),
        'bad_hits': len(bh),
        'cleaned_hits': len(ch),
        'manual_hits': len(mh),
        'good_support': gsup,
        'bad_support': bsup,
        'cleaned_support': csup,
        'manual_support': msup,
        'precision_vs_bad_support': precision,
        'lift_good_vs_bad': lift,
    }


def rule_non_triviality_v3(preds):
    kinds = [predicate_kind_v3(p) for p in preds]
    names = ' '.join(p.name.lower() for p in preds)
    has_strong = any(k == 'positive_ingredient' for k in kinds)
    has_quality = any(k == 'quality_context' for k in kinds)
    has_avoid = any(k == 'bad_avoid_context' for k in kinds)
    has_motif = 'motif:' in names or 'proxy:' in names or 'field:' in names or 'relhist:' in names or 'role:' in names
    # Penalize pure gates and pure atomic statistics.
    score = 0.15
    if has_strong:
        score += 0.35
    if has_quality:
        score += 0.18
    if has_avoid:
        score += 0.10
    if has_motif:
        score += 0.22
    if len(set(kinds)) >= 3:
        score += 0.10
    if all(k == 'quality_context' for k in kinds):
        score *= 0.25
    if ('x_count' in names or 'cycle_count' in names or 'min_length' in names or 'connected_components' in names) and not has_motif and not any(x in names for x in ['orbit', 'interlock', 'enclosure', 'spine', 'port']):
        score *= 0.50
    return max(0.0, min(1.0, score))


def old_rule_orthogonality_v3(preds, old_rules):
    # Reuse token/feature Jaccard, but make the report explicit that this is de-dup only.
    base = old_rule_orthogonality(preds, old_rules)
    base['nearest_old_rule_by_token'] = base.pop('nearest_old_rule', None)
    base['max_old_token_jaccard'] = base.pop('max_old_jaccard', 0.0)
    # We do not execute old rules here, so coverage orthogonality is reported as N/A.
    base['nearest_old_rule_by_coverage'] = None
    base['max_old_coverage_jaccard'] = 0.0
    return base


def score_quality_gate_v3(p, good_rows, bad_rows, cleaned_rows, manual_rows, old_rules):
    ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
    ortho = old_rule_orthogonality_v3([p], old_rules)
    contrast = max(0.0, ev['good_support'] - ev['bad_support'])
    score = 0.45 * contrast + 0.25 * ev['precision_vs_bad_support'] + 0.15 * generality_score(ev['good_support']) + 0.10 * ev['manual_support'] + 0.05 * ortho['orthogonality']
    return {**ev, **ortho, 'score': score, 'predicate': p.name, 'role': 'quality_gate', 'kind': predicate_kind_v3(p)}


def score_positive_ingredient_v3(p, good_rows, bad_rows, cleaned_rows, manual_rows, old_rules):
    ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
    ortho = old_rule_orthogonality_v3([p], old_rules)
    contrast = max(0.0, ev['good_support'] - ev['bad_support'])
    # Ingredients may be shared with Bad, so we do not require low bad_support.
    score = 0.35 * contrast + 0.20 * ev['precision_vs_bad_support'] + 0.15 * generality_score(ev['good_support']) + 0.15 * ev['cleaned_support'] + 0.05 * ev['manual_support'] + 0.10 * ortho['orthogonality']
    return {**ev, **ortho, 'score': score, 'predicate': p.name, 'role': 'positive_ingredient', 'kind': predicate_kind_v3(p)}


def mine_v3_contextual_rules(good_rows, bad_rows, cleaned_rows, manual_rows, old_rules, min_good_hits=5, top_k=40):
    all_rows = good_rows + bad_rows + cleaned_rows + manual_rows
    atoms = build_atomic_predicates(all_rows)
    atoms += make_bad_avoid_predicates()

    quality_atoms = [p for p in atoms if is_quality_context_v3(p)]
    positive_atoms = [p for p in atoms if is_positive_ingredient_v3(p)]
    weak_atoms = [p for p in atoms if is_weak_structural_context_v3(p)]
    avoid_atoms = [p for p in atoms if predicate_kind_v3(p) == 'bad_avoid_context']
    bad_atoms = [p for p in atoms if is_bad_failure_ingredient_v3(p)]

    quality_scored = []
    for p in quality_atoms:
        ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
        if ev['good_hits'] >= max(2, min_good_hits // 2):
            quality_scored.append(score_quality_gate_v3(p, good_rows, bad_rows, cleaned_rows, manual_rows, old_rules))
    quality_scored.sort(key=lambda x: x['score'], reverse=True)

    ingredient_scored = []
    for p in positive_atoms + weak_atoms:
        ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
        if ev['good_hits'] >= max(2, min_good_hits // 2):
            row = score_positive_ingredient_v3(p, good_rows, bad_rows, cleaned_rows, manual_rows, old_rules)
            row['role'] = 'positive_ingredient' if is_positive_ingredient_v3(p) else 'structural_context_ingredient'
            ingredient_scored.append(row)
    ingredient_scored.sort(key=lambda x: x['score'], reverse=True)

    # Bad failure rows from subtype definitions + bad-enriched motif atoms.
    failure_rows = []
    for name, fn, desc in BAD_SUBTYPE_DEFS:
        p = Predicate(name, 'bad_subtype:' + name, 'is', True, fn)
        ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
        failure_rows.append({
            'role': 'bad_failure_rule',
            'failure_name': name,
            'description': desc,
            **ev,
            'bad_minus_good': ev['bad_support'] - ev['good_support'],
            'score': max(0.0, ev['bad_support'] - ev['good_support']) + 0.15 * ev['bad_support'],
        })
    for p in bad_atoms:
        ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
        if ev['bad_support'] > ev['good_support'] and ev['bad_hits'] >= max(3, min_good_hits):
            failure_rows.append({
                'role': 'bad_failure_ingredient',
                'failure_name': p.name,
                'description': 'Bad-enriched motif/field ingredient; use as negative context, not as positive rule.',
                **ev,
                'bad_minus_good': ev['bad_support'] - ev['good_support'],
                'score': max(0.0, ev['bad_support'] - ev['good_support']) + 0.10 * ev['bad_support'],
            })
    failure_rows.sort(key=lambda x: x['score'], reverse=True)

    # Candidate contextual rules.
    # Require at least one positive ingredient and at least one context, so shallow gates cannot dominate.
    top_quality_pred_names = {r['predicate'] for r in quality_scored[:45]}
    top_ingredient_pred_names = {r['predicate'] for r in ingredient_scored[:60] if r['role'] == 'positive_ingredient'}
    top_weak_pred_names = {r['predicate'] for r in ingredient_scored[:35] if r['role'] == 'structural_context_ingredient'}
    q_atoms = [p for p in quality_atoms if p.name in top_quality_pred_names]
    pos_atoms = [p for p in positive_atoms if p.name in top_ingredient_pred_names]
    weak_context_atoms = [p for p in weak_atoms if p.name in top_weak_pred_names]
    avoid_atoms = avoid_atoms[:7]

    candidates = []
    seen_names = set()

    def add_candidate(combo, family):
        # no duplicate features unless one is NOT_bad context
        non_bad_feats = [p.feature for p in combo if not p.feature.startswith('bad_subtype:')]
        if len(non_bad_feats) != len(set(non_bad_feats)):
            return
        has_pos = any(is_positive_ingredient_v3(p) for p in combo)
        has_quality = any(is_quality_context_v3(p) for p in combo)
        has_avoid = any(predicate_kind_v3(p) == 'bad_avoid_context' for p in combo)
        has_weak = any(is_weak_structural_context_v3(p) for p in combo)
        if not has_pos:
            return
        if not (has_quality or has_avoid or has_weak):
            return
        name = ' AND '.join(p.name for p in combo)
        if name in seen_names:
            return
        seen_names.add(name)
        ev = eval_rule_v3(combo, good_rows, bad_rows, cleaned_rows, manual_rows)
        if ev['good_hits'] < min_good_hits:
            return
        if ev['good_support'] < 0.025:
            return
        # context rules can allow bad support, but should have some contrast or high precision.
        contrast = max(0.0, ev['good_support'] - ev['bad_support'])
        if contrast < 0.012 and ev['precision_vs_bad_support'] < 0.34:
            return
        ortho = old_rule_orthogonality_v3(combo, old_rules)
        nontriv = rule_non_triviality_v3(combo)
        gen = generality_score(ev['good_support'])
        manual_align = ev['manual_support']
        cleaned_align = ev['cleaned_support']
        div = diversity_bonus(combo)
        # Bad support is not forced to zero. It is interpreted as context dependence.
        score = (
            0.24 * contrast +
            0.17 * ev['precision_vs_bad_support'] +
            0.14 * gen +
            0.18 * ortho['orthogonality'] +
            0.11 * nontriv +
            0.08 * cleaned_align +
            0.04 * manual_align +
            0.04 * div
        )
        candidates.append({
            'score': score,
            'rule_name': name,
            'role': 'contextual_independent_rule',
            'rule_family': family,
            'predicates': [{'name': p.name, 'feature': p.feature, 'op': p.op, 'value': p.value, 'kind': predicate_kind_v3(p)} for p in combo],
            **ev,
            'bad_support_note': 'allowed: Bad may share ingredients; rule should be judged with bad subtype/context',
            'generality_score': gen,
            'non_triviality_score': nontriv,
            'diversity_bonus': div,
            **ortho,
            'write_to_rule_cache': False,
            'review_note': 'review-only candidate; old rules used only for orthogonality/de-dup',
        })

    # 2-predicate: positive + quality/weak
    for p in pos_atoms[:45]:
        for q in (q_atoms[:35] + weak_context_atoms[:20]):
            add_candidate((p, q), 'ingredient_plus_context')
    # 3-predicate: positive + quality + avoid or weak
    for p in pos_atoms[:35]:
        for q in q_atoms[:25]:
            for a in avoid_atoms[:7]:
                add_candidate((p, q, a), 'ingredient_plus_quality_plus_bad_avoid')
            for w in weak_context_atoms[:12]:
                add_candidate((p, q, w), 'ingredient_plus_quality_plus_structural_context')
    # motif + motif + quality for richer non-trivial rules.
    for i, p1 in enumerate(pos_atoms[:25]):
        for p2 in pos_atoms[i+1:25]:
            for q in q_atoms[:12]:
                add_candidate((p1, p2, q), 'two_ingredients_plus_quality')

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return {
        'quality_gate_candidates': quality_scored[:top_k],
        'positive_ingredient_candidates': ingredient_scored[:top_k],
        'bad_failure_rules': failure_rows[:top_k],
        'contextual_independent_rules': candidates[:top_k],
    }



# =============================================================================
# 4.7 V4 strict contextual miner
# =============================================================================
# V3 successfully separated gate/ingredient/failure/contextual rule, but its
# candidate generator still allowed degenerate semantic predicates such as
#   enclosure_score>=0 / interlock_score>=0 / orbit_score>=0
# Those are always-true lower bounds, so they are not real topology ingredients.
# V4 fixes that by requiring a positive ingredient to be semantically non-trivial.

SEMANTIC_SCORE_FEATURES = {
    'spine_score', 'enclosure_score', 'interlock_score', 'orbit_score'
}

# For score-like semantic features, values <= these thresholds are treated as
# not meaningful enough to serve as a positive ingredient.
STRICT_SEMANTIC_MIN_VALUE = {
    'spine_score': 0.24,
    'enclosure_score': 0.35,
    'interlock_score': 0.16,
    'orbit_score': 0.22,
}

STRICT_POSITIVE_MOTIF_ALLOW = [
    'proxy:orbit_or_seal_field',
    'proxy:nested_or_multi_cycle',
    'proxy:vertical_spine_branching',
    'proxy:interlocked_ports',
    'proxy:ring_core_spine_ports',
]


def _p_value_float(p, default=None):
    try:
        return float(p.value)
    except Exception:
        return default


def is_degenerate_semantic_predicate_v4(p):
    """Return True for predicates that look semantic but are basically always-on."""
    f = str(p.feature).lower()
    n = str(p.name).lower()
    if f in SEMANTIC_SCORE_FEATURES:
        val = _p_value_float(p, None)
        if p.op == '>=' and (val is None or val <= 1e-9):
            return True
        # A <= threshold on positive semantic scores is usually an absence/failure signal,
        # not a positive topology ingredient.
        if p.op == '<=':
            return True
    # Guard against string-form always-on predicates.
    for feat in SEMANTIC_SCORE_FEATURES:
        if n.startswith(feat + '>=0') or n.startswith(feat + '>=0.0'):
            return True
        if n.startswith(feat + '<='):
            return True
    return False


def motif_is_strict_positive_v4(name):
    n = str(name).lower()
    if not n.startswith('motif:'):
        return False
    body = n[len('motif:'):]
    if any(x in body for x in BAD_MOTIF_HINTS):
        return False
    if any(x in body for x in STRICT_POSITIVE_MOTIF_ALLOW):
        return True
    # Field motifs become positive only when they contain a visible structure field:
    # a spine plus enclosure or interlock. field:spineb0_enclb0_interb0 is excluded.
    if body.startswith('field:'):
        has_spine = ('spineb1' in body or 'spineb2' in body or 'spineb3' in body)
        has_encl = ('enclb2' in body or 'enclb3' in body or 'enclb4' in body)
        has_inter = ('interb1' in body or 'interb2' in body or 'interb3' in body)
        return has_spine and (has_encl or has_inter)
    # Relation histogram motifs: require T/X or cycle context. Pure E2E chain is not positive.
    if body.startswith('relhist:'):
        return ('_x1' in body or '_x2' in body or '_t1' in body or '_t2' in body or '_c1' in body or '_c2' in body or '_c3' in body)
    # Role motifs: high degree + branch/port bins are useful as ingredients, but low-degree role motifs are not.
    if body.startswith('role:'):
        return any(x in body for x in ['maxd3', 'maxd4', 'maxd5', 'maxd6']) and any(x in body for x in ['_b1', '_b2', '_b3', '_b4', 'portb2', 'portb3', 'portb4'])
    # Capacity motifs: keep only connected/compact forms with cycle context, not generic stroke count.
    if body.startswith('capacity:'):
        return ('cc1' in body) and any(x in body for x in ['_c1', '_c2', '_c3', '_c4'])
    return False


def is_strict_positive_ingredient_v4(p):
    """Positive ingredient must be genuinely structural, not a trivial score bound."""
    n = str(p.name).lower()
    f = str(p.feature).lower()
    if is_degenerate_semantic_predicate_v4(p):
        return False
    if n.startswith('motif:'):
        return motif_is_strict_positive_v4(n)
    if f in SEMANTIC_SCORE_FEATURES:
        val = _p_value_float(p, None)
        if p.op != '>=' or val is None:
            return False
        return val >= STRICT_SEMANTIC_MIN_VALUE.get(f, 0.1)
    if f == 'port_count_proxy':
        val = _p_value_float(p, None)
        # Only high port count can be a positive ingredient; <= is a readability/quality context.
        return p.op == '>=' and val is not None and val >= 14
    return False


def is_structural_context_v4(p):
    if is_strict_positive_ingredient_v4(p):
        return False
    k = predicate_kind_v3(p)
    n = str(p.name).lower()
    f = str(p.feature).lower()
    if k in {'structural_context', 'quality_context', 'bad_avoid_context'}:
        return True
    # cycles/X/T can act as context; alone they are not an independent rule.
    if any(x in f for x in ['cycle_count', 'x_count', 't_count', 'branch_ratio', 'max_degree', 'topology_event_count', 'mean_curvature', 'curve_ratio']):
        return True
    if n.startswith('motif:') and not motif_is_strict_positive_v4(n):
        return True
    return False


def ingredient_label_v4(p):
    n = str(p.name).lower()
    f = str(p.feature).lower()
    if 'ring_core_spine_ports' in n:
        return 'ring_core_spine_port'
    if 'orbit_or_seal' in n or f == 'orbit_score':
        return 'orbit_seal'
    if 'nested_or_multi_cycle' in n:
        return 'nested_portal'
    if 'interlocked_ports' in n or f == 'interlock_score':
        return 'interlock_port'
    if 'vertical_spine_branching' in n or f == 'spine_score':
        return 'spine_branch'
    if f == 'enclosure_score' or 'enclb' in n:
        return 'enclosure_field'
    if n.startswith('motif:field:'):
        return 'structure_field'
    if n.startswith('motif:relhist:'):
        return 'relation_motif'
    if n.startswith('motif:role:'):
        return 'role_port_motif'
    return 'semantic_topology'


def grammar_hint_v4(preds):
    labels = {ingredient_label_v4(p) for p in preds if is_strict_positive_ingredient_v4(p)}
    names = ' '.join(str(p.name).lower() for p in preds)
    if 'ring_core_spine_port' in labels:
        return 'production: sample_ring_core(); grow_spine(); bind_side_ports(); project_physical_constraints()'
    if 'orbit_seal' in labels and ('nested_portal' in labels or 'cycle_count' in names):
        return 'production: sample_outer_seal(); add_inner_orbit(); bind_ports_to_shell(); mutate_curve_flow()'
    if 'interlock_port' in labels:
        return 'production: sample_primary_component(); attach_secondary_by_typed_ports(); enforce_single_component()'
    if 'spine_branch' in labels and 'enclosure_field' in labels:
        return 'production: grow_axis_spine(); place_enclosure_field(); attach_curve_residuals_to_ports()'
    if 'nested_portal' in labels:
        return 'production: sample_nested_portal(); add_core_spine(); add_optional_hook_or_chord()'
    return 'production: sample_topology_motif(); bind_typed_ports(); apply_nonmechanical_geometry_mutation()'


def score_strict_positive_ingredient_v4(p, good_rows, bad_rows, cleaned_rows, manual_rows, old_rules):
    ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
    ortho = old_rule_orthogonality_v3([p], old_rules)
    contrast = ev['good_support'] - ev['bad_support']
    # Real ingredients may be shared; still require non-negative or near-neutral contrast unless strongly cleaned-aligned.
    score = (
        0.32 * max(0.0, contrast) +
        0.18 * ev['precision_vs_bad_support'] +
        0.16 * generality_score(ev['good_support']) +
        0.14 * ev['cleaned_support'] +
        0.06 * min(ev['manual_support'], 0.35) +
        0.14 * ortho['orthogonality']
    )
    return {**ev, **ortho, 'score': score, 'predicate': p.name, 'role': 'strict_positive_ingredient', 'kind': ingredient_label_v4(p)}


def rule_non_triviality_v4(preds):
    strict_pos = [p for p in preds if is_strict_positive_ingredient_v4(p)]
    has_quality = any(is_quality_context_v3(p) for p in preds)
    has_avoid = any(predicate_kind_v3(p) == 'bad_avoid_context' for p in preds)
    has_motif = any(str(p.name).lower().startswith('motif:') for p in preds)
    labels = {ingredient_label_v4(p) for p in strict_pos}
    score = 0.10
    if strict_pos:
        score += 0.38
    if len(labels) >= 2:
        score += 0.16
    if has_motif:
        score += 0.14
    if has_quality:
        score += 0.12
    if has_avoid:
        score += 0.12
    # Avoid rules that are just one broad semantic score and quality gate.
    if len(strict_pos) == 1 and not has_motif and len(preds) <= 2:
        score -= 0.10
    return max(0.0, min(1.0, score))


def mine_v4_strict_contextual_rules(good_rows, bad_rows, cleaned_rows, manual_rows, old_rules, min_good_hits=5, top_k=40):
    all_rows = good_rows + bad_rows + cleaned_rows + manual_rows
    atoms = build_atomic_predicates(all_rows)
    atoms += make_bad_avoid_predicates()

    quality_atoms = [p for p in atoms if is_quality_context_v3(p)]
    strict_pos_atoms = [p for p in atoms if is_strict_positive_ingredient_v4(p)]
    weak_context_atoms = [p for p in atoms if (not is_strict_positive_ingredient_v4(p)) and is_structural_context_v4(p) and predicate_kind_v3(p) not in {'quality_context', 'bad_avoid_context'}]
    avoid_atoms = [p for p in atoms if predicate_kind_v3(p) == 'bad_avoid_context']
    bad_atoms = [p for p in atoms if is_bad_failure_ingredient_v3(p)]

    quality_scored = []
    for p in quality_atoms:
        ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
        if ev['good_hits'] >= max(2, min_good_hits // 2):
            quality_scored.append(score_quality_gate_v3(p, good_rows, bad_rows, cleaned_rows, manual_rows, old_rules))
    quality_scored.sort(key=lambda x: x['score'], reverse=True)

    ingredient_scored = []
    for p in strict_pos_atoms:
        ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
        if ev['good_hits'] >= max(2, min_good_hits // 2):
            # Keep ingredients that are at least weakly Good/cleaned aligned.
            if ev['good_support'] + 0.05 < ev['bad_support'] and ev['cleaned_support'] < 0.20:
                continue
            ingredient_scored.append(score_strict_positive_ingredient_v4(p, good_rows, bad_rows, cleaned_rows, manual_rows, old_rules))
    ingredient_scored.sort(key=lambda x: x['score'], reverse=True)

    structural_scored = []
    for p in weak_context_atoms:
        ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
        if ev['good_hits'] >= max(2, min_good_hits // 2):
            row = score_positive_ingredient_v3(p, good_rows, bad_rows, cleaned_rows, manual_rows, old_rules)
            row['role'] = 'structural_context_ingredient'
            row['kind'] = predicate_kind_v3(p)
            structural_scored.append(row)
    structural_scored.sort(key=lambda x: x['score'], reverse=True)

    failure_rows = []
    for name, fn, desc in BAD_SUBTYPE_DEFS:
        p = Predicate(name, 'bad_subtype:' + name, 'is', True, fn)
        ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
        failure_rows.append({
            'role': 'bad_failure_rule',
            'failure_name': name,
            'description': desc,
            **ev,
            'bad_minus_good': ev['bad_support'] - ev['good_support'],
            'score': max(0.0, ev['bad_support'] - ev['good_support']) + 0.15 * ev['bad_support'],
        })
    for p in bad_atoms:
        ev = eval_rule_v3([p], good_rows, bad_rows, cleaned_rows, manual_rows)
        if ev['bad_support'] > ev['good_support'] and ev['bad_hits'] >= max(3, min_good_hits):
            failure_rows.append({
                'role': 'bad_failure_ingredient',
                'failure_name': p.name,
                'description': 'Bad-enriched motif/field ingredient; use as negative context, not as positive rule.',
                **ev,
                'bad_minus_good': ev['bad_support'] - ev['good_support'],
                'score': max(0.0, ev['bad_support'] - ev['good_support']) + 0.10 * ev['bad_support'],
            })
    failure_rows.sort(key=lambda x: x['score'], reverse=True)

    top_quality_pred_names = {r['predicate'] for r in quality_scored[:40]}
    top_pos_pred_names = {r['predicate'] for r in ingredient_scored[:45]}
    top_struct_pred_names = {r['predicate'] for r in structural_scored[:30]}
    q_atoms = [p for p in quality_atoms if p.name in top_quality_pred_names]
    pos_atoms = [p for p in strict_pos_atoms if p.name in top_pos_pred_names]
    struct_atoms = [p for p in weak_context_atoms if p.name in top_struct_pred_names]
    avoid_atoms = avoid_atoms[:7]

    candidates = []
    seen_names = set()

    def add_candidate(combo, family):
        non_bad_feats = [p.feature for p in combo if not str(p.feature).startswith('bad_subtype:')]
        if len(non_bad_feats) != len(set(non_bad_feats)):
            return
        strict_pos = [p for p in combo if is_strict_positive_ingredient_v4(p)]
        if not strict_pos:
            return
        has_quality = any(is_quality_context_v3(p) for p in combo)
        has_avoid = any(predicate_kind_v3(p) == 'bad_avoid_context' for p in combo)
        has_struct = any((p not in strict_pos) and is_structural_context_v4(p) for p in combo)
        if not (has_quality or has_avoid or has_struct):
            return
        # At least one truly non-trivial topology ingredient must remain after filters.
        if any(is_degenerate_semantic_predicate_v4(p) for p in strict_pos):
            return
        name = ' AND '.join(p.name for p in combo)
        if name in seen_names:
            return
        seen_names.add(name)
        ev = eval_rule_v3(combo, good_rows, bad_rows, cleaned_rows, manual_rows)
        if ev['good_hits'] < min_good_hits or ev['good_support'] < 0.025:
            return
        contrast = ev['good_support'] - ev['bad_support']
        # Stricter than V3: a candidate should either contrast Bad or have strong cleaned/manual support with an avoid condition.
        if contrast < 0.02 and not (has_avoid and ev['cleaned_support'] >= 0.35 and ev['precision_vs_bad_support'] >= 0.18):
            return
        ortho = old_rule_orthogonality_v3(combo, old_rules)
        nontriv = rule_non_triviality_v4(combo)
        gen = generality_score(ev['good_support'])
        div = diversity_bonus(combo)
        bad_penalty = min(0.18, 0.20 * ev['bad_support'])
        score = (
            0.28 * max(0.0, contrast) +
            0.18 * ev['precision_vs_bad_support'] +
            0.14 * gen +
            0.16 * ortho['orthogonality'] +
            0.14 * nontriv +
            0.06 * ev['cleaned_support'] +
            0.02 * min(ev['manual_support'], 0.45) +
            0.04 * div -
            bad_penalty
        )
        labels = sorted({ingredient_label_v4(p) for p in strict_pos})
        candidates.append({
            'score': score,
            'rule_name': name,
            'role': 'strict_contextual_independent_rule',
            'rule_family': family,
            'ingredient_labels': '+'.join(labels),
            'predicates': [{'name': p.name, 'feature': p.feature, 'op': p.op, 'value': p.value, 'kind': ('strict_positive_ingredient' if is_strict_positive_ingredient_v4(p) else predicate_kind_v3(p))} for p in combo],
            **ev,
            'bad_support_note': 'allowed but penalized; this rule is valid only with listed context/avoid constraints',
            'generality_score': gen,
            'non_triviality_score': nontriv,
            'diversity_bonus': div,
            'generation_hint': grammar_hint_v4(combo),
            **ortho,
            'write_to_rule_cache': False,
            'review_note': 'V4 strict candidate; convert accepted rule to typed-port graph grammar production after human review',
        })

    # Prefer 3-predicate rules with avoid condition; these are closer to actual contextual rules.
    for p in pos_atoms[:35]:
        for q in q_atoms[:25]:
            for a in avoid_atoms[:7]:
                add_candidate((p, q, a), 'strict_ingredient_plus_quality_plus_bad_avoid')
    # Positive + structural context + quality.
    for p in pos_atoms[:30]:
        for sctx in struct_atoms[:18]:
            for q in q_atoms[:14]:
                add_candidate((p, sctx, q), 'strict_ingredient_plus_structural_context_plus_quality')
    # Two positive ingredients + quality, likely grammar-like motifs.
    for i, p1 in enumerate(pos_atoms[:24]):
        for p2 in pos_atoms[i+1:24]:
            for q in q_atoms[:12]:
                add_candidate((p1, p2, q), 'two_strict_ingredients_plus_quality')
    # Two positive ingredients + avoid, allowing geometry to be freer.
    for i, p1 in enumerate(pos_atoms[:18]):
        for p2 in pos_atoms[i+1:18]:
            for a in avoid_atoms[:5]:
                add_candidate((p1, p2, a), 'two_strict_ingredients_plus_bad_avoid')

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return {
        'quality_gate_candidates': quality_scored[:top_k],
        'positive_ingredient_candidates': ingredient_scored[:top_k],
        'structural_context_candidates': structural_scored[:top_k],
        'bad_failure_rules': failure_rows[:top_k],
        'contextual_independent_rules': candidates[:top_k],
    }

# Preserve the existing main/report code while swapping in the stricter miner.
mine_v3_contextual_rules = mine_v4_strict_contextual_rules

def emit_review_rule_json_v3(contextual_rules, output_path):
    review = []
    for i, r in enumerate(contextual_rules):
        review.append({
            'rule_id': f'independent_contextual_candidate_{i:04d}',
            'source': 'good_bad_contextual_rule_miner_v3_review_only',
            'depends_on_existing_rule': False,
            'status': 'review_only_not_auto_accepted',
            'score': r.get('score'),
            'rule_name': r.get('rule_name'),
            'rule_family': r.get('rule_family'),
            'predicates': r.get('predicates', []),
            'supports': {
                'good_support': r.get('good_support'),
                'bad_support': r.get('bad_support'),
                'cleaned_support': r.get('cleaned_support'),
                'manual_support': r.get('manual_support'),
                'precision_vs_bad_support': r.get('precision_vs_bad_support'),
            },
            'orthogonality': {
                'nearest_old_rule_by_token': r.get('nearest_old_rule_by_token'),
                'max_old_token_jaccard': r.get('max_old_token_jaccard'),
                'nearest_old_rule_by_coverage': r.get('nearest_old_rule_by_coverage'),
                'max_old_coverage_jaccard': r.get('max_old_coverage_jaccard'),
            },
            'notes': {
                'bad_support_note': r.get('bad_support_note'),
                'review_note': r.get('review_note'),
                'next_step': 'convert accepted rule into typed-port graph grammar production, not an if-else template',
            }
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(review, f, ensure_ascii=False, indent=2)


# =============================================================================
# 5. Report printing
# =============================================================================

class Reporter:
    def __init__(self):
        self.lines = []
    def write(self, s=''):
        print(s)
        self.lines.append(str(s))
    def section(self, title):
        self.write('\n' + '=' * 92)
        self.write(title)
        self.write('=' * 92)
    def subsection(self, title):
        self.write('\n' + '-' * 88)
        self.write(title)
        self.write('-' * 88)


def print_table(rep, rows, cols, max_rows=20):
    rows = rows[:max_rows]
    if not rows:
        rep.write('(empty)')
        return
    widths = []
    for c in cols:
        widths.append(max(len(c), min(80, max(len(str(r.get(c, ''))) for r in rows))))
    header = ' | '.join(c.ljust(widths[i]) for i, c in enumerate(cols))
    rep.write(header)
    rep.write('-' * len(header))
    for r in rows:
        vals = []
        for i, c in enumerate(cols):
            v = r.get(c, '')
            if isinstance(v, float):
                v = fmt_float(v)
            s = str(v)
            if len(s) > widths[i]:
                s = s[:max(0, widths[i]-3)] + '...'
            vals.append(s.ljust(widths[i]))
        rep.write(' | '.join(vals))


def feature_delta_table(summaries, label_a='good', label_b='bad', top_k=25):
    A = summaries.get(label_a, {})
    B = summaries.get(label_b, {})
    rows = []
    for feat in NUMERIC_FEATURES:
        if feat not in A or feat not in B:
            continue
        am = A[feat]['mean']; bm = B[feat]['mean']
        # normalized delta using pooled spread proxy
        spread = max(1e-6, (A[feat]['p75'] - A[feat]['p25'] + B[feat]['p75'] - B[feat]['p25']) / 2.0)
        rows.append({
            'feature': feat,
            f'{label_a}_mean': am,
            f'{label_b}_mean': bm,
            'delta': am - bm,
            'norm_delta': (am - bm) / spread,
            f'{label_a}_median': A[feat]['median'],
            f'{label_b}_median': B[feat]['median'],
        })
    rows.sort(key=lambda x: abs(x['norm_delta']), reverse=True)
    return rows[:top_k]


def main():
    parser = argparse.ArgumentParser(description='V4 strict contextual independent topology rule miner: separates quality gates, positive ingredients, bad failure modes, and contextual rules.')
    parser.add_argument('--max-files-per-dir', type=int, default=None, help='limit json files per directory for quick debug')
    parser.add_argument('--max-records-per-group', type=int, default=None, help='limit records per label group')
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--min-good-hits', type=int, default=5)
    parser.add_argument('--old-rule-max-files', type=int, default=None)
    parser.add_argument('--emit-rule-json', action='store_true', help='emit review-only contextual independent rule JSON candidates')
    args = parser.parse_args()

    t0 = time.time()
    rep = Reporter()
    rep.section('Morpheme Independent Rule Miner V4 Strict Contextual')
    rep.write(f'SCRIPT_DIR              = {SCRIPT_DIR}')
    rep.write(f'CHAR_GLYPH_DIR          = {CHAR_GLYPH_DIR}')
    rep.write(f'PCG_GOOD_DIR            = {PCG_GOOD_DIR}')
    rep.write(f'PCG_BAD_DIR             = {PCG_BAD_DIR}')
    rep.write(f'PCG_CLEANED_DIR         = {PCG_CLEANED_DIR}')
    rep.write(f'MANUAL_ANNOTATIONS_DIR  = {MANUAL_ANNOTATIONS_DIR}')
    rep.write(f'MORPHEME_OUTPUT_TREE    = {MORPHEME_OUTPUT_TREE_DIR}')
    rep.write(f'MORPHEME_NEW_RULE_CACHE = {MORPHEME_NEW_RULE_CACHE_DIR}')

    groups = {}
    load_meta = {}
    for label, path in [
        ('good', PCG_GOOD_DIR),
        ('bad', PCG_BAD_DIR),
        ('cleaned', PCG_CLEANED_DIR),
        ('manual', MANUAL_ANNOTATIONS_DIR),
    ]:
        recs, meta = load_records_from_dir(path, label, max_files=args.max_files_per_dir, max_records=args.max_records_per_group)
        groups[label] = recs
        load_meta[label] = meta

    rep.section('01 Data loading summary')
    for label in ['good', 'bad', 'cleaned', 'manual']:
        meta = load_meta[label]
        rep.write(f'{label:8s}: records={len(groups[label]):6d}, files={meta["files"]:5d}, bad_json={meta["bad_json"]:4d}, truncated={meta.get("truncated")}')

    rep.write('\nNote: cleaned is treated as positive-ish reference, but candidate rules are mainly Good vs Bad; Manual is alignment/reference.')

    old_rules, old_file_count = load_old_rules(max_files=args.old_rule_max_files)
    rep.section('02 Old rule inventory for orthogonality / de-dup only')
    rep.write(f'old rule json files scanned = {old_file_count}')
    rep.write(f'old rule-like objects found = {len(old_rules)}')
    feature_vocab = Counter()
    token_vocab = Counter()
    for r in old_rules:
        feature_vocab.update(r.get('features', []))
        token_vocab.update(r.get('tokens', []))
    rep.write(f'old rule feature vocab size = {len(feature_vocab)}')
    rep.write(f'old rule token vocab size   = {len(token_vocab)}')
    rep.subsection('Top old-rule feature tokens')
    rows = [{'token': k, 'count': v} for k, v in (feature_vocab + token_vocab).most_common(30)]
    print_table(rep, rows, ['token', 'count'], max_rows=30)

    # Extract features
    feature_rows = {}
    for label, recs in groups.items():
        rows = []
        for rec in recs:
            try:
                rows.append(extract_features(rec))
            except Exception as e:
                # keep going; bad schema should not break full audit
                pass
        feature_rows[label] = rows

    rep.section('03 Feature extraction summary')
    for label, rows in feature_rows.items():
        rep.write(f'{label:8s}: feature_rows={len(rows):6d}')
        if rows:
            sample = rows[0]
            rep.write(f'  sample_id={sample.get("id")} motifs={sample.get("_motifs", [])[:5]} source={sample.get("source_file", "")[:100]}')

    summaries = {label: summarize_numeric(rows, label) for label, rows in feature_rows.items()}
    rep.section('04 Key numeric distribution by label')
    key_feats = [
        'stroke_count', 'connected_components', 'cycle_count', 'topology_event_count',
        'E2E_count', 'T_count', 'X_count', 'axis_ratio', 'diagonal_ratio', 'curve_ratio',
        'spine_score', 'enclosure_score', 'interlock_score', 'orbit_score', 'fragmentation_risk',
    ]
    for feat in key_feats:
        line = [feat]
        for label in ['good', 'bad', 'cleaned', 'manual']:
            s = summaries.get(label, {}).get(feat)
            if s:
                line.append(f'{label}: mean={fmt_float(s["mean"],3)} med={fmt_float(s["median"],3)} p25={fmt_float(s["p25"],3)} p75={fmt_float(s["p75"],3)}')
        rep.write(' | '.join(line))

    rep.section('05 Good vs Bad feature deltas')
    delta_rows = feature_delta_table(summaries, 'good', 'bad', top_k=args.top_k)
    print_table(rep, delta_rows, ['feature', 'good_mean', 'bad_mean', 'delta', 'norm_delta', 'good_median', 'bad_median'], max_rows=args.top_k)
    rep.write('\nInterpretation: large delta means Good and Bad differ; but if both means are high, this is not a sufficient rule. Check shared motifs below.')

    rep.section('05B Bad subtype clustering / failure mode decomposition')
    bad_subtypes = bad_subtype_rows(feature_rows['good'], feature_rows['bad'])
    print_table(rep, bad_subtypes, ['bad_subtype', 'bad_hits', 'bad_support', 'good_support', 'bad_minus_good', 'description'], max_rows=len(bad_subtypes))
    rep.write('\nInterpretation: Bad samples are structured negatives. These subtypes should become separate negative contexts, not one mixed Bad label.')


    rep.section('06 Motif support: Good-enriched / Bad-enriched / Shared')
    motif_cmp = compare_motifs(feature_rows['good'], feature_rows['bad'], feature_rows['manual'], top_k=args.top_k)
    rep.subsection('06A Good-enriched motifs: likely positive ingredients')
    print_table(rep, motif_cmp['good_enriched'], ['motif', 'good_support', 'bad_support', 'manual_support', 'good_minus_bad'], max_rows=args.top_k)
    rep.subsection('06B Bad-enriched motifs: structures that often fail or need constraints')
    print_table(rep, motif_cmp['bad_enriched'], ['motif', 'good_support', 'bad_support', 'manual_support', 'bad_minus_good'], max_rows=args.top_k)
    rep.subsection('06C Shared motifs: exist in both Good and Bad, so they are context-dependent, not sufficient')
    print_table(rep, motif_cmp['shared'], ['motif', 'good_support', 'bad_support', 'manual_support', 'shared_strength'], max_rows=args.top_k)
    rep.subsection('06D Manual-aligned motifs: closer to human annotation distribution')
    print_table(rep, motif_cmp['manual_aligned'], ['motif', 'good_support', 'bad_support', 'manual_support', 'manual_alignment'], max_rows=args.top_k)

    rep.section('07 V4 strict separated rule mining: gates / ingredients / failure modes / contextual rules')
    v3 = mine_v3_contextual_rules(
        feature_rows['good'],
        feature_rows['bad'],
        feature_rows['cleaned'],
        feature_rows['manual'],
        old_rules,
        min_good_hits=args.min_good_hits,
        top_k=args.top_k,
    )
    quality_gates = v3['quality_gate_candidates']
    positive_ingredients = v3['positive_ingredient_candidates']
    bad_failure_rules = v3['bad_failure_rules']
    contextual_rules = v3['contextual_independent_rules']
    candidates = contextual_rules

    rep.write('V4 design:')
    rep.write('  - quality gates are not treated as topology rules; they are only context / filters.')
    rep.write('  - positive ingredients are topology ingredients, not sufficient rules by themselves.')
    rep.write('  - bad failure rules explain structured negative cases.')
    rep.write('  - contextual independent rules require a non-trivial strict positive ingredient + context; old rules are used only for de-dup/orthogonality.')

    rep.subsection('07A Quality gate candidates: useful filters, not independent topology rules')
    print_table(rep, quality_gates, [
        'score', 'predicate', 'role', 'good_support', 'bad_support', 'cleaned_support', 'manual_support',
        'precision_vs_bad_support', 'orthogonality', 'nearest_old_rule_by_token', 'max_old_token_jaccard'
    ], max_rows=args.top_k)

    rep.subsection('07B Strict positive ingredient candidates: non-trivial topology ingredients, not sufficient alone')
    print_table(rep, positive_ingredients, [
        'score', 'predicate', 'role', 'kind', 'good_support', 'bad_support', 'cleaned_support', 'manual_support',
        'precision_vs_bad_support', 'orthogonality', 'nearest_old_rule_by_token', 'max_old_token_jaccard'
    ], max_rows=args.top_k)

    rep.subsection('07C Bad failure rules: structured negatives / constraints')
    print_table(rep, bad_failure_rules, [
        'score', 'failure_name', 'role', 'good_support', 'bad_support', 'bad_minus_good', 'cleaned_support', 'manual_support', 'description'
    ], max_rows=args.top_k)

    rep.subsection('07D Strict contextual independent rule candidates: review-first, possible new rule cache entries')
    print_table(rep, contextual_rules, [
        'score', 'rule_name', 'role', 'rule_family', 'good_support', 'bad_support', 'cleaned_support', 'manual_support',
        'precision_vs_bad_support', 'generality_score', 'non_triviality_score', 'orthogonality',
        'nearest_old_rule_by_token', 'max_old_token_jaccard'
    ], max_rows=args.top_k)

    rep.section('08 Design hints for next model/rule miner')
    rep.write('1. Orthogonal rule discovery should read old rules, but only for duplicate suppression:')
    rep.write('   candidate_rule_features vs old_rule_features -> graph/token similarity -> reject if too close.')
    rep.write('2. Bad samples should not be treated as topology-empty negatives.')
    rep.write('   Shared motifs show which structures need contextual constraints rather than direct rejection.')
    rep.write('3. Good-enriched motifs are positive ingredients; Bad-enriched motifs are failure modes; Shared motifs are neutral primitives.')
    rep.write('4. The next real miner should replace proxy motifs with canonical attributed subgraph mining:')
    rep.write('   stroke node + E2E/T/X edge + cycle/port attributes + angle/position/curvature buckets.')
    rep.write('5. The final independent rule score should combine:')
    rep.write('   Good coverage + Bad contrast + Manual alignment + MDL/generalization + Orthogonality-to-old-rules + Non-triviality.')

    summary = {
        'paths': {
            'script_dir': str(SCRIPT_DIR),
            'char_glyph_dir': str(CHAR_GLYPH_DIR),
            'good_dir': str(PCG_GOOD_DIR),
            'bad_dir': str(PCG_BAD_DIR),
            'cleaned_dir': str(PCG_CLEANED_DIR),
            'manual_dir': str(MANUAL_ANNOTATIONS_DIR),
            'morpheme_output_tree': str(MORPHEME_OUTPUT_TREE_DIR),
            'morpheme_new_rule_cache': str(MORPHEME_NEW_RULE_CACHE_DIR),
        },
        'load_meta': load_meta,
        'record_counts': {k: len(v) for k, v in groups.items()},
        'feature_row_counts': {k: len(v) for k, v in feature_rows.items()},
        'old_rule_count': len(old_rules),
        'old_rule_feature_vocab_top': (feature_vocab + token_vocab).most_common(80),
        'numeric_summaries': summaries,
        'feature_deltas_good_bad': delta_rows,
        'motif_comparison': motif_cmp,
        'bad_subtypes': bad_subtypes if 'bad_subtypes' in locals() else [],
        'quality_gate_candidates': quality_gates if 'quality_gates' in locals() else [],
        'positive_ingredient_candidates': positive_ingredients if 'positive_ingredients' in locals() else [],
        'bad_failure_rules': bad_failure_rules if 'bad_failure_rules' in locals() else [],
        'contextual_independent_rules': contextual_rules if 'contextual_rules' in locals() else [],
        'runtime_seconds': time.time() - t0,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = OUTPUT_DIR / 'independent_rule_miner_v4_strict_contextual_report.txt'
    json_path = OUTPUT_DIR / 'independent_rule_miner_v4_strict_contextual_summary.json'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rep.lines))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


    # Extra CSV outputs for easier review.
    def write_csv(path, rows):
        if not rows:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write('empty\n')
            return
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in keys})

    import csv
    write_csv(OUTPUT_DIR / 'bad_subtype_summary.csv', bad_subtypes if 'bad_subtypes' in locals() else [])
    write_csv(OUTPUT_DIR / 'quality_gate_candidates.csv', quality_gates if 'quality_gates' in locals() else [])
    write_csv(OUTPUT_DIR / 'positive_ingredient_candidates.csv', positive_ingredients if 'positive_ingredients' in locals() else [])
    write_csv(OUTPUT_DIR / 'bad_failure_rules.csv', bad_failure_rules if 'bad_failure_rules' in locals() else [])
    write_csv(OUTPUT_DIR / 'contextual_independent_rule_candidates.csv', contextual_rules if 'contextual_rules' in locals() else [])
    if getattr(args, 'emit_rule_json', False):
        emit_review_rule_json_v3(contextual_rules if 'contextual_rules' in locals() else [], OUTPUT_DIR / 'candidate_independent_rules_v4_review_only.json')
    flat_motif_rows = []
    for group_name, group_rows in motif_cmp.items():
        for r in group_rows:
            rr = dict(r)
            rr['group'] = group_name
            flat_motif_rows.append(rr)
    write_csv(OUTPUT_DIR / 'motif_support_summary.csv', flat_motif_rows)

    rep.section('09 Output files')
    rep.write(f'Wrote text report: {txt_path}')
    rep.write(f'Wrote json summary: {json_path}')
    rep.write(f"Wrote bad subtype csv: {OUTPUT_DIR / 'bad_subtype_summary.csv'}")
    rep.write(f"Wrote quality gates csv: {OUTPUT_DIR / 'quality_gate_candidates.csv'}")
    rep.write(f"Wrote positive ingredients csv: {OUTPUT_DIR / 'positive_ingredient_candidates.csv'}")
    rep.write(f"Wrote bad failure rules csv: {OUTPUT_DIR / 'bad_failure_rules.csv'}")
    rep.write(f"Wrote contextual rules csv: {OUTPUT_DIR / 'contextual_independent_rule_candidates.csv'}")
    if getattr(args, 'emit_rule_json', False):
        rep.write(f"Wrote review rule json: {OUTPUT_DIR / 'candidate_independent_rules_v4_review_only.json'}")
    rep.write(f"Wrote motif support csv: {OUTPUT_DIR / 'motif_support_summary.csv'}")
    rep.write(f'Runtime seconds: {fmt_float(time.time() - t0, 2)}')

if __name__ == '__main__':
    main()
