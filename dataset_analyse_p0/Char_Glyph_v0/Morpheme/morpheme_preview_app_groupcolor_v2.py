# -*- coding: utf-8 -*-
r"""
morpheme_preview_app_groupcolor_v2.py

放置：
    dataset_analyse_p0/Char_Glyph_v0/Morpheme/morpheme_preview_app.py
或：
    dataset_analyse_p0/Char_Glyph_v0/Morpheme/morpheme_preview_app_groupcolor_v2.py

作用：
    从 output_tree 读取拓扑语素库，随机预览一个语素：
      - 左上：parent morpheme cluster prototype
      - 左下：同一个 composition example 里的 parent，并按 child_a / child_b 完美着色
      - 右上：同一个 example 的 child_a
      - 右下：同一个 example 的 child_b
      - 下方文本显示 topology_way / example / coverage

重要：
    如果 output_tree 是旧 builder 生成的，composition.examples 里没有 parent_example_segments，
    那么只能用 source_sid 近似映射，复杂图可能出现灰线。
    要实现 parent 边被 child 完美分割，请先使用：
        morpheme_tree_builder_EXISTING_MIRROR_ONLY_EXAMPLE_GEOM.py
    重新生成 output_tree。
"""

from __future__ import annotations

import os
import json
import random
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TREE_DIR = os.path.join(SCRIPT_DIR, "output_tree")

COLOR_CHILD_A = "#d64b4b"
COLOR_CHILD_B = "#3d6fd6"
COLOR_OVERLAP = "#8a45b8"
COLOR_OTHER = "#888888"
COLOR_TEXT = "#1f1f1f"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sharded_rows(tree_dir, prefix):
    rows = []
    manifest_path = os.path.join(tree_dir, f"{prefix}_manifest.json")
    if os.path.exists(manifest_path):
        mani = load_json(manifest_path)
        for sh in mani.get("shards", []):
            p = os.path.join(tree_dir, sh.get("file", ""))
            if os.path.exists(p):
                obj = load_json(p)
                rows.extend(obj.get("rows", []))
        return rows

    for fn in sorted(os.listdir(tree_dir)):
        if fn.startswith(prefix + "_shard_") and fn.endswith(".json"):
            obj = load_json(os.path.join(tree_dir, fn))
            rows.extend(obj.get("rows", []))
    return rows


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def seg_identity(seg, fallback_i=None):
    for k in ("source_sid", "local_id", "sid", "stroke_id", "glyph_stroke_index"):
        if k in seg:
            return str(seg.get(k))
    if fallback_i is None:
        return None
    return str(fallback_i + 1)


def bbox_of_segments(segments):
    pts = []
    for s in segments or []:
        p0, p1 = s.get("p0"), s.get("p1")
        if isinstance(p0, list) and len(p0) == 2:
            pts.append((safe_float(p0[0]), safe_float(p0[1])))
        if isinstance(p1, list) and len(p1) == 2:
            pts.append((safe_float(p1[0]), safe_float(p1[1])))
    if not pts:
        return (0, 0, 1, 1)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def transform_point(x, y, bbox, W, H, pad=24):
    xmin, ymin, xmax, ymax = bbox
    bw = max(1e-6, xmax - xmin)
    bh = max(1e-6, ymax - ymin)
    scale = min((W - 2 * pad) / bw, (H - 2 * pad) / bh)
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    px = W * 0.5 + (x - cx) * scale
    py = H * 0.5 - (y - cy) * scale
    return px, py


def draw_segments(canvas, segments, title="", width=3, jitter=False,
                  default_color="black", color_by_seg_id=None,
                  label_color="blue", show_ids=True, title_suffix=None,
                  color_by_group=False):
    canvas.delete("all")
    canvas.update_idletasks()
    W = int(canvas.winfo_width() or 300)
    H = int(canvas.winfo_height() or 220)
    title_text = title if title_suffix is None else f"{title} | {title_suffix}"
    canvas.create_text(10, 10, anchor="nw", text=title_text, font=("Arial", 12, "bold"), fill=COLOR_TEXT)

    if not segments:
        canvas.create_text(W/2, H/2, text="No prototype/example segments", fill="gray")
        return

    bbox = bbox_of_segments(segments)
    rng = random.Random(1234 if not jitter else random.randint(0, 999999))

    for i, s in enumerate(segments):
        p0 = s.get("p0", [0, 0])
        p1 = s.get("p1", [1, 0])
        x0, y0 = safe_float(p0[0]), safe_float(p0[1])
        x1, y1 = safe_float(p1[0]), safe_float(p1[1])

        if jitter:
            dx0, dy0 = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
            dx1, dy1 = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
            x0, y0 = x0 + dx0, y0 + dy0
            x1, y1 = x1 + dx1, y1 + dy1

        px0, py0 = transform_point(x0, y0, bbox, W, H)
        px1, py1 = transform_point(x1, y1, bbox, W, H)

        sid = seg_identity(s, i)
        color = default_color

        if color_by_group:
            group = str(s.get("group", "other"))
            if group == "child_a":
                color = COLOR_CHILD_A
            elif group == "child_b":
                color = COLOR_CHILD_B
            elif group == "overlap":
                color = COLOR_OVERLAP
            else:
                color = COLOR_OTHER
        elif color_by_seg_id and sid in color_by_seg_id:
            color = color_by_seg_id[sid]

        canvas.create_line(px0, py0, px1, py1, width=width, capstyle="round", fill=color)
        if show_ids:
            mx, my = (px0 + px1) / 2, (py0 + py1) / 2
            canvas.create_text(mx, my, text=sid, fill=label_color, font=("Arial", 9))

    x0, y0 = transform_point(bbox[0], bbox[1], bbox, W, H)
    x1, y1 = transform_point(bbox[2], bbox[3], bbox, W, H)
    canvas.create_rectangle(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), outline="#dddddd")


def infer_parent_child_color_map(parent_segments, comp):
    color_map = {}
    if not comp:
        return color_map, {"child_a": [], "child_b": [], "overlap": [], "other": []}

    exs = comp.get("examples") or []
    ex0 = exs[0] if exs else {}
    child_a = set(str(x) for x in (ex0.get("child_a_source_sids") or []))
    child_b = set(str(x) for x in (ex0.get("child_b_source_sids") or []))

    groups = {"child_a": [], "child_b": [], "overlap": [], "other": []}
    for i, seg in enumerate(parent_segments or []):
        sid = seg_identity(seg, i)
        in_a = sid in child_a
        in_b = sid in child_b
        if in_a and in_b:
            color_map[sid] = COLOR_OVERLAP
            groups["overlap"].append(sid)
        elif in_a:
            color_map[sid] = COLOR_CHILD_A
            groups["child_a"].append(sid)
        elif in_b:
            color_map[sid] = COLOR_CHILD_B
            groups["child_b"].append(sid)
        else:
            color_map[sid] = COLOR_OTHER
            groups["other"].append(sid)
    return color_map, groups


def group_summary_from_segments(segments):
    out = {"child_a": [], "child_b": [], "overlap": [], "other": []}
    for i, s in enumerate(segments or []):
        g = str(s.get("group", "other"))
        sid = seg_identity(s, i)
        if g in out:
            out[g].append(sid)
        else:
            out["other"].append(sid)
    return out


def example_has_exact_geometry(comp):
    exs = comp.get("examples") or []
    if not exs:
        return False
    ex0 = exs[0]
    return bool(ex0.get("parent_example_segments"))


class MorphemePreviewApp:
    def __init__(self, root, tree_dir):
        self.root = root
        self.tree_dir = tree_dir
        self.root.title("Topology Morpheme Preview - exact child grouping")

        self.nodes = {}
        self.comps_by_parent = defaultdict(list)
        self.all_comps = []
        self.current_parent_id = None
        self.current_comp = None
        self.current_grouped = {"child_a": [], "child_b": [], "overlap": [], "other": []}
        self.current_exact = False

        self.stable_only_var = tk.BooleanVar(value=True)
        self.has_comp_only_var = tk.BooleanVar(value=True)
        self.jitter_style_var = tk.BooleanVar(value=True)
        self.stroke_min_var = tk.StringVar(value="1")
        self.stroke_max_var = tk.StringVar(value="8")
        self.id_var = tk.StringVar()

        self._load_data()
        self._build_ui()
        self.random_parent()

    def _load_data(self):
        if not os.path.isdir(self.tree_dir):
            raise FileNotFoundError(f"tree_dir not found: {self.tree_dir}")

        nodes = load_sharded_rows(self.tree_dir, "morpheme_nodes")
        comps = load_sharded_rows(self.tree_dir, "composition_alternatives")

        self.nodes = {n.get("morpheme_id"): n for n in nodes if n.get("morpheme_id")}
        self.all_comps = comps
        self.comps_by_parent = defaultdict(list)
        for c in comps:
            p = c.get("parent_morpheme_id")
            if p:
                self.comps_by_parent[p].append(c)

        for p in list(self.comps_by_parent.keys()):
            self.comps_by_parent[p].sort(key=lambda x: (-int(x.get("support_count") or 0), x.get("composition_id", "")))

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text=f"tree_dir: {self.tree_dir}").grid(row=0, column=0, columnspan=10, sticky="w")

        ttk.Checkbutton(top, text="stable only", variable=self.stable_only_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(top, text="has composition", variable=self.has_comp_only_var).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(top, text="random preview style", variable=self.jitter_style_var).grid(row=1, column=2, sticky="w")

        ttk.Label(top, text="stroke min").grid(row=1, column=3, sticky="e")
        ttk.Entry(top, textvariable=self.stroke_min_var, width=5).grid(row=1, column=4, sticky="w")
        ttk.Label(top, text="max").grid(row=1, column=5, sticky="e")
        ttk.Entry(top, textvariable=self.stroke_max_var, width=5).grid(row=1, column=6, sticky="w")

        ttk.Button(top, text="Random Parent", command=self.random_parent).grid(row=1, column=7, padx=4)
        ttk.Button(top, text="Random Rule", command=self.random_rule).grid(row=1, column=8, padx=4)

        ttk.Label(top, text="morpheme_id").grid(row=2, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.id_var, width=28).grid(row=2, column=1, columnspan=2, sticky="we")
        ttk.Button(top, text="Open ID", command=self.open_id).grid(row=2, column=3, padx=4)
        ttk.Button(top, text="Redraw", command=self.redraw).grid(row=2, column=4, padx=4)

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        self.parent_canvas = tk.Canvas(main, width=420, height=260, bg="white")
        self.parent_group_canvas = tk.Canvas(main, width=420, height=260, bg="white")
        self.child_a_canvas = tk.Canvas(main, width=300, height=220, bg="white")
        self.child_b_canvas = tk.Canvas(main, width=300, height=220, bg="white")

        self.parent_canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.parent_group_canvas.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.child_a_canvas.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.child_b_canvas.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="both", expand=True)

        self.info_text = tk.Text(bottom, height=18, wrap="word")
        self.info_text.pack(fill="both", expand=True)

    def _eligible_ids(self):
        try:
            mn = int(self.stroke_min_var.get())
        except Exception:
            mn = 1
        try:
            mx = int(self.stroke_max_var.get())
        except Exception:
            mx = 99

        ids = []
        for mid, node in self.nodes.items():
            sc = int(node.get("stroke_count") or 0)
            if sc < mn or sc > mx:
                continue
            if self.stable_only_var.get() and not node.get("is_stable"):
                continue
            if self.has_comp_only_var.get() and not self.comps_by_parent.get(mid):
                continue
            ids.append(mid)
        return ids

    def random_parent(self):
        ids = self._eligible_ids()
        if not ids:
            messagebox.showwarning("No morpheme", "没有符合过滤条件的语素。可以取消 stable only / has composition。")
            return
        self.current_parent_id = random.choice(ids)
        self.id_var.set(self.current_parent_id)
        self.random_rule()

    def random_rule(self):
        if not self.current_parent_id:
            return
        comps = self.comps_by_parent.get(self.current_parent_id, [])
        self.current_comp = random.choice(comps) if comps else None
        self.redraw()

    def open_id(self):
        mid = self.id_var.get().strip()
        if mid not in self.nodes:
            messagebox.showerror("Not found", f"morpheme_id not found: {mid}")
            return
        self.current_parent_id = mid
        self.random_rule()

    def redraw(self):
        if not self.current_parent_id:
            return
        parent = self.nodes.get(self.current_parent_id)
        if not parent:
            return

        jitter = self.jitter_style_var.get()
        parent_title = f"Cluster Parent {self.current_parent_id} | {parent.get('stroke_count')} strokes"
        draw_segments(self.parent_canvas, parent.get("prototype_segments", []),
                      title=parent_title, width=4, jitter=jitter,
                      default_color="black", label_color="blue", show_ids=True)

        comp = self.current_comp
        self.current_exact = False
        grouped = {"child_a": [], "child_b": [], "overlap": [], "other": []}

        if comp:
            exs = comp.get("examples") or []
            ex0 = exs[0] if exs else {}
            child_ids = list(comp.get("child_morpheme_ids", []))

            if ex0.get("parent_example_segments"):
                self.current_exact = True
                parent_example_segments = ex0.get("parent_example_segments", [])
                child_a_segments = ex0.get("child_a_example_segments", [])
                child_b_segments = ex0.get("child_b_example_segments", [])
                grouped = group_summary_from_segments(parent_example_segments)

                draw_segments(
                    self.parent_group_canvas,
                    parent_example_segments,
                    title="Example Parent grouped by child",
                    width=5,
                    jitter=jitter,
                    color_by_group=True,
                    label_color=COLOR_TEXT,
                    show_ids=True,
                    title_suffix="A=red B=blue other=gray; exact example geometry",
                )
                draw_segments(
                    self.child_a_canvas,
                    child_a_segments,
                    title=f"Example Child A {ex0.get('child_a_morpheme_id', child_ids[0] if child_ids else '?')}",
                    width=4,
                    jitter=jitter,
                    default_color=COLOR_CHILD_A,
                    label_color=COLOR_CHILD_A,
                    show_ids=True,
                )
                draw_segments(
                    self.child_b_canvas,
                    child_b_segments,
                    title=f"Example Child B {ex0.get('child_b_morpheme_id', child_ids[1] if len(child_ids)>1 else '?')}",
                    width=4,
                    jitter=jitter,
                    default_color=COLOR_CHILD_B,
                    label_color=COLOR_CHILD_B,
                    show_ids=True,
                )
            else:
                # fallback for old output_tree
                parent_segments = parent.get("prototype_segments", [])
                color_map, grouped = infer_parent_child_color_map(parent_segments, comp)
                ca = self.nodes.get(child_ids[0]) if len(child_ids) >= 1 else None
                cb = self.nodes.get(child_ids[1]) if len(child_ids) >= 2 else None

                draw_segments(
                    self.parent_group_canvas,
                    parent_segments,
                    title="Cluster Parent grouped by child",
                    width=5,
                    jitter=jitter,
                    default_color=COLOR_OTHER,
                    color_by_seg_id=color_map,
                    label_color=COLOR_TEXT,
                    show_ids=True,
                    title_suffix="fallback source_sid mapping; rerun builder for exact",
                )
                draw_segments(
                    self.child_a_canvas,
                    ca.get("prototype_segments", []) if ca else [],
                    title=f"Cluster Child A {child_ids[0] if child_ids else '?'}",
                    width=4,
                    jitter=jitter,
                    default_color=COLOR_CHILD_A,
                    label_color=COLOR_CHILD_A,
                    show_ids=True,
                )
                draw_segments(
                    self.child_b_canvas,
                    cb.get("prototype_segments", []) if cb else [],
                    title=f"Cluster Child B {child_ids[1] if len(child_ids)>1 else '?'}",
                    width=4,
                    jitter=jitter,
                    default_color=COLOR_CHILD_B,
                    label_color=COLOR_CHILD_B,
                    show_ids=True,
                )
        else:
            draw_segments(self.parent_group_canvas, [], title="No composition rule")
            draw_segments(self.child_a_canvas, [], title="No composition rule")
            draw_segments(self.child_b_canvas, [], title="No composition rule")

        self.current_grouped = grouped
        self._write_info()

    def _write_info(self):
        parent = self.nodes.get(self.current_parent_id, {})
        comp = self.current_comp
        grouped = self.current_grouped or {}

        lines = []
        lines.append("=== Parent Morpheme ===")
        lines.append(json.dumps({
            "morpheme_id": parent.get("morpheme_id"),
            "kind": parent.get("kind"),
            "stroke_count": parent.get("stroke_count"),
            "is_stable": parent.get("is_stable"),
            "support_count": parent.get("support_count"),
            "source_glyph_count": parent.get("source_glyph_count"),
            "relation_hist": parent.get("relation_hist"),
            "angle_class_hist": parent.get("angle_class_hist"),
            "angle_stats": parent.get("angle_stats"),
            "degree_hist_mode": parent.get("degree_hist_mode"),
            "cycle_rank_mode": parent.get("cycle_rank_mode"),
            "symmetry": parent.get("symmetry"),
            "composition_count": parent.get("composition_count"),
            "source_glyphs_head": (parent.get("source_glyphs") or [])[:10],
        }, ensure_ascii=False, indent=2))

        lines.append("\n=== Composition Rule ===")
        if comp:
            ex0 = (comp.get("examples") or [{}])[0]
            lines.append(json.dumps({
                "composition_id": comp.get("composition_id"),
                "parent_morpheme_id": comp.get("parent_morpheme_id"),
                "child_morpheme_ids_aggregate_sorted": comp.get("child_morpheme_ids"),
                "example_child_a_morpheme_id": ex0.get("child_a_morpheme_id"),
                "example_child_b_morpheme_id": ex0.get("child_b_morpheme_id"),
                "support_count": comp.get("support_count"),
                "source_glyph_count": comp.get("source_glyph_count"),
                "topology_way_hash": comp.get("topology_way_hash"),
                "topology_way": comp.get("topology_way"),
                "example_geometry_exact": self.current_exact,
                "example_head_without_geometry": {
                    k: v for k, v in ex0.items()
                    if k not in ("parent_example_segments", "child_a_example_segments", "child_b_example_segments")
                },
            }, ensure_ascii=False, indent=2))
        else:
            lines.append("No composition alternative for this morpheme.")

        other = grouped.get("other", [])
        lines.append("\n=== Parent Edge Coverage ===")
        lines.append(json.dumps({
            "exact_example_geometry": self.current_exact,
            "child_A_parent_segments": grouped.get("child_a", []),
            "child_B_parent_segments": grouped.get("child_b", []),
            "overlap_parent_segments": grouped.get("overlap", []),
            "other_parent_segments": other,
            "perfect_split": bool(self.current_exact and len(other) == 0 and len(grouped.get("overlap", [])) == 0),
            "legend": {
                "child_A": COLOR_CHILD_A,
                "child_B": COLOR_CHILD_B,
                "overlap": COLOR_OVERLAP,
                "other": COLOR_OTHER,
            },
            "note": "如果 exact_example_geometry=false，说明 output_tree 是旧 builder 生成的，复杂图会因 cluster prototype 与 composition example 不一致而出现灰线。请重新运行 EXAMPLE_GEOM 版 builder。",
        }, ensure_ascii=False, indent=2))

        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree-dir", default=DEFAULT_TREE_DIR)
    args = p.parse_args()

    root = tk.Tk()
    root.geometry("1280x900")
    MorphemePreviewApp(root, os.path.abspath(args.tree_dir))
    root.mainloop()


if __name__ == "__main__":
    main()
