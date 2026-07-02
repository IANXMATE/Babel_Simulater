# -*- coding: utf-8 -*-
r"""
morpheme_preview_app.py

放置：
    dataset_analyse_p0/Char_Glyph_v0/Morpheme/morpheme_preview_app.py

作用：
    从 output_tree 读取拓扑语素库，随机预览一个语素：
      - parent morpheme 的拓扑母线样子
      - 能组成它的 child morphemes
      - child -> parent 的 topology_way / composition rule
      - 支持 stable only / has composition only / stroke_count 过滤
      - 可随机 parent，也可输入 morpheme_id 查看指定语素

运行：
    cd dataset_analyse_p0/Char_Glyph_v0/Morpheme
    python morpheme_preview_app.py

可选：
    python morpheme_preview_app.py --tree-dir output_tree
"""

from __future__ import annotations

import os
import json
import random
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from collections import defaultdict, Counter
import math


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TREE_DIR = os.path.join(SCRIPT_DIR, "output_tree")


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

    # fallback
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


def bbox_of_segments(segments):
    pts = []
    for s in segments:
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


def draw_segments(canvas, segments, title="", width=3, jitter=False):
    canvas.delete("all")
    W = int(canvas.winfo_width() or 300)
    H = int(canvas.winfo_height() or 220)
    canvas.create_text(10, 10, anchor="nw", text=title, font=("Arial", 12, "bold"))

    if not segments:
        canvas.create_text(W/2, H/2, text="No prototype segments", fill="gray")
        return

    bbox = bbox_of_segments(segments)
    rng = random.Random(1234 if not jitter else random.randint(0, 999999))

    for i, s in enumerate(segments):
        p0 = s.get("p0", [0, 0])
        p1 = s.get("p1", [1, 0])
        x0, y0 = safe_float(p0[0]), safe_float(p0[1])
        x1, y1 = safe_float(p1[0]), safe_float(p1[1])

        if jitter:
            # 只是 preview 的随机样式，不改变语素数据。
            dx0, dy0 = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
            dx1, dy1 = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
            x0, y0 = x0 + dx0, y0 + dy0
            x1, y1 = x1 + dx1, y1 + dy1

        px0, py0 = transform_point(x0, y0, bbox, W, H)
        px1, py1 = transform_point(x1, y1, bbox, W, H)
        canvas.create_line(px0, py0, px1, py1, width=width, capstyle="round")
        mx, my = (px0 + px1) / 2, (py0 + py1) / 2
        canvas.create_text(mx, my, text=str(s.get("source_sid", s.get("local_id", i+1))), fill="blue", font=("Arial", 9))

    # bbox outline
    x0, y0 = transform_point(bbox[0], bbox[1], bbox, W, H)
    x1, y1 = transform_point(bbox[2], bbox[3], bbox, W, H)
    canvas.create_rectangle(min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1), outline="#dddddd")


class MorphemePreviewApp:
    def __init__(self, root, tree_dir):
        self.root = root
        self.tree_dir = tree_dir
        self.root.title("Topology Morpheme Preview")

        self.nodes = {}
        self.comps_by_parent = defaultdict(list)
        self.all_comps = []
        self.current_parent_id = None
        self.current_comp = None

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

        ttk.Label(top, text=f"tree_dir: {self.tree_dir}").grid(row=0, column=0, columnspan=8, sticky="w")

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

        self.parent_canvas = tk.Canvas(main, width=420, height=300, bg="white")
        self.child_a_canvas = tk.Canvas(main, width=300, height=220, bg="white")
        self.child_b_canvas = tk.Canvas(main, width=300, height=220, bg="white")

        self.parent_canvas.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=4, pady=4)
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
        title = f"Parent {self.current_parent_id} | {parent.get('stroke_count')} strokes"
        draw_segments(self.parent_canvas, parent.get("prototype_segments", []), title=title, width=4, jitter=jitter)

        comp = self.current_comp
        if comp:
            child_ids = list(comp.get("child_morpheme_ids", []))
            ca = self.nodes.get(child_ids[0]) if len(child_ids) >= 1 else None
            cb = self.nodes.get(child_ids[1]) if len(child_ids) >= 2 else None
            draw_segments(self.child_a_canvas, ca.get("prototype_segments", []) if ca else [], title=f"Child A {child_ids[0] if child_ids else '?'}", width=3, jitter=jitter)
            draw_segments(self.child_b_canvas, cb.get("prototype_segments", []) if cb else [], title=f"Child B {child_ids[1] if len(child_ids)>1 else '?'}", width=3, jitter=jitter)
        else:
            draw_segments(self.child_a_canvas, [], title="No composition rule")
            draw_segments(self.child_b_canvas, [], title="No composition rule")

        self._write_info()

    def _write_info(self):
        parent = self.nodes.get(self.current_parent_id, {})
        comp = self.current_comp

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

        lines.append("\\n=== Composition Rule ===")
        if comp:
            lines.append(json.dumps({
                "composition_id": comp.get("composition_id"),
                "parent_morpheme_id": comp.get("parent_morpheme_id"),
                "child_morpheme_ids": comp.get("child_morpheme_ids"),
                "support_count": comp.get("support_count"),
                "source_glyph_count": comp.get("source_glyph_count"),
                "topology_way_hash": comp.get("topology_way_hash"),
                "topology_way": comp.get("topology_way"),
                "examples": comp.get("examples", [])[:5],
            }, ensure_ascii=False, indent=2))
        else:
            lines.append("No composition alternative for this morpheme.")

        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "\\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree-dir", default=DEFAULT_TREE_DIR)
    args = p.parse_args()

    root = tk.Tk()
    root.geometry("1100x780")
    app = MorphemePreviewApp(root, os.path.abspath(args.tree_dir))
    root.mainloop()


if __name__ == "__main__":
    main()
