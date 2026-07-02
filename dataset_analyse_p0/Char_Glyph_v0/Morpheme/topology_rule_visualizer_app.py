# -*- coding: utf-8 -*-
r"""
topology_rule_visualizer_app.py

放置位置：
    dataset_analyse_p0/Char_Glyph_v0/Morpheme/topology_rule_visualizer_app.py

运行：
    cd /Users/cuijiaxing03/BST/Babel_Simulater/dataset_analyse_p0/Char_Glyph_v0/Morpheme
    python topology_rule_visualizer_app.py

功能：
    1. 读取 output_tree 里的 morpheme_nodes / composition_alternatives。
    2. 读取 new_rule_cache/auto_rules_*.json，并合并 auto_pattern_xxx。
    3. 选择 rule，显示该 rule 命中的代表语素。
    4. 支持 top_score / weighted_random 两种模式。
    5. 支持“重新随机”。
    6. 支持批量导出当前筛选规则的 PNG contact sheet。
    7. 支持重新发现 auto rules 并写入 new_rule_cache。

说明：
    new_rule_cache 里缓存的是 auto_discovered rules。
    builtin / derived rules 不需要缓存，每次由 topology_pattern_diffusion_api_v2.py 构建。
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as e:
    raise RuntimeError("需要 tkinter。") from e

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except Exception as e:
    raise RuntimeError("需要 pillow：pip install pillow") from e


SCRIPT_DIR = Path(__file__).resolve().parent
TREE_DIR = SCRIPT_DIR / "output_tree"
CACHE_DIR = SCRIPT_DIR / "new_rule_cache"
EXPORT_DIR = CACHE_DIR / "rule_visual_exports"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import topology_pattern_diffusion_api_v2 as api
except Exception as e:
    raise RuntimeError(
        "无法导入 topology_pattern_diffusion_api_v2.py。\n"
        "请确认它和本脚本在同一个 Morpheme 目录下。\n"
        f"当前目录: {SCRIPT_DIR}\n"
        f"错误: {repr(e)}"
    )


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p: Path):
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def latest_auto_rule_cache_file() -> Optional[Path]:
    if not CACHE_DIR.exists():
        return None
    fs = sorted(CACHE_DIR.glob("auto_rules_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return fs[0] if fs else None


def load_latest_discovered_rules() -> Optional[Dict[str, Any]]:
    p = latest_auto_rule_cache_file()
    if p is None:
        return None
    d = load_json(p)
    d["_loaded_cache_path"] = str(p)
    return d


def get_font(size=12):
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT10 = get_font(10)
FONT11 = get_font(11)
FONT12 = get_font(12)
FONT14 = get_font(14)


def seg_pair(seg: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not isinstance(seg, dict):
        return None
    if "p0" in seg and "p1" in seg:
        try:
            p0 = np.asarray(seg["p0"], dtype=np.float32)
            p1 = np.asarray(seg["p1"], dtype=np.float32)
            if p0.shape == (2,) and p1.shape == (2,):
                return p0, p1
        except Exception:
            return None
    for k in ["mother_bezier", "bezier", "path"]:
        v = seg.get(k)
        if isinstance(v, list) and len(v) == 4:
            try:
                P = np.asarray(v, dtype=np.float32)
                if P.shape == (4, 2):
                    return P[0], P[3]
            except Exception:
                pass
    return None


def node_segments(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ["prototype_segments", "example_segments", "segments"]:
        v = node.get(k)
        if isinstance(v, list) and v:
            return v
    return []


def normalized_pairs(segments: List[Dict[str, Any]], w: int, h: int, margin=18):
    pairs, pts = [], []
    for s in segments:
        pp = seg_pair(s)
        if pp is None:
            continue
        p0, p1 = pp
        pairs.append((p0, p1))
        pts.extend([p0, p1])
    if not pts:
        return []
    P = np.stack(pts).astype(np.float32)
    mn, mx = P.min(axis=0), P.max(axis=0)
    center = (mn + mx) / 2
    size = max(float(mx[0] - mn[0]), float(mx[1] - mn[1]), 1e-6)
    scale = min(w - 2 * margin, h - 2 * margin) / size
    out = []
    for p0, p1 in pairs:
        q0 = (p0 - center) * scale + np.array([w / 2, h / 2], dtype=np.float32)
        q1 = (p1 - center) * scale + np.array([w / 2, h / 2], dtype=np.float32)
        out.append(((float(q0[0]), float(q0[1])), (float(q1[0]), float(q1[1]))))
    return out


def draw_card(node: Dict[str, Any], score: float, rule_id: str, size=(235, 185)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([1, 1, w - 2, h - 2], radius=8, outline=(200, 200, 200), width=1)

    mid = str(node.get("morpheme_id", "?"))
    sc = node.get("stroke_count", "?")
    sup = node.get("support_count", "?")
    glyphs = node.get("source_glyph_count", "?")
    stable = "stable" if node.get("is_stable") else "unstable"

    d.text((8, 6), f"{mid}  score={score:.3f}"[:38], fill=(30, 70, 160), font=FONT11)

    x0, y0, x1, y1 = 8, 28, w - 8, h - 42
    d.rectangle([x0, y0, x1, y1], fill=(250, 250, 250), outline=(230, 230, 230))

    area_w, area_h = x1 - x0, y1 - y0
    pairs = normalized_pairs(node_segments(node), area_w, area_h, margin=16)
    if pairs:
        for (a, b) in pairs:
            d.line([x0 + a[0], y0 + a[1], x0 + b[0], y0 + b[1]], fill=(20, 20, 20), width=4)
        for (a, b) in pairs:
            r = 3
            d.ellipse([x0 + a[0] - r, y0 + a[1] - r, x0 + a[0] + r, y0 + a[1] + r], fill=(190, 50, 50))
            d.ellipse([x0 + b[0] - r, y0 + b[1] - r, x0 + b[0] + r, y0 + b[1] + r], fill=(50, 130, 70))
    else:
        d.text((w / 2 - 34, h / 2 - 10), "no segments", fill=(150, 150, 150), font=FONT12)

    d.text((8, h - 34), f"{stable} | strokes={sc} | support={sup} | glyphs={glyphs}"[:45], fill=(70, 70, 70), font=FONT10)
    d.text((8, h - 18), f"rule: {rule_id}"[:45], fill=(110, 110, 110), font=FONT10)
    return img


def contact_sheet(samples: List[Tuple[Dict[str, Any], float]], rule_id: str, cols=4, card_size=(235, 185)) -> Image.Image:
    cols = max(1, int(cols))
    rows = max(1, int(np.ceil(len(samples) / cols)))
    cw, ch = card_size
    pad, title_h = 12, 52
    W = cols * cw + (cols + 1) * pad
    H = title_h + rows * ch + (rows + 1) * pad
    img = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.text((pad, 12), f"Rule: {rule_id}", fill=(20, 20, 20), font=FONT14)
    d.text((pad, 32), f"samples={len(samples)}", fill=(90, 90, 90), font=FONT11)
    for i, (node, score) in enumerate(samples):
        r, c = divmod(i, cols)
        x = pad + c * (cw + pad)
        y = title_h + pad + r * (ch + pad)
        img.paste(draw_card(node, score, rule_id, size=card_size), (x, y))
    return img


class RuleVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Topology Rule Visualizer - new_rule_cache / auto_pattern viewer")
        self.root.geometry("1500x920")

        self.nodes = []
        self.comps = []
        self.node_by_id = {}
        self.registry = {}
        self.discovered = None
        self.scored = {}
        self.visible_rule_ids = []
        self.current_rule_id = None
        self.current_samples = []
        self.photos = []
        self.random_seed = int(time.time()) % 1000000

        self.build_ui()
        self.load_all(initial=True)

    def build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(main, width=330)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Rule source").pack(anchor="w")
        self.source_var = tk.StringVar(value="all")
        cb = ttk.Combobox(left, textvariable=self.source_var, values=["all", "builtin", "derived_manual", "auto_discovered"], state="readonly")
        cb.pack(fill=tk.X, pady=(0, 4))
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_rule_list())

        ttk.Label(left, text="Search").pack(anchor="w")
        self.search_var = tk.StringVar(value="")
        ent = ttk.Entry(left, textvariable=self.search_var)
        ent.pack(fill=tk.X, pady=(0, 4))
        ent.bind("<KeyRelease>", lambda e: self.refresh_rule_list())

        self.rule_list = tk.Listbox(left, height=34, exportselection=False)
        self.rule_list.pack(fill=tk.BOTH, expand=True)
        self.rule_list.bind("<<ListboxSelect>>", self.on_rule_select)

        f1 = ttk.Frame(left)
        f1.pack(fill=tk.X, pady=5)
        ttk.Button(f1, text="Reload", command=self.load_all).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(f1, text="Rediscover auto", command=self.rediscover).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        f2 = ttk.Frame(left)
        f2.pack(fill=tk.X, pady=2)
        ttk.Button(f2, text="Clear cache", command=self.clear_cache).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(f2, text="Batch export", command=self.batch_export).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        top = ttk.Frame(right)
        top.pack(fill=tk.X)

        self.mode_var = tk.StringVar(value="weighted_random")
        ttk.Label(top, text="Mode").pack(side=tk.LEFT)
        ttk.Combobox(top, textvariable=self.mode_var, values=["top_score", "weighted_random"], state="readonly", width=16).pack(side=tk.LEFT, padx=4)

        self.n_var = tk.IntVar(value=24)
        ttk.Label(top, text="samples").pack(side=tk.LEFT)
        ttk.Spinbox(top, from_=1, to=96, textvariable=self.n_var, width=6).pack(side=tk.LEFT, padx=4)

        self.min_score_var = tk.DoubleVar(value=0.18)
        ttk.Label(top, text="min score").pack(side=tk.LEFT)
        ttk.Spinbox(top, from_=0.0, to=1.0, increment=0.02, textvariable=self.min_score_var, width=7).pack(side=tk.LEFT, padx=4)

        self.stable_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="stable only", variable=self.stable_var, command=self.refresh_current).pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="重新随机", command=self.rerandom).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Export current PNG", command=self.export_current).pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.status_var).pack(fill=tk.X, pady=5)

        self.detail = tk.Text(right, height=10, wrap="word")
        self.detail.pack(fill=tk.X, pady=(0, 6))

        frame = ttk.Frame(right)
        frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(frame, bg="#eeeeee")
        self.vs = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.hs = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vs.set, xscrollcommand=self.hs.set)
        self.vs.pack(side=tk.RIGHT, fill=tk.Y)
        self.hs.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.grid = ttk.Frame(self.canvas)
        self.win = self.canvas.create_window((0, 0), window=self.grid, anchor="nw")
        self.grid.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.win, width=max(e.width, 1000)))

    def status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def load_all(self, initial=False):
        try:
            self.status("Loading output_tree ...")
            self.nodes, self.comps = api.load_morpheme_tree(str(TREE_DIR))
            self.node_by_id = {n.get("morpheme_id"): n for n in self.nodes if n.get("morpheme_id")}

            base = api.derive_builtin_rules(api.build_default_rule_registry())
            self.discovered = load_latest_discovered_rules()
            if self.discovered and self.discovered.get("rules"):
                self.registry = api.merge_auto_rules(base, self.discovered)
            else:
                self.registry = base

            self.status("Scoring morphemes ...")
            self.scored = api.score_morphemes_by_rules(self.nodes, self.registry)

            self.refresh_rule_list()
            cache_status = api.get_new_rule_cache_status(str(CACHE_DIR))
            cache_path = self.discovered.get("_loaded_cache_path") if self.discovered else None
            self.status(
                f"nodes={len(self.nodes)} comps={len(self.comps)} rules={len(self.registry.get('rules', {}))} "
                f"cache={cache_status.get('size_mb')}MB file={cache_path or 'None'}"
            )

            if initial and self.rule_list.size() > 0:
                self.rule_list.selection_set(0)
                self.on_rule_select()
        except Exception:
            traceback.print_exc()
            messagebox.showerror("Load failed", traceback.format_exc())

    def refresh_rule_list(self):
        self.rule_list.delete(0, tk.END)
        source = self.source_var.get()
        q = self.search_var.get().strip().lower()
        rows = []
        for rid, rule in self.registry.get("rules", {}).items():
            src = str(rule.get("source", ""))
            if source != "all":
                if source == "derived_manual":
                    if src not in ("derived_manual", "derived", "manual_derived"):
                        continue
                elif src != source:
                    continue
            text = f"{rid} [{src}]"
            blob = text + " " + str(rule.get("description", "")) + " " + str(rule.get("aliases", ""))
            if q and q not in blob.lower():
                continue
            rows.append((rid, text))
        rows.sort(key=lambda x: (self.registry["rules"][x[0]].get("source", ""), x[0]))
        self.visible_rule_ids = [x[0] for x in rows]
        for _, text in rows:
            self.rule_list.insert(tk.END, text)

    def on_rule_select(self, event=None):
        sel = self.rule_list.curselection()
        if not sel:
            return
        i = int(sel[0])
        if i >= len(self.visible_rule_ids):
            return
        self.current_rule_id = self.visible_rule_ids[i]
        self.refresh_current()

    def candidates_for_rule(self, rid):
        out = []
        mn = float(self.min_score_var.get())
        stable_only = bool(self.stable_var.get())
        for mid, pack in self.scored.items():
            node = self.node_by_id.get(mid)
            if node is None:
                continue
            if stable_only and not node.get("is_stable"):
                continue
            if mid == "M_LINE" and rid != "line":
                continue
            s = safe_float((pack.get("scores") or {}).get(rid), 0.0)
            if s >= mn:
                out.append((node, s))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def select_samples(self, rid):
        cands = self.candidates_for_rule(rid)
        n = max(1, int(self.n_var.get()))
        if not cands:
            return []
        if self.mode_var.get() == "top_score":
            return cands[:n]
        rng = random.Random(self.random_seed + (abs(hash(rid)) % 1000000))
        weights = np.asarray([max(1e-6, s) ** 2 for _, s in cands], dtype=np.float64)
        weights = weights / weights.sum()
        count = min(n, len(cands))
        idxs = rng.choices(range(len(cands)), weights=weights.tolist(), k=count * 3)
        seen, out = set(), []
        for i in idxs:
            mid = cands[i][0].get("morpheme_id")
            if mid in seen:
                continue
            seen.add(mid)
            out.append(cands[i])
            if len(out) >= count:
                break
        if len(out) < count:
            for x in cands:
                mid = x[0].get("morpheme_id")
                if mid not in seen:
                    out.append(x)
                    seen.add(mid)
                    if len(out) >= count:
                        break
        return out

    def refresh_current(self):
        if not self.current_rule_id:
            return
        rid = self.current_rule_id
        self.current_samples = self.select_samples(rid)
        self.render_detail(rid)
        self.render_grid(rid, self.current_samples)

    def rerandom(self):
        self.random_seed = random.randint(0, 999999999)
        self.refresh_current()

    def render_detail(self, rid):
        rule = self.registry.get("rules", {}).get(rid, {})
        cands = self.candidates_for_rule(rid)
        info = {
            "rule_id": rid,
            "source": rule.get("source"),
            "description": rule.get("description"),
            "aliases": rule.get("aliases"),
            "parent_rules": rule.get("parent_rules"),
            "term_count": len(rule.get("terms") or []),
            "candidate_count_above_threshold": len(cands),
            "shown_count": len(self.current_samples),
            "terms": rule.get("terms"),
        }
        if rid.startswith("auto_pattern_") and self.discovered:
            for c in self.discovered.get("clusters", []):
                if c.get("auto_pattern_id") == rid:
                    info["auto_cluster_size"] = c.get("size")
                    info["auto_label"] = c.get("auto_label")
                    info["auto_cluster_samples"] = [s.get("morpheme_id") for s in c.get("samples", [])[:16]]
                    info["center_rule_scores_top"] = sorted((c.get("center_rule_scores") or {}).items(), key=lambda kv: kv[1], reverse=True)[:12]
                    break
        self.detail.delete("1.0", tk.END)
        self.detail.insert(tk.END, json.dumps(info, ensure_ascii=False, indent=2))

    def render_grid(self, rid, samples):
        for w in self.grid.winfo_children():
            w.destroy()
        self.photos.clear()
        if not samples:
            ttk.Label(self.grid, text="No matched morphemes. Try lower min score or disable stable only.").grid(row=0, column=0, padx=20, pady=20)
            return
        cols = 4
        for i, (node, score) in enumerate(samples):
            img = draw_card(node, score, rid, size=(235, 185))
            ph = ImageTk.PhotoImage(img)
            self.photos.append(ph)
            lab = ttk.Label(self.grid, image=ph)
            lab.grid(row=i // cols, column=i % cols, padx=8, pady=8, sticky="n")
            lab.bind("<Button-1>", lambda e, n=node, s=score: self.popup_node(n, s))

    def popup_node(self, node, score):
        win = tk.Toplevel(self.root)
        win.title(str(node.get("morpheme_id", "?")))
        win.geometry("660x600")
        img = draw_card(node, score, self.current_rule_id or "", size=(600, 430))
        ph = ImageTk.PhotoImage(img)
        win._photo = ph
        ttk.Label(win, image=ph).pack(padx=10, pady=10)
        t = tk.Text(win, height=10, wrap="word")
        t.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        mid = node.get("morpheme_id")
        pack = self.scored.get(mid, {})
        info = {
            "morpheme_id": mid,
            "score_for_current_rule": score,
            "stroke_count": node.get("stroke_count"),
            "support_count": node.get("support_count"),
            "source_glyph_count": node.get("source_glyph_count"),
            "is_stable": node.get("is_stable"),
            "top_scores": sorted((pack.get("scores") or {}).items(), key=lambda kv: kv[1], reverse=True)[:25],
            "features": pack.get("features", {}),
        }
        t.insert(tk.END, json.dumps(info, ensure_ascii=False, indent=2))

    def clear_cache(self):
        try:
            ret = api.clear_new_rule_cache(str(CACHE_DIR))
            messagebox.showinfo("Cache cleared", json.dumps(ret, ensure_ascii=False, indent=2))
            self.load_all()
        except Exception:
            messagebox.showerror("Clear failed", traceback.format_exc())

    def rediscover(self):
        try:
            self.status("Rediscovering auto rules ...")
            base = api.derive_builtin_rules(api.build_default_rule_registry())
            discovered = api.discover_new_pattern_rules_cached(
                self.nodes,
                base,
                k=16,
                stable_only=True,
                min_support=1,
                min_source_glyphs=1,
                use_cache=True,
                force_rebuild=True,
                cache_dir=str(CACHE_DIR),
                cache_limit_mb=90,
                delete_cache_if_over_limit=True,
            )
            self.discovered = discovered
            self.registry = api.merge_auto_rules(base, discovered)
            self.scored = api.score_morphemes_by_rules(self.nodes, self.registry)
            self.refresh_rule_list()
            messagebox.showinfo("Rediscover done", f"auto_rules={len(discovered.get('rules', {}))}\ncache={json.dumps(discovered.get('_cache', {}), ensure_ascii=False, indent=2)}")
            self.status("Rediscover done.")
        except Exception:
            traceback.print_exc()
            messagebox.showerror("Rediscover failed", traceback.format_exc())

    def export_current(self):
        if not self.current_rule_id:
            return
        try:
            ensure_dir(EXPORT_DIR)
            rid = self.current_rule_id
            samples = self.current_samples or self.select_samples(rid)
            img = contact_sheet(samples, rid, cols=4)
            out = EXPORT_DIR / f"rule_{rid}_{time.strftime('%Y%m%d_%H%M%S')}.png"
            img.save(out)
            messagebox.showinfo("Export done", f"saved:\n{out}")
        except Exception:
            messagebox.showerror("Export failed", traceback.format_exc())

    def batch_export(self):
        try:
            ensure_dir(EXPORT_DIR)
            out_dir = EXPORT_DIR / f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
            ensure_dir(out_dir)
            summary = []
            for i, rid in enumerate(self.visible_rule_ids):
                self.status(f"Exporting {i + 1}/{len(self.visible_rule_ids)}: {rid}")
                samples = self.select_samples(rid)
                if not samples:
                    summary.append({"rule_id": rid, "sample_count": 0, "png": None})
                    continue
                img = contact_sheet(samples, rid, cols=4)
                png = out_dir / f"{i:03d}_{rid}.png"
                img.save(png)
                summary.append({
                    "rule_id": rid,
                    "source": self.registry.get("rules", {}).get(rid, {}).get("source"),
                    "sample_count": len(samples),
                    "png": str(png),
                    "top_morphemes": [
                        {
                            "morpheme_id": n.get("morpheme_id"),
                            "score": float(s),
                            "stroke_count": n.get("stroke_count"),
                            "support_count": n.get("support_count"),
                            "source_glyph_count": n.get("source_glyph_count"),
                            "is_stable": n.get("is_stable"),
                        }
                        for n, s in samples
                    ],
                })
            save_json({
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rule_count": len(self.visible_rule_ids),
                "mode": self.mode_var.get(),
                "sample_per_rule": int(self.n_var.get()),
                "min_score": float(self.min_score_var.get()),
                "stable_only": bool(self.stable_var.get()),
                "summary": summary,
            }, out_dir / "batch_summary.json")
            self.status(f"Batch export done: {out_dir}")
            messagebox.showinfo("Batch export done", f"saved:\n{out_dir}")
        except Exception:
            traceback.print_exc()
            messagebox.showerror("Batch export failed", traceback.format_exc())


def main():
    root = tk.Tk()
    RuleVisualizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
