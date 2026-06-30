import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image, ImageDraw, ImageFont
from skimage import morphology
from skan import Skeleton, summarize
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt, map_coordinates


# =========================
# CONFIG
# =========================
CANVAS_SIZE = 256


# =========================
# 1. Glyph Render
# =========================
def render_unicode_glyph(font_path, unicode_hex, size=256):
    if unicode_hex.startswith("U+"):
        unicode_hex = unicode_hex[2:]

    char = chr(int(unicode_hex, 16))
    font = ImageFont.truetype(font_path, int(size * 0.8))

    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    left, top, right, bottom = font.getbbox(char)

    w, h = right - left, bottom - top

    x = (size - w) / 2 - left
    y = (size - h) / 2 - top

    draw.text((x, y), char, font=font, fill=0)

    arr = np.array(img).astype(np.float32) / 255.0
    binary = arr < 0.5

    return arr, binary, char


# =========================
# 2. Skeleton Graph
# =========================
def build_graph(binary):
    skel = morphology.skeletonize(binary)
    obj = Skeleton(skel)
    branch = summarize(obj)

    G = nx.Graph()

    for idx, row in branch.iterrows():
        coords = obj.path_coordinates(idx)
        path = np.column_stack([coords[:, 1], coords[:, 0]])

        src, dst = int(row["node-id-src"]), int(row["node-id-dst"])
        G.add_edge(src, dst, path=path)

    return G, skel


def extract_paths(G):
    return [d["path"] for _, _, d in G.edges(data=True)]


# =========================
# 3. Split Stroke
# =========================
def smooth_path(path, w=2):
    if len(path) < 2 * w + 1:
        return path

    return np.array([
        path[i-w:i+w+1].mean(axis=0)
        for i in range(w, len(path)-w)
    ])


def is_collinear(p0, p1, p2, eps=0.01):
    v1 = p1 - p0
    v2 = p2 - p1

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return True

    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return abs(1 - cos) < eps


def split_curve(path, k=5):
    path = smooth_path(path)

    if len(path) < 12:
        return [path]

    segs = []
    start = 0

    for i in range(k, len(path) - k):

        p0 = path[i - k]
        p1 = path[i]
        p2 = path[i + k]

        if is_collinear(p0, p1, p2):
            continue

        window = path[i-k:i+k]
        cov = np.cov(window.T)
        eig = np.sort(np.linalg.eigvals(cov))
        linearity = eig[0] / (eig[1] + 1e-6)

        if linearity < 0.03:
            continue

        v1 = p1 - p0
        v2 = p2 - p1

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-6 or n2 < 1e-6:
            continue

        cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angle = np.degrees(np.arccos(cos))

        if angle > 165:
            if i - start > 8:
                segs.append(path[start:i])
                start = i

    segs.append(path[start:])

    return [s for s in segs if len(s) > 5]


# =========================
# 4. Bezier Fit
# =========================
def cubic(P, t):
    t = np.asarray(t)
    if t.ndim == 1:
        t = t[:, None]

    mt = 1 - t
    return (
        mt**3 * P[0]
        + 3 * mt**2 * t * P[1]
        + 3 * mt * t**2 * P[2]
        + t**3 * P[3]
    )


def fit_bezier(path):
    path = np.asarray(path, dtype=np.float32)

    if len(path) < 2:
        return None

    t = np.linspace(0, 1, len(path))[:, None]

    P0, P3 = path[0], path[-1]
    P1 = P0 + (P3 - P0) * 0.33
    P2 = P0 + (P3 - P0) * 0.66

    def loss(x):
        P = np.array([P0, x[:2], x[2:], P3])
        return (cubic(P, t) - path).reshape(-1)

    res = least_squares(loss, np.r_[P1, P2])

    return np.array([P0, res.x[:2], res.x[2:], P3])


# =========================
# 5. Width
# =========================
def estimate_width(P, dt_map):
    t = np.linspace(0, 1, 40)[:, None]
    curve = cubic(P, t)

    coords = np.vstack([curve[:, 1], curve[:, 0]])
    w = map_coordinates(dt_map, coords, order=1)

    return float(np.mean(w))


# =========================
# 6. LOOP EXTRACTION (🔥 FIX CORE)
# =========================
def extract_loops(G):
    cycles = nx.cycle_basis(G)
    loops = []

    for cycle in cycles:
        pts = []

        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]

            if G.has_edge(u, v):
                pts.append(G[u][v]["path"])

        if len(pts) == 0:
            continue

        pts = np.concatenate(pts, axis=0)

        if len(pts) < 10:
            continue

        # close loop explicitly
        pts = np.vstack([pts, pts[0]])

        loops.append(pts)

    return loops


# =========================
# 7. FULL PIPELINE
# =========================
def build_sample(font_path, unicode_hex):
    img, binary, char = render_unicode_glyph(font_path, unicode_hex)

    dt = distance_transform_edt(binary)

    G, skel = build_graph(binary)

    paths = extract_paths(G)
    loops = extract_loops(G)

    tokens = []

    # -------- open strokes
    for p in paths:
        segments = split_curve(p)

        for s in segments:
            P = fit_bezier(s)
            if P is None:
                continue

            w = estimate_width(P, dt)

            tokens.append({
                "type": "stroke",
                "char": char,
                "unicode": unicode_hex,
                "bezier": P,
                "width": w,
                "length": len(s)
            })

    # -------- loop strokes (FIXED)
    for lp in loops:
        # 核心修复：单根贝塞尔无法画圆，必须将其强制切分为 4 段
        num_splits = 4
        n = len(lp)
        step = n // num_splits
        
        for i in range(num_splits):
            start_idx = i * step
            # 关键：每段的结尾要包含下一段的开头（+1），保证切分后首尾相连不断带
            end_idx = (i + 1) * step + 1 if i < num_splits - 1 else n
            
            segment = lp[start_idx:end_idx]
            
            if len(segment) < 4:
                continue
                
            P = fit_bezier(segment)
            if P is None:
                continue

            w = estimate_width(P, dt)

            tokens.append({
                "type": "loop",
                "char": char,
                "unicode": unicode_hex,
                "bezier": P,
                "width": w,
                "length": len(segment)
            })

    return img, binary, skel, paths, tokens, dt


# =========================
# 8. VISUALIZATION
# =========================
def draw_bezier(canvas, P, width, color):
    ts = np.linspace(0, 1, 80)[:, None]
    mt = 1 - ts

    curve = (
        mt**3 * P[0]
        + 3 * mt**2 * ts * P[1]
        + 3 * mt * ts**2 * P[2]
        + ts**3 * P[3]
    )

    r = max(1, int(width * 1.2))

    for x, y in curve.astype(int):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < canvas.shape[0] and 0 <= yy < canvas.shape[1]:
                    canvas[yy, xx] = color


def plot_show(img, binary, skel, paths, tokens, dt_map):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Glyph")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(binary, cmap="gray")
    axes[0, 1].set_title("Binary")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(skel, cmap="gray")
    axes[0, 2].set_title("Skeleton")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(dt_map, cmap="magma")
    axes[1, 0].set_title("DT")
    axes[1, 0].axis("off")

    cmap = cm.get_cmap("tab10", 10)

    recon = np.ones((*binary.shape, 3))

    for i, t in enumerate(tokens):
        P = t["bezier"]
        w = t["width"]

        color = cmap(i % 10)[:3]

        # loop tokens slightly different color
        if t["type"] == "loop":
            color = np.array([1.0, 0.3, 0.3])

        draw_bezier(recon, P, w, color)

    axes[1, 1].imshow(recon)
    axes[1, 1].set_title("Bezier Tokens (Stroke + Loop)")
    axes[1, 1].axis("off")

    heat = np.zeros(binary.shape)

    for t in tokens:
        P = t["bezier"]
        w = t["width"]

        ts = np.linspace(0, 1, 60)[:, None]
        mt = 1 - ts

        curve = (
            mt**3 * P[0]
            + 3 * mt**2 * ts * P[1]
            + 3 * mt * ts**2 * P[2]
            + ts**3 * P[3]
        )

        for x, y in curve.astype(int):
            if 0 <= x < heat.shape[0] and 0 <= y < heat.shape[1]:
                heat[y, x] += w

    axes[1, 2].imshow(heat, cmap="hot")
    axes[1, 2].set_title("Width Heatmap")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    font_path = "../alien_tensors_raw/NotoSansTamil[wdth,wght].ttf"
    unicode_hex = "U+0BB4"

    img, binary, skel, paths, tokens, dt = build_sample(font_path, unicode_hex)

    print("tokens:", len(tokens))
    print(tokens[0])

    plot_show(img, binary, skel, paths, tokens, dt)