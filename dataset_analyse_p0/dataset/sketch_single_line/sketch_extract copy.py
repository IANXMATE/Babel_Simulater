import os
import random
import glob
import json
import torch
import numpy as np
import networkx as nx

from PIL import Image, ImageDraw, ImageFont
from skimage import morphology
from skan import Skeleton, summarize
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt, map_coordinates

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# =========================
# CONFIG
# =========================
TARGET_DIR = "../alien_tensors_raw"
CANVAS_SIZE = 256
MAX_BEZIER_ERROR = 1.5


# =========================
# DEBUG VISUALIZATION
# =========================
def plot_show(
    target_img,
    binary,
    skeleton,
    paths,
    tokens
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(target_img, cmap="gray")
    axes[0, 0].set_title("Target")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(binary, cmap="gray")
    axes[0, 1].set_title("Binary")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(skeleton, cmap="gray")
    axes[0, 2].set_title("Skeleton")
    axes[0, 2].axis("off")

    # strokes
    vis = np.zeros((*binary.shape, 3))
    cmap = cm.get_cmap("tab10", max(1, len(paths)))

    for i, p in enumerate(paths):
        for pt in p:
            x, y = int(pt[1]), int(pt[0])
            if 0 <= x < binary.shape[0] and 0 <= y < binary.shape[1]:
                vis[x, y] = cmap(i)[:3]

    axes[1, 0].imshow(vis)
    axes[1, 0].set_title("Paths")
    axes[1, 0].axis("off")

    # bezier render
    recon = np.ones((*binary.shape, 3))

    for t in tokens:
        P = np.array(t["bezier"])
        color = cmap(0)[:3]

        ts = np.linspace(0, 1, 80)[:, None]
        mt = 1 - ts

        curve = (
            mt**3 * P[0]
            + 3 * mt**2 * ts * P[1]
            + 3 * mt * ts**2 * P[2]
            + ts**3 * P[3]
        )

        for p in curve.astype(int):
            x, y = int(p[0]), int(p[1])
            if 0 <= x < binary.shape[0] and 0 <= y < binary.shape[1]:
                recon[x, y] = color

    axes[1, 1].imshow(recon)
    axes[1, 1].set_title("Bezier")
    axes[1, 1].axis("off")

    heat = np.zeros(binary.shape)
    for t in tokens:
        P = np.array(t["bezier"])
        ts = np.linspace(0, 1, 60)[:, None]
        mt = 1 - ts

        curve = (
            mt**3 * P[0]
            + 3 * mt**2 * ts * P[1]
            + 3 * mt * ts**2 * P[2]
            + ts**3 * P[3]
        )

        for p in curve.astype(int):
            x, y = int(p[0]), int(p[1])
            if 0 <= x < heat.shape[0] and 0 <= y < heat.shape[1]:
                heat[x, y] += 1

    axes[1, 2].imshow(heat, cmap="hot")
    axes[1, 2].set_title("Token Heatmap")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# SAMPLE
# =========================
def get_random_glyph():
    font_files = glob.glob(os.path.join(TARGET_DIR, "*.[ot]tf"))
    font_path = random.choice(font_files)
    font = ImageFont.truetype(font_path, 180)

    chars = ["S", "之", "乙", "弓", "B", "W", "O", "口"]
    char = random.choice(chars)

    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(img)

    bbox = font.getbbox(char)

    draw.text(
        ((CANVAS_SIZE - (bbox[2]-bbox[0]))/2,
         (CANVAS_SIZE - (bbox[3]-bbox[1]))/2),
        char,
        font=font,
        fill=0
    )

    img = np.array(img).astype(np.float32) / 255.0
    binary = img < 0.5

    return img, binary, char


# =========================
# GRAPH
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
# CURVE SPLIT (SAFE)
# =========================
def split_curve(path, k=4):
    if len(path) < 10:
        return [path]

    segs = []
    start = 0

    for i in range(k, len(path) - k):
        v1 = path[i] - path[i-k]
        v2 = path[i+k] - path[i]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-6 or n2 < 1e-6:
            continue

        cos = np.clip(np.dot(v1, v2)/(n1*n2), -1, 1)
        angle = np.degrees(np.arccos(cos))

        if angle > 150:
            if i - start > 5:
                segs.append(path[start:i])
                start = i

    segs.append(path[start:])
    return [s for s in segs if len(s) > 3]


# =========================
# FIXED BEZIER (BUG FIX HERE)
# =========================
def cubic(P, t):
    t = np.asarray(t)
    if t.ndim == 1:
        t = t[:, None]

    P = np.asarray(P)

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
        pred = cubic(P, t)
        return (pred - path).reshape(-1)

    res = least_squares(loss, np.r_[P1, P2])

    return np.array([P0, res.x[:2], res.x[2:], P3])


# =========================
# WIDTH
# =========================
def estimate_width(P, dt_map):
    t = np.linspace(0, 1, 40)[:, None]
    curve = cubic(P, t)

    coords = np.vstack([curve[:,1], curve[:,0]])
    w = map_coordinates(dt_map, coords, order=1)

    return float(np.mean(w))


# =========================
# PIPELINE
# =========================
def build_sample():
    img, binary, char = get_random_glyph()
    dt = distance_transform_edt(binary)

    G, skel = build_graph(binary)
    paths = extract_paths(G)

    tokens = []

    for p in paths:
        segs = split_curve(p)

        for s in segs:
            P = fit_bezier(s)
            if P is None:
                continue

            w = estimate_width(P, dt)

            tokens.append({
                "char": char,
                "bezier": P.tolist(),
                "width": w
            })

    return img, binary, skel, paths, tokens


# =========================
# RUN
# =========================
if __name__ == "__main__":
    img, binary, skel, paths, tokens = build_sample()

    print("tokens:", len(tokens))
    print(json.dumps(tokens[0], indent=2, ensure_ascii=False))

    plot_show(img, binary, skel, paths, tokens)