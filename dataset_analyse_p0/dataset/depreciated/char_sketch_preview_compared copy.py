import os, glob, random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.basePen import BasePen
from scipy.spatial import Voronoi, KDTree

# =========================
# Adaptive Bezier Flatten
# =========================
def adaptive_quad(p0, p1, p2, tol=1.5):
    chord = np.linalg.norm(np.array(p2)-np.array(p0))
    cont = np.linalg.norm(np.array(p1)-np.array(p0)) + np.linalg.norm(np.array(p2)-np.array(p1))
    if abs(cont - chord) < tol:
        return [p2]
    p01 = ((p0[0]+p1[0])/2,(p0[1]+p1[1])/2)
    p12 = ((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)
    p012 = ((p01[0]+p12[0])/2,(p01[1]+p12[1])/2)
    return adaptive_quad(p0,p01,p012,tol) + adaptive_quad(p012,p12,p2,tol)

class FlattenPen(BasePen):
    def __init__(self,glyphSet,density=15):
        super().__init__(glyphSet)
        self.polylines=[]
        self.curr=[]
        self.start=(0,0)
        self.density=density
    def _moveTo(self,p):
        if self.curr: self.polylines.append(self.curr)
        self.curr=[p]
        self.start=p
    def _lineTo(self,p):
        self.curr.append(p)
    def _qCurveToOne(self,p1,p2):
        pts=adaptive_quad(self.curr[-1],p1,p2)
        self.curr.extend(pts)
    def _curveToOne(self,p1,p2,p3):
        p0=self.curr[-1]
        ts=np.linspace(0,1,self.density)
        for t in ts[1:]:
            x=(1-t)**3*p0[0]+3*(1-t)**2*t*p1[0]+3*(1-t)*t**2*p2[0]+t**3*p3[0]
            y=(1-t)**3*p0[1]+3*(1-t)**2*t*p1[1]+3*(1-t)*t**2*p2[1]+t**3*p3[1]
            self.curr.append((x,y))
    def _closePath(self):
        if self.curr and self.curr[-1]!=self.start:
            self.curr.append(self.start)
        if self.curr:
            self.polylines.append(self.curr)
        self.curr=[]

# =========================
# Even-Odd 内部检测
# =========================
def is_inside(x,y,polylines):
    inside=False
    for poly in polylines:
        n=len(poly)
        for i in range(n):
            x1,y1=poly[i]
            x2,y2=poly[(i+1)%n]
            if ((y1>y) != (y2>y)):
                inter=(x2-x1)*(y-y1)/(y2-y1+1e-9)+x1
                if x<inter: inside=not inside
    return inside

# =========================
# Voronoi Skeleton + Prune
# =========================
def voronoi_skeleton(polylines):
    points=[p for poly in polylines for p in poly]
    if len(points)<4: return nx.Graph()
    vor=Voronoi(points)
    G=nx.Graph()
    bbox=[min(p[0] for p in points),max(p[0] for p in points),
          min(p[1] for p in points),max(p[1] for p in points)]
    size=max(bbox[1]-bbox[0],bbox[3]-bbox[2])
    valid=set(i for i,v in enumerate(vor.vertices) if is_inside(v[0],v[1],polylines))
    # 去重 Voronoi 顶点
    vert_map = {tuple(v):i for i,v in enumerate(vor.vertices)}
    for ridge in vor.ridge_vertices:
        if ridge[0]!=-1 and ridge[1]!=-1:
            if ridge[0] in valid and ridge[1] in valid:
                p1,p2=vor.vertices[ridge[0]],vor.vertices[ridge[1]]
                dist=np.linalg.norm(p1-p2)
                if dist>size*0.5: continue
                G.add_edge(tuple(p1),tuple(p2),weight=dist)
    # prune
    changed=True
    while changed:
        changed=False
        for n in list(G.nodes):
            if G.degree(n)==1:
                neighbor=list(G.neighbors(n))[0]
                if G[n][neighbor]['weight']<size*0.02:
                    G.remove_node(n)
                    changed=True
    return G

# =========================
# Edge DFS + Junctions + Width
# =========================
def decompose_graph_with_width(G, boundary_points):
    tree=KDTree(boundary_points)
    junctions=list({n for n,d in G.degree() if d>2})
    Gb=G.copy()
    for j in junctions:
        for n in list(Gb.neighbors(j)):
            Gb.remove_edge(j,n)
    segments=[]
    tensor=[]
    for comp in nx.connected_components(Gb):
        if len(comp)<2: continue
        sub=Gb.subgraph(comp)
        edges=list(nx.edge_dfs(sub))
        if not edges: continue
        path=[edges[0][0]]
        for u,v in edges: path.append(v)
        segments.append(path)
        # 计算局部笔画宽度
        for pt in path:
            dist,_=tree.query(pt)
            tensor.append((pt[0],pt[1],dist))
    return segments,junctions,tensor

# =========================
# 主循环
# =========================
TARGET_DIR="alien_tensors_raw"
files=glob.glob(os.path.join(TARGET_DIR,"*.[ot]tf"))
sample=random.sample(files,min(5,len(files)))

plt.style.use('dark_background')
fig, axes=plt.subplots(len(sample),2,figsize=(8,1.5*len(sample)))
if len(sample)==1: axes=np.array([axes])

for i,f in enumerate(sample):
    font=TTFont(f)
    cmap=font.getBestCmap()
    glyph_set=font.getGlyphSet()
    chars=list(cmap.keys())
    random.shuffle(chars)
    target_pen=None
    target_char=""
    for code in chars:
        name=cmap[code]
        rec=RecordingPen()
        glyph_set[name].draw(rec)
        pen=FlattenPen(glyph_set,density=20)
        rec.replay(pen)
        if sum(len(p) for p in pen.polylines)>30:
            target_pen=pen
            target_char=chr(code)
            break
    if not target_pen: continue

    # 左侧：Bezier 矢量
    ax_left=axes[i,0]
    all_x,all_y=[],[]
    for poly in target_pen.polylines:
        px=[p[0] for p in poly]; py=[p[1] for p in poly]
        all_x.extend(px); all_y.extend(py)
        ax_left.plot(px,py,color='#00e5ff',alpha=0.8)
    padding=max(max(all_x)-min(all_x),max(all_y)-min(all_y))*0.15
    ax_left.set_xlim(min(all_x)-padding,max(all_x)+padding)
    ax_left.set_ylim(min(all_y)-padding,max(all_y)+padding)
    ax_left.set_aspect('equal'); ax_left.axis('off')
    ax_left.set_title(f"Flattened Bezier: {target_char}",color='#c9d1d9')

    # 右侧：Voronoi骨架+tensor
    ax_right=axes[i,1]
    G=voronoi_skeleton(target_pen.polylines)
    boundary_points=[p for poly in target_pen.polylines for p in poly]
    segments,junctions,tensor=decompose_graph_with_width(G,boundary_points)

    # 绘制骨架
    colors=plt.cm.spring(np.linspace(0,1,len(segments)))
    for seg,color in zip(segments,colors):
        skel_x=[p[0] for p in seg]; skel_y=[p[1] for p in seg]
        ax_right.plot(skel_x,skel_y,color=color,lw=2.5,alpha=0.9)

    # 绘制 Junctions
    if junctions:
        jx=[p[0] for p in junctions]; jy=[p[1] for p in junctions]
        ax_right.scatter(jx,jy,color='red',marker='*',s=50,zorder=5)

    ax_right.set_xlim(min(all_x)-padding,max(all_x)+padding)
    ax_right.set_ylim(min(all_y)-padding,max(all_y)+padding)
    ax_right.set_aspect('equal'); ax_right.axis('off')
    ax_right.set_title(f"Voronoi Skeleton: Strokes={len(segments)}",color='#00ff41')

plt.tight_layout()
plt.show()

# 输出 tensor
for pt in tensor[:20]:
    print(f"x={pt[0]:.1f}, y={pt[1]:.1f}, width={pt[2]:.2f}")