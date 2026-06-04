import torch
import torch.nn as nn
import numpy as np

class StrokeRouterMLP(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x): 
        return self.net(x)

class ReplayBuffer:
    def __init__(self):
        self.features, self.labels = [], []
    def add(self, feature, label):
        self.features.append(feature)
        self.labels.append(label)

def extract_pairwise_features(path_a, path_b, dt_map):
    """提取两个笔画片段之间的几何特征，输入给模型预测是否应该合并"""
    ends_a, ends_b = [path_a[0], path_a[-1]], [path_b[0], path_b[-1]]
    min_dist, best_a_idx, best_b_idx = float('inf'), 0, 0
    for i, ea in enumerate(ends_a):
        for j, eb in enumerate(ends_b):
            dist = np.linalg.norm(ea - eb)
            if dist < min_dist: 
                min_dist, best_a_idx, best_b_idx = dist, i, j
                
    dist_feat = np.clip(min_dist / 10.0, 0, 1)
    step = min(4, len(path_a)-1, len(path_b)-1)
    if step < 1: step = 1
    
    vec_a = path_a[step] - path_a[0] if best_a_idx == 0 else path_a[-1-step] - path_a[-1]
    vec_b = path_b[step] - path_b[0] if best_b_idx == 0 else path_b[-1-step] - path_b[-1]
    norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    
    cos_theta = 0 if norm_a < 1e-5 or norm_b < 1e-5 else np.dot(vec_a, vec_b) / (norm_a * norm_b)
    len_ratio = min(len(path_a), len(path_b)) / max(len(path_a), len(path_b))
    
    w_a = dt_map[int(ends_a[best_a_idx][0]), int(ends_a[best_a_idx][1])]
    w_b = dt_map[int(ends_b[best_b_idx][0]), int(ends_b[best_b_idx][1])]
    w_ratio = min(w_a, w_b) / (max(w_a, w_b) + 1e-5)
    
    return [dist_feat, cos_theta, len_ratio, w_ratio]