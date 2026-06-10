import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StrokeFeatureExtractor:
    """
    🌟 核心几何特征解耦引擎
    将原始贝塞尔曲线拆解为: [起点x, 起点y, 长度, 角度, 归一化母线(8维), 宽度]
    """
    @staticmethod
    def decompose(mother_bezier, width_array):
        mb = np.array(mother_bezier, dtype=np.float32) # shape (4, 2)
        p0, p1, p2, p3 = mb[0], mb[1], mb[2], mb[3]
        
        # 1. 提取起点
        start_x, start_y = p0[0], p0[1]
        
        # 2. 提取长度和角度 (基于首尾连线)
        delta = p3 - p0
        length = np.linalg.norm(delta)
        
        # 极小笔画防崩溃护栏
        if length < 1e-3:
            angle = 0.0
            length_safe = 1.0
            norm_mb = mb - p0 # 仅平移，不缩放旋转
        else:
            angle = np.arctan2(delta[1], delta[0]) # 弧度制 (-pi 到 pi)
            length_safe = length
            
            # 🌟 剥离空间信息：平移 -> 旋转 -> 缩放
            # 平移 (起点归零)
            shifted_mb = mb - p0
            
            # 旋转矩阵 (逆向旋转 -angle，把首尾连线强行按在 X 轴上)
            cos_a, sin_a = np.cos(-angle), np.sin(-angle)
            rot_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a,  cos_a]
            ])
            rotated_mb = np.dot(shifted_mb, rot_matrix.T)
            
            # 缩放 (除以长度，把 p3 强行按在 (1.0, 0.0) 的位置)
            norm_mb = rotated_mb / length_safe
            
        # 此时的 norm_mb，p0 永远是 (0,0)，p3 永远是 (1,0) (闭合曲线除外)
        # ... (前面的平移、旋转、缩放逻辑保持完全不变) ...
        norm_bezier_info = norm_mb.flatten() # 8 维向量
        
        # 🌟 修复：宏观物理信息扩展为 8 维 (加入 4 个控制点的宽度)
        spatial_info = np.concatenate([
            [start_x, start_y, length, angle], 
            width_array
        ]).astype(np.float32)
        
        # VQ 目标变成绝对纯粹的 8 维骨架
        vq_target_info = norm_bezier_info.astype(np.float32) 
        
        return spatial_info, vq_target_info

class AlienStrokeDataset(Dataset):
    def __init__(self, config_path="model_config.json"):
        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.anno_dir = os.path.join(SCRIPT_DIR, self.config['data_config']['annotations_dir'])
        
        self.min_len = self.config['data_config']['min_stroke_length']
        
        self.spatial_features = [] # 存储宏观特征 [N, 4]
        self.vq_features = []      # 存储 VQ 目标特征 [N, 9] (归一化母线 + 宽度)
        self.stroke_sources = []   # 记录属于哪个字，方便后续追溯
        
        self._build_dataset()
        
    def _build_dataset(self):
        print(f"🔍 正在从 {self.anno_dir} 提取单笔画数据...")
        json_files = [f for f in os.listdir(self.anno_dir) if f.endswith('.json')]
        
        valid_strokes_count = 0
        for file in json_files:
            file_path = os.path.join(self.anno_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for hex_key, strokes in data.items():
                for stroke in strokes:
                    mb = stroke.get('mother_bezier')
                    if not mb or len(mb) != 4: continue
                    
                    # 🌟 修复：严格提取 4 维线宽
                    w_raw = stroke.get('width_bezier', stroke.get('width', [6.0, 6.0, 6.0, 6.0]))
                    if isinstance(w_raw, (list, np.ndarray)):
                        if len(w_raw) == 4:
                            widths = np.array(w_raw, dtype=np.float32)
                        else:
                            widths = np.full(4, np.mean(w_raw), dtype=np.float32)
                    else:
                        widths = np.full(4, float(w_raw), dtype=np.float32)
                    
                    spatial, vq_target = StrokeFeatureExtractor.decompose(mb, widths)
                    
                    # 过滤掉太小的无意义噪点笔画
                    if spatial[2] < self.min_len: 
                        continue
                        
                    self.spatial_features.append(spatial)
                    self.vq_features.append(vq_target)
                    self.stroke_sources.append(hex_key)
                    valid_strokes_count += 1
                    
        print(f"✅ 数据管道构建完成！共提取出 {valid_strokes_count} 根有效笔画。")
        
    def __len__(self):
        return len(self.vq_features)
        
    def __getitem__(self, idx):
        # 第一阶段 (训练 VQ-VAE) 时，我们只关心 vq_target
        # spatial_features 会在未来的 Transformer 序列训练时用到
        return {
            "vq_input": torch.tensor(self.vq_features[idx]), 
            "spatial": torch.tensor(self.spatial_features[idx]),
            "hex_key": self.stroke_sources[idx]
        }

# ==========================================
# 🚀 Step 1 测试入口
# ==========================================
if __name__ == "__main__":
    # 实例化数据集
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset = AlienStrokeDataset(config_path=os.path.join(SCRIPT_DIR, "model_config.json"))
    
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n📊 --- 数据解耦示例 ---")
        print(f"字符来源: {sample['hex_key']}")
        print(f"宏观空间特征 [起点x, 起点y, 长度, 角度]:\n{sample['spatial'].numpy()}")
        print(f"VQ-VAE目标 [归一化 p0~p3 坐标, 宽度]:\n{sample['vq_input'].numpy()}")
        
        # 验证归一化是否成功 (p0 应该是 [0,0], p3 应该是 [1,0])
        vq_inp = sample['vq_input'].numpy()
        print(f"💡 校验：归一化起点 p0 = [{vq_inp[0]:.2f}, {vq_inp[1]:.2f}]")
        print(f"💡 校验：归一化终点 p3 = [{vq_inp[6]:.2f}, {vq_inp[7]:.2f}]")