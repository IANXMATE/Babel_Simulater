import os
import json
import torch
import numpy as np
import pickle

# 严格遵守你的路径读取规范
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(SCRIPT_DIR, "model_config.json")

# 引入之前的解耦引擎和模型 (确保在同一目录下)
try:
    from step1_data_pipeline import StrokeFeatureExtractor
    from step2_train_vqvae import BezierVQVAE
except ImportError:
    print("❌ 无法导入 step1_data_pipeline 或 step2_train_vqvae，请确保它们在同一目录下。")
    exit(1)

def tokenize_full_dataset():
    # 1. 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    anno_dir = os.path.join(SCRIPT_DIR, config['data_config']['annotations_dir'])
    save_dir = os.path.join(SCRIPT_DIR, config['train_config']['save_dir'])
    model_path = os.path.join(save_dir, "stroke_vqvae_best.pth")


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    # 2. 加载训好的 VQ-VAE 模型
    print("🧠 正在加载 VQ-VAE 大脑...")
    mc = config['model_config']
    model = BezierVQVAE(
        input_dim=mc['input_dim'], hidden_dim=mc['hidden_dim'],
        latent_dim=mc['latent_dim'], num_tokens=mc['num_tokens'],
        commitment_cost=mc['commitment_cost']
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ 模型权重加载成功！")
    else:
        print(f"❌ 找不到权重文件 {model_path}，请先运行 step2 训练模型！")
        return

    # 3. 遍历标注集，进行 Token 翻译
    print(f"🔤 正在翻译字典... (目标文件夹: {anno_dir})")
    json_files = [f for f in os.listdir(anno_dir) if f.endswith('.json')]
    
    tokenized_dataset = {} # 结构: { "U+XXXX": [stroke1, stroke2, ...] }
    min_len = config['data_config']['min_stroke_length']
    
    total_chars = 0
    total_strokes = 0
    
    with torch.no_grad(): # 翻译过程不需要梯度
        for file in json_files:
            file_path = os.path.join(anno_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for hex_key, strokes in data.items():
                char_sequence = []
                for stroke in strokes:
                    mb = stroke.get('mother_bezier')
                    if not mb or len(mb) != 4: continue
                    
                    # 🌟 修复：在 step3 中也严格提取 4 维线宽
                    w_raw = stroke.get('width_bezier', stroke.get('width', [6.0, 6.0, 6.0, 6.0]))
                    if isinstance(w_raw, (list, np.ndarray)):
                        if len(w_raw) == 4:
                            widths = np.array(w_raw, dtype=np.float32)
                        else:
                            widths = np.full(4, np.mean(w_raw), dtype=np.float32)
                    else:
                        widths = np.full(4, float(w_raw), dtype=np.float32)
                    
                    # (1) 几何解耦 (此时 widths 是一个 shape 为 (4,) 的 numpy array)
                    spatial, vq_target = StrokeFeatureExtractor.decompose(mb, widths)

                    if spatial[2] < min_len: continue
                    
                    # (2) 让模型看一眼形状，给出 Token ID
                    vq_input = torch.tensor(vq_target, dtype=torch.float32).unsqueeze(0).to(device)
                    z_e = model.encoder(vq_input)
                    _, _, token_id_tensor = model.vq_layer(z_e)
                    token_id = int(token_id_tensor[0].item())
                    
                    # (3) 组装这个笔画的“单词” (离散形状 + 连续空间位置)
                    stroke_word = {
                        "token_id": token_id,         
                        "start_x": float(spatial[0]), 
                        "start_y": float(spatial[1]), 
                        "length": float(spatial[2]),  
                        "angle": float(spatial[3]),
                        "width_0": float(spatial[4]), # 🌟 起点宽度
                        "width_1": float(spatial[5]), # 🌟 控制点1宽度
                        "width_2": float(spatial[6]), # 🌟 控制点2宽度
                        "width_3": float(spatial[7])  # 🌟 终点宽度
                    }

                    char_sequence.append(stroke_word)
                    total_strokes += 1
                
                # 只有包含有效笔画的字才收录
                if len(char_sequence) > 0:
                    tokenized_dataset[hex_key] = char_sequence
                    total_chars += 1

    # 4. 保存为供 Transformer 训练使用的数据集
    output_path = os.path.join(SCRIPT_DIR, "tokenized_fonts_dataset.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(tokenized_dataset, f)
        
    print(f"\n🎉 翻译完成！")
    print(f"📊 统计: 共翻译了 {total_chars} 个字符，包含 {total_strokes} 根笔画。")
    print(f"💾 数据已打包保存至: {output_path}")

    # --- 打印一个字看看长啥样 ---
    if total_chars > 0:
        sample_key = list(tokenized_dataset.keys())[0]
        print(f"\n🧐 抽查字符 '{sample_key}' 的 Token 序列:")
        for idx, s in enumerate(tokenized_dataset[sample_key]):
            print(f"  笔画 {idx+1}: [Token {s['token_id']:04d}] | 位置:({s['start_x']:.1f}, {s['start_y']:.1f}) | 宽:[{s['width_0']:.1f}, {s['width_1']:.1f}, {s['width_2']:.1f}, {s['width_3']:.1f}]")

if __name__ == "__main__":
    tokenize_full_dataset()