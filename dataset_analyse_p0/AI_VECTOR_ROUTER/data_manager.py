import os
import json

class DatasetManager:
    def __init__(self, script_dir, font_filename):
        self.font_filename = font_filename
        
        # 确保双文件夹存在
        self.anno_dir = os.path.join(script_dir, "annotations")
        self.meta_dir = os.path.join(script_dir, "metadata")
        os.makedirs(self.anno_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        
        # 定义文件路径
        self.anno_json_path = os.path.join(self.anno_dir, f"{font_filename}.json")
        self.meta_json_path = os.path.join(self.meta_dir, f"{font_filename}_meta.json")
        
        # 初始化内存数据结构
        self.annotated_outlines = {}
        self.meta_data = {"banned": [], "raw_edges": {}}
        
        # 实例化时自动读取
        self.load_data()

    def load_data(self):
        if os.path.exists(self.anno_json_path):
            with open(self.anno_json_path, 'r', encoding='utf-8') as f:
                self.annotated_outlines = json.load(f)
        if os.path.exists(self.meta_json_path):
            with open(self.meta_json_path, 'r', encoding='utf-8') as f:
                self.meta_data = json.load(f)

    def save_data(self):
        with open(self.anno_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotated_outlines, f, indent=2, ensure_ascii=False)
        with open(self.meta_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.meta_data, f, indent=2, ensure_ascii=False)