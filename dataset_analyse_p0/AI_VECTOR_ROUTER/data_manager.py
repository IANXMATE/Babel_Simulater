import os
import json
import glob

class DatasetManager:
    def __init__(self, script_dir, font_filename, max_action_mb=40):
        self.font_filename = font_filename
        self.max_action_mb = max_action_mb # 文件切分阈值 (默认 40MB)
        
        # 1. 确保三文件夹存在 (保留你的逻辑，增加 action_dir)
        self.anno_dir = os.path.join(script_dir, "annotations")
        self.meta_dir = os.path.join(script_dir, "metadata")
        self.action_dir = os.path.join(script_dir, "action_logs")
        
        os.makedirs(self.anno_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.action_dir, exist_ok=True)
        
        # 2. 定义核心文件路径 (保留你的逻辑)
        self.anno_json_path = os.path.join(self.anno_dir, f"{font_filename}.json")
        self.meta_json_path = os.path.join(self.meta_dir, f"{font_filename}_meta.json")
        
        # 3. 初始化内存数据结构
        self.annotated_outlines = {}
        self.meta_data = {"banned": [], "raw_edges": {}}
        self.action_data = {} # 🌟 新增：专门存时光机录像的字典
        
        # 实例化时自动读取
        self.load_data()

    def load_data(self):
        # --- 保留你原有的安全读取逻辑 ---
        if os.path.exists(self.anno_json_path):
            with open(self.anno_json_path, 'r', encoding='utf-8') as f:
                self.annotated_outlines = json.load(f)
        if os.path.exists(self.meta_json_path):
            with open(self.meta_json_path, 'r', encoding='utf-8') as f:
                self.meta_data = json.load(f)
                
        # --- 🌟 新增：极度安全的 action_logs 读取逻辑 ---
        files_to_load = []
        
        # 优先级1：绝对精准定位你已经存在的旧版单一文件 (不带后缀)
        base_file = os.path.join(self.action_dir, f"{self.font_filename}_actions.json")
        if os.path.exists(base_file):
            files_to_load.append(base_file)
            
        # 优先级2：加载未来可能生成的切分文件 (_1.json, _2.json 等)
        chunk_pattern = os.path.join(self.action_dir, f"{self.font_filename}_actions_*.json")
        files_to_load.extend(glob.glob(chunk_pattern))
        
        # 开始合并读取
        for file_path in files_to_load:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.action_data.update(data) # 把数据合并进内存字典
            except Exception as e:
                print(f"⚠️ Error loading action log {os.path.basename(file_path)}: {e}")

    def save_data(self):
        # --- 完全保留你原有的保存逻辑 ---
        with open(self.anno_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotated_outlines, f, indent=2, ensure_ascii=False)
        with open(self.meta_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.meta_data, f, indent=2, ensure_ascii=False)

    def save_action_log(self, hex_key, serialized_log):
        """🌟 新增功能：自动处理超大录像带的切分与保存"""
        # 1. 更新当前字符的录像到内存
        self.action_data[hex_key] = serialized_log
        
        # 2. 查找并清理旧文件，防止产生脏数据
        old_base = os.path.join(self.action_dir, f"{self.font_filename}_actions.json")
        if os.path.exists(old_base):
            try: os.remove(old_base)
            except OSError: pass
            
        old_chunks = glob.glob(os.path.join(self.action_dir, f"{self.font_filename}_actions_*.json"))
        for old_file in old_chunks:
            try: os.remove(old_file)
            except OSError: pass
            
        # 3. 动态计算体积并分块 (基于 MB 阈值)
        max_bytes = self.max_action_mb * 1024 * 1024
        chunks = []
        current_chunk = {}
        current_size = 2 # 空 "{}" 的基础字节数
        
        for k, v in self.action_data.items():
            # 计算单个字符转换成紧凑 JSON 后的字节大小
            item_str = json.dumps({k: v}, ensure_ascii=False, separators=(',', ':'))
            item_size = len(item_str.encode('utf-8'))
            
            if current_size + item_size > max_bytes and current_chunk:
                chunks.append(current_chunk)
                current_chunk = {}
                current_size = 2
                
            current_chunk[k] = v
            current_size += item_size
            
        if current_chunk:
            chunks.append(current_chunk)
            
        # 4. 写入磁盘
        for i, chunk in enumerate(chunks):
            # 如果体积很小只生成了1个块，就恢复成你以前熟悉的旧名字
            if len(chunks) == 1:
                chunk_path = os.path.join(self.action_dir, f"{self.font_filename}_actions.json")
            else:
                # 否则自动加后缀 _1, _2
                chunk_path = os.path.join(self.action_dir, f"{self.font_filename}_actions_{i+1}.json")
            
            with open(chunk_path, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, separators=(',', ':'), ensure_ascii=False)
    
    def delete_character(self, hex_key):
        """🌟 彻底抹除某个字符的所有物理记录"""
        # 1. 删最终标签
        self.annotated_outlines.pop(hex_key, None)
        # 2. 删元数据快照
        if hex_key in self.meta_data.get("raw_edges", {}):
            del self.meta_data["raw_edges"][hex_key]
        self.save_data()
        
        # 3. 删动作录像带
        if hex_key in self.action_data:
            del self.action_data[hex_key]
            # 为了让磁盘上的文件也同步删除，我们复用已有的写入逻辑
            # 将内存中剩下的（已经被删除了那个字符的）录像带重新切分打包
            self.save_action_log("TEMP_TRIGGER", None)
            if "TEMP_TRIGGER" in self.action_data:
                del self.action_data["TEMP_TRIGGER"] # 清理触发器