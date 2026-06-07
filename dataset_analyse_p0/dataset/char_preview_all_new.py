import os
import json
import webbrowser
import base64
from fontTools.ttLib import TTFont

# ==========================================
# ⚙️ 全局配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_DIR = "alien_tensors_storage"
OUTPUT_HTML = "rune_viewer.html"
WHITELIST_FILE = "font_whitelist.json"
SCRIPTS_FILE = "scripts.txt"

TARGET_DIR = os.path.join(SCRIPT_DIR, TARGET_DIR)
OUTPUT_HTML = os.path.join(SCRIPT_DIR, OUTPUT_HTML)
WHITELIST_FILE = os.path.join(SCRIPT_DIR, WHITELIST_FILE)
SCRIPTS_FILE = os.path.join(SCRIPT_DIR, SCRIPTS_FILE)

def parse_unicode_scripts(filepath):
    """解析官方 Scripts.txt，生成 {标准化语种名: [(start_hex, end_hex), ...]} 的映射表"""
    script_ranges = {}
    if not os.path.exists(filepath):
        print(f"⚠️ 警告: 找不到 {filepath}，将只能使用盲切模式。")
        return script_ranges
        
    print(f"📖 正在加载 Unicode 护城河: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line: continue
            
            parts = line.split(';')
            if len(parts) != 2: continue
            
            code_range, script_name_raw = parts[0].strip(), parts[1].strip()
            script_norm = script_name_raw.replace('_', '').replace('-', '').replace(' ', '').lower()
            
            if '..' in code_range:
                start, end = int(code_range.split('..')[0], 16), int(code_range.split('..')[1], 16)
            else:
                start = end = int(code_range, 16)
                
            if script_norm not in script_ranges:
                script_ranges[script_norm] = []
            script_ranges[script_norm].append((start, end))
            
    return script_ranges

def get_filtered_glyphs(ttf_path, valid_ranges, is_fallback):
    """根据 Unicode 白名单区间（或盲切模式）读取纯净字库"""
    try:
        font = TTFont(ttf_path)
        items = []
        cmap = font.getBestCmap()
        for codepoint, glyph_name in cmap.items():
            keep = False
            
            if not is_fallback:
                # 🎯 精准制导：必须落在官方分配的专属十六进制区间内
                for start, end in valid_ranges:
                    if start <= codepoint <= end:
                        keep = True
                        break
            else:
                # 🪓 盲切模式：抛弃 0~0x02AF 范围内的所有人类高频字符 (拉丁、数字、标点)
                if codepoint > -1:
                    keep = True
                    
            if keep:
                items.append({
                    'char': chr(codepoint), 
                    'hex': hex(codepoint)[2:].upper().zfill(4)
                })
        return items
    except Exception as e:
        print(f"❌ 无法读取 {ttf_path}: {e}")
        return []

def generate_html_dashboard():
    print("🚀 启动全息浏览器流水线 (Unicode 元数据制导版)...")
    
    if not os.path.exists(TARGET_DIR): return print(f"❌ 找不到文件夹: {TARGET_DIR}")
    if not os.path.exists(WHITELIST_FILE): return print(f"❌ 找不到 {WHITELIST_FILE}，请先运行爬虫脚本。")

    # 1. 装载 Unicode 护城河规则
    unicode_rules = parse_unicode_scripts(SCRIPTS_FILE)
    
    # 2. 读取爬虫阶段生成的字体元数据白名单
    with open(WHITELIST_FILE, 'r', encoding='utf-8') as f:
        whitelist_meta = json.load(f)
    
    font_data = {}
    font_faces_css = ""
    
    # 3. 遍历提取
    for filename in os.listdir(TARGET_DIR):
        if filename.lower().endswith(('.ttf', '.otf')):
            filepath = os.path.join(TARGET_DIR, filename)
            font_name = os.path.splitext(filename)[0]
            
            # 获取该字体的语种信息
            meta = whitelist_meta.get(filename, {"primary_script": "UNKNOWN"})
            script_norm = meta['primary_script'].replace('_', '').replace('-', '').replace(' ', '').lower()
            
            # 决定使用精准白名单还是盲切
            is_fallback = False
            valid_ranges = []
            if script_norm in unicode_rules:
                valid_ranges = unicode_rules[script_norm]
            else:
                is_fallback = True
                
            # 提取纯净字符
            filtered_items = get_filtered_glyphs(filepath, valid_ranges, is_fallback)
            
            if filtered_items:
                font_data[font_name] = filtered_items
                
                # Base64 编码，完美嵌入 HTML (避免跨域或路径问题)
                with open(filepath, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                fmt = "truetype" if filename.lower().endswith('.ttf') else "opentype"
                font_faces_css += f"@font-face {{ font-family: '{font_name}'; src: url(data:font/{fmt};base64,{encoded}); }}\n        "
                
                mode_str = "🎯 白名单模式" if not is_fallback else "🪓 盲切模式"
                print(f"  ✅ {font_name} [{mode_str}]: 提纯保留了 {len(filtered_items)} 个异星字形")

    # 4. 生成极客风 HTML
    html_content = f"""<!DOCTYPE html>
    <html lang="en"><head><meta charset="UTF-8"><style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: sans-serif; padding: 20px; }}
        {font_faces_css}
        .controls {{ display: flex; gap: 10px; margin-bottom: 20px; align-items: center; }}
        button {{ background: #238636; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }}
        select {{ background: #161b22; color: white; border: 1px solid #30363d; padding: 8px; border-radius: 4px; outline: none; }}
        .grid {{ display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px; }}
        .card {{ height: 120px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 1px solid #30363d; border-radius: 6px; background: #161b22; transition: border-color 0.2s; }}
        .card:hover {{ border-color: #00ff41; }}
        .glyph-display {{ font-size: 44px; color: #00ff41; }}
        .label {{ font-size: 14px; color: #58a6ff; margin-top: 10px; font-family: monospace, sans-serif !important; font-weight: bold; }}
    </style></head>
    <body>
        <h1 style="color: #00ff41; font-family: monospace;">🛸 Alien Tensor Glyph Preview Dashboard</h1>
        <div class="controls">
            <button onclick="prev()">◀ Prev</button>
            <select id="sel"></select>
            <button onclick="next()">Next ▶</button>
        </div>
        <div class="grid" id="grid"></div>
        <script>
            const data = {json.dumps(font_data, ensure_ascii=False)}, names = Object.keys(data);
            const sel = document.getElementById('sel'), grid = document.getElementById('grid');
            let idx = 0;
            names.forEach((n, i) => sel.appendChild(new Option(n + ' (Yield: ' + data[n].length + ' tensors)', i)));
            function render() {{
                sel.value = idx;
                grid.innerHTML = data[names[idx]].map(i => `<div class="card">
                    <div class="glyph-display" style="font-family:'${{names[idx]}}'">${{i.char}}</div>
                    <div class="label">U+${{i.hex}}</div>
                </div>`).join('');
            }}
            function prev() {{ idx = (idx - 1 + names.length) % names.length; render(); }}
            function next() {{ idx = (idx + 1) % names.length; render(); }}
            sel.onchange = (e) => {{ idx = e.target.value; render(); }};
            if (names.length > 0) render();
        </script>
    </body></html>"""
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f: 
        f.write(html_content)
    webbrowser.open('file://' + os.path.abspath(OUTPUT_HTML))
    print(f"\n🎉 提纯预览面板已生成并自动打开: {OUTPUT_HTML}")

if __name__ == "__main__": 
    generate_html_dashboard()