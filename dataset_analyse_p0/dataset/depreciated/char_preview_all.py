import os
import json
import webbrowser
import base64
from fontTools.ttLib import TTFont

TARGET_DIR = "alien_tensors_raw"
OUTPUT_HTML = "rune_viewer.html"
RULES_FILE = "rules.json"

def load_and_parse_rules():
    """读取本地 rules.json，并将其解析为计算机可读的十进制区间"""
    # 如果文件不存在，自动生成一个 Demo 规则文件
    if not os.path.exists(RULES_FILE):
        demo_rules = {
            "人类基础拉丁字母及标点": ["0000", "0000"],
        }
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(demo_rules, f, ensure_ascii=False, indent=4)
        print(f"📝 未找到 {RULES_FILE}，已自动生成 Demo 规则文件。")
        rules_dict = demo_rules
    else:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            rules_dict = json.load(f)
            
    # 解析规则：将十六进制字符串转为 (start, end) 的十进制元组列表
    parsed_ranges = []
    print("🛡️ 正在装载拦截规则:")
    for rule_name, bounds in rules_dict.items():
        if len(bounds) == 2:
            start_dec = int(bounds[0], 16)
            end_dec = int(bounds[1], 16)
            parsed_ranges.append((start_dec, end_dec))
            print(f"  - [{rule_name}]: 拦截 U+{bounds[0]} 到 U+{bounds[1]}")
            
    return parsed_ranges

def is_excluded(codepoint, exclude_ranges):
    """判断一个字符地址是否落在任何一个黑名单区间内"""
    for start, end in exclude_ranges:
        if start <= codepoint <= end:
            return True
    return False

def get_filtered_glyphs(ttf_path, exclude_ranges):
    """根据黑名单区间读取字库"""
    try:
        font = TTFont(ttf_path)
        items = []
        cmap = font.getBestCmap()
        for codepoint, glyph_name in cmap.items():
            # 核心过滤逻辑：如果不在排除区间内，才予以保留
            if not is_excluded(codepoint, exclude_ranges):
                items.append({
                    'char': chr(codepoint), 
                    'hex': hex(codepoint)[2:].upper().zfill(4)
                })
        return items
    except Exception as e:
        print(f"❌ 无法读取 {ttf_path}: {e}")
        return []

def generate_html_dashboard():
    print("🚀 启动数据提取流水线 (JSON规则驱动版)...")
    
    if not os.path.exists(TARGET_DIR): 
        print(f"❌ 找不到文件夹: {TARGET_DIR}")
        return

    # 1. 装载拦截区间
    exclude_ranges = load_and_parse_rules()
    
    font_data = {}
    font_faces_css = ""
    
    # 2. 遍历提取
    for filename in os.listdir(TARGET_DIR):
        if filename.lower().endswith(('.ttf', '.otf')):
            filepath = os.path.join(TARGET_DIR, filename)
            font_name = os.path.splitext(filename)[0]
            
            # 传入排除区间进行提取
            filtered_items = get_filtered_glyphs(filepath, exclude_ranges)
            
            if filtered_items:
                font_data[font_name] = filtered_items
                with open(filepath, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                fmt = "truetype" if filename.lower().endswith('.ttf') else "opentype"
                font_faces_css += f"@font-face {{ font-family: '{font_name}'; src: url(data:font/{fmt};base64,{encoded}); }}"
                print(f"  ✅ {font_name}: 拦截后保留了 {len(filtered_items)} 个有效字形")

    # 3. 生成 HTML (已包含标签字体修复)
    html_content = f"""<!DOCTYPE html>
    <html lang="en"><head><meta charset="UTF-8"><style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: sans-serif; padding: 20px; }}
        {font_faces_css}
        .controls {{ display: flex; gap: 10px; margin-bottom: 20px; align-items: center; }}
        button {{ background: #238636; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }}
        .grid {{ display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px; }}
        .card {{ height: 120px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 1px solid #30363d; border-radius: 6px; background: #161b22; }}
        .glyph-display {{ font-size: 40px; }}
        .label {{ font-size: 14px; color: #58a6ff; margin-top: 10px; font-family: monospace, sans-serif !important; font-weight: bold; }}
    </style></head>
    <body>
        <h1>🛸 Filtered Glyph Preview</h1>
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
            names.forEach((n, i) => sel.appendChild(new Option(n + ' (' + data[n].length + ')', i)));
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
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f: f.write(html_content)
    webbrowser.open('file://' + os.path.abspath(OUTPUT_HTML))
    print(f"\n🎉 规则过滤版面板已生成: {OUTPUT_HTML}")

if __name__ == "__main__": generate_html_dashboard()