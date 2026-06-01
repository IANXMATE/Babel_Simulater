import os
import json
import webbrowser
import base64
from fontTools.ttLib import TTFont

TARGET_DIR = "alien_tensors_raw"
OUTPUT_HTML = "rune_viewer.html"

def get_all_glyphs(ttf_path):
    """物理原始读取：不对任何字符进行过滤或清洗"""
    try:
        font = TTFont(ttf_path)
        items = []
        cmap = font.getBestCmap()
        for codepoint, glyph_name in cmap.items():
            items.append({
                'char': chr(codepoint), 
                'hex': hex(codepoint)[2:].upper().zfill(4)
            })
        return items
    except Exception as e:
        print(f"❌ 无法读取 {ttf_path}: {e}")
        return []

def generate_html_dashboard():
    print("🚀 正在加载所有字库，执行全量展示 (RAW 模式)...")
    
    if not os.path.exists(TARGET_DIR): 
        print(f"❌ 找不到文件夹: {TARGET_DIR}")
        return

    font_data = {}
    font_faces_css = ""
    
    for filename in os.listdir(TARGET_DIR):
        if filename.lower().endswith(('.ttf', '.otf')):
            filepath = os.path.join(TARGET_DIR, filename)
            font_name = os.path.splitext(filename)[0]
            
            all_items = get_all_glyphs(filepath)
            
            if all_items:
                font_data[font_name] = all_items
                with open(filepath, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                fmt = "truetype" if filename.lower().endswith('.ttf') else "opentype"
                font_faces_css += f"@font-face {{ font-family: '{font_name}'; src: url(data:font/{fmt};base64,{encoded}); }}"
                print(f"  ✅ {font_name}: 加载了 {len(all_items)} 个字符")

    # 生成 HTML (修复了 label 的字体继承问题)
    html_content = f"""<!DOCTYPE html>
    <html lang="en"><head><meta charset="UTF-8"><style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: sans-serif; padding: 20px; }}
        {font_faces_css}
        .controls {{ display: flex; gap: 10px; margin-bottom: 20px; align-items: center; }}
        button {{ background: #238636; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }}
        .grid {{ display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px; }}
        .card {{ height: 120px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 1px solid #30363d; border-radius: 6px; background: #161b22; }}
        
        /* 核心修复：强制标签使用标准代码字体，不继承外星字体 */
        .label {{ font-size: 14px; color: #58a6ff; margin-top: 10px; font-family: monospace, sans-serif !important; font-weight: bold; }}
        /* 将外星字体仅限制在符文展示区域 */
        .glyph-display {{ font-size: 40px; }}
        
    </style></head>
    <body>
        <h1>🛸 Full Raw Glyph Preview</h1>
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
                // 注意这里：我们将内联样式移动到了 glyph-display 的 div 上，不再污染外层卡片
                grid.innerHTML = data[names[idx]].map(i => `<div class="card">
                    <div class="glyph-display" style="font-family:'${{names[idx]}}'">${{i.char}}</div>
                    <div class="label">U+${{i.hex}}</div>
                </div>`).join('');
            }}
            function prev() {{ idx = (idx - 1 + names.length) % names.length; render(); }}
            function next() {{ idx = (idx + 1) % names.length; render(); }}
            sel.onchange = (e) => {{ idx = e.target.value; render(); }};
            render();
        </script>
    </body></html>"""
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f: f.write(html_content)
    webbrowser.open('file://' + os.path.abspath(OUTPUT_HTML))

if __name__ == "__main__": generate_html_dashboard()