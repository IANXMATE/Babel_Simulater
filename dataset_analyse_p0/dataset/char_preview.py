import os
import json
import webbrowser
import base64
from fontTools.ttLib import TTFont

TARGET_DIR = "alien_tensors_raw"
OUTPUT_HTML = "rune_viewer.html"

def get_font_characters(ttf_path):
    """物理提取 TTF 内部字符，并执行强力【人类语言大清洗】"""
    try:
        font = TTFont(ttf_path)
        chars = set()
        for table in font['cmap'].tables:
            for codepoint, glyph_name in table.cmap.items():
                # 🚀 深度防稀释隔离机制：
                # 1. 强制阻断 0x024F 以内的一切基础英文、数字、拉丁字母扩展
                # 2. 阻断 0xE000-0xF8FF 的私有空白造字区
                if codepoint > 0x024F and not (0xE000 <= codepoint <= 0xF8FF):
                    chars.add(chr(codepoint))
        return list(chars)
    except Exception as e:
        print(f"❌ 解析 {ttf_path} 失败: {e}")
        return []

def generate_html_dashboard():
    print("🔬 正在扫描本地字库，物理清洗拉丁字符集...")
    
    if not os.path.exists(TARGET_DIR):
        print(f"❌ 找不到文件夹 {TARGET_DIR}，请检查字体文件是否已就绪。")
        return

    font_data = {}
    font_faces_css = ""
    
    for filename in os.listdir(TARGET_DIR):
        if filename.lower().endswith(('.ttf', '.otf')):
            filepath = os.path.join(TARGET_DIR, filename)
            font_name = os.path.splitext(filename)[0]
            
            valid_chars = get_font_characters(filepath)
            
            if len(valid_chars) > 0:
                font_data[font_name] = valid_chars
                
                # 压扁二进制字体直接进行 Base64 内存焊接，击穿本地 CORS 跨域壁垒
                with open(filepath, "rb") as f:
                    encoded_font = base64.b64encode(f.read()).decode("utf-8")
                
                format_str = "truetype" if filename.lower().endswith('.ttf') else "opentype"
                
                font_faces_css += f"""
                @font-face {{
                    font-family: '{font_name}';
                    src: url(data:font/{format_str};charset=utf-8;base64,{encoded_font}) format('{format_str}');
                }}
                """
                print(f"  ✅ {font_name}: 捕获 {len(valid_chars)} 个纯净非拉丁字形零件 (已就地装填)")

    if not font_data:
        print("❌ 未在当前文件夹中找到任何有效的非拉丁字符集，请核对字库。")
        return

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>🛸 Alien Rune Visualizer</title>
        <style>
            body {{ background-color: #0d1117; color: #c9d1d9; font-family: monospace; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
            {font_faces_css}
            .header {{ text-align: center; margin-bottom: 20px; }}
            .controls {{ display: flex; gap: 15px; margin-bottom: 30px; align-items: center; }}
            button {{ background: #238636; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; transition: 0.2s; }}
            button:hover {{ background: #2ea043; }}
            button:disabled {{ background: #444; color: #888; cursor: not-allowed; }}
            select {{ background: #161b22; color: #c9d1d9; border: 1px solid #30363d; padding: 10px; border-radius: 6px; font-size: 16px; }}
            .grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; width: 100%; max-width: 1000px; }}
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; height: 160px; display: flex; justify-content: center; align-items: center; font-size: 80px; position: relative; box-shadow: 0 4px 12px rgba(0,0,0,0.5); transition: 0.3s; }}
            .card:hover {{ border-color: #58a6ff; box-shadow: 0 0 15px rgba(88,166,255,0.3); }}
            .unicode-label {{ position: absolute; bottom: 8px; right: 10px; font-size: 12px; color: #8b949e; font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛸 Alien Rune Tensor Preview</h1>
            <p>100% Non-Latin Purified. Bypassing macOS Local File CORS.</p>
        </div>
        
        <div class="controls">
            <button id="btnPrev">◀ Prev</button>
            <select id="fontSelector"></select>
            <button id="btnNext">Next ▶</button>
            <div style="width: 20px;"></div>
            <button id="btnRoll" style="background: #1f6feb;">🎲 Roll 10 Glyphs</button>
        </div>

        <div class="grid" id="glyphGrid"></div>

        <script>
            const fontData = {json.dumps(font_data, ensure_ascii=False)};
            const fontNames = Object.keys(fontData);
            const selector = document.getElementById('fontSelector');
            const grid = document.getElementById('glyphGrid');
            let currentIndex = 0;

            fontNames.forEach((name, idx) => {{
                let opt = document.createElement('option');
                opt.value = idx;
                opt.textContent = name + ' (' + fontData[name].length + ' glyphs)';
                selector.appendChild(opt);
            }});

            function renderGrid() {{
                const fontName = fontNames[currentIndex];
                const chars = fontData[fontName];
                
                let sampleSize = Math.min(10, chars.length);
                let shuffled = chars.slice().sort(() => 0.5 - Math.random());
                let selected = shuffled.slice(0, sampleSize);

                grid.innerHTML = '';
                selected.forEach(char => {{
                    let div = document.createElement('div');
                    div.className = 'card';
                    div.style.fontFamily = "'" + fontName + "', monospace";
                    div.textContent = char;
                    
                    let hex = char.codePointAt(0).toString(16).toUpperCase().padStart(4, '0');
                    let label = document.createElement('span');
                    label.className = 'unicode-label';
                    label.textContent = 'U+' + hex;
                    div.appendChild(label);
                    
                    grid.appendChild(div);
                }});
                
                selector.value = currentIndex;
                document.getElementById('btnPrev').disabled = (currentIndex === 0);
                document.getElementById('btnNext').disabled = (currentIndex === fontNames.length - 1);
            }}

            document.getElementById('btnRoll').addEventListener('click', renderGrid);
            selector.addEventListener('change', (e) => {{ currentIndex = parseInt(e.target.value); renderGrid(); }});
            document.getElementById('btnPrev').addEventListener('click', () => {{ if (currentIndex > 0) {{ currentIndex--; renderGrid(); }} }});
            document.getElementById('btnNext').addEventListener('click', () => {{ if (currentIndex < fontNames.length - 1) {{ currentIndex++; renderGrid(); }} }});

            if (fontNames.length > 0) renderGrid();
        </script>
    </body>
    </html>
    """

    html_path = os.path.abspath(OUTPUT_HTML)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\n🎉 纯净版全息预览面板已强力构建: {html_path}")
    print("🌐 正在唤醒本地默认浏览器，见证真正的异星符文...")
    webbrowser.open('file://' + html_path)

if __name__ == "__main__":
    generate_html_dashboard()