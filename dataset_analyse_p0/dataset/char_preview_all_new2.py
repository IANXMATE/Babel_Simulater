import os
import json
import webbrowser
import base64
from fontTools.ttLib import TTFont

# ==========================================
# ⚙️ 全局配置
# ==========================================
TARGET_DIR = "alien_tensors_raw"
OUTPUT_HTML = "rune_viewer.html"
WHITELIST_FILE = "font_whitelist.json"
SCRIPTS_FILE = "Scripts.txt"
RULES_FILE = "rules.json"

def load_and_parse_rules():
    """解析双向规则库：包含全局兜底黑名单 (排除) 和 字体专属白名单 (保留)"""
    if not os.path.exists(RULES_FILE):
        demo_rules = {
            "DEFAULT_BLACKLIST": {
                "人类基础拉丁字母及标点": ["0000", "02AF"]
            },
            "CUSTOM_WHITELIST": {
                "Daedriccalligraphy-Regular": [
                    ["E000", "F8FF"]
                ],
                "Demo-Wildcard-Font": []
            }
        }
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(demo_rules, f, ensure_ascii=False, indent=4)
        print(f"📝 未找到 {RULES_FILE}，已自动生成 [双向控制] 规则模板。")
        rules_dict = demo_rules
    else:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            rules_dict = json.load(f)
            
    # 1. 解析兜底黑名单 (排除区间)
    fallback_exclude_ranges = []
    print("🛡️ 正在装载全局兜底黑名单 (DEFAULT_BLACKLIST):")
    for rule_name, bounds in rules_dict.get("DEFAULT_BLACKLIST", {}).items():
        if len(bounds) == 2:
            start_dec, end_dec = int(bounds[0], 16), int(bounds[1], 16)
            fallback_exclude_ranges.append((start_dec, end_dec))
            print(f"  - [{rule_name}]: 拦截 U+{bounds[0]} 到 U+{bounds[1]}")

    # 2. 解析字体专属白名单 (保留区间)
    custom_include_rules = {}
    print("🎛️ 正在装载字体专属白名单 (CUSTOM_WHITELIST):")
    for font_key, ranges in rules_dict.get("CUSTOM_WHITELIST", {}).items():
        # 💡 核心升级：如果 value 为空列表，转化为全量通配符 (0 到 10FFFF)
        if not ranges:
            custom_include_rules[font_key] = [(0, 0x10FFFF)]
            print(f"  - [{font_key}]: 🔓 免检通道开启 (全量无差别提取)")
        else:
            parsed_ranges = []
            for bounds in ranges:
                if len(bounds) == 2:
                    parsed_ranges.append((int(bounds[0], 16), int(bounds[1], 16)))
            custom_include_rules[font_key] = parsed_ranges
            print(f"  - [{font_key}]: 强制定制保留 {len(parsed_ranges)} 个区间")
            
    return fallback_exclude_ranges, custom_include_rules

def parse_unicode_scripts(filepath):
    """加载 Unicode 官方字典"""
    script_ranges = {}
    if not os.path.exists(filepath):
        print(f"⚠️ 警告: 找不到 {filepath}。")
        return script_ranges
        
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

def get_filtered_glyphs(ttf_path, valid_ranges, is_fallback, fallback_exclude_ranges):
    """核心过滤器：处理保留或剔除逻辑"""
    try:
        font = TTFont(ttf_path)
        items = []
        cmap = font.getBestCmap()
        for codepoint, glyph_name in cmap.items():
            keep = False
            if not is_fallback:
                # 🎯 白名单模式 (命中区间则保留)
                for start, end in valid_ranges:
                    if start <= codepoint <= end:
                        keep = True
                        break
            else:
                # 🛡️ 黑名单模式 (默认保留，命中区间则剔除)
                keep = True
                for start, end in fallback_exclude_ranges:
                    if start <= codepoint <= end:
                        keep = False
                        break
                        
            if keep:
                items.append({'char': chr(codepoint), 'hex': hex(codepoint)[2:].upper().zfill(4)})
        return items
    except Exception as e:
        print(f"❌ 无法读取 {ttf_path}: {e}")
        return []

def generate_html_dashboard():
    print("\n🚀 启动全息浏览器流水线 (三级瀑布流 + 全量通配版)...")
    
    if not os.path.exists(TARGET_DIR): 
        return print(f"❌ 找不到文件夹: {TARGET_DIR}")

    # 装载三大防线
    fallback_exclude_ranges, custom_include_rules = load_and_parse_rules()
    unicode_rules = parse_unicode_scripts(SCRIPTS_FILE)
    
    whitelist_meta = {}
    if os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, 'r', encoding='utf-8') as f:
            whitelist_meta = json.load(f)
    
    font_data = {}
    font_faces_css = ""
    cnt = 0
    for filename in os.listdir(TARGET_DIR):
        if filename.lower().endswith(('.ttf', '.otf')):
            cnt += 1
            filepath = os.path.join(TARGET_DIR, filename)
            font_name = os.path.splitext(filename)[0]
            
            is_fallback = False
            valid_ranges = []
            mode_str = ""
            
            # ===============================================
            # 🚦 核心路由逻辑：三级瀑布流 (Waterfall Pipeline)
            # ===============================================
            
            # [Tier 1] 最高优先级：rules.json 里的定制白名单
            if font_name in custom_include_rules:
                valid_ranges = custom_include_rules[font_name]
                is_fallback = False
                if valid_ranges == [(0, 0x10FFFF)]:
                    mode_str = f"🔓 rules.json 免检通道 (全量解析)"
                else:
                    mode_str = f"🎛️ rules.json 强制定制保留"
                
            else:
                # [Tier 2] 官方字典查阅 (元数据 -> 文件名嗅探)
                meta = whitelist_meta.get(filename, {"primary_script": "UNKNOWN", "valuable_subsets": []})
                matched_subsets = []
                
                # 尝试用 subsets 去 Scripts.txt 里查
                for subset in meta.get('valuable_subsets', []):
                    subset_norm = subset.replace('_', '').replace('-', '').replace(' ', '').lower()
                    if subset_norm in unicode_rules:
                        valid_ranges.extend(unicode_rules[subset_norm])
                        matched_subsets.append(subset)
                
                if len(valid_ranges) > 0:
                    is_fallback = False
                    mode_str = f"🎯 子集精准定位 ({', '.join(matched_subsets)})"
                    
                else:
                    # 尝试用 文件名 去 Scripts.txt 里嗅探兜底
                    clean_filename = font_name.replace('_', '').replace('-', '').replace(' ', '').lower()
                    sniffed = False
                    for known_script in unicode_rules.keys():
                        if len(known_script) > 3 and known_script in clean_filename:
                            valid_ranges.extend(unicode_rules[known_script])
                            is_fallback = False
                            mode_str = f"🧠 文件名嗅探匹配 ({known_script})"
                            sniffed = True
                            break
                            
                    if not sniffed:
                        # [Tier 3] 最低优先级：触发 rules.json 兜底黑名单
                        is_fallback = True
                        mode_str = f"🛡️ 默认黑名单拦截 ({meta.get('primary_script', 'UNKNOWN')})"
                        
            # ===============================================

            # 执行提取
            filtered_items = get_filtered_glyphs(filepath, valid_ranges, is_fallback, fallback_exclude_ranges)
            
            if filtered_items:
                font_data[font_name] = filtered_items
                
                with open(filepath, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                fmt = "truetype" if filename.lower().endswith('.ttf') else "opentype"
                font_faces_css += f"@font-face {{ font-family: '{font_name}'; src: url(data:font/{fmt};base64,{encoded}); }}\n        "
                
                print(f"  ✅ {font_name} [{mode_str}]: 提纯保留了 {len(filtered_items)} 个异星字形")
    print(f"共有{cnt}个文件")
    # 生成 HTML
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