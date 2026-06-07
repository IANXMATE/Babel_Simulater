import os
import json
from fontTools.ttLib import TTFont
from fontTools import subset  # 🌟 新增：引入字库子集化切割工具

# ==========================================
# ⚙️ 全局配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIR = "alien_tensors_raw"
TARGET_DIR = os.path.join(SCRIPT_DIR, DIR)
# 🌟 新增：提纯后的目标存储文件夹
STORAGE_DIR = os.path.join(SCRIPT_DIR, "alien_tensors_storage") 

WHITELIST_FILE = "font_whitelist.json"
SCRIPTS_FILE = "Scripts.txt"
RULES_FILE = "rules.json"

WHITELIST_FILE = os.path.join(SCRIPT_DIR, "font_whitelist") 
SCRIPTS_FILE = os.path.join(SCRIPT_DIR, "Scripts.txt") 
RULES_FILE = os.path.join(SCRIPT_DIR, "rules.json") 

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

def get_valid_codepoints(ttf_path, valid_ranges, is_fallback, fallback_exclude_ranges):
    """核心过滤器：返回需要保留的 Unicode Codepoint 集合"""
    try:
        font = TTFont(ttf_path)
        keep_codepoints = set()
        cmap = font.getBestCmap()
        
        if not cmap:
            return keep_codepoints

        for codepoint in cmap.keys():
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
                keep_codepoints.add(codepoint)
                
        font.close()
        return keep_codepoints
    except Exception as e:
        print(f"❌ 无法读取 {ttf_path}: {e}")
        return set()

def process_and_export_fonts():
    print("\n🚀 启动异星字体提纯切割流水线 (三级瀑布流 -> 物理导出)...")
    
    if not os.path.exists(TARGET_DIR): 
        return print(f"❌ 找不到输入文件夹: {TARGET_DIR}")

    # 🌟 自动创建输出文件夹
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)
        print(f"📁 已创建输出文件夹: {STORAGE_DIR}")

    # 装载三大防线
    fallback_exclude_ranges, custom_include_rules = load_and_parse_rules()
    unicode_rules = parse_unicode_scripts(SCRIPTS_FILE)
    
    whitelist_meta = {}
    if os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, 'r', encoding='utf-8') as f:
            whitelist_meta = json.load(f)
    
    cnt = 0
    success_cnt = 0
    
    for filename in os.listdir(TARGET_DIR):
        if filename.lower().endswith(('.ttf', '.otf')):
            cnt += 1
            filepath = os.path.join(TARGET_DIR, filename)
            output_filepath = os.path.join(STORAGE_DIR, filename)
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
                
                # 🌟 修复：将局部变量从 subset 改为 subset_name，防止覆盖 fontTools.subset 模块
                for subset_name in meta.get('valuable_subsets', []):
                    subset_norm = subset_name.replace('_', '').replace('-', '').replace(' ', '').lower()
                    if subset_norm in unicode_rules:
                        valid_ranges.extend(unicode_rules[subset_norm])
                        matched_subsets.append(subset_name)
                
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

            # 1. 过滤获取需要保留的 Unicode 编码点集合
            codepoints_to_keep = get_valid_codepoints(filepath, valid_ranges, is_fallback, fallback_exclude_ranges)
            
            if not codepoints_to_keep:
                print(f"  ⚠️ {font_name} [{mode_str}]: 过滤后无字形保留，跳过导出。")
                continue
                
            # 2. 🌟 物理切割字体 (Subset)
            try:
                # 此时 subset 安全地指向 fontTools.subset 模块
                options = subset.Options()
                options.name_IDs = '*' 
                options.name_legacy = True
                options.name_languages = '*'
                options.layout_features = '*'
                options.recommended_glyphs = True
                options.notdef_outline = True
                
                subsetter = subset.Subsetter(options=options)
                subsetter.populate(unicodes=codepoints_to_keep)
                
                font = TTFont(filepath)
                subsetter.subset(font)
                font.save(output_filepath)
                font.close()
                
                success_cnt += 1
                print(f"  ✅ {font_name} [{mode_str}]: 提纯成功！({len(codepoints_to_keep)} 个字形) -> 已保存至 alien_tensors_storage")
            except Exception as e:
                print(f"  ❌ {font_name} [{mode_str}]: 切割保存失败 -> {e}")
                
    print(f"\n🎉 任务完成！共处理了 {cnt} 个字体文件，成功导出 {success_cnt} 个提纯后的字体到 {STORAGE_DIR}")

if __name__ == "__main__": 
    process_and_export_fonts()