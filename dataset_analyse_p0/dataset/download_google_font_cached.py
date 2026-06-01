import os
import urllib.parse
import requests
import zipfile
import io
import re
import json

# ==========================================
# ⚙️ 全局配置区
# ==========================================
DATASET_FILE = "dataset.txt"

TARGET_DIR = "alien_tensors_raw"       # 【临时收件箱】爬虫只能往这里写数据（每次启动清空）
STORAGE_DIR = "alien_tensors_storage"  # 【归档区】爬虫只在这里判断是否有 .otf / .ttf
WHITELIST_FILE = os.path.join(TARGET_DIR, "font_whitelist.json") # 供下游脚本使用的元数据也写在 raw 里

# 🔑 GitHub Token https://github.com/settings/tokens
GITHUB_TOKEN = "" 

# ==========================================
# 🛠️ 核心功能引擎 (严格限制写入 output_dir)
# ==========================================
def parse_metadata_text(pb_text, font_name):
    primary_scripts = re.findall(r'primary_script:\s*"([^"]+)"', pb_text)
    subsets = re.findall(r'subsets:\s*"([^"]+)"', pb_text)
    
    ignore_subsets = {"latin", "latin-ext", "menu", "cyrillic", "greek"}
    valuable_subsets = [s for s in subsets if s not in ignore_subsets]
    
    return {
        "primary_script": primary_scripts[0] if primary_scripts else "UNKNOWN",
        "valuable_subsets": valuable_subsets,
        "all_subsets": subsets
    }

def fallback_to_zip_download(font_display_name, output_dir, whitelist_data):
    api_download_url = f"https://fonts.google.com/download?family={urllib.parse.quote(font_display_name)}"
    try:
        response = requests.get(api_download_url, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=15)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                ttf_files = [name for name in z.namelist() if name.lower().endswith(('.ttf', '.otf'))]
                for ttf_filename in ttf_files:
                    extracted_path = z.extract(ttf_filename, path=output_dir)
                    basename = os.path.basename(extracted_path)
                    print(f"    ✅ [降级通道] 提取成功: {basename}")
                    whitelist_data[basename] = {"primary_script": "UNKNOWN", "valuable_subsets": []}
        else:
            print(f"    ❌ [降级通道] 失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"    ❌ [降级通道] 发生异常: {e}")

def get_full_ttf_from_github(font_display_name, output_dir, whitelist_data):
    clean_name = font_display_name.replace(' ', '').replace('+', '').lower()
    print(f"  [GitHub 直连] 正在请求: {clean_name}")
    
    licenses = ['ofl', 'ufl', 'apache']
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    for lic in licenses:
        api_url = f"https://api.github.com/repos/google/fonts/contents/{lic}/{clean_name}"
        try:
            resp = requests.get(api_url, headers=headers, timeout=10)
            
            if resp.status_code in (403, 429):
                print("    🚨 触发 GitHub API 频率限制，转入降级通道...")
                fallback_to_zip_download(font_display_name, output_dir, whitelist_data)
                return

            if resp.status_code == 200:
                files = resp.json()
                ttf_files = [f for f in files if f['name'].lower().endswith(('.ttf', '.otf'))]
                
                if ttf_files:
                    target_file = next((f for f in ttf_files if 'Regular' in f['name']), ttf_files[0])
                    ttf_name = target_file['name']
                    
                    print(f"    🔗 锁定字体: {ttf_name}")
                    font_resp = requests.get(target_file['download_url'], headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                    
                    # 写入 raw 文件夹
                    with open(os.path.join(output_dir, ttf_name), "wb") as f:
                        f.write(font_resp.content)
                    
                    pb_file = next((f for f in files if f['name'] == 'METADATA.pb'), None)
                    if pb_file:
                        pb_resp = requests.get(pb_file['download_url'], headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                        pb_filename = f"{os.path.splitext(ttf_name)[0]}_METADATA.pb"
                        
                        # PB 文件也写入 raw 文件夹供下游脚本使用
                        with open(os.path.join(output_dir, pb_filename), "w", encoding="utf-8") as f:
                            f.write(pb_resp.text)
                            
                        meta_info = parse_metadata_text(pb_resp.text, ttf_name)
                        print(f"    📜 元数据: {meta_info['valuable_subsets']}")
                        whitelist_data[ttf_name] = meta_info
                    else:
                        print("    ⚠️ 未找到 METADATA.pb，退回盲切模式。")
                        whitelist_data[ttf_name] = {"primary_script": "UNKNOWN", "valuable_subsets": []}

                    print(f"    ✅ 下载完成！")
                    return
        except requests.exceptions.RequestException:
            continue
            
    print(f"    ❌ GitHub 库中未找到，转入降级通道...")
    fallback_to_zip_download(font_display_name, output_dir, whitelist_data)

def download_google_font(url, output_dir, whitelist_data):
    parsed_url = urllib.parse.urlparse(url)
    font_display_name = urllib.parse.unquote(parsed_url.path.strip('/').split('/')[-1]).replace('+', ' ').split('?')[0]
    get_full_ttf_from_github(font_display_name, output_dir, whitelist_data)

def download_dafont(url, output_dir, whitelist_data):
    parsed_url = urllib.parse.urlparse(url)
    filename = parsed_url.path.strip('/').split('/')[-1]
    if not filename.endswith('.font'): return
    font_name = filename.replace('.font', '')
    
    print(f"  [DaFont] 正在请求: {font_name}")
    try:
        response = requests.get(f"https://dl.dafont.com/dl/?f={font_name}", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for f_name in [n for n in z.namelist() if n.lower().endswith(('.ttf', '.otf'))]:
                basename = os.path.basename(f_name)
                # 写入 raw 文件夹
                with z.open(f_name) as source, open(os.path.join(output_dir, basename), "wb") as target:
                    target.write(source.read())
                print(f"    ✅ 提取落盘: {basename}")
                whitelist_data[basename] = {"primary_script": "PUA", "valuable_subsets": ["Private Use Area"]}
    except Exception as e:
        print(f"    ❌ DaFont 失败: {e}")

# ==========================================
# 🛑 [核心] 纯物理字体比对嗅探器
# ==========================================
def check_if_font_in_storage(url, storage_dir):
    """
    只查 storage_dir 里面有没有对应的 .ttf 或 .otf 实体文件。
    没有 JSON 判断，没有 Marker，一切以实体归档文件为准。
    """
    if not os.path.exists(storage_dir):
        return None
        
    parsed_url = urllib.parse.urlparse(url)
    if "dafont.com" in url:
        raw_name = parsed_url.path.strip('/').split('/')[-1].replace('.font', '')
    else:
        raw_name = urllib.parse.unquote(parsed_url.path.strip('/').split('/')[-1]).split('?')[0]
        
    search_key = raw_name.replace('+', '').replace('-', '').replace('_', '').replace(' ', '').lower()
    
    for fname in os.listdir(storage_dir):
        # 【严格限制】仅比对 .ttf 和 .otf 文件
        if not fname.lower().endswith(('.ttf', '.otf')): 
            continue 
        
        clean_fname = fname.replace('-', '').replace('_', '').replace(' ', '').lower()
        if search_key in clean_fname:
            return fname 
            
    return None

# ==========================================
# 🚀 主控流水线
# ==========================================
def main():
    print("🛸 异星符文爬虫 V9.4 (严格物理归档制导版) 启动！")
    
    if not os.path.exists(DATASET_FILE): 
        print(f"❌ 致命错误: 找不到 {DATASET_FILE}")
        return 
        
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(STORAGE_DIR, exist_ok=True)
    
    # 🌟 启动时自动清空临时收件箱 (raw)
    for f in os.listdir(TARGET_DIR): 
        file_path = os.path.join(TARGET_DIR, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"🧹 已清空临时收件箱 ({TARGET_DIR})。")
    
    whitelist_data = {}
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith(('#', '%'))]
    
    lines = list(dict.fromkeys(lines))
        
    for url in lines:
        print(f"\n🔄 准备处理: {url}")
        
        # 🌟 核心拦截逻辑：去归档区 (storage) 寻找是否有对应的 .otf / .ttf
        archived_file = check_if_font_in_storage(url, STORAGE_DIR)
        
        if archived_file:
            print(f"  ⏩ [跳过] 归档区发现实体文件 `{archived_file}`，确认已处理过。")
            continue
            
        # 未被归档的字体，全部下载至 TARGET_DIR (raw)
        if "fonts.google.com" in url:
            download_google_font(url, TARGET_DIR, whitelist_data)
        elif "dafont.com" in url:
            download_dafont(url, TARGET_DIR, whitelist_data)
            
        # 生成的白名单也仅保存在 TARGET_DIR 供下游流水线使用
        with open(WHITELIST_FILE, "w", encoding="utf-8") as f:
            json.dump(whitelist_data, f, ensure_ascii=False, indent=4)
        
    print("\n" + "="*50)
    print(f"🎉 本次增量任务结束！")
    print(f"📥 待处理的原始数据已放入: {TARGET_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()