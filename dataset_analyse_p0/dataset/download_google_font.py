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
TARGET_DIR = "alien_tensors_raw"
WHITELIST_FILE = "font_whitelist.json"

# 🔑 [关键配置] 在这里填入你的 GitHub Token (ghp_xxxxxx)
# 如果为空，每小时限额 60 次；填入后额度提升至 5000 次/小时！
#### "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" 
GITHUB_TOKEN = "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" 

# ==========================================
# 🛠️ 核心功能引擎
# ==========================================
def parse_metadata_text(pb_text, font_name):
    """提取 METADATA.pb 中的关键语种/字符集信息"""
    primary_scripts = re.findall(r'primary_script:\s*"([^"]+)"', pb_text)
    subsets = re.findall(r'subsets:\s*"([^"]+)"', pb_text)
    
    # 过滤掉人类常见的无用子集
    ignore_subsets = {"latin", "latin-ext", "menu", "cyrillic", "greek"}
    valuable_subsets = [s for s in subsets if s not in ignore_subsets]
    
    return {
        "primary_script": primary_scripts[0] if primary_scripts else "UNKNOWN",
        "valuable_subsets": valuable_subsets,
        "all_subsets": subsets
    }

def fallback_to_zip_download(font_display_name, output_dir, whitelist_data):
    """降级通道：调用 Google 隐藏的打包接口"""
    api_download_url = f"https://fonts.google.com/download?family={urllib.parse.quote(font_display_name)}"
    try:
        response = requests.get(api_download_url, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=15)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                ttf_files = [name for name in z.namelist() if name.lower().endswith('.ttf')]
                for ttf_filename in ttf_files:
                    extracted_path = z.extract(ttf_filename, path=output_dir)
                    basename = os.path.basename(extracted_path)
                    print(f"    ✅ [降级通道] 提取成功: {basename}")
                    # 降级通道没有 PB 文件，标记为 UNKNOWN
                    whitelist_data[basename] = {"primary_script": "UNKNOWN", "valuable_subsets": []}
        else:
            print(f"    ❌ [降级通道] 失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"    ❌ [降级通道] 发生异常: {e}")

def get_full_ttf_from_github(font_display_name, output_dir, whitelist_data):
    """带 Token 提权与 403 自动降级的 GitHub 直连引擎"""
    clean_name = font_display_name.replace(' ', '').replace('+', '').lower()
    print(f"  [GitHub 直连] 正在搜寻目标与元数据: {clean_name}")
    
    licenses = ['ofl', 'ufl', 'apache']
    headers = {
        "User-Agent": "Mozilla/5.0", 
        "Accept": "application/vnd.github.v3+json"
    }
    
    # 注入提权 Token
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    for lic in licenses:
        api_url = f"https://api.github.com/repos/google/fonts/contents/{lic}/{clean_name}"
        try:
            resp = requests.get(api_url, headers=headers, timeout=10)
            
            # 🚨 增加精准的限流拦截与报警
            if resp.status_code in (403, 429):
                print("    🚨 [警告] 触发 GitHub API 频率限制 (403)！")
                print("    👉 请在代码顶部配置 GITHUB_TOKEN，或等待一小时后重试。")
                print("    🔄 正在自动降级，尝试使用 Google Fonts ZIP 通道强拉...")
                fallback_to_zip_download(font_display_name, output_dir, whitelist_data)
                return

            if resp.status_code == 200:
                files = resp.json()
                ttf_files = [f for f in files if f['name'].lower().endswith('.ttf')]
                
                if ttf_files:
                    # 1. 强拉 TTF 实体
                    target_file = next((f for f in ttf_files if 'Regular' in f['name']), ttf_files[0])
                    ttf_url = target_file['download_url']
                    ttf_name = target_file['name']
                    
                    print(f"    🔗 锁定字体: {ttf_name}")
                    font_resp = requests.get(ttf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                    with open(os.path.join(output_dir, ttf_name), "wb") as f:
                        f.write(font_resp.content)
                    
                    # 2. 搜寻并拉取 METADATA.pb
                    pb_file = next((f for f in files if f['name'] == 'METADATA.pb'), None)
                    if pb_file:
                        pb_resp = requests.get(pb_file['download_url'], headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                        pb_text = pb_resp.text
                        
                        pb_filename = f"{os.path.splitext(ttf_name)[0]}_METADATA.pb"
                        with open(os.path.join(output_dir, pb_filename), "w", encoding="utf-8") as f:
                            f.write(pb_text)
                            
                        meta_info = parse_metadata_text(pb_text, ttf_name)
                        print(f"    📜 解析元数据 -> 语种: [{meta_info['primary_script']}], 独有子集: {meta_info['valuable_subsets']}")
                        whitelist_data[ttf_name] = meta_info
                    else:
                        print("    ⚠️ 未找到 METADATA.pb，退回盲切模式。")
                        whitelist_data[ttf_name] = {"primary_script": "UNKNOWN", "valuable_subsets": []}

                    print(f"    ✅ 下载完成！")
                    return
        except requests.exceptions.RequestException as e:
            continue
            
    print(f"    ❌ 致命失败：未找到 {font_display_name} 的源码。尝试启动降级通道...")
    fallback_to_zip_download(font_display_name, output_dir, whitelist_data)

def download_google_font(url, output_dir, whitelist_data):
    parsed_url = urllib.parse.urlparse(url)
    font_display_name = urllib.parse.unquote(parsed_url.path.strip('/').split('/')[-1]).replace('+', ' ').split('?')[0]
    get_full_ttf_from_github(font_display_name, output_dir, whitelist_data)

def download_dafont(url, output_dir, whitelist_data):
    """DaFont 没有官方 PB 文件，统一打上 PUA (私有区) 标签"""
    parsed_url = urllib.parse.urlparse(url)
    filename = parsed_url.path.strip('/').split('/')[-1]
    if not filename.endswith('.font'): return
    font_name = filename.replace('.font', '')
    
    print(f"  [DaFont] 锁定目标: {font_name}")
    api_url = f"https://dl.dafont.com/dl/?f={font_name}"
    try:
        response = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for f_name in [n for n in z.namelist() if n.lower().endswith(('.ttf', '.otf'))]:
                basename = os.path.basename(f_name)
                with z.open(f_name) as source, open(os.path.join(output_dir, basename), "wb") as target:
                    target.write(source.read())
                print(f"    ✅ 提取落盘: {basename} (由于是 DaFont，自动标记为 PUA 优先)")
                whitelist_data[basename] = {"primary_script": "PUA", "valuable_subsets": ["Private Use Area"]}
    except Exception as e:
        print(f"    ❌ DaFont 失败: {e}")

# ==========================================
# 🚀 主控流水线
# ==========================================
def main():
    print("🛸 异星符文爬虫 V7.1 (终极防断流 + GitHub 源码版) 启动！")
    
    if not os.path.exists(DATASET_FILE): 
        print(f"❌ 致命错误: 在当前目录下找不到 {DATASET_FILE} 文件！")
        print(f"💡 请先创建一个 {DATASET_FILE}，并把你的字体链接按行粘贴进去。")
        return 
        
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 安全清空旧数据
    for f in os.listdir(TARGET_DIR): 
        file_path = os.path.join(TARGET_DIR, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    if os.path.exists(WHITELIST_FILE): 
        os.remove(WHITELIST_FILE)
    print(f"🧹 已清空工作区，准备拉取全新的满血张量。")
    
    whitelist_data = {}
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith(('#', '%'))]
        
    for url in lines:
        print(f"\n🔄 正在请求: {url}")
        if "fonts.google.com" in url:
            download_google_font(url, TARGET_DIR, whitelist_data)
        elif "dafont.com" in url:
            download_dafont(url, TARGET_DIR, whitelist_data)
            
    # 将解析到的所有元数据统一保存为 JSON
    with open(WHITELIST_FILE, "w", encoding="utf-8") as f:
        json.dump(whitelist_data, f, ensure_ascii=False, indent=4)
        
    print("\n" + "="*50)
    print(f"🎉 全部入库！满血字体已落盘至目录: {TARGET_DIR}")
    print(f"📜 元数据指纹库已生成: {WHITELIST_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()