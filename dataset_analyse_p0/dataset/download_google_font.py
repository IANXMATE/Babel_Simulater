import os
import urllib.parse
import requests
import zipfile
import io

def get_full_ttf_from_github(font_display_name, output_dir):
    """
    终极黑科技：绕过 Google Fonts Web 层的动态切片阉割，
    直接去 Google Fonts 的 GitHub 源码仓库拉取满血版 TTF！
    """
    # 转换名字：比如 "Noto Sans Nabataean" -> "notosansnabataean"
    clean_name = font_display_name.replace(' ', '').replace('+', '').lower()
    print(f"  [GitHub 直连] 正在 Google 源码库中搜寻目标: {clean_name}")
    
    # Google Fonts 仓库的三大开源协议文件夹
    licenses = ['ofl', 'ufl', 'apache']
    
    headers = {
        # 伪装并请求 JSON 格式
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.github.v3+json"
    }
    
    for lic in licenses:
        api_url = f"https://api.github.com/repos/google/fonts/contents/{lic}/{clean_name}"
        try:
            resp = requests.get(api_url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                files = resp.json()
                # 找出所有的 ttf 文件
                ttf_files = [f for f in files if f['name'].lower().endswith('.ttf')]
                
                if ttf_files:
                    # 优先下载 Regular 常规体，如果没有就拿第一个
                    target_file = next((f for f in ttf_files if 'Regular' in f['name']), ttf_files[0])
                    download_url = target_file['download_url']
                    file_name = target_file['name']
                    
                    print(f"    🔗 锁定金库源码: {file_name} (准备强拉...)")
                    
                    # 开始下载真正的二进制实体
                    font_resp = requests.get(download_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                    font_resp.raise_for_status()
                    
                    target_path = os.path.join(output_dir, file_name)
                    with open(target_path, "wb") as f:
                        f.write(font_resp.content)
                        
                    print(f"    ✅ 满血版提取成功！体积: {len(font_resp.content) / 1024:.1f} KB")
                    return
        except requests.exceptions.RequestException:
            continue
            
    print(f"    ❌ 致命失败：在 Google GitHub 源码库中未找到 {font_display_name} 的未切割实体。")

def download_google_font(url, output_dir):
    parsed_url = urllib.parse.urlparse(url)
    path_parts = parsed_url.path.strip('/').split('/')
    font_name_raw = path_parts[-1]
    font_display_name = urllib.parse.unquote(font_name_raw).replace('+', ' ')
    
    # 拦截某些特殊的查询参数残留
    if '?' in font_display_name:
        font_display_name = font_display_name.split('?')[0]
        
    get_full_ttf_from_github(font_display_name, output_dir)

def download_dafont(url, output_dir):
    """处理 DaFont 链接 (稳定，保持不变)"""
    parsed_url = urllib.parse.urlparse(url)
    filename = parsed_url.path.strip('/').split('/')[-1]
    
    if not filename.endswith('.font'):
        return
        
    font_name = filename.replace('.font', '')
    print(f"  [DaFont] 锁定目标: {font_name}")
    api_url = f"https://dl.dafont.com/dl/?f={font_name}"
    
    try:
        response = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
            
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            font_files = [n for n in z.namelist() if n.lower().endswith(('.ttf', '.otf'))]
            for f_name in font_files:
                basename = os.path.basename(f_name)
                target_path = os.path.join(output_dir, basename)
                with z.open(f_name) as source, open(target_path, "wb") as target:
                    target.write(source.read())
                print(f"    ✅ 提取落盘: {basename}")
                
    except Exception as e:
        print(f"    ❌ DaFont 解压或请求失败: {e}")

def main():
    print("🛸 异星符文爬虫 V5.0 (GitHub 源码直连版) 启动！")
    filepath = input("👉 请输入清单文件路径 (例如 dataset.txt): ").strip()
    
    if not os.path.exists(filepath):
        print(f"❌ 找不到文件: {filepath}")
        return
        
    output_dir = "alien_tensors_raw"
    os.makedirs(output_dir, exist_ok=True)
    
    # 清理一下上一轮下载的假洋鬼子 (防止文件混淆)
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
    print(f"🧹 已清空上一轮的切片碎片。")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        url = line.strip()
        if not url or url.startswith('#') or url.startswith('%'):
            continue
            
        print(f"\n🔄 正在请求: {url}")
        
        if "fonts.google.com" in url:
            download_google_font(url, output_dir)
        elif "dafont.com" in url:
            download_dafont(url, output_dir)
            
    print("\n🎉 提取完毕！真正的满血异星符文已入库。")

if __name__ == "__main__":
    main()