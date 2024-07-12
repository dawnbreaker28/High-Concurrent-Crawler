import concurrent.futures
import requests
from bs4 import BeautifulSoup
import csv

def read_tsv(file_path):
    urls = []
    with open(file_path, newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            url = row[0].strip()
            # 去除字节字符串前缀并确保 URL 有方案
            if url.startswith("b'"):
                url = url[2:-1]
            if not url.startswith('http://') and not url.startswith('https://'):
                url = 'http://' + url
            urls.append(url)
    return urls

def fetch_url(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_news_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    news_section = soup.find('div', class_='news')  # 根据实际网页结构修改选择器
    if news_section:
        return news_section.get_text(strip=True)
    return None

def process_url(url):
    html = fetch_url(url)
    if html:
        return extract_news_content(html)
    return None

def save_to_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for url, content in data.items():
            file.write(f"URL: {url}\nContent:\n{content}\n\n")

def main(input_file, output_file):
    urls = read_tsv(input_file)
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(process_url, url): url for url in urls}

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                content = future.result()
                if content:
                    results[url] = content
            except Exception as exc:
                print(f"Error processing {url}: {exc}")

    save_to_file(results, output_file)

if __name__ == '__main__':
    input_tsv_file = 'test.tsv'  # 指定输入的.tsv文件路径
    output_text_file = 'news_contents.txt'  # 指定输出的文件路径
    main(input_tsv_file, output_text_file)
