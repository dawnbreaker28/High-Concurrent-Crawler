import os
import re

def read_urls(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def sanitize_filename(url):
    # 只保留字母、数字和一些常见字符，将其他字符替换为下划线
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', url)

def save_content(url, content, result_dir):
    filename = os.path.join(result_dir, sanitize_filename(url) + '.html')
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


        
