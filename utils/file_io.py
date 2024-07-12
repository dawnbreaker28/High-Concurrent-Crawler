import os

def read_urls(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def save_content(url, content, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    filename = os.path.join(result_dir, url.replace('/', '_') + '.txt')
    with open(filename, 'w') as file:
        file.write(content)
