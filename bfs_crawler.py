import requests
from bs4 import BeautifulSoup
from collections import deque

def fetch_page(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_page(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if not href.startswith('http'):
            href = requests.compat.urljoin(base_url, href)
        links.append(href)
    return links

def bfs_crawl(start_url, max_depth):
    queue = deque([(start_url, 0)])
    visited = set([start_url])
    
    while queue:
        url, depth = queue.popleft()
        if depth >= max_depth:
            continue
        
        html = fetch_page(url)
        if html is None:
            continue
        
        print(f"Depth {depth}: {url}")
        
        links = parse_page(html, start_url)
        for link in links:
            if link not in visited:
                visited.add(link)
                queue.append((link, depth + 1))

if __name__ == "__main__":
    base_url = "https://www.bbc.com/"  # 替换为你的起始URL
    bfs_crawl(base_url, 3)
