import requests
from concurrent.futures import ThreadPoolExecutor
from utils import logger

def fetch_content(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return url, response.text
    except requests.RequestException as e:
        logger.log(f"Error fetching {url}: {e}")
        return url, ""

def fetch_all(urls, max_workers=10):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_content, url): url for url in urls}
        for future in futures:
            url = futures[future]
            try:
                url, content = future.result()
                results[url] = content
            except Exception as e:
                logger.log(f"Error fetching {url}: {e}")
                results[url] = ""
    return results
