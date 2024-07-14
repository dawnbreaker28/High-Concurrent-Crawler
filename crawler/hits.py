import concurrent.futures
from collections import defaultdict
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import numpy as np

def fetch_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return url, response.content
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return url, None

def build_graph(urls, max_workers=10):
    graph = defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_url, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url, content = future.result()
            if content:
                base_url = "{0.scheme}://{0.netloc}".format(urlparse(url))
                soup = BeautifulSoup(content, 'html.parser')
                for a in soup.find_all('a', href=True):
                    link = urljoin(base_url, a['href'])
                    if link.startswith('http'):
                        graph[url].append(link)
    return graph

def initialize_scores(graph):
    nodes = set(graph.keys())
    for neighbors in graph.values():
        nodes.update(neighbors)
    authority_scores = {node: 1.0 for node in nodes}
    hub_scores = {node: 1.0 for node in nodes}
    return authority_scores, hub_scores

def update_scores(graph, authority_scores, hub_scores):
    new_authority_scores = {node: 0.0 for node in authority_scores}
    new_hub_scores = {node: 0.0 for node in hub_scores}

    for node in graph:
        for neighbor in graph[node]:
            new_authority_scores[neighbor] += hub_scores[node]

    for node in graph:
        for neighbor in graph[node]:
            new_hub_scores[node] += new_authority_scores[neighbor]

    return new_authority_scores, new_hub_scores

def normalize_scores(scores):
    norm = np.linalg.norm(list(scores.values()))
    for node in scores:
        scores[node] /= norm
    return scores

def hits(graph, max_iterations=100, tol=1e-6):
    authority_scores, hub_scores = initialize_scores(graph)

    for _ in range(max_iterations):
        new_authority_scores, new_hub_scores = update_scores(graph, authority_scores, hub_scores)
        new_authority_scores = normalize_scores(new_authority_scores)
        new_hub_scores = normalize_scores(new_hub_scores)

        if (np.allclose(list(authority_scores.values()), list(new_authority_scores.values()), atol=tol) and
            np.allclose(list(hub_scores.values()), list(new_hub_scores.values()), atol=tol)):
            break

        authority_scores, hub_scores = new_authority_scores, new_hub_scores

    return authority_scores, hub_scores

def filter_authorities(urls):
    graph = build_graph(urls)
    authority_scores, _ = hits(graph)
    sorted_authorities = sorted(authority_scores.items(), key=lambda x: x[1], reverse=True)
    return [url for url, score in sorted_authorities]

if __name__ == "__main__":
    urls = [
        "https://www.bbc.com",
        "https://www.cnn.com",
        "https://www.nytimes.com"
    ]

    top_authorities = filter_authorities(urls)
    print("Top authority URLs:", top_authorities)
