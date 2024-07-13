from crawler import fetcher, parser
from utils import logger

def breadth_first_search(urls, depth=3):
    visited = set()
    to_visit = urls
    for d in range(depth):
        next_to_visit = []
        logger.log(f"Depth {d}: Visiting {len(to_visit)} URLs")
        results = fetcher.fetch_all(to_visit)
        for url, content in results.items():
            if content:
                links = parser.parse_links(url, content)
                for link in links:
                    if link not in visited:
                        visited.add(link)
                        next_to_visit.append(link)
        to_visit = next_to_visit
    return visited
