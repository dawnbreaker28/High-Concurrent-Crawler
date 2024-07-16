from crawler import fetcher, parser, bfs, hits
from utils import file_io, logger
import config

def main():
    urls = file_io.read_urls(config.SOURCE_FILE)
    print(f"Read {len(urls)} URLs from source file.")
    results, graph = bfs.breadth_first_search(urls, depth=config.SEARCH_DEPTH)
    print(f"BFS completed, found {len(results)} URLs.")
    authority_urls = hits.filter_authorities(results, graph)
    print(f"Filtered {len(authority_urls)} authority URLs.")
    for url in authority_urls:
        url, content = fetcher.fetch_content(url)
        if content:
            file_io.save_content(url, content, config.RESULT_DIR)
            print(f"Saved content for {url}.")
        else:
            print(f"Failed to fetch content for {url}.")
    print("Crawling complete.")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
