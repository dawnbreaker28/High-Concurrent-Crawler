from crawler import fetcher, parser, bfs, hits
from utils import file_io, logger
import config

def main():
    urls = file_io.read_urls(config.SOURCE_FILE)
    results = bfs.breadth_first_search(urls, depth=config.SEARCH_DEPTH)
    authority_urls = hits.filter_authorities(results)
    for url in authority_urls:
        content = fetcher.fetch_content(url)
        file_io.save_content(url, content, config.RESULT_DIR)
    print("Crawling complete.")

if __name__ == "__main__":
    main()
