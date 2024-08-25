from producer_api import fetcher, parser, bfs, hits, send_kafka
from utils import file_io, logger, kafka_helper, kafka_instance
from categorize_service import categorizer
from collector_service import collector
import config
from kafka import KafkaConsumer
import json
import threading


def main():

    kafka_helper.create_topics()

    # producer_thread = kafka_helper.KafkaProducerThread(kafka_instance.producer, "producer.news")
    
    collector_thread = threading.Thread(
        target=kafka_helper.start_consumer,
        args=('categorizer.news', 'categorizer_group', collector.consume_and_store)
    )
    collector_thread.start()

    categorizer_thread = threading.Thread(
        target=kafka_helper.start_consumer,
        args=('producer.news', 'producer_group', categorizer.consume_and_produce)
    )
    categorizer_thread.start()

    urls = file_io.read_urls(config.SOURCE_FILE)
    print(f"Read {len(urls)} URLs from source file.")
    results, graph = bfs.breadth_first_search(urls, depth=config.SEARCH_DEPTH)
    print(f"BFS completed, found {len(results)} URLs.")
    authority_urls = hits.filter_authorities(results, graph)
    print(f"Filtered {len(authority_urls)} authority URLs.")
    
    for url in authority_urls:
        url, content = fetcher.fetch_content(url)
        if content:
            # file_io.save_content(url, content, config.RESULT_DIR)
            print(f"Saved content.")
            output = {"content": content}
            send_kafka.send_message_unknown_type("producer.news", output)
        else:
            print(f"Failed to fetch content for.")
    print("Crawling complete.")

    categorizer_thread.join()
    collector_thread.join()

if __name__ == "__main__":
    main()
