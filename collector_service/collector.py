from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
from concurrent.futures import ThreadPoolExecutor
from .keyword_extract import preprocess_and_extract_keywords
from utils.elasticsearch_instance import ESClient
import threading
import json

# 创建Elasticsearch实例
es = ESClient.get_instance()

executor = ThreadPoolExecutor(max_workers=10)  # 你可以根据需要调整线程池大小

def store_to_elasticsearch(index, data):
    # 使用线程池来异步执行 Elasticsearch 请求
    future = executor.submit(es.index, index=index, body={"doc": data, "doc_as_upsert":True})
    future.add_done_callback(lambda f: print(f"Data stored to Elasticsearch: {f.result()}"))


def consume_and_store(message):
    if isinstance(message, str):
        message = json.loads(message)
        print("message is String")
    
    category = message['category']
    text = message['text']
    print(f"Received message from categorizer: {category}")
    
    if not category:
        print("Error: No category provided, cannot store to Elasticsearch.")
        return

    keywords_with_scores = preprocess_and_extract_keywords(text)
    # 将消息存储到Elasticsearch中
    print(keywords_with_scores)
    store_to_elasticsearch("news", {"text": text, "keywords_with_score": keywords_with_scores, "category": category})


if __name__ == "__main__":
    consume_and_store()
