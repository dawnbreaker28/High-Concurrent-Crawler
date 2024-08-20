from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
import json

# 创建Elasticsearch实例
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

def store_to_elasticsearch(index, data):
    es.index(index=index, body=data)

def consume_and_store(message):
    # # 创建Kafka消费者，订阅'categorizer.news'主题
    # consumer = KafkaConsumer('categorizer.news', bootstrap_servers='localhost:9092',
    #                          value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    
     # If message_data is still a JSON string, decode it
    
    

    if isinstance(message, str):
        message = json.loads(message)
        print("message is String")
    
    category = message['category']
    text = message['text']
    print(f"Received message from categorizer: {category}")
    
    if not category:
        print("Error: No category provided, cannot store to Elasticsearch.")
        return

    # 将消息存储到Elasticsearch中
    store_to_elasticsearch(category, {"text": text})

if __name__ == "__main__":
    consume_and_store()
