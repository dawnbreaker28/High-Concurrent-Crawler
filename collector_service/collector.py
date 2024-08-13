from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
import json

# 创建Elasticsearch实例
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

def store_to_elasticsearch(index, data):
    es.index(index=index, body=data)

def consume_and_store():
    # 创建Kafka消费者，订阅'categorizer.news'主题
    consumer = KafkaConsumer('categorizer.news', bootstrap_servers='localhost:9092',
                             value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    
    for message in consumer:
        # 消息内容已经反序列化为字典
        msg_data = message.value
        category = msg_data.get('category')
        text = msg_data.get('text')
        
        print(f"received message from categorizer: {category}")

        if not category:
            print("Error: No category provided, cannot store to Elasticsearch.")
            continue

        # 将消息存储到Elasticsearch中
        store_to_elasticsearch(category, {"text": text})

if __name__ == "__main__":
    consume_and_store()
