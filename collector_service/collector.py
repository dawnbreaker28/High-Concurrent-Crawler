from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
import json

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

def store_to_elasticsearch(index, data):
    es.index(index=index, body=data)

def consume_and_store(message):
    # consumer = KafkaConsumer('categorizer.news', bootstrap_servers='localhost:9092',
    #                          value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    # for message in consumer:
    category, text = message
    print(f"received message from categorizer {category}")
    store_to_elasticsearch(category, text)

if __name__ == "__main__":
    consume_and_store()
