from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
import json

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

def store_to_elasticsearch(index, doc_type, data):
    es.index(index=index, doc_type=doc_type, body=data)

def consume_and_store():
    consumer = KafkaConsumer('categorizer.news', bootstrap_servers='localhost:9092',
                             value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    for message in consumer:
        news = message.value
        store_to_elasticsearch('news', 'doc', news)

if __name__ == "__main__":
    consume_and_store()
