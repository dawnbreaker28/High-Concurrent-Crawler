import requests
from bs4 import BeautifulSoup
import json
from kafka import KafkaProducer

def send_to_kafka(topic, data):
    producer = KafkaProducer(bootstrap_servers='localhost:9092',
                             value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    producer.send(topic, data)
    producer.flush()