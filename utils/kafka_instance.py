# producer_instance.py
from kafka import KafkaProducer
import json

# 创建全局的KafkaProducer实例
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    # value_serializer=lambda v: json.dumps(v).encode('utf-8') 自动触发
)

