from transformers import pipeline
from kafka import KafkaConsumer, KafkaProducer
import json

classifier = pipeline('text-classification')

def classify_news(news):
    result = classifier(news['content'])
    return result[0]['label']

def consume_and_produce():
    consumer = KafkaConsumer('producer.news', bootstrap_servers='localhost:9092',
                             value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    producer = KafkaProducer(bootstrap_servers='localhost:9092',
                             value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    for message in consumer:
        news = message.value
        category = classify_news(news)
        news['category'] = category
        producer.send('categorizer.news', news)
        producer.flush()

if __name__ == "__main__":
    consume_and_produce()
