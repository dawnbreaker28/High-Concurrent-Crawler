from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaConsumer
import json
import threading

def create_topics():
    admin_client = KafkaAdminClient(
        bootstrap_servers="localhost:9092",
        client_id='test_client'
    )

    topic_list = []
    topic_list.append(NewTopic(name="producer.news", num_partitions=1, replication_factor=1))
    topic_list.append(NewTopic(name="categorizer.news", num_partitions=1, replication_factor=1))
    topic_list.append(NewTopic(name="collector.news", num_partitions=1, replication_factor=1))

    admin_client.create_topics(new_topics=topic_list, validate_only=False)

# Function to start a Kafka consumer
def start_consumer(topic, group_id, consume_and_store_func):
    # Initialize Kafka consumer
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=['localhost:9092'],  # Specify the Kafka broker address
        auto_offset_reset='earliest',          # Start reading from the earliest messages
        enable_auto_commit=True,               # Enable automatic offset commit
        group_id=group_id                      # Consumer group ID
    )
    for message in consumer:
        # Decode the Kafka message
        message_content = message.value.decode('utf-8')
        message_data = json.loads(message_content)
        # Call the appropriate consume_and_store function
        consume_and_store_func(message_data)

def start_producer(topic, bootstrap_servers='localhost:9092'):
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    while True:
        message = {"key": "value", "timestamp": time.time()}
        producer.send(topic, value=message)
        print(f"Sent message to {topic}: {message}")
        time.sleep(5)