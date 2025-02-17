from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaConsumer, KafkaProducer
import json
import threading
import time
from kafka.errors import TopicAlreadyExistsError
from categorize_service import categorizer, categorized_news_handler
from collector_service import collector
from publisher_api import counter

class KafkaConsumerThread:
    def __init__(self, topic, group, handler):
        self.topic = topic
        self.group = group
        self.handler = handler
        self.thread = threading.Thread(target=self.start_consumer)

    def start_consumer(self):
        start_consumer(self.topic, self.group, self.handler)

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()

class KafkaThreadManager:
    def __init__(self):
        self.consumers = []

    def start_all(self):
        self.consumers = [
        KafkaConsumerThread('counter.news', 'counter_group', counter.count_and_cluster),
        KafkaConsumerThread('categorizer.news', 'categorizer_group', collector.consume_and_store),
        KafkaConsumerThread('producer.news', 'producer_group', categorizer.consume_and_produce),
        ]
        labels = [ "sport", "business",  "tech", "entertainment",  "politics"]
        for label in labels:
            self.consumers.append(KafkaConsumerThread(label, label, categorized_news_handler.consume_and_store_news))
        
        for consumer in self.consumers:
            consumer.start()

    def end_all(self):
        for consumer in self.consumers:
            consumer.join()

class KafkaProducerThread(threading.Thread):
    def __init__(self, producer, topic, message):
        threading.Thread.__init__(self)
        self.producer = producer
        self.topic = topic
        self.message = message

    def run(self):
        try:
            self.producer.send(self.topic, self.message)
            print(f"Message sent to {self.topic}")
            self.producer.flush()  # Ensure the message is sent
        except Exception as e:
            print(f"Failed to send message to {self.topic}: {e}")

def create_topics():
    admin_client = KafkaAdminClient(
        bootstrap_servers="localhost:9092",
        client_id='test_client'
    )

    topic_list = []
    topic_list.append(NewTopic(name="producer.news", num_partitions=1, replication_factor=1))
    topic_list.append(NewTopic(name="categorizer.news", num_partitions=1, replication_factor=1))
    topic_list.append(NewTopic(name="collector.news", num_partitions=1, replication_factor=1))

    try:
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
    except TopicAlreadyExistsError as e:
        print(f"Topics already exist: {e}")

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
    for message in consumer:  # message:ConsumerRecord
        # Decode the Kafka message
        message_content = message.value.decode('utf-8') #.value->bytes, .decode->string
        # print(f"message received for {topic} and content {message_content}")
        try: # json.loads->dict
            message_data = json.loads(message_content)  # Ensure message is deserialized correctly
        except json.JSONDecodeError:
            print(f"Failed to decode JSON message")
            continue  
        # Call the appropriate consume_and_store function
        # Check if the message is in the expected format (i.e., a dict)
        if not isinstance(message_data, dict):
            print(f"Unexpected message format, skipping")
            continue  # Skip if the message is not a dictionary
        
        # Call the appropriate consume_and_store function
        consume_and_store_func(message_data)

