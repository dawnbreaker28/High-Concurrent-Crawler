import threading
from utils.kafka_instance import producer
import json

def send_message(topic, message): 
    """
    使用全局KafkaProducer实例发送消息
    Sends a message to the specified Kafka topic.

    Args:
        topic (str): The name of the Kafka topic.
        message (bytes): The message to send. Must be in bytes format.
        
    Output:
        None: This function sends a message and doesn't return any value.
    """
    try:
        producer.send(topic, message)
        producer.flush()  # Ensure the message is sent immediately
        print(f"Message sent to {topic}")
    except Exception as e:
        print(f"Failed to send message to {topic}: {e}")


def send_message_unknown_type(topic, content):
    if isinstance(content, bytes):
        message_json = content
    else:
        try:
            message_json = json.dumps(content).encode('utf-8')
            print(f"Serialized message")  # Debug log
        except Exception as e:
            print(f"Failed to serialize message")
            return
    
    try:
        producer.send(topic, message_json)
        producer.flush()
        print(f"Message sent to {topic}")
    except Exception as e:
        print(f"Failed to send message: {e}")



def start_producer_thread(topic, message):
    """创建并启动一个线程来发送消息"""
    producer_thread = threading.Thread(
        target=send_message,
        args=(topic, message)
    )
    producer_thread.start()
    return producer_thread