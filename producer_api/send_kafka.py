import threading
from utils.kafka_instance import producer
import json

def send_message(topic, message):
    """使用全局KafkaProducer实例发送消息"""
    producer.send(topic, message)
    producer.flush()  # 确保消息被立即发送
    print(f"Message sent to {topic}")


def send_message_to_unknown_type(topic, content):
    # 假设消费者期望接收的消息是JSON格式的
    if isinstance(content, bytes):
        message = content.decode('utf-8')
    else:
        message = {'html_content': content}
    
    try:
        message_json = json.dumps(message)
        print("Serialized message: ")  # Debug log
    except Exception as e:
        print("Failed to serialize message:")
        return
    
    producer.send(topic, message_json.encode('utf-8'))
    producer.flush()


def start_producer_thread(topic, message):
    """创建并启动一个线程来发送消息"""
    producer_thread = threading.Thread(
        target=send_message,
        args=(topic, message)
    )
    producer_thread.start()
    return producer_thread