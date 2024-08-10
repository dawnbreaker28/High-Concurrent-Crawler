import threading
from utils.kafka_instance import producer

def send_message(topic, message):
    """使用全局KafkaProducer实例发送消息"""
    producer.send(topic, message)
    producer.flush()  # 确保消息被立即发送
    print(f"Message sent to {topic}")

def start_producer_thread(topic, message):
    """创建并启动一个线程来发送消息"""
    producer_thread = threading.Thread(
        target=send_message,
        args=(topic, message)
    )
    producer_thread.start()
    return producer_thread