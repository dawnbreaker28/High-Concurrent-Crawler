from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from kafka import KafkaConsumer, KafkaProducer
from producer_api import send_kafka
import json

# 预先加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# 预先初始化分类器
classifier = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer, device=0)  # 使用GPU


def classify_news(message):

    # 如果 message 是字符串，先解析为字典
    if isinstance(message, str):
        try:
            message = json.loads(message)
        except json.JSONDecodeError as e:
            print("Error decoding JSON")
            return None, None
    elif not isinstance(message, dict):
        print(f"Unexpected message format: {type(message)}")
        return None, None  
    html_content = message.get('html_content', '')  # 从消息中获取 HTML 内容
    if not html_content:
        print("Error: html_content is None or empty")
        return None, None

    # 尝试解析HTML内容
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception as e:
        print("Error parsing HTML content")
        return None, None
    text = soup.get_text()

    # 定义可能的新闻分类标签
    candidate_labels = ['sports', 'politics', 'technology', 'entertainment', 'business']

    # 进行分类预测
    result = classifier(text, candidate_labels)
    return result['labels'][0], text

def consume_and_produce(message):

    # 接收并处理消息
    print("message received from producer.news")
    category, text = classify_news(message)

    if category is None:
        print("Failed to classify message.")
        return

    # 创建输出消息
    output = {
        'category': category,
        'text': text
    }

    # 将消息转换为JSON格式并发送
    output_json = json.dumps(output)
    print("send message to collector")
    send_kafka.send_message('categorizer.news', output_json)


if __name__ == "__main__":
    consume_and_produce()
