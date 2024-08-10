from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from kafka import KafkaConsumer, KafkaProducer
import json

classifier = pipeline('text-classification')

def classify_news(message):

    # 如果 message 是字符串，先解析为字典
    if isinstance(message, str):
        try:
            message = json.loads(message)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None, None
    
    html_content = message.get('html_content', '')  # 从消息中获取 HTML 内容
    if not html_content:
        print("Error: html_content is None or empty")
        return None, None

    # 尝试解析HTML内容
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception as e:
        print(f"Error parsing HTML content: {e}")
        return None, None
    text = soup.get_text()

    # 加载预训练的BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # 使用pipeline进行分类
    classifier = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer)

    # 定义可能的新闻分类标签
    candidate_labels = ['sports', 'politics', 'technology', 'entertainment', 'business']

    # 进行分类预测
    result = classifier(text, candidate_labels)
    return result[0]['label'], text

def consume_and_produce(message):
    # consumer = KafkaConsumer('producer.news', bootstrap_servers='localhost:9092',
    #                          value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    producer = KafkaProducer(bootstrap_servers='localhost:9092',
                             value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    # for message in consumer:
    print("message received from producer.news")
    category, text= classify_news(message)
    # message['category'] = category
    output = [category, text]
    producer.send('categorizer.news', output)
    print("send message to collector")
    producer.flush()

if __name__ == "__main__":
    consume_and_produce()
