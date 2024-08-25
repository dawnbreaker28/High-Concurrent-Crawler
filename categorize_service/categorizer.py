from bs4 import BeautifulSoup
from kafka import KafkaConsumer, KafkaProducer
from producer_api import send_kafka
import json
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import config

def parse_text(message):
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
    
    # 获取 HTML 内容
    html_content = message.get('content', '')  
    print("get content")
    if not html_content:
        print("Error: html_content is None or empty")
        return None, None

    # 尝试解析HTML内容
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception as e:
        print("Error parsing HTML content")
        return None, None

    # 移除 script 和 style 元素
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()  # 完全移除标签

    # 移除导航、页眉和页脚（如果适用）
    for nav_footer in soup.find_all(['nav', 'footer']):
        nav_footer.decompose()

    # 尝试提取 <article> 或 <div> 内的内容，如果没有找到则提取整个页面的文本
    article = soup.find('article') or soup.find('div', class_='article-body')

    # 如果找到 <article> 或 <div class="article-body"> 标签，则获取其文本，否则获取整个页面的文本
    text = article.get_text() if article else soup.get_text()

    # 进一步清理文本，去除多余的空格
    text = ' '.join(text.split())  # 移除多余的空白字符

    return text

def classify_news(text):

    # 加载保存的模型和tokenizer
    model_dir = config.MODEL_DIR  # 替换为你保存模型的目录
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # 假设你的模型有5个类别
    label_map = {0: "sport", 1: "business", 2: "tech", 3: "entertainment", 4: "politics"}

    # 将文本tokenize并转换为输入ID和注意力掩码
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # 将模型设置为评估模式
    model.eval()

    # 将数据传递给模型进行预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # 获取预测的标签
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = label_map[predicted_class_id]

    # 打印结果
    print(f"Predicted label for the whole file: {predicted_label}")

    return predicted_label

def consume_and_produce(message):

    # 接收并处理消息
    print("message received from producer.news")
    text = parse_text(message)
    print(text[:300])
    # category = classify_news(text) # for testing 
    category = "sports"

    if category is None:
        print("Failed to classify message.")
        return

    # 创建输出消息
    output = {
        'category': "sports",
        'text': "testtext"
    }

    # 将消息转换为JSON格式并发送
    output_json = json.dumps(output)
    print("send message to collector")
    send_kafka.send_message('categorizer.news', output_json.encode('utf-8'))  # 添加 encode('utf-8')


if __name__ == "__main__":
    consume_and_produce()
