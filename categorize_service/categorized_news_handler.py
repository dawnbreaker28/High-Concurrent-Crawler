import redis
import json
from kafka import KafkaConsumer

# 初始化 Redis 连接
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

def consume_and_store_news(news_data):
    try:
        # 假设新闻数据中有 'id' 和 'content' 字段
        news_id = news_data.get('id')
        news_content = news_data.get('content')

        if news_id and news_content:
            # 使用 Redis 的 HASH 结构存储新闻
            redis_client.hset(f'news:{news_id}', 'content', news_content)

            # 如果有其他字段，也可以继续存储
            redis_client.hset(f'news:{news_id}', 'category', news_data.get('category', 'unknown'))
            redis_client.hset(f'news:{news_id}', 'timestamp', news_data.get('timestamp', ''))

            print(f"News {news_id} stored in Redis.")
        else:
            print("Invalid news data received.")
    except Exception as e:
        print(f"Error storing news to Redis: {e}")

if __name__ == '__main__':
    consume_and_store_news()
