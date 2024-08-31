from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json


# 初始化Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# 定义新闻计数器和触发聚类的阈值
news_count = 0
CLUSTER_THRESHOLD = 1000

# 定义从Elasticsearch获取新闻并进行聚类的函数
def count_and_cluster():
    news_count += 1
    # 检查是否达到聚类触发阈值
    if news_count >= CLUSTER_THRESHOLD:
        cluster_news()
        news_count = 0  # 重置计数器


def cluster_news():

    # 从Elasticsearch中获取新闻数据
    result = es.search(index="news_index", body={"query": {"match_all": {}}}, size=10000)
    texts = [hit['_source']['text'] for hit in result['hits']['hits']]

    # 计算TF-IDF矩阵
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    # 聚类算法（例如KMeans）
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)

    # 将聚类结果存回Elasticsearch（可以根据需求优化保存的内容）
    for i, cluster_id in enumerate(kmeans.labels_):
        doc_id = result['hits']['hits'][i]['_id']
        es.update(index="news_index", id=doc_id, body={"doc": {"cluster_id": int(cluster_id)}})

    print("Clustering completed and stored in Elasticsearch.")
