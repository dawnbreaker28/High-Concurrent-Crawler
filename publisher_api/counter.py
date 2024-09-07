from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from utils import elasticsearch_instance
import json


# 初始化Elasticsearch
es = elasticsearch_instance.ESClient()

# 定义新闻计数器和触发聚类的阈值
news_count = 0
CLUSTER_THRESHOLD = 100

# 定义从Elasticsearch获取新闻并进行聚类的函数
def count_and_cluster():
    news_count += 1
    # 检查是否达到聚类触发阈值
    if news_count >= CLUSTER_THRESHOLD:
        print("Threshold reached")
        cluster_news()
        news_count = 0  # 重置计数器


def cluster_news():

    # 从Elasticsearch中获取新闻数据
    result = es.search(index="news", body={"query": {"match_all": {}}}, size=10000)
    # 构建特征矩阵
    feature_matrix = []
    for hit in result['hits']['hits']:
        keywords_with_scores = hit['_source']['keywords_with_scores']
        print("keywords get:")
        print(keywords_with_scores)
        feature_vector = [score for _, score in keywords_with_scores]
        feature_matrix.append(feature_vector)
    
    # 转换为numpy矩阵
    import numpy as np
    X = np.array(feature_matrix)
    
    # 聚类算法（例如KMeans）
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)

    # 将聚类结果存回Elasticsearch
    for i, cluster_id in enumerate(kmeans.labels_):
        doc_id = result['hits']['hits'][i]['_id']
        es.update(index="news_index", id=doc_id, body={"doc": {"cluster_id": int(cluster_id)}})

    print("Clustering completed and stored in Elasticsearch.")
