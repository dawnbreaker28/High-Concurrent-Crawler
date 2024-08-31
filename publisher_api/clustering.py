from sklearn.cluster import KMeans
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils.elasticsearch_instance import ESClient

es = ESClient.get_instance()

def cluster_news(news_texts, num_clusters=5):
    # 使用TF-IDF对新闻数据进行向量化
    vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    X = vectorizer.fit_transform(news_texts)
    
    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    
    return kmeans.labels_

def update_news_clusters(news_texts, labels):
    # 将聚类标签更新到Elasticsearch中的新闻文档
    for i, text in enumerate(news_texts):
        es.update_by_query(
            index="news_index",
            body={
                "query": {"match": {"text": text}},
                "script": {
                    "source": "ctx._source.cluster_id = params.cluster_id",
                    "params": {"cluster_id": int(labels[i])}
                }
            }
        )