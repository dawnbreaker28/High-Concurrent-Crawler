from sklearn.cluster import KMeans
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils.elasticsearch_instance import ESClient
from publisher_api.clustering import cluster_news, update_news_clusters

es = ESClient.get_instance()

def fetch_news_data():
    # 从Elasticsearch中提取新闻数据
    res = es.search(index="news_index", body={"query": {"match_all": {}}})
    news_texts = [hit["_source"]["text"] for hit in res['hits']['hits']]
    return news_texts

def publisher():
    # 从Elasticsearch中提取新闻数据
    news_texts = fetch_news_data()

    # 对新闻数据进行聚类分析
    labels = cluster_news(news_texts, num_clusters=5)

    # 将聚类结果更新到Elasticsearch中
    update_news_clusters(news_texts, labels)


