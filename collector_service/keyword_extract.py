import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict

# 下载停止词
nltk.download('stopwords')

# 停止词和词干提取器
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# 步骤 2: 计算TF-IDF
vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# 步骤 1: 数据预处理
def preprocess(text):
    # 去除标点和特殊字符
    text = re.sub(r'\W', ' ', text)
    # 转为小写
    text = text.lower()
    # 去除停止词
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # 词干提取
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def preprocess_and_extract_keywords(news_text):
    # 预处理和关键词提取代码与之前类似
    # 对新闻文本进行预处理...
    preprocessed_text = preprocess(news_text)
    
    # 使用TF-IDF提取关键词
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
    feature_names = vectorizer.get_feature_names_out()
    keywords = extract_keywords(tfidf_matrix, feature_names)
    
    return keywords


# 步骤 3: 关键词提取
def extract_keywords(tfidf_matrix, feature_names, top_n=5):
    keywords_with_scores = []
    for row in tfidf_matrix:
        # 获取每个词的TF-IDF分数
        row_array = row.toarray().flatten()
        # 按TF-IDF分数排序并提取最高的top_n个特征
        sorted_items = row_array.argsort()[-top_n:][::-1]
        # 获取关键词和对应的分数
        keywords_with_scores.append([(feature_names[i], row_array[i]) for i in sorted_items])
    return keywords_with_scores

# # 步骤 4: 聚类
# num_clusters = 5  # 假设我们需要5个簇
# km = KMeans(n_clusters=num_clusters)
# km.fit(tfidf_matrix)

# clusters = km.labels_.tolist()

# # 步骤 5: 生成集群结果
# clustered_news = defaultdict(list)
# for i, cluster in enumerate(clusters):
#     clustered_news[cluster].append(news_data[i])

# # 打印集群结果
# for cluster_id, cluster_news in clustered_news.items():
#     print(f"Cluster {cluster_id}:")
#     cluster_keywords = extract_keywords(tfidf_matrix[clusters == cluster_id], feature_names, top_n=5)
#     print("Keywords:", cluster_keywords)
#     for news in cluster_news:
#         print(news[:200])  # 打印前200个字符作为示例
#     print("\n")
