from flask import Flask, jsonify
from elasticsearch import Elasticsearch

app = Flask(__name__)
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

@app.route('/news', methods=['GET'])
def get_news():
    res = es.search(index='news', body={"query": {"match_all": {}}})
    return jsonify(res['hits']['hits'])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
