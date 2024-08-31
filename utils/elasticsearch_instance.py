from elasticsearch import Elasticsearch

class ESClient:
    _instance = None

    @staticmethod
    def get_instance():
        if ESClient._instance is None:
            ESClient._instance = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
        return ESClient._instance