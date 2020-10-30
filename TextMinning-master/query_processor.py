import re
from dictionary import Dictionary
import json

from preprocess.normalizer import Normalizer


class QueryProcessor:
    def __init__(self):
        pass

    @staticmethod
    def pre_process(query):
        query = Normalizer.normalize(query)
        return query

    @staticmethod
    def parse(query, cat=None, source=None):
        query = QueryProcessor.pre_process(query)
        jquery = []
        delta = 0
        for m in re.finditer(r'"([^"]*)"', query):
            word = query[m.start() - delta:m.end() - delta].strip("\"\'")
            words = word.split(' ')
            words = Normalizer.remove_stopwords([Normalizer.lem(normalized) for normalized in words if Normalizer.lem(normalized) != ""])
            if len(words) != 0:
                jquery.append({
                    'type': '2',
                    'term': words
                })
            query = query[:m.start() - delta] + query[m.end() - delta:]
            delta = m.end() - m.start()

        delta = 0
        for m in re.finditer(r'[!]\w*', query):
            word = query[m.start() - delta:m.end() - delta].strip("! ")
            word = Normalizer.remove_stopwords([Normalizer.lem(word)])
            word = [x for x in word if x != '']
            if len(word) != 0:
                jquery.append({
                    'type': '0',
                    'term': word
                })
            query = query[:m.start() - delta] + query[m.end() - delta:]
            delta = m.end() - m.start()

        p = re.compile('\\b[\\w+\u200c\\w+]+\\b')
        for m in re.finditer(p, query):
            word = query[m.start():m.end()].strip("\"\' ")
            word = Normalizer.remove_stopwords([Normalizer.lem(word)])
            if len(word) != 0:
                jquery.append({
                    'type': '1',
                    'term': word
                })

        if cat is not None:
            jquery.append({
                'type': 'cat',
                'cat-name': cat
            })

        if source is not None:
            jquery.append({
                'type': 'source',
                'source-url': source
            })

        return jquery
        # return json.dumps(jquery)

    def optimize_query(self, query):
        pass

if __name__ == '__main__':
    # query = '"استقلال از به برای من تیم" !کتاب امریکا !از بر او'
    query = ''
    jq = QueryProcessor().parse(query)
    print(jq)
