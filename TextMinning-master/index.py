import json

import pandas
from pre_processor import DocTokenizer
from dictionary import Dictionary
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import glob
import os
from mongoengine import connect
from model.news import NewsModel

doc_tokenizer = DocTokenizer()
data_set = '../IR-project-data-phase-3-100k'

connect('IREngine', host="mongodb://localhost:27017")


# def doc2file(doc_id):
def to_db():
    idx = 0
    for filename in glob.glob(os.path.join(data_set, '*.csv')):
        df = pandas.read_csv(filename, encoding='utf_8')

        for inn in range(df.shape[0]):

            idx += 1
            print('saving' + str(idx))
            doc = df.loc[inn, :]
            content = doc_tokenizer.clean_html(doc['content'])
            publish_date = doc['publish_date']
            title = doc['title']
            url = doc['url']
            summary = doc['summary']
            if type(summary) is float:
                summary = ' '
            meta_tags = json.loads(doc['meta_tags'])
            thumbnail = doc['thumbnail']
            if type(thumbnail) is float:
                thumbnail = 'https://www.bvfd.com/wp-content/uploads/2015/12/placeholder.jpg'
            if type(title) is float:
                title = "خبر"
            news = NewsModel()
            news.meta_tags = meta_tags
            news.thumbnail = thumbnail
            news.url = url
            news.content = content
            news.publish_date = publish_date
            news.summary = summary
            news.title = title
            news.news_id = idx - 1
            news.save()


def get_news_by_id(doc):
    return NewsModel.objects(news_id=doc)


def indexing():
    print("Creating index.txt...")
    dictionary = Dictionary()
    to_db()
    doc2term = dict()
    docId = 0
    for filename in glob.glob(os.path.join(data_set, '*.csv')):
        print("Creating Index of ", filename)
        df = pandas.read_csv(filename, encoding='utf_8')
        for doc in df['content']:
            # total_tokens = np.zeros(df.shape[0]) #zeapf and heapf law
            # total_terms = np.zeros(df.shape[0])  #zeapf and heapf law
            # doc2term_file = open('./doc2term.txt', 'w', encoding='utf_8')
            print('Indexing doc ', docId)
            positionals = doc_tokenizer.get_tokens(doc)

            # if i != 0:
            # total_tokens[i] = total_tokens[i - 1]               #zeapf and heapf law
            terms = []
            # doc2term_file.write(str(i) + ' => ')
            doc2term[docId] = list()

            for positional in positionals:
                # dictionary.add_term_to_dictionary(positional, i)
                # total_tokens[i] = total_tokens[i] + len(positional[1])      #zeapf and heapf law
                # terms.append((positional[0], len(positional[1])))           #zeapf and heapf law
                # doc2term_file.write(str(positional[0]) + ':' + str(len(positional[1])) + ',')
                doc2term[docId].append((str(positional[0]), len(positional[1])))
                if dictionary.existed_in_dictionary(positional[0]):
                    dictionary.add_term_to_dictionary(positional, docId)
                    # temp = dictionary.terms_cf[positional[0]]               #zeapf and heapf law
                else:
                    new_posting = list()
                    new_posting.append((docId, positional[1]))
                    dictionary.dictionary[positional[0]] = (1, new_posting)
                    # temp = 0  # zeapf and heapf law
            docId += 1
        print("Done")
        # dictionary.terms_cf[positional[0]] = len(positional[1]) + temp      #zeapf and heapf law
        # doc2term_file.write('\n')

        # total_terms[i] = len(dictionary.dictionary.keys())

    # heaps_law(total_tokens, total_terms)
    # zipfs_law(dictionary.terms_cf)
    # dict = dictionary.get_dictionary()
    # print(dictionary.get_dictionary()['ایران'])

    with open('inverted_index_500.pickle', 'wb') as handle:
        pickle.dump(dictionary.get_dictionary(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('doc2term_index_500.pickle', 'wb') as handle:
        pickle.dump(doc2term, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('./index.txt', 'w', encoding="utf_8") as f:
    #     for key in sorted(dict.keys()):
    #         f.writelines([key, " => ", str(dict[key]), "\n"])
    print("Don!")


def fetch_result(doc_list):
    # df = pandas.read_csv(data_set)
    result = []
    for doc_id in doc_list:
        # content = df['content'][doc_id]
        # content = doc_tokenizer.clean_html(content)
        # content = doc_tokenizer.text_normalizer(content)
        #
        # publish_date = df['publish_date'][doc_id]
        # title = df['title'][doc_id]
        # url = df['url'][doc_id]
        # summary = df['summary'][doc_id]
        # if type(summary) is float:
        #     summary = ' '
        # meta_tags = df['meta_tags'][doc_id]
        # thumbnail = df['thumbnail'][doc_id]
        # if type(thumbnail) is float:
        #     thumbnail = 'https://www.bvfd.com/wp-content/uploads/2015/12/placeholder.jpg'
        # result.append(news)
        q = get_news_by_id(doc_id)
        print(q)
        if len(q) > 0:
            nw = q[0]
            news = {'content': str(nw.content), 'publish_date': str(nw.publish_date), 'title': str(nw.title), 'url': str(nw.url),
                    'summary': str(nw.summary),
                    'meta_tags': nw.meta_tags, 'thumbnail': str(nw.thumbnail)}
            result.append(news)
        else:
            return []
    return result


def heaps_law(total_tokens, total_terms):
    total_tokens = np.log10(total_tokens)
    total_terms = np.log10(total_terms)
    x = np.linspace(0, total_tokens[len(total_tokens) - 1], 2000)
    y = math.log(40, 10) + (1 / 2) * x
    plt.plot(total_tokens, total_terms)
    plt.plot(x, y, '--')
    plt.xlabel('log10 T')
    plt.ylabel('log10 M')
    plt.title('Heap`s law')
    plt.savefig('./plots/heaps.png')
    plt.show()


def zipfs_law(terms_cf):
    sorted_terms_cf = [(k, terms_cf[k]) for k in sorted(terms_cf, key=terms_cf.get, reverse=True)]
    total_cf = []
    for k, v in sorted_terms_cf:
        total_cf.append(v)
    total_cf = np.array(total_cf)
    total_cf = np.log10(total_cf)
    print(total_cf)
    total_ranks = np.arange(len(total_cf))
    total_ranks = total_ranks + 1
    total_ranks = np.log10(total_ranks)
    print(total_ranks)
    x = np.linspace(0, total_ranks[len(total_ranks) - 1], 2000)
    y = math.log(10000, 10) - x
    plt.plot(total_ranks, total_cf)
    plt.plot(x, y, '--')
    plt.xlabel('log10 rank')
    plt.ylabel('log10 cf')
    plt.title('Zipf`s law')
    plt.savefig('./plots/zipfs.png')
    plt.show()


if __name__ == '__main__':
    # indexing()
    # caltest()
    to_db()
    a = get_news_by_id([10304, 12])
    print(get_news_by_id([10304, 12]))
    # print(fetch_result([108]))
    # doclist = [15]
    # result = fetch_result(doclist)
    # print(result)
    # for r in result:
    #     print(r['summary'])
