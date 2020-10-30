from collections import defaultdict
import re
import math
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import pickle
from scipy.sparse import csr_matrix
import copy

# TODO
# Implement dictionary as B-tree


class Dictionary:
    def __init__(self):
        # self.dictionary = defaultdict(list)
        self.dictionary = dict()
        # self.doc2vec = defaultdict(list)
        # self.doc2vec = dict()
        self.inverted_index_filename = 'inverted_index_all'
        self.doc2term_index_filename = 'doc2term_index_all'
        self.doc2vec_filename = 'doc2vec_all'
        self.doc2vec = []
        self.doc_set = set()
        self.terms_cf = defaultdict(int)
        self.weighting_scheme = 2

    def existed_in_dictionary(self, token):
        return token in self.dictionary.keys()

    def add_term_to_dictionary(self, positional, doc_id):
        current_df = self.dictionary[positional[0]][0]
        current_postings = list(self.dictionary[positional[0]][1])
        current_postings.append((doc_id, positional[1]))
        self.dictionary[positional[0]] = (current_df+1, current_postings)
        # self.dictionary[positional[0]].append((doc_id, positional[1]))

    def get_dictionary(self):
        return self.dictionary

    def get_posting_list(self, term):
        return self.dictionary.get(term)

    def get_all_doc_id(self):
        return self.doc_set

    def compressed_doc2vec(self, docId, terms):
        compressed_vector = {}
        for term in terms:
            posting = self.get_posting_list(term)
            df = posting[0]
            posting = posting[1]
            tf = 0
            for i in range(len(posting)):
                if posting[i][0] == docId:
                    tf = len(posting[i][1])
                    break
            tfidf_score = self.tf_idf(tf, df, weighting_scheme=self.weighting_scheme)
            compressed_vector[term] = tfidf_score
        return compressed_vector

    def query2vec(self, terms):
        term_tf_df = dict()
        query_vector = np.zeros(len(self.dictionary.keys()))
        term2key = list(self.dictionary.keys())
        for term in terms:
            if term in term_tf_df.keys():
                (tf, df) = term_tf_df[term]
                term_tf_df[term] = (tf + 1, df)
            else:
                try:
                    term_tf_df[term] = (1, self.get_posting_list(term)[0])
                except:
                    term_tf_df[term] = (1, 0)
        max_tf = 1
        for value in term_tf_df.values():
            if value[0] > max_tf:
                max_tf = value[0]

        for term in term_tf_df:
            if term in self.dictionary.keys():
                query_vector[term2key.index(term)] = self.query_tf_idf(term_tf_df[term][0],
                                                     term_tf_df[term][1], max_tf,
                                                     weighting_scheme=self.weighting_scheme)
                # term_tf_df[term] = self.query_tf_idf(term_tf_df[term][0],
                #                                      term_tf_df[term][1], max_tf,
                #                                      weighting_scheme=self.weighting_scheme)
        # return term_tf_df
        return query_vector
    # def docId2vec(self, doc_id):
    #     try:
    #         return self.doc2vec[doc_id]
    #     except:
    #         return

    def get_postings(self, terms):
        postings = []
        for term in terms:
            w = self.dictionary.get(term)
            if w is not None:
                postings.append(w)
            else:
                postings.append((0, []))
        return postings

    def tf_idf(self, tf, df, weighting_scheme=None):
        switcher = {
            1: tf * math.log(len(self.get_all_doc_id()) / (df + 0.25)),
            2: (1 + math.log(tf)),
            3: (1 + math.log(tf)) * math.log(len(self.get_all_doc_id()) / (df + 0.25))
        }
        default = (1 + math.log(tf)) * math.log(len(self.get_all_doc_id()) / (df + 0.25))
        return switcher.get(weighting_scheme, default)

    def query_tf_idf(self, tf, df, max_tf, weighting_scheme=None):
        # print(max_tf)
        switcher = {
            1: (0.5 + 0.5 * tf / max_tf) * math.log(len(self.get_all_doc_id()) / (df + 0.25)),
            2: math.log(1 + (len(self.get_all_doc_id()) / (df + 0.25))),
            3: (1 + math.log(tf)) * math.log(len(self.get_all_doc_id()) / (df + 0.25))
        }
        default = (1 + math.log(tf)) * math.log(len(self.get_all_doc_id()) / (df + 0.25))
        return switcher.get(weighting_scheme, default)

    def init(self):
        import time
        # TODO: LOADING INVERTED_INDEX
        print("Loading dictionary and postings...", end='\t')

        with open('../' + self.inverted_index_filename + '.pickle', 'rb') as inverted_index:

            dictionary = pickle.load(inverted_index)
        self.dictionary = dictionary

        print("Done!")
        print("Loading document tf-idf...")
        self.load_doc2vec()
        return self.dictionary

    def save_dov2vec(self, doc2term):
        # with open('../' + self.doc2term_index_filename + '.pickle', 'rb') as doc2term:
        #     doc2term_index = pickle.load(doc2term)

        doc2term_index = doc2term
        print(len(doc2term_index))
        self.doc_set = doc2term_index
        term2key = list(self.dictionary.keys())

        for key in sorted(doc2term_index.keys()):
            doc_vector = np.zeros(len(self.dictionary.keys()))
            print('Creating vector for doc: ' + str(key))
            for tup in doc2term_index[key]:
                term = str(tup[0])
                tf = int(tup[1])
                df = int(self.get_posting_list(term)[0])
                tf_idf = self.tf_idf(tf, df, weighting_scheme=self.weighting_scheme)
                doc_vector[term2key.index(term)] = tf_idf
            # self.doc2vec[key] = doc_vector
            # tmp_doc2vec.append(csr_matrix(doc_vector))
            self.doc2vec.append(csr_matrix(doc_vector))

        with open('../' + self.doc2vec_filename + '.pickle', 'wb') as handle:
            pickle.dump(self.doc2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_doc2vec(self):
        with open('../' + self.doc2vec_filename + '.pickle', 'rb') as doc2vec:
            doc2vec_tmp = pickle.load(doc2vec)
        print(len(doc2vec_tmp))
        self.doc_set = doc2vec_tmp
        self.doc2vec = doc2vec_tmp


if __name__ == '__main__':
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    d = Dictionary()
    d.init()
    # print(d.dictionary['استرالیا'])
    # print(len(d.doc2vec[1]))
    # print(type(d.doc2vec[1]))
    # print(d.doc2vec[4])
    # print(np.count_nonzero(d.doc2vec[0]))
    # query = 'گاو گاو گاو گاو مربی ورزشی ورزشی ایران'
    # print(d.get_posting_list('استرالیا'))
    # print(d.compressed_doc2vec(1337, ['استرالیا']))
    # print(d.compressed_doc2vec(30, ['استرالیا']))
    # print(d.query2vec(query.split(' ')))
    # print(d.doc2vec[5])
    # print(len(d.doc2vec[7]))

    # print(p)
    # strs = '[(0, [6, 27, 36, 106, 136, 164]), (1, [27, 49, 84, 106, 193, 215]), (4, [52, 56])]'
    # ss = [" ".join(x.split()) for x in re.split(r'[()]', strs) if x.strip()]
    # tokens = list(filter(','.__ne__, ss))
    # tokens = list(filter('['.__ne__, tokens))
    # tokens = list(filter(']'.__ne__, tokens))
    #
    # print(tokens)
    # for token in tokens:
    #     ss = [" ".join(x.split()) for x in re.split(r'[, ]', token) if x.strip()]
    #     ss = token.split(', ')
    #     ss[1] = ss[1][1:]
    #     ss[len(ss)-1] = ss[len(ss)-1][0:(len(ss[len(ss)-1]) - 1)]
    #     print(ss)
