import time

from clustering.kmeans import KMeans
from dictionary import Dictionary
from query_processor import QueryProcessor
import numpy as np
import heapq
import itertools
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
from classification import Classification
from index import fetch_result


class SearchEngine:
    def __init__(self):
        self.dictionary = Dictionary()
        self.dictionary.init()
        final_train_doc_index = 995
        self.classification = Classification(final_train_doc_index, knn_mode=True)

        # train_data = []
        # for i in range(0, final_train_doc_index):
        #     train_data.append(self.dictionary.doc2vec[i].toarray()[0])
        # self.classification.learn(train_data)
        # print('complete learned ...')

        # final_test_doc_index = 158200
        # counter = 1
        # while 1:
        #     test_data = []
        #     doc_ids = []
        #     if (final_test_doc_index - final_train_doc_index) < 1000:
        #         for i in range(final_train_doc_index, final_test_doc_index):
        #             test_data.append(self.dictionary.doc2vec[i].toarray()[0])
        #             doc_ids.append(i)
        #         self.classification.classify_test_data(test_data, doc_ids)
        #         break
        #     else:
        #         for i in range(final_train_doc_index, final_train_doc_index + 1000):
        #             test_data.append(self.dictionary.doc2vec[i].toarray()[0])
        #             doc_ids.append(i)
        #         self.classification.classify_test_data(test_data, doc_ids)
        #         print('complete phase: ' + str(final_train_doc_index))
        #         counter += 1
        #         final_train_doc_index += 1000
        #
        # # TODO: SAVE CLASS LABELS
        # self.classification.save_labels()

        # TODO: LOAD CLASS LABELS
        self.classification.load_labels()
        print('complete classification ...')

        KMeans.init()
        # KMeans.create(self.dictionary.doc2vec)
        print('complete clustering ...')

        self.clustering_mode = True

    def search(self, jquery, start_time):
        query_size = len(jquery)
        if query_size == 0:
            return

        cat_query = False
        input_category = None
        if jquery[query_size - 1]['type'] == 'cat':
            self.clustering_mode = False
            cat_query = True
            tmp = jquery.pop()
            input_category = int(tmp['cat-name'])
            query_size = query_size - 1

        all_terms = []
        for i in range(query_size):
            obj = jquery[i]
            if obj['type'] != '0':
                for term in obj['term']:
                    all_terms.append(term)

        jquery.sort(key=lambda x: x['type'], reverse=True)
        priorities = [obj['type'] for obj in jquery]
        terms = jquery[0]['term']
        postings = self.dictionary.get_postings(terms)
        postings = [tup[1] for tup in postings]

        if priorities[0] == '2':
            results = self.exact_term_search(postings)
        elif priorities[0] == '1' and not cat_query and not self.clustering_mode:
            results = postings[0]
        elif priorities[0] == '0':
            results = self.negation(self.dictionary.get_all_doc_id(), postings[0])

        if query_size == 1:
            if cat_query:
                category_docs = self.classification.get_class_type(input_category)
                category_docs = [(doc_id,) for doc_id in category_docs]
                if priorities[0] == '1':
                    results = category_docs
                else:
                    results = self.basic_intersect(results, category_docs)

            if self.clustering_mode:
                # TODO: NOT CHECKING EXACT-WORD QUERY
                if priorities[0] == '1':
                    clustering_docs = self.get_related_docs_from_vector(self.dictionary.query2vec(all_terms))
                    # print(clustering_docs)
                    # clustering_docs = [(doc_id,) for doc_id in clustering_docs]
                    results = clustering_docs

            ordered_results = self.k_most_similarity(200, results, all_terms, self.dictionary.query2vec(all_terms))
            return self.to_json_result(ordered_results, time.time() - start_time)

        if cat_query:
            category_docs = self.classification.get_class_type(input_category)
            category_docs = [(doc_id,) for doc_id in category_docs]
            if priorities[0] == '1':
                results = category_docs
            else:
                results = self.basic_intersect(results, category_docs)
            # print(category_docs)
            # print(results)

        if self.clustering_mode:
            clustering_docs = self.get_related_docs_from_vector(self.dictionary.query2vec(all_terms))
            # clustering_docs = [(doc_id[0],) for doc_id in clustering_docs]
            if priorities[0] == '1':
                results = clustering_docs
            else:
                results = self.basic_intersect(results, clustering_docs)
            # results = clustering_docs
            # print('clustring docs:')
            # print(clustering_docs)
            # print(results)
            # results = self.basic_intersect(results, clustering_docs)

        for i in range(1, query_size):
            postings = self.dictionary.get_postings(jquery[i]['term'])
            postings = [tup[1] for tup in postings]
            if jquery[i]['type'] == '2':
                exact_term = self.exact_term_search(postings)
                results = self.basic_intersect(results, exact_term)
            elif jquery[i]['type'] == '1' and not cat_query and not self.clustering_mode:
                results = self.basic_intersect(results, postings[0])
            elif jquery[i]['type'] == '0':
                results = self.basic_intersect_negation(results, postings[0])
            if len(results) == 0:
                print("result not found")
                return self.to_json_result(results, time.time() - start_time)

        # if cat_query:
        #     category_docs = self.classification.get_class_type(input_category)
        #     category_docs = [(doc_id,) for doc_id in category_docs]
        #     print(category_docs)
        #     print(results)
        #     results = self.basic_intersect(results, category_docs)
        #
        # if self.clustering_mode:
        #     clustering_docs = self.get_related_docs_from_vector(self.dictionary.query2vec(all_terms))
        #     clustering_docs = [(doc_id[0][0],) for doc_id in clustering_docs]
        #     print(clustering_docs)
        #     results = self.basic_intersect(results, clustering_docs)
        ordered_results = self.k_most_similarity(200, results, all_terms, self.dictionary.query2vec(all_terms))
        return self.to_json_result(ordered_results, time.time() - start_time)

    @staticmethod
    def get_related_docs_from_vector(vector):
        clus = KMeans.get_k_means()
        most_clusters = clus.get_cluster(vector)
        all_res = []
        for cluster in most_clusters:
            all_res.extend(clus.get_cluster_data(cluster))

        res = list(itertools.zip_longest(all_res, [], fillvalue=[]))
        return res

    @staticmethod
    def negation(superset, p):
        dict_p = dict(p)
        res = [(term,) for term in superset if term not in dict_p]
        return res

    @staticmethod
    def basic_intersect(p1, p2):
        result = []
        i1 = i2 = 0
        while i1 < len(p1) and i2 < len(p2):
            if p1[i1][0] == p2[i2][0]:
                result.append((p1[i1][0], list(set(p1[i1][1]).union(p2[i2][1]))))
                i1 += 1
                i2 += 1
            elif p1[i1][0] < p2[i2][0]:
                i1 += 1
            else:
                i2 += 1
        # print('Intersect {} and {} is {}'.format(str(p1), str(p2), str(result)))
        return result

    @staticmethod
    def basic_intersect_negation(p1, np2):
        result = []
        i1 = 0
        i2 = 0
        while i1 < len(p1) and i2 < len(np2):
            if p1[i1][0] < np2[i2][0]:
                if len(p1[i1]) == 1:
                    result.append((p1[i1][0],))
                else:
                    result.append((p1[i1][0], p1[i1][1]))
                i1 += 1
            elif p1[i1][0] == np2[i2][0]:
                i1 += 1
                i2 += 1
            else:
                i2 += 1
        if i2 >= len(np2):
            [result.append(x) for x in p1[i1:]]
        # print('Intersect {} and Not {} is {}'.format(str(p1), str(np2), str(result)))
        return result

    @staticmethod
    def basic_exact_term_search(p1, p2, k):
        results = []
        i1 = 0
        i2 = 0
        while i1 < len(p1) and i2 < len(p2):
            if p1[i1][0] == p2[i2][0]:
                l = []
                pp1 = p1[i1][1]
                pp2 = p2[i2][1]
                for position in pp1:
                    if (position + k) in pp2:
                        l.append(position)
                if len(l) != 0:
                    results.append((p1[i1][0], l))
                i1 += 1
                i2 += 1
            elif p1[i1][0] < p2[i2][0]:
                i1 += 1
            else:
                i2 += 1
        return results

    @staticmethod
    def to_json_result(results, time):
        output = []
        for doc_id in results:
            if len(doc_id) == 1:
                output.append({
                    'id': doc_id[0]
                })
            else:
                output.append({
                    'id': doc_id[0],
                    'positions': doc_id[1]
                })
        return {'results': output, 'time': time}

    def exact_term_search(self, posting_list):
        k = 1
        result = posting_list[0]
        for i in range(1, len(posting_list)):
            result = self.basic_exact_term_search(result, posting_list[i], k)
            if len(result) == 0:
                break
            k += 1

        for i in range(len(result)):
            x = []
            for j in range(0, len(posting_list)):
                [x.append(position + j) for position in result[i][1]]
            temp = list(result[i])
            temp[1] = x
            result[i] = tuple(temp)
        return result

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def compressed_cosine_similarity(cVec1, cVec2):
        # print('query_vec: ' + str(cVec1))
        # print('doc: ' + str(cVec2))
        # print("-----------------------------------")
        # # score = 0
        score = np.dot(cVec1, cVec2)
        # for term in cVec1.keys():
        #     score += cVec1[term] * cVec2[term]
        # print(score)
        return score

    def k_most_similarity(self, k, results, query_terms, query_vector):
        # heap_list = []
        # order_results = []
        # print(results)
        # v2 = query_vector.toarray()[0]
        # q_len = np.linalg.norm(v2)
        # for res in results:
        #     print(res)
        #     print("here0")
        #     doc_id = res[0]
        #     print("here1")
        #     # vector = self.dictionary.compressed_doc2vec(doc_id, query_terms)
        #     # print(doc_id)
        #     # print(len(self.dictionary.doc2vec))
        #     # print(self.dictionary.doc2vec[doc_id])
        #     doc_vector = self.dictionary.doc2vec[doc_id].toarray()[0]
        #     print("here2")
        #     print(doc_vector)
        #     print(query_vector)
        #     # print(doc_id)
        #     score = (np.dot(v2, doc_vector) / q_len) / np.linalg.norm(doc_vector)
        #     print("here3")
        #     heap_list.append((-score, res))
        #     print("here4")
        # k_largest = heapq.nsmallest(k, heap_list)
        # for item in k_largest:
        #     order_results.append(item[1])
        # return order_results

        vecs = []
        print(results)
        v2 = query_vector
        if type(v2) == sp.sparse.csr.csr_matrix:
            v2 = query_vector.toarray()[0]
        for res in results:
            print(res)
            print("here0")
            doc_id = res[0]
            print("here1")
            # vector = self.dictionary.compressed_doc2vec(doc_id, query_terms)
            # print(doc_id)
            # print(len(self.dictionary.doc2vec))
            # print(self.dictionary.doc2vec[doc_id])
            vecs.append(self.dictionary.doc2vec[doc_id])
            # print("here2")
            # print(doc_vector)
            # print(query_vector)
            # # print(doc_id)
            # score = (np.dot(v2, doc_vector) / q_len) / np.linalg.norm(doc_vector)
            # print("here3")
            # heap_list.append((-score, res))
            # print("here4")
        data = sp.sparse.vstack(vecs, format='csr')
        score = cosine_similarity(data, [v2], True)
        score.reshape(score.shape[0])
        heap_list = list(zip(-score, results))
        k_largest = heapq.nsmallest(k, heap_list)
        rem = list(zip(*k_largest))[1]
    
        return rem


if __name__ == '__main__':
    query = 'تولیدات'
    jquery = QueryProcessor().parse(query)
    print(jquery)
    eng = SearchEngine()
    # start = time.time()
    output = eng.search(jquery)

    # end = time.time()
    # print("Elapsed " + str(end - start  ))
    # for doc in output:
    #     news = fetch_result([doc['id']])

    print(output)
