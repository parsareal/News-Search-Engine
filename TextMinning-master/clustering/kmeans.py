import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.cluster import KMeans as KM
from sklearn.metrics.pairwise import cosine_similarity


from clustering.clusterer import clusterer
from dictionary import Dictionary

file_post = '_cluster.pkl'


def euc_dist(X, Y=None, Y_norm_squared=None, squared=False):
    # return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
    return cosine_similarity(X, Y)


KM.euclidean_distances = euc_dist


class KMeans(clusterer):

    def __init__(self):

        try:
            self.k_index = 3
            self.labels_list = list(np.load("../label" + file_post + '.npy', allow_pickle=True))
            self.RSS_list = list(np.load("../rss" + file_post + '.npy', allow_pickle=True))
            self.centroids_list = np.load("../cent" + file_post + '.npy', allow_pickle=True)
            # np.save("cent" + file_post, self.centroids_list[self.k_index])
        except Exception as e:
            print(e)
            print("error")
            self.k_index = 0
            self.labels_list = []
            self.RSS_list = []
            self.centroids_list = []

    def cluster_list_k(self, k_list, data):
        data = sp.vstack(data, format='csr')
        # self.data = pd.DataFrame.sparse.from_spmatrix(v)
        # data = pd.DataFrame(sp.vstack(data, format='csr').toarray())
        # data = np.array(data)

        self.k_index = 0
        self.labels_list = []
        self.RSS_list = []
        self.centroids_list = []
        ks = []
        for k in k_list:
            # print('\n')
            print('\n' + str(k) + ":", end=" ")
            # # try:
            # labels, RSS, cents = KMeans.cluster_with_k(k, data)
            # self.RSS_list.append(RSS)
            # self.labels_list.append(np.array(labels))
            # self.centroids_list.append(cents)
            # ks.append(k)
            # # except:
            # #     print('Failed', end='')
            # try:
            sk = KM(k, n_init=1, max_iter=30).fit(data)
            self.labels_list.append(np.array(sk.labels_))
            self.centroids_list.append(sk.cluster_centers_)
            self.RSS_list.append(sk.inertia_)
            ks.append(k)
            # except:
            #     print('Failed', end='')

        plt.plot(ks, self.RSS_list)
        plt.savefig('./cluster_rss')
        plt.show()
        np.save("index" + file_post, [self.k_index])
        np.save("label" + file_post, self.labels_list)
        np.save("rss" + file_post, self.RSS_list)
        np.save("cent" + file_post, self.centroids_list)

    def set_k_index(self, idx):
        self.k_index = idx

    @staticmethod
    def cluster_with_k(k, data: np.ndarray):
        centroids_idx = np.random.randint(data.shape[0], size=k);
        centroids = data.loc[centroids_idx, :]
        data_trans = data.transpose()
        data_len = np.linalg.norm(data, axis=1)
        feature_num = data.shape[1]

        r_list = []
        label_list = []
        centroids_list = []
        for j in range(1):
            print(str(j), end=' ')
            for i in range(50):
                centroids_len = np.linalg.norm(centroids, axis=1)
                centroids_len = np.transpose(np.array([centroids_len]))
                cent_data = centroids @ data_trans
                # cent_data = np.matmul(centroids, data_trans)
                tmp1 = centroids_len.reshape(k, 1)
                tmp2 = data_len.reshape(1, data.shape[0])
                tmp3 = np.matmul(tmp1, tmp2)
                cent_data = cent_data / tmp3
                labels = cent_data.idxmax(axis=0)
                data['label'] = labels
                # sample = df.sample(k)
                # df.at[sample.index[:], 'label'] = np.arange(k)
                centroids = data.groupby(['label']).mean()
                if centroids.shape[0] < k:
                    le = k - centroids.shape[0]
                    for ii in range(centroids.shape[0], centroids.shape[0] + le):
                        centroids.loc[np.random.randint(1000, 100000)] = np.random.normal(np.zeros(feature_num),
                                                                                          np.ones(feature_num) * 100)
                centroids = centroids.to_numpy()
                del data['label']

            centroids_len = np.linalg.norm(centroids, axis=1)
            centroids_len = np.transpose(np.array([centroids_len]))
            cent_data = centroids @ data_trans
            # cent_data = np.matmul(centroids, data_trans)
            tmp1 = centroids_len.reshape(k, 1)
            tmp2 = data_len.reshape(1, data.shape[0])
            tmp3 = np.matmul(tmp1, tmp2)
            cent_data = cent_data / tmp3
            labels = cent_data.idxmax(axis=0)
            data['label'] = labels
            # sample = df.sample(k)
            # df.at[sample.index[:], 'label'] = np.arange(k)
            centroids = data.groupby(['label']).mean()
            joi = data.join(centroids, on='label', rsuffix="cent", lsuffix="data")
            m = joi.loc[:, '0data': (str(feature_num - 1) + 'data')].to_numpy()
            n = joi.loc[:, '0cent': (str(feature_num - 1) + 'cent')].to_numpy()
            RSS = ((m - n) ** 2).sum()
            r_list.append(RSS)
            label_list.append(labels)
            centroids = centroids.to_numpy()
            centroids_list.append(centroids)
            del data['label']
        maxx = np.argmax(r_list)
        return label_list[maxx], r_list[maxx], centroids_list[maxx]

    def get_cluster_data(self, id):
        res = np.argwhere(self.labels_list[self.k_index] == id)
        res = res.reshape(res.shape[0])
        return res

    def get_cluster(self, vector):
        cents = self.centroids_list
        cents_len = np.linalg.norm(cents)
        vec_len = np.linalg.norm(vector)
        print(vec_len)
        print(cents_len)
        print()
        return (((cents * vector).sum(1) / cents_len) / vec_len).argsort()[-1:]

    @staticmethod
    def create(data):
        global k_means
        k_means.cluster_list_k(range(2, 150, 5), data)

    @staticmethod
    def get_k_means():
        global k_means
        return k_means

    @staticmethod
    def init():
        global k_means
        k_means = KMeans()
        return k_means


k_means = None
