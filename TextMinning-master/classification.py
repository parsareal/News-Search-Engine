from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas
from collections import defaultdict
import pickle


class Classification:
    def __init__(self, train_size, knn_mode):
        print('load labels for classification...')
        self.labels_file = '../all_labels.csv'
        self.labels = []
        self.train_size = train_size
        self.data_classes = dict()
        if knn_mode:
            self.classifier = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
        else:
            self.classifier = GaussianNB()
        df = pandas.read_csv(self.labels_file)
        for i in range(self.train_size):
            self.labels.append(df['Label'][i])
            # self.data_classes[df['Label'][i]].append(i)
            if df['Label'][i] in self.data_classes.keys():
                self.data_classes[df['Label'][i]].append(i)
            else:
                self.data_classes[df['Label'][i]] = list()
                self.data_classes[df['Label'][i]].append(i)

    def classify_a_doc(self, doc_vec, doc_id):
        class_label = self.classifier.predict(doc_vec)
        # print('doc class: ' + str(class_label))
        self.data_classes[class_label[0]].append(doc_id)

    def classify_test_data(self, test_data, doc_ids):
        labels = self.classifier.predict(test_data)
        # print(labels[0])
        # print(type(labels[0]))
        for i in range(len(labels)):
            label = labels[i]
            doc_id = doc_ids[i]
            if label in self.data_classes.keys():
                self.data_classes[label].append(doc_id)
            else:
                self.data_classes[label] = list()
                self.data_classes[label].append(doc_id)
            # self.data_classes[label[0]].append(doc_id)

    def learn(self, train_data):
        print('train data loaded for classification...')
        self.classifier.fit(train_data, self.labels)
        print('train complete')

    def get_class_type(self, cat):
        # print(self.data_classes[cat])
        return self.data_classes[int(cat)]

    def save_labels(self):
        with open('../classification_all.pickle', 'wb') as handle:
            pickle.dump(self.data_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_labels(self):
        with open('../classification_all.pickle', 'rb') as class_labels:
            loaded_data_classes = pickle.load(class_labels)
        self.data_classes = loaded_data_classes


if __name__ == '__main__':
    c = Classification()
    print(c.labels)
    print(c.data_classes)
