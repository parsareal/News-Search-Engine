
from preprocess.normalizer import Normalizer

result = set()
with open('../resources/short.txt', 'r', encoding="utf_8") as f:
    with open('../resources/short1.txt', 'w', encoding="utf_8") as f1:
        stopwords = f.read()
        stopwords = stopwords.splitlines()
        for word in stopwords:
            sp = word.split(',')
            f1.writelines(Normalizer.normalize(sp[0]) + "," + Normalizer.normalize(sp[1]) + '\n')

