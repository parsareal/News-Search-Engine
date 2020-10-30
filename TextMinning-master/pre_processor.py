import re
from preprocess.normalizer import Normalizer
from bs4 import BeautifulSoup
from PersianStemmer import PersianStemmer

stopwords = []


class DocTokenizer:

    @staticmethod
    def clean_html(raw_html):
        clean_text = BeautifulSoup(raw_html, features="html.parser")
        for sc in clean_text(['script', 'style']):
            sc.extract()
        return clean_text.get_text(' ')

    @staticmethod
    def text_normalizer(doc):
        return Normalizer.normalize(doc)

    @staticmethod
    def text_tokenizer(doc):

        return Normalizer.tokenize(doc)

    @staticmethod
    def remove_stopwords(tokens):
        Normalizer.remove_stopwords(tokens)
        return tokens

    @staticmethod
    def lem(token):
        return Normalizer.lem(token)

    @staticmethod
    def stem(word):
        return Normalizer.stem(word)

    def stem_tokens(self, tokens):
        new_tokens = []
        for token in tokens:
            new_tokens.append(self.lem(token))
        return new_tokens

    def get_tokens(self, doc):
        clean_text = self.clean_html(doc)
        clean_text = self.text_normalizer(clean_text)

        # tokens = word_tokenize(clean_text)
        tokens = self.text_tokenizer(clean_text)
        tokens = self.stem_tokens(tokens)
        tokens = self.remove_stopwords(tokens)
        unique_tokens = list(dict.fromkeys(tokens))

        # positional_tokens = clean_text.split(' ')
        # positional_tokens = self.stem_tokens(positional_tokens)
        positional = []
        for token in unique_tokens:
            indices = [i for i, x in enumerate(tokens) if x == token]
            positional.append((token, indices))
        return positional


if __name__ == '__main__':
    d = DocTokenizer()
    print(d.text_normalizer(''))
#     import pandas
#     df = pandas.read_csv('IR-F19-Project01-Input.csv')
#     doc = d.clean_html(df['content'][0])
#     tokens = d.text_tokenizer(doc)
#     # print(re.split('\W+', 'Words, words, words.'))
#     print(tokens)
