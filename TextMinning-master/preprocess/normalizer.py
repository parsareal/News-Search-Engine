import re

from preprocess.lemmatizer import Lemmatizer
from scripts import utils

translation_src, translation_dst = '“”آ ىكي', '""ا یکی'
translation_src += "\u200b\u200d_-,.\n"
translation_dst += "\u200c\u200c     "
must_remove = "۰۱۲۳۴۵۶۷۸۹٪ًٌٍَُِّـ0123456789%;:,؛،'#\\/"
translation_src += must_remove
translation_dst += ''.join(' ' for i in range(len(must_remove)))
remove_range = (128512, 128591)  # 😀
maketrans = lambda A, B: dict((ord(a), b) for a, b in zip(A, B))
translator = maketrans(translation_src, translation_dst)

bad_ends = ['ات', 'ان', 'ترین', 'تر', 'م', 'ت', 'ش', 'یی', 'ی', 'ها', 'ٔ', '‌ا', '‌']
print("Loading phrases...", end="\t")
phrases = utils.read_file_lines('../resources/phrases.txt')
phrases = set(phrases)
print("Done!")

multi_dictation = []

print("Loading short phrases...", end='\t')
shorts = dict()
tmp = utils.read_file_lines("../resources/short.txt")
for item in tmp:
    sp = item.split(',', maxsplit=1)
    shorts[sp[0]] = sp[1]
print("Done!")

print("Loading stopwords...", end="\t")
stopwords = set(utils.read_file_lines("../resources/stopwords.txt"))
print("Done!")


class Normalizer:

    @staticmethod
    def normalize(text: str):
        text = Normalizer._character_normalization(text)
        text = Normalizer._word_normalization(text)
        return text

    @staticmethod
    def _character_normalization(text):
        text = text.lower()
        text = Normalizer._remove_bad(text)
        text = Normalizer._character_translate(text)
        text = Normalizer._fix_white_space(text)
        return text

    @staticmethod
    def _word_normalization(text):
        text = Normalizer._fix_short_phrases(text)
        text = Normalizer._fix_phrases(text)
        text = Normalizer._handle_multi_dic(text)
        return text

    @staticmethod
    def _character_translate(text: str):
        """ Replaces Arabic and non-Persian characters with Persian characters ones and unifying half-spaces """
        return text.translate(translator)

    @staticmethod
    def _remove_bad(text: str):
        """ Removes emojis and Erab """
        for char in must_remove:
            text = text.replace(char, '')
        for char_code in range(remove_range[0], remove_range[1] + 1):
            c = chr(char_code)
            text = text.replace(c, "")
        return text

    @staticmethod
    def tokenize(text):
        result = []
        try:
            result = text.split(" ")
        except:
            return result
        return result

    @staticmethod
    def stem(text, words):
        for end in bad_ends:
            if text.endswith(end):
                text = text[:-len(end)]
                if text in words:
                    return text

        if text.endswith('ۀ'):
            text = text[:-1] + 'ه'
        return Normalizer._fix_white_space(text)

    @staticmethod
    def lem(text):
        global lemmatizer
        text = lemmatizer.lemmatize(text)
        return Normalizer._fix_white_space(text)

    @staticmethod
    def _fix_phrases(text):
        for phrase in phrases:
            text = text.replace(phrase, Normalizer.iget_whole_phrase(phrase))
            text = Normalizer._fix_white_space(text)
        return text

    @staticmethod
    def iget_whole_phrase(ph):
        ph = ph.replace(" ", '\u200c')
        return ph

    @staticmethod
    def _fix_white_space(text):
        text = re.sub("\s+", " ", text)
        text = text.strip()
        return text

    @staticmethod
    def _handle_multi_dic(text):
        for multi_set in multi_dictation:
            for word in multi_set:
                text = text.replace(word, multi_set[0])
        return text

    @staticmethod
    def _fix_short_phrases(text):
        for key in shorts.keys():
            pat = re.compile('^\s*' + key + '\s+|\s+' + key + '\s+|\s+' + key + '\s*$')
            text = re.sub(pat, " " + shorts[key] + " ", text)
        return text

    @staticmethod
    def is_stopword(txt):
        return txt in stopwords

    @staticmethod
    def remove_stopwords(tokens):
        tokens = [x for x in tokens if x not in stopwords]
        return tokens


print("Loading multi dictation words...", end='\t')
md = utils.read_file_lines("../resources/multi_dictation.txt")
for multi in md:
    word_list = []
    for item in multi.split(','):
        word_list.append(Normalizer._character_normalization(item))
    multi_dictation.append(word_list)
print("Done!")

print("Loading lemmatizer...", end="\t")
lemmatizer = Lemmatizer(Normalizer.stem)
print("Done!")
