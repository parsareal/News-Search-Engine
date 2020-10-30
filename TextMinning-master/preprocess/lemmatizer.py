from __future__ import unicode_literals
from hazm.utils import default_words, default_verbs
from hazm.WordTokenizer import WordTokenizer


class Lemmatizer(object):

    def __init__(self, stemmer, words_file=default_words, verbs_file=default_verbs, joined_verb_parts=True):
        self.verbs = {}
        self.stemmer = stemmer

        tokenizer = WordTokenizer(words_file=words_file, verbs_file=verbs_file)
        self.words = tokenizer.words

        if verbs_file:
            self.verbs['است'] = '#است'
            for verb in tokenizer.verbs:
                for tense in self.conjugations(verb):
                    self.verbs[tense] = verb
            if joined_verb_parts:
                for verb in tokenizer.verbs:
                    bon = verb.split('#')[0]
                    for after_verb in tokenizer.after_verbs:
                        self.verbs[bon + 'ه_' + after_verb] = verb
                        self.verbs['ن' + bon + 'ه_' + after_verb] = verb
                    for before_verb in tokenizer.before_verbs:
                        self.verbs[before_verb + '_' + bon] = verb

    def lemmatize(self, word, pos=''):
        if not pos and word in self.words:
            return word

        if (not pos or pos == 'V') and word in self.verbs:
            return self.verbs[word]

        if pos.startswith('AJ') and word[-1] == 'ی':
            return word

        if pos == 'PRO':
            return word

        if word in self.words:
            return word

        stem = self.stemmer(word, self.words)
        if stem and stem in self.words:
            return stem

        return word

    def conjugations(self, verb):

        past, present = verb.split('#')
        ends = ['م', 'ی', '', 'یم', 'ید', 'ند']

        if verb == '#هست':
            return ['هست' + end for end in ends] + ['نیست' + end for end in ends]

        past_simples = [past + end for end in ends]
        past_imperfects = ['می‌' + item for item in past_simples]
        ends = ['ه‌ام', 'ه‌ای', 'ه', 'ه‌ایم', 'ه‌اید', 'ه‌اند']
        past_narratives = [past + end for end in ends]

        imperatives = ['ب' + present, 'ن' + present]

        if present.endswith('ا') or present in ('آ', 'گو'):
            present = present + 'ی'

        ends = ['م', 'ی', 'د', 'یم', 'ید', 'ند']
        present_simples = [present + end for end in ends]
        present_imperfects = ['می‌' + item for item in present_simples]
        present_subjunctives = [item if item.startswith('ب') else 'ب' + item for item in present_simples]
        present_not_subjunctives = ['ن' + item for item in present_simples]

        with_nots = lambda items: items + list(map(lambda item: 'ن' + item, items))
        aa_refinement = lambda items: list(map(lambda item: item.replace('بآ', 'بیا').replace('نآ', 'نیا'), items)) if \
        items[0].startswith('آ') else items
        return aa_refinement(
            with_nots(past_simples) + with_nots(present_simples) + with_nots(past_imperfects) + with_nots(
                past_narratives) + with_nots(present_simples) + with_nots(
                present_imperfects) + present_subjunctives + present_not_subjunctives + imperatives)
