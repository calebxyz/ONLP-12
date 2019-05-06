## general imports
import random
import itertools 
from pprint import pprint  
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split  # data splitter
from sklearn.linear_model import LogisticRegression
import re
import copy as cp


## project supplied imports
from submission_specs.SubmissionSpec12 import SubmissionSpec12

class Submission(SubmissionSpec12):
    ''' a contrived poorely performing solution for question one of this Maman '''
    __TAG_IDX  = 1
    __WORD_IDX = 0
    __START_TAG = "<s>"
    __END_TAG = "<e>"
    __NUM_OF_PATHS = 3

    def __init__(self):
        self._tag_set = np.array('ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split())
        self._tag_to_num = {tag:idx for idx, tag in enumerate(self._tag_set)}
        self._lrm = LogisticRegression(multi_class='multinominal', solver='lbfgs', max_iter=1000)
        self._ngrams = set()

    def _estimate_transition_probabilites(self, annotated_sentences):
        pass

    def _estimate_emission_probabilites(self, annotated_sentences):
        pass

    def _get_word_ngrams(self, min_ngram_len, max_ngram_len, token):
        '''
        a helper function we use to build all character ngrams upon initialization,
        in case we choose to use character ngrams as features
        '''

        word_ngrams = set()

        # per ngram length
        for n in range(min_ngram_len, max_ngram_len + 1):
            # sliding window iterate the token to extract its ngrams
            for idx in range(len(token) - n + 1):
                ngram = token[idx: idx + n]
                word_ngrams.add(ngram)

        return word_ngrams  # return value used for test only

    def _get_vocabulary(self, sentences):
        V = set()
        grams = set()


        for sentence in sentences:
            for idx, token in enumerate(sentence):
                gram = self._create_trigram(sentence, idx)
                V.add(token[self.__WORD_IDX])
                grams.add(gram)

        #set unique words and unique trigrams
        self._word_count = len(V)
        self._trigram_count = len(grams)
        return V

    def _create_ngrams_list(self, sentences, min_ngram=1, max_ngram=3):
        V = self._get_vocabulary(sentences)
        for t in V:
            self._ngrams |= self._get_word_ngrams(min_ngram, max_ngram, t)

    def _create_vectors(self, sentences):
        self._3gram_dict = set()
        y = np.zeros(self._trigram_count)
        X = np.zeros(self._trigram_count, dtype=list)

        #location in vect
        loc = 0
        for sentence in sentences:
            for idx, pair in enumerate(sentence):
                x = self._vectorize(sentence, idx)
                if x is not None:
                    X[loc] = x
                    y[loc] = self._tag_to_num[pair[self.__TAG_IDX]]
                loc += 1

        assert len(y) == len(X)

        return X, y

    def _word_vectorize(self, word, tag):
        '''size of the vector is size of ngrams * 3 + is all upper + starts with capital
        + has numbers and dash + has a number + special letter + word shape capitals to regulars, num of capitals,
        num of regulars, num of punct, num of numbers and N for number of states'''
        vect = np.zeros(len(self._ngrams)*3 + 10 + len(self._tag_set))
        offset = 0
        offest_count = 3

        #check if word contains starts with or ends with one of the ngrams
        for ngram in self._ngrams:
            if ngram in word:
                vect[offset] = 1
            if word.startswith(ngram):
                vect[offset + 1] = 1
            if word.endswith(ngram):
                vect[offset + 2] = 1
            offset += offest_count

        caps = re.findall("[A-Z]", word)
        nums = re.findall("[0-9]", word)
        lower = re.findall("[a-z]", word)
        punct = re.findall("[.?\-,\"]+", word)

        #is all upper
        if not re.findall("^[A-Z]+$", word):
            vect[offset] = 1
        offset += 1

        #starts with capital
        if re.findall("^[A-Z][A-Za-z0-9]+$", word):
            vect[offset] = 1
        offset += 1

        #has only capitals dash and numbers
        if re.findall("^[A-Z]+-[0-9]+", word):
            vect[offset] = 1
        offset += 1

        #contains a number
        if re.findall("[0-9]", word):
            vect[offset] = 1
        offset += 1

        if 'a' in word or 'e' in word or 'u' in word or 'o' in word or 'i' in word:
            vect[offset] = 1
        offset += 1

        vect[offset] = len(caps)
        vect[offset + 1] = len(lower)
        vect[offset + 2] = len(nums)
        vect[offset + 3] = len(punct)

        vect[offset + 4 + self._tag_to_num[tag]] = 1

        return vect

    def _create_trigram(self, sentence, idx):
        gram = [(None, None), (None, None), (None, None)]

        def assign_grams(gid, id):
            grm = (sentence[id][self.__WORD_IDX], sentence[id][self.__TAG_IDX])
            gram[gid] = tuple(grm)

        assign_grams(1, idx)

        if idx > 0:
            assign_grams(0, idx - 1)

        if idx < len(sentence) - 1:
            assign_grams(2, idx + 1)

        return tuple(gram)

    def _vectorize(self, sentence, idx):
        gram = self._create_trigram(sentence, idx)

        if gram not in self._3gram_dict:
            self._3gram_dict.add(gram)

            vec1 = self._word_vectorize(gram[1][0], gram[1][1])

            if gram[0][0] and gram[0][1]:
                vec2 = self._word_vectorize(gram[0][0], gram[0][1])
            else:
                # an all-zeros vector the length of the word vector (this is arbitrary)
                vec2 = [0] * len(vec1)

            if gram[2][0] and gram[2][1]:
                vec3 = self._word_vectorize(gram[2][0], gram[2][1])
            else:
                # an all-zeros vector the length of the word vector (this is arbitrary)
                vec3 = [0] * len(vec1)

            # our feature vector
            return vec1 + vec2 + vec3

        return None

    '''def _viterbi(self, sentence):
        if sentence is None:
            return
        N = self._N
        T = len(sentence)
        viterbi_mat = np.zeros((N, T))
        backpointer = np.full((N, T), -1)

        #init the lettece matrix
        for s in range(N):
            if sentence[0] in self._emission_probs:
                viterbi_mat[s, 0] = self._emission_probs[sentence[0]][s]*self._pis[s]

        def getMaxByFunc(func, o_t, s, t):
            if o_t in self._emission_probs:
                b_ot = self._emission_probs[o_t][s]
            else:
                b_ot = self._smooth(self._tag_set[s])

            vitmax = np.multiply(viterbi_mat[:, t - 1], self._transition_probs[:, s]) * b_ot
            return func(vitmax)

        for t in range(1, T):
            for s in range(N):
                o_t = sentence[t]
                viterbi_mat[s, t] = getMaxByFunc(np.max, o_t, s, t)
                backpointer[s, t] = getMaxByFunc(np.argmax, o_t, s, t)

        return self._calc_best_paths(T, viterbi_mat, backpointer)'''

    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''
        self._create_ngrams_list(annotated_sentences)
        X, y = self._create_vectors(annotated_sentences)
        self._lrm.fit(X, y)
        return self

    def predict(self, sentence):
        prediction, _ = (None, None)
        return prediction
