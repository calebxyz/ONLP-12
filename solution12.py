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
    __START_GRAM = (None, None)
    __END_GRAM   = (None, None)

    def __init__(self):
        self._tag_set = np.array('ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split())
        self._tag_to_num = {tag:idx for idx, tag in enumerate(self._tag_set)}
        self._lrm = LogisticRegression(multi_class='multinominal', solver='lbfgs', max_iter=1000)
        self._ngrams = set()
        self._N = len(self._tag_set)
        self._pis = np.zeros(self._N, dtype=np.float64)

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

    def _calc_pis(self, grams):
        start_grams = dict()
        for k, v in grams.items():
            if k[0] == self.__START_GRAM:
                work_gram = (None, k[1][1])
                if work_gram not in start_grams:
                    start_grams[work_gram] = v
                else:
                    start_grams[work_gram] += v

        for idx in range(self._N):
            tag = self._tag_set[idx]
            gram = (None, self._tag_set[idx])
            if gram in start_grams:
                self._pis[idx] = start_grams[gram] / self._tag_count[tag]

    def _get_vocabulary(self, sentences):
        '''
        calculates all needed initial parts , trigrams, number of words set of words
        counts tags and pis
        :param sentences: sentences to inspect
        :return: vocabulary
        '''
        V = set()
        self._tri_grams = dict()

        self._total_ngrams = 0
        self.total_words = 0
        self._tag_count = dict()
        for sentence in sentences:
            for idx, token in enumerate(sentence):
                gram = self._create_trigram(sentence, idx)
                V.add(token[self.__WORD_IDX])
                if gram in self._tri_grams.keys():
                    self._tri_grams[gram] += 1
                else:
                    self._tri_grams[gram] = 1
                if token[self.__TAG_IDX] in  self._tag_count.keys():
                    self._tag_count[token[self.__TAG_IDX]] += 1
                else:
                    self._tag_count[token[self.__TAG_IDX]] = 1
                self._total_ngrams += 1
                self.total_words += 1

        self._calc_pis(self._tri_grams)

        self._word_count = len(V)
        self._trigram_count = len(self._tri_grams)
        return V

    def _create_ngrams_list(self, sentences, min_ngram=2, max_ngram=2):
        V = self._get_vocabulary(sentences)
        for t in V:
            self._ngrams |= self._get_word_ngrams(min_ngram, max_ngram, t)

    def _create_vectors(self):
        #TODO: check if we can train on one ngram at a time or we need to multiply the ngrams
        '''y = np.zeros(self._total_ngrams)
        X = np.zeros(self._total_ngrams, dtype=list)'''
        y = np.zeros(len(self._tri_grams))
        X = np.zeros(len(self._tri_grams), dtype=list)

        #location in vect
        loc = 0
        for gram, times in self._tri_grams.items():
            fv = self._vectorize(gram)
            c  = self._tag_to_num[gram[1][1]]

            '''y[loc:loc+times] = c

            for t in range(times):
                X[loc+t] = fv

            loc += times'''

            y[loc] = c
            X[loc] = fv
            loc += 1

        assert len(y) == len(X)

        return X, y

    def _word_vectorize(self, word, vect, start=0, main_word=False):
        '''size of the vector is size of ngrams * 3 + is all upper + starts with capital
        + has numbers and dash + has a number + special letter + word shape capitals to regulars, num of capitals,
        num of regulars, num of punct, num of numbers and N for number of states'''
        offset = start
        offest_count = 2

        if main_word:
            #check if word contains starts with or ends with one of the ngrams
            for ngram in self._ngrams:
                '''if ngram in word:
                    vect[offset] = 1'''
                if word.startswith(ngram):
                    vect[offset] = 1
                if word.endswith(ngram):
                   vect[offset + 1] = 1
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

        return vect

    def _create_trigram(self, sentence, idx):
        gram = [self.__START_GRAM, self.__START_GRAM, self.__END_GRAM]

        def assign_grams(gid, id):
            grm = (sentence[id][self.__WORD_IDX], sentence[id][self.__TAG_IDX])
            gram[gid] = tuple(grm)

        assign_grams(1, idx)

        if idx > 0:
            assign_grams(0, idx - 1)

        if idx < len(sentence) - 1:
            assign_grams(2, idx + 1)

        return tuple(gram)

    def _vectorize(self, gram):
        # the size of the vector is fv_len and two words between it
        vector = np.zeros(self._fv_size + 2*self._fv_size_no_ngrams)

        self._word_vectorize(gram[1][self.__WORD_IDX], vector, 0, True)

        if gram[0] != self.__START_GRAM:
             self._word_vectorize(gram[0][self.__WORD_IDX], vector, self._fv_size)

        if gram[2] != self.__END_GRAM:
             self._word_vectorize(gram[2][self.__WORD_IDX], vector, self._fv_size+self._fv_size_no_ngrams)

        # our feature vector
        return vector

    def _get_lrm_prediction(self, sentence, idx, state):
        X = np.zeros(self._fv_size)
        self._word_vectorize(sentence[idx], self._tag_set[state], X)
        return self._lrm.predict_proba(X)

    def _viterbi(self, sentence):
        if sentence is None:
            return
        N = self._N
        T = len(sentence)
        viterbi_mat = np.zeros((N, T))
        backpointer = np.full((N, T), -1)
        #init the lettece matrix
        for s in range(N):
            pred = self._get_lrm_prediction(sentence, 0, s)
            viterbi_mat[s, 0] = pred[s]*self._pis[s]

        for t in range(1, T):
            for s in range(N):
                pred = self._get_lrm_prediction(sentence, t, s)
                viterbi_mat[s, t] = np.max(pred)
                backpointer[s, t] = np.argmax(pred)

        best_path_probe = np.max(viterbi_mat[:, T - 1])
        best_back_pointer = int(np.argmax(viterbi_mat[:, T - 1]))

        best_path = list()  # [self._tag_set[int(best_back_pointer)]]
        best_path.append(self._tag_set[best_back_pointer])
        for t in reversed(range(0, T - 1)):
            next_tag = int(backpointer[np.argmax(viterbi_mat[:, t + 1]), t + 1])
            best_path.append(self._tag_set[next_tag])

        return best_path[::-1], best_path_probe

    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''
        self._create_ngrams_list(annotated_sentences)
        self._fv_size = len(self._ngrams) * 2 + 10  # size of the feature vector
        self._fv_size_no_ngrams = 10 + len(self._tag_set)
        X, y = self._create_vectors()
        self._lrm.fit(X, y)
        return self

    def predict(self, sentence):
        prediction, _ = (None, None)
        return prediction
