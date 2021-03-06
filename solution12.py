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
    __SPECIAL_FEATURES_NUM = 11

    def __init__(self):
        self._tag_set = np.array('ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split())
        self._tag_to_num = {tag:idx for idx, tag in enumerate(self._tag_set)}
        self._lrm = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        self._ngrams = set()
        self._N = len(self._tag_set)
        self._pis = np.zeros(self._N, dtype=np.float64)
        self._states_per = sorted(set(list(itertools.product(range(self._N), repeat=3))))
        self._states_len = len(self._states_per)

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

    def _create_ngrams_list(self, sentences, min_ngram=1, max_ngram=2):
        V = self._get_vocabulary(sentences)
        for t in V:
            self._ngrams |= self._get_word_ngrams(min_ngram, max_ngram, t)

    def _create_vectors(self, sentences):
        #TODO: check if we can train on one ngram at a time or we need to multiply the ngrams
        y = np.zeros(self._total_ngrams)
        X = [[0]] * self._total_ngrams
        #X = np.zeros((self._total_ngrams, self._fv_size))
        '''y = np.zeros(len(self._tri_grams))
        X = [0] * len(self._tri_grams)'''

        #location in vect
        loc = 0
        for gram, times in self._tri_grams.items():
            fv = self._vectorize(gram)
            c  = self._tag_to_num[gram[1][1]]

            y[loc:loc+times] = c

            for t in range(times):
                X[loc+t] = fv

            loc += times

            '''y[loc] = c
            X[loc] = fv
            loc += 1'''

        assert len(y) == len(X)

        return X, y

    def _word_vectorize(self, word, tag, vect, start=0, main_word=False):
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

        if 'ing' in word:
            vect[offset] = 1
        offset += 1

        tmpWord = word.lower()
        if 'a' in tmpWord or 'e' in tmpWord or 'u' in tmpWord or 'o' in tmpWord or 'i' in tmpWord:
            vect[offset] = 1
        offset += 1

        if 'ing' in tmpWord:
            vect[offset] = 1
        offset += 1

        vect[offset] = len(caps)
        vect[offset + 1] = len(lower)
        vect[offset + 2] = len(nums)
        vect[offset + 3] = len(punct)

        vect[offset + 4 + self._tag_to_num[tag]] = 1

        return vect

    def _create_trigram(self, sentence, idx, states=None):
        gram = [self.__START_GRAM, self.__START_GRAM, self.__END_GRAM]

        def assign_grams(gid, id):
            try:
                if not states:
                    grm = (sentence[id][self.__WORD_IDX], sentence[id][self.__TAG_IDX])
                else:
                    grm = (sentence[id], self._tag_set[states[gid]])
            except:
                grm = (None, None)

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

        self._word_vectorize(gram[1][self.__WORD_IDX], gram[1][self.__TAG_IDX], vector, 0, True)

        if gram[0] != self.__START_GRAM:
            self._word_vectorize(gram[0][self.__WORD_IDX], gram[0][self.__TAG_IDX], vector, self._fv_size)

        if gram[2] != self.__END_GRAM:
            self._word_vectorize(gram[2][self.__WORD_IDX], gram[2][self.__TAG_IDX], vector, self._fv_size+self._fv_size_no_ngrams)

        # our feature vector
        return vector

    def _get_lrm_prediction(self, sentence, idx, states=None):
        gram = self._create_trigram(sentence, idx, states)
        X = self._vectorize(gram).reshape(1, -1)
        return self._lrm.predict_proba(X)

    def _get_bulk_lrm_prediction(self, sent, idx):
        grams = list(map(self._create_trigram, itertools.repeat(sent, self._states_len),
                         itertools.repeat(idx, self._states_len), self._states_per))
        X = list(map(self._vectorize, grams))

        return self._lrm.predict_proba(X)

    def _prepare_prediction_data(self, raw_lrm_pred):
        N = self._N
        pred_i = np.zeros((N, N))
        pred_j = np.zeros((N, N))
        i_loc = 0
        for i in range(0, self._states_len, N*N):
            for j in range(N):
                loc = i + j*N
                pred_j[j, :] = raw_lrm_pred[loc:loc+N, j]
            pred_i[i_loc, :] = np.max(pred_j, 1)[:]
            i_loc += 1
        return pred_i

    def _get_word_bulk_predictions(self, sent, idx):
        return self._prepare_prediction_data(self._get_bulk_lrm_prediction(sent, idx))

    def _get_sentence_bulk_prediction(self, sentence):
        '''
        creates a vector of bulk predictions ordered by words
        :param sentence: the sentence to extract from
        :return: returns a vector of prediction matrixes ([works, NxN transitions of (si, (sj,sk))])
        '''
        return np.array(list(map(self._get_word_bulk_predictions, itertools.repeat(sentence, len(sentence)), range(len(sentence)))))

    def _viterbi(self, sentence):
        if sentence is None:
            return
        N = self._N
        T = len(sentence)
        viterbi_mat = np.zeros((N, T))
        backpointer = np.full((N, T), -1)

        ''' get prediction for all words'''
        preds = self._get_sentence_bulk_prediction(sentence)

        #init the lettece matrix
        '''for s in range(N):
            pred = self._get_lrm_prediction(sentence, 0, [-1, s, s+1])
            viterbi_mat[s, 0] = pred[0][s]*self._pis[s]'''

        '''init first values in the viterbi matrix , the predictions are allocated by words in the sentence 
        so we need to look at the first matrix take the maximum value to the transition s_i,s_j,s_k which is simplified 
        after the transition to s_i,(s_j,s_k) and multiply it by pis set vector'''
        viterbi_mat[:, 0] = np.max(preds[0], 0)*self._pis

        for t in range(1, T):
            for s_i in range(N):
                vitmax = preds[t][s_i, :] * viterbi_mat[:, t - 1]
                viterbi_mat[s_i, t] = np.max(vitmax)
                backpointer[s_i, t] = np.argmax(vitmax)

        best_path_probe = np.max(viterbi_mat[:, T - 1])
        best_back_pointer = int(np.argmax(viterbi_mat[:, T - 1]))

        best_path = list()  # [self._tag_set[int(best_back_pointer)]]

        for t in reversed(range(0, T - 1)):
            next_tag = int(backpointer[np.argmax(viterbi_mat[:, t + 1]), t + 1])
            best_path.append(self._tag_set[next_tag])

        best_path.append(self._tag_set[best_back_pointer])

        return best_path[::-1], best_path_probe

    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''
        self._create_ngrams_list(annotated_sentences)
        self._fv_size = len(self._ngrams) * 2 + self.__SPECIAL_FEATURES_NUM + len(self._tag_set)  # size of the feature vector
        self._fv_size_no_ngrams = self.__SPECIAL_FEATURES_NUM + len(self._tag_set)
        X, y = self._create_vectors(annotated_sentences)
        self._lrm.fit(X, y)
        return self

    def predict(self, sentence):
        prediction, _ = self._viterbi(sentence)
        return prediction
