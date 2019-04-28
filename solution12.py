## general imports
import random
import itertools 
from pprint import pprint  
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split  # data splitter
from sklearn.linear_model import LogisticRegression
import re


## project supplied imports
from submission_specs.SubmissionSpec12 import SubmissionSpec12

class Submission(SubmissionSpec12):
    ''' a contrived poorely performing solution for question one of this Maman '''
    __TAG_IDX  = 1
    __WORD_IDX = 0
    __SEPERATOR = "#=<*|*>=#"
    __XTAG_PLACE_HOLD = "{}" + __SEPERATOR + "{}"

    def __init__(self):
        self._tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        self._tag_count = dict()
        self._tag_to_num = dict()
        for idx, tag in enumerate(self._tag_set):
            self._tag_to_num[tag] = idx
        self._delta = 0.1
        self._N = len(self._tag_set)

    def _smooth(self, tag, numerator=0):
        return (numerator + self._delta) / (self._tag_count[tag] + self._delta*self._N)

    def _create_xtag(self, x1, x2):
        return self.__XTAG_PLACE_HOLD.format(x1, x2)

    def _count_tag_bigrams(self, annotated_sentences):
        counts = dict()
        for sent in annotated_sentences:
            sent_len = len(sent)
            for idx in range(sent_len - 1):
                XTag = self._create_xtag(sent[idx][self.__TAG_IDX], sent[idx + 1][self.__TAG_IDX])
                if not XTag in counts:
                    counts[XTag] = 1
                else:
                    counts[XTag] = counts[XTag] + 1
        return counts

    def _count_tags(self, annotated_sentences):
        self._tag_count = dict()
        for sent in annotated_sentences:
            for p in sent:
                tag = p[self.__TAG_IDX]
                if not tag in self._tag_count:
                    self._tag_count[tag] = 1
                else:
                    self._tag_count[tag] = self._tag_count[tag] + 1

    def _estimate_transition_probabilites(self, annotated_sentences):
        bigrams_counts = self._count_tag_bigrams(annotated_sentences)
        self._transition_probs = dict()
        for sent in annotated_sentences:
            sent_len = len(sent)
            for idx in range(sent_len-1):
                tag1 = sent[idx][self.__TAG_IDX]
                tag2 = sent[idx+1][self.__TAG_IDX]
                Xtag = self._create_xtag(tag1, tag2)
                if not Xtag in self._transition_probs:
                    self._transition_probs[Xtag] = self._smooth(tag1, bigrams_counts[Xtag])  #bigrams_counts[Xtag] / self._tag_count[tag1]

    def _count_tag_to_word_pairs(self, annotated_sentences):
        counts = dict()
        for sent in annotated_sentences:
            for p in sent:
                xtag = self._create_xtag(p[self.__WORD_IDX], p[self.__TAG_IDX])
                if not xtag in counts:
                    counts[xtag] = 1
                else:
                    counts[xtag] = counts[xtag] + 1
        return counts

    def _estimate_emission_probabilites(self, annotated_sentences):
        bigrams_counts = self._count_tag_to_word_pairs(annotated_sentences)
        self._emission_probs = dict()
        for sent in annotated_sentences:
            for p in sent:
                tag = p[self.__TAG_IDX]
                word = p[self.__WORD_IDX]
                XTag = self._create_xtag(word, tag)
                if not XTag in self._emission_probs:
                    self._emission_probs[XTag] = self._smooth(tag, bigrams_counts[XTag]) #bigrams_counts[XTag] / self._tag_count[tag]

    def _break_XTag(self, XTag):
        return XTag.split(self.__SEPERATOR)

    def _reshape_probabilities(self):
        '''
        reshapes the dictionaries that were learned into a ndarrays, that will be used in the viterbi algorithm
        _transition_probs will be reshaped to NXN ndarray and _emission_probs will be reshaped to a dictionary that
        will hold the an vector of N
        :return: self
        '''
        N = self._N
        tp = np.zeros((N,N))
        ep = dict()

        #reshape the transition probabilities
        for k,v in self._transition_probs.items():
            tags = self._break_XTag(k)
            tp[self._tag_to_num[tags[0]], self._tag_to_num[tags[1]]] = v
        self._transition_probs = tp

        #reshape emission probabilities
        for k,v in self._emission_probs.items():
            word_tag = self._break_XTag(k)
            if not word_tag[0] in ep:
                ep[word_tag[0]] = np.zeros(N)
            ep[word_tag[0]][self._tag_to_num[word_tag[1]]] = v

        self._emission_probs = ep

        return self

    def _viterbi(self, sentence):
        if sentence is None:
            return
        N = self._N
        T = len(sentence)
        viterbi_mat = np.zeros((N, T))
        backpointer = np.zeros((N, T))

        #init the lettece matrix
        for s in range(N):
            if sentence[0] in self._emission_probs:
                viterbi_mat[s, 0] = self._emission_probs[sentence[0]][s]

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

        best_path_probe = np.max(viterbi_mat[:, T-1])
        best_back_pointer = int(np.argmax(viterbi_mat[:, T-1]))

        best_path = list() #[self._tag_set[int(best_back_pointer)]]
        for t in range(1, T):
            next = int(np.max(backpointer[:, t]))
            best_path.append(self._tag_set[next])
        best_path.append(self._tag_set[best_back_pointer])

        return best_path, best_path_probe

    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''

        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))

        self._count_tags(annotated_sentences)
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)
        self._reshape_probabilities()
        
        return self 

    def predict(self, sentence):
        prediction, _ = self._viterbi(sentence)
        #prediction = [random.choice(self._tag_set) for segment in sentence]
        assert (len(prediction) == len(sentence))
        return prediction
            