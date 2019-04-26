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

    def __init__(self):
        self._tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        self._tag_count = dict()

    def _create_xtag(self, x1, x2):
        return "{}|{}".format(x1, x2)

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
                    self._transition_probs[Xtag] = bigrams_counts[Xtag] / self._tag_count[tag1]

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
                    self._emission_probs[XTag] = bigrams_counts[XTag] / self._tag_count[tag]

    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''

        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))

        self._count_tags(annotated_sentences)
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)
        
        return self 

    def predict(self, sentence):
        prediction = [random.choice(self._tag_set) for segment in sentence]
        assert (len(prediction) == len(sentence))
        return prediction
            