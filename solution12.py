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

    def _count_tag(self, tag, annotated_sentences):
        count = 0
        for sent in annotated_sentences:
            for p in sent:
                if p[self.__TAG_IDX] == tag:
                    count = count + 1
        return count

    def _count_bigrams(self, tag1, tag2, annotated_sentences):
        count = 0
        for sent in annotated_sentences:
            sent_len = len(sent)
            for idx in range(sent_len - 1):
                tag1_ = sent[idx][self.__TAG_IDX]
                tag2_ = sent[idx + 1][self.__TAG_IDX]
                if tag1 == tag1_ and tag2 == tag2_:
                    count = count + 1
        return count

    def _estimate_transition_probabilites(self, annotated_sentences):
        bigrams_counts = dict()
        self._transition_probs = dict()
        for sent in annotated_sentences:
            sent_len = len(sent)
            for idx in range(sent_len-1):
                tag1 = sent[idx][self.__TAG_IDX]
                tag2 = sent[idx+1][self.__TAG_IDX]
                Xtag = tag1 + "|" + tag2
                if not Xtag in self._transition_probs:
                    if not tag1 in self._tag_count:
                        self._tag_count[tag1] = self._count_tag(tag1, annotated_sentences)
                    if not Xtag in bigrams_counts:
                        bigrams_counts[Xtag] = self._count_bigrams(tag1, tag2, annotated_sentences)
                    self._transition_probs[Xtag] = bigrams_counts[Xtag] / self._tag_count[tag1]

    def _count_tag_to_word(self, tag, word, annotated_sentences):
        count = 0
        for sent in annotated_sentences:
            for p in sent:
                if p[self.__WORD_IDX] == word and p[self.__TAG_IDX] == tag:
                    count = count + 1
        return count

    def _estimate_emission_probabilites(self, annotated_sentences):
        bigrams_counts = dict()
        self._emission_probs = dict()
        for sent in annotated_sentences:
            for p in sent:
                tag = p[self.__TAG_IDX]
                word = p[self.__WORD_IDX]
                XTag = word + "|" + tag
                if not XTag in self._emission_probs:
                    if not tag in self._tag_count:
                        self._tag_count[tag] = self._count_tag(tag, annotated_sentences)
                    if not XTag in bigrams_counts:
                        bigrams_counts[XTag] = self._count_tag_to_word(tag, word, annotated_sentences)
                    self._emission_probs[XTag] = bigrams_counts[XTag] / self._tag_count[tag]


    def train(self, annotated_sentences):    
        ''' trains the HMM model (computes the probability distributions) '''

        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)
        
        return self 

    def predict(self, sentence):
        prediction = [random.choice(self._tag_set) for segment in sentence]
        assert (len(prediction) == len(sentence))
        return prediction
            