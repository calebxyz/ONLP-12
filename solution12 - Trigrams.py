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
import itertools
class Submission(SubmissionSpec12):
    ''' a contrived poorely performing solution for question one of this Maman '''

    def __init__(self):
        self.N = 3
        self.tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        self.e = dict()

        self.p_tag_set = list(itertools.product(self.tag_set, repeat=self.N - 1))
        for i in range(1, self.N - 1):
            self.p_tag_set += list(("SOS",) * i + x for x in itertools.product(self.tag_set, repeat=max(0, self.N - i)))

        self.all_tag_set = []
        for size in range(1, self.N + 1):
            self.all_tag_set += [("SOS",) * size]
            self. all_tag_set += list(itertools.product(self.tag_set, repeat=size))
            for i in range(1, size):
                self.all_tag_set += list(("SOS",) * i + x for x in itertools.product(self.tag_set, repeat=max(0, size - i)))

        #self.start_tags = list(("SOS",) * (self.N - 1) + (t,) for t in self.tag_set)

        self.t = dict.fromkeys(self.all_tag_set, 0)
        self.t["ALL"] = 0

    def _estimate_emission_probabilites(self, annotated_sentences):
        for sentence in annotated_sentences:
            for idx, wordTagPair in enumerate(sentence):
                pair = (wordTagPair[1], wordTagPair[0])

                if pair not in self.e:
                    self.e[pair] = 1
                else:
                    self.e[pair] += 1

                if pair[0] not in self.e:
                    self.e[pair[0]] = 1
                else:
                    self.e[pair[0]] += 1
    
    def _estimate_transition_probabilites(self, annotated_sentences):
        for sentence in annotated_sentences:
            for size in range(1, self.N):
                self.t[("SOS",) * size] += 1
                self.t["ALL"] += 1

            for idx, wordTagPair in enumerate(sentence):
                self.t["ALL"] += 1
                for size in range(1, self.N + 1):
                    arr = [0] * size
                    for tplIdx, i in enumerate(range(idx - (size - 1), idx + 1)):
                        if i < 0:
                            arr[tplIdx] = "SOS"
                        else:
                            arr[tplIdx] = sentence[i][1]

                    self.t[tuple(arr)] += 1

    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''

        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)

        return self

    def maxViterbi(self, sentence, s, t):
        maxVal = 0
        maxState = 0
        for pS in range(0, len(self.p_tag_set)):
            if self.p_tag_set[pS][1:] == self.p_tag_set[s][:-1]:
                value = self.viterbiMat[pS][t-1] * self.getProb(self.p_tag_set[pS], (self.p_tag_set[s][-1],),  sentence[t]) #self.viterbiMat[pS, t] * P('s' | "s'") * P("w|s")
                if maxVal < value:
                    maxVal = value
                    maxState = pS

        return (maxState, maxVal)


    def bestPathViterbi(self, t):
        maxProb = 0
        maxState = 0
        for s in range(0, len(self.p_tag_set)):
            value = self.viterbiMat[s][t]
            if maxProb < value:
                maxProb = value
                maxState = s

        return (maxState, maxProb)


    def getProb(self, pTag, tag, word):
        totalTag = pTag + tag
        emitTag = tag + (word,)
        prob = 0
        mul =0
        if len(pTag) == 0:
            mul = 0.1
        elif len(pTag) == 1:
            mul = 0.3
        elif len(pTag) == 2:
            mul = 0.6

        if emitTag not in self.e:
            return 0

        if len(pTag) > 0:
            prob += self.getProb(pTag[1:], tag, word)

            if totalTag in self.t and pTag in self.t and self.t[pTag] > 0 and emitTag in self.e:
                prob += mul * self.t[totalTag] / self.t[pTag]
        else:
            if totalTag in self.t and emitTag in self.e:
                prob += mul * self.t[totalTag] / self.t["ALL"]   # P("s|prev s") * P("w|s")

        if len(pTag) == self.N - 1:
            return prob * self.e[emitTag] / self.e[tag[0]]
        else:
            return prob




    def viterbi(self, sentence):
        # size of matrix n x m
        self.viterbiMat = [[0 for i in range(len(sentence))] for j in range(len(self.p_tag_set))]
        # size of matrix n x m
        self.backpointerMat = [[0 for i in range(len(sentence))] for j in range(len(self.p_tag_set))]

        for s in range(0, len(self.p_tag_set)):
            self.viterbiMat[s][0] = self.getProb(("SOS",) + self.p_tag_set[s][:-1], (self.p_tag_set[s][-1],), sentence[0])  #P("s|<s>") * P("w|s")
            self.backpointerMat[s][0] = 0

        for t in range(1, len(sentence)):
            for s in range(0, len(self.p_tag_set)):
                self.backpointerMat[s][t], self.viterbiMat[s][t] = self.maxViterbi(sentence, s, t)

        bestpathpointer, bestpathprob = self.bestPathViterbi(len(sentence) - 1)

        states = [0] * len(sentence)
        for t in reversed(range(0, len(sentence))):
            states[t] = self.p_tag_set[bestpathpointer][-1]
            bestpathpointer = self.backpointerMat[bestpathpointer][t]

        # print(sentence)
        # print(states)
        #del states[-1]
        return states

    def predict(self, sentence):
        prediction = self.viterbi(sentence)
        assert (len(prediction) == len(sentence))
        return prediction