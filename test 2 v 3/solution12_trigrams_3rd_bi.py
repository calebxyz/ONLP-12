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
        self.N = 2
        self.tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        self.e = dict()

        self.n_tag_set = list(itertools.product(self.tag_set, repeat=self.N))
        for i in range(1, self.N):
            self.n_tag_set += list(("SOS",) * i + x for x in itertools.product(self.tag_set, repeat=max(0, self.N - i)))

        self.p_tag_set = list(itertools.product(self.tag_set, repeat=self.N - 1))
        for i in range(1, self.N - 1):
            self.p_tag_set += list(("SOS",) * i + x for x in itertools.product(self.tag_set, repeat=max(0, (self.N - 1)- i)))

        self.all_tag_set = []
        for size in range(1, self.N + 1):
            self.all_tag_set += [("SOS",) * size]
            self. all_tag_set += list(itertools.product(self.tag_set, repeat=size))
            for i in range(1, size):
                self.all_tag_set += list(("SOS",) * i + x for x in itertools.product(self.tag_set, repeat=max(0, size - i)))

        self.p_tag_set_indexer = {}
        for idxS in range(0, len(self.p_tag_set)):
            if idxS not in self.p_tag_set_indexer:
                self.p_tag_set_indexer[idxS] = []
                for idxPS in range(0, len(self.p_tag_set)):
                    if self.p_tag_set[idxS][:-1] == self.p_tag_set[idxPS][1:]:
                        self.p_tag_set_indexer[idxS].append(idxPS)


        self.t = dict.fromkeys(self.all_tag_set, 0)
        self.t["ALL"] = 0

        self.tp = dict()
        self.ep = dict()
        self.delta = 0.1
        self.V = 1


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
                    self.V += 1
                else:
                    self.e[pair[0]] += 1

        for k, v in self.e.items():
            if isinstance(k, str):
                self.ep[k] = self.delta / (self.e[k] + self.delta * self.V)
            else:
                self.ep[k] = (self.e[k] + self.delta) / (self.e[k[0]] + self.delta * self.V)



    
    def _estimate_transition_probabilites(self, annotated_sentences):
        for sentence in annotated_sentences:
            for size in range(1, self.N):
                self.t[("SOS",) * size] += self.N - size

            self.t["ALL"] += self.N - 1 #count of SOS in start of the sentence

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

        for k, v in self.t.items():
            if not isinstance(k, str) and len(k) > 1:
                val = self.t[k[:-1]]
                if val > 0:
                    self.tp[k] = v / val
                else:
                    self.tp[k] = val
            else:
                self.tp[k] = v / self.t["ALL"]



    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''

        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)
        self.lambdas = self.deleted_interpolation()
        return self

    def maxViterbi(self, sentence, s, t):
        maxVal = 0
        maxState = 0
        for pS in self.p_tag_set_indexer[s]: #range(0, len(self.p_tag_set)):
            pSTag = self.p_tag_set[pS]
            isEq = True
            for i in range(1, self.N - 1):
                if pSTag[i] != self.p_tag_set[s][i-1]:
                    isEq = False
                    break

            if isEq == True:
                value = self.viterbiMat[pS][t-1] * self.getProb(pSTag + (self.p_tag_set[s][-1],),  sentence[t]) #self.viterbiMat[pS, t] * P('s' | "s'") * P("w|s")
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


    def getProb(self, tag, word):
        prob = 0
        mul = 1

        if self.N > 2:
            mul = self.lambdas[len(tag)-1]

        if len(tag) == self.N:
            emitTag = (tag[-1],) + (word,)
            if emitTag not in self.ep:
                if self.delta == 0:
                    return 0
                else:
                    eprob = self.ep[tag[-1]]
            else:
                eprob = self.ep[emitTag] #/ self.e[tag[-1]]

            if self.N > 2:
                return (mul * self.tp[tag] + self.getProb(tag[1:], word)) * eprob
            else:
                return self.tp[tag] * eprob

        elif len(tag) > 1:
            prob = self.getProb(tag[1:], word)

        return (mul * self.tp[tag]) + prob


    def deleted_interpolation(self):
        lambdas = [0.0] * self.N
        allTagCounts = self.t["ALL"]

        for tag in self.n_tag_set:
            if self.t[tag] > 0:
                cases = [0.0] * self.N
                for size in range(2, self.N + 1):
                    #dominator is less by 1
                    if self.t[tag[self.N - (size - 1):]] - 1 == 0:
                        cases[size - 1] = 0
                    else:
                        cases[size - 1] = (self.t[tag[self.N - size:]] - 1)/(self.t[tag[self.N - (size - 1):]] - 1)

                #size of one tag
                cases[0] = (self.t[tag[self.N - 1:]] - 1) / allTagCounts

                maxPos = cases.index(max(cases))
                lambdas[maxPos] += self.t[tag]

        sum_lambdas = sum(lambdas)
        return [x / sum_lambdas for x in lambdas]





    def viterbi(self, sentence):
        # size of matrix n x m
        self.viterbiMat = [[0 for i in range(len(sentence))] for j in range(len(self.p_tag_set))]
        # size of matrix n x m
        self.backpointerMat = [[0 for i in range(len(sentence))] for j in range(len(self.p_tag_set))]

        for s in range(0, len(self.p_tag_set)):
            self.viterbiMat[s][0] = self.getProb(("SOS",) + self.p_tag_set[s], sentence[0])  #P("s|<s>") * P("w|s")
            self.backpointerMat[s][0] = 0

        for t in range(1, len(sentence)):
            for s in range(0, len(self.p_tag_set)):
                self.backpointerMat[s][t], self.viterbiMat[s][t] = self.maxViterbi(sentence, s, t)

        bestpathpointer, bestpathprob = self.bestPathViterbi(len(sentence) - 1)

        states = [0] * len(sentence)
        for t in reversed(range(0, len(sentence))):
            states[t] = self.p_tag_set[bestpathpointer][-1]
            bestpathpointer = self.backpointerMat[bestpathpointer][t]

        return states

    def predict(self, sentence):
        prediction = self.viterbi(sentence)
        assert (len(prediction) == len(sentence))
        return prediction