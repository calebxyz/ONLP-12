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

    def __init__(self):
        self.tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        self.e = dict()
        self.e["SOS"] = 0
        self.e["EOS"] = 0

        self.t = dict()
        self.t["SOS"] = 0
        self.t["EOS"] = 0

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
            for idx, wordTagPair in enumerate(sentence):
                pair = (type(None), type(None))
                if idx == 0:
                    pair = ("SOS", wordTagPair[1])
                else:
                    pair = (sentence[idx - 1][1], wordTagPair[1])

                if pair not in self.t:
                    self.t[pair] = 1
                else:
                    self.t[pair] += 1

                if pair[0] not in self.t:
                    self.t[pair[0]] = 1
                else:
                    self.t[pair[0]] += 1

            pair = (sentence[len(sentence) - 1][1], "EOS")

            if pair not in self.t:
                self.t[pair] = 1
            else:
                self.t[pair] += 1

            if pair[0] not in self.t:
                self.t[pair[0]] = 1
            else:
                self.t[pair[0]] += 1

    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''

        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)

        return self

    def maxViterbi(self, sentence, s, t):
        maxVal = -1
        maxState = self.tag_set[0]
        for pS in range(0, len(self.tag_set)):
            value = self.viterbiMat[pS][t-1] * self.getProb(self.tag_set[pS], self.tag_set[s],  sentence[t]) #self.viterbiMat[pS, t] * P('s' | "s'") * P("w|s")
            if maxVal < value:
                maxVal = value
                maxState = pS

        return (maxState, maxVal)


    def bestPathViterbi(self, t):
        maxProb = -1
        maxState = self.tag_set[0]
        for s in range(0, len(self.tag_set)):
            value = self.viterbiMat[s][t]
            if maxProb < value:
                maxProb = value
                maxState = s

        return (maxState, maxProb)


    def getProb(self, pTag, tag, word):
        if word == "EOS":
            if (pTag, "EOS") not in self.t:
                return 0
            else:
                return self.t[(pTag, "EOS")] / self.t[(pTag)]
        elif (pTag, tag) in self.t and (tag, word) in self.e:
            return self.t[(pTag, tag)] / self.t[(pTag)] * self.e[(tag, word)] / self.e[(tag)]  # P("s|prev s") * P("w|s")
        else:
            return 0


    def viterbi(self, sentence):

        #sentence = sentence + ["EOS"]

        # size of matrix n x m
        self.viterbiMat = [[0 for i in range(len(sentence))] for j in range(len(self.tag_set))]
        # size of matrix n x m
        self.backpointerMat = [[0 for i in range(len(sentence))] for j in range(len(self.tag_set))]

        for s in range(0, len(self.tag_set)):
            self.viterbiMat[s][0] = self.getProb("SOS", self.tag_set[s], sentence[0])  #P("s|<s>") * P("w|s")
            self.backpointerMat[s][0] = 0

        for t in range(1, len(sentence)):
            for s in range(0, len(self.tag_set)):
                self.backpointerMat[s][t], self.viterbiMat[s][t] = self.maxViterbi(sentence, s, t)

        bestpathpointer, bestpathprob = self.bestPathViterbi(len(sentence) - 1)

        states = [0] * len(sentence)
        for t in reversed(range(0, len(sentence))):
            states[t] = self.tag_set[bestpathpointer]
            bestpathpointer = self.backpointerMat[bestpathpointer][t]

        # print(sentence)
        # print(states)
        #del states[-1]
        return states

    def predict(self, sentence):
        prediction = self.viterbi(sentence)
        assert (len(prediction) == len(sentence))
        return prediction