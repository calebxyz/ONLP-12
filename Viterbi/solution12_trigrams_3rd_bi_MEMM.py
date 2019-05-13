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
        self.tag_setIndexDic = dict()
        for i in range(len(self.tag_set)):
            self.tag_setIndexDic[self.tag_set[i]] = i
        self.tag_setIndexDic["SOS"] = len(self.tag_set)


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
        self.LRM = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

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

    def _word_vectorize(self, word):
        ''' vectorizes a word while ignoring its context '''

        # ngram occurences as prefix
        numOfSeq = 10
        sval = ord('א')
        eval = ord('ת')
        totalABChars = eval - sval + 1
        vec1 = [0] * (len(self.ngrams) * 3 + (totalABChars * numOfSeq) + 1)

        offestSecondNgram = len(self.ngrams)
        offestThirdNgram = len(self.ngrams) * 2
        offestAB = len(self.ngrams) * 3
        offsetAfterAB = (len(self.ngrams) * 3 + (totalABChars * numOfSeq))

        for idx, ngram in enumerate(self.ngrams):
            if ngram in word:
                vec1[idx] = 1
            if word.startswith(ngram):
                vec1[offestSecondNgram + idx] = 1
            if word.endswith(ngram):
                vec1[offestThirdNgram + idx] = 1

        for index in range(0, min(len(word) - 1, numOfSeq)):
            val = ord(word[index])
            if val >= sval and val <= eval:
                vec1[(offestAB + (totalABChars * index)) + val - sval] = 1

        vec1[offsetAfterAB] = int(re.search("\d", word) != None)

        return vec1

    def vectorize(self, sentence, token_index):
        '''
        the vectorization endpoint to be called by the class user.
        this specific implementation is an example function for
        vectorizing a token in sentence context.

        the arguments follow the semantics defined in the abstract class
        '''

        ##############################
        # features from/of the token #
        ##############################

        token = sentence[token_index]

        vec1 = self._word_vectorize(token[0])
        ################################
        # features from/of its context #
        ################################

        vec2 = [0] * len(self.tag_setIndexDic)

        if token_index > 0:
            prev_token = sentence[token_index - 1]  # the previous token in the sentence
            vec2[self.tag_setIndexDic[prev_token[1]]] = 1
        else:
            vec2[len(self.tag_setIndexDic) - 1] = 1 #"SOS"

        # if token_index < len(sentence) - 1:
        #     next_token = sentence[token_index + 1]  # the previous token in the sentence
        #     vec3 = self._word_vectorize(next_token)
        # else:
        #     # an all-zeros vector the length of the word vector (this is arbitrary)
        #     vec3 = [0] * len(vec1)

        # our feature vector
        return vec1 + vec2 # + vec3

    def trainLR(self, data):
        '''
        the training endpoint to be called by the class user.
        see the abstract class for the arguments spec.
        '''

        ## beginning of added section ##

        # get all unique tokens
        unique_tokens = set()
        for sentence in data:
            for pair in sentence:
                unique_tokens.add(pair[0])  # add the token form

        # extract all 1-grams
        min_ngram_len = 1
        max_ngram_len = 2

        self.ngrams = set()
        for token in unique_tokens:
            self.ngrams |= self._get_word_ngrams(min_ngram_len, max_ngram_len, token)  # adding to the set

        ## end of added section ##

        ## vectorizing the data
        X = []
        y = []
        for sentence in data:
            for i in range(len(sentence)):
                X.append(self.vectorize(sentence, i))
                y.append(self.tag_setIndexDic[sentence[i][1]])
        # X = list(map(lambda datum: self.vectorize(*datum),
        #              data))  # the asterisk unpacks the (sentence, index) tuple into a function arguments list for the function being called

        assert len(X) == len(y)

        model = LogisticRegression(solver='liblinear', multi_class='auto')
        #
        # sample_weights = []
        #
        # for label in y:
        #     if label == 0:
        #         weight = 1
        #     elif label == 1:
        #         weight = 1
        #     elif label == 2:
        #         weight = 1
        #     sample_weights.append(weight)

        model.fit(X, y)

        self.LRM = model




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
        self.trainLR(annotated_sentences)

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
                value = self.viterbiMat[pS][t-1] * self.getProbPred([(pSTag[0], pSTag[0]), (sentence[0], (self.p_tag_set[s][-1],))])
                #self.getProb(pSTag + (self.p_tag_set[s][-1],),  sentence[t]) #self.viterbiMat[pS, t] * P('s' | "s'") * P("w|s")
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

    def getProbPred(self, sentence):
        vect = self.vectorize(sentence, 1)
        pred = self.LRM.predict_proba([vect])[0]
        result = [0] * len(self.tag_setIndexDic)
        for idx in range(len(pred)):
            result[self.LRM.classes_[idx]] = pred[idx]
        return result[self.tag_setIndexDic[sentence[1][1][0]]]





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
            self.viterbiMat[s][0] = self.getProbPred([("SOS","SOS"), (sentence[0], self.p_tag_set[s])])
                #self.getProb(("SOS",) + self.p_tag_set[s], sentence[0])  #P("s|<s>") * P("w|s")
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