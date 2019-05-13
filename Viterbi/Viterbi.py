a = dict()
b = dict()

def learn(sentence):
    for word in sentence.split():
        if word in a:
            a[word] += 1
        else:
            a[word] = 1




def maxViterbi(viterbiMat, stateGraph, s, t):
    maxVal = -1
    maxState = stateGraph[0]
    for pS in range(0, len(stateGraph)):
        value = viterbiMat[pS, t] * P('s'|"s'")*P("w|s")
        if(maxVal < value)
            maxVal = value
            maxState = pS

    return (maxState,maxVal)

def bestPathViterbi(viterbiMat, stateGraph, t):
    maxProb = -1
    maxState = stateGraph[0]
    for s in range(0, len(stateGraph)):
        value = viterbiMat[s, t]
        if(maxProb < value)
            maxProb = value
            maxState = s

    return (maxState, maxProb)



def viterbi(sentence, stateGraph):
    words = sentence.split()
    # size of matrix n x m

    viterbiMat = [[0 for i in range(n)] for j in range(m)]
    # size of matrix n x m
    backpointerMat = [[0 for i in range(n)] for j in range(m)]

    for s in range(0, len(stateGraph)):
        viterbiMat[s, 0] = P("s|<s>") * P("w|s")
        backpointerMat[s, 0] = 0

    for t in range(1, len(words) - 1):
        for s in range(0, len(stateGraph) - 1):
            backpointerMat[s, t], viterbiMat[s, t] = maxViterbi(viterbiMat, stateGraph, s, t - 1)

    bestpathpointer, bestpathprob = bestPathViterbi(viterbiMat, stateGraph, t)

    states = []
    for t in range(len(stateGraph) - 1, 0):
        states.insert(backpointerMat[bestpathpointer, t])
        bestpathpointer = backpointerMat[bestpathpointer, t]

    states.reverse()
    print(states)


states = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()