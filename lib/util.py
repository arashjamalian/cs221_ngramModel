import random
import math

CONST_START_WORD = "__s__"
CONST_END_WORD = "__e__"


# Sliding function (borrowed from hw3)
def sliding(xs, windowSize):
    for i in range(1, len(xs) + 1):
        yield xs[max(0, i - windowSize):i]


def ngramWindow(wordSeg, windowSize):
    if len(wordSeg) < windowSize:
        startWords = [CONST_START_WORD for _ in range(len(wordSeg), windowSize)]
        startWords = startWords + wordSeg
        return tuple(startWords)
    else:
        return tuple(wordSeg)


# Function: Weighted Random Choice (borrowed from hw7)
def weightSample(weightDict):
    weights = []
    elems = []
    for elem in sorted(weightDict):
        weights.append(weightDict[elem])
        elems.append(elem)
    total = sum(weights)
    key = random.uniform(0, total)
    runningTotal = 0.0
    for i in range(len(weights)):
        weight = weights[i]
        runningTotal += weight
        if runningTotal > key:
            chosenIndex = i
            return elems[chosenIndex]
    raise Exception('Should not reach here')


def computePerplexity(testSentance, ng):
    textList = testSentance.strip().split(" ")
    widnowSize = ng.windowSizeList[-1]
    ngramList = [ngramWindow(segWord, widnowSize) for segWord in sliding(textList, widnowSize)]
    sumProb = 0.0
    notExistProb = ng.kSmoothingFactor / ng.vocabSize
    for ngram in ngramList:
        for wIndex, wSize in enumerate(reversed(ng.windowSizeList)):
            #wSize = widnowSize
            #print("wSize: %s" % wSize)
            if ngram[wIndex:] in ng.nGramProb[wSize]:
                sumProb += math.log((ng.nGramProb[wSize][ngram[wIndex:]]), 2)
                #sumProb *= 1.0 / ng.nGramProb[ngram]
                break

        else:
            sumProb += math.log(notExistProb, 2)
            # sumProb *= 1.0 / notExistProb

    #print("nCount : %s cCount: %s" % (notCount, cCount))
    modelPerplexity = 2 ** (-1 / len(ngramList) * sumProb)
    #modelPerplexity = sumProb ** (1.0 / float(len(ngramList)))
    return modelPerplexity
