

import os
import collections
import math
import json
import random
import re

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
def WeightSample(weightDict):
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


class MarkovModel(object):
    def __init__(self, nGramModel, windowSize, domain, topic):
        self.domain = domain
        self.topic = topic
        self.nGramModel = nGramModel
        self.windowSize = windowSize

    def startState(self):
        return [CONST_START_WORD for _ in range(self.windowSize)]

    def _generateBiasedWeights(self, weightDict, biasedDomainTopic):
        # To generate content with specific domain/topic, we search
        # for the ngram only in ngrams from same domain/topic.
        # If the model does is not trained for the speicifed domain/topic,
        # we use the full ngram list.

        biasedWeights = {}
        totalWeights = 0.0
        for currWord, (weight, cDomainTopic) in weightDict.items():
            if cDomainTopic == biasedDomainTopic:
                biasedWeights.update({currWord: weight})

            totalWeights += weight
            biasedWeights.update({currWord: weight})

        #normalize
        for word, weight in biasedWeights.items():
            biasedWeights[word] = weight / totalWeights

        return biasedWeights

    # return next word
    def generate(self, cState):
        cDict = {}
        ngramKey = tuple(cState[1:])
        # Given we are generating n-grams from our trained mode,
        # we should not reach the case that ngram is not found
        if ngramKey not in self.nGramModel:
            raise Exception("Generated article should not have unknown ngrams")

        # To generate content with specific domain/topic, we search
        # for the ngram only in ngrams from same domain/topic.
        # If the model  is not trained for the specified domain/topic,
        # we use ngrams from all the trained domains/topics.
        biasedDomainTopic = (self.domain, self.topic)
        if biasedDomainTopic in self.nGramModel[ngramKey]:
            for weight, currWord in self.nGramModel[ngramKey][biasedDomainTopic]:
                cDict.update({currWord: weight})
        else:
            for domainTopic in self.nGramModel[ngramKey]:
                for weight, currWord in self.nGramModel[ngramKey][domainTopic]:
                    cDict.update({currWord: weight})

        sampleWord = WeightSample(cDict)
        # construct next state
        cState = [word for word in cState[1:]]
        cState.append(sampleWord)
        cState = tuple(cState)
        return cState


class NGramModel(object):
    def __init__(self, windowSize, filters=[]):
        self.windowSize = windowSize
        self.ngramCount = collections.Counter()
        self.ntotalCounts = collections.Counter()
        self.ngramDomainTopicDict = {}
        self.nGramPdf = {}
        self.nGramProb = collections.defaultdict(float)
        self.dataDir = None
        self.filters = filters

    def _count(self):
        for (currDir, _, fileList) in os.walk(self.dataDir):
            for filename in fileList:
                # extract results folder
                #if self.topic in filename:
                # filter for specific domain and/or filename if given
                skipFile = True
                for fl in self.filters:
                    if fl in filename:
                        skipFile = False
                        break

                if skipFile:
                    #print("%s skipped" % filename)
                    continue

                if filename.endswith('.json'):
                    fullName = os.path.join(currDir, filename)
                    #print("filename : %s" % fullName)
                    with open(fullName, "r") as f:
                        cData = json.load(f)
                        articleText = cData.get("article", None)
                        domain = cData.get("domain", None)
                        topic = cData.get("theme", None)
                        domainTopicKey = (domain, topic)
                        if not articleText:
                            print("%s skipped" % filename)
                            continue

                    textList = articleText.strip().split(" ")
                    #print("textList : %s" % textList)
                    ngramList = [ngramWindow(wordSeg, self.windowSize) for wordSeg in sliding(textList, self.windowSize)]
                    # Add last words
                    prevWord = list(ngramList[-1])
                    lastWord = prevWord[1:] + [CONST_END_WORD]
                    ngramList.append(tuple(lastWord))

                    for ngram in ngramList:
                        ngramKey = ngram[:-1]
                        if ngramKey not in self.ngramDomainTopicDict:
                            self.ngramDomainTopicDict[ngramKey] = [domainTopicKey]
                            continue

                        self.ngramDomainTopicDict[ngramKey].append(domainTopicKey)

                    self.ngramCount.update(ngramList)
                    self.ntotalCounts.update([x[:-1] for x in ngramList])

    def _nGramProb(self):
        for ngram in self.ngramCount:
            totalCount = self.ntotalCounts[ngram[0:-1]]
            ngramCount = self.ngramCount[ngram]
            ngramKey = ngram[:-1]
            for domainTopicKey in self.ngramDomainTopicDict.get(ngramKey, []):
                if ngramKey not in self.nGramPdf:
                    self.nGramPdf[ngramKey] = {domainTopicKey: [(float(ngramCount) / totalCount, ngram[-1])]}
                    continue

                if domainTopicKey not in self.nGramPdf[ngramKey]:
                    self.nGramPdf[ngramKey][domainTopicKey] = [(float(ngramCount) / totalCount, ngram[-1])]

                self.nGramPdf[ngramKey][domainTopicKey].append((float(ngramCount) / totalCount, ngram[-1]))
                self.nGramProb[ngram] += float(ngramCount) / totalCount

    def countNgrams(self, newsSource, baseDir="."):
        # File path is <newSource>/<*topic*.json>
        self.dataDir = os.path.join(baseDir, newsSource)
        self._count()

    def generateModel(self):
        self._nGramProb()
        return self.nGramPdf
        #return math.log(totalCount + VOCAB_SIZE) - math.log(ngramCount + 1)


def computePerplexity(testSentance, ng):
    textList = testSentance.strip().split(" ")
    ngramList = [ngramWindow(segWord, ng.windowSize) for segWord in sliding(textList, ng.windowSize)]
    sumProb = 0.0
    for ngram in ngramList:
        if ngram in ng.nGramProb:
            sumProb += ng.nGramProb[ngram]

    modelPerplexity = 2 ** (-1 / len(ngramList) * sumProb)
    print("ngram sumProb: %s" % modelPerplexity)

windowSize = 3
baseDir = "../"
#filters = ["EDUCATION", "EXTREMISM"]
filters = ["EDUCATION", "EXTREMISM"]
ng = NGramModel(windowSize, filters=filters)
#ng.countNgrams("npr.org", baseDir)
ng.countNgrams("cnn.com", baseDir)

#ng = NGramModel(windowSize, "cnn.com", "EDUCATION", "../")
#ng = NGramModel(windowSize, "foxnews.com", "EDUCATION", "../")
#ng = NGramModel(windowSize, "cnn.com", "EXTREMISM", "../")
#ng = NGramModel(windowSize, "foxnews.com", "EXTREMISM", "../")
#ng = NGramModel(windowSize, "foxnews.com", "TAX_FNCACT_PRESIDENTS", "../")
#ng = NGramModel(windowSize, "cnn.com", "TAX_FNCACT_PRESIDENTS", "../")
#ng = NGramModel(windowSize, "npr.org", "TAX_FNCACT_PRESIDENTS", "../")

nGramModel = ng.generateModel()
#for key in nGramModel:
#    print("%s: %s" % (list(key), nGramModel[key]))

maxLength = 1000
mv = MarkovModel(nGramModel, windowSize, "cnn.com", "EDUCATION")
cState = mv.startState()
article = []
for _ in range(maxLength):
    cState = mv.generate(cState)
    cWord = cState[-1]
    if cWord == CONST_END_WORD:
        print("reached end of the article")
        break

    article.append(cState[-1])

print(" ".join(article))

#computePerplexity("college admissions scandal reinforced for many what they have long believed", ng)
computePerplexity("college isuu pycharm reinforced for many what they have long believed", ng)



