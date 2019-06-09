import os
import collections
import json
from lib.util import CONST_END_WORD
from lib.util import sliding, ngramWindow

import logging

logger = logging.getLogger(__name__)


class NGramModel(object):
    def __init__(self, windowSizeList, generatePdf=False, filters=[]):
        self.windowSizeList = windowSizeList
        if 1 not in self.windowSizeList:
            self.windowSizeList.append(1)

        self.ngramCount = {}
        self.ntotalCounts = {}
        self.ngramDomainTopicDict = {}
        self.nGramPdf = {}
        self.nGramProb = {}
        self.baseDir = None
        self.domainList = []
        self.filters = filters
        self.vocabSize = 0
        self.kSmoothingFactor = 0.00001
        self.generatePdf = generatePdf
        #self.kSmoothingFactor = 0.0001

    def _count(self):
        windowSizeList = self.windowSizeList
        # Needed to get vocab size
        if 1 not in self.windowSizeList:
            windowSizeList.append(1)

        for wSize in windowSizeList:
            ngramCount = collections.Counter()
            ntotalCounts = collections.Counter()
            for domain in self.domainList:
                dataDir = os.path.join(self.baseDir, domain)
                for (currDir, _, fileList) in os.walk(dataDir):
                    for filename in fileList:

                        if 'all' not in self.filters:
                            skipFile = True
                            for fl in self.filters:
                                if fl in filename:
                                    skipFile = False
                                    break

                            if skipFile:
                                continue

                        if filename.endswith('.json'):
                            fullName = os.path.join(currDir, filename)
                            logger.debug("filename : %s" % fullName)
                            with open(fullName, "r") as f:
                                cData = json.load(f)
                                articleText = cData.get("article", None)
                                domain = cData.get("domain", None)
                                topic = cData.get("theme", None)
                                domainTopicKey = (domain, topic)
                                if not articleText:
                                    logger.debug("%s skipped" % filename)
                                    continue

                            textList = articleText.strip().split(" ")
                            # Update vocab size
                            logger.debug("textList : %s" % textList)
                            ngramList = [ngramWindow(wordSeg, wSize) for wordSeg in sliding(textList, wSize)]
                            # Add last words
                            prevWord = list(ngramList[-1])
                            lastWord = prevWord[1:] + [CONST_END_WORD]
                            ngramList.append(tuple(lastWord))

                            #for ngram in ngramList:
                            #    ngramKey = ngram[:-1]
                            #    if ngramKey not in self.ngramDomainTopicDict:
                            #        self.ngramDomainTopicDict[ngramKey] = [domainTopicKey]
                            #        continue
                            #
                            #    self.ngramDomainTopicDict[ngramKey].append(domainTopicKey)

                            ngramCount.update(ngramList)
                            if wSize != 1:
                                ntotalCounts.update([x[:-1] for x in ngramList])

            self.ngramCount[wSize] = ngramCount
            if wSize != 1:
                self.ntotalCounts[wSize] = ntotalCounts

        self.vocabSize = sum([wordCount for _, wordCount in self.ngramCount[1].items()])

    def _nGramProb(self):
        vSizeFraction = self.kSmoothingFactor * float(self.vocabSize)
        for wSize in self.windowSizeList:
            nGramProb = collections.defaultdict(float)
            nGramPdf = {}
            for ngram in self.ngramCount[wSize]:
                ngramCount = self.ngramCount[wSize][ngram]
                if wSize != 1:
                    totalCount = self.ntotalCounts[wSize][ngram[0:-1]]
                else:
                    totalCount = self.vocabSize

                if self.generatePdf:
                    ngramKey = ngram[:-1]
                    if ngramKey not in self.nGramPdf:
                        nGramPdf[ngramKey] = [(float(ngramCount) / totalCount, ngram[-1])]
                        continue

                    nGramPdf[ngramKey].append((float(ngramCount) / totalCount, ngram[-1]))

                #ngramKey = ngram[:-1]
                #for domainTopicKey in self.ngramDomainTopicDict.get(ngramKey, []):
                #    if ngramKey not in self.nGramPdf:
                #        self.nGramPdf[ngramKey] = {domainTopicKey: [(float(ngramCount) / totalCount, ngram[-1])]}
                #        continue
                #
                #    if domainTopicKey not in self.nGramPdf[ngramKey]:
                #        self.nGramPdf[ngramKey][domainTopicKey] = [(float(ngramCount) / totalCount, ngram[-1])]
                #
                #    self.nGramPdf[ngramKey][domainTopicKey].append((float(ngramCount) / totalCount, ngram[-1]))

                #self.nGramProb[ngram] += float(ngramCount) / totalCount
                nGramProb[ngram] += (float(ngramCount) + self.kSmoothingFactor) / (vSizeFraction + totalCount)

            self.nGramProb[wSize] = nGramProb
            self.nGramPdf[wSize] = nGramPdf
            #self._normalizeprob()

    def _normalizeprob(self):
        probListTotal = float(sum([mProb for _, mProb in self.nGramProb.items()]))
        for cNgram, pVal in self.nGramProb.items():
            self.nGramProb[cNgram] = pVal / probListTotal

        #for cNgram in self.nGramPdf:
        #    for topic, pValList in self.nGramPdf[cNgram].items():
        #        print(pValList)
        #        self.nGramProb[cNgram][topic] = [ap / probListTotal for ap in pValList]


    def countNgrams(self, domainList, baseDir="."):
        self.baseDir = baseDir
        self.domainList = domainList
        self._count()

    def generateModel(self):
        self._nGramProb()
        return self.nGramPdf
        #return self.nGramProb

