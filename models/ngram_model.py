import os
import collections
import json
from lib.util import CONST_END_WORD
from lib.util import sliding, ngramWindow

import logging

logger = logging.getLogger(__name__)


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
                            logger.warning("%s skipped" % filename)
                            continue

                    textList = articleText.strip().split(" ")
                    logger.debug("textList : %s" % textList)
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

    def countNgrams(self, baseDir="."):
        self.dataDir = baseDir
        self._count()

    def generateModel(self):
        self._nGramProb()
        return self.nGramPdf
        #return math.log(totalCount + VOCAB_SIZE) - math.log(ngramCount + 1)

