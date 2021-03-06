#!/usr/bin/env python3.4

import os
import json
import logging
import sys
import time
import datetime

from argparse import ArgumentParser
from models.markov_model import MarkovModel
from models.ngram_model import NGramModel
from lib.util import CONST_END_WORD
from lib.util import computePerplexity

FORMAT = "%(asctime)s: %(name)s:%(lineno)d (%(process)d/%(threadName)s) - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


parser = ArgumentParser(description="Generate article based on input domain and topic using ngram model")
parser.add_argument('--filters', '-f', action='append',
                    help='Filters for data collection')
parser.add_argument('--domainList', '-d', action='append', required=True,
                    help='Specify domain (news source) for training ngram')
parser.add_argument('--generateDomain', '-D', action='store',
                    help='Specify domain for generating article', required=True)
parser.add_argument('--generateTopic', '-t', action='store',
                    help='Specify topic for generating article', required=True)
parser.add_argument('--sizeList', '-s', action='append',
                    help='Number of of words in ngram')
parser.add_argument('--basedir', '-b', action='store',
                    help='directory for train data', required=True)
parser.add_argument('--length', '-l', action='store', default=1000,
                    help='Maximum number of words in generated article')
parser.add_argument('--perplexity', '-p', action='store_true', default=False,
                    help='Compute perplexity on test data')
parser.add_argument('--generate', '-g', action='store_true', default=False,
                    help='generate article')
parser.add_argument('--testDir', '-T', action='store',
                    help='directory for test data')


args = parser.parse_args()

logger.info("args: %s" % args)

filters = []
if args.filters:
    filters = args.filters

maxLength = int(args.length)
windowSizeList = [int(size) for size in args.sizeList]
baseDir = args.basedir
domainList = args.domainList
generateDomain = args.generateDomain
generateTopic = args.generateTopic

startTime = time.time()
ng = NGramModel(windowSizeList, generatePdf=args.generate, filters=filters)
ng.countNgrams(domainList, baseDir)
nGramModel = ng.generateModel()

ngTypeCount = [key for key, _ in ng.ngramCount.items()]

print("vocab size: %s" % ng.vocabSize)
print("ngram types: %s" % len(ngTypeCount))
for size in windowSizeList:
    print("size of ngram model %s: %s" % (size, sys.getsizeof(ng.nGramProb[size])))

print("Time elapsed: %s" % str(datetime.timedelta(seconds=time.time()-startTime)))

if args.generate:
    # For generating sentences use max window size
    windowSize = max(windowSizeList)
    mv = MarkovModel(nGramModel, windowSize, generateDomain, generateTopic)
    cState = mv.startState()
    article = []
    for _ in range(maxLength):
        cState = mv.generate(cState)
        cWord = cState[-1]
        if cWord == CONST_END_WORD:
            logger.debug("reached end of the article")
            break

        article.append(cState[-1])

    print(" ".join(article))

if args.perplexity:
    if not args.testDir:
        raise Exception("Need to input test directory")

    pvaluesList = []
    articleText = ""
    for (currDir, _, fileList) in os.walk(args.testDir):
        for filename in fileList:
            if filename.endswith('.json'):
                fullName = os.path.join(currDir, filename)
                # print("filename : %s" % fullName)
                with open(fullName, "r") as f:
                    cData = json.load(f)
                    articleText += cData.get("article", None)
                    #articleText = cData.get("article", None)
                    #pvaluesList.append(computePerplexity(articleText, ng))

    #print("articleText: %s" % articleText)
    pvaluesList.append(computePerplexity(articleText, ng))

    logger.debug("pvaluesList: %s" % pvaluesList)
    logger.info("max perplexity: %s" % max(pvaluesList))
    logger.info("mean perplexity: %s" % (sum(pvaluesList)/len(pvaluesList)))
    logger.info("min perplexity: %s" % min(pvaluesList))
