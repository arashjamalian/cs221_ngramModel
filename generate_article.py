#!/usr/bin/env python3.4

from argparse import ArgumentParser
from markov_model import MarkovModel
from ngram_model import NGramModel
from util import CONST_END_WORD
from util import computePerplexity
import os
import json

parser = ArgumentParser(description="Generate article based on input domain and topic using ngram model")
parser.add_argument('--filters', '-f', action='append',
                    help='Filters for data collection')
parser.add_argument('--domain', '-d', action='store', required=True,
                    help='Specify domain (news source) for generating article')
parser.add_argument('--topic', '-t', action='store',
                    help='Specify topic for generating article', required=True)
parser.add_argument('--size', '-s', action='store', default=3,
                    help='Number of of words in ngram')
parser.add_argument('--basedir', '-b', action='store',
                    help='directory for train data', required=True)
parser.add_argument('--length', '-l', action='store', default=1000,
                    help='Maximum number of words in generated article')
parser.add_argument('--perplexity', '-p', action='store_true', default=False,
                    help='Compute perplexity on test data')
parser.add_argument('--testDir', '-T', action='store',
                    help='directory for test data')


args = parser.parse_args()

print("args: %s" % args)

filters = []
if args.filters:
    filters = args.filters

maxLength = int(args.length)
windowSize = int(args.size)
baseDir = args.basedir
domain = args.domain
topic = args.topic

ng = NGramModel(windowSize, filters=filters)
ng.countNgrams(baseDir)
nGramModel = ng.generateModel()


mv = MarkovModel(nGramModel, windowSize, domain, topic)
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

if args.perplexity:
    if not args.testDir:
        raise Exception("Need to input test directory")

    pvaluesList = []
    for (currDir, _, fileList) in os.walk(args.testDir):
        for filename in fileList:
            if filename.endswith('.json'):
                fullName = os.path.join(currDir, filename)
                # print("filename : %s" % fullName)
                with open(fullName, "r") as f:
                    cData = json.load(f)
                    articleText = cData.get("article", None)
                    pvaluesList.append(computePerplexity(articleText, ng))

    print("pvaluesList: %s" % pvaluesList)
    print("max perplexity: %s" % max(pvaluesList))
    print("mean perplexity: %s" % (sum(pvaluesList)/len(pvaluesList)))
    print("min perplexity: %s" % min(pvaluesList))
