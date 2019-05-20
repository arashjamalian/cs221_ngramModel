

from argparse import ArgumentParser
from markov_model import MarkovModel
from ngram_model import NGramModel
from util import CONST_END_WORD
from util import computePerplexity

parser = ArgumentParser(description="Run ngram")
parser.add_argument('--filters', '-f', action='append',
                    help='Filters for data collection')
parser.add_argument('--domain', '-d', action='store', required=True,
                    help='Specify domain (news source) for generating article')
parser.add_argument('--topic', '-t', action='store',
                    help='Specify topic for generating article', required=True)
parser.add_argument('--size', '-s', action='store', default=3,
                    help='Number of of words in ngram')
parser.add_argument('--basedir', '-b', action='store',
                    help='base directory for input data', required=True)
parser.add_argument('--length', '-l', action='store', default=1000,
                    help='Maximum number of words in generated article')

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
ng.countNgrams("cnn.com", baseDir)
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

#computePerplexity("college admissions scandal reinforced for many what they have long believed", ng)
computePerplexity("college isuu pycharm reinforced for many what they have long believed", ng)
