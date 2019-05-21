compatible with python3.4 and python3.7
Example run command:
python3.4 generate_article.py -f "EDUCATION" -d cnn.com -t EDUCATION  -b /tmp/trainData/cnn.com -p -T /tmp/testData/foxnews.com

python3.4 generate_article.py -h
usage: generate_article.py [-h] [--filters FILTERS] --domain DOMAIN --topic
                           TOPIC [--size SIZE] --basedir BASEDIR
                           [--length LENGTH] [--perplexity]
                           [--testDir TESTDIR]

Generate article based on input domain and topic using ngram model

optional arguments:
  -h, --help            show this help message and exit
  --filters FILTERS, -f FILTERS
                        Filters for data collection
  --domain DOMAIN, -d DOMAIN
                        Specify domain (news source) for generating article
  --topic TOPIC, -t TOPIC
                        Specify topic for generating article
  --size SIZE, -s SIZE  Number of of words in ngram
  --basedir BASEDIR, -b BASEDIR
                        directory for train data
  --length LENGTH, -l LENGTH
                        Maximum number of words in generated article
  --perplexity, -p      Compute perplexity on test data
  --testDir TESTDIR, -T TESTDIR
                        directory for test data