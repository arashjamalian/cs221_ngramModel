compatible with python3.4 and python3.7

Example run command:

python3.4 generate_article.py -f "all" -d foxnews.com -d cnn.com -D cnn.com -t EDUCATION  -b /tmp/trainData/ -T /tmp/testData/cnn.com -s 1 -s 2 -s 3 -p

arguments:
  -h, --help            show this help message and exit
  --filters FILTERS, -f FILTERS
                        Filters for data collection

  --domainList DOMAINLIST, -d DOMAINLIST
                        Specify domain (news source) for training ngram

  --generateDomain GENERATEDOMAIN, -D GENERATEDOMAIN
                        Specify domain for generating article

  --generateTopic GENERATETOPIC, -t GENERATETOPIC
                        Specify topic for generating article

  --sizeList SIZELIST, -s SIZELIST
                        Number of of words in ngram

  --basedir BASEDIR, -b BASEDIR
                        directory for train data

  --length LENGTH, -l LENGTH
                        Maximum number of words in generated article

  --perplexity, -p      Compute perplexity on test data

  --generate, -g        generate article

  --testDir TESTDIR, -T TESTDIR
                        directory for test data
