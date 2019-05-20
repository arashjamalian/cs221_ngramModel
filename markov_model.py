from util import CONST_START_WORD
from util import weightSample


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

        sampleWord = weightSample(cDict)
        # construct next state
        cState = [word for word in cState[1:]]
        cState.append(sampleWord)
        cState = tuple(cState)
        return cState


