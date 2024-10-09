import math
from Datum import Datum
from Sentence import Sentence
from Corpus import Corpus
from UniformModel import UniformModel
from UnigramModel import UnigramModel
from BackoffModel import BackoffModel
from SmoothUnigramModel import SmoothUnigramModel
from SmoothBigramModel import SmoothBigramModel
from CustomModel import CustomModel
from EditModel import EditModel
from SpellingResult import SpellingResult
import types
import re, collections


class SpellCorrect:
    """Spelling corrector for sentences. Holds edit model, language model and the corpus."""

    def __init__(self, lm, corpus):
        self.languageModel = lm
        self.editModel = EditModel('data/count_1edit.txt', corpus)
        all_misspells = [x.strip(" ").strip("\n").split("->") for x in open("data/misspellings.txt").readlines()]
        self.MISSPELLS = {}
        for corr_og in all_misspells:
            self.MISSPELLS.update({corr_og[0]: corr_og[1].split(", ")})

    def correctSentence(self, sentence, use_misspell_dict=False):
        """Assuming exactly one error per sentence, returns the most probable corrected sentence.
           Sentence is a list of words."""

        if len(sentence) == 0:
            return []

        bestSentence = sentence[:]  # copy of sentence
        bestScore = float('-inf')

        for i in range(1, len(sentence) - 1):  # ignore <s> and </s>
            # TODO: select the maximum probability sentence here, according to the noisy channel model.
            # Tip: self.editModel.editProbabilities(word) gives edits and log-probabilities according to your edit model.
            #      You should iterate through these values instead of enumerating all edits.
            # Tip: self.languageModel.score(trialSentence) gives log-probability of a sentence
            # Tip: self.MISSPELLS contains common misspellings in lower case, and their corresponding corrections in a list
            
            
            word = sentence[i]
            candidates = [(word, 0)]
            
            candidates += self.editModel.editProbabilities(word)
            
            if use_misspell_dict and word.lower() in self.MISSPELLS:
                for correction in self.MISSPELLS[word.lower()]:
                    candidates.append((correction, 5))
        
            
            for candidate_word, edit_log_prob in candidates:
                trialSentence = sentence[:i] + [candidate_word] + sentence[i + 1:]
                languageModel_log_prob = self.languageModel.score(trialSentence)
                total_log_prob = languageModel_log_prob + edit_log_prob
                if total_log_prob > bestScore:
                    bestScore = total_log_prob
                    bestSentence = trialSentence
            
            
            '''if use_misspell_dict:
                pass
            pass'''

        return bestSentence

    def evaluate(self, corpus, use_misspell_dict=False):
        """Tests this speller on a corpus, returns a SpellingResult"""
        numCorrect = 0
        numTotal = 0
        testData = corpus.generateTestCases()
        for sentence in testData:
            if sentence.isEmpty():
                continue
            errorSentence = sentence.getErrorSentence()
            hypothesis = self.correctSentence(errorSentence, use_misspell_dict)
            if sentence.isCorrection(hypothesis):
                numCorrect += 1
            numTotal += 1
        return SpellingResult(numCorrect, numTotal)

    def correctCorpus(self, corpus):
        """Corrects a whole corpus, returns a JSON representation of the output."""
        string_list = []  # we will join these with commas,  bookended with []
        sentences = corpus.corpus
        for sentence in sentences:
            uncorrected = sentence.getErrorSentence()
            corrected = self.correctSentence(uncorrected)
            word_list = '["%s"]' % '","'.join(corrected)
            string_list.append(word_list)
        output = '[%s]' % ','.join(string_list)
        return output


def main():
    """Trains all of the language models and tests them on the dev data. Change devPath if you
       wish to do things like test on the training data."""

    trainPath = 'data/tagged-train.dat'
    trainingCorpus = Corpus(trainPath)

    devPath = 'data/tagged-dev.dat'
    devCorpus = Corpus(devPath)
    

    
    print('Unigram Language Model: ')
    unigramLM = UnigramModel(trainingCorpus)
    unigramSpell = SpellCorrect(unigramLM, trainingCorpus)
    unigramOutcome = unigramSpell.evaluate(devCorpus)
    print(str(unigramOutcome))

    print('Uniform Language Model: ')
    uniformLM = UniformModel(trainingCorpus)
    uniformSpell = SpellCorrect(uniformLM, trainingCorpus)
    uniformOutcome = uniformSpell.evaluate(devCorpus)
    print(str(uniformOutcome)) 
    
   
    print('Smooth Unigram Language Model: ')
    smoothUnigramLM = SmoothUnigramModel(trainingCorpus)
    smoothUnigramSpell = SpellCorrect(smoothUnigramLM, trainingCorpus)
    smoothUnigramOutcome = smoothUnigramSpell.evaluate(devCorpus)
    print(str(smoothUnigramOutcome))
    smoothUnigramOutcome = smoothUnigramSpell.evaluate(devCorpus, use_misspell_dict=True)
    print(str(smoothUnigramOutcome))

    print('Smooth Bigram Language Model: ')
    smoothBigramLM = SmoothBigramModel(trainingCorpus)
    smoothBigramSpell = SpellCorrect(smoothBigramLM, trainingCorpus)
    smoothBigramOutcome = smoothBigramSpell.evaluate(devCorpus)
    print(str(smoothBigramOutcome))
    smoothBigramOutcome = smoothBigramSpell.evaluate(devCorpus, use_misspell_dict=True)
    print(str(smoothBigramOutcome))

    print('Backoff Language Model: ')
    backoffLM = BackoffModel(trainingCorpus)
    backoffSpell = SpellCorrect(backoffLM, trainingCorpus)
    backoffOutcome = backoffSpell.evaluate(devCorpus)
    print(str(backoffOutcome))
    backoffOutcome = backoffSpell.evaluate(devCorpus, use_misspell_dict=True)
    print(str(backoffOutcome))
    

    print('Custom Language Model: ')
    customLM = CustomModel(trainingCorpus)
    customSpell = SpellCorrect(customLM, trainingCorpus)
    customOutcome = customSpell.evaluate(devCorpus)
    print(str(customOutcome))
    

if __name__ == "__main__":
    main()
