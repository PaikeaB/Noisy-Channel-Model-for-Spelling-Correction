import math, collections

#trigram backoff model
class CustomModel:

    def __init__(self, corpus):
        """Initial custom language model and structures needed by this mode"""
        self.trigramCounts = collections.defaultdict(lambda: 0)
        self.bigramCounts = collections.defaultdict(lambda: 0)
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.vocab_size = 0
     
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
        """
        # TODO your code here
        self.unigramCounts["UNK"] = 0
        
        for sentence in corpus.corpus:
            
            prev1 = None
            prev2 = None
            
            for datum in sentence.data:
                word = datum.word
                
                self.unigramCounts[word] += 1
                self.total += 1
                
                if prev1 is not None:
                    bigram = (prev1, word)
                    self.bigramCounts[bigram] += 1
                
                if prev2 is not None and prev1 is not None:
                    trigram = (prev2, prev1, word)
                    self.trigramCounts[trigram] += 1
                    
                prev2 = prev1
                prev1 = word

        self.vocab_size = len(self.unigramCounts)
    
    def score(self, sentence):
        """ With list of strings, return the log-probability of the sentence with language model. Use
            information generated from train.
        """
        # TODO your code here
        
        score = 0.0
        prev2 = None
        prev1 = None
        backoff = 0.4
        
        for word in sentence:
            if word not in self.unigramCounts:
              
                word = "UNK"
            
            
            if prev2 is not None and prev1 is not None:
                trigram = (prev2, prev1, word)
                trigram_count = self.trigramCounts[trigram]
                bigram_count = self.bigramCounts[(prev2, prev1)]
                
                if trigram_count > 0:
                    smoothed_trigram_prob = trigram_count / bigram_count
                    score += math.log(smoothed_trigram_prob)
                
                else:
                    bigram = (prev1, word)
                    bigram_count = self.bigramCounts[bigram]
                    unigram_count = self.unigramCounts[prev1]
                    
                    if bigram_count > 0:
                        smoothed_bigram_prob = bigram_count / unigram_count
                        score += math.log(backoff * smoothed_bigram_prob)
                        
                    else:
                        unigram_count = self.unigramCounts[word]
                        smoothed_unigram_prob = (unigram_count + 1) / (self.total + self.vocab_size)
                        score += math.log(backoff**2 * smoothed_unigram_prob)
                
                
            elif prev1 is not None:
                bigram = (prev1, word)
                bigram_count = self.bigramCounts[bigram]
                unigram_count = self.unigramCounts[prev1]
                
                if bigram_count > 0:
                    smoothed_bigram_prob = bigram_count / unigram_count
                    score += math.log(backoff * smoothed_bigram_prob)
                    
                else:
                    unigram_count = self.unigramCounts[word]
                    smoothed_unigram_prob = (unigram_count + 1) / (self.total + self.vocab_size)
                    score += math.log(backoff * smoothed_unigram_prob)
                    
            else:
                unigram_count = self.unigramCounts[word]
                smoothed_unigram_prob = (unigram_count + 1) / (self.total + self.vocab_size)
                score += math.log(smoothed_unigram_prob)
                
                
            prev2 = prev1
            prev1 = word
            
        return score
        

