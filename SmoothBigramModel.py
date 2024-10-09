import math, collections


class SmoothBigramModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.bigramCounts = collections.defaultdict(lambda: 0)
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.vocab_size = 0
        

        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        # TODO your code here
        # Tip: To get words from the corpus, try
        #    for sentence in corpus.corpus:
        #       for datum in sentence.data:
        #         word = datum.word
        
        self.unigramCounts["UNK"] = 0
        for sentence in corpus.corpus:
            prev = None
            
            for datum in sentence.data:
                word = datum.word
                self.unigramCounts[word] += 1
                self.total += 1
            
                if prev is not None:
                    bigram = (prev, word)
                    self.bigramCounts[bigram] += 1
                
                prev = word
                
                
        self.vocab_size = len(self.unigramCounts)


    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here
        score = 0.0
        prev = None
        
        for word in sentence:
            if word not in self.unigramCounts:
                word = "UNK"
                
            unigram_count = self.unigramCounts[word]
            smoothed_unigram_prob = (unigram_count + 1) / (self.total + self.vocab_size)
            score += math.log(smoothed_unigram_prob)
                
            if prev is not None:
                
                if prev not in self.unigramCounts:
                    prev = "UNK"
                    
                bigram = (prev, word)
                bigram_count = self.bigramCounts[bigram]
                
                unigram_count = self.unigramCounts[prev]
                
                smoothed_bigram_prob = (bigram_count + 1) / (unigram_count + self.vocab_size)
                score += math.log(smoothed_bigram_prob)
                
                
            
            prev = word
        
        return score
