import math, collections


class SmoothUnigramModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        
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
            for datum in sentence.data:
                word = datum.word
                
                self.unigramCounts[word] += 1
                self.total += 1
                
    
        self.vocab_size = len(self.unigramCounts)
        #pass

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here
        score = 0.0
        
        for word in sentence: 
            
            if word not in self.unigramCounts:
                word = "UNK"
                
            count = self.unigramCounts[word]
            score += math.log(count + 1)
            score -= math.log(self.total + self.vocab_size)
            
           
        
        return score
