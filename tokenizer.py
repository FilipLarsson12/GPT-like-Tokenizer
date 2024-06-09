class Tokenizer:
    def __init__(self):
        self.vocab = {}

    def train(self, text, vocab_size, verbose=False):
        self.vocab = {bytes([idx]): idx for idx in range(256)}

        add_elms = vocab_size - 256

    
    def find_pairs(self, text):
        counts = {}
        for ch1, ch2 in zip(text, text[1:]):
            if (ch1, ch2) in counts:
                counts[(ch1, ch2)] += 1
            else:
                counts[(ch1, ch2)] = 1
        return counts

