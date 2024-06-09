class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

    # Simple helper function to encode the text
    def encode_text(self, text):
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        return tokens

    # Find all the character-pairs in a text and return them in a dict
    def find_pairs(self, text):
        counts = {}
        for ch1, ch2 in zip(text, text[1:]):
            if (ch1, ch2) in counts:
                counts[(ch1, ch2)] += 1
            else:
                counts[(ch1, ch2)] = 1

        # Sort the pair counts to be descending (most frequently-occuring pair at position 0)
        counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        return counts


    def train(self, text, vocab_size, verbose=False):
        text = self.encode_text(text)

        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        add_elms = vocab_size - 256
        pair_counts = self.find_pairs(text)
        print(pair_counts)
        pairs_to_add = list(pair_counts.items())[:add_elms]
        

