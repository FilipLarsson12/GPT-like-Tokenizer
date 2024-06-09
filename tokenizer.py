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
    def count_pairs(self, text):
        counts = {}
        for ch1, ch2 in zip(text, text[1:]):
            if (ch1, ch2) in counts:
                counts[(ch1, ch2)] += 1
            else:
                counts[(ch1, ch2)] = 1

        # Sort the pair counts to be descending (most frequently-occuring pair at position 0)
        counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        return counts
    
    def merge(self, text, pair, new_id):
        i = 0
        while i < (len(text)-1):
            current_pair = (text[i], text[i+1])
            if current_pair == pair:
                text.pop(i)
                text.pop(i)
                text.insert(i, new_id)
                self.merges[pair] = new_id
            i += 1
        return text


    def train(self, text, vocab_size, verbose=False):
        text = self.encode_text(text)

        add_elms = vocab_size - 256

        print("Text before merges: ", text)
        for i in range(add_elms):
            pair_counts = self.count_pairs(text)
            print(pair_counts)
            pair_to_replace = list(pair_counts.items())[0][0]

            print(f"Replacing all occurences of: {pair_to_replace} with {256+i}")

            text = self.merge(text, pair_to_replace, 256+i)

        print("Text after merges: ", text)
        print("Merges: ", self.merges)
