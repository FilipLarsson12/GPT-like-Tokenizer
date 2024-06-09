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

        # Building out our Vocabulary
        # Base Vocab
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # + All new ids based on the most frequently occuring pairs in the text
        for (id1, id2), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[id1] + self.vocab[id2]
        
    # Takes in a text and returns the encoding by the tokenizer
    def encode(self, text):
        # Encode the text with utf-8 to start
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        # Find all pairs in the tokens
        while True and len(tokens) >= 2:
            pairs = list(k for k, v in self.count_pairs(tokens).items())
            # Replace the pairs in tokens that has been reassigned to new ids in merges dict.
            # Start with the pairs that appear at the start of merges and move downwards
            best_pair_to_replace = None
            lowest_value_in_merges = 256+len(self.merges.items())
            for pair in pairs:
                if pair in self.merges and self.merges[pair] < lowest_value_in_merges:
                    best_pair_to_replace = pair
                    lowest_value_in_merges = self.merges[pair]

            if best_pair_to_replace:
                # Replace the pair with its new token_id
                tokens = self.merge(tokens, best_pair_to_replace, lowest_value_in_merges)
            else:
                # If no pair in our tokens exist in merges then we cant merge pairs anymore and we return the encoded text
                return tokens
        # We come here if we have only a single token with no possible pairs
        return tokens
    

    # Takes the encoding from the tokenizer and returns the text
    def decode(self, tokens):
        tokens = b"".join(self.vocab[id] for id in tokens)
        text = tokens.decode("utf-8", errors="replace")
        return text


    # return the n longest tokens in our vocabulary to get a sense of the longest tokens that we created during training
    def get_longest_tokens(self, n):
        longest_tokens = []
        for i in range(n):
            longest_tokens.append(self.vocab[len(self.vocab.items())-n+i])
        return longest_tokens
    
    # take in string and return a new text but with * signs between the tokens created by the tokenizer
    def visualize_tokens(self, string):
        tokens = self.encode(string)
        
