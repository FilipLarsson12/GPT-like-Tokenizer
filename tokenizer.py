import regex as re
import tiktoken

class Tokenizer:
    def __init__(self):
        # Vocabulary
        self.vocab = {}
        # Keep track of which tokens merge into which other tokens
        self.merges = {}
        # Identical Regex pattern that GPT-4 uses to separate the training text to counteract multiple tokens for the same word
        # For example we dont want separate tokens for: 'hello.', 'hello!' and 'hello?'. We probably just want: 'hello'.
        self.regex_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

        # These can be ignored just 

    # Simple helper function to encode the text
    def basic_encode(self, text):
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        return tokens

    # Helper function to merge two dictionaries
    def merge_dictionaries(self, dict1, dict2):
        merged_dict = dict1.copy()  

        for key, value in dict2.items():
            if key in merged_dict:
                merged_dict[key] += value 
            else:
                merged_dict[key] = value 

        return merged_dict

    # Find all the character-pairs in a text and return them in a dict
    def count_pairs(self, text):
        counts = {}
        for ch1, ch2 in zip(text, text[1:]):
            if (ch1, ch2) in counts:
                counts[(ch1, ch2)] += 1
            else:
                counts[(ch1, ch2)] = 1

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
        splitted_text = re.findall(self.regex_pattern, text)
        text = self.basic_encode(text)
        splitted_text = [self.basic_encode(part) for part in splitted_text]
        print(splitted_text)
        add_elms = vocab_size - 256

        for i in range(add_elms):
            overall_counts = {}
            for part in splitted_text:
                
                pair_counts = self.count_pairs(part)
                
                overall_counts = self.merge_dictionaries(overall_counts, pair_counts)
            # Sort the pair counts to be descending (most frequently-occuring pair at position 0)
            overall_counts = dict(sorted(overall_counts.items(), key=lambda item: item[1], reverse=True))

            if len(overall_counts.items()) > 0:
                pair_to_replace = list(overall_counts.items())[0][0]
                print(f"Replacing all occurences of: {pair_to_replace} with {256+i}")
                splitted_text = [self.merge(part, pair_to_replace, 256+i) for part in splitted_text]

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
        longest_tokens = [token.decode('utf-8') for token in longest_tokens]
        longest_tokens.reverse()
        return longest_tokens
    
    # take in string and return a new text but with * signs between the tokens created by the tokenizer
    def visualize_tokens(self, string):
        tokens = self.encode(string)
        tokens = [self.vocab[token] for token in tokens]
        tokens = b"*".join(tokens)
        tokens = tokens.decode('utf-8')
        return tokens


    # These functions will be used just to test the Tokenizer against the real GPT-4 tokenizer. 
    # The GPT-4 Merges dictionary can only be obtained by: 
    # Either training the Tokenizer on the identical training data as GPT-4 Tokenizer
    # Or simply copying the complete merges dict directly from OpenAI githu
    # The merges is necessary to be able to produce identical tokens as GPT-4 Tokenizer.

    # Takes a merged token and return the pair inside it that got merged first
    def swap_positions(self, mergeable_ranks):
        for i in range(256):
            # Convert i to bytes
            i_bytes = bytes([i])
            
            # Check if i_bytes is a key in mergeable_ranks
            if i_bytes in mergeable_ranks:
                # Get the rank corresponding to i_bytes
                rank = mergeable_ranks[i_bytes]
                
                # Find the bytes key for the rank value
                rank_bytes = bytes([rank])
                
                # Swap the positions if rank_bytes is also a key in mergeable_ranks
                if rank_bytes in mergeable_ranks:
                    # Swap the values
                    mergeable_ranks[i_bytes], mergeable_ranks[rank_bytes] = mergeable_ranks[rank_bytes], mergeable_ranks[i_bytes]
        
        return mergeable_ranks

    def find_first_merge(self, mergeable_ranks, token, max_rank):
        parts = [bytes([b]) for b in token]
        while True:
            min_id = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_id = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_id is not None
            parts = parts[:min_id] + [parts[min_id] + parts[min_id + 1]] + parts[min_id + 2:]
        return parts
    

    def recover_gpt4_merges_and_vocab(self):
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = {}
        for token, rank in mergeable_ranks.items():
            if len(token) == 1:
                continue
            pair = tuple(self.find_first_merge(mergeable_ranks, token, rank))
            assert len(pair) == 2
            ix0 = mergeable_ranks[pair[0]]
            ix1 = mergeable_ranks[pair[1]]            
            self.merges[(ix0, ix1)] = rank

        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (ix0, ix1), rank in self.merges.items():
            vocab[rank] = vocab[ix0] + vocab[ix1]
        self.vocab = vocab

        mergeable_ranks = self.swap_positions(mergeable_ranks)
