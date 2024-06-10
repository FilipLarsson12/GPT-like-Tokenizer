from tokenizer import Tokenizer
import tiktoken

def main():

    file_path = 'train_text.txt'

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        text = file.read()

    # Initializing and training the tokenizer, size of vocabulary is a hyperparameter, I chose 400 for example
    my_tokenizer = Tokenizer()
    my_tokenizer.train(text, 500)

    text2 = my_tokenizer.decode(my_tokenizer.encode(text))
    print(text2 == text)

    tokens = my_tokenizer.visualize_tokens("I like to play tennis")
    print(tokens)

    n = 20
    longest_tokens = my_tokenizer.get_longest_tokens(n)
    print(f"{n} longest tokens in our vocabulary: {longest_tokens}")

    """"
    Test to see if my tokenizer produces identical Encoding + Decoding as the GPT-4 tokenizer
    Obviously they don't produce the same tokens since they have been trained 
    wildly different training data but fun to visualize differences.
    """
    test_string = "hello world!!!? (안녕하세요!) lol123 😉"

    # Importing the GPT-4 Tokenizer
    gpt4_tokenizer = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
    gpt4_enc = gpt4_tokenizer.encode(test_string)
    gpt4_dec = gpt4_tokenizer.decode(gpt4_enc) # get the same text back

    # Using my own Tokenizer
    my_tokenizer.recover_gpt4_merges_and_vocab()
    my_tokenizer_enc = my_tokenizer.encode(test_string)
    my_tokenizer_dec = my_tokenizer.decode(my_tokenizer_enc)

    # Printing Results
    print("Results: \n----------------")
    print(f"GPT-4 Tokenizer: \nOriginal String: {test_string}\nEncoding: {gpt4_enc}\nEncoding -> Decoding: {gpt4_dec}")
    print("- - - - - - - - - - - - - - - -")
    print(f"My Tokenizer: \nOriginal String: {test_string}\nEncoding: {my_tokenizer_enc}\nEncoding -> Decoding: {my_tokenizer_dec}")



# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
