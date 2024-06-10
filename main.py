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
    Fun Test to Visualize the differences of my tokenizers encodings compared to the GPT-4 tokenizer
    Obviously they don't produce the same tokens since they have been trained 
    wildly different training data but fun to visualize differences.

    """
    test_string = "welcome everyone!!!? (你好!) haha456 😎"

    # Importing the GPT-4 Tokenizer
    gpt4_tokenizer = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
    gpt4_enc = gpt4_tokenizer.encode(test_string)
    gpt4_dec = gpt4_tokenizer.decode(gpt4_enc) # get the same text back

    # Using my own Tokenizer
    my_tokenizer_enc = my_tokenizer.encode(test_string)
    my_tokenizer_dec = my_tokenizer.decode(my_tokenizer_enc)

    # Printing Results
    print("Results: \n----------------")
    print(f"Original String: {test_string}")
    print("----------------")
    print(f"GPT-4 Tokenizer: \nEncoding: {gpt4_enc}\nEncoding -> Decoding: {gpt4_dec}")
    print("----------------")
    print(f"My Tokenizer: \nEncoding: {my_tokenizer_enc}\nEncoding -> Decoding: {my_tokenizer_dec}")

    # Interesting to see that my tokenizer encodes the sentence with much more tokens than GPT-4
    # This is completely expected however as their vocabulary sizes and training sets differ a lot in size.


# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
