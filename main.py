from tokenizer import Tokenizer

def main():

    file_path = 'train_text.txt'

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        text = file.read()

    # Initializing and training the tokenizer, size of vocabulary is a hyperparameter, I chose 400 for example
    my_tokenizer = Tokenizer()
    my_tokenizer.train(text, 700)

    text2 = my_tokenizer.decode(my_tokenizer.encode(text))
    print(text2 == text)

    tokens = my_tokenizer.visualize_tokens("tennis")
    print(tokens)

    n = 20
    longest_tokens = my_tokenizer.get_longest_tokens(n)
    print(f"{n} longest tokens in our vocabulary: {longest_tokens}")


# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
