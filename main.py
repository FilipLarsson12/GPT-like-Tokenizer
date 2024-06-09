from tokenizer import Tokenizer

def main():

    file_path = 'train_text.txt'

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        text = file.read()

    # Initializing and training the tokenizer, size of vocabulary is a hyperparameter, I chose 400 for example
    my_tokenizer = Tokenizer()
    my_tokenizer.train(text, 300)


# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
