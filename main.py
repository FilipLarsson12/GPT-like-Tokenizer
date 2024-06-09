from tokenizer import Tokenizer

def main():

    file_path = 'train_text.txt'

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        text = file.read()

    # Initializing and training the tokenizer
    my_tokenizer = Tokenizer()
    my_tokenizer.train(text, 280)

    # Performing some tests
    string = "hello world"
    encoding = my_tokenizer.encode(string)

    print(f"Encoding: {encoding} of string: {my_tokenizer.encode_text(string)}")

    print(my_tokenizer.decode(my_tokenizer.encode(string)))

    text2 = my_tokenizer.decode(my_tokenizer.encode(text))
    print(text == text2)

    ten_longest_tokens = my_tokenizer.get_longest_tokens(10)
    print(ten_longest_tokens)

# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
