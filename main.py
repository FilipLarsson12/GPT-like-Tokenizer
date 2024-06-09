from tokenizer import Tokenizer

def main():

    file_path = 'train_text.txt'

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        text = file.read()

    my_tokenizer = Tokenizer()
    my_tokenizer.train(text, 276)
    string = "hello world"
    encoding = my_tokenizer.encode(string)
    print(f"Encoding: {encoding} of string: {my_tokenizer.encode_text(string)}")
    print(my_tokenizer.decode(my_tokenizer.encode(string)))

# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
