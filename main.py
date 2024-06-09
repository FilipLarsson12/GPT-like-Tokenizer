from tokenizer import Tokenizer

def main():

    my_tokenizer = Tokenizer()
    my_tokenizer.train("hhhhhhh", 260)
    string = "hhhejj"
    encoding = my_tokenizer.encode(string)
    print(f"Encoding: {encoding} of string: {my_tokenizer.encode_text(string)}")
    # HALLOOOOOOOOOOOOO

# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
