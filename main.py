from tokenizer import Tokenizer

def main():

    my_tokenizer = Tokenizer()
    my_tokenizer.train("abc", 3)
    pairs = my_tokenizer.find_pairs("abcdefg")
    print(pairs)
    
# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
