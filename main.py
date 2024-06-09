from tokenizer import Tokenizer

def main():

    my_tokenizer = Tokenizer()
    pairs = my_tokenizer.find_pairs("aaaaaaffggggeeeeeeeeee")
    print(pairs)
    # HALLOOOOOOOOOOOOO

# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()
