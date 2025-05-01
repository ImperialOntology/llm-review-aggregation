import nltk


def main():
    # setup nltk
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    nltk.download('stopwords')


if __name__ == "__main__":
    main()
