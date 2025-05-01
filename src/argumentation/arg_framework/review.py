import re
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
from src.argumentation.sentiment.bert_dataset import MAX_SEQ_LEN
from anytree import PostOrderIter
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()


class Review:
    """
    Represents a review for a product, containing phrases and votes for arguments.

    Attributes:
        SENTIMENT_THRESHOLD (float): Threshold for determining strong sentiments.
        PHRASE_MAX_WORDS (float): Maximum number of words allowed in a phrase.
        product: The product associated with the review.
        id: The ID of the review.
        body: The text content of the review.
        phrases: A list of extracted phrases from the review.
        votes: Votes based on sentiments extracted from the review phrases.
    """

    SENTIMENT_THRESHOLD = 0.95
    PHRASE_MAX_WORDS = MAX_SEQ_LEN * 0.3

    def __init__(self, review_id, review_body, product):
        """
        Initialize a Review object.

        Args:
            review_id (int): The ID of the review.
            review_body (str): The text content of the review.
            product: The product associated with the review.
        """
        self.product = product
        self.id = review_id
        self.body = review_body
        self.phrases = Review.extract_phrases(self.body, product)
        self.votes = {}

    @staticmethod
    def extract_phrases(review_body, product):
        """
        Extract phrases from the review body.
        Phrases are extracted by splitting sentences using specific conjunctions
        and filtering based on the maximum word count.
        The extracted phrases are then converted into Phrase objects.

        Args:
            review_body (str): The text content of the review.
            product: The product associated with the review.

        Returns:
            list: A list of Phrase objects extracted from the review.
        """
        sentences = sent_tokenize(review_body)
        texts = []
        for sentence in sentences:
            texts += re.split(' but | although | though | otherwise | however | unless | whereas | despite |<br />',
                              sentence)
        texts = filter(lambda t: len(t.split()) <
                       Review.PHRASE_MAX_WORDS, texts)
        phrases = [Phrase(text, product) for text in texts]
        return phrases

    def get_votes(self):
        """
        Retrieve and normalize votes for arguments.
        Votes are based on sentiment scores extracted from the review phrases.

        Returns:
            dict: A dictionary mapping arguments to their normalized votes.
        """
        for arg, sentiment in [(arg, sentiment) for phrase in self.phrases for arg, sentiment in phrase.votes.items()]:
            if arg not in self.votes or abs(sentiment) > abs(self.votes[arg]):
                self.votes[arg] = sentiment
        # normalize
        for arg in self.votes:
            self.votes[arg] = 1 if self.votes[arg] > 0 else -1
        self.augment_votes()
        return self.votes

    def augment_votes(self):
        """
        Total augment votes based on votes of its descendants in the 
        ontology tree.
        """
        arguments = [node for node in PostOrderIter(self.product.root)]
        for argument in arguments:
            if argument not in self.votes:
                polar_sum = 0
                for subfeat in argument.children:
                    if subfeat in self.votes:
                        polar_sum += self.votes[subfeat]
                if polar_sum != 0:
                    self.votes[argument] = 1 if polar_sum > 0 else -1

    def is_voting(self):
        """
        Check if the review contains any phrases with votes.

        Returns:
            bool: True if the review contains voting phrases, False otherwise.
        """
        return any(len(p.votes) > 0 for p in self.phrases)


class Phrase:
    """
    Represents a phrase extracted from a review, containing arguments/aspects and their sentiment votes.

    Attributes:
        product: The product associated with the phrase.
        text: The text content of the phrase.
        spans: A list of token spans in the phrase.
        tokens: A list of tokens in the phrase.
        args: A list of arguments matched to the phrase.
        votes: A dictionary mapping arguments to their votes.
    """

    def __init__(self, text, product):
        """
        Initialize a Phrase object.

        Args:
            text (str): The text content of the phrase.
            product: The product associated with the phrase.
        """
        self.product = product
        self.text = text
        self.spans = list(tokenizer.span_tokenize(text))
        self.tokens = [text[start:end] for start, end in self.spans]
        self.args = self.get_args()
        self.votes = {}

    def get_args(self):
        """
        Retrieve arguments that match the phrase.

        Returns:
            list: A list of Arg objects representing matched arguments.
        """
        argument_matches = []
        arguments = [node for node in PostOrderIter(self.product.root)]
        while len(arguments) > 0:
            arg = arguments.pop(0)
            for term in self.product.glossary[arg]:
                matches = [Arg(arg, ' '.join(term), start, end)
                           for start, end in Phrase.matching_subsequences(term, self.tokens)]
                if matches:
                    argument_matches += matches
                    self.remove_ancestors(arg, arguments)
                    break
        return argument_matches

    def remove_ancestors(self, node, l):
        """
        Remove all ancestors of a node from a list.

        Args:
            node: The node whose ancestors should be removed.
            l (list): The list to modify.
        """
        if node.parent is not None:
            try:
                l.remove(node.parent)
            except ValueError:
                pass
            self.remove_ancestors(node.parent, l)

    def num_args(self):
        """
        Get the number of arguments matched to the phrase.

        Returns:
            int: The number of matched arguments.
        """
        return len(self.args)

    def get_votes(self):
        """
        Retrieve votes for arguments from the phrase.

        Returns:
            dict: A dictionary mapping arguments to their votes.
        """
        for arg in self.args:
            if (abs(arg.sentiment) > Review.SENTIMENT_THRESHOLD and
                    (arg.node not in self.votes or abs(arg.sentiment) > abs(self.votes[arg.node]))):
                self.votes[arg.node] = arg.sentiment
        return self.votes

    def get_vote(self, node):
        """
        Get the vote for a specific argument node.

        Args:
            node: The argument node.

        Returns:
            int: The vote for the argument node.
        """
        return self.votes[node]

    def get_arg_mentions(self, node):
        """
        Get mentions of a specific argument node in the phrase.

        Args:
            node: The argument node.

        Returns:
            list: A list of tuples containing the argument form and its span.
        """
        mentions = []
        for arg in self.args:
            if arg.node == node:
                start, end = self.spans[arg.start][0], self.spans[arg.end - 1][1]
                mentions.append((arg.form, start, end))
        return mentions

    def n_args(self):
        """
        Get the number of arguments in the phrase.

        Returns:
            int: The number of arguments.
        """
        return len(self.args)

    @staticmethod
    def matching_subsequences(l_sub, l):
        """
        Find matching subsequences in a list.

        Args:
            l_sub (list): The subsequence to match.
            l (list): The list to search.

        Returns:
            list: A list of tuples representing the start and end indices of matches.
        """
        sub_idxs = []
        len_sub = len(l_sub)
        for i in range(len(l)):
            if l[i:i+len_sub] == l_sub:
                sub_idxs.append((i, i+len_sub))
        return sub_idxs


class Arg:
    """
    Represents an argument / ontology tree node / aspect matched in a phrase.

    Attributes:
        node: The argument node in the product's ontology.
        form: The textual form of the argument, i.e. the node name.
        start: The start index of the argument in the phrase.
        end: The end index of the argument in the phrase.
        sentiment: The sentiment associated with the argument.
    """

    def __init__(self, node, form, start, end):
        """
        Initialize an Arg object.

        Args:
            node: The argument node in the product's ontology.
            form (str): The textual form of the argument.
            start (int): The start index of the argument in the phrase.
            end (int): The end index of the argument in the phrase.
        """
        self.node = node
        self.form = form
        self.start = start
        self.end = end
        self.sentiment = None

    def set_sentiment(self, sentiment):
        """
        Set the sentiment for the argument.

        Args:
            sentiment (float): The sentiment value.
        """
        self.sentiment = sentiment
