from anytree import PostOrderIter
from functools import reduce
from src.argumentation.arg_framework.product import Product
from src.argumentation.arg_framework.review import Review
from src.argumentation.arg_framework.argument import Argument
import re


class Framework:
    """
    A framework for analyzing arguments, extracting votes, and calculating argument strengths
    based on reviews and product features. It uses a BERT-based sentiment analyzer to determine
    the polarity of arguments and applies gradual semantics to compute argument strengths.

    Attributes:
        bert_analyzer: The BERT-based sentiment analyzer.
        product_id: The ID of the product being analyzed.
        product: The product object containing argument and feature nodes.
        product_node: The root node of the ontology tree correspondign to the product.
        arguments: The argument nodes of the product.
        features: The feature nodes of the product.
        qbaf: The Quantitative Bipolar Argumentation Framework (QBAF) analysis results.
        argument_polarities: The polarities of arguments (positive or negative).
        strengths: The computed strengths of arguments.
    """

    HIGH_SENTIMENT_THRESHOLD = 0.99

    def __init__(self, product: Product, product_id, review_df, bert_analyser, method=None):
        """
        Initialize the Framework object.

        Args:
            product (Product): The product object containing argument and feature nodes.
            product_id: The ID of the product being analyzed.
            review_df: A DataFrame containing reviews for the product.
                With columns 'id' for product ids and 'content' for review texts.
            bert_analyser: The BERT-based sentiment analyzer.
        """
        self.bert_analyzer = bert_analyser
        self.product_id = product_id
        self.product = product
        self.product_node = self.product.root
        self.arguments = self.product.argument_nodes
        self.features = self.product.feature_nodes

        # get reviews
        review_df = review_df
        reviews = [Review(row['id'], row['content'], self.product)
                   for _, row in review_df.iterrows()]

        # extract augmented votes
        self.extract_votes(reviews)

        # get aggregates
        ra, self.vote_sum, self.vote_phrases = self.get_aggregates(reviews)

        # get qbaf from ra
        self.qbaf, self.argument_polarities = self.get_qbaf(ra, len(reviews))

        # apply gradual semantics
        self.strengths = self.get_strengths(self.qbaf)

    def print(self):
        """
        Print the QBAF structure, argument strengths, polarities, and votes.
        """
        print('qbaf:')
        print(self.qbaf)
        for argument in self.arguments:
            print(argument.name)
            print('  strength:', self.strengths[argument])
            print(
                '  polarity:', 'positive' if self.argument_polarities[argument] else 'negative')
            print('  votes:')
            print('    direct: {} positive, {} negative'.format(len(self.supporting_phrases(argument)),
                                                                len(self.attacking_phrases(argument))))
            print('    augmented sum: {}'.format(self.vote_sum[argument]))

    def get_bert_sentiments(self, data):
        """
        Get sentiment polarities for a batch of data using the BERT analyzer.

        Args:
            data: A list of data items to analyze.

        Returns:
            List of sentiment polarities.
        """
        return list(self.bert_analyzer.get_batch_sentiment_polarity(data))

    def extract_votes(self, reviews):
        """
        Extract votes (sentiments) for arguments from reviews.

        Args:
            reviews: A list of Review objects.
        """
        labelled_phrases = [(phrase.text, arg.start, arg.end) for review in reviews for phrase in review.phrases for arg
                            in phrase.args]

        sentiments = self.get_bert_sentiments(labelled_phrases)

        for review in reviews:
            for phrase in review.phrases:
                for arg in phrase.args:
                    sentiment = sentiments.pop(0)
                    arg.set_sentiment(sentiment)

    def get_aggregates(self, reviews):
        """
        Aggregate votes and phrases for arguments from reviews.

        Args:
            reviews: A list of Review objects.

        Returns:
            Tuple containing:
                - ra: A list of aggregated votes.
                - vote_sum: A dictionary mapping arguments to their vote sums.
                - vote_phrases: A dictionary mapping arguments to their associated phrases.
        """
        ra = []
        vote_sum = {arg: 0 for arg in self.arguments}
        vote_phrases = {arg: [] for arg in self.arguments}
        for review in reviews:
            for phrase in review.phrases:
                for arg, sentiment in phrase.get_votes().items():
                    # {'phrase': phrase.text, 'sentiment': sentiment, 'n_args': len(phrase.args)}
                    vote_phrases[arg].append(phrase)
            for arg, sentiment in review.get_votes().items():
                ra.append({'review_id': review.id,
                          'argument': arg, 'vote': sentiment})
                vote_sum[arg] += sentiment
        return ra, vote_sum, vote_phrases

    def get_qbaf(self, ra, review_count):
        """
        Construct the Quantitative Bipolar Argumentation Framework (QBAF) from aggregated votes.

        Args:
            ra: A list of aggregated votes.
            review_count: The total number of reviews.

        Returns:
            Tuple containing:
                - qbaf: The QBAF structure.
                - argument_polarities: A dictionary mapping arguments to their polarities.
        """
        # sums of all positive and negative votes for arguments
        argument_sums = {}
        for argument in self.arguments:
            argument_sums[argument] = 0
            for r in ra:
                if r['argument'] == argument:
                    argument_sums[argument] += r['vote']

        # calculate attack/support relations
        argument_polarities = {}
        supporters = {r: [] for r in self.arguments}
        attackers = {r: [] for r in self.arguments}
        for r in self.arguments:
            argument_polarities[r] = argument_sums[r] >= 0
            for subf in r.children:
                if (argument_sums[r] >= 0 and argument_sums[subf] >= 0) or (argument_sums[r] < 0 and argument_sums[subf] < 0):
                    supporters[r].append(subf)
                elif (argument_sums[r] >= 0 and argument_sums[subf] < 0) or (argument_sums[r] < 0 and argument_sums[subf] >= 0):
                    attackers[r].append(subf)

        # calculate base scores for arguments
        base_strengths = {self.product_node: 0.5 + 0.5 *
                          argument_sums[self.product_node] / review_count}
        for feature in self.features:
            base_strengths[feature] = abs(
                argument_sums[feature]) / review_count

        qbaf = {'supporters': supporters, 'attackers': attackers,
                'base_strengths': base_strengths}
        return qbaf, argument_polarities

    @staticmethod
    def combined_strength(args):
        """
        Calculate the combined strength of a list of arguments.

        Args:
            args: A list of argument strengths.

        Returns:
            The combined strength derived from strengths of provided arguments.
        """
        if len(args) != 0:
            return 1 - reduce(lambda x, y: x * y, map(lambda v: 1 - v, args))
        return 0

    @staticmethod
    def argument_strength(base_score, attacker_strengths, supporter_strengths):
        """
        Calculate the strength of an argument based on its base score, combined attacker strengths,
        and combined supporter strengths.

        Args:
            base_score: The base score of the argument.
            attacker_strengths: A list of strengths of attacking arguments.
            supporter_strengths: A list of strengths of supporting arguments.

        Returns:
            The calculated argument strength.
        """
        attack = Framework.combined_strength(attacker_strengths)
        support = Framework.combined_strength(supporter_strengths)
        if attack > support:
            return base_score - (base_score * abs(attack - support))
        elif attack < support:
            return base_score + ((1 - base_score) * abs(attack - support))
        return base_score

    def get_strengths(self, qbaf):
        """
        Apply DF-QUAD gradual semantics to calculate argument strengths.

        Args:
            qbaf: The QBAF structure.

        Returns:
            A dictionary mapping arguments to their augmented strengths.
        """
        strengths = {}
        arguments = [node for node in PostOrderIter(self.product_node)]
        for argument in arguments:
            attacker_strengths = []
            supporter_strengths = []
            for child in argument.children:
                if child in qbaf['attackers'][argument]:
                    attacker_strengths.append(strengths[child])
                elif child in qbaf['supporters'][argument]:
                    supporter_strengths.append(strengths[child])
            strengths[argument] = Framework.argument_strength(qbaf['base_strengths'][argument], attacker_strengths,
                                                              supporter_strengths)
        return strengths

    def get_strongest_supporting_subfeature(self, argument):
        """
        Get the strongest supporting subfeature for an argument.

        Args:
            argument: The argument to analyze.

        Returns:
            The strongest supporting subfeature, or None if no supporters exist.
        """
        supporters = self.qbaf['supporters'][argument]
        if len(supporters) == 0:
            return None
        supporter_strengths = {s: self.strengths[s] for s in supporters}
        return max(supporter_strengths, key=supporter_strengths.get)

    def get_strongest_attacking_subfeature(self, argument):
        """
        Get the strongest attacking subfeature for an argument.

        Args:
            argument: The argument to analyze.

        Returns:
            The strongest attacking subfeature, or None if no attackers exist.
        """
        attackers = self.qbaf['attackers'][argument]
        if len(attackers) == 0:
            return None
        attacker_strengths = {a: self.strengths[a] for a in attackers}
        return max(attacker_strengths, key=attacker_strengths.get)

    def liked_argument(self, argument):
        """
        Determine if an argument is liked based on its polarity.

        Args:
            argument: The argument to analyze.

        Returns:
            True if the argument is liked (positive polarity), False otherwise.
        """
        return self.argument_polarities[argument]

    def supported_argument(self, argument):
        """
        Determine if an argument is supported by its strongest supporting subfeature.

        Args:
            argument: The argument to analyze.

        Returns:
            True if the argument is supported, False otherwise.
        """
        supp = self.get_strongest_supporting_subfeature(argument)
        return supp is not None and self.strengths[supp] > 0

    def attacked_argument(self, argument):
        """
        Determine if an argument is attacked by its strongest attacking subfeature.

        Args:
            argument: The argument to analyze.

        Returns:
            True if the argument is attacked, False otherwise.
        """
        att = self.get_strongest_attacking_subfeature(argument)
        return att is not None and self.strengths[att] > 0

    def best_supporting_phrase(self, argument):
        """
        Get the best supporting phrase for an argument.

        Args:
            argument: The argument to analyze.

        Returns:
            The best supporting phrase, or None if no suitable phrases exist.
        """
        phrases = [phrase for phrase in self.supporting_phrases(argument)
                   if phrase.n_args() == 1 and Framework.is_well_formatted(phrase.text)]
        if len(phrases) == 0:
            return None
        top_5 = list(
            sorted(phrases, key=lambda p: p.get_vote(argument), reverse=True))[:5]
        return max(top_5, key=lambda p: len(p.text))

    def best_attacking_phrase(self, argument):
        """
        Get the best attacking phrase for an argument.

        Args:
            argument: The argument to analyze.

        Returns:
            The best attacking phrase, or None if no suitable phrases exist.
        """
        phrases = [phrase for phrase in self.attacking_phrases(argument)
                   if phrase.n_args() == 1 and Framework.is_well_formatted(phrase.text)]
        if len(phrases) == 0:
            return None
        top_5 = list(sorted(phrases, key=lambda p: p.get_vote(argument)))[:5]
        return max(top_5, key=lambda p: len(p.text))

    @staticmethod
    def is_well_formatted(phrase):
        """
        Check if a phrase is well-formatted. E.g. contains only alphanumeric characters
        and punctuation.

        Args:
            phrase: The phrase to check.

        Returns:
            True if the phrase is well-formatted, False otherwise.
        """
        return re.match('^[-a-zA-Z0-9();,./!?\'" ]*$', phrase)

    def supporting_phrases(self, argument):
        """
        Get all supporting phrases for an argument.

        Args:
            argument: The argument to analyze.

        Returns:
            A list of supporting phrases.
        """
        return list(filter(lambda phrase: phrase.get_vote(argument) > 0, self.vote_phrases[argument]))

    def attacking_phrases(self, argument):
        """
        Get all attacking phrases for an argument.

        Args:
            argument: The argument to analyze.

        Returns:
            A list of attacking phrases.
        """
        return list(filter(lambda phrase: phrase.get_vote(argument) < 0, self.vote_phrases[argument]))

    def get_argument_graph(self):
        """
        Generate the argument graph for the product for plotting purposes.

        Returns:
            The root Argument object representing the argument graph.
        """
        return self.create_arg(self.product_node, 120)

    def create_arg(self, arg_node, size):
        """
        Recursively create an Argument object for the argument graph for plotting.

        Args:
            arg_node: The current argument node.
            size: The size of the argument node.

        Returns:
            The created Argument object.
        """
        supporters = [self.create_arg(supp_node, size - 20)
                      for supp_node in self.qbaf['supporters'][arg_node]]
        attackers = [self.create_arg(att_node, size - 20)
                     for att_node in self.qbaf['attackers'][arg_node]]
        phrase = self.best_supporting_phrase(
            arg_node) if self.argument_polarities[arg_node] else self.best_attacking_phrase(arg_node)
        return Argument(arg_node.name, self.argument_polarities[arg_node], supporters, attackers, phrase, size)

    def get_product_strength_percentage(self):
        """
        Get the product's strength as a percentage as the strength of the root of
        the corresponding ontology tree.

        Returns:
            The product's strength percentage.
        """
        return self.strengths[self.product_node] * 100
