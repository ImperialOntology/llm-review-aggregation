from nltk.stem import WordNetLemmatizer
from os.path import isfile
import pickle
import os
import sys
# Add your desired parent directories here
parent_directories = ['..', '../..']
_ = [sys.path.insert(1, os.path.join(root, d))
     for root, dirs, _ in os.walk(os.getcwd())
     for d in dirs + parent_directories]


wnl = WordNetLemmatizer()


class Product:
    """
    Represents a product with its ontology tree, including feature nodes, and a glossary of synonyms.

    Attributes:
        root: The root node of the product's ontology tree.
        feature_nodes: A list of feature nodes derived from the root's descendants.
        argument_nodes: A list of all nodes represented as arguments.
        glossary: A dictionary mapping ontology nodes to their synonym aspects.
        singularities: A dictionary indicating whether each argument node's name is singular 
            i.e. stays same after lematization.
    """

    FILE_DIR = '/vol/bitbucket/yh2024/thesis/ada_project/ADA-X/server/agent/extracted_products/'
    FILE_EXTENSION = '.pkl'

    def __init__(self, root, syn_dict=None):
        """
        Initialize a Product object.

        Args:
            root: The root node of the product's ontology tree.
            syn_dict (dict, optional): A dictionary mapping node names to their synonym aspects.
        """
        self.root = root
        self.feature_nodes = [n for n in root.descendants]
        self.argument_nodes = [root] + self.feature_nodes
        if syn_dict is not None:
            self.glossary = {a_node: [syn.split(' ') for syn in syns]
                             for a, syns in syn_dict.items() for a_node in self.argument_nodes if a_node.name == a}
        else:
            self.glossary = {a_node: [
                syn.split(' ') for syn in a_node.synset] for a_node in self.argument_nodes}
        self.singularities = {a_node: wnl.lemmatize(
            a_node.name) == a_node.name for a_node in self.argument_nodes}

    def argument_node_for_id(self, id):
        """
        Retrieve an argument node by its ID.

        Args:
            id (int): The ID of the argument node.

        Returns:
            The argument node corresponding to the given ID.
        """
        return self.argument_nodes[id]

    def name(self):
        """
        Get the name of the product.

        Returns:
            str: The name of the product (root node's name).
        """
        return self.root.name

    def print_latex(self):
        """
        Print the LaTeX representation of the product's ontology_tree.
        """
        self.node_to_latex(self.root, 2)

    def node_to_latex(self, node, indent):
        """
        Recursively generate and print the LaTeX representation of a node and its children.

        Args:
            node: The current node to process.
            indent (int): The indentation level for the LaTeX output.
        """
        node_latex = '{\\textbf{' + \
            Product.aspect_to_latex(node.name.split(' ')) + '}'
        for syn_words in [syn_words for syn_words in self.glossary[node] if ' '.join(syn_words) != node.name]:
            node_latex += ', {}'.format(Product.aspect_to_latex(syn_words))
        node_latex += '}'
        if node.children:
            print('{}[{}'.format(' ' * indent, node_latex))
            for child in node.children:
                self.node_to_latex(child, indent + 2)
            print('{}]'.format(' ' * indent))
        else:
            print('{}[{}]'.format(' ' * indent, node_latex))

    @staticmethod
    def aspect_to_latex(words):
        """
        Convert a list of words into a LaTeX-formatted string.

        Args:
            words (list): A list of words to format.

        Returns:
            str: The LaTeX-formatted string.
        """
        return ' '.join(map(lambda w: '\\mbox{' + w + '}', words))

    @staticmethod
    def get_product(folder_name, cat, method=None):
        """
        Load a serialized product from a file.

        Args:
            folder_name (str): The folder containing the product file.
            cat (str): The category of the product.
            method (str, optional): The method used for generating the product file.

        Returns:
            Product: The deserialized Product object.

        Raises:
            Exception: If the product file does not exist.
        """
        if method == 'ours':
            method = None
        path = Product.FILE_DIR + folder_name + '/' + cat + \
            ('_{}'.format(method) if method else '') + Product.FILE_EXTENSION
        if isfile(path):
            f = open(path, 'rb')
            product: Product = pickle.load(f)
            f.close()
            return product
        else:
            raise Exception(
                'No representation found for product {} at {}'.format(cat, path))
