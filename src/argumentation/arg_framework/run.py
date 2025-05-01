from anytree import Node, PreOrderIter
import pandas as pd
from tqdm import tqdm

from src.argumentation.sentiment.bert_analyzer import BertAnalyzer
from src.argumentation.arg_framework.product import Product
from src.argumentation.arg_framework.framework import Framework
from logger import logger


def convert_dict_to_tree(tree_dict: dict) -> Node:
    """
    Converts a dictionary representation of a tree to a Node object

    :param tree_dict: dictionary representation of a tree
          {root_name : {child_name_1: {child_name_1_1: None, 
                                       child_name_1_2: None}, 
                        child_name_2: None}}

    :return: Node object pointing to the root of the tree, name is equal to dictionary key
    """
    root_name = list(tree_dict.keys())[0]
    root = Node(root_name)
    child_dir = tree_dict[root_name]

    def _construct_tree(current_node, current_dict):
        if isinstance(current_dict, dict):
            for child_name, child_info in current_dict.items():
                child_node = Node(child_name, parent=current_node)
                _construct_tree(child_node, child_info)

    _construct_tree(root, child_dir)

    return root


def get_arguments(frameworks, tree):
    """
    Returns the argumentation frameworks as a list of dictionaries

    Each dictionary represents an argumentation framework and is structured as follows:
    {
        "root_node_name": [strength, children_dict],
    }
    where children_dict is a dictionary of the same structure as the parent dictionary.
    children_dict = {
        "child_node_name_1": [strength, children_dict],
        "child_node_name_2": [strength, children_dict],
        ...
     }

    children_dict can be None if the node is a leaf node.

    :param frameworks: dict of Framework objects, {product_id: Framework}
    :param tree: root Node of the ontology tree

    :return: pandas dataframe of arguments and their metadata with following columns:
        - product_id
        - aspect
        - polarity
        - strength
        - strongest_support_phrase
        - strongest_attack_phrase
        - strongest_support_feature
        - strongest_attack_feature
    """
    fw_df = []
    nodes = [node for node in PreOrderIter(tree)]

    for product_id, fw in frameworks.items():
        for node in nodes:
            attack_phrase = fw.best_attacking_phrase(node)
            attack_phrase = attack_phrase.text if attack_phrase else None
            support_phrase = fw.best_supporting_phrase(node)
            support_phrase = support_phrase.text if support_phrase else None

            attack_feature = fw.get_strongest_attacking_subfeature(node)
            attack_feature = attack_feature.name if attack_feature else None
            support_feature = fw.get_strongest_supporting_subfeature(node)
            support_feature = support_feature.name if support_feature else None

            fw_df.append({
                'product_id': product_id,
                'aspect': node.name,
                'polarity': fw.argument_polarities[node],
                'strength': fw.strengths[node],
                'strongest_support_phrase': support_phrase,
                'strongest_attack_phrase': attack_phrase,
                'strongest_support_feature': support_feature,
                'strongest_attack_feature': attack_feature,
            })
    fw_df = pd.DataFrame(fw_df)
    return fw_df


def run_argumentative_analysis(reviews_df, tree,
                               synsets, model_path='./SentimentBertModel.model'):
    """
    Run the argumentative analysis on the reviews and tree.

    :param reviews_df: dataframe with columns ['id', 'content', 'product_id']
    :param tree: ontology tree as a dictionary
    :param synsets: dictionary with synsets as keys and their corresponding words as values
    :param model_path: path to the saved sentiment BERT model

    :return: dict of Framework objects, {product_id: Framework}
    """
    bert_analyzer = BertAnalyzer()
    logger.info("Loading BERT model...")
    bert_analyzer.load_saved(model_path)

    shortlisted_synsets = [node.name for node in PreOrderIter(tree)]

    synset_dict = {concept: synset for concept, synset in synsets.items()
                   if concept in shortlisted_synsets}
    product = Product(tree, synset_dict)
    frameworks = dict()
    logger.info("Creating frameworks...")
    total = len(reviews_df['product_id'].unique())
    for product_id, product_review_df in tqdm(reviews_df.groupby('product_id'), total=total):
        fw = Framework(product, product_id, product_review_df, bert_analyzer)
        frameworks[product_id] = fw
    return frameworks
