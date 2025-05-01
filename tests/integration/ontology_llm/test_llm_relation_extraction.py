from src.ontology.ontology_llm.relations.run import run_relation_extraction
from src.constants import PROMPT_BASED_LLM_MODEL_NAME, CACHE_DIR
import json
import pandas as pd
from anytree import RenderTree


def test_relation_extraction_and_ontology_tree_construction():
    """
    should produce the following tree structure

    {'product': {'makeup': None, 'liner': None, 'packaging': None}}

    Node('/product', idx=0)
    ├── Node('/product/makeup', idx=1)
    ├── Node('/product/liner', idx=2)
    └── Node('/product/packaging', idx=3)
    """
    # load data
    with open("tests/data/concepts_10_products.json", "r") as file:
        data = json.load(file)
    root_name = data["root_name"]
    concept_counts = data["synset_counts"]
    synsets = data["synsets"]

    reviews_list = pd.read_csv(
        "tests/data/reviews_of_10_products.csv")["content"].tolist()

    # extract relations and construct ontology tree
    tree = run_relation_extraction(reviews_list=reviews_list,
                                   root_name=root_name,
                                   synsets=synsets,
                                   synset_counts=concept_counts,
                                   cache_dir=CACHE_DIR,
                                   model_name=PROMPT_BASED_LLM_MODEL_NAME,
                                   batch_size=8)

    # print results
    print('Tree pretty---'*10)
    print(RenderTree(tree))


if __name__ == '__main__':
    test_relation_extraction_and_ontology_tree_construction()
