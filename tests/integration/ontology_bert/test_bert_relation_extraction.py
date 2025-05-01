import json
import os
import pandas as pd
from anytree import RenderTree, Node
from src.ontology.ontology_bert.relations.run import run_relation_extraction, train_relation_extractor


def test_relation_extraction_and_ontology_tree_construction():
    """
    should produce the following tree structure

    Node('/product', idx=0)
    ├── Node('/product/packaging', idx=4)
    │   └── Node('/product/packaging/price', idx=1)
    └── Node('/product/makeup', idx=3)
        └── Node('/product/makeup/bag', idx=2)
    """
    # train relation extractor
    test_bert_path = "test_relation_extractor_bert.pt"
    train_relation_extractor(
        size=2000,  # decrease for faster testing
        valid_frac=0.05,
        batch_size=8,
        bert_relation_extractor_path=test_bert_path,
        categories=['camera', 'backpack', 'cardigan', 'guitar', 'laptop'],
        path='bert-train-data/relation_extraction_datasets',
        random_state=42
    )

    assert os.path.exists(test_bert_path), "BERT model file not found after training."

    # load data
    with open("tests/data/concepts_10_products.json", "r") as file:
        data = json.load(file)
    root_name = data["root_name"]
    concept_counts = data["synset_counts"]
    synsets = data["synsets"]

    reviews_list = pd.read_csv(
        "tests/data/reviews_of_10_products.csv")["content"].tolist()

    # extract relations and construct ontology tree
    tree = run_relation_extraction(reviews_list,
                                   root_name,
                                   synsets,
                                   concept_counts,
                                   batch_size=8,
                                   njobs=4,
                                   bert_relation_extractor_path=test_bert_path)

    assert isinstance(tree, Node), "Tree should anytree Node."
    assert tree.name == root_name, "Tree root name should match the provided root name."

    # print results
    print('Tree pretty---'*10)
    print(RenderTree(tree))

    # Clean up the test model file
    if os.path.exists(test_bert_path):
        os.remove(test_bert_path)
        print(f"Removed test model file: {test_bert_path}")


if __name__ == '__main__':
    test_relation_extraction_and_ontology_tree_construction()
