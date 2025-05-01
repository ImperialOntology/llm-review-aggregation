import pytest
import json
import pandas as pd
from collections import Counter
from src.ontology.ontology_bert.concepts.run import run_concept_extraction
from src.constants import CACHE_DIR


@pytest.mark.integtest
def test_concept_extraction():
    reviews = pd.read_csv(
        "tests/data/reviews_of_10_products.csv")["content"].tolist()
    aspect_frequency = json.load(open("tests/data/bert_aspect_freq_10_products.json"))

    root_name = "product"

    # Create a concept extraction manager

    synset_counts, synsets = run_concept_extraction(root_name=root_name,
                                                    reviews_list=reviews,
                                                    aspect_frequency=aspect_frequency,
                                                    cache_dir=CACHE_DIR,
                                                    top_k_aspects_to_keep=10,
                                                    njobs=4)

    print('synsets: ', synsets)
    print('synset_counts: ', synset_counts)


if __name__ == "__main__":
    test_concept_extraction()
