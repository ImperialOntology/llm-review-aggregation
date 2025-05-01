import pytest
import pandas as pd
from collections import Counter
from src.ontology.ontology_llm.concepts.run import run_concept_extraction
from src.constants import CACHE_DIR


@pytest.mark.integtest
def test_concept_extraction():
    reviews = pd.read_csv(
        "tests/data/reviews_of_10_products.csv")["content"].tolist()
    aspect_lists = pd.read_csv(
        "tests/data/aspects_10_products.csv")["aspects"].tolist()
    aspect_lists = [eval(aspects) for aspects in aspect_lists]
    aspect_frequency = Counter([aspect for aspects in aspect_lists for aspect in aspects])

    root_name = "product"

    # Create a concept extraction manager

    synset_counts, synsets = run_concept_extraction(root_name=root_name,
                                                    reviews_list=reviews,
                                                    aspect_frequency=aspect_frequency,
                                                    cache_dir=CACHE_DIR,
                                                    top_k_aspects_to_keep=10)

    print('synsets: ', synsets)
    print('synset_counts: ', synset_counts)


if __name__ == "__main__":
    test_concept_extraction()
