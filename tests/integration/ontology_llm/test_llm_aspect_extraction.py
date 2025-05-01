import pytest
import pandas as pd
from collections import Counter
from pathlib import Path
from src.ontology.ontology_llm.aspects.run import run_aspect_extraction
from src.constants import CACHE_DIR


@pytest.mark.integtest
def test_aspect_extraction(delete_output=True):
    reviews = pd.read_csv(
        "tests/data/reviews_of_10_products.csv")["content"].tolist()[:16]
    output_file = "integtest_aspect_extraction_output.csv"
    aspect_counter = run_aspect_extraction(reviews, output_file, CACHE_DIR, batch_size=8)

    aspects = []
    with open(output_file, 'r') as f:
        for line in f:
            aspect_list = eval(line.strip())
            assert isinstance(aspect_list, list), f"Expected a list, got {type(aspect_list)}"
            aspects.append(aspect_list)

    assert len(aspects) == len(reviews)
    assert isinstance(aspect_counter, Counter)

    for review, aspect_list in zip(reviews, aspects):
        print('Review:', review)
        print('---'*10)
        print('Aspects:', aspect_list)
        print('==='*10)
    print('Aspect counter:', aspect_counter)

    if delete_output:
        Path(output_file).unlink()


if __name__ == "__main__":
    test_aspect_extraction()
