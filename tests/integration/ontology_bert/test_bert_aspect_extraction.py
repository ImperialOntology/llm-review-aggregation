import os
from collections import Counter
import pandas as pd
from src.ontology.ontology_bert.aspects.run import train_term_extractor, run_aspect_extraction


def test_bert_training_and_aspect_extraction():
    """
    Test the BERT model training and aspect extraction.
    """

    test_bert_path = "test_term_extractor_bert.pt"
    train_term_extractor(
        size=200,
        valid_frac=0.05,
        batch_size=8,
        bert_entity_extractor_path=test_bert_path,
        categories=['camera', 'backpack', 'cardigan', 'guitar', 'laptop'],
        path='bert-train-data/term_extraction_datasets',
        random_state=42
    )

    assert os.path.exists(test_bert_path), "BERT model file not found after training."

    reviews = pd.read_csv("tests/data/reviews_of_10_products.csv")["content"].tolist()
    aspect_counts = run_aspect_extraction(
        reviews_list=reviews,
        batch_size=8,
        bert_entity_extractor_path=test_bert_path,
        njobs=4
    )
    assert isinstance(aspect_counts, Counter), "Aspect counts should be a Counter object."
    assert len(aspect_counts) > 0, "Aspect counts should not be empty."

    print("Aspect counts:", aspect_counts)

    # Clean up the test model file
    if os.path.exists(test_bert_path):
        os.remove(test_bert_path)
        print(f"Removed test model file: {test_bert_path}")


if __name__ == "__main__":
    test_bert_training_and_aspect_extraction()
    print("Test passed.")
