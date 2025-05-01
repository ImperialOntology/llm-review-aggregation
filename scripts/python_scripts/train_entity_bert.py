import time
from argparse import ArgumentParser
from src.ontology.ontology_bert.aspects.run import train_term_extractor
from src.constants import BERT_ENTITY_DATASET_PATH, BERT_TRAIN_CATEGORIES
from logger import logger


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to save the model to")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Batch size for training")
    parser.add_argument("--data-path", type=str, required=False, default=BERT_ENTITY_DATASET_PATH,
                        help="Path to the dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()
    logger.info("Training BERT entity extractor...")
    train_term_extractor(
        size=None,
        valid_frac=0.05,
        batch_size=args.batch_size,
        bert_entity_extractor_path=args.model_path,
        categories=BERT_TRAIN_CATEGORIES,
        path=args.data_path,
        random_state=42
    )
    end = time.time()
    logger.info(f"BERT entity extractor trained in {end - start:.2f} seconds")
    logger.info(f"Model saved to {args.model_path}")


if __name__ == '__main__':
    main()
