import time

from argparse import ArgumentParser

from datasets import load_dataset
from pathlib import Path

from src.argumentation.sentiment.sentiment_annotation import prepare_bert_trained_dataset
from src.argumentation.sentiment.bert_analyzer import BertAnalyzer
from logger import logger


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to save the model to")
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    # load data
    SemEvalData_train = load_dataset(
        "alexcadillon/SemEval2014Task4", "laptops", split='train', cache_dir=cache_dir)
    SemEvalData_train_xml_loc = cache_dir / 'SemEvalData_train.xml'
    SemEvalData_test = load_dataset(
        "alexcadillon/SemEval2014Task4", "laptops", split='test', cache_dir=cache_dir)
    SemEvalData_test_xml_loc = cache_dir / 'SemEvalData_test.xml'

    # Collect the train, test and tril file
    prepare_bert_trained_dataset(SemEvalData_train, SemEvalData_train_xml_loc)
    prepare_bert_trained_dataset(SemEvalData_test, SemEvalData_test_xml_loc)

    # train bert model
    start = time.time()
    BA = BertAnalyzer()
    model_path = Path(args.model_path)
    # To train the model
    BA.train(SemEvalData_train_xml_loc, model_path)

    # evaluate the model
    # will just log the metrics,
    BA.evaluate(SemEvalData_test_xml_loc)
    end = time.time()
    logger.info(f"Model trained and evaluated in {end - start:.2f} seconds")


if __name__ == '__main__':
    main()
