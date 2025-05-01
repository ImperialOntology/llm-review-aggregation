# base_bert_extractor.py
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import time
import numpy as np
from sklearn import metrics

from src.ontology.ontology_bert.base.bert_dataset_base import BaseDataset
from src.constants import (
    LEARNING_RATE,
    MAX_GRAD_NORM,
    N_EPOCHS,
    WARM_UP_FRAC
)
from logger import logger

device = torch.device('cuda')


# loss criterion
loss_criterion = CrossEntropyLoss()


class BaseBertExtractor:
    """
    Abstract base class for BERT-based extractors (e.g., entity and relation).
    """
    model_class = None  # Placeholder for the model class, to be set in subclasses

    def __init__(self, batch_size: int):
        assert self.model_class is not None, "model_class must be set in subclasses"
        self.net = self.model_class()
        self.net.to(device)
        self.batch_size = batch_size

    @classmethod
    def load_saved(cls, path: str,  batch_size: int):
        """Loads a saved instance of the extractor."""
        extractor = cls(batch_size)
        extractor.net.load_state_dict(torch.load(path, map_location=device))
        extractor.net.to(device)
        extractor.net.eval()
        return extractor

    @classmethod
    def train_and_validate(cls,
                           train_data: BaseDataset,
                           valid_data: BaseDataset = None,
                           batch_size: int = 32,
                           save_file: str = 'extractor.pt'):
        """Trains and optionally validates the extractor."""
        extractor = cls(batch_size)
        extractor.train(train_data, valid_data)
        torch.save(extractor.net.state_dict(), save_file)
        return extractor

    @classmethod
    def train_eval(cls,
                   train_data: BaseDataset,
                   eval_data: BaseDataset,
                   batch_size: int = 32):
        """Trains and evaluates the extractor on a separate evaluation set."""
        extractor = cls(batch_size)
        extractor.train(train_data)
        accuracy, _, _, _, _ = extractor.evaluate(eval_data)
        return accuracy

    def train(self, train_data, valid_data=None):
        """Trains the extractor on the given training data."""
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  collate_fn=train_data.generate_batch)

        # set up optimizer with weight decay
        optimiser = Adam(self.net.parameters(), lr=LEARNING_RATE)

        # set up scheduler for learning rate
        n_training_steps = len(train_loader) * N_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimiser,
            num_warmup_steps=int(WARM_UP_FRAC * n_training_steps),
            num_training_steps=n_training_steps
        )

        start = time.time()

        for epoch_idx in range(N_EPOCHS):
            self.net.train()
            batch_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                # move batch to gpu
                inputs = tuple(i.to(device) for i in batch[:-1])  # All elements except the last (labels/instances)
                target = batch[-1].to(device)

                # zero param gradients
                optimiser.zero_grad()

                # forward pass
                output_scores = self.net(*inputs)

                # backward pass
                loss = loss_criterion(output_scores, target)
                loss.backward()

                # clip gradient norm
                clip_grad_norm_(parameters=self.net.parameters(), max_norm=MAX_GRAD_NORM)

                # optimise
                optimiser.step()

                # update learning rate
                scheduler.step()

                # print interim stats
                batch_loss += loss.item()
                if (batch_idx + 1) % self.print_frequency == 0:
                    progress = (batch_idx + 1) / len(train_loader)
                    avg_loss = batch_loss / self.print_frequency
                    logger.info(
                        f'Epoch: {epoch_idx + 1} -- Progress: {progress:.4f} -- Batch: {batch_idx + 1} -- Avg Loss: {avg_loss:.4f}')
                    batch_loss = 0.0

            logger.info('Epoch done')

            if valid_data is not None:
                self.print_evaluate(valid_data)

        end = time.time()
        logger.info(f'Training took {end - start:.2f} seconds')

    def print_evaluate(self, test_data):
        """Evaluates the model and prints the results."""
        accuracy, precision, recall, f1, cm = self.evaluate(test_data)

        logger.info('Accuracy:', accuracy)
        logger.info('Precision:', precision)
        logger.info('Recall:', recall)
        logger.info('Macro F1:', f1)
        logger.info('Confusion Matrix:')
        logger.info(cm)

    def evaluate(self, data):
        """Evaluates the model on the given data."""
        test_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=4,
                                 collate_fn=data.generate_batch)

        self.net.eval()

        outputs = []
        targets = []

        with torch.no_grad():
            for batch in test_loader:
                # move batch to gpu
                inputs = tuple(i.to(device) for i in batch[:-1])
                target = batch[-1].to(device)

                # forward pass
                output_scores = self.net(*inputs)
                _, output_labels = torch.max(output_scores.data, 1)

                outputs += output_labels.tolist()
                targets += target.tolist()

        assert len(outputs) == len(targets)

        correct = (np.array(outputs) == np.array(targets))
        accuracy = correct.sum() / correct.size
        precision = metrics.precision_score(targets, outputs, average=None)
        recall = metrics.recall_score(targets, outputs, average=None)
        f1 = metrics.f1_score(targets, outputs, labels=range(self.num_classes), average='macro')
        cm = metrics.confusion_matrix(targets, outputs, labels=range(self.num_classes))
        return accuracy, precision, recall, f1, cm

    @property
    def num_classes(self):
        """Returns the number of classes. To be implemented by subclasses."""
        raise NotImplementedError

    @property
    def print_frequency(self):
        """Defines the frequency of printing training stats. Can be overridden in subclasses."""
        return 250
