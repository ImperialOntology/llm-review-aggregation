import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.argumentation.sentiment.tdbertnet import TDBertNet
from src.argumentation.sentiment.bert_dataset import BertDataset, Instance, polarity_indices, generate_batch
import time
import numpy as np
from sklearn import metrics
from logger import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
MAX_EPOCHS = 6
LEARNING_RATE = 0.00002
loss_criterion = nn.CrossEntropyLoss()


def loss(outputs, labels):
    return loss_criterion(outputs, labels)


class BertAnalyzer:
    def __init__(self):
        self.net = None

    def load_saved(self, path):
        self.net = TDBertNet(len(polarity_indices))
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
        # initialise GPU
        self.net.to(device)

    def train(self, data_file, save_path, mask_target=False):
        '''
        Find tuning the bert model

        Args:
        dataset_file: path to the xml file
        save_path: save path of the model
        '''
        train_data = BertDataset.from_file(data_file, mask_target=mask_target)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                  collate_fn=generate_batch)

        self.net = TDBertNet(len(polarity_indices))
        self.net.to(device)

        optimiser = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

        start = time.time()

        for epoch in range(MAX_EPOCHS):
            batch_loss = 0.0
            for i, batch in enumerate(train_loader):

                # send batch to gpu
                input_ids, attn_mask, target_indices, labels = tuple(
                    i.to(device) for i in batch)

                # zero param gradients
                optimiser.zero_grad()

                # forward pass
                outputs = self.net(input_ids, attn_mask, target_indices)

                # backward pass
                l = loss(outputs, labels)
                l.backward()

                # optimise
                optimiser.step()

                # print interim stats every 10 batches
                batch_loss += l.item()
                if i % 10 == 9:
                    logger.info('epoch:', epoch + 1, '-- batch:', i +
                                1, '-- avg loss:', batch_loss / 10)
                    batch_loss = 0.0

        end = time.time()
        logger.info('Training took', end - start, 'seconds')

        torch.save(self.net.state_dict(), save_path)

    def evaluate(self, data_file, mask_target=False):
        '''
        Evaluate the model performance
        data_file: path to the xml file

        '''
        test_data = BertDataset.from_file(data_file, mask_target=mask_target)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                 collate_fn=generate_batch)

        predicted = []
        truths = []
        with torch.no_grad():
            for info in test_loader:
                input_ids, attn_mask, target_indices, labels = tuple(
                    i.to(device) for i in info)
                outputs = self.net(input_ids, attn_mask, target_indices)
                _, pred = torch.max(outputs.data, 1)
                predicted += pred.tolist()
                truths += labels.tolist()

        correct = (np.array(predicted) == np.array(truths))
        accuracy = correct.sum() / correct.size
        logger.info('accuracy:', accuracy)

        cm = metrics.confusion_matrix(
            truths, predicted, labels=range(len(polarity_indices)))
        logger.info('confusion matrix:')
        logger.info(cm)

        f1 = metrics.f1_score(truths, predicted, labels=range(
            len(polarity_indices)), average='macro')
        logger.info('macro F1:', f1)

    def get_batch_sentiment_polarity_from_file(self, data):
        '''
        Get the polairty for a certian trial input
        '''
        # originally is frmo_data, currently from file for convinent purposes
        dataset = BertDataset.from_file(data)
        loader = DataLoader(dataset, batch_size=128, shuffle=False,
                            num_workers=4, collate_fn=generate_batch)

        self.net.eval()

        predicted = []
        with torch.no_grad():
            for input_ids, attn_mask, target_indices, _ in loader:
                input_ids, attn_mask, target_indices = tuple(
                    i.to(device) for i in [input_ids, attn_mask, target_indices])
                outputs = self.net(input_ids, attn_mask, target_indices)
                batch_val, batch_pred = torch.max(outputs.data, 1)
                predicted += [BertAnalyzer.get_polarity(val, pred)
                              for val, pred in zip(batch_val, batch_pred)]

        return predicted

    def get_batch_sentiment_polarity(self, data):
        '''
        Get the polairty for a certian trial input
        '''
        dataset = BertDataset.from_data(data)
        loader = DataLoader(dataset, batch_size=128, shuffle=False,
                            num_workers=4, collate_fn=generate_batch)

        self.net.eval()

        predicted = []
        with torch.no_grad():
            for input_ids, attn_mask, target_indices, _ in loader:
                input_ids, attn_mask, target_indices = tuple(
                    i.to(device) for i in [input_ids, attn_mask, target_indices])
                outputs = self.net(input_ids, attn_mask, target_indices)
                batch_val, batch_pred = torch.max(outputs.data, 1)
                predicted += [BertAnalyzer.get_polarity(val, pred)
                              for val, pred in zip(batch_val, batch_pred)]

        return predicted

    def get_sentiment_polarity(self, text, char_from, char_to):
        '''
        A more detailed output 
        '''
        instance = Instance(text, char_from, char_to)
        tokens, tg_from, tg_to = instance.get(mask_target=False)
        text, target_indices = instance.to_tensor()

        with torch.no_grad():
            outputs, attentions = self.net(text, target_indices)

        val, pred = torch.max(outputs.data, 1)
        return BertAnalyzer.get_polarity(val, pred)

    @staticmethod
    def get_polarity(val, pred):
        if pred == 0:
            # positive
            return val
        elif pred == 1:
            # negative
            return -val
        else:
            # neutral or conflicted
            return 0
