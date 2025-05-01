import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
from src.ontology.ontology_bert.relations.relation_dataset import RelationDataset
from src.ontology.ontology_bert.relations.rel_bert_net import RelBertNet
from src.ontology.ontology_bert.base.bert_extractor_base import BaseBertExtractor
from src.constants import RELATION_NUM_CLASSES

device = torch.device('cuda')


class BertRelExtractor(BaseBertExtractor):
    model_class = RelBertNet

    def __init__(self, batch_size: int = 16):
        super().__init__(batch_size)

    @property
    def num_classes(self):
        return RELATION_NUM_CLASSES

    @property
    def print_frequency(self):
        return 500

    def extract_relations(self, n_aspects, data: RelationDataset):
        """
        Extract relation probabilities between aspects from the dataset using the trained model.

        Args:
            n_aspects (int): The total number of aspects in the dataset.
            data (RelationDataset): The dataset containing relation instances between aspects.

        Returns:
            tuple: A tuple containing:
                - prob_matrix (np.ndarray): A matrix where each entry (i, j) represents the accumulated 
                  probability of a relation between aspect i and aspect j.
                - count_matrix (np.ndarray): A matrix where each entry (i, j) represents the count of 
                  relation instances between aspect i and aspect j.
        """
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=4,
                            collate_fn=RelationDataset.generate_production_batch)

        self.net.eval()

        prob_matrix = np.zeros((n_aspects, n_aspects))
        count_matrix = np.zeros((n_aspects, n_aspects))

        with torch.no_grad():
            for input_ids, attn_mask, fst_indices, snd_indices, instances in loader:
                # send batch to gpu
                input_ids, attn_mask, fst_indices, snd_indices = tuple(i.to(device) for i in
                                                                       [input_ids, attn_mask,
                                                                        fst_indices, snd_indices])

                # forward pass
                output_scores = softmax(self.net(input_ids, attn_mask, fst_indices, snd_indices), dim=1)
                rel_scores = output_scores.narrow(1, 1, 2)

                for ins, scores in zip(instances, rel_scores.tolist()):
                    forward_score, backward_score = scores
                    prob_matrix[ins.snd_a][ins.fst_a] += forward_score
                    prob_matrix[ins.fst_a][ins.snd_a] += backward_score
                    count_matrix[ins.snd_a][ins.fst_a] += 1
                    count_matrix[ins.fst_a][ins.snd_a] += 1

        return prob_matrix, count_matrix
