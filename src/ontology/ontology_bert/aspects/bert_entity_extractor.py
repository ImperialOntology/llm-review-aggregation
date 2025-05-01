from src.ontology.ontology_bert.base.bert_extractor_base import BaseBertExtractor
from typing import List
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import statistics
from src.ontology.ontology_bert.aspects.entity_bert_net import EntityBertNet
from src.ontology.ontology_bert.aspects.entity_dataset import EntityDataset, EntityInstance
from src.constants import ENTITY_NUM_CLASSES

device = torch.device('cuda')


class BertEntityExtractor(BaseBertExtractor):
    model_class = EntityBertNet

    def __init__(self, batch_size: int = 32):
        super().__init__(batch_size)

    @property
    def num_classes(self):
        return ENTITY_NUM_CLASSES

    @property
    def print_frequency(self):
        return 250

    def extract_aspect_probabilities(self, entities: List[EntityInstance], data: EntityDataset):
        """
        Extract probabilities for aspects (entities) from the dataset using the trained model.

        Args:
            entities (List[EntityInstance]): A list of entity instances to extract probabilities for.
            data (EntityDataset): The dataset containing the input data for the entities.

        Returns:
            dict: A dictionary mapping each entity to its average probability score. If no scores are available
                  for an entity, its value will be `None`.
        """
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=4,
                            collate_fn=EntityDataset.generate_production_batch)

        self.net.eval()

        probs = {entity: [] for entity in entities}

        with torch.no_grad():
            for input_ids, attn_mask, entity_indices, instances in loader:
                # send batch to gpu
                input_ids, attn_mask, entity_indices = tuple(i.to(device) for i in [input_ids, attn_mask,
                                                                                    entity_indices])

                # forward pass
                output_scores = softmax(self.net(input_ids, attn_mask, entity_indices), dim=1)
                entity_scores = output_scores.narrow(1, 1, 1).flatten()

                for ins, score in zip(instances, entity_scores.tolist()):
                    probs[ins.entity].append(score)

        return {t: statistics.mean(t_probs) if len(t_probs) > 0 else None for t, t_probs in probs.items()}
