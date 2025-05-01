from scipy.special import comb
import numpy as np


class SynsetExtractor:
    """
    A class for extracting and clustering aspects into synsets based on word embeddings similarity.

    Attributes:
        similarity_threshold (float): The threshold for determining similarity between aspects.
        num_clustering_levels (int): The number of clustering levels to consider for connectivity.
    """

    def __init__(self,
                 similarity_threshold: float = 0.15,
                 num_clustering_levels: int = 1):
        """
        Initialize the SynsetExtractor.

        Args:
            similarity_threshold (float): The threshold for determining similarity between aspects.
            num_clustering_levels (int): The number of clustering levels to consider for connectivity.
        """
        self.similarity_threshold = similarity_threshold
        self.num_clustering_levels = num_clustering_levels

    @staticmethod
    def _connected(idx1, idx2, m, k):
        """
        Check if two indices are connected within k levels in the similarity matrix.

        Args:
            idx1 (int): The first index.
            idx2 (int): The second index.
            m (np.ndarray): The similarity matrix.
            k (int): The number of levels to check for connectivity.

        Returns:
            bool: True if the indices are connected, False otherwise.
        """
        if k == 0:
            return False
        if m[idx1][idx2] != 0:
            return True
        conn = False
        for idx3 in range(len(m[idx1])):
            if m[idx1][idx3] != 0:
                conn = conn or SynsetExtractor._connected(idx3, idx2, m, k-1)
        return conn

    @staticmethod
    def _clique_similarity(c, m):
        """
        Calculate the similarity of a clique based on average relation and density.

        Args:
            c (set): A set of indices representing the clique.
            m (np.ndarray): The similarity matrix.

        Returns:
            float: The similarity score of the clique.
        """
        if len(c) == 1:
            return 1
        avg_rel = np.mean([m[idx1][idx2]
                           for idx1 in c for idx2 in c if idx1 < idx2])
        density = sum(1 for idx1 in c for idx2 in c if idx1 <
                      idx2 and m[idx1][idx2] != 0) / comb(len(c), 2)
        return avg_rel * density

    def cluster_aspects(self, root_aspect, counts, wv_model):
        """
        Cluster aspects into synsets based on word embeddings similarity.

        Args:
            root_aspect (str): The root aspect to prioritize in clustering.
            counts (dict): A dictionary mapping aspects to their counts.
            wv_model: A word embedding model with similarity and synonym-checking methods.

        Returns:
            tuple: A tuple containing:
                - synset_counts (dict): A dictionary mapping aspects to their total counts.
                - synsets (dict): A dictionary mapping aspects to their synonym groups.
        """
        aspects = list(counts.keys())
        syn_m = np.zeros((len(aspects), len(aspects)))
        for idx1, a1 in enumerate(aspects):
            for idx2, a2 in enumerate(aspects):
                syn_m[idx1][idx2] = wv_model.similarity(a1, a2) \
                    if wv_model.are_syns(a1, a2, self.similarity_threshold) else 0
        cliques = {frozenset({idx2 for idx2 in range(len(aspects))
                              if idx1 == idx2 or self._connected(idx1, idx2, syn_m, self.num_clustering_levels)})
                   for idx1 in range(len(aspects))}
        non_overlapping = []
        for c in sorted(cliques, key=lambda c: self._clique_similarity(c, syn_m), reverse=True):
            if not any(idx in existing_c for existing_c in non_overlapping for idx in c):
                non_overlapping.append(c)
        all_appointments = non_overlapping + [frozenset({idx}) for idx in range(len(aspects))
                                              if not any(idx in c for c in non_overlapping)]

        synsets = list(
            map(lambda c: {aspects[idx] for idx in c}, all_appointments))
        synsets = {max(group, key=counts.get): list(set(group))
                   for group in synsets}
        if root_aspect not in synsets:
            for key, synset in synsets.items():
                if root_aspect in synset:
                    del synsets[key]
                    synsets[root_aspect] = synset
                    break

        # remove aspect synonyms and reorder list based on sum of all synonym counts
        aspects = [aspect for aspect in aspects if aspect in synsets.keys()]
        synset_counts = {aspect: sum(
            counts[syn] for syn in synsets[aspect]) for aspect in aspects}

        return synset_counts, synsets
