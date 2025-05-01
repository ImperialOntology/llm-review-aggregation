import numpy as np
from anytree import Node, LoopError


class TreeBuilder:
    """
    Class to build an ontology tree based on the meronym scores matrix of concepts.
    """

    def __init__(self, root_name):
        self.root_name = root_name
        self.tree = None

    def get_relatedness(self, concepts, concept_counts, meronym_matrix):
        """
        Calculates the relatedness between aspects based on the extracted relations.
        """
        norm_matrix = np.array([concept_counts[concept]
                               for concept in concepts])
        normalised = meronym_matrix / norm_matrix

        # ensure root does not get filtered out
        root_idx = concepts.index(self.root_name)
        normalised[root_idx][root_idx] = 1

        # Calculate row sums
        row_sums = normalised.sum(axis=1)
        non_zero_concepts = (row_sums != 0)
        filtered_norm_matrix = (
            normalised[non_zero_concepts][:, non_zero_concepts])
        filtered_concepts = [aspect for i, aspect in enumerate(
            concepts) if non_zero_concepts[i]]
        return filtered_norm_matrix, filtered_concepts

    @staticmethod
    def _get_parent(idx, relatedness_matrix):
        """
        Returns the parent of the concept at the given index based on the relatedness matrix.
        """
        row = relatedness_matrix[idx]
        max_idx = np.argmax(row)
        return max_idx, row[max_idx]

    def build_tree(self, filtered_norm_matrix, filtered_concepts):
        """
        Builds the ontology tree based on the relatedness matrix.

        Here are the design choices made for this ontology tree generation algorithm
        - sort all the nodes by their relatedness to their parent
        - add the most related node as child of its corresponding parent if no loop is created by doing so
        - if loop is created, add the node as child of the root node

        - NOTE: nodes being their own parent is treated as anomaly, 
            instead the second most related node is considered as parent

        :param filtered_norm_matrix: The relatedness matrix,
            contains relatedness scores between concepts aligned with filtered_concepts.
        :param filtered_concepts: The filtered list of concepts,
        """
        # Set relatedness of nodes to themselves to 0
        np.fill_diagonal(filtered_norm_matrix, 0)

        # Add dictionary to hold all nodes
        nodes = []
        for idx, concept in enumerate(filtered_concepts):
            node = Node(concept, parent=None)
            node.idx = idx
            nodes.append(node)

        # Root node
        root_idx = filtered_concepts.index(self.root_name)
        root = nodes[root_idx]
        self.tree = root

        # concept ids sorted by reverse order of relatedness to their parent
        non_root_ids = [i for i in range(len(filtered_concepts)) if i != root_idx]
        sorted_concept_ids = sorted(
            non_root_ids,
            key=lambda idx: self._get_parent(idx, filtered_norm_matrix)[1],
            reverse=True
        )

        for idx in sorted_concept_ids:
            parent_idx, _ = self._get_parent(idx, filtered_norm_matrix)
            current_node = nodes[idx]
            parent_candidate = nodes[parent_idx]

            try:
                current_node.parent = parent_candidate
            except LoopError:
                # If a loop is created, add the node as child of the root
                current_node.parent = root

        return self.tree

    @staticmethod
    def convert_tree_to_dict(tree):
        """
        Returns the ontology tree as a dictionary.
        """
        # iterate through tree and build dictionary

        tree_dict = {}

        def _get_children(node):
            if not node.children:
                return None  # leaf node
            children = {}
            for child in node.children:
                children[child.name] = _get_children(child)
            return children

        tree_dict[tree.name] = _get_children(tree)
        tree_dict = tree_dict

        return tree_dict
