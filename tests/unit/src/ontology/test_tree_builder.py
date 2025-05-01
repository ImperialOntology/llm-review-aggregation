import pytest
import numpy as np
from anytree import Node, PreOrderIter

from src.ontology.tree_builder import TreeBuilder

@pytest.fixture
def tree_builder():
    tree_builder = TreeBuilder("product")
    return tree_builder


def test_build_tree_with_only_root(tree_builder):
    """
    Test that the tree is built correctly with only the root node
    """
    result_tree = tree_builder.build_tree(
        filtered_norm_matrix=np.array([[0]]),
        filtered_concepts=["product"],)

    assert isinstance(result_tree, Node)
    assert len(list(PreOrderIter(result_tree))) == 1  # Only the root node
    assert result_tree.name == "product"
    assert result_tree.parent is None


def test_not_a_parent_to_itself(tree_builder):
    """
    Test that a node is not its own parent even if it is most
    related to itself. In this case the node's parent should be the second most
    related node.
    """
    relation_matrix = np.array([[0, 0],
                                [0.3, 0.9]])
    concepts = ["product", "price"]
    result_tree = tree_builder.build_tree(
        filtered_norm_matrix=relation_matrix,
        filtered_concepts=concepts)
    assert isinstance(result_tree, Node)
    assert len(list(PreOrderIter(result_tree))) == 2
    assert result_tree.name == "product"
    assert result_tree.parent is None
    assert result_tree.children[0].name == "price"
    assert result_tree.children[0].parent == result_tree


def test_root_has_no_parent(tree_builder):
    """
    Test that the root node has no parent.
    """
    relation_matrix = np.array([[0, 0.9],
                                [0.3, 0.5]])
    concepts = ["product", "price"]
    result_tree = tree_builder.build_tree(
        filtered_norm_matrix=relation_matrix,
        filtered_concepts=concepts)
    assert isinstance(result_tree, Node)
    assert len(list(PreOrderIter(result_tree))) == 2
    assert result_tree.name == "product"
    assert result_tree.parent is None
    assert result_tree.children[0].name == "price"
    assert result_tree.children[0].parent == result_tree


def test_when_loop_then_added_as_child_of_root(tree_builder):
    """
    Test that when there is a loop for a given node, the node is added as a direct child
    of the root node.
    """
    relation_matrix = np.array([[0, 0, 0],
                                [0.3, 0, 0.9],  # quality wants to be child of texture
                                [0, 0.8, 0]])  # texture wants to be child of quality
    concepts = ["product", "quality", "texture"]
    result_tree = tree_builder.build_tree(
        filtered_norm_matrix=relation_matrix,
        filtered_concepts=concepts)
    assert isinstance(result_tree, Node)
    assert len(list(PreOrderIter(result_tree))) == 3
    assert result_tree.name == "product"
    assert result_tree.parent is None

    assert len(result_tree.children) == 1
    texture_node = result_tree.children[0]
    assert texture_node.name == "texture"  # texture became child of root

    assert len(texture_node.children) == 1
    quality_node = texture_node.children[0]
    assert quality_node.name == "quality"
