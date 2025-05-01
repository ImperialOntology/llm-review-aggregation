from typing import Dict, Optional, Any, Callable
from tqdm import tqdm
import json
import time

from src.llm_judge.llm_manager import Gemini
from src.llm_judge.prompt_manager import LlmJudgePromptManager
from logger import logger


def run_aspect_judging(
    llm: Gemini,
    category_name: str,
    aspect_frequency: Dict[str, int],
    save_to_db: bool = False,
    aspect_extraction_id: Optional[int] = None,
    db_update_callback: Optional[Callable] = None,
    sleep_time: int = 3,
) -> Dict[str, int]:
    """
    Run the aspect judging process.

    Args:
        llm: LLM instance for generating judgements
        category_name: Name of the category for context
        aspect_frequency: Dictionary mapping aspects to their frequencies
        save_to_db: Whether to save results to database (default: False)
        aspect_extraction_id: ID of the aspect extraction (needed for DB updates)
        db_update_callback: Callback function to update scores in the database

    Returns:
        Dict mapping aspects to their scores
    """
    logger.info(f"Starting aspect judging for category '{category_name}'...")

    # Initialize prompt manager
    prompt_manager = LlmJudgePromptManager()

    # Judge each aspect with progress bar
    aspect_scores = {}
    for aspect in tqdm(aspect_frequency.keys(), desc="Judging Aspects"):
        # Get prompt
        prompt = prompt_manager.construct_aspect_judge_prompt(
            product=category_name,
            term=aspect
        )

        # Generate judgment
        result = llm.generate(
            messages=[prompt],
            temperature=0.3,
            top_p=0.95,
            max_new_tokens=128
        )
        # Extract score
        score = prompt_manager.process_response(result)
        aspect_scores[aspect] = score

        # Update score in database if save_to_db is True and callback is provided
        if save_to_db and db_update_callback and aspect_extraction_id:
            db_update_callback(
                aspect_extraction_id=aspect_extraction_id,
                aspect=aspect,
                score=score
            )

        time.sleep(sleep_time)  # Sleep to avoid rate limits

    return aspect_scores


def run_ontology_judging(
    llm: Gemini,
    category_name: str,
    ontology_tree: Any,
    save_to_db: bool = False,
    ontology_extraction_id: Optional[int] = None,
    db_update_callback: Optional[Callable] = None,
    sleep_time: int = 3,
) -> Dict[str, int]:
    """
    Run the ontology judging process.

    Args:
        llm: LLM instance for generating judgements
        category_name: Name of the category for context
        ontology_tree: Tree structure of ontology relations
        save_to_db: Whether to save results to database (default: False)
        ontology_extraction_id: ID of the ontology extraction (needed for DB updates)
        db_update_callback: Callback function to update scores in the database

    Returns:
        Dict mapping relations to their scores
    """
    logger.info(f"Starting ontology judging for category '{category_name}'...")

    # Initialize prompt manager
    prompt_manager = LlmJudgePromptManager()

    # Extract relations from ontology tree
    relation_scores = {}
    # Check if ontology_tree is a json string or a dictionary
    if isinstance(ontology_tree, str):
        ontology_dict = json.loads(ontology_tree)
    else:
        ontology_dict = ontology_tree

    def extract_relations(node, parent=None, relations=[]):
        if isinstance(node, dict):
            for child, subtree in node.items():
                if parent is not None:
                    # Get prompt for this relation
                    relations.append((parent, child))

                relations = extract_relations(subtree, child, relations)
        return relations

    # Start recursive extraction from root
    relations = extract_relations(ontology_dict)

    for parent, child in tqdm(relations, desc="Judging Relations"):
        prompt = prompt_manager.construct_relation_judge_prompt(
            category=category_name,
            parent=parent,
            child=child
        )

        # Generate judgment
        result = llm.generate(
            messages=[prompt],
            temperature=0.3,
            top_p=0.95,
            max_new_tokens=128
        )

        # Extract score
        score = prompt_manager.process_response(result)
        relation = f"{parent} -> {child}"
        relation_scores[relation] = score
        logger.info(f"Relation: {relation}, Score: {score}")

        # Sleep to avoid rate limits
        time.sleep(sleep_time)

    # Calculate average score and update database if save_to_db is True and callback is provided
    if relation_scores and save_to_db and db_update_callback and ontology_extraction_id:
        average_score = sum(relation_scores.values()) / len(relation_scores)
        # Update average score in database
        db_update_callback(
            ontology_extraction_id=ontology_extraction_id,
            score=average_score
        )

    return relation_scores
