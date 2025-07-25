import re
from abc import ABC
from src.data.llm_judge_prompts import ASPECT_JUDGE_PROMPT, RELATION_JUDGE_PROMPT
from src.base.prompt_manager import BasePromptManager


class LlmJudgePromptManager(BasePromptManager):
    """
    Prompt manager for LLM judge tasks that evaluate aspects and relations.
    Provides methods to construct prompts for judging aspects and relations,
    and process the responses to extract scores.
    """

    @staticmethod
    def process_response(generated_text):
        """
        Process the generated text to extract the score.
        
        Args:
            generated_text (str): The text generated by the LLM
            
        Returns:
            int: The extracted score (0 if extraction fails)
        """
        generated_score = re.findall(r'Score\**\s*:\**\s*\**(?:\[\[)?(\d+)(?:\]\])?', generated_text)
        try:
            score = int(generated_score[0])
        except Exception:
            score = 0
        return score

    def construct_aspect_judge_prompt(self, product, term):
        """
        Construct a prompt for judging if a term is an appropriate aspect of a product.
        
        Args:
            product (str): The product name
            term (str): The aspect term to evaluate
            
        Returns:
            str: The formatted prompt
        """
        formatted_prompt = ASPECT_JUDGE_PROMPT.format(
            product=product,
            term=term
        )
        
        return formatted_prompt
        
    def construct_relation_judge_prompt(self, category, parent, child):
        """
        Construct a prompt for judging if a child-parent relation is appropriate.
        
        Args:
            category (str): The product category
            parent (str): The parent node
            child (str): The child node
            
        Returns:
            str: The formatted prompt
        """
        formatted_prompt = RELATION_JUDGE_PROMPT.format(
            category=category,
            parent=parent,
            child=child
        )
        
        return formatted_prompt
