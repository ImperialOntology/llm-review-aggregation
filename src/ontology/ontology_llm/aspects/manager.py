from tqdm import tqdm
import json
from pathlib import Path
from src.base.hf_llm_wrapper import HuggingFaceLLMWrapper
from src.ontology.ontology_llm.base.manager_base import ExtractionManager
from src.ontology.ontology_llm.aspects.prompt_manager import SentimentAspectPromptManager
from src.constants import CACHE_DIR


class PromptedAspectExtractionManager(ExtractionManager):
    """Manage aspect extraction by using an llm wrapper and a prompt manager.
    """

    def __init__(self,
                 llm: HuggingFaceLLMWrapper,
                 prompt_manager: SentimentAspectPromptManager,
                 batch_size=32,
                 max_new_tokens=200,
                 temperature=0.001,
                 top_p=0.95,
                 repetition_penalty=1.0,) -> None:
        super().__init__(
            llm=llm,
            prompt_manager=prompt_manager,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

    def extract_aspects_from_reviews(self, reviews: list, output_file=Path(CACHE_DIR)/"aspects.csv"):
        """Extract aspects from a list of reviews.

        Returns: List of lists of aspects extracted from the reviews.

        The aspects are also written to a file in the output_file path.
        """
        messages = [self.prompt_manager.get_prompt(review) for review in reviews]
        all_responses = self._batch_generate(messages, self.prompt_manager.json_grammar)
        all_aspects = [self.prompt_manager.process_response(review, response)
                       for review, response in zip(reviews, all_responses)]

        # cache just in case
        with open(output_file, "w") as f:
            for aspects in tqdm(all_aspects):
                f.write(f"{json.dumps(aspects)}\n")

        return all_aspects
