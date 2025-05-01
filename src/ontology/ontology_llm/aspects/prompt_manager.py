from nltk import pos_tag
from nltk.tokenize import word_tokenize
import re

from src.data.n_shot_examples import sem_eval_nshot_examples_json
from src.base.prompt_manager import BasePromptManager


class SentimentAspectPromptManager(BasePromptManager):
    """
    Prompts LLM to extract aspects from a review and assign a sentiment to each aspect.
    This is a technique that makes aspect extraction more precise.
    """

    def __init__(self):
        # Load json grammar
        with open("src/data/json_grammar/json_aspects_grammar.ebnf", "r") as file:
            self.json_grammar = file.read()

        self.instruction = (f"You are provided with customer reviews of various products from Amazon. "
                            f"Your task is to identify and extract specific aspects of the product "
                            f"mentioned in each review and label the sentiment associated with each aspect. "
                            f"Each aspect refers to a particular feature, attribute, or component of the product. "
                            f"The sentiment can be classified as positive, negative, or neutral. "
                            f"Identify aspects using the exact words used in the review, do not make your own aspects.")

        self.primer = "\n[Start of Examples]\n" + \
            "\n".join(sem_eval_nshot_examples_json) + \
            "\n[End of Examples]\n"

    @staticmethod
    def process_response(review_text, generated_text):
        """Process the generated text to extract aspects from the review text.
            Filter only aspects that are nouns and only keep aspects that have exact matches in the text.
        """
        aspects_and_polarities = re.findall(
            r'"aspect":\s*"(.*?)",\s*"polarity":\s*"(.*?)"', generated_text)
        aspects = [aspect for aspect, _ in aspects_and_polarities]

        tokens = word_tokenize(review_text)
        pos_tags = pos_tag(tokens)

        noun_aspects = []
        for aspect in aspects:
            aspect_terms = aspect.split()
            for i in range(len(tokens) - len(aspect_terms) + 1):
                exact_match = True
                contains_noun = False
                for j in range(len(aspect_terms)):
                    if tokens[i + j] != aspect_terms[j]:
                        exact_match = False
                        break
                    if pos_tags[i + j][1].startswith('NN'):
                        contains_noun = True
                if exact_match and contains_noun:
                    noun_aspects.append(aspect)
                    break

        return noun_aspects

    def get_prompt(self, review_text):
        """Create a prompt that asks the LLM to extract aspects from a review and assign a sentiment to each aspect.
            The prompt includes instructions, examples, and the review text.
        """
        template = f"\n[Start of Review]\n{review_text}[End of Review]"

        prompt = self.instruction + self.primer + template

        return prompt
