import re
from src.base.prompt_manager import BasePromptManager
from src.data.n_shot_examples import concept_examples


class ConceptPromptManager(BasePromptManager):
    """
    Prompts LLM to determine whether a candidate aspect term should be included in a product aspect ontology as a concept.
    """

    def __init__(self):
        # Load json grammar
        with open("src/data/json_grammar/json_concept_grammar.ebnf", "r") as file:
            self.json_grammar = file.read()
        self.instruction = (f"You are provided with a list of candidate aspect terms related to a specific product "
                            f"in an e-commerce context. The goal is to determine whether each term should be included "
                            f"as part of a product aspect ontology. A product aspect ontology consists of a root entity, "
                            f"which is the product itself, and various aspects that represent features/sub-features or "
                            f"components/sub-components of the product.\n"
                            f"For each candidate term, evaluate its relevance and appropriateness for inclusion in the ontology. "
                            f"Consider the following guidelines:\n"
                            f"Relevance: The term must directly relate to a specific feature, sub-feature, component, or sub-component of the product.\n"
                            f"Specificity: The term should not be overly broad or overly narrow. It must clearly identify a distinct aspect of the product, but able to generalise across multiple products.\n"
                            f"Hierarchy: Consider whether the term represents a primary feature or a more granular sub-feature/component.\n"
                            f"For each candidate aspect term, respond with \"yes\" if the term should be included in the product aspect ontology "
                            f"and \"no\" if it should not be included. Additionally, provide a brief explanation for your decision.")

        self.primer = "\n[Start of Examples]\n" + \
            "\n".join(concept_examples) + "\n[End of Examples]\n"

    @staticmethod
    def process_response(generated_text):
        """Process the generated text to extract whether the answer is yes or no.
        """
        answer = re.findall(r'"answer":\s*"(.*?)"', generated_text)
        if answer:
            return answer[0] == "yes"
        return False

    def get_prompt(self, product, aspect):
        """Create a prompt that asks the LLM to answer a yes/no question of 
        whether the aspect should be included in the ontology as a concept or no.
        """
        template = f"Product: {product}\nCandidate Aspect: {aspect}"

        prompt = self.instruction + self.primer + template

        return prompt
