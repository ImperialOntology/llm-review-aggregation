import re
from src.base.prompt_manager import BasePromptManager
from src.data.n_shot_examples import selected_context_examples


class RelationPromptManager(BasePromptManager):
    """
    Prompt manager for the relation extraction task.
    Generates prompts for the relation extraction task and processes the model output.
    """

    def __init__(self):
        # Load json grammar
        with open("src/data/json_grammar/json_relations_grammar.ebnf", "r") as file:
            self.json_grammar = file.read()

        self.instruction = (f"You are provided with a sentence and two aspects extracted from the sentence. "
                            f"Your task is to determine if there is a meronym (part-whole) relationship between "
                            f"these two aspects. A meronym relationship exists when one aspect is a part of "
                            f"another aspect. Use common sense and the sentence as context for the identified "
                            f"relationship, if any.")
        self.primer = "\n[Start of Examples]\n" + \
            "\n".join(selected_context_examples) + "\n[End of Examples]\n"

    @ staticmethod
    def process_response(generated_text, aspect1, aspect2):
        """
        Process the generated text to determine if the first aspect term is a child of the second aspect term.

        Returns:
        - is_first_aspect_child: Boolean indicating if the first aspect term is a child of the second aspect term.
        - score: Confidence score associated with the prediction. Currently constant and set to 1.
        """
        meronymn = re.findall(
            r'"part":\s*"(.*?)",\s*"whole":\s*"(.*?)"', generated_text)

        if meronymn:
            part, whole = meronymn[0]
            # part, whole, score = meronymn[0]
            if (part, whole) == (aspect1, aspect2):
                is_first_aspect_child = True
                return is_first_aspect_child, float(1)
            elif (part, whole) == (aspect2, aspect1):
                is_first_aspect_child = False
                return is_first_aspect_child, float(1)
        return None

    def get_prompt(self, sentence, aspect1, aspect2):
        """Create a prompt that asks the LLM to identify if
        there is a meronym (part-whole) relationship between two aspects in a sentence.

        Returns:
        - prompt: Prompt to be fed to the LLM.
        - json_grammar: JSON grammar to be used for the LLM.
        """
        template = f'Sentence: "{sentence}"\nAspect1: "{
            aspect1}"\nAspect2: "{aspect2}"'

        prompt = self.instruction + self.primer + template

        json_grammar = (self.json_grammar
                        .replace("ASPECT1_PLACEHOLDER", f"{aspect1}")
                        .replace("ASPECT2_PLACEHOLDER", f"{aspect2}"))

        return prompt, json_grammar
