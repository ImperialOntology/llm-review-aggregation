from abc import ABC, abstractmethod
from typing import List

class BaseLLMWrapper(ABC):
    """
    An "interface" for various LLM manager objects.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[str],
        grammar=None,
        print_result=False,
        seed=42,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.0,
        trim_response=True,
    ):
        pass
