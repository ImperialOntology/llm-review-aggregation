import re
import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from typing import List
from src.constants import CACHE_DIR
from src.base.llm_wrapper import BaseLLMWrapper


class HuggingFaceLLMWrapper(BaseLLMWrapper):
    """
    A wrapper for Hugging Face's language models, providing functionality for model initialization,
    tokenization, and text generation with optional grammar constraints.

    Attributes:
        device (str): The device on which the model is loaded (e.g., "cuda:0" or "cpu").
        model: The Hugging Face causal language model.
        tokenizer: The tokenizer associated with the model.
    """

    def __init__(
        self,
        model_name,
        device="cuda:0",
        cache_dir=CACHE_DIR,
        quantization="4bit",
    ) -> None:
        """
        Initialize the HuggingFaceLLMWrapper.

        Args:
            model_name (str): The name of the Hugging Face model to load.
            device (str): The device to load the model on (default: "cuda:0").
            cache_dir (str): The directory to cache the model and tokenizer (default: CACHE_DIR).
            quantization (str): The quantization type ("4bit", "8bit", or None) for model optimization.
        """
        super().__init__()

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            device_map={"": self.device},  # Force model to be on 1 GPU
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Optimize for batch generation

    def generate(
        self,
        messages: List[str],
        grammar=None,
        seed=42,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.0,
        trim_response=True,
    ):
        """
        Generate responses for a batch of input messages using the language model.

        Args:
            messages (List[str]): A list of input messages to generate responses for.
            grammar (optional): A grammar constraint to apply during generation.
            seed (int): The random seed for reproducibility (default: 42).
            max_new_tokens (int): The maximum number of new tokens to generate (default: 128).
            do_sample (bool): Whether to sample during generation (default: True).
            temperature (float): Sampling temperature for controlling randomness (default: 0.3).
            top_p (float): Nucleus sampling probability (default: 0.95).
            repetition_penalty (float): Penalty for repeated tokens (default: 1.0).
            trim_response (bool): Whether to trim special tokens from the response (default: True).

        Returns:
            List[str]: A list of generated responses corresponding to the input messages.
        """
        transformers.set_seed(seed)

        # Batch preparation
        formatted_prompts = []
        for message in messages:
            # Format each message as a single user instruction
            conv = [{"role": "user", "content": message}]
            formatted = self.tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=True,
                tokenize=False
            )
            formatted_prompts.append(formatted)

        # Batch tokenization with padding
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,  # Chat template already includes special tokens
        )

        # Prepare grammar constraints if needed
        grammar_processor = None
        if grammar:
            grammar_constraint = IncrementalGrammarConstraint(
                grammar, "root", self.tokenizer)
            grammar_processor = [GrammarConstrainedLogitsProcessor(grammar_constraint)]

        # Move inputs to device
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Batch generation
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            logits_processor=grammar_processor,
        )

        # Batch decoding
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        # Post-processing
        if trim_response:
            pattern = re.compile(r'\[INST\].*?\[/INST\]|<s>|</s>', re.DOTALL)
            responses = [
                re.sub(pattern, '', resp).strip()
                for resp in responses
            ]

        return responses
