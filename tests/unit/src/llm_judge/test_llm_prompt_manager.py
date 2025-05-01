from unittest.mock import patch

from src.data.llm_judge_prompts import ASPECT_JUDGE_PROMPT, RELATION_JUDGE_PROMPT
from src.base.prompt_manager import BasePromptManager
from src.llm_judge.prompt_manager import LlmJudgePromptManager


class TestLlmJudgePromptManager:
    def setup_method(self):
        self.prompt_manager = LlmJudgePromptManager()
        
        # Sample test data
        self.product = "smartphone"
        self.term = "battery life"
        self.category = "electronics"
        self.parent = "hardware"
        self.child = "processor"
    
    def test_process_response_with_valid_score(self):
        # Test different score format variations
        test_cases = [
            "The evaluation is complete. Score: 4",
            "After analyzing, I give this a Score: 5",
            "Score** : 3",
            "Score: **2**",
            "Score:**[[1]]",
        ]
        
        expected_scores = [4, 5, 3, 2, 1]
        
        for text, expected in zip(test_cases, expected_scores):
            result = self.prompt_manager.process_response(text)
            assert result == expected
    
    def test_process_response_with_invalid_format(self):
        test_cases = [
            "No score provided",
            "The evaluation is Score: high",
            "Score: excellent",
            "",
        ]
        
        for text in test_cases:
            result = self.prompt_manager.process_response(text)
            assert result == 0
    
    def test_construct_aspect_judge_prompt(self):
        expected_prompt = ASPECT_JUDGE_PROMPT.format(
            product=self.product,
            term=self.term
        )
        
        result = self.prompt_manager.construct_aspect_judge_prompt(
            product=self.product,
            term=self.term
        )
        
        assert result == expected_prompt
        assert self.product in result
        assert self.term in result
    
    def test_construct_relation_judge_prompt(self):
        expected_prompt = RELATION_JUDGE_PROMPT.format(
            category=self.category,
            parent=self.parent,
            child=self.child
        )
        
        result = self.prompt_manager.construct_relation_judge_prompt(
            category=self.category,
            parent=self.parent,
            child=self.child
        )
        
        assert result == expected_prompt
        assert self.category in result
        assert self.parent in result
        assert self.child in result
    
    def test_inheritance(self):
        assert isinstance(self.prompt_manager, BasePromptManager)
    
    @patch('re.findall')
    def test_process_response_exception_handling(self, mock_findall):
        # Setup mock to raise an exception
        mock_findall.return_value = []
        
        result = self.prompt_manager.process_response("Some response text")
        assert result == 0
        
        # Test with a list that doesn't contain a valid integer
        mock_findall.return_value = ["not_a_number"]
        result = self.prompt_manager.process_response("Score: not_a_number")
        assert result == 0