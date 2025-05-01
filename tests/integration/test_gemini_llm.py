import pytest
from src.llm_judge.llm_manager import Gemini

# user input api key
gemini_api_key = input("Enter your Gemini API key: ").strip()


@pytest.mark.integtest
def test_gemini_llm():
    # Initialize LLM
    llm = Gemini(gemini_api_key=gemini_api_key)

    # Test simple generation
    result = llm.generate(messages=["Hello"], print_result=True)

    # Assertions
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

    # Print results for inspection
    print(f"Generated response: {result}")
    print("===" * 10)


if __name__ == "__main__":
    test_gemini_llm()
