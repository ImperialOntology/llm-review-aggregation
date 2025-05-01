import time
from typing import List
import google.generativeai as genai
import google.api_core.exceptions

from src.base.llm_wrapper import BaseLLMWrapper
from logger import logger


class Gemini(BaseLLMWrapper):
    def __init__(
        self,
        gemini_api_key,
        model_name="gemini-1.5-flash",
    ) -> None:
        super().__init__()

        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(
        self,
        messages: List[str],
        print_result=False,
        max_new_tokens=128,
        temperature=0.3,
        top_p=0.95,
    ):
        while True:
            try:
                chat = self.model.start_chat(history=[])

                # Build message history with retry logic
                for i, message in enumerate(messages[:-1]):  # Process all messages except the last one
                    if i % 2 == 0:  # User message
                        while True:
                            try:
                                chat.send_message(message)
                                break
                            except google.api_core.exceptions.ResourceExhausted as e:
                                logger.warning(f"Error during history building: {e}")
                                logger.warning("⚠️ API rate limit exceeded. Retrying after 10 seconds...")
                                time.sleep(10)
                    else:  # Model message (adding to history)
                        chat._history.append({"role": "model", "parts": [{"text": message}]})

                # Send final message with retry logic
                last_message = messages[-1] if messages else "Hello, can you help me?"
                response = chat.send_message(
                    last_message,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                )
                result = response.text
                break  # Exit the main loop if everything succeeded

            except google.api_core.exceptions.ResourceExhausted as e:
                logger.warning(f"Error: {e}")
                logger.warning("⚠️ API rate limit exceeded. Retrying after 10 seconds...")
                time.sleep(10)  # Wait before retrying
            except google.api_core.exceptions.InvalidArgument as e:
                logger.error(f"🚨 Invalid argument error: {e}")
                raise e  # Raise invalid argument errors as they won't be fixed by retrying
            except google.api_core.exceptions.GoogleAPICallError as e:
                logger.warning(f"⚠️ API call error: {e}")
                time.sleep(5)  # Wait before retrying
            except Exception as e:
                logger.warning(f"🚨 Unexpected error: {e}")
                time.sleep(10)  # Wait before retrying

        if print_result:
            print(result, flush=True)

        return result
