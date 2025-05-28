from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, history: list[dict]) -> str:
        """
        Generates a response from the LLM.

        Args:
            prompt: The current user prompt.
            history: A list of previous messages in the conversation.
                     Each message is a dict with "role" and "content" keys.

        Returns:
            The generated response string.
        """
        pass
