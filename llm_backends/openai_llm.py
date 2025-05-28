import openai
from llm_backends.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str, history: list[dict]) -> str:
        messages = [{"role": m["role"], "content": m["content"]}
                    for m in history]
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        return full_response
