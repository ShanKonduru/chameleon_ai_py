import google.generativeai as genai
from llm_backends.base_llm import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, prompt: str, history: list[dict]) -> str:
        # Gemini's history format is slightly different
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        convo = self.model.start_chat(history=gemini_history)
        convo.send_message(prompt)
        return convo.last.text
