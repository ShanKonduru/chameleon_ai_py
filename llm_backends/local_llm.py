from transformers import pipeline
from llm_backends.base_llm import BaseLLM


class LocalLLM(BaseLLM):
    def __init__(self, model_name_or_path: str = "distilbert/distilgpt2"):
        # You might need to download the model, adjust for larger models
        self.generator = pipeline("text-generation", model=model_name_or_path)

    def generate_response(self, prompt: str, history: list[dict]) -> str:
        # For simple local models, history handling might be limited
        # You'd need more sophisticated prompt engineering for true conversational context
        full_prompt = "\n".join(
            [f"{m['role']}: {m['content']}" for m in history]) + f"\nuser: {prompt}\nassistant:"

        # Adjust max_new_tokens for desired response length
        response = self.generator(
            full_prompt, max_new_tokens=50, num_return_sequences=1)

        # Extract only the newly generated text, not the input prompt
        generated_text = response[0]['generated_text']

        # Remove the input prompt part from the generated text
        if generated_text.startswith(full_prompt):
            return generated_text[len(full_prompt):].strip()
        return generated_text.strip()
