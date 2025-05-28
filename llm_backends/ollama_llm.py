import ollama
from llm_backends.base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = "llama3.2:latest", base_url: str = "http://localhost:11434"):
        """
        Initializes the Ollama LLM.

        Args:
            model_name: The name of the Ollama model to use (e.g., "llama2:13b", "llama3.1:latest").
            base_url: The URL where your Ollama server is running.
        """
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)
        # Optional: Check if the model exists and is pulled
        try:
            self.client.show(self.model_name)
            print(f"Ollama model '{self.model_name}' is available.")
        except ollama.ResponseError as e:
            print(
                f"Warning: Ollama model '{self.model_name}' not found locally. Attempting to pull... {e}")
            try:
                self.client.pull(self.model_name)
                print(f"Ollama model '{self.model_name}' pulled successfully.")
            except ollama.ResponseError as pull_e:
                print(
                    f"Error pulling Ollama model '{self.model_name}': {pull_e}")
                raise ValueError(
                    f"Could not initialize Ollama with model '{self.model_name}'. Is Ollama running and is the model name correct?")

    def generate_response(self, prompt: str, history: list[dict]) -> str:
        """
        Generates a response using the specified Ollama model.

        Args:
            prompt: The current user prompt.
            history: A list of previous messages in the conversation.
                     Each message is a dict with "role" and "content" keys.

        Returns:
            The generated response string.
        """
        messages = [{"role": m["role"], "content": m["content"]}
                    for m in history]
        messages.append({"role": "user", "content": prompt})

        # Ollama's chat endpoint handles streaming by default
        response_stream = self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=True  # Ensures we get chunks, which is good for larger responses
        )

        full_response = ""
        for chunk in response_stream:
            if chunk['message']['content']:
                full_response += chunk['message']['content']
        return full_response
