from llm_backends.base_llm import BaseLLM
from llm_backends.ollama_llm import OllamaLLM
from llm_backends.openai_llm import OpenAILLM
from llm_backends.gemini_llm import GeminiLLM
from llm_backends.local_llm import LocalLLM


class LLMFactory:
    @staticmethod
    def get_llm(llm_type: str, **kwargs) -> BaseLLM:
        if llm_type.lower() == "openai":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI LLM.")
            return OpenAILLM(api_key=api_key, model_name=kwargs.get("model_name", "gpt-3.5-turbo"))
        elif llm_type.lower() == "gemini":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("Gemini API key is required for Gemini LLM.")
            return GeminiLLM(api_key=api_key, model_name=kwargs.get("model_name", "gemini-pro"))
        elif llm_type.lower() == "local":
            model_name_or_path = kwargs.get(
                "model_name_or_path", "distilbert/distilgpt2")
            return LocalLLM(model_name_or_path=model_name_or_path)
        elif llm_type.lower() == "ollama":
            # Pass model_name and optionally base_url to LocalLLM
            # Default to llama2 if not specified
            model_name = kwargs.get("model_name", "llama2:latest")
            # Default Ollama URL
            base_url = kwargs.get("base_url", "http://localhost:11434")
            return OllamaLLM(model_name=model_name, base_url=base_url)
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")
