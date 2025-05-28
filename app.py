import streamlit as st
from llm_backends.llm_factory import LLMFactory
from llm_backends.base_llm import BaseLLM
import ollama # Import ollama to potentially list local models dynamically

st.set_page_config(page_title="Chameleon AI Chatbot")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_type" not in st.session_state:
    st.session_state.llm_type = "OpenAI"  # Default LLM

# --- Sidebar for LLM Selection and Configuration ---
st.sidebar.title("LLM Settings")
llm_options = ["OpenAI", "Gemini", "Local", "Ollama"]
selected_llm_type = st.sidebar.selectbox(
    "Choose LLM Backend:", llm_options, key="llm_selector")

# Update LLM type in session state if changed
if selected_llm_type != st.session_state.llm_type:
    st.session_state.llm_type = selected_llm_type
    st.session_state.messages = [] # Clear chat history on LLM change
    st.rerun() # <--- CHANGE THIS LINE FROM st.experimental_rerun()
    

api_key_placeholder = {}
ollama_base_url = "http://localhost:11434" # Default Ollama URL

if st.session_state.llm_type == "OpenAI":
    api_key_placeholder["openai"] = st.sidebar.text_input("OpenAI API Key", type="password",
                                                          value=st.secrets.get("OPENAI_API_KEY", ""))
    model_name = st.sidebar.text_input(
        "OpenAI Model Name", value="gpt-3.5-turbo")
elif st.session_state.llm_type == "Gemini":
    api_key_placeholder["gemini"] = st.sidebar.text_input("Gemini API Key", type="password",
                                                          value=st.secrets.get("GEMINI_API_KEY", ""))
    model_name = st.sidebar.text_input("Gemini Model Name", value="gemini-pro")
elif st.session_state.llm_type == "Local":
    model_name = st.sidebar.text_input(
        "Local Model Path/Name (Hugging Face)", value="distilbert/distilgpt2")
elif st.session_state.llm_type == "Ollama":
    ollama_base_url = st.sidebar.text_input("Ollama Server URL", value="http://localhost:11434")
    # Dynamically fetch available Ollama models
    available_ollama_models = []
    try:
        ollama_client = ollama.Client(host=ollama_base_url)
        models_info = ollama_client.list()['models']
        print(models_info)
        available_ollama_models = [m['model'] for m in models_info]
    except ollama.ResponseError:
        st.sidebar.warning(f"Could not connect to Ollama server at {ollama_base_url}. Is Ollama running?")
    except Exception as e:
        st.sidebar.warning(f"Error listing Ollama models: {e}")
    
    if available_ollama_models:
        model_name = st.sidebar.selectbox(
            "Select Ollama Model:", 
            options=available_ollama_models, 
            key="ollama_model_selector"
        )
    else:
        model_name = st.sidebar.text_input(
            "Ollama Model Name (e.g., llama2:latest)", 
            value="llama3.2:latest" # Fallback if no models are listed
        )
        
# --- LLM Instance Creation (cached) ---


# Don't hash the LLM object itself
# @st.cache_resource(hash_funcs={BaseLLM: lambda _: None})
def get_llm_instance(llm_type: str, **kwargs) -> BaseLLM:
    try:
        return LLMFactory.get_llm(llm_type, **kwargs)
    except ValueError as e:
        st.error(f"Error initializing LLM: {e}")
        return None


llm_instance = None
if st.session_state.llm_type == "OpenAI":
    if api_key_placeholder.get("openai"):
        llm_instance = get_llm_instance(
            "openai", api_key=api_key_placeholder["openai"], model_name=model_name)
    else:
        st.warning("Please enter your OpenAI API key in the sidebar.")
elif st.session_state.llm_type == "Gemini":
    if api_key_placeholder.get("gemini"):
        llm_instance = get_llm_instance(
            "gemini", api_key=api_key_placeholder["gemini"], model_name=model_name)
    else:
        st.warning("Please enter your Gemini API key in the sidebar.")
elif st.session_state.llm_type == "Local":
    llm_instance = get_llm_instance("local", model_name_or_path=model_name)
elif st.session_state.llm_type == "Ollama":
    llm_instance = get_llm_instance("ollama", model_name_or_path=model_name, base_url=ollama_base_url)


st.title("Chameleon AI Chatbot")

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Chat Input and Response Generation ---
if prompt := st.chat_input("Say something..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if llm_instance:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm_instance.generate_response(
                    prompt, st.session_state.messages)
                st.write(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            st.warning(
                "LLM not initialized. Please configure API keys or check local model path.")
