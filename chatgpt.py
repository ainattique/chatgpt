# chatgpt.py
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Load environment keys
load_dotenv(find_dotenv(), override=True)
api_key = os.getenv("OPENAI_API_KEY", "")

def get_chat_response(messages, model_name="gpt-4o-mini", temperature=0.5, system_role=None):
    """
    Orchestrates the LangChain ChatOpenAI stream initialization.
    """
    # Initialize the model instance with streaming enabled
    chat_model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        streaming=True
    )
    
    # Prepend system context if provided
    full_messages = []
    if system_role:
        full_messages.append(SystemMessage(content=system_role))
    
    full_messages.extend(messages)
    
    # Return the stream object to be consumed by the UI
    return chat_model.stream(full_messages)
