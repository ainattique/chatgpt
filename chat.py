# chat.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
# Import the backend processor you created
from chatgpt import get_chat_response 

st.set_page_config(
    page_title="Aura AI — Your Personal Assistant",
    page_icon="✨",
    layout="wide"
)

# Initialize Session Memory Slots
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_role" not in st.session_state:
    st.session_state.system_role = "You are a helpful, thoughtful assistant."
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-4o-mini"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5

# Sidebar Controls Configuration Window
with st.sidebar:
    st.title("Settings Menu ⚙️")
    st.session_state.model_name = st.selectbox(
        "Select Model Target", 
        ["gpt-4o-mini", "gpt-4o"]
    )
    st.session_state.temperature = st.slider(
        "Creativity Index", 
        0.0, 1.0, float(st.session_state.temperature), 0.1
    )
    st.session_state.system_role = st.text_area(
        "System Prompt Role", 
        value=st.session_state.system_role
    )
    if st.button("Reset Chat Window", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.subheader("Aura Custom ChatGPT 🤖")

# Render historical conversation elements
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Prompt Entry Monitoring Frame
user_prompt = st.chat_input("Send a message to Aura...")
if user_prompt:
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                # Call backend function to generate standard stream chunks
                stream = get_chat_response(
                    messages=st.session_state.messages,
                    model_name=st.session_state.model_name,
                    temperature=st.session_state.temperature,
                    system_role=st.session_state.system_role
                )
                # Stream directly down into the Chat Frame viewport 
                response_content = st.write_stream(
                    chunk.content for chunk in stream if isinstance(chunk.content, str)
                )
            st.session_state.messages.append(AIMessage(content=response_content))
        except Exception as e:
            st.error(f"Operational Exception: {e}")
