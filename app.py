import os
import uuid
import docx
import openpyxl
import pandas as pd
import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page Configuration & Custom Theme Styling
st.set_page_config(
    page_title="Aura AI — Production RAG Co-Pilot",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://googleapis.com');
:root {
  --ink: #292347; --purple: #7657e8; --line: rgba(120, 102, 190, .17);
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background: radial-gradient(ellipse at 10% 0%, rgba(228,91,183,.1), transparent 33%),
              radial-gradient(ellipse at 95% 8%, rgba(84,169,245,.1), transparent 30%),
              linear-gradient(135deg, #fbfaff 0%, #f6f3ff 48%, #f7fbff 100%);
  color: var(--ink);
}
.brand { display:flex; align-items:center; gap:12px; margin: 4px 0 26px; }
.brand-icon {
  display:flex; align-items:center; justify-content:center; width:44px; height:44px;
  border-radius:14px; font-size:22px; color:white;
  background:linear-gradient(135deg,#7757eb,#e15ab6 62%,#68b8fa);
}
.brand-name { font-family:'Space Grotesk',sans-serif; font-size:21px; font-weight:700; color:#30264e; }
.chat-heading { display:flex; align-items:center; gap:12px; margin: 4px 0 22px; }
.chat-heading .mini-orb { width:40px; height:40px; display:flex; align-items:center; justify-content:center; border-radius:12px; background:linear-gradient(140deg,#e8ddff,#ffe1f4,#dff2ff); font-size:20px; }
.chat-heading h2 { font-family:'Space Grotesk',sans-serif; font-size:22px; margin:0; color:#30264e; }
div[data-testid="stChatMessage"] { background:rgba(255,255,255,.75); border:1px solid rgba(150,126,215,.14); border-radius:16px; box-shadow:0 4px 12px rgba(70,47,130,.02); }
</style>
""", unsafe_allow_html=True)

# 2. Environment Initialization
load_dotenv(find_dotenv(), override=True)
api_key = os.getenv("OPENAI_API_KEY", "")

# 3. Session State Management
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "system_role" not in st.session_state:
    st.session_state.system_role = "You are a helpful assistant. Use the provided context to answer accurately. If you don't know, state it clearly."
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-4o-mini"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.4

def init_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Session",
        "messages": [],
        "vectorstore": None
    }
    st.session_state.active_chat = chat_id

if not st.session_state.active_chat or st.session_state.active_chat not in st.session_state.chats:
    init_new_chat()

current_chat = st.session_state.chats[st.session_state.active_chat]

# 4. Multi-Modal Document File Extractors
def extract_text_from_pdf(file_bytes):
    pdf = PdfReader(BytesIO(file_bytes))
    return "".join([page.extract_text() or "" for page in pdf.pages])

def extract_text_from_docx(file_bytes):
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_xlsx(file_bytes):
    wb = openpyxl.load_workbook(BytesIO(file_bytes), data_only=True)
    text_data = []
    for sheet in wb.sheetnames:
        text_data.append(f"--- Sheet: {sheet} ---")
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
        text_data.append(df.to_string(index=False))
    return "\n".join(text_data)

# 5. Core Pipeline RAG Processing Engine
def build_vector_index(uploaded_files):
    combined_raw_text = ""
    for file in uploaded_files:
        bytes_data = file.read()
        if file.name.endswith(".pdf"):
            combined_raw_text += f"\n[Document: {file.name}]\n" + extract_text_from_pdf(bytes_data)
        elif file.name.endswith(".docx"):
            combined_raw_text += f"\n[Document: {file.name}]\n" + extract_text_from_docx(bytes_data)
        elif file.name.endswith((".xlsx", ".xls")):
            combined_raw_text += f"\n[Document: {file.name}]\n" + extract_text_from_xlsx(bytes_data)
            
    if not combined_raw_text.strip():
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(combined_raw_text)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# 6. Sidebar Application Configurations
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">✨</div><div><div class="brand-name">Aura AI</div></div></div>', unsafe_allow_html=True)
    
    if st.button("＋ New chat", use_container_width=True):
        init_new_chat()
        st.rerun()
        
    st.markdown("##### Conversations")
    for cid, data in list(st.session_state.chats.items())[::-1]:
        btn_label = ("● " if cid == st.session_state.active_chat else "○ ") + data["title"][:25]
        if st.button(btn_label, key=f"nav_{cid}", use_container_width=True):
            st.session_state.active_chat = cid
            st.rerun()
            
    st.divider()
    st.markdown("##### Knowledge Documents")
    uploaded_docs = st.file_uploader("Upload contextual files", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)
    if uploaded_docs:
        if st.button("Index Documents", use_container_width=True):
            with st.spinner("Building vector memory store..."):
                current_chat["vectorstore"] = build_vector_index(uploaded_docs)
                st.success("Indexing complete!")
                
    st.divider()
    st.markdown("##### Hyperparameters")
    st.session_state.model_name = st.selectbox("LLM Model Target", ["gpt-4o-mini", "gpt-4o"])
    st.session_state.temperature = st.slider("Generation Temp", 0.0, 1.0, float(st.session_state.temperature), 0.1)

# 7. Main Chat UI Window Frame Interface
st.markdown('<div class="chat-heading"><div class="mini-orb">🤖</div><div><h2>Aura AI Engine</h2></div></div>', unsafe_allow_html=True)

for msg in current_chat["messages"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        st.markdown(msg.content)

user_query = st.chat_input("Message Aura AI...")
if user_query:
    current_chat["messages"].append(HumanMessage(content=user_query))
    if current_chat["title"] == "New Session":
        current_chat["title"] = user_query.strip()[:25]
        
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)
        
    with st.chat_message("assistant", avatar="🤖"):
        context = ""
        if current_chat["vectorstore"] is not None:
            retriever = current_chat["vectorstore"].as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(user_query)
            context = "\n\n".join([d.page_content for d in docs])
            
        system_instructions = st.session_state.system_role
        if context:
            system_instructions += f"\n\nContext information to strictly use:\n{context}"
            
        llm_messages = [SystemMessage(content=system_instructions)] + current_chat["messages"]
        
        try:
            model_instance = ChatOpenAI(
                model=st.session_state.model_name,
                temperature=st.session_state.temperature,
                openai_api_key=api_key,
                streaming=True
            )
            response_stream = st.write_stream(
                chunk.content for chunk in model_instance.stream(llm_messages)
                if isinstance(chunk.content, str)
            )
            current_chat["messages"].append(AIMessage(content=response_stream))
        except Exception as err:
            st.error(f"Execution Exception encountered: {str(err)}")
