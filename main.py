import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tempfile

import streamlit as st
from streamlit_lottie import st_lottie
import json


import asyncio
import nest_asyncio
nest_asyncio.apply()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

    

# ----------------------------
# Load API keys
load_dotenv()

# Init LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             temperature=0.1,
                             max_tokens=1000)

st.set_page_config(page_title="📊 Chat With Pdf's - (Finance Domain)", layout="wide")
# st.title("📊 Finance Management Expert")
# st.write("Upload a finance/management PDF and ask questions about it.")
# st.write("⚠️ Prototype limitation: Please upload PDF <= 20 pages.")

# --------------------------------------------------------------------------------------------

# Load Lottie file
def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Adjust columns: more balanced ratio
col1, col2, col3 = st.columns([3, 1, 1])  # Middle column (col2) for the animation

with col1:
    st.title("📊 Chat With Pdf's")
    st.write("Upload a finance/management PDF and ask questions about it.")
    st.write("⚠️ Prototype limitation: Please upload PDF <= 20 pages.")

with col2:
    lottie_animation = load_lottie("assets/Book_Reading.json")
    st_lottie(lottie_animation, key="finance", height=300, width=300, speed=0.3)

# ----------------------------
# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    docs = text_splitter.split_documents(data)

    # Create embeddings + vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 8}
                                )

    # System Prompt
    system_prompt = (
        "You are an expert financial assistant. "
        "Use the retrieved PDF text to answer the user’s query. "
        "Keep your answer as close to the original wording as possible. "
        "Always provide the *complete explanation* from the retrieved PDF." 
        "If a section introduces a list (e.g., aspects, differences, advantages), ensure you include all listed items."
        "Do not add external examples or details unless they are explicitly mentioned in the PDF. "
        "If the content is limited, provide a concise summary instead of elaborating.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # ----------------------------
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Chat input
    user_input = st.chat_input("Ask a question about the PDF...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])  # ✅ Supports bullet points, bold, etc