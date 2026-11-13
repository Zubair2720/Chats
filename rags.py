
import streamlit as st
import os
import tempfile
import uuid

from dotenv import load_dotenv, find_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------
# 1. Configuration
# ---------------------------------

load_dotenv(find_dotenv(), override=True)

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# API KEY
API_KEY = (
    os.getenv("Gemini_API_Key")
    or os.getenv("GEMINI_API_KEY")
)

if not API_KEY:
    st.error("Gemini API Key not found. Please check your .env file.")
    st.stop()

# Gemini Client
client = genai.Client(api_key=API_KEY)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

# ---------------------------------
# 2. Session State
# ---------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ---------------------------------
# 3. Sidebar
# ---------------------------------

with st.sidebar:

    st.title("📚 PDF RAG Assistant")

    st.write("Upload your PDF and ask questions.")

    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"]
    )

    process_button = st.button(
        "Process PDF",
        use_container_width=True
    )

    st.divider()

    st.subheader("About")

    st.write("""
    This application uses:

    - Google Gemini AI
    - LangChain
    - Chroma Vector Database
    - Google Embeddings
    - Retrieval Augmented Generation
    """)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------
# 4. PDF Processing
# ---------------------------------

if process_button:

    if uploaded_file is None:
        st.warning("Please upload a PDF first.")

    else:

        with st.spinner("Processing your PDF..."):

            temp_path = None

            try:

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                # Load PDF
                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                # Text Splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                texts = text_splitter.split_documents(docs)

                if not texts:
                    raise ValueError("No readable text found in this PDF.")

                # Create Chroma Vector Database
                vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    collection_name=f"pdf_{uuid.uuid4().hex[:12]}",
                    persist_directory=None
                )

                # Store in Session
                st.session_state.vector_store = vector_store
                st.session_state.chat_history = []
                st.session_state.pdf_name = uploaded_file.name

                st.success("PDF processed successfully!")

                st.info(
                    f"Total pages: {len(docs)} | "
                    f"Total chunks: {len(texts)}"
                )

            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

# ---------------------------------
# 5. Main Frontend
# ---------------------------------

st.title("🤖 AI PDF Chat Assistant")

st.caption(
    "Ask questions from your PDF using Google Gemini and RAG."
)

if st.session_state.pdf_name:

    st.success(
        f"Active Document: {st.session_state.pdf_name}"
    )

else:

    st.info("👈 Upload a PDF from the sidebar to get started.")

st.divider()

# ---------------------------------
# 6. Display Chat History
# ---------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(source)

# ---------------------------------
# 7. User Question
# ---------------------------------

question = st.chat_input(
    "Ask something about your PDF..."
)

if question:

    if st.session_state.vector_store is None:

        st.warning("Please upload and process a PDF first.")

    else:

        # Display user question
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve Relevant Documents
        with st.chat_message("assistant"):

            with st.spinner("Searching your document..."):

                try:

                    results = (
                        st.session_state.vector_store
                        .similarity_search(question, k=3)
                    )

                    context = "\n\n".join(
                        doc.page_content for doc in results
                    )

                    # RAG Prompt
                    rag_prompt = ChatPromptTemplate.from_template("""
                    You are a helpful AI assistant.

                    Answer the question using only the
                    following context.

                    If the answer is not available in
                    the context, say:

                    "I could not find this information
                    in the document."

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
                    """)

                    prompt = rag_prompt.invoke({
                        "context": context,
                        "question": question
                    }).to_string()

                    # Gemini Response
                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt
                    )

                    answer = response.text or "No answer generated."

                    st.markdown(answer)

                    # Source Information
                    sources = []

                    for doc in results:

                        page = doc.metadata.get("page", "Unknown")

                        sources.append(
                            f"Page {page + 1 if isinstance(page, int) else page}"
                        )

                    with st.expander("📄 View Sources"):
                        for source in sources:
                            st.write(source)

                    # Save Assistant Response
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    st.error(f"Something went wrong: {str(e)}")

# ---------------------------------
# 8. Footer
# ---------------------------------

st.divider()

st.caption(
    "Built with Streamlit | LangChain | Chroma | Google Gemini"
)
