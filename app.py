import tempfile
import os 

from pathlib import Path
from typing import List, Optional

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma

from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

st.markdown("""
<style>
section[data-testid="stSidebar"] {display: none !important;}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
def get_api_key() -> str:
    # Streamlit Cloud uses st.secrets; local can also use secrets.toml
    key = None
    if "OPENAI_API_KEY" in st.secrets:
        key = st.secrets["OPENAI_API_KEY"]
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to Streamlit Secrets or env var.")
    os.environ["OPENAI_API_KEY"] = key
    return key


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def load_documents(file_path: str):
    lower = file_path.lower()
    if lower.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    if lower.endswith(".txt"):
        return TextLoader(file_path, encoding="utf-8").load()
    raise ValueError("Unsupported file type. Upload a .pdf or .txt file.")


def split_documents(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def format_docs(docs: List) -> str:
    if not docs:
        return ""
    return "\n\n".join([f"[Source {i+1}] {d.page_content}" for i, d in enumerate(docs)])


def build_vectorstore(
    chunks,
    db_type: str,
    persist_dir: str,
    embeddings,
    rebuild: bool = False,
):
    os.makedirs(persist_dir, exist_ok=True)

    if db_type == "faiss":
        # FAISS persistence uses index.faiss + index.pkl
        faiss_exists = (
            os.path.exists(os.path.join(persist_dir, "index.faiss"))
            or os.path.exists(os.path.join(persist_dir, "index.pkl"))
        )
        if faiss_exists and not rebuild:
            return FAISS.load_local(
                persist_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(persist_dir)
        return vs

    if db_type == "chroma":
        chroma_exists = os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))
        if chroma_exists and not rebuild:
            return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

        vs = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        try:
            vs.persist()
        except Exception:
            pass
        return vs

    raise ValueError("db_type must be 'faiss' or 'chroma'.")


def build_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a document QA assistant. Use ONLY the provided context. "
                "If the answer is not in the context, say: 'I don't know based on the document.'",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
        ]
    )

    format_docs_runnable = RunnableLambda(format_docs)

    # Explicit RunnableSequence (also valid to use pipe '|', which creates a sequence internally)
    chain = RunnableSequence(
        {
            "context": retriever | format_docs_runnable,
            "question": RunnablePassthrough(),
        },
        prompt,
        llm,
        StrOutputParser(),
    )
    return chain


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG Document QA (Q2→Q4)", layout="wide")
st.title("Deployable RAG Document Q/A (LangChain Runnables)")

with st.sidebar:
    st.header("Settings")
    db_type = st.selectbox("Vector DB", ["faiss", "chroma"], index=0)
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
    top_k = st.slider("Retriever top-k", 1, 10, 4, 1)
    rebuild_db = st.checkbox("Rebuild vector DB", value=True)

    llm_model = st.text_input("LLM model", value="gpt-4o-mini")
    embed_model = st.text_input("Embedding model", value="text-embedding-3-small")

st.subheader("1) Upload document")
uploaded = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])

if uploaded:
    try:
        get_api_key()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Persist directory (note: Streamlit Community Cloud filesystem is ephemeral)
    persist_dir = "./vector_db"

    if "rag_chain" not in st.session_state or rebuild_db or st.button("Build / Rebuild Index"):
        with st.spinner("Building vector database..."):
            file_path = save_uploaded_file(uploaded)
            docs = load_documents(file_path)
            chunks = split_documents(docs, chunk_size, chunk_overlap)

            embeddings = OpenAIEmbeddings(model=embed_model)
            vectorstore = build_vectorstore(
                chunks=chunks,
                db_type=db_type,
                persist_dir=persist_dir,
                embeddings=embeddings,
                rebuild=rebuild_db,
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
            llm = ChatOpenAI(model=llm_model, temperature=0)

            st.session_state.rag_chain = build_rag_chain(retriever, llm)
            st.session_state.db_type = db_type
            st.session_state.persist_dir = persist_dir

        st.success(f"Vector DB ready ({db_type}) at {persist_dir}")

    st.subheader("2) Ask a question")
    q = st.text_input("Your question", placeholder="e.g., Summaries the above documment.")
    if st.button("Answer") and q.strip():
        with st.spinner("Retrieving context and generating answer..."):
            answer = st.session_state.rag_chain.invoke(q)
        st.markdown("### Answer")
        st.write(answer)

else:
    st.info("Upload a PDF/TXT to start.")
