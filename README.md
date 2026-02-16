# Q4 — RAG Document Q/A (Streamlit Deployment)

This project is a Retrieval-Augmented Generation (RAG) Document Question Answering app built with LangChain Runnables and deployed on Streamlit.

*Live App:* https://maarij.streamlit.app/

## Features
- Upload a PDF/TXT document
- Splits text into chunks (RecursiveCharacterTextSplitter)
- Creates embeddings and stores them in a vector DB (FAISS/Chroma)
- Retrieves top-k relevant chunks and answers using an LLM (RAG)

## Tech Stack
- Python, Streamlit
- LangChain (Runnables)
- FAISS  (Vector Store)
- OpenAI (Embeddings + Chat Model)
