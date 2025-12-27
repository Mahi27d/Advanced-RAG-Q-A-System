import os
import faiss
import streamlit as st
import numpy as np
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader

from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import HuggingFaceEndpoint

from secret_api_keys import huggingface_api_key


# ------------------------------------------------------------------
# ENV SETUP
# ------------------------------------------------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key


# ------------------------------------------------------------------
# INPUT PROCESSING
# ------------------------------------------------------------------
def process_input(input_type, input_data):
    documents = []

    # -------------------- LINK --------------------
    if input_type == "Link":
        for url in input_data:
            if url.strip():
                loader = WebBaseLoader(url)
                documents.extend(loader.load())

        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = [doc.page_content for doc in text_splitter.split_documents(documents)]

    # -------------------- PDF --------------------
    elif input_type == "PDF":
        pdf_reader = PdfReader(BytesIO(input_data.read()))
        full_text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                full_text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_text(full_text)

    # -------------------- TEXT --------------------
    elif input_type == "Text":
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_text(input_data)

    # -------------------- DOCX --------------------
    elif input_type == "DOCX":
        doc = Document(BytesIO(input_data.read()))
        full_text = "\n".join([p.text for p in doc.paragraphs])

        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_text(full_text)

    # -------------------- TXT --------------------
    elif input_type == "TXT":
        full_text = input_data.read().decode("utf-8")

        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_text(full_text)

    else:
        raise ValueError("Unsupported input type")

    # ------------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    # ------------------------------------------------------------------
    # FAISS VECTOR STORE (MANUAL, FAST, STABLE)
    # ------------------------------------------------------------------
    sample_embedding = np.array(embeddings.embed_query("sample"))
    dimension = sample_embedding.shape[0]

    index = faiss.IndexFlatL2(dimension)

    vectorstore = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vectorstore.add_texts(texts)

    return vectorstore


# ------------------------------------------------------------------
# QA PIPELINE
# ------------------------------------------------------------------
def answer_question(vectorstore, query):
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        token=huggingface_api_key,
        temperature=0.6,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
    )

    result = qa({"query": query})
    return result["result"]


# ------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Advanced RAG Q&A", layout="wide")
    st.title("📄 Advanced RAG Question Answering System")

    input_type = st.selectbox(
        "Select Input Type",
        ["Link", "PDF", "Text", "DOCX", "TXT"]
    )

    input_data = None

    if input_type == "Link":
        count = st.number_input(
            "Number of URLs",
            min_value=1,
            max_value=20,
            step=1
        )
        urls = []
        for i in range(int(count)):
            urls.append(st.text_input(f"URL {i + 1}"))
        input_data = urls

    elif input_type == "Text":
        input_data = st.text_area("Enter Text", height=200)

    elif input_type == "PDF":
        input_data = st.file_uploader("Upload PDF", type=["pdf"])

    elif input_type == "TXT":
        input_data = st.file_uploader("Upload TXT", type=["txt"])

    elif input_type == "DOCX":
        input_data = st.file_uploader("Upload DOCX", type=["docx", "doc"])

    if st.button("Process Input"):
        if not input_data:
            st.warning("Please provide valid input.")
        else:
            with st.spinner("Processing and building vector index..."):
                vectorstore = process_input(input_type, input_data)
                st.session_state["vectorstore"] = vectorstore
                st.success("Documents processed successfully.")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask a Question")
        if st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                answer = answer_question(st.session_state["vectorstore"], query)
                st.markdown("### Answer")
                st.write(answer)


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
