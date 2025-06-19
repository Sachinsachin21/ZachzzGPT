
import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()

st.title("ZachzzGPT: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_groq.pkl"
main_placeholder = st.empty()

llm = ChatGroq(
    temperature=0.7,
    model_name="llama3-70b-8192",
    api_key="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"
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=[url for url in urls if url])
    main_placeholder.text("🔄 Loading URLs...")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=100)
    main_placeholder.text("🔄 Splitting text...")
    docs = splitter.split_documents(data)

    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("🔄 Building vector store...")
    time.sleep(1)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    st.success("✅ FAISS vector store saved!")

query = main_placeholder.text_input("Ask a question about the articles:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        result = chain({"question": query}, return_only_outputs=True)

        st.header("📌 Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("🔗 Sources")
            for src in sources.split("\n"):
                st.write(src)