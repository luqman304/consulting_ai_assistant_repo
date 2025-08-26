"""
Consulting AI Assistant — Streamlit RAG App
- Upload consulting PDFs (case studies, decks, contracts)
- Build embeddings (OpenAI or HF) and store in ChromaDB
- Ask questions with conversational context
"""
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="Consulting AI Assistant", page_icon="💼", layout="wide")
st.title("💼 Consulting AI Assistant")
st.write("Upload PDFs and ask questions. Choose embeddings provider in the sidebar.")

provider = st.sidebar.selectbox("Embeddings Provider", ["OpenAI", "HuggingFace (local)"])
model = st.sidebar.text_input("Chat model (OpenAI)", value="gpt-4o-mini")
persist_dir = st.sidebar.text_input("ChromaDB directory", value="chroma_db")

uploaded = st.file_uploader("Upload 1 or more PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded:
    docs = []
    for uf in uploaded:
        path = os.path.join("uploads", uf.name)
        os.makedirs("uploads", exist_ok=True)
        with open(path, "wb") as f:
            f.write(uf.read())
        for page in PyPDFLoader(path).load():
            docs.append(page)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    if provider == "OpenAI":
        embed_fn = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        embed_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vs = Chroma.from_documents(chunks, embedding=embed_fn, persist_directory=persist_dir)
    vs.persist()
    st.success(f"Indexed {len(chunks)} chunks into {persist_dir}")

if os.path.exists(persist_dir):
    st.info("Using existing ChromaDB persistence")
    if provider == "OpenAI":
        llm = ChatOpenAI(model=model, temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))
        retriever = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))).as_retriever(search_kwargs={"k":4})
    else:
        # Fallback: try OpenAI for chat if key exists, else a dummy
        if os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(model=model, temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))
        else:
            class DummyLLM:
                def __call__(self, *args, **kwargs):
                    return "Set OPENAI_API_KEY for better answers. Context retrieved below."
            llm = DummyLLM()
        retriever = Chroma(persist_directory=persist_dir, embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")).as_retriever(search_kwargs={"k":4})

    chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    if "history" not in st.session_state:
        st.session_state.history = []

    q = st.text_input("Ask a question (e.g., 'Key risks from the proposal?')")
    if st.button("Ask") and q:
        try:
            res = chain.invoke({"question": q, "chat_history": st.session_state.history})
            answer = res["answer"] if isinstance(res, dict) else str(res)
        except Exception as e:
            answer = f"Error: {e}"
        st.session_state.history.append((q, answer))

    if st.session_state.history:
        st.subheader("Conversation")
        for i, (qq, aa) in enumerate(st.session_state.history):
            st.markdown(f"**Q{i+1}:** {qq}")
            st.write(aa)
else:
    st.warning("No persisted database yet. Upload PDFs to create one.")

st.caption("Tip: Export OPENAI_API_KEY to enable OpenAI embeddings/chat. Without it, HF embeddings + dummy chat will still demo retrieval.")
