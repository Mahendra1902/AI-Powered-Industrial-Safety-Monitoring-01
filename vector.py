import os
from glob import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Constants
DOCS_PATH = "incident_docs"
DB_PATH = "incident_faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def build_vector_db():
    if not os.path.exists(DOCS_PATH) or not glob(f"{DOCS_PATH}/*.txt"):
        print("No incident documents found to build the vector DB.")
        return

    print("Loading documents...")
    documents = []
    for filepath in glob(f"{DOCS_PATH}/*.txt"):
        loader = TextLoader(filepath)
        documents.extend(loader.load())

    print("Splitting documents...")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Creating FAISS index...")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(DB_PATH)
    print(f"Vector database saved at '{DB_PATH}'.")

if __name__ == "__main__":
    build_vector_db()
