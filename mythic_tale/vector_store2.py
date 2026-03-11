import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")
BOOK_FOLDER = os.path.join(BASE_DIR, "document")


def load_vectorstore(embeddings):

    print("Book folder path:", BOOK_FOLDER)
    print("Files inside:", os.listdir(BOOK_FOLDER))

    # Vector store
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    # Parent document store
    store = InMemoryStore()

    # Parent chunks (large context)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    # Child chunks (used for embeddings)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    # Parent-child retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    print("\nLoading documents...\n")

    all_docs = []

    for file in os.listdir(BOOK_FOLDER):

        if file.endswith(".pdf"):

            path = os.path.join(BOOK_FOLDER, file)

            print(f"Processing book: {file}")

            loader = PyPDFLoader(path)

            docs = loader.load()

            book_name = file.replace(".pdf", "")

            # add metadata
            for doc in docs:
                doc.metadata["book"] = book_name
                doc.metadata["source"] = file

            all_docs.extend(docs)

    # IMPORTANT STEP (Parent-Child indexing happens here)
    # We need to add documents in batches to avoid Chroma's max batch size limit (5461)
    batch_size = 200
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        print(f"Adding batch {i//batch_size + 1} ({len(batch)} documents)...")
        retriever.add_documents(batch)

    print(f"\nTotal documents added: {len(all_docs)}")

    return retriever