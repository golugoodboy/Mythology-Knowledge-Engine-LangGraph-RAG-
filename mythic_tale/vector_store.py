import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

VECTOR_DB_PATH = "vector_db"
BOOK_FOLDER = "document"

print("Book folder path:", BOOK_FOLDER)
print("Files inside:", os.listdir(BOOK_FOLDER))

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 200
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 50
)

store = InMemoryStore()

retriever_parent = ParentDocumentRetriever(
    vectorstore = vector_db,
    child_splitter = child_splitter,
    parent_splitter = parent_splitter,
    docstore = store
)

def load_vectorstore(embeddings):

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):

        print("Loading existing vector database...")

        vector_db = Chroma(
            persist_directory = VECTOR_DB_PATH,
            embedding_function = embeddings
        )

        return vector_db
    
    else:

        print("Building vector database...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 200
    )

    all_chunks = []

    for file in os.listdir(BOOK_FOLDER):

        if file.endswith(".pdf"):

            path = os.path.join(BOOK_FOLDER, file)

            loader = PyPDFLoader(path)
            docs = loader.load()

            chunks = splitter.split_documents(docs)

            book_name = file.replace(".pdf", "")

            print(f"Processing {file}...")
            print(f"Total chunks: {len(chunks)}")

            for chunk in chunks:

                chunk.metadata["book"] = book_name
                chunk.metadata["source"] = file

            all_chunks.extend(chunks)

    vector_db = Chroma.from_documents(
        documents = all_chunks,
        embedding = embeddings,
        persist_directory = VECTOR_DB_PATH
    )

    vector_db.persist()

    return vector_db