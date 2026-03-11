import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain.messages import SystemMessage, HumanMessage
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo


#loading the file
print("Loading the Document")
loader = PyPDFLoader(r"C:\Users\Goludev\Desktop\langchain\mythic_tale\document\shrimad-valmiki-ramayana.pdf")
data = loader.load()

#splitting the text
print("Splitting Text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)
chunks = text_splitter.split_documents(data)

for chunk in chunks:
    chunk.metadata["book"] = "Ramcharitmanas"
    chunk.metadata["author"] = "Tulsidas"
    chunk.metadata["type"] = "verse"


#metadata field info
metadata_field_info = [
    AttributeInfo(
        name="book",
        description="The mythology book where the text comes from",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="Author of the mythology book",
        type="string",
    ),
    AttributeInfo(
        name="type",
        description="Type of text such as verse, story, explanation",
        type="string",
    ),
]


#now embeddings
print("Converting to Vector")
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

document_content_description = "Verses and explanations from Indian mythology texts"

#vector database
vector_db = Chroma.from_documents(documents = chunks, embedding = embeddings, persist_directory = r"C:\Users\Goludev\Desktop\langchain\mythic_tale\vector_db")

print("Your database is ready")

#llm define
print("Model is Loaded")
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2-7B-Instruct",
    task="text-generation",
    max_new_tokens = 300,
    do_sample = False,
    temperature=0.1,
    repetition_penalty=1.1,
    huggingfacehub_api_token = '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'
)

model = ChatHuggingFace(llm = llm)

#retriever
#retriever = vector_db.as_retriever(search_kwargs = {"k" : 6})
retriever = SelfQueryRetriever.from_llm(
    model,
    vector_db,
    document_content_description,
    metadata_field_info,
    verbose=True
)

while True:

    query = input("Ask a question about the Ramayana (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    print("\n Retrieved Documents :\n")

    for i, doc in enumerate(docs):
        print(f"Document {i+1}: ")
        print(doc.page_content[:500])
        #print("Source:", doc.metadata)

    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt
    messages = [
        SystemMessage(
            content="""You are a mythology assistant.

Answer ONLY using the provided context.
If the answer is not present in the context, say:

"I could not find the answer in the provided text."
"""
        ),
        HumanMessage(
            content=f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Provide a clear and concise answer.
"""
        )
    ]

    # Generate response
    response = model.invoke(messages)

    print("\nAnswer:\n")
    print(response.content)
    print("\n" + "-" * 50 + "\n")
    print("\nSources:\n")

    for doc in docs:
        print(f"Page: {doc.metadata.get('page')}")



