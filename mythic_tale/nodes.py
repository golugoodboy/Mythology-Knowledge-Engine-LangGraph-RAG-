from langchain_core.messages import SystemMessage, HumanMessage
from reranker import rerank_documents
from compressor import create_compressor

def retrieve(state, retriever):
    print("\n--- Retrieving Documents ---")
    question = state["rewritten_query"]

    filter_book = None

    if "ramayana" in question.lower():
        filter_book = "shrimad-valmiki-ramayana"
    
    if "chanakya" in question.lower():
        filter_book = "Chanakya - Chanakya Neeti (2018)"

    if filter_book:

        docs = retriever.vectorstore.similarity_search(
            question,
            k=8,
            filter={"book": filter_book}
        )

    else:

        docs = retriever.invoke(question)

    print(f"Question: {question}\n")
    print(f"\nTotal retrieved documents: {len(docs)}\n")

    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}")
        print("Source:", doc.metadata)
        print(doc.page_content[:400])
        print("\n------------------------\n")
        

    return {"documents" : docs}

#------------------------------------------------------------------------------------#

def generate(state, model):
    question = state["question"]
    docs = state["documents"]

    if len(docs) == 0:
        return {"answer": "No relevant content found in the knowledge base."}
    

    context = "\n\n".join([doc.page_content for doc in docs])


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
{question}

Provide a clear and concise answer.
"""
        )
    ]

    response = model.invoke(messages)

    return {"answer" : response.content}

#----------------------------------------------------------------------------------------#
def rewrite_query(state, model):
    print("\n--- Rewriting Query ---")
    question = state["question"]

    prompt = f"""
Rewrite the following question to improve document retrieval.
Focus on important keywords like book names, characters, authors.

Question:
{question}
"""

    response = model.invoke(prompt)

    rewritten_query = response.content.strip()

    print("Question :", question)
    print("Rewritten Query :", rewritten_query)

    return {"question" : question, "rewritten_query" : rewritten_query}

#-------------------------------------------------------------------------------------------#

def rerank(state):
    print("\n--- Reranking Documents ---")
    question = state["rewritten_query"]
    docs = state["documents"]

    ranked_docs = rerank_documents(question, docs)

    print("\nTotal ranked documents: {}\n".format(len(ranked_docs)))

    return {"documents" : ranked_docs}


#------------------------------------------------------------------------------------------#

def compress_context(state, model):
    print("\n--- Compressing Context ---")
    question = state["rewritten_query"]
    docs = state["documents"]

    compressor = create_compressor(model)
    compressed_docs = compressor.compress_documents(docs, question)

    print("\nTotal compressed documents: {}\n".format(len(compressed_docs)))

    return {"documents" : compressed_docs}

