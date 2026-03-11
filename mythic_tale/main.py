import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from embeddings import load_embeddings
from vector_store2 import load_vectorstore
from retrieval import create_retriever

from nodes import retrieve, generate

from langgraph_page import build_graph

# LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2-7B-Instruct",
    task="text-generation",
    max_new_tokens = 300,
    do_sample = False,
    temperature=0.1,
    repetition_penalty=1.1,
    huggingfacehub_api_token = ''
)


model = ChatHuggingFace(llm=llm)


# RAG setup
embeddings = load_embeddings()
#vector_db = load_vectorstore(embeddings)
#retriever = create_retriever(model, vector_db)
retriever = load_vectorstore(embeddings)


# LangGraph
graph = build_graph(retriever, model)


print("\nMythology AI Assistant Ready\n")


while True:

    question = input("Ask mythology question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    result = graph.invoke({"question": question})

    print("\nAnswer:\n")
    print(result["answer"])