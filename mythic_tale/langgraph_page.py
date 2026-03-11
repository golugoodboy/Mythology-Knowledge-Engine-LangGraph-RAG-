from retrieval import create_retriever
from generation import generate
from graph import State
from langgraph.graph import StateGraph, START, END
from nodes import retrieve, generate, rewrite_query, rerank, compress_context
from embeddings import load_embeddings
from vector_store2 import load_vectorstore

def build_graph(retriever, model):

    workflow = StateGraph(State)

    workflow.add_node(
        "rewrite_query",
        lambda state: rewrite_query(state, model)
    )

    workflow.add_node(
        "retrieve",
        lambda state: retrieve(state, retriever)
    )

    workflow.add_node(
        "rerank",
        lambda state: rerank(state)
    )

    workflow.add_node(
        "generate",
        lambda state: generate(state, model)
    )

    workflow.add_node(
        "compress_context",
        lambda state: compress_context(state, model)
    )

    workflow.set_entry_point("rewrite_query")

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "compress_context")
    workflow.add_edge("compress_context", "generate")
    workflow.add_edge("generate", END)

    graph = workflow.compile()

    return graph



