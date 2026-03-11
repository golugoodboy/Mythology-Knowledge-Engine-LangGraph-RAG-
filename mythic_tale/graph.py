from typing import TypedDict, List
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    rewritten_query : str
    documents: List[Document]
    answer: str



