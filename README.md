# Mythology-Knowledge-Engine-LangGraph-RAG-
An AI-powered knowledge system that answers questions about mythology texts such as Ramayana and Chanakya Neeti using Retrieval Augmented Generation (RAG) and LangGraph workflows.  The system ingests mythology books as PDFs, builds a semantic vector database, retrieves relevant passages, and generates grounded answers using a Large Language Model.

# Mythology Knowledge Engine (LangGraph + RAG)

An AI-powered knowledge system that answers questions about mythology texts such as **Ramayana** and **Chanakya Neeti** using **Retrieval Augmented Generation (RAG)** and **LangGraph workflows**.

The system ingests mythology books as PDFs, builds a semantic vector database, retrieves relevant passages, and generates grounded answers using a Large Language Model.

---

## Features

* Multi-book mythology knowledge base (Ramayana, Chanakya Neeti)
* Retrieval-Augmented Generation (RAG) pipeline
* LangGraph workflow orchestration
* Parent–Child document retrieval for better context
* Multilingual semantic search (English queries over Hindi/Sanskrit texts)
* Query rewriting for improved retrieval
* Document reranking using cross-encoder models
* Persistent vector database using Chroma
* Modular architecture for easy extension

---

## System Architecture

User Question
↓
Query Rewrite Node
↓
Parent–Child Document Retrieval
↓
Document Reranking
↓
LLM Answer Generation

This pipeline ensures answers are generated **from retrieved documents rather than LLM memory**, reducing hallucinations.

---

## Technologies Used

* **LangChain**
* **LangGraph**
* **HuggingFace Models**
* **Sentence Transformers**
* **Chroma Vector Database**
* **CrossEncoder Reranker**
* **Python**

---

## Project Structure

```
mythic_tale/

├── app.py
│
├── graph/
│   ├── nodes.py
│   ├── workflow.py
│   └── state.py
│
├── rag/
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── reranker.py
│   └── compressor.py
│
├── document/
│   ├── Ramayana.pdf
│   └── Chanakya_Neeti.pdf
│
├── vector_db/
├── parent_store/
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/mythology-knowledge-engine.git
cd mythology-knowledge-engine
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file and add your HuggingFace API key:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## Running the Application

Start the system:

```
python app.py
```

Example interaction:

```
Ask mythology question: What does Chanakya say about friendship?
```

The system will retrieve relevant passages from the knowledge base and generate an answer grounded in the source documents.

---

## Example Questions

You can try questions such as:

* What does Chanakya say about friendship?
* What advice does Chanakya give about foolish people?
* What lesson about duty does the Ramayana teach?
* What does Ramayana say about dharma?

---

## Future Improvements

Planned enhancements include:

* Hybrid retrieval (vector + keyword search)
* Metadata-based filtering (characters, topics)
* Agentic routing between mythology books
* Context compression for efficient token usage
* Support for additional mythology texts (Mahabharata, Bhagavad Gita)

---

## Learning Outcomes

This project demonstrates how to build a **production-style RAG system** with:

* structured AI workflows
* modular retrieval pipelines
* scalable document ingestion
* grounded LLM responses

---

## License

MIT License

---

## Author

Developed as part of an exploration into **AI knowledge systems and RAG architectures for large text corpora**.

