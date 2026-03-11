from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, docs, top_k = 3):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker_model.predict(pairs, convert_to_numpy=True)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in ranked[:top_k]]

    return top_docs
    
    