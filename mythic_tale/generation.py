from graph import State

def generate(state : State):
    print("Generating Answer...")
    question = state["question"]
    docs = state["documents"]

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
    