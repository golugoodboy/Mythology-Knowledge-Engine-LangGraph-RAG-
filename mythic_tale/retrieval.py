from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo



def create_retriever(model, vector_db):

    metadata_field_info = [

        AttributeInfo(
            name="book",
            description="The mythology book",
            type="string"
        ),

        AttributeInfo(
            name="author",
            description="Author of the text",
            type="string"
        ),
    ]

    document_description = "Mythology verses and explanations"

    retriever = SelfQueryRetriever.from_llm(
        model,
        vector_db,
        document_description,
        metadata_field_info,
        verbose=True
    )

    return retriever
    

    