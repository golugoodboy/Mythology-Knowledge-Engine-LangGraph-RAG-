from langchain_classic.retrievers.document_compressors import LLMChainExtractor

def create_compressor(chat_model):

    compressor = LLMChainExtractor.from_llm(chat_model)

    return compressor