import os
from ragtest import RAG_Pipeline
from chain import Chain


def invoke_pipeline():
    base = os.path.dirname(__file__)
    doc_path = os.path.join(base, "data", "Comply Analytics.pdf")
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"document not found: {doc_path}")

    rag = RAG_Pipeline(doc_path)
    rag_chain = Chain()

    retriever = rag.implement_rag()
    chain = rag_chain.call_chain(retriever)

    result = chain.invoke("give me key 10 points")
    return result


if __name__ == "__main__":
    response = invoke_pipeline()
    print("response:", response)
