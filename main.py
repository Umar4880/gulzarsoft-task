from .ragtest import RAG_Pipeline
from .chain import Chain


def invoke_pipeline(self):
    rag = RAG_Pipeline("ragtest\data\Comply Analytics.pdf")
    rag_chain = Chain()

    retriver = rag.implement_rag()
    chain = rag_chain.call_chain(retriver)
    result = chain.invoke("give me key 10 points")

    return result


if __name__ == "__main__":
    response = invoke_pipeline()
    print(response)