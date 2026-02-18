from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter



class RAG_Pipeline:
    def __init__(self, doc_path: str = None) -> None:
        if not doc_path:
            raise ValueError("Invalid Path")
        
        self.api_key = "AIzaSyBL_NQArluP4BOA53aqP6uMvGUD08V_CQo"

        self.path = doc_path
        self.loader = PyPDFLoader(doc_path) if doc_path.endswith(".pdf") else Docx2txtLoader(doc_path)
        self.splitter = RecursiveCharacterTextSplitter()

    def load_docs(self):
        """load documents"""
        try:
            docs = self.loader.load()
            print("docs loaded...")
            return docs
        except Exception as e:
            raise ValueError("docs not get laod", e)
    
    def split_docs(self, docs):
        """split docs"""
        try:
            splited_docs = self.splitter.split_documents(docs)
            print("doc splitted...")
            return splited_docs
        except Exception as e:
            raise ValueError("docs not get split", e)
    
    def store_docs(self, docs):
        """store docs"""

        try:

            retriever = FAISS.from_documents(
                documents=docs,
                embedding=GoogleGenerativeAIEmbeddings(
                    model ="gemini-embedding-001",
                    api_key=self.api_key),
            )
            print("store into database")

            return retriever.as_retriever(search_kwargs={"k":5})
        except Exception as e:
            raise ValueError("docs not get store", e)
    
    def implement_rag(self):
        """load, split and store"""

        docs = self.load_docs()
        split = self.split_docs(docs)
        storer = self.store_docs(split)

        return storer
    
    
    def retrieve_docs(self, query, retriever):
        """retrieve docs"""
        try:

            docs = retriever.invoke(query)
            print("docs retrieved ", docs)
            return docs
        except Exception as e:
            raise ValueError("Erro in retrieving docs", e)
    

