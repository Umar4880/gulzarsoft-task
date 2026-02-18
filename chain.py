from .ragtest import RAG_Pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class Chain:
    def __init__(self) -> None:
        pass

    def generate_prompt(self):
        """generate prompt"""

        prompt = """your are a senior ai assistant. your task is to analyze given text and provide helpful answer:
        __Critical__:
        - Keep tone professional and friendly
        - Answer always should be grounded to provided context
        - Deny gentely if the context not provided or question out of the context
         \n\n
        **context: ** {context}
        """
        template = ChatPromptTemplate.from_messages([
            ('system', prompt),
            ('human', "{input}")
        ])

        return template
    
    def init_model(self):
        """initalize model"""

        model = ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash",
            api_key = "AIzaSyBL_NQArluP4BOA53aqP6uMvGUD08V_CQo",
        )

        return model 
    

    def call_chain(self, retriever):
        """make chain"""
        
        prompt = self.generate_prompt()
        model = self.init_model()
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        
        return chain