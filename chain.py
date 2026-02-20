import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def generate_prompt(self):
        prompt = (
            "you are a senior ai assistant. your task is to analyze given text "
            "and provide helpful answer:\n"
            "__Critical__:\n"
            "- Keep tone professional and friendly\n"
            "- Answer always should be grounded to provided context\n"
            "- Deny gently if the context is not provided or the question is out "
            "of scope\n\n"
            "**context:** {context}\n"
        )
        return ChatPromptTemplate.from_messages([
            ('system', prompt),
            ('human', "{input}")
        ])

    def init_model(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be set in environment")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            temperature=1,
            max_tokens=1024,
        )

    def call_chain(self, retriever):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | self.generate_prompt()
            | self.init_model()
            | StrOutputParser()
        )
        return chain