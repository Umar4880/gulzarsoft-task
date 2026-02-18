# RAG Pipeline & Chain

This folder contains two main files for a Retrieval-Augmented Generation (RAG) system using LangChain and Google Gemini models:

## Files

- **ragtest.py**: Implements the `RAG_Pipeline` class for document loading, splitting, storing, and retrieval.
- **chain.py**: Implements the `Chain` class for prompt generation, model initialization, and chaining retrieval with generative AI.

---

## ragtest.py

### RAG_Pipeline
- **Purpose**: Handles end-to-end document processing for RAG.
- **Features:**
  - Loads PDF or DOCX files
  - Splits documents into chunks
  - Stores chunks in a FAISS vector database using Google Gemini embeddings
  - Retrieves relevant chunks for a given query

#### Example Usage
```python
rag = RAG_Pipeline("example.pdf")
docs = rag.load_docs()
split = rag.split_docs(docs)
storer = rag.store_docs(split)
retrieved = rag.retrieve_docs("your query", storer)
```

---

## chain.py

### Chain
- **Purpose**: Builds a prompt and connects the retriever to a generative model for answering questions.
- **Features:**
  - Generates a professional, context-grounded prompt
  - Initializes Google Gemini model
  - Creates a chain combining context retrieval and generative AI

#### Example Usage
```python
from .ragtest import RAG_Pipeline
from .chain import Chain

rag = RAG_Pipeline("example.pdf")
docs = rag.load_docs()
split = rag.split_docs(docs)
retriever = rag.store_docs(split)

chain = Chain()
lcel_chain = chain.make_chain(retriever)
result = lcel_chain.invoke({"input": "your question"})
```

---

## Requirements
- Python 3.8+
- Install dependencies from requirements.txt

## Notes
- Requires a valid Google Gemini API key
- Only PDF and DOCX files are supported for document loading

---

## License
This project is for educational and research purposes.
