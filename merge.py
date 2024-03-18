import hashlib
from fastapi import FastAPI, File, HTTPException, UploadFile
import pandas as pd
from pydantic import BaseModel
from typing import List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from io import BytesIO
from pdfplumber import PDF
from docx import Document as DocxDocument

import requests

from features import process_document, process_webpage
from readcsv import process_csv

app = FastAPI()

os.environ["OPENAI_API_KEY"] = "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"

# Store the vector_store object for each uploaded file or URL
vector_stores = {}

class UserQuery(BaseModel):
    url: str
    query: str

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate the search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query, vector_store, chat_history=[]):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke(
        {"chat_history": chat_history, "input": user_query}
    )

    return response["answer"]

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        if file.content_type == "application/pdf":
            pdf_bytes = await file.read()  # Read the contents of the file into a bytes object
            pdf_file = BytesIO(pdf_bytes)  # Create a file-like object that can be read by PDF from pdfplumber
            vector_stores[file.filename] = process_document(pdf_file)

        elif file.content_type == "text/plain":
            vector_stores[file.filename] = process_document(file.file.read().decode("utf-8"))

        elif file.content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            doc = DocxDocument(BytesIO(file.file.read()))
            full_text = "\n".join([p.text for p in doc.paragraphs])
            vector_stores[file.filename] = process_document(full_text)

        else:
            raise HTTPException(status_code=400, detail="Invalid file type")

        return {"message": "File processed successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/process_csv/")
async def process_csv_route(csv_file: UploadFile):
    try:
        if csv_file.content_type != "text/csv":
            raise HTTPException(status_code=400, detail="Invalid file type")

        encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(csv_file.file, encoding=encoding)
                vector_stores[csv_file.filename] = process_csv(df)
                break
            except UnicodeDecodeError:
                continue

        return {"message": "CSV processed successfully"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat/")
async def chat(filename: str, user_query: str):
    try:
        # Get the vector_store object for the uploaded file or URL
        vector_store = vector_stores.get(filename)
        if vector_store is None:
            raise HTTPException(status_code=400, detail="No vector store found for the given filename")

        # Call the vector_store function with the user query
        response = get_response(user_query, vector_store)

        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/url_chat/")
async def answer(user_query: UserQuery):
    url = user_query.url
    query = user_query.query

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    if url.endswith(".pdf"):
        response = requests.get(url)
        pdf_file = BytesIO(response.content)
        vector_store = process_document(pdf_file)
    elif url.startswith("http"):
        vector_store = get_vectorstore_from_url(url)
    else:
        raise HTTPException(status_code=400, detail="Invalid URL")

    if not os.path.exists(".vector_stores"):
        os.makedirs(".vector_stores")

    response = get_response(query, vector_store)

    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008)
