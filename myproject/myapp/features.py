from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document
import requests
from io import BytesIO
from pdfplumber import PDF
import textract
from .webask import get_response, get_vectorstore_from_url

def process_document(document):
    raw_text = ""
    if isinstance(document, (PDF, BytesIO)):
        
        if isinstance(document, PDF):
            for i, page in enumerate(document.pages):
                content = page.extract_text()
                if content:
                    raw_text += content
        else:
            with PDF(document) as pdf:
                for i, page in enumerate(pdf.pages):
                    content = page.extract_text()
                    if content:
                        raw_text += content
    elif isinstance(document, str):
        raw_text = document
    elif isinstance(document, bytes):
        raw_text = document.decode("utf-8")
    else:
        return {"error": "Invalid document type"}

    if not raw_text:
        return {"error": "No text found in the uploaded document"}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    if texts:
        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        def answer_query(query):
            docs = document_search.similarity_search(query)
            response = chain.run(input_documents=docs, question=query)
            return response

        return answer_query

    else:
        return lambda: "No text found in the uploaded document."



def process_webpage(url):
    if url:
        if url.endswith(".pdf"):
            response = requests.get(url)
            pdf_file = BytesIO(response.content)
            return process_document(pdf_file)
        else:
            vector_store = get_vectorstore_from_url(url)

            def answer_query(query):
                response = get_response(query, vector_store)  # Pass vector_store
                return response

            return answer_query
