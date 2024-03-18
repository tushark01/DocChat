from django.shortcuts import render

# Create your views here.
import os
from io import BytesIO
from fastapi import HTTPException
import pandas as pd
from pdfplumber import PDF
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings


from docx import Document as DocxDocument
from .serializers import UserQuerySerializer
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import requests

# Import your other functions from the original code
from .features import process_document, process_webpage
from .readcsv import process_csv
os.environ["OPENAI_API_KEY"] = "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"

vector_stores = {}

# Functions from the original code
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

def get_response(user_query, vector_store):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke(
        {"input": user_query}
    )

    return response["answer"]

# View classes
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

@method_decorator(csrf_exempt, name='dispatch')
class UploadFileView(APIView):
    def post(self, request):
        try:
            file = request.FILES.get('file')
            if file:
                if file.content_type == "application/pdf":
                    pdf_bytes = file.read()
                    pdf_file = BytesIO(pdf_bytes)
                    vector_stores[file.name] = process_document(pdf_file)
                elif file.content_type == "text/plain":
                    vector_stores[file.name] = process_document(file.read().decode("utf-8"))
                elif file.content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                    doc = DocxDocument(BytesIO(file.read()))
                    full_text = "\n".join([p.text for p in doc.paragraphs])
                    vector_stores[file.name] = process_document(full_text)
                else:
                    return Response({"error": "Invalid file type"}, status=status.HTTP_400_BAD_REQUEST)

                return Response({"message": "File processed successfully"}, status=status.HTTP_200_OK)
            else:
                return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class ProcessCSVView(APIView):
    def post(self, request):
        try:
            csv_file = request.FILES.get('csv_file')
            if csv_file and csv_file.content_type == "text/csv":
                encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings_to_try:
                    try:
                        csv_data = csv_file.read()  # Read the file content
                        df = pd.read_csv(BytesIO(csv_data), encoding=encoding)  # Pass the file content to read_csv
                        vector_stores[csv_file.name] = process_csv(df)
                        break
                    except UnicodeDecodeError:
                        continue

                return Response({"message": "CSV processed successfully"}, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Invalid file type"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ChatView(APIView):
    def post(self, request):
        try:
            filename = request.data.get('filename')
            user_query = request.data.get('user_query')

            vector_store = vector_stores.get(filename)
            print("vector_store", vector_store)
            print("*********VS***********")
            if vector_store is None:
                raise HTTPException(status_code=400, detail="No vector store found for the given filename")

            print("Passsssssssssssss")

            # response = get_response(user_query, vector_store)
            response = vector_store(user_query)
            print("________________________")
            return Response({"response": response})
            #return {"response": response}
        except Exception as e:
            return Response({"error": str(e)})

        
class AnswerView(APIView):
    serializer_class = UserQuerySerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            url = serializer.validated_data['url']
            query = serializer.validated_data['query']

            if not url:
                return Response({"detail": "URL is required"}, status=status.HTTP_400_BAD_REQUEST)

            if url.endswith(".pdf"):
                response = requests.get(url)
                pdf_file = BytesIO(response.content)
                vector_store = process_document(pdf_file)
            elif url.startswith("http"):
                vector_store = get_vectorstore_from_url(url)
            else:
                return Response({"detail": "Invalid URL"}, status=status.HTTP_400_BAD_REQUEST)

            if not os.path.exists(os.path.join(settings.BASE_DIR, ".vector_stores")):
                os.makedirs(os.path.join(settings.BASE_DIR, ".vector_stores"))

            response = get_response(query, vector_store)

            return Response({"answer": response})
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        