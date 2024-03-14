from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain
from langchain.tools.base import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests
from io import BytesIO
import streamlit as st
from pdfplumber import PDF
import textract

def process_document(document, chat_history):
    if isinstance(document, (PDF, BytesIO)):
        raw_text = ""
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
    else:
        raw_text = textract.process(document).decode("utf-8")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    if texts:
        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        query = st.text_input("Enter your query")
        if st.button("Submit Query"):
            docs = document_search.similarity_search(query)
            response = chain.run(input_documents=docs, question=query)
            chat_history.append((query, response))
            st.write(response)
    else:
        st.write("No text found in the uploaded document.")

def process_webpage(url, chat_history):
    class WebpageQATool(BaseTool):
        name = "query_webpage"
        description = "Browse a webpage and retrieve the information and answers relevant to the question. Please use bullet points to list the answers"
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            length_function=len,
        )
        qa_chain: BaseCombineDocumentsChain

        def _run(self, url: str, question: str) -> str:
            response = requests.get(url)
            page_content = response.text
            docs = [Document(page_content=page_content, metadata={"source": url})]
            web_docs = self.text_splitter.split_documents(docs)
            results = []
            for i in range(0, len(web_docs), 4):
                input_docs = web_docs[i : i + 4]
                window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
                results.append(f"Response from window {i} - {window_result}")
            results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
            return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)

        async def _arun(self, url: str, question: str) -> str:
            raise NotImplementedError

    if url:
        llm = ChatOpenAI(temperature=0.5)
        query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

        query = st.text_input("Enter your query")
        if st.button("Get Answers"):
            if query:
                final_answer = query_website_tool._run(url, query)
                chat_history.append((query, final_answer))
                st.write(final_answer)

def display_chat_history(chat_history):
    st.subheader("Chat History")
    for i, (query, response) in enumerate(chat_history, start=1):
        st.write(f"Query {i}: {query}")
        st.write(f"Response {i}: {response}")
        st.write("--" * 20)
