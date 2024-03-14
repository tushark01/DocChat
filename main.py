import os
import streamlit as st
from streamlit_ace import st_ace
from pdfplumber import PDF
from docx import Document as DocxDocument
import textract
from features import process_document, process_webpage, display_chat_history
import requests
from io import BytesIO

os.environ["OPENAI_API_KEY"] = ""

def extract_text(document):
    raw_text = ""
    if isinstance(document, PDF):
        for page in document.pages:
            content = page.extract_text()
            if content:
                raw_text += content
    elif isinstance(document, DocxDocument):
        for paragraph in document.paragraphs:
            raw_text += paragraph.text
    else:
        raw_text = textract.process(document).decode("utf-8")
    return raw_text

def main():
    st.set_page_config(page_title="Document Query App", page_icon=":mag:", layout="wide")

    upload_option = st.radio("Choose input method", ["Upload File", "Enter URL"])
    chat_history = []

    if upload_option == "Upload File":
        with st.sidebar:
            file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])
        if file is not None:
            if file.type == "application/pdf":
                with st.spinner("Extracting text..."):
                    with PDF(file) as pdf:
                        raw_text = extract_text(pdf)
                process_document(raw_text, chat_history)
            elif file.type == "text/plain":
                process_document(file.read().decode("utf-8"), chat_history)
            elif file.type in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"}:
                doc = DocxDocument(file)
                process_document(extract_text(doc), chat_history)
    elif upload_option == "Enter URL":
        with st.sidebar:
            url = st.text_input("Enter the URL")
        if url:
            if url.endswith(".pdf"):
                response = requests.get(url)
                pdf_file = BytesIO(response.content)
                with st.spinner("Extracting text..."):
                    with PDF(pdf_file) as pdf:
                        raw_text = extract_text(pdf)
                process_document(raw_text, chat_history)
            else:
                process_webpage(url, chat_history)

    display_chat_history(chat_history)

if __name__ == "__main__":
    main()
