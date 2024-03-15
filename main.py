import os
import streamlit as st
from streamlit_ace import st_ace
from pdfplumber import PDF
from docx import Document as DocxDocument
import textract
from features import process_document, process_webpage, display_chat_history
import requests
from io import BytesIO

os.environ["OPENAI_API_KEY"] = "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"

def main():
    st.title("Document Query App")

    #!Check if chat_history is not already in the session state and initialize it if needed
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    #*Allow user to choose the input method: Upload File or Enter URL
    upload_option = st.radio("Choose input method", ["Upload File", "Enter URL"])

    if upload_option == "Upload File":

        #*Provide a file uploader for the user to upload a file
        file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])
        if file is not None:

            #!Process the uploaded file based on its type
            if file.type == "application/pdf":
                process_document(file, st.session_state.chat_history)
            elif file.type == "text/plain":
                process_document(file.read().decode("utf-8"), st.session_state.chat_history)
            elif file.type in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"}:
                doc = DocxDocument(file)
                full_text = "\n".join([p.text for p in doc.paragraphs])
                process_document(full_text, st.session_state.chat_history)

    elif upload_option == "Enter URL":

        #*Get the URL entered by the user
        url = st.text_input("Enter the URL")
        if url:

            #*Process the URL based on its extension
            if url.endswith(".pdf"):
                #*Send a GET request to the URL and retrieve the content
                response = requests.get(url)

                #*Store the content in a BytesIO object
                pdf_file = BytesIO(response.content)

                #*Process the PDF content using the PDF class from the pdfplumber library
                process_document(pdf_file, st.session_state.chat_history)
            else:
                #*Process the webpage using the process_webpage function from the features module
                process_webpage(url, st.session_state.chat_history)

    #!Display the chat history using the display_chat_history function from the features module
    display_chat_history(st.session_state.chat_history)

if __name__ == "__main__":
    main()
