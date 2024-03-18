from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
from features import process_document, process_webpage, get_response
from readcsv import process_csv
from io import BytesIO
from pdfplumber import PDF
from docx import Document as DocxDocument
import pandas as pd
import requests

app = FastAPI()

os.environ["OPENAI_API_KEY"] = ""

# Store the vector_store object for each uploaded file or URL
vector_stores = {}

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
        response = vector_store(user_query)

        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
