from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_csv(df):
    csv_text = df.to_csv(index=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(csv_text)

    if texts:
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        def answer_query(query):
            docs = vector_store.similarity_search(query)
            response = chain.run(input_documents=docs, question=query)
            return response

        return answer_query

    else:
        return lambda: "No text found in the uploaded CSV file."
