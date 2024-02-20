#conda python11
import streamlit as st
import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

DB_FAISS_PATH = "vectorstore/db_faiss"

def main():
    st.title("CSV Uploader")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # Load CSV data
        data = pd.read_csv(uploaded_file)
        st.write("CSV file contents:")
        st.write(data)

        # Split the text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(data)

        st.write("Number of text chunks:", len(text_chunks))

        # Download Sentence Transformers Embedding From Hugging Face
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        model = AutoModel.from_pretrained("microsoft/DialoGPT-large")
        embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Convert the text chunks into embeddings and save the embeddings into FAISS Knowledge Base
        docsearch = FAISS.from_documents(text_chunks, embeddings)

        docsearch.save_local(DB_FAISS_PATH)

        llm = CTransformers(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1
        )

        qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

        while True:
            chat_history = []
            # query = "What is the value of  GDP per capita of Finland provided in the data?"
            query = st.text_input("Input Prompt:")
            if query == 'exit':
                st.write('Exiting')
                sys.exit()
            if query == '':
                continue
            result = qa({"question": query, "chat_history": chat_history})
            st.write("Response:", result['answer'])

if __name__ == "__main__":
    main()