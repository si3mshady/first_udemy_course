from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import logging

def main():
    try:
        st.set_page_config(page_title="GPT++", page_icon=None, layout="wide", initial_sidebar_state="expanded")
        load_dotenv()
        st.header("Your Personal Assistant🗣️💬")

        # upload file
        pdf = st.file_uploader("Upload your PDF", type="pdf")

        # extract the text and create embeddings
        if pdf is not None:
            with pdf.open("rb") as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                embeddings = OpenAIEmbeddings()
                splitter = CharacterTextSplitter()
                chunks = splitter.split_text(text)
                knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        while True:
            if pdf is None:
                st.warning("Please upload a PDF file.")
                break
            user_question = st.text_input("Query your PDF for insight:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)
                st.code(response, language="python")

    except Exception as e:
        logging.exception("An error occurred:")

if __name__ == '__main__':
    main()
