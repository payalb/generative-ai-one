import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY=""

st.header("Interview Process")
with st.sidebar:
    st.title("Your docs")
    file = st.file_uploader("Upload your resume ", type="pdf")

#extract text
if file is not None:
    pdf_reader=PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text+=page.extract_text()
       # st.write(text)

# break into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size= 1000, chunk_overlap=150, length_function=len)

    chunks = text_splitter.split_text(text)
   # st.write(chunks)

# generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# creating vector
    vector_store = FAISS.from_texts(chunks, embeddings)

# GET USER QUESTION
    user_question = st.text_input("Enter the question here")
# DO SIMILARITY SEARCH
    if user_question:
        match = vector_store.similarity_search(user_question)
      #  st.write(match)
# OUTPUT RESULTS: take question, relevant docs, pass to LLM, generate output.
        llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, temperature= 0
                         , max_tokens = 200, model_name = "gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(question = user_question, input_documents = match)
        st.write(response)
