import streamlit as st
import glob
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

st.set_page_config(page_title="Perfume Zone AI", page_icon="✨", layout="centered")
st.title("Perfume Zone AI✨")


with st.sidebar:
    st.title("Perfume Zone ✨")
    st.header("Contact")
    col1,col2=st.columns(2)
    with col1:
        st.link_button(label='Website',url='https://perfumezonebd.com',use_container_width=True)
    with col2:
        st.link_button(label='Whatsapp', url='https://wa.me/1625338214?text=Ami%20ekti%20perfume%20oil%20kinte%20chai',use_container_width=True)
    st.link_button(label='Facebook', url='https://www.facebook.com/perfumezone0/',use_container_width=True)

    st.markdown("---")
    st.header("Developer Information")
    st.markdown("""
            **Mohiuddin Mahady**  
            *BSc in CSE*  
            Mymensingh Engineering College  
            *(Affiliated with Dhaka University)*
            """)
    col3, col4 = st.columns([1, 1])
    with col3:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/mohiuddin-mahady/", use_container_width=True)
    with col4:
        st.link_button("Github", 'https://www.github.com/mahady13', use_container_width=True)
#fixed them as comments for future fast ci/cd
# available_models={
#     "Ling 3 Flash": "inclusionai/ling-3.0-flash:free",
#     "Google Gemma 4-26b-a4b": "google/gemma-4-26b-a4b-it:free",
#     "Nvidia Nemotron 3 Ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
#     "Nvidia Nano Omni": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
#     "Nemotron 3 Super": "nvidia/nemotron-3-super-120b-a12b:free",
#     "Cohere: North Mini Code": "cohere/north-mini-code:free",
#     "PoolSide Laguna S2.1": "poolside/laguna-s-2.1:free",
#     "PoolSide Laguna XS2.1": "poolside/laguna-xs-2.1:free",
#     "OpenAI: gpt-oss-20b": "openai/gpt-oss-20b:free",
#     "Auto Free Router": "openrouter/free",
# }
PRIMARY_MODEL="inclusionai/ling-3.0-flash:free"
BACKUP_MODEL="openrouter/free"

api_key=os.getenv("OPENROUTER_API_KEY")
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
pdf_files = glob.glob("./assets/*.pdf")

@st.cache_resource
def load_vectorstore():
    assist_directory="./assets"
    persist_directory='./chromadb'
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectorstore=Chroma(
            persist_directory=persist_directory,embedding_function=embedding,
        )
        return vectorstore

    elif pdf_files:
        loader = PyPDFDirectoryLoader(assist_directory)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        split_text = splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(
            documents=split_text,
            embedding=embedding,
            persist_directory=persist_directory
        )
        return vectorstore
    return None

vectorstore=load_vectorstore()
