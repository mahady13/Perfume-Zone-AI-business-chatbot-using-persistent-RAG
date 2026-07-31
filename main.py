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
