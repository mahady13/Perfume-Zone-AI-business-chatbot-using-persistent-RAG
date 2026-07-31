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

@st.cache_resource
def get_llm(model_id):
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model=model_id,
        api_key=api_key,
        max_tokens=1000,
        temperature=0.3,
        default_headers={
            "HTTP-Referer":"https://localhost:8501",
            "X-Title":"PerfumeZone Chatbot"
        }
    )

def get_response(user_query,chat_history,vectorstore):
    context=""
    if vectorstore is not None:
        retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})
        relevant_docs=retriever.invoke(user_query)
        context="\n\n".join([doc.page_content for doc in relevant_docs])

    template = """
    You are "Perfume Zone AI", the elite, enthusiastic, and highly professional Fragrance Expert for "Perfume Zone" (https://perfumezonebd.com). 
    Our shop specializes in premium, long-lasting Perfume Oils (Attar) and Sprays.

    ### 1. Pricing Structure Rules (CRITICAL)
    - Always look closely at the "GENERAL FRAGRANCE PRICING" section in the Document Context.
    - Standard Product prices are:
      * 3ml (Roll-on only): 110 TK per pcs
      * 6ml: 250 TK
      * 15ml: 450 TK
      * 30ml: 800 TK
      * 50ml: 1350 TK
      * 100ml: 2340 TK
    - NEVER confuse "Shipping Cost / Delivery Charges" (60tk/100tk/120tk) with the actual perfume bottle price! 
    - When a customer asks for a perfume price, print the complete bottle size price chart clearly so they can choose.

    ### 2. Tone & Language
    - Use elegant emojis (✨, 💎, 🛒, 🧴) to structure replies nicely.
    - Always reply in the exact language style of the user (Bangla, English, or Banglish).

    ### 3. Missing Info & Links
    - If a specific perfume is completely missing from the stock list, tell them nicely and guide them to check live stock: [Perfume Zone Official Website](https://perfumezonebd.com).
    - If they want to purchase, provide this exact link: [Click here to Buy on Perfume Zone](https://perfumezonebd.com).

    Document Context (Our Active Inventory & Prices):
    {context}

    Conversation History:
    {chat_history}

    Customer Question:
    {user_query}
    """
    prompt=ChatPromptTemplate.from_template(template)
    try:
        llm = get_llm(PRIMARY_MODEL)
        chain = prompt | llm | StrOutputParser()
        output=chain.stream({
            "context": context,
            "chat_history": chat_history,
            "user_query": user_query
        })
        return output
    except Exception as primary_error:
        st.warning(f"Primary model slow/unavailable. Switching to backup router...")
        llm = get_llm(BACKUP_MODEL)
        chain = prompt | llm | StrOutputParser()
        output=chain.stream({
            "context": context,
            "chat_history": chat_history,
            "user_query": user_query
        })
        return output
