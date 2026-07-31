import streamlit as st
import glob
import os
from streamlit_carousel import carousel
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
st.set_page_config(page_title="Perfume Zone AI", page_icon="✨", layout="centered")

st.markdown("""
    <style>
    .stCarousel, [data-testid="stHtml"] iframe, .carousel-item img {
        max-height: 380px !important; 
        object-fit: cover !important;    
        border-radius: 12px !important;
        margin: 0 auto !important;
    }

    [data-testid="element-container"] {
        margin-bottom: 0.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Perfume Zone AI✨",text_alignment="center")

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
# embedding=HuggingFaceInferenceAPIEmbeddings(
#     api_key=hf_api_key,
#     hf_api_token=hf_api_key,
#     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

gemini_key = os.getenv("GEMINI_API_KEY")
embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    api_key=gemini_key
)

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

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
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
        retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content="আসসালামু আলাইকুম! পারফিউম জোনে আপনাকে স্বাগতম। ✨ আজকে কোন চমৎকার সুগন্ধি দিয়ে আপনার মন ভালো করব বলুন? 🍊")
    ]

perfume_banners = [
    {
        "title": "",
        "text": "",
        "img": "https://scontent.fdac24-2.fna.fbcdn.net/v/t39.30808-6/739012691_1003739512471032_851677454991239113_n.jpg?stp=dst-jpg_tt6&cstp=mx1080x1080&ctp=s1080x1080&_nc_cat=111&ccb=1-7&_nc_sid=833d8c&_nc_eui2=AeH6_pNoQiSjTfpXQNRXqFi2r384q7P09X-vfzirs_T1f7641NC7cBCUVlQI4R6oSwYTfEwcHG3bNWf3oKnuLJ_X&_nc_ohc=3xxLUoLpzEMQ7kNvwEHvelh&_nc_oc=Adrz5itzM3nLaiinLIsTfo5im4iQpWWq5vYdOTkZGHmxgv6oKUl1cBx3gWKqZHUxR7djtGZMNU7iJF4piuy_k5v8&_nc_zt=23&_nc_ht=scontent.fdac24-2.fna&_nc_gid=ifm1qUIIp29AHGtXuH08uA&_nc_ss=7b2a8&oh=00_AQFL-UUkw5HX_CE-bmibPo9YjEfILITNbRV0QMXpUq-caA&oe=6A71FCBD",
    },
    {
        "title": "",
        "text": "",
        "img": "https://scontent.fdac24-5.fna.fbcdn.net/v/t39.30808-6/725199212_990942237084093_1693890862720007805_n.jpg?stp=dst-jpg_tt6&cstp=mx1080x1080&ctp=s1080x1080&_nc_cat=101&ccb=1-7&_nc_sid=833d8c&_nc_eui2=AeGwD1p8YjyLJdx3e02z68VFCzwnvYJXC0kLPCe9glcLSeKvcZrEv5iVRsGyo_0Zcb_qoVKl9hFf3Q-HuG3efgV2&_nc_ohc=rgl8mYgO6I4Q7kNvwFhBamD&_nc_oc=AdoWm6QJ0weL3zis5HKyWVQqXByEFaTk5wIKt32RINjJ4RgHkKdUxuIePVpUbJesbRCAzC_jWk_c1prRKGrIGzAR&_nc_zt=23&_nc_ht=scontent.fdac24-5.fna&_nc_gid=ihg0OWm0287_NkBjBHR_HQ&_nc_ss=7b2a8&oh=00_AQEzOAzVfwRUCrKYwUh3dla719IhgkqKisr4ZAjxbtZ2jg&oe=6A71C75C",
    },
    {
        "title": "",
        "text": "",
        "img": "https://scontent.fdac24-5.fna.fbcdn.net/v/t39.30808-6/746026711_1012032108308439_306530557166542663_n.jpg?stp=dst-jpg_tt6&cstp=mx1254x1254&ctp=s1254x1254&_nc_cat=101&ccb=1-7&_nc_sid=833d8c&_nc_eui2=AeElUKi9Iy_becyQ7xwYcrP6wF6mHxeg4JLAXqYfF6Dgktzp0R9eWbeXXJxe1xnWNKweLruSJQ2efQ7dhM6drf_7&_nc_ohc=IJAn4rumjaMQ7kNvwEYqDvu&_nc_oc=Adp4F4CLY0_5JcFuuh62v_wPfSrgLOTSSswAgPUSmb-maeQr2aG99lbdkQ3CIZ0aH_URMsbGdujLCDx9N6DqU0bh&_nc_zt=23&_nc_ht=scontent.fdac24-5.fna&_nc_gid=h-1BCjN6j3K9j65bKwSAlg&_nc_ss=7b2a8&oh=00_AQF_GDOmcViTGRhpgrIcanGFu5KKBjINM4zhuMCE-6Ymzg&oe=6A71DBDA",
    },
    {
        "title": "",
        "text": "",
        "img": "https://scontent.fdac24-5.fna.fbcdn.net/v/t39.30808-6/635454359_889551030556548_8145614045911123595_n.jpg?stp=dst-jpg_tt6&cstp=mx1080x1080&ctp=s1080x1080&_nc_cat=105&ccb=1-7&_nc_sid=833d8c&_nc_eui2=AeFlWvooZ6_eBS0Pm8opxFbf04D3O_q7bMPTgPc7-rtsw_TiIUPjdQdJruouDbWScsIm2RAKH4dt9exI4qDbQIFi&_nc_ohc=_aJL_Bw4WQEQ7kNvwFFo9J9&_nc_oc=Adosi8nldCC2fnC3n3EZxKjLRVfc9jDAQWyEiHYaVrp-_jcOeZESGRnc_XKiDt2nmVcCmzEQrZw4Ft-nVsHzSU3b&_nc_zt=23&_nc_ht=scontent.fdac24-5.fna&_nc_gid=exrOejJF5E3YX6h_rlcOYQ&_nc_ss=7b2a8&oh=00_AQG3BHAhyeFkk_9TGR-E6B-jKIiVhkZTD0-CiIOM-MsP_w&oe=6A71CC8F",
    },
    {
        "title": "",
        "text": "",
        "img": "https://scontent.fdac198-2.fna.fbcdn.net/v/t39.30808-6/738691597_1004330489078601_7107487147553559495_n.jpg?stp=dst-jpg_tt6&cstp=mx1080x1080&ctp=s1080x1080&_nc_cat=107&ccb=1-7&_nc_sid=833d8c&_nc_eui2=AeFs0VIv8f2UuvHxcRRiQFGGDw5TOknRCI4PDlM6SdEIjgZao1U1jRTv_YSxO_8aGDVr0DWBPtERRynaz9dnAcL0&_nc_ohc=ltkkSsHeAbUQ7kNvwH0LDI7&_nc_oc=AdoaIuiXljn9k-s-bSA0RqE4_04-js0pm-QRvI21l0aUwapscdM9mBn0htXgaeHDL0xhsXn8RuwD52GZqqD83nKk&_nc_zt=23&_nc_ht=scontent.fdac198-2.fna&_nc_gid=dAhSp7_ppd22XwOm7UtYsA&_nc_ss=7b2a8&oh=00_AQGWTPnBEy6IX1Qn0kw9qn4ZormZNFns9TeIJqOi7fstxw&oe=6A71E2AD",
    }
]

st.write("---")
carousel(items=perfume_banners, width=1.0)
st.write("---")

st.subheader("💬 Chat with Perfume Zone AI")
for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

user_query=st.chat_input("Type your message here")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        response=st.write_stream(get_response(user_query,st.session_state.chat_history,vectorstore))
        st.session_state.chat_history.append(AIMessage(content=response))

    except Exception as e:
        st.exception(e)