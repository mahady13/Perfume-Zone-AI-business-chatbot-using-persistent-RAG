import streamlit as st
import os
from streamlit_carousel import carousel
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever



load_dotenv()
st.set_page_config(
    page_title="Perfume Zone AI - Fragrance Assistant",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://perfumezonebd.com',
        'Report a bug': "https://github.com/mahady13/Perfume-Zone-AI-business-chatbot-using-persistent-RAG/issues",
        'About': "AI-Powered Fragrance Assistant for Perfume Zone BD"
    }
)
st.title("✨ Perfume Zone AI Assistant ✨",text_alignment="center")
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

    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #f0f2f6;
    }

    .stButton button {
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .sidebar-content {
        padding: 1rem 0;
    }

    .sidebar-section {
        margin-bottom: 1.5rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🏪 Perfume Zone")
    st.header("📱 Contact Us")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button(
            label='🌐 Website',
            url='https://perfumezonebd.com',
            use_container_width=True
        )
    with col2:
        st.link_button(
            label='💬 WhatsApp',
            url='https://wa.me/1625338214?text=Ami%20ekti%20perfume%20oil%20kinte%20chai',
            use_container_width=True
        )
    st.link_button(
        label='📘 Facebook',
        url='https://www.facebook.com/perfumezone0/',
        use_container_width=True
    )

    st.markdown("---")

    st.header("👨‍💻 Developer")
    with st.container():
        st.markdown("""
            **Mohiuddin Mahady**  
            *BSc in Computer Science & Engineering*  
            Mymensingh Engineering College  
            *(Affiliated with Dhaka University)*

            ---
            🚀 **Built with:**
            - LangChain
            - Groq API
            - OpenRouter API
            - HuggingFace Embeddings
            - ChromaDB
            - Streamlit
        """)

    col3, col4 = st.columns([1, 1])
    with col3:
        st.link_button(
            "🔗 LinkedIn",
            "https://www.linkedin.com/in/mohiuddin-mahady/",
            use_container_width=True
        )
    with col4:
        st.link_button(
            "💻 GitHub",
            'https://www.github.com/mahady13',
            use_container_width=True
        )

# PRIMARY_MODEL="inclusionai/ling-3.0-flash:free"
# groq_llama="llama-3.1-8b-instant"
# groq_openai="openai/gpt-oss-120b"
# BACKUP_MODEL="openrouter/free"

# api_key=os.getenv("OPENROUTER_API_KEY")

@st.cache_resource
def load_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embeddings

embedding=load_embedding()

@st.cache_resource
def load_vectorstore():
    asset_directory="./assets"
    persist_directory='./chromadb'
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectorstore=Chroma(
            persist_directory=persist_directory,embedding_function=embedding,
        )
        return vectorstore
    # elif os.path.exists(asset_directory) and os.listdir(asset_directory):
    #     loader=PyPDFDirectoryLoader(asset_directory)
    #     docs=loader.load()
    #
    #     splitter=RecursiveCharacterTextSplitter(
    #         separators=["\n\n", "\n"],
    #         chunk_size=180,
    #         chunk_overlap=20
    #     )
    #     chunks=splitter.split_documents(docs)
    #     vectorstore=Chroma.from_documents(
    #         documents=chunks,
    #         embedding=embedding,
    #         persist_directory=persist_directory
    #     )
    #     return vectorstore
    return None

vectorstore=load_vectorstore()

MODEL_CONFIGS = [

    {
        "provider": "groq1",
        "model": "openai/gpt-oss-120b",
        "api_key": os.getenv("GROQ_API_KEY"),
        "class": ChatGroq
    },

    {
        "provider": "groq2",
        "model": "openai/gpt-oss-20b",
        "api_key": os.getenv("GROQ_API_KEY"),
        "class": ChatGroq
    },

    {
        "provider": "groq3",
        "model": "qwen/qwen3.6-27b",
        "api_key": os.getenv("GROQ_API_KEY"),
        "class": ChatGroq
    },

    {
        "provider": "openrouter1",
        "model": "openrouter/free",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
        "class": ChatOpenAI
    },

    {
        "provider": "openrouter2",
        "model": "mistralai/mistral-7b-instruct:free",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
        "class": ChatOpenAI
    }
]


@st.cache_resource
def get_llm_with_fallback():

    for config in MODEL_CONFIGS:
        try:
            if not config.get("api_key"):
                continue
            if config["provider"] == "groq1":
                llm = ChatGroq(
                    model=config["model"],
                    api_key=config["api_key"],
                    temperature=0.3,
                    max_tokens=1000,
                    timeout=20
                )

            elif config["provider"] == "groq2":
                llm = ChatGroq(
                    model=config["model"],
                    api_key=config["api_key"],
                    temperature=0.3,
                    max_tokens=500,
                    timeout=20
                )

            elif config["provider"] == "groq3":
                llm = ChatGroq(
                    model=config["model"],
                    api_key=config["api_key"],
                    temperature=0.3,
                    max_tokens=500,
                    timeout=20
                )

            elif config["provider"] == "openrouter2":
                llm = ChatOpenAI(
                    model=config["model"],
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                    temperature=0.3,
                    max_tokens=1000,
                    timeout=20
                )
            elif config['provider']=='openrouter1':
                llm=ChatOpenAI(
                    model=config["model"],
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                    temperature=0.3,
                    max_tokens=1000,
                    timeout=20
                )

            with st.expander("Running Model"):
                st.markdown(f"Using: {config['provider']} - {config['model']}")
            return llm

        except Exception as e:
            st.warning(f"⚠️{config['provider']} failed: {str(e)[:50]}")
            continue

    st.error("❌ No working LLM found. Please Contact Our Facebook Page or Whatsapp")
    return None


def get_response(user_query,chat_history,vectorstore):
    context=""
    if vectorstore is not None:
        vector_retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":10})
        all_docs = vectorstore._collection.get(include=['documents'])
        all_texts = all_docs['documents']
        bm25_retriever = BM25Retriever.from_texts(all_texts)
        bm25_retriever.k = 5
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

        relevant_docs=ensemble_retriever.invoke(user_query)
        context="\n\n".join([doc.page_content for doc in relevant_docs])
    trimmed_history = chat_history[-3:]
    template = """
    You are "Perfume Zone AI" - expert fragrance consultant for Perfume Zone BD (https://perfumezonebd.com). Specializing in premium Attar oils and sprays.

    ### PRICING (ALWAYS use this chart):
    3ml (roll-on): 110 TK | 6ml: 250 TK | 15ml: 450 TK | 30ml: 800 TK | 50ml: 1350 TK | 100ml: 2340 TK
    ⚠️ NEVER confuse with shipping: Dhaka 60tk | Suburban 100tk | Outside 120tk
    ️️⚠️ NEVER give discounts until total price is above 1500,if above 1500tk,either give free delivery or give 10% discounts.
    If customer wants to but,drive them to our website/whatsapp/phone number
    
    ### PRODUCT KNOWLEDGE:
    • Check context for Men's, Women's, Unisex, or New Arrivals
    • Mention fragrance notes
    • Suggest Top 10 Best Sellers when relevant
    • If exact perfume missing, recommend similar alternatives

    ### PRODUCT TYPES:
    • Spray: Halal alcohol-based
    • Roll-on: 100% perfume oil (alcohol-free)
    • 3ml ONLY roll-on, all other sizes both available

    ### CRITICAL INSTRUCTION:
    - If the exact product name (like "Creed Aventus") is found in the context, if its found written in the context then IT IS IN STOCK, ALWAYS confirm this.
    - ONLY recommend similar items only if the original is strictly confirmed missing from the context.
    - Be careful: Do not confuse a CLONE (যেমন Armaf) with the ORIGINAL (Creed). 
    - If you find the original, DO NOT recommend the clone.

    ### RESPONSE STYLE:
    • Reply in user's language (Bangla/English/Banglish)
    • Use emojis (professional,luxurious, cool vibes) sparingly
    • Show full price chart ONLY WHEN price queries
    • Be helpful, concise, and professional

    ### Context:
    {context}

    ### History:
    {chat_history}

    ### Question:
    {user_query}

    Provide accurate, helpful response. If unsure, suggest contacting 01625-338214.
    """
    prompt=ChatPromptTemplate.from_template(template)
    llm = get_llm_with_fallback()
    chain = prompt | llm | StrOutputParser()
    output=chain.stream({
    "context": context,
    "chat_history": trimmed_history,
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

st.markdown("---")
with st.container():
    carousel(items=perfume_banners, width=1.0)
st.markdown("---")

st.subheader("💬 Chat with Perfume Zone AI")
for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message("assistant",avatar="✨"):
            st.markdown(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("user",avatar="👤"):
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

st.markdown("---")
with st.expander("💡 Quick Tips for Using the Chat"):
    st.markdown("""
        - **Ask about prices:** "What's the price of Dior Sauvage?"
        - **Get recommendations:** "Recommend a fragrance for summer"
        - **Check availability:** "Do you have Creed Aventus in stock?"
        - **Women's fragrances:** "Show me women's floral perfumes"
        - **Best sellers:** "What are your top 10 selling fragrances?"
        - **Delivery info:** "What's the delivery charge for Dhaka?"
        - **Product types:** "What's the difference between spray and roll-on?"
    """)
st.sidebar.markdown("---")
if st.sidebar.checkbox("Stats", value=False):
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    st.sidebar.write(f"Vector Store: {'✅ Loaded' if vectorstore else '❌ Not Available'}")
    st.sidebar.write(f"Embedding Model: {'✅ Loaded' if embedding else '❌ Not Available'}")
    st.sidebar.write(f"Total Message You've Sent: {len(st.session_state.chat_history)}")
