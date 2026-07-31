# ✨ Perfume Zone AI
### Intelligent RAG-Powered Customer Support Chatbot for Perfume Zone

Perfume Zone AI is a Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, **ChromaDB**, and **OpenRouter LLMs**.

Instead of relying only on the language model, the chatbot retrieves information directly from the company's knowledge base (PDF documents), allowing it to answer customer questions with accurate product information, pricing, stock details, and business policies.

---

## 🚀 Live Demo

> Add your Streamlit deployment link here

```
https://perfumezoneai.streamlit.app
```

---
# ✨ Features

- 🤖 AI-powered customer support
- 📚 Retrieval-Augmented Generation (RAG)
- 📄 Reads company PDF documents automatically
- 💾 Persistent Chroma Vector Database
- 🌍 Multilingual Support (Bangla, English & Banglish)
- 🧠 Conversational Memory
- 🔍 Semantic Search
- ⚡ OpenRouter LLM Integration
- 🔄 Automatic Fallback Model
- 🖼️ Promotional Image Carousel
- 📱 Clean Streamlit Interface

---

# 🏗️ Tech Stack

### Frontend

- Streamlit

### LLM

- OpenRouter API
- Ling 3 Flash
- OpenRouter Auto Router (Fallback)

### RAG

- LangChain
- ChromaDB
- Recursive Character Text Splitter

### Embeddings

- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

### Document Loader

- PyPDFDirectoryLoader

---

# 📂 Project Structure

```
PerfumeZone-AI/
│
├── assets/                 # Knowledge base PDFs
├── chromadb/               # Persistent Vector Database
├── app.py                  # Main Streamlit App
├── requirements.txt
├── .env
├── README.md
└── screenshots/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/PerfumeZone-AI.git

cd PerfumeZone-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env`

```env
OPENROUTER_API_KEY=your_api_key_here
```

Run

```bash
streamlit run app.py
```

---

# 📚 How It Works

```text
Customer Question
        │
        ▼
Retrieve Relevant PDF Chunks
        │
        ▼
Generate Embeddings
        │
        ▼
Semantic Search (ChromaDB)
        │
        ▼
Relevant Context
        │
        ▼
OpenRouter LLM
        │
        ▼
Final AI Response
```

---

# 🧠 RAG Workflow

1. PDFs are loaded from the `assets` directory.
2. Documents are split into smaller chunks.
3. Chunks are converted into vector embeddings.
4. Embeddings are stored inside ChromaDB.
5. User asks a question.
6. Similar document chunks are retrieved.
7. Retrieved context is sent to the LLM.
8. The chatbot generates an accurate response.

---

# 🌍 Supported Languages

- 🇧🇩 Bangla
- 🇺🇸 English
- 🔀 Banglish

The chatbot automatically responds in the user's preferred language.

---

# ✨ Future Improvements

- Voice Assistant
- Product Recommendation Engine
- Order Tracking
- WhatsApp Integration
- Facebook Messenger Integration
- Admin Dashboard
- Analytics
- Customer Feedback Collection

---

# 🛠 Requirements

- Python 3.10+
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings
- OpenRouter API

---

# 👨‍💻 Developer

**Mohiuddin Mahady**

B.Sc. in Computer Science & Engineering

Mymensingh Engineering College  
(University of Dhaka Affiliated)

### Connect with me

LinkedIn:
https://linkedin.com/in/mohiuddin-mahady

GitHub:
https://github.com/mahady13

---

# ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub!

It motivates me to build more AI projects.

---

## License

This project is licensed under the MIT License.