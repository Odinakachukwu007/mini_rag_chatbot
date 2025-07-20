# 🧠 General Conference RAG Chatbot

This is a **Mini Retrieval-Augmented Generation (RAG) chatbot** built with Azure OpenAI, Pinecone, and Streamlit. It allows users to ask questions about LDS General Conference talks and receive accurate, context-rich answers based on the retrieved content.

---

## 📁 Project Structure

```
mini_rag_chatbot/
│
├── ask_question.py             # Main CLI chatbot logic
├── ask_question_web.py         # Streamlit-based web UI version
├── embed_and_chunk.py          # Script to embed and chunk documents into Pinecone
├── settings.py                 # Environment and API config using Pydantic
├── .env                        # Stores secret keys and environment variables
├── requirements.txt            # Python dependencies
├── /chunks                    # Folder for saved chunked JSON files
└── /docs                      # Folder with LDS General Conference transcripts
```

---

## 🚀 What This Project Does

- Embeds General Conference talks using Azure OpenAI Embeddings
- Saves those embeddings to Pinecone for fast semantic search
- Accepts user queries and searches the most relevant chunks
- Sends those results back to GPT-3.5 Turbo to generate a high-quality answer
- Available both in **terminal mode** and with a **Streamlit web interface**

---

## ✅ Technologies Used

- 🧠 Azure OpenAI (Chat + Embeddings)
- 🌲 Pinecone (Vector Database)
- 🔤 tiktoken (Tokenizer)
- 🌐 Streamlit (Frontend Web App)
- 🐍 Python + Pydantic for config handling

---

## 📈 Key Steps I Took

### 1. Setup
- Created a virtual environment using `venv`
- Installed all dependencies via `requirements.txt`
- Setup `.env` and `settings.py` with API keys and configs

### 2. Document Preprocessing
- Loaded General Conference transcripts
- Cleaned and chunked the text into token-friendly segments
- Generated and stored embeddings in Pinecone

### 3. Chatbot Interface
- Created `ask_question.py` for terminal interaction
- Created `ask_question_web.py` using Streamlit for a beautiful web UI

---

## ⚠️ Key Errors I Solved

- **Pinecone Import Issue**: Switched from old `pinecone-client` to new `pinecone` package.
- **Execution Policy Error**: Solved PowerShell script restrictions by adjusting system policy.
- **Model Attribute Error**: Fixed by renaming `chat_model` to `model_name` in both `.env` and `settings.py`.
- **Missing API Config**: Ensured `.env` was fully synced with `settings.py` and properly loaded.

---

## 🌍 How to Run This Project

### 🔹 Option 1: Terminal
```bash
# Activate environment
.\env\Scripts\activate

# Run the chatbot
python ask_question.py
```

### 🔹 Option 2: Web App (Streamlit)
```bash
# Activate environment
.\env\Scripts\activate

# Run web app
streamlit run ask_question_web.py
```

---

## 📚 Sample Questions You Can Try

- What did President Nelson teach about personal revelation?
- What is the importance of the Holy Ghost in our daily lives?
- How did leaders encourage faith and endurance?

---

## 💼 Why This Project Stands Out

- Combines real-world tools (Pinecone, Azure OpenAI) in a real-world use case
- Demonstrates understanding of vector search, embedding, prompt engineering, and user interaction
- Includes both a command-line and a polished web UI
- Handles errors gracefully and uses professional coding practices

---

## 👨🏽‍💻 Made with 💙 by Prosper Anosike
Intern – AI Developer | BYU-Pathway Worldwide | July 2025
