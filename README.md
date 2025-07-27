
# 🧠 General Conference RAG Chatbot

This is a professional Retrieval-Augmented Generation (RAG) chatbot built with Azure OpenAI, Pinecone, and Streamlit. It allows users to ask questions about LDS General Conference talks and receive accurate, context-rich answers based on the retrieved content.

---

## 📁 Project Structure

```
mini_rag_chatbot/
│
├── ask_question.py             # Main CLI chatbot logic
├── ask_question_web.py         # Streamlit-based web UI version
├── embed_and_upload.py         # Script to embed and upload documents to Pinecone
├── settings.py                 # Environment and API config using Pydantic
├── .env                        # Stores secret keys and environment variables
├── requirements.txt            # Python dependencies
├── data/                       # Folder for CSV and source data files
├── notebooks/                  # Jupyter notebooks for data exploration
└── env/                        # Python virtual environment
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


## 📈 Key Steps

### 1. Setup
- Create a virtual environment using `venv`
- Install all dependencies via `requirements.txt`
- Setup `.env` and `settings.py` with API keys and configs

### 2. Document Preprocessing & Upload
- Load General Conference transcripts from CSV
- Clean and chunk the text into token-friendly segments
- Generate and store embeddings in Pinecone using `embed_and_upload.py`

### 3. Chatbot Interface
- Use `ask_question.py` for terminal interaction
- Use `ask_question_web.py` for a polished Streamlit web UI

---


## ⚠️ Key Issues & Solutions

- **Pinecone Import Issue**: Switched from old `pinecone-client` to new `pinecone` package.
- **Execution Policy Error**: Solved PowerShell script restrictions by adjusting system policy.
- **Model Attribute Error**: Fixed by renaming `chat_model` to `model_name` in both `.env` and `settings.py`.
- **Missing API Config**: Ensured `.env` was fully synced with `settings.py` and properly loaded.
- **Upload/Embedding Issue (Jorge Alberto Feedback)**: Fixed a bug where the upload script was not finding the correct CSV file or uploading all required metadata. Now, `embed_and_upload.py` robustly loads, chunks, and uploads all data fields, and the retrieval scripts display all metadata (source, speaker, title) as required.

---


## 🌍 How to Run This Project

### 🔹 Option 1: Terminal
```powershell
# Activate environment (PowerShell)
& .\env\Scripts\Activate.ps1

# Run the chatbot
python ask_question.py
```

### 🔹 Option 2: Web App (Streamlit)
```powershell
# Activate environment (PowerShell)
& .\env\Scripts\Activate.ps1

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


---

## 🛠️ Troubleshooting

- If you see errors about missing packages, run `pip install -r requirements.txt` after activating your environment.
- If the upload script (`embed_and_upload.py`) cannot find your CSV, ensure the path is correct and the file is in the `data/` folder.
- For PowerShell execution policy errors, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating your environment.
- If sources, speaker, or title are missing in retrieval results, re-run the upload script to refresh Pinecone with complete metadata.

---

## 👨🏽‍💻 Made with 💙 by Prosper Anosike
Intern – AI Developer | BYU-Pathway Worldwide | July 2025
