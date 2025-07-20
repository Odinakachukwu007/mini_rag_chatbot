import streamlit as st
from settings import settings
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken

# Page setup
st.set_page_config(page_title="General Conference Chatbot", page_icon="📖")
st.title("🧠 General Conference RAG Chatbot")
st.markdown("Ask me anything about LDS General Conference talks.")

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Setup Azure OpenAI client
try:
    client = AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_api_version,
        azure_endpoint=settings.azure_endpoint
    )
except Exception as e:
    st.error(f"⚠️ Azure OpenAI client failed to initialize: {e}")
    st.stop()

# Setup Pinecone client (using new official SDK)
try:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
except Exception as e:
    st.error(f"⚠️ Pinecone client failed to initialize: {e}")
    st.stop()

# Main input box
query = st.text_input("💬 Your question:")

if query:
    with st.spinner("🔎 Thinking..."):
        try:
            # Get embedding for the query
            embed_response = client.embeddings.create(
                input=[query],
                model=settings.embedding_model
            )
            query_embedding = embed_response.data[0].embedding

            # Search Pinecone
            results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            retrieved_texts = [match['metadata']['text'] for match in results['matches']]

            # Build context and messages
            context = "\n\n".join(retrieved_texts)
            system_msg = {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on LDS General Conference talks."
            }
            user_msg = {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }

            # Chat completion
            answer = client.chat.completions.create(
                model=settings.model_name,
                messages=[system_msg, user_msg],
                max_tokens=512,
                temperature=0.7
            )

            # Display answer
            st.markdown("### 🧠 Answer:")
            st.success(answer.choices[0].message.content.strip())

            # Display context
            st.markdown("### 📚 Retrieved Contexts:")
            for i, text in enumerate(retrieved_texts, 1):
                st.markdown(f"**{i}.** {text[:300]}...")

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
