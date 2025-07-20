import os
from openai import AzureOpenAI
from pinecone import Pinecone
from settings import settings
import tiktoken

# Initialize Azure OpenAI client
embedding_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=settings.azure_endpoint,
    api_key=settings.azure_openai_api_key,
)

chat_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=settings.azure_endpoint,
    api_key=settings.azure_openai_api_key,
)

# Initialize Pinecone
pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index_name)

# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Function to get embedding for user question
def get_embedding(text):
    response = embedding_client.embeddings.create(
        input=[text],
        model=settings.embedding_model
    )
    return response.data[0].embedding

# Function to ask question
def ask_question(question):
    print(f"\n🔍 Searching for answers to: {question}")
    
    # Get the embedding for the user question
    query_embedding = get_embedding(question)

    # Search Pinecone for similar vectors
    search_response = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    # Extract matched text chunks
    retrieved_chunks = [match.metadata['text'] for match in search_response.matches]
    context = "\n---\n".join(retrieved_chunks)

    # Prompt for GPT
    system_prompt = "You are a helpful assistant. Use the following conference talk excerpts to answer the question."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    # Call Chat Completion API
    response = chat_client.chat.completions.create(
        messages=messages,
        model=settings.model_name,
        temperature=0.7,
        max_tokens=800
    )

    # Output
    print("\n🧠 Answer:\n")
    print(response.choices[0].message.content)

# Run the script
if __name__ == "__main__":
    user_input = input("🤖 Ask a question about General Conference talks: ")
    ask_question(user_input)
