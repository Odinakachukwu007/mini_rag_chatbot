import os
import pandas as pd
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec

# ===================
# Load environment variables
# ===================
load_dotenv()

# === Azure OpenAI Info ===
AZURE_ENDPOINT = "https://poa82-mdbpygav-eastus2.cognitiveservices.azure.com/"
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # Paste in .env
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
AZURE_API_VERSION = "2024-12-01-preview"

# === Pinecone Info ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-demo")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_VECTOR_DIMENSION = int(os.getenv("PINECONE_VECTOR_DIMENSION", 1536))

# ===================
# Initialize Clients
# ===================
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone Index (only if not existing)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_VECTOR_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# ===================
# Load and Process Data
# ===================
df = pd.read_csv("mini_rag_chatbot/Notebooks/data/general-conference-talks.csv")
documents = df.apply(lambda row: f"{row['title']} - {row['speaker']}: {row['text']}", axis=1)

# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, max_tokens=300):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# Chunk the documents
all_chunks = []
for doc in documents:
    all_chunks.extend(chunk_text(doc))

metadata_list = [{"text": chunk} for chunk in all_chunks]

# ===================
# Generate Embeddings + Upload to Pinecone
# ===================
batch_size = 100
for i in tqdm(range(0, len(all_chunks), batch_size)):
    batch_texts = all_chunks[i:i + batch_size]
    batch_meta = metadata_list[i:i + batch_size]

    # Create embeddings
    response = client.embeddings.create(
        input=batch_texts,
        model=EMBEDDING_DEPLOYMENT
    )

    embeds = [record.embedding for record in response.data]
    ids = [f"id-{i+j}" for j in range(len(embeds))]

    # Upsert to Pinecone
    to_upsert = list(zip(ids, embeds, batch_meta))
    index.upsert(vectors=to_upsert, namespace=PINECONE_NAMESPACE)

print("✅ Embedding and upload complete!")
