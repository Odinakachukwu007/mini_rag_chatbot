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

"""
Validate all required environment variables and settings.
"""
required_env_vars = [
    "AZURE_OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "PINECONE_INDEX_NAME", "PINECONE_NAMESPACE", "PINECONE_VECTOR_DIMENSION"
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"❌ Missing required environment variables: {', '.join(missing_vars)}")

# === Azure OpenAI Info ===
AZURE_ENDPOINT = "https://poa82-mdbpygav-eastus2.cognitiveservices.azure.com/"
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
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


# Create Pinecone Index only if not existing
if PINECONE_INDEX_NAME not in pc.list_indexes():
    print(f"📌 Index '{PINECONE_INDEX_NAME}' not found. Creating it...")
    try:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_VECTOR_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
        print(f"✅ Index '{PINECONE_INDEX_NAME}' created.")
    except Exception as create_err:
        if "ALREADY_EXISTS" in str(create_err):
            print(f"⚠️ Index '{PINECONE_INDEX_NAME}' already exists. Skipping creation.")
        else:
            print(f"❌ Error creating index: {create_err}")
            raise
else:
    print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists.")

index = pc.Index(PINECONE_INDEX_NAME)

# ===================
# Clear old embeddings (optional)
# ===================
print(f"⚠️ Clearing old embeddings in namespace '{PINECONE_NAMESPACE}'...")
try:
    index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
    print(f"✅ Cleared namespace '{PINECONE_NAMESPACE}'.")
except Exception as e:
    print(f"⚠️ Could not clear namespace '{PINECONE_NAMESPACE}'. It may not exist yet. Continuing...")

# ===================
# Load and Process Data
# ===================
csv_path = "notebooks/Data/general-conference-talks.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)

print(f"📂 Loaded {len(df)} conference talks.")
print("📌 Columns available:", list(df.columns))

# Determine which column holds the text
TEXT_COLUMN = "text" if "text" in df.columns else "content"

# Tokenizer for chunking
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, max_tokens=300):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# ===================
# Chunk the data + prepare metadata
# ===================
all_chunks = []
metadata_list = []


for _, row in tqdm(df.iterrows(), total=len(df)):
    # Handle missing text gracefully
    title = row.get("title", "")
    speaker = row.get("speaker", "")
    speaker_role = row.get("speaker_role", "")
    content = row.get(TEXT_COLUMN, "")
    source = row.get("source", row.get("url", ""))
    if pd.isna(content):
        content = ""

    # Only use actual speaker names, not talk titles or other fields
    if speaker.strip() == "" or speaker.strip().lower() == "the first vision":
        speaker = "Unknown Speaker"

    # Chunk only the content, not the combined metadata
    chunks = chunk_text(content)

    for chunk in chunks:
        all_chunks.append(chunk)
        metadata_list.append({
            "title": title,
            "speaker": speaker,
            "speaker_role": speaker_role,
            "content": chunk,
            "source": source,
            "text": chunk
        })

print(f"📝 Total chunks created: {len(all_chunks)}")

# ===================
# Save chunks to CSV (for reference)
# ===================
output_csv = "chunked_conference_talks.csv"
chunk_df = pd.DataFrame(metadata_list)
chunk_df.to_csv(output_csv, index=False)
print(f"💾 Chunked data saved to {output_csv}")

# ===================
# Generate Embeddings + Upload to Pinecone
# ===================
"""
Generate embeddings and upload to Pinecone in safe batches.
"""
batch_size = 100
max_batch_size = 100  # Azure OpenAI embedding API limit is typically 100
if batch_size > max_batch_size:
    print(f"⚠️ Batch size {batch_size} exceeds API limit. Using {max_batch_size} instead.")
    batch_size = max_batch_size

for i in tqdm(range(0, len(all_chunks), batch_size)):
    batch_texts = all_chunks[i:i + batch_size]
    batch_meta = metadata_list[i:i + batch_size]

    # Create embeddings
    try:
        response = client.embeddings.create(
            input=batch_texts,
            model=EMBEDDING_DEPLOYMENT
        )
        embeds = [record.embedding for record in response.data]
        ids = [f"id-{i+j}" for j in range(len(embeds))]
    except Exception as embed_err:
        print(f"❌ Error generating embeddings for batch {i//batch_size+1}: {embed_err}")
        continue

    # Upsert to Pinecone with metadata
    to_upsert = list(zip(ids, embeds, batch_meta))
    try:
        index.upsert(vectors=to_upsert, namespace=PINECONE_NAMESPACE)
    except Exception as upsert_err:
        print(f"❌ Error upserting to Pinecone for batch {i//batch_size+1}: {upsert_err}")

print("✅ Embedding and upload complete!")
