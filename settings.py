from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file

class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint: str = os.getenv("AZURE_ENDPOINT")
    model_name: str = os.getenv("MODEL_NAME")
    embedding_model: str = os.getenv("EMBEDDING_MODEL_NAME")  # e.g., "text-embedding-3-small"
    azure_api_version: str = os.getenv("AZURE_API_VERSION")
    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE")
    pinecone_vector_dimension: int = int(os.getenv("PINECONE_VECTOR_DIMENSION", 1536))

settings = Settings()
