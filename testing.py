from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="test_key")  # Fake key just for testing
print("✅ Pinecone module imported and initialized (expected to fail on key).")
