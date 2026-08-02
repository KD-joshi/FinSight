import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "finsight")
    dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    
    if not api_key:
        print("Error: PINECONE_API_KEY not found in .env")
        return

    print(f"Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating Serverless Index '{index_name}' with dimension {dimension}...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index '{index_name}' created successfully!")
    else:
        print(f"Index '{index_name}' already exists.")

if __name__ == "__main__":
    init_pinecone()
