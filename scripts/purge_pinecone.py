#!/usr/bin/env python3
"""
Purge Pinecone Namespaces.

Usage:
  python scripts/purge_pinecone.py                  (Deletes everything in the index)
  python scripts/purge_pinecone.py --namespace <id> (Deletes a specific session namespace)
"""

import os
import sys
import argparse
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Purge Pinecone Vector Database.")
    parser.add_argument("--namespace", type=str, help="Specific namespace to delete.", default=None)
    args = parser.parse_args()

    # Load .env
    load_dotenv()
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("Error: PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in .env")
        sys.exit(1)

    try:
        from pinecone import Pinecone
    except ImportError:
        print("Error: Pinecone SDK not found. Run `pip install pinecone-client`")
        sys.exit(1)

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    try:
        index = pc.Index(index_name)
    except Exception as e:
        print(f"Error connecting to index '{index_name}': {e}")
        sys.exit(1)

    if args.namespace:
        print(f"Deleting all vectors in namespace '{args.namespace}'...")
        index.delete(delete_all=True, namespace=args.namespace)
        print(f"Namespace '{args.namespace}' successfully purged.")
    else:
        print(f"WARNING: You are about to delete ALL data across ALL namespaces in index '{index_name}'.")
        confirm = input("Type 'DELETE' to confirm: ")
        if confirm == "DELETE":
            # To delete all namespaces, we must fetch their names first
            stats = index.describe_index_stats()
            namespaces = stats.get('namespaces', {}).keys()
            
            if not namespaces:
                print("No namespaces found. The database is already empty.")
            else:
                for ns in namespaces:
                    print(f"Deleting namespace: {ns}")
                    index.delete(delete_all=True, namespace=ns)
                print("All namespaces successfully purged.")
        else:
            print("Operation aborted.")

if __name__ == "__main__":
    main()
