#!/usr/bin/env python3
"""
Re-ingestion Script for Archived Sessions.

If a session's Pinecone namespace was purged by the TTL cleanup job,
this script can be used to read the session history from SQLite,
extract any URLs or context that were discussed, and re-ingest them
into Pinecone under the same session namespace.

Usage:
  python scripts/reingest_session.py <thread_id>
"""

import os
import sys
import sqlite3
import re
from dotenv import load_dotenv

# We can reuse the web surfer to re-ingest URLs
try:
    from finsight.tools.web_surfer import surf_and_ingest
except ImportError:
    surf_and_ingest = None

def extract_urls(text: str) -> list[str]:
    """Extract URLs from a text string."""
    url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
    return url_pattern.findall(text)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/reingest_session.py <thread_id>")
        sys.exit(1)
        
    thread_id = sys.argv[1]
    
    load_dotenv()
    db_path = "checkpoints.sqlite"
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        sys.exit(1)
        
    try:
        conn = sqlite3.connect(db_path)
        # LangGraph checkpoints are binary blobs. We will use the langgraph API to read them.
        
        import asyncio
        from langgraph.checkpoint.sqlite import SqliteSaver
        from finsight.api.dependencies import get_rag_agent
        
        checkpointer = SqliteSaver(conn)
        
        # We need the compiled graph to read the state
        agent = get_rag_agent()
        config = {"configurable": {"thread_id": thread_id}}
        
        state = agent.get_state(config)
        
        if not state or not hasattr(state, 'values') or 'chat_history' not in state.values:
            print(f"No chat history found for thread {thread_id}.")
            sys.exit(0)
            
        messages = state.values['chat_history']
        
        print(f"Found {len(messages)} messages in session {thread_id}.")
        
        urls_to_ingest = set()
        
        for msg in messages:
            urls = extract_urls(msg.content)
            urls_to_ingest.update(urls)
            
        if not urls_to_ingest:
            print("No URLs found in chat history to re-ingest.")
            print("Note: User-uploaded PDFs cannot be recovered automatically unless saved to disk.")
            sys.exit(0)
            
        print(f"Found {len(urls_to_ingest)} URLs to re-ingest: {urls_to_ingest}")
        
        if surf_and_ingest:
            for url in urls_to_ingest:
                print(f"Re-ingesting {url}...")
                # We can trick the surfer to just download the URL
                surf_and_ingest(url, namespace=thread_id)
            print("Re-ingestion complete!")
        else:
            print("Web surfer module not found. Cannot re-ingest automatically.")
            
    except Exception as e:
        print(f"Failed to process session: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
