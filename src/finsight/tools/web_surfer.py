import logging
import os
import tempfile
import requests
from urllib.parse import urlparse
from typing import Optional

from tavily import TavilyClient
from llama_parse import LlamaParse
from config.settings import settings
from finsight.ingestion.document_processor import chunk_document
from finsight.ingestion.vector_store import get_vector_store
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def surf_and_ingest(query: str, namespace: str = "finsight", extra_metadata: dict = None) -> list[Document]:
    """Search the web for a query, download relevant PDFs or docs, parse with LlamaParse, and ingest to Pinecone."""
    extra_metadata = extra_metadata or {}
    if not settings.tavily_api_key or not settings.llama_parse_api_key:
        logger.error("Missing API keys for Tavily or LlamaParse.")
        return []

    tavily = TavilyClient(api_key=settings.tavily_api_key)
    parser = LlamaParse(api_key=settings.llama_parse_api_key, result_type="markdown")
    vector_store = get_vector_store(namespace)

    logger.info(f"Surfing the web for: {query}")
    
    try:
        # Search for content
        response = tavily.search(query=query, search_depth="advanced", max_results=5)
        results = response.get("results", [])
        
        if not results:
            logger.warning("No web results found.")
            return []

        ingested_docs = []

        for res in results:
            url = res.get("url")
            title = res.get("title", "Web Document")
            content = res.get("content", "")
            
            logger.info(f"Found web resource: {title} ({url})")
            
            if url and url.endswith(".pdf"):
                logger.info("Downloading PDF for LlamaParse...")
                # Download PDF
                try:
                    pdf_resp = requests.get(url, timeout=15)
                    pdf_resp.raise_for_status()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_resp.content)
                        tmp_path = tmp.name
                        
                    logger.info("Parsing with LlamaParse...")
                    parsed_docs = parser.load_data(tmp_path)
                    
                    if parsed_docs:
                        markdown_text = "\n".join([d.text for d in parsed_docs])
                        metadata = {"source": url, "title": title, "type": "pdf_web"}
                        metadata.update(extra_metadata)
                        chunks = chunk_document(markdown_text, metadata=metadata)
                        
                        logger.info(f"Ingesting {len(chunks)} chunks from {title}...")
                        vector_store.add_documents(chunks)
                        ingested_docs.extend(chunks)
                        
                    os.remove(tmp_path)
                except Exception as e:
                    logger.error(f"Failed to process PDF {url}: {e}")
            else:
                # Raw text ingestion
                logger.info("Ingesting raw web text...")
                metadata = {"source": url, "title": title, "type": "html_web"}
                metadata.update(extra_metadata)
                chunks = chunk_document(content, metadata=metadata)
                if chunks:
                    vector_store.add_documents(chunks)
                    ingested_docs.extend(chunks)
                    
        return ingested_docs

    except Exception as e:
        logger.error(f"Web surfing failed: {e}")
        return []
