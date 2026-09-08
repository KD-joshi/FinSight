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
        response = tavily.search(
            query=query, 
            search_depth="advanced", 
            max_results=5,
            exclude_domains=["youtube.com", "forbes.com", "bloomberg.com", "wsj.com", "ft.com", "seekingalpha.com", "barrons.com"]
        )
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
            
            # SEC EDGAR index page fix: if URL is an index page, find the primary filing link
            if url and "sec.gov/Archives/edgar/data/" in url and (url.endswith("-index.htm") or url.endswith("-index.html")):
                try:
                    import re
                    logger.info(f"Detecting SEC EDGAR index page. Fetching actual filing link...")
                    # Add User-Agent because sec.gov blocks requests without a specific format
                    headers = {"User-Agent": "FinSight App (finsight@example.com)"}
                    idx_resp = requests.get(url, headers=headers, timeout=10)
                    idx_resp.raise_for_status()
                    
                    # Look for the primary document which usually matches the accession number or has a .htm extension
                    # The format is <a href="/Archives/edgar/data/...">
                    matches = re.findall(r'<a href="(/Archives/edgar/data/[^"]+\.htm(?:l)?)"', idx_resp.text)
                    if matches:
                        # Find the first one that is NOT an index page
                        primary_links = [m for m in matches if not m.endswith("-index.htm") and not m.endswith("-index.html")]
                        if primary_links:
                            new_url = "https://www.sec.gov" + primary_links[0]
                            logger.info(f"Resolved EDGAR index to primary document: {new_url}")
                            url = new_url
                            
                            # Refetch the content for the actual document
                            doc_resp = requests.get(url, headers=headers, timeout=10)
                            doc_resp.raise_for_status()
                            content = doc_resp.text
                except Exception as e:
                    logger.warning(f"Failed to resolve SEC index page {url}: {e}")

            
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
