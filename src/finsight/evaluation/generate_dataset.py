"""Generate a synthetic gold-standard evaluation dataset.

This script fetches a random subset of ingested SEC document chunks
from Qdrant and uses our Groq LLM to automatically generate diverse,
realistic financial questions and their ground-truth answers.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config.settings import settings
from finsight.utils.llm_provider import get_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("eval_dataset_gen")


def fetch_sample_documents(limit: int = 50) -> list[Document]:
    logger.info(f"Connecting to Pinecone to fetch {limit} sample documents...")
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    # Fetch a random vector or just query with a dummy vector to get points
    # Since we just need text chunks, we can query with zeros.
    dummy_vector = [0.0] * 384 # MiniLM-L6-v2 dimension
    
    response = index.query(
        vector=dummy_vector,
        top_k=limit,
        include_metadata=True,
        namespace="finsight" # default testing namespace
    )

    documents = []
    for match in response.matches:
        metadata = match.metadata or {}
        page_content = metadata.get("text", "")
        if len(page_content) < 300:
            continue
        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info(f"Fetched {len(documents)} valid documents for generation.")
    return documents


def generate_dataset(num_questions: int = 20) -> None:
    docs = fetch_sample_documents(limit=50)
    if not docs:
        logger.error("No documents fetched. Aborting.")
        return

    logger.info("Initializing Groq LLM for question generation...")
    from dotenv import dotenv_values
    env_vars = dotenv_values(".env")
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analyst. Read the following SEC filing extract and generate ONE realistic question that an investor might ask, which can be answered strictly using this text. Provide the ground-truth answer as well."),
        ("human", "Context:\n{context}\n\nRespond in JSON format with two keys: 'question' and 'ground_truth'.")
    ])
    
    chain = prompt | llm | JsonOutputParser()

    dataset = []
    logger.info(f"Generating {num_questions} synthetic test questions...")
    
    for i, doc in enumerate(docs[:num_questions]):
        try:
            result = chain.invoke({"context": doc.page_content})
            dataset.append({
                "question": result["question"],
                "ground_truth": result["ground_truth"],
                "context": doc.page_content,
                "metadata": json.dumps(doc.metadata)
            })
            logger.info(f"Generated question {i+1}/{num_questions}")
        except Exception as e:
            logger.warning(f"Failed to generate for doc {i}: {e}")

    eval_dir = settings.project_root / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "synthetic_eval_dataset.csv"
    
    df = pd.DataFrame(dataset)
    df.to_csv(out_path, index=False)
    
    logger.info(f"Successfully generated {len(df)} questions!")
    logger.info(f"Dataset saved to: {out_path}")


if __name__ == "__main__":
    generate_dataset()
