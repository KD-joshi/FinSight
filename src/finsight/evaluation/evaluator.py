"""Run RAGAS evaluation on the generated synthetic dataset."""

from __future__ import annotations

import logging
import json
import pandas as pd
from datasets import Dataset

# Langchain
from langchain_core.documents import Document

import sys
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

# RAGAS
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
)

# App
from config.settings import settings
from src.finsight.main import _init_llm, _init_retriever
from src.finsight.agents.graph import build_rag_agent
from src.finsight.utils.embeddings import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("evaluator")


def run_evaluation() -> None:
    eval_dataset_path = settings.project_root / "data" / "eval" / "synthetic_eval_dataset.csv"
    if not eval_dataset_path.exists():
        logger.error(f"Dataset not found at {eval_dataset_path}. Please run generate_dataset.py first.")
        return

    logger.info("Loading test dataset...")
    df = pd.read_csv(eval_dataset_path)
    
    # Ragas needs specific column names: question, ground_truth, answer, contexts
    # We already have question, ground_truth. We need to fill answer and contexts using our agent.
    
    logger.info("Initializing Agent...")
    from dotenv import dotenv_values
    env_vars = dotenv_values(".env")
    primary_llm, fallback_llm = _init_llm(env_vars)
    retriever = _init_retriever(env_vars, settings.qdrant_collection_name)
    from finsight.utils.embeddings import get_embeddings
    embeddings = get_embeddings()
    agent_graph = build_rag_agent(retriever, primary_llm, fallback_llm)

    answers = []
    contexts_list = []

    logger.info(f"Running agent on {len(df)} questions...")
    for idx, row in df.iterrows():
        question = row["question"]
        logger.info(f"[{idx+1}/{len(df)}] Asking: {question}")
        
        try:
            # Run our LangGraph agent
            result = agent_graph.invoke(
                {
                    "question": question,
                    "max_retries": 1
                },
                config={"configurable": {"thread_id": f"eval_thread_{idx}"}}
            )
            
            # The agent state outputs "generation" and "documents"
            answers.append(result.get("generation", "No answer generated."))
            
            # Ragas expects a list of string contexts for each question
            docs = result.get("documents", [])
            contexts_list.append([doc.page_content for doc in docs])
            
        except Exception as e:
            logger.error(f"Agent failed on question {idx+1}: {e}")
            answers.append("Error generating answer.")
            contexts_list.append([])

    # Append to dataframe
    df["answer"] = answers
    df["contexts"] = contexts_list

    # Convert to HuggingFace Dataset (required by RAGAS)
    dataset = Dataset.from_pandas(df)

    logger.info("Starting RAGAS evaluation...")
    
    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
        ],
        llm=primary_llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    # Save results
    results_df = result.to_pandas()
    out_path = eval_dataset_path.parent / "evaluation_results.csv"
    results_df.to_csv(out_path, index=False)
    
    logger.info(f"Evaluation complete! Results saved to {out_path}")
    logger.info("\nAggregate Scores:")
    for metric_name, score in result.items():
        logger.info(f"- {metric_name}: {score:.4f}")


if __name__ == "__main__":
    run_evaluation()
