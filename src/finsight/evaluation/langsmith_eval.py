"""Run LangSmith evaluation on the generated synthetic dataset."""

from __future__ import annotations

import logging
import pandas as pd
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResult, run_evaluator

from config.settings import settings
from finsight.api.dependencies import get_rag_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("langsmith_eval")

# Initialize LangSmith Client (will now pick up LANGCHAIN_ENDPOINT from .env)
client = Client()

def predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Target function that wraps the LangGraph agent for evaluation."""
    question = inputs["question"]
    agent_graph = get_rag_agent()
    
    # Run the graph
    result = agent_graph.invoke(
        {
            "question": question,
            "max_retries": 1
        },
        config={
            "configurable": {
                "thread_id": "langsmith_eval",
                "skip_human_consent": True
            }
        }
    )
    
    # Return outputs to be evaluated
    return {
        "generation": result.get("generation", "No answer generated."),
        "documents": [doc.page_content for doc in result.get("documents", [])],
        "route": result.get("route", "unknown")
    }

@run_evaluator
def answer_length_evaluator(run, example) -> EvaluationResult:
    """A basic evaluator that checks if an answer was generated."""
    prediction = run.outputs.get("generation", "")
    score = 1.0 if len(prediction) > 20 else 0.0
    return EvaluationResult(key="has_answer", score=score)

@run_evaluator
def document_retrieval_evaluator(run, example) -> EvaluationResult:
    """Checks if the agent successfully retrieved context."""
    docs = run.outputs.get("documents", [])
    score = 1.0 if len(docs) > 0 else 0.0
    return EvaluationResult(key="has_context", score=score)

def run_evaluation(dataset_name: str = "FinSight Synthetic Dataset"):
    eval_dataset_path = settings.project_root / "data" / "eval" / "synthetic_eval_dataset.csv"
    if not eval_dataset_path.exists():
        logger.error(f"Dataset not found at {eval_dataset_path}")
        return

    logger.info("Loading test dataset...")
    df = pd.read_csv(eval_dataset_path)
    
    # Convert dataframe to LangSmith Dataset
    logger.info(f"Uploading dataset '{dataset_name}' to LangSmith...")
    
    # Try to find if dataset already exists, otherwise create
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        logger.info("Found existing dataset.")
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Synthetic dataset for evaluating FinSight RAG agent"
        )
        
        # Add examples
        inputs = [{"question": row["question"]} for _, row in df.iterrows()]
        # If we had ground truth, we'd add it to outputs here
        outputs = [{"ground_truth": row.get("ground_truth", "")} for _, row in df.iterrows()]
        
        client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset.id,
        )
        logger.info("Created new dataset and uploaded examples.")

    logger.info("Starting LangSmith evaluation...")
    
    experiment_results = evaluate(
        predict,
        data=dataset_name,
        evaluators=[answer_length_evaluator, document_retrieval_evaluator],
        experiment_prefix="finsight-agent-eval",
        max_concurrency=1, # Keep at 1 to prevent FlashRank concurrency crash
    )
    
    logger.info("Evaluation complete! View results at https://smith.langchain.com")

if __name__ == "__main__":
    run_evaluation("FinSight Varied Dataset")
