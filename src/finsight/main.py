"""FinSight CLI — Interactive financial document Q&A.

Starts an interactive terminal session for querying SEC filings and
financial documents using the agentic RAG pipeline.

Usage:
    python -m finsight.main
    python -m finsight.main --collection finsight_sec_filings
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

# ======================================================================
# Constants
# ======================================================================

DEFAULT_COLLECTION = "finsight_docs"

BANNER = r"""
╔═══════════════════════════════════════════════════════════════╗
║   _____ _       ____  _       _     _                        ║
║  |  ___(_)_ __ / ___|(_) __ _| |__ | |_                     ║
║  | |_  | | '_ \\___ \| / _` | '_ \| __|                    ║
║  |  _| | | | | |___) | | (_| | | | | |_                     ║
║  |_|   |_|_| |_|____/|_|\__, |_| |_|\__|                    ║
║                          |___/                               ║
║                                                              ║
║  Agentic RAG for Financial Intelligence                      ║
║  Powered by LangGraph • Groq • Qdrant                       ║
╚═══════════════════════════════════════════════════════════════╝
"""

# Rich theme for consistent styling
FINSIGHT_THEME = Theme({
    "info": "dim cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",
    "question": "bold magenta",
    "answer": "white",
})

console = Console(theme=FINSIGHT_THEME)
logger = logging.getLogger("finsight")


# ======================================================================
# Initialization helpers
# ======================================================================

def _setup_logging(verbose: bool = False) -> None:
    """Configure structured logging with Rich handler.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
        )],
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


def _load_settings() -> dict[str, str]:
    """Load environment variables and validate required API keys.

    Returns:
        Dictionary of configuration values.

    Raises:
        SystemExit: If required API keys are missing.
    """
    # Load from .env file (project root)
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path)

    required_keys = {
        "GROQ_API_KEY": "https://console.groq.com",
        "GOOGLE_API_KEY": "https://aistudio.google.com",
        "QDRANT_URL": "https://cloud.qdrant.io",
        "QDRANT_API_KEY": "https://cloud.qdrant.io",
    }

    settings: dict[str, str] = {}
    missing: list[str] = []

    for key, signup_url in required_keys.items():
        value = os.getenv(key, "")
        if not value:
            missing.append(f"  • {key} — Get it at {signup_url}")
        settings[key] = value

    if missing:
        console.print("\n[error]Missing required API keys:[/error]")
        for m in missing:
            console.print(f"[warning]{m}[/warning]")
        console.print(
            "\n[info]Set them in your .env file or as environment variables.[/info]"
        )
        console.print(
            f"[info]Expected .env location: {env_path}[/info]\n"
        )
        sys.exit(1)

    # Optional keys
    settings["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    settings["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
    settings["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    return settings


def _init_llm(settings: dict[str, str]):
    """Initialize the primary LLM and small fallback LLM using the unified provider.

    Args:
        settings: Configuration dictionary with API keys.

    Returns:
        Tuple of (primary_llm, fallback_llm).
    """
    from finsight.utils.llm_provider import get_llm_with_fallback
    from config.settings import settings as global_settings

    primary_llm = get_llm_with_fallback()
    logger.info("Primary LLM initialized with 3-tier waterfall fallback (Groq -> Gemini -> Cohere)")

    fallback_llm = get_llm_with_fallback(max_tokens=256)
    logger.info("Small LLM initialized with 3-tier waterfall fallback (Groq -> Gemini -> Cohere)")

    return primary_llm, fallback_llm


def _init_retriever(settings: dict[str, str], collection_name: str):
    """Initialize the hybrid retriever (Qdrant + BM25).

    Connects to Qdrant Cloud, loads existing documents, and builds
    the BM25 index.

    Args:
        settings: Configuration dictionary with API keys.
        collection_name: Qdrant collection name.

    Returns:
        HybridRetriever instance.
    """
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from finsight.rag.retriever import HybridRetriever

    from finsight.utils.embeddings import get_embeddings
    embeddings = get_embeddings()

    # Connect to Qdrant Cloud
    qdrant_client = QdrantClient(
        url=settings["QDRANT_URL"],
        api_key=settings["QDRANT_API_KEY"],
    )

    # Check if collection exists
    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name not in collection_names:
        console.print(
            f"\n[warning]Collection '{collection_name}' not found in Qdrant.[/warning]"
        )
        console.print(
            "[info]Available collections: "
            f"{collection_names if collection_names else '(none)'}[/info]"
        )
        console.print(
            "\n[info]Run the ingestion pipeline first:[/info]"
        )
        console.print(
            "  python -m finsight.ingestion.sec_downloader --tickers AAPL TSLA GOOGL\n"
        )
        sys.exit(1)

    # Initialize vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    logger.info("Connected to Qdrant collection: %s", collection_name)

    # Load all documents from Qdrant for BM25 index
    # Scroll through the entire collection to build the sparse index
    all_points = []
    offset = None
    scroll_limit = 100

    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=scroll_limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)

        if next_offset is None:
            break
        offset = next_offset

    # Convert Qdrant points to LangChain Documents
    from langchain_core.documents import Document

    documents: list[Document] = []
    for point in all_points:
        payload = point.payload or {}
        page_content = payload.get("page_content", payload.get("text", ""))
        metadata = payload.get("metadata", {})

        # If metadata is stored flat in payload, extract known fields
        if not metadata:
            metadata = {
                k: v for k, v in payload.items()
                if k not in ("page_content", "text")
            }

        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info("Loaded %d documents from Qdrant for BM25 index.", len(documents))

    # Build hybrid retriever
    retriever = HybridRetriever(
        vector_store=vector_store,
        documents=documents,
        k=60,
    )

    return retriever


def _init_agent(retriever, primary_llm, fallback_llm):
    """Build the LangGraph agentic RAG pipeline.

    Args:
        retriever: HybridRetriever instance.
        primary_llm: Primary LLM for generation.
        fallback_llm: Fallback LLM for grading/routing.

    Returns:
        Compiled LangGraph agent.
    """
    from finsight.agents.graph import build_rag_agent

    agent = build_rag_agent(
        retriever=retriever,
        llm=primary_llm,
        fallback_llm=fallback_llm,
        max_retries=3,
    )
    logger.info("LangGraph agent compiled successfully.")
    return agent


# ======================================================================
# Interactive Q&A Loop
# ======================================================================

def _run_query(agent, question: str, thread_id: str) -> str:
    """Run a single query through the agent pipeline.

    Args:
        agent: Compiled LangGraph agent.
        question: User's question.
        thread_id: Conversation thread ID for checkpointing.

    Returns:
        Generated answer string.
    """
    config = {"configurable": {"thread_id": thread_id}}

    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse.langchain import CallbackHandler
        langfuse_handler = CallbackHandler()
        config["callbacks"] = [langfuse_handler]

    initial_state = {
        "question": question,
        "max_retries": 3,
    }

    with console.status("[info]Thinking...[/info]", spinner="dots"):
        result = agent.invoke(initial_state, config=config)

    return result.get("generation", "No answer generated.")


def _interactive_loop(agent) -> None:
    """Run the interactive Q&A loop.

    Args:
        agent: Compiled LangGraph agent.
    """
    thread_id = str(uuid.uuid4())

    console.print(
        "\n[info]Type your financial question below. "
        "Commands: /quit, /new (new conversation), /help[/info]\n"
    )

    while True:
        try:
            question = Prompt.ask("[question]❓ Ask FinSight[/question]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[info]Goodbye! 👋[/info]")
            break

        question = question.strip()
        if not question:
            continue

        # Handle commands
        if question.lower() in ("/quit", "/exit", "/q"):
            console.print("[info]Goodbye! 👋[/info]")
            break

        if question.lower() in ("/new", "/reset"):
            thread_id = str(uuid.uuid4())
            console.print("[success]Started new conversation.[/success]\n")
            continue

        if question.lower() in ("/help", "/?"):
            console.print(Panel(
                "[info]Commands:[/info]\n"
                "  /quit  — Exit FinSight\n"
                "  /new   — Start a new conversation\n"
                "  /help  — Show this help\n\n"
                "[info]Example questions:[/info]\n"
                "  • What was Apple's total revenue in fiscal year 2024?\n"
                "  • Compare Tesla and Google's net income growth\n"
                "  • What risk factors did AAPL disclose in their 10-K?\n"
                "  • How has Microsoft's gross margin changed over the last 3 years?",
                title="FinSight Help",
                border_style="cyan",
            ))
            continue

        # Run the query
        try:
            answer = _run_query(agent, question, thread_id)

            console.print()
            console.print(Panel(
                Markdown(answer),
                title="📊 FinSight Answer",
                border_style="green",
                padding=(1, 2),
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[warning]Query cancelled.[/warning]\n")
        except Exception as e:
            logger.exception("Error processing query.")
            console.print(f"\n[error]Error: {e}[/error]\n")


# ======================================================================
# Entry Point
# ======================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="FinSight — Agentic RAG for Financial Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m finsight.main\n"
            "  python -m finsight.main --collection my_filings --verbose\n"
        ),
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the FinSight CLI."""
    args = parse_args()

    # Setup
    _setup_logging(verbose=args.verbose)
    console.print(BANNER, style="bold cyan")

    # 1. Load settings & validate API keys
    console.print("[info]Loading configuration...[/info]")
    settings = _load_settings()
    console.print("[success]✓ Configuration loaded[/success]")

    # 2. Initialize LLM provider
    console.print("[info]Initializing LLM providers...[/info]")
    primary_llm, fallback_llm = _init_llm(settings)
    console.print("[success]✓ LLM providers ready[/success]")

    # 3. Initialize retriever
    console.print("[info]Connecting to Qdrant & building retriever...[/info]")
    retriever = _init_retriever(settings, args.collection)
    console.print("[success]✓ Hybrid retriever ready[/success]")

    # 4. Build the LangGraph agent
    console.print("[info]Compiling LangGraph agent...[/info]")
    agent = _init_agent(retriever, primary_llm, fallback_llm)
    console.print("[success]✓ Agent compiled and ready[/success]")

    # 5. Start interactive loop
    _interactive_loop(agent)


if __name__ == "__main__":
    main()
