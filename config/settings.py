"""Centralised application settings loaded from environment variables.

Uses ``pydantic-settings`` to parse ``.env`` files with type coercion,
validation, and sensible defaults so every module can simply do::

    from config.settings import settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────────────────────────────
# Resolve project root (two levels up from this file: config/settings.py)
# ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings sourced from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ─────────────────────────────────────────────────────
    groq_api_key: str = Field(
        default="",
        description="Groq Cloud API key (primary LLM provider).",
    )
    google_api_key: str = Field(
        default="",
        description="Google AI Studio API key (embeddings + fallback LLM).",
    )

    # ── Pinecone Cloud ─────────────────────────────────────────────────
    pinecone_api_key: str = Field(
        default="",
        description="Pinecone API key.",
    )
    # LlamaParse & Web Tools
    llama_parse_api_key: Optional[str] = Field(default=None)
    tavily_api_key: Optional[str] = Field(default=None)

    # Logging & Path
    pinecone_index_name: str = Field(
        default="finsight",
        description="Name of the Pinecone index.",
    )
    pinecone_environment: str = Field(
        default="us-east-1-aws",
        description="Pinecone environment/region.",
    )

    # ── Cohere (Fallback) ────────────────────────────────────────────
    cohere_api_key: str = Field(
        default="",
        description="Cohere API key for fallback LLM.",
    )

    # ── Langfuse Observability ───────────────────────────────────────
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # ── Model Configuration ──────────────────────────────────────────
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Primary Groq chat model.",
    )
    groq_fallback_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Smaller Groq model used when the primary is rate-limited.",
    )
    gemini_model: str = Field(
        default="gemini-3.6-flash",
        description="Google Gemini model used as a secondary fallback LLM.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace embedding model identifier.",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Dimensionality of the embedding vectors (384 for all-MiniLM-L6-v2).",
    )

    # ── Chunking Defaults ────────────────────────────────────────────
    chunk_size: int = Field(
        default=1024,
        description="Target chunk size (characters) for document splitting.",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between consecutive chunks.",
    )

    # ── Retrieval Defaults ───────────────────────────────────────────
    retrieval_top_k: int = Field(
        default=6,
        description="Number of top documents to retrieve.",
    )

    # ── LLM Behaviour ───────────────────────────────────────────────
    llm_temperature: float = Field(
        default=0.0,
        description="Temperature for LLM generation (0 = deterministic).",
    )
    llm_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in LLM response.",
    )
    llm_max_retries: int = Field(
        default=3,
        description="Maximum retries on transient LLM failures.",
    )

    # ── SEC EDGAR ────────────────────────────────────────────────────
    sec_edgar_email: str = Field(
        default="finsight@example.com",
        description="Email for SEC EDGAR fair-access identification.",
    )
    sec_edgar_company: str = Field(
        default="FinSight",
        description="Company name for SEC EDGAR identification.",
    )

    # ── Validators ───────────────────────────────────────────────────
    @field_validator("embedding_dimension")
    @classmethod
    def _positive_dim(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embedding_dimension must be positive")
        return v

    # ── Computed Paths ───────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        """Absolute path to the project root directory."""
        return _PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        """Root data directory."""
        return _PROJECT_ROOT / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Directory for raw downloaded filings."""
        path = _PROJECT_ROOT / "data" / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def processed_data_dir(self) -> Path:
        """Directory for processed / chunked documents."""
        path = _PROJECT_ROOT / "data" / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_dir(self) -> Path:
        """Directory for miscellaneous cache files."""
        path = _PROJECT_ROOT / "data" / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def has_groq_key(self) -> bool:
        """Return *True* if a Groq API key is configured."""
        return bool(self.groq_api_key and self.groq_api_key != "your_groq_api_key_here")

    def has_google_key(self) -> bool:
        """Return *True* if a Google API key is configured."""
        return bool(self.google_api_key and self.google_api_key != "your_google_api_key_here")

    def has_pinecone_config(self) -> bool:
        """Return *True* if Pinecone is configured."""
        return bool(
            self.pinecone_api_key
            and self.pinecone_api_key != "your_pinecone_api_key_here"
        )

    def has_cohere_key(self) -> bool:
        """Return *True* if Cohere is configured."""
        return bool(
            self.cohere_api_key
            and self.cohere_api_key != "your_cohere_api_key_here"
        )

    def has_langfuse_config(self) -> bool:
        """Return *True* if Langfuse credentials are configured."""
        return bool(
            self.langfuse_public_key
            and self.langfuse_public_key != "your_langfuse_public_key_here"
        )


# ── Module-level singleton ───────────────────────────────────────────
settings = Settings()
