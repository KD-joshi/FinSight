"""Simple exact-match semantic cache to avoid hitting rate limits for repeated queries."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_FILE = Path("query_cache.json")

class QueryCache:
    """A simple persistent exact-match query cache."""

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self.cache: dict[str, str] = {}
        self._load()

    def _load(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info("Loaded %d queries from cache.", len(self.cache))
            except Exception as e:
                logger.warning("Failed to load query cache: %s", e)
                self.cache = {}

    def _save(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save query cache: %s", e)

    def get(self, query: str) -> Optional[str]:
        """Get the cached answer for a query."""
        normalized_query = query.strip().lower()
        if normalized_query in self.cache:
            logger.info("Cache hit for query: %.50s", normalized_query)
            return self.cache[normalized_query]
        return None

    def set(self, query: str, answer: str):
        """Cache the answer for a query."""
        normalized_query = query.strip().lower()
        self.cache[normalized_query] = answer
        self._save()

# Global instance
query_cache = QueryCache()
