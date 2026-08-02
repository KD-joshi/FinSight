"""Download SEC filings (10-K, 10-Q) from EDGAR.

Uses the ``sec-edgar-downloader`` library to fetch filings for a list
of tickers and saves them into a structured ``data/raw/`` directory.

Usage (programmatic)::

    from src.finsight.ingestion.sec_downloader import download_filings

    paths = download_filings(
        tickers=["AAPL", "MSFT"],
        filing_types=["10-K", "10-Q"],
        num_filings=3,
    )

Usage (CLI)::

    python -m src.finsight.ingestion.sec_downloader \\
        --tickers AAPL MSFT \\
        --filings 10-K 10-Q \\
        --num 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

from sec_edgar_downloader import Downloader

from config.settings import settings

logger = logging.getLogger(__name__)

# Default filing types of interest for financial analysis
DEFAULT_FILING_TYPES: list[str] = ["10-K", "10-Q"]


def download_filings(
    tickers: Sequence[str],
    *,
    filing_types: Optional[Sequence[str]] = None,
    num_filings: int = 5,
    output_dir: Optional[Path] = None,
    after_date: Optional[date] = None,
    before_date: Optional[date] = None,
) -> list[Path]:
    """Download SEC filings for the given tickers.

    Parameters
    ----------
    tickers:
        Stock ticker symbols (e.g. ``["AAPL", "MSFT"]``).
    filing_types:
        SEC filing types to download (default: ``["10-K", "10-Q"]``).
    num_filings:
        Maximum number of filings *per ticker per type* to download.
    output_dir:
        Root directory to save filings (default: ``data/raw/``).
    after_date:
        Only download filings filed *after* this date.
    before_date:
        Only download filings filed *before* this date.

    Returns
    -------
    list[Path]
        Paths to directories where filings were saved.
    """
    filing_types = filing_types or DEFAULT_FILING_TYPES
    output_dir = output_dir or settings.raw_data_dir

    logger.info(
        "Downloading filings for tickers=%s, types=%s, num=%d → %s",
        list(tickers),
        list(filing_types),
        num_filings,
        output_dir,
    )

    # Initialise the SEC EDGAR downloader with identification
    downloader = Downloader(
        company_name=settings.sec_edgar_company,
        email_address=settings.sec_edgar_email,
        download_folder=str(output_dir),
    )

    downloaded_paths: list[Path] = []

    for ticker in tickers:
        ticker_upper = ticker.upper().strip()
        for filing_type in filing_types:
            logger.info(
                "  ↳ %s / %s (limit=%d)", ticker_upper, filing_type, num_filings
            )
            try:
                downloader.get(
                    filing_type,
                    ticker_upper,
                    limit=num_filings,
                    after=after_date,
                    before=before_date,
                )

                # The library saves to <output_dir>/sec-edgar-filings/<TICKER>/<TYPE>/
                filing_dir = output_dir / "sec-edgar-filings" / ticker_upper / filing_type
                if filing_dir.exists():
                    downloaded_paths.append(filing_dir)
                    count = sum(1 for _ in filing_dir.rglob("*") if _.is_file())
                    logger.info(
                        "    ✓ Downloaded %d file(s) → %s", count, filing_dir
                    )
                else:
                    logger.warning(
                        "    ⚠ Expected directory not found: %s", filing_dir
                    )

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "    ✗ Failed to download %s/%s: %s",
                    ticker_upper,
                    filing_type,
                    exc,
                )

    logger.info(
        "Download complete. %d filing director(ies) saved.", len(downloaded_paths)
    )
    return downloaded_paths


def list_downloaded_filings(output_dir: Optional[Path] = None) -> dict[str, list[Path]]:
    """List all previously downloaded filings grouped by ticker.

    Parameters
    ----------
    output_dir:
        Root directory to scan (default: ``data/raw/``).

    Returns
    -------
    dict[str, list[Path]]
        Mapping of ``TICKER`` → list of filing file paths.
    """
    output_dir = output_dir or settings.raw_data_dir
    sec_root = output_dir / "sec-edgar-filings"

    result: dict[str, list[Path]] = {}

    if not sec_root.exists():
        logger.info("No filings found at %s", sec_root)
        return result

    for ticker_dir in sorted(sec_root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        files = sorted(f for f in ticker_dir.rglob("*") if f.is_file())
        result[ticker_dir.name] = files

    return result


# ======================================================================
# CLI
# ======================================================================

def _parse_date(date_str: str) -> date:
    """Parse a ``YYYY-MM-DD`` date string."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download SEC filings from EDGAR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Ticker symbols (e.g. AAPL MSFT GOOGL).",
    )
    parser.add_argument(
        "--filings",
        nargs="+",
        default=DEFAULT_FILING_TYPES,
        help="Filing types to download (default: 10-K 10-Q).",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Max filings per ticker per type (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/raw/).",
    )
    parser.add_argument(
        "--after",
        type=_parse_date,
        default=None,
        help="Only filings after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--before",
        type=_parse_date,
        default=None,
        help="Only filings before this date (YYYY-MM-DD).",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    args = _build_parser().parse_args()

    paths = download_filings(
        tickers=args.tickers,
        filing_types=args.filings,
        num_filings=args.num,
        output_dir=args.output_dir,
        after_date=args.after,
        before_date=args.before,
    )

    if paths:
        print(f"\n✅ Saved to {len(paths)} director(ies):")
        for p in paths:
            print(f"   {p}")
    else:
        print("\n⚠️  No filings were downloaded.", file=sys.stderr)
        sys.exit(1)
