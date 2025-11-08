"""
Data extraction module for the Sentiment Analysis project.

Purpose
-------
Load raw datasets from CSV/TSV/JSON/NDJSON into a clean pandas DataFrame with:
- Robust snake_case column normalization, including camelCase -> snake_case.
- Light dtype coercion for common numeric and datetime fields.
- Built-in expected schema for the provided Google Play reviews dataset.
- A minimal CLI to quickly verify files from the terminal.

Dataset schema (expected)
-------------------------
Original headers (camelCase)   -> Normalized headers (snake_case)
reviewId                       -> review_id
userName                       -> user_name
userImage                      -> user_image
content                        -> content
score                          -> score
thumbsUpCount                  -> thumbs_up_count
reviewCreatedVersion           -> review_created_version
at                             -> at
replyContent                   -> reply_content
repliedAt                      -> replied_at
sortOrder                      -> sort_order
appId                          -> app_id

Notes
-----
- CSV delimiter is auto-inferred among [',', ';', '\\t', '|'] so a tab-separated
  file named *.csv works without extra flags.
- Datetime fields are parsed as UTC when possible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence, Union, Iterable

import pandas as pd

SUPPORTED_EXTS = {".csv", ".tsv", ".json", ".ndjson"}

# Expected columns for your dataset (after normalization).
DEFAULT_EXPECTED_COLUMNS: Sequence[str] = [
    "review_id",
    "user_name",
    "user_image",
    "content",
    "score",
    "thumbs_up_count",
    "review_created_version",
    "at",
    "reply_content",
    "replied_at",
    "sort_order",
    "app_id",
]


class DataExtractionError(Exception):
    """Raised when data extraction or validation fails."""


def _to_snake(name: str) -> str:
    """
    Convert a string to snake_case.

    Steps:
    - Trim
    - Replace spaces/hyphens with underscores
    - Insert underscores before capitals in camelCase/PascalCase
    - Lowercase
    - Replace non-alnum (except underscore) with underscore
    - Collapse consecutive underscores
    """
    s = name.strip().replace(" ", "_").replace("-", "_")
    # Insert underscores before capitals: "reviewId" -> "review_Id"
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s)
    s = s.lower()
    s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)
    s = re.sub(r"_+", "_", s)
    return s


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with snake_case, ASCII-safe column names."""
    df = df.copy()
    df.columns = [_to_snake(str(c)) for c in df.columns]
    return df


def _coerce_common_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `df` with light dtype coercion.

    - Numeric: {"score", "thumbs_up_count"}
    - Datetime (UTC): {"at", "replied_at", "created_at", "timestamp"}
    """
    df = df.copy()

    for col in df.columns:
        if col in {"score", "thumbs_up_count"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        if col in {"at", "replied_at", "created_at", "timestamp"}:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    return df


def detect_format(path: Union[str, Path]) -> str:
    """Infer file format from extension."""
    ext = Path(path).suffix.lower()
    if ext in {".csv"}:
        return "csv"
    if ext in {".tsv"}:
        return "tsv"
    if ext in {".json", ".ndjson"}:
        return "json"
    raise DataExtractionError(f"Unsupported file extension: {ext}")


def _infer_sep(sample: str, candidates: Iterable[str] = (",", ";", "\t", "|")) -> str:
    """
    Infer delimiter by choosing the candidate that yields the most columns
    on the header line. Falls back to comma.
    """
    header = sample.splitlines()[0] if sample else ""
    if not header:
        return ","
    best_sep, best_count = ",", 1
    for sep in candidates:
        count = header.count(sep) + 1
        if count > best_count:
            best_sep, best_count = sep, count
    return best_sep


def load_raw_data(
    path: Union[str, Path],
    expected_columns: Optional[Sequence[str]] = DEFAULT_EXPECTED_COLUMNS,
    encoding: Optional[str] = "utf-8",
    csv_sep: Optional[str] = None,
    json_orient_records: bool = True,
) -> pd.DataFrame:
    """
    Load a dataset with validation and light normalization.

    Parameters
    ----------
    path : str | Path
        File path to the raw dataset. Supports CSV, TSV, JSON, NDJSON.
    expected_columns : sequence of str or None, default DEFAULT_EXPECTED_COLUMNS
        Expected column names after normalization. Use None to disable.
    encoding : str, default 'utf-8'
        Text encoding.
    csv_sep : str or None
        CSV delimiter override. If None, inferred from the header among
        [',', ';', '\\t', '|'].
    json_orient_records : bool, default True
        If True and JSON starts with '[', parse as a list of records.
        Otherwise parse as NDJSON.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise DataExtractionError(f"Unsupported file extension: {path.suffix}")

    fmt = detect_format(path)
    try:
        if fmt == "csv":
            if csv_sep is None:
                # Read a small sample to infer separator
                sample = path.open("r", encoding=encoding).readline()
                sep = _infer_sep(sample)
            else:
                sep = csv_sep
            df = pd.read_csv(path, encoding=encoding, sep=sep)
        elif fmt == "tsv":
            df = pd.read_csv(path, encoding=encoding, sep="\t")
        else:  # json or ndjson via content sniffing
            raw = path.read_text(encoding=encoding).strip()
            if not raw:
                raise DataExtractionError("Empty JSON file.")
            if raw[0] == "[" and json_orient_records:
                data = json.loads(raw)
                df = pd.DataFrame(data)
            else:
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
                df = pd.DataFrame(records)
    except pd.errors.EmptyDataError as e:
        raise DataExtractionError("Empty or malformed file.") from e
    except ValueError as e:
        raise DataExtractionError(f"Pandas failed to parse file: {e}") from e
    except json.JSONDecodeError as e:
        raise DataExtractionError(f"Invalid JSON: {e}") from e

    df = _normalize_columns(df)
    df = _coerce_common_types(df)

    if expected_columns is not None:
        expected_norm = {_to_snake(c) for c in expected_columns}
        missing = expected_norm.difference(df.columns)
        if missing:
            raise DataExtractionError(f"Missing expected columns: {sorted(missing)}")

    if df.empty:
        raise DataExtractionError("Loaded DataFrame is empty.")

    return df


if __name__ == "__main__":
    # Minimal CLI for quick checks.
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Load and validate a raw dataset.")
    p.add_argument("path", type=str, help="Path to dataset (csv/tsv/json/ndjson)")
    p.add_argument(
        "--no-default-expected",
        action="store_true",
        help="Disable built-in expected schema validation.",
    )
    p.add_argument("--expected", type=str, nargs="*", help="Extra expected column names")
    p.add_argument("--encoding", type=str, default="utf-8", help="Text encoding")
    p.add_argument("--sep", dest="csv_sep", type=str, default=None, help="CSV delimiter override")
    p.add_argument(
        "--json-array",
        dest="json_orient_records",
        action="store_true",
        help="Treat JSON as an array of records if it starts with '[' (default).",
    )
    p.add_argument(
        "--no-json-array",
        dest="json_orient_records",
        action="store_false",
        help="Force NDJSON parsing even if the JSON starts with '['.",
    )
    p.set_defaults(json_orient_records=True)

    args = p.parse_args()
    expected = None if args.no_default_expected else list(DEFAULT_EXPECTED_COLUMNS)
    if args.expected:
        # Merge user-provided expected with the default set (unless disabled)
        expected = (expected or []) + list(args.expected)

    try:
        df = load_raw_data(
            args.path,
            expected_columns=expected,
            encoding=args.encoding,
            csv_sep=args.csv_sep,
            json_orient_records=args.json_orient_records,
        )
        print(f"OK | rows={len(df)} cols={df.shape[1]} columns={list(df.columns)}")
    except Exception as e:
        print(f"ERROR | {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
