"""
Unit tests for src/data_extraction.py

Scope
-----
- Separator inference for tab-separated files named *.csv
- Column normalization (camelCase -> snake_case)
- Expected schema validation
- JSON array and NDJSON parsing
- Type coercion for numeric and datetime fields
- CLI smoke test

Test layout assumptions
-----------------------
Project root:
- src/data_extraction.py
- tests/unit/test_data_extraction.py  (this file)

These tests add <project_root>/src to sys.path explicitly.
"""

from _future_ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make src importable without installing as a package
PROJECT_ROOT = Path(_file_).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_extraction import (  # noqa: E402
    DataExtractionError,
    load_raw_data,
    detect_format,
)

# Common headers from the provided dataset (tab-separated in the real file)
RAW_HEADERS = [
    "reviewId",
    "userName",
    "userImage",
    "content",
    "score",
    "thumbsUpCount",
    "reviewCreatedVersion",
    "at",
    "replyContent",
    "repliedAt",
    "sortOrder",
    "appId",
]

NORMALIZED_HEADERS = [
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


def _write_tab_csv(path: Path, rows: list[list[str]]) -> None:
    """Write a tab-separated file with RAW_HEADERS and rows."""
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(RAW_HEADERS) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def test_load_csv_tab_separated_infers_sep(tmp_path: Path) -> None:
    """
    A tab-separated file named *.csv should be parsed correctly and columns normalized.
    """
    p = tmp_path / "dataset.csv"
    _write_tab_csv(
        p,
        rows=[
            [
                "gp:123",
                "Alice",
                "http://img",
                "Good app",
                "5",
                "10",
                "1.2.3",
                "2020-10-27 21:24:41",
                "",
                "",
                "newest",
                "com.anydo",
            ]
        ],
    )

    df = load_raw_data(p)
    assert list(df.columns) == NORMALIZED_HEADERS
    # Type coercion
    assert pd.api.types.is_numeric_dtype(df["score"])
    assert pd.api.types.is_numeric_dtype(df["thumbs_up_count"])
    assert isinstance(df["at"].dtype, pd.DatetimeTZDtype)
    # Values
    assert df.loc[0, "score"] == 5
    assert df.loc[0, "thumbs_up_count"] == 10
    assert str(df.loc[0, "sort_order"]) == "newest"


def test_missing_expected_columns_raises(tmp_path: Path) -> None:
    """
    If a required column is missing after normalization, raise DataExtractionError.
    """
    p = tmp_path / "bad.csv"
    # Drop "appId"
    headers = RAW_HEADERS[:-1]
    with p.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        f.write("\t".join(["x"] * len(headers)) + "\n")

    with pytest.raises(DataExtractionError) as err:
        load_raw_data(p)  # uses built-in expected schema
    assert "Missing expected columns" in str(err.value)


def test_json_array_parsing(tmp_path: Path) -> None:
    """
    JSON array of records should load and normalize columns.
    """
    p = tmp_path / "data.json"
    payload = [
        {
            "reviewId": "gp:1",
            "userName": "Bob",
            "userImage": "http://img",
            "content": "ok",
            "score": 3,
            "thumbsUpCount": 0,
            "reviewCreatedVersion": "1.0",
            "at": "2020-10-27 08:18:40",
            "replyContent": None,
            "repliedAt": None,
            "sortOrder": "newest",
            "appId": "com.anydo",
        }
    ]
    p.write_text(json.dumps(payload), encoding="utf-8")

    df = load_raw_data(p)
    assert set(NORMALIZED_HEADERS).issubset(df.columns)
    assert isinstance(df["at"].dtype, pd.DatetimeTZDtype)
    assert pd.notna(df.loc[0, "at"])


def test_ndjson_parsing(tmp_path: Path) -> None:
    """
    NDJSON should load when the file does not start with '['.
    """
    p = tmp_path / "data.ndjson"
    lines = [
        json.dumps(
            {
                "reviewId": "gp:2",
                "userName": "Eve",
                "userImage": "",
                "content": "meh",
                "score": "2",
                "thumbsUpCount": "1",
                "reviewCreatedVersion": "",
                "at": "2020-10-27 14:03:28",
                "replyContent": "",
                "repliedAt": "",
                "sortOrder": "newest",
                "appId": "com.anydo",
            }
        )
    ]
    p.write_text("\n".join(lines), encoding="utf-8")

    df = load_raw_data(p, json_orient_records=False)
    assert "review_id" in df.columns
    assert df.loc[0, "score"] == 2
    assert df.loc[0, "thumbs_up_count"] == 1
    assert pd.notna(df.loc[0, "at"])


def test_detect_format_unsupported_extension(tmp_path: Path) -> None:
    """
    Unsupported extension should raise in detect_format or load.
    """
    p = tmp_path / "data.xlsx"
    p.write_text("dummy", encoding="utf-8")
    with pytest.raises(DataExtractionError):
        detect_format(p)


def test_cli_smoke_ok(tmp_path: Path) -> None:
    """
    Run the module as a script and verify a successful 'OK |' line.
    """
    # Build a minimal valid tab-separated file
    ds = tmp_path / "cli.csv"
    _write_tab_csv(
        ds,
        rows=[
            [
                "gp:42",
                "Zed",
                "",
                "works",
                "4",
                "7",
                "",
                "2020-10-27 10:00:00",
                "",
                "",
                "newest",
                "com.anydo",
            ]
        ],
    )

    script = SRC_DIR / "data_extraction.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(ds)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "OK | rows=1" in proc.stdout