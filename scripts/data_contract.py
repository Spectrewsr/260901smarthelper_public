"""Shared data contract and deterministic loaders for the local demo.

The raw data is intentionally kept as simple, reviewable UTF-8 CSV/JSONL files.
This module is the one place where their column names are interpreted, so the
indexer, validator and web application do not gradually drift apart.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


COMPANY_REQUIRED_FIELDS = (
    "company_id",
    "company_name",
    "district",
    "industry_track",
    "summary",
    "source_id",
    "source_url",
    "verified_date",
    "confidence",
)

SOURCE_REQUIRED_FIELDS = (
    "source_id",
    "title",
    "publisher",
    "url",
    "accessed_date",
    "source_type",
)

SECTOR_REQUIRED_FIELDS = (
    "sector_id",
    "name",
    "overview",
    "source_ids",
    "source_urls",
    "verified_date",
)

ASSET_REQUIRED_FIELDS = (
    "asset_id",
    "name",
    "asset_type",
    "description",
    "source_ids",
    "source_urls",
    "verified_date",
    "confidence",
)

MULTI_VALUE_FIELDS = frozenset(
    {
        "aliases",
        "products",
        "capabilities",
        "input_materials",
        "output_products",
        "target_customer_industries",
        "supply_chain_role",
        "source_ids",
        "source_urls",
        "keywords",
        "upstream",
        "midstream",
        "downstream",
        "transport_modes",
        "serves_sectors",
    }
)

_LIST_SEPARATORS = re.compile(r"[，,；;|]+")


@dataclass(frozen=True)
class DataPaths:
    """Resolved paths for one data collection.

    ``data_dir`` may be either ``demo/data`` or ``demo/data/raw``.  Keeping this
    resolution in one place makes the tools usable before and after generated
    artifacts are added under ``data/derived``.
    """

    root: Path
    raw: Path
    derived: Path

    @property
    def companies(self) -> Path:
        return self.raw / "companies.csv"

    @property
    def sources(self) -> Path:
        return self.raw / "sources.csv"

    @property
    def sectors(self) -> Path:
        return self.raw / "sector_profiles.jsonl"

    @property
    def assets(self) -> Path:
        return self.raw / "regional_assets.jsonl"


def resolve_data_paths(data_dir: str | Path) -> DataPaths:
    """Resolve a data root without creating or modifying files."""

    supplied = Path(data_dir).expanduser().resolve()
    if supplied.name.lower() == "raw":
        return DataPaths(root=supplied.parent, raw=supplied, derived=supplied.parent / "derived")
    raw = supplied / "raw" if (supplied / "raw").is_dir() else supplied
    root = supplied if raw.name.lower() == "raw" else supplied
    # If callers pass a directory containing the CSV files directly, generated
    # files still live next to it in ``derived`` rather than under ``raw``.
    if raw == supplied:
        root = supplied
    return DataPaths(root=root, raw=raw, derived=root / "derived")


def clean_text(value: Any) -> str:
    """Return a safely stripped text value; CSV blanks become an empty string."""

    if value is None:
        return ""
    return str(value).replace("\ufeff", "").strip()


def split_values(value: Any) -> list[str]:
    """Split the project's comma/semicolon-separated cells deterministically."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = [clean_text(item) for item in value]
    else:
        raw_values = _LIST_SEPARATORS.split(clean_text(value))
    seen: set[str] = set()
    items: list[str] = []
    for raw in raw_values:
        item = clean_text(raw)
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            items.append(item)
    return items


def string_or_list(value: Any) -> str | list[str]:
    """Normalize a JSONL field without erasing its human-authored wording."""

    if isinstance(value, list):
        return split_values(value)
    return clean_text(value)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a UTF-8 (optionally BOM-prefixed) CSV with stable, trimmed keys."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        rows: list[dict[str, str]] = []
        for row in reader:
            # DictReader uses None for values past the header.  Ignore those
            # malformed extras here and let the validator give a useful error.
            rows.append(
                {
                    clean_text(key): clean_text(value)
                    for key, value in row.items()
                    if key is not None
                }
            )
        return rows


def read_csv_headers(path: str | Path) -> list[str]:
    """Read CSV headers even when the file has no data rows."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []
    return [clean_text(item) for item in header]


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON Lines and retain its line number for validation diagnostics."""

    rows: list[dict[str, Any]] = []
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{jsonl_path}:{line_number} must be a JSON object")
            item = {clean_text(key): value for key, value in item.items()}
            item["_line_number"] = line_number
            rows.append(item)
    return rows


def is_http_url(value: Any) -> bool:
    parsed = urlparse(clean_text(value))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_iso_date(value: Any) -> bool:
    text = clean_text(value)
    if not text:
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def parse_coordinate(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def normalize_company(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the public CSV row into one stable, app-facing record.

    The aliases make the retriever convenient for the user interface while the
    canonical fields remain exactly those documented in the raw CSV contract.
    """

    raw = {clean_text(key): clean_text(value) for key, value in row.items()}
    record: dict[str, Any] = {
        "company_id": raw.get("company_id", ""),
        "company_name": raw.get("company_name", raw.get("name", "")),
        "aliases": split_values(raw.get("aliases", "")),
        "english_name": raw.get("english_name", ""),
        "district": raw.get("district", ""),
        "park": raw.get("park", ""),
        "address": raw.get("address", ""),
        "latitude": parse_coordinate(raw.get("latitude", "")),
        "longitude": parse_coordinate(raw.get("longitude", "")),
        "industry_track": raw.get("industry_track", raw.get("sector", "")),
        "industry_subtrack": raw.get("industry_subtrack", raw.get("industry", "")),
        "supply_chain_role": split_values(raw.get("supply_chain_role", "")),
        "products": split_values(raw.get("products", "")),
        "capabilities": split_values(raw.get("capabilities", "")),
        "input_materials": split_values(raw.get("input_materials", "")),
        "output_products": split_values(raw.get("output_products", "")),
        "target_customer_industries": split_values(raw.get("target_customer_industries", "")),
        "summary": raw.get("summary", raw.get("description", raw.get("main_business", ""))),
        "source_ids": split_values(raw.get("source_id", raw.get("source_ids", ""))),
        "source_urls": split_values(raw.get("source_url", raw.get("source_urls", ""))),
        "source_date": raw.get("source_date", ""),
        "verified_date": raw.get("verified_date", ""),
        "confidence": raw.get("confidence", raw.get("confidence_level", "")),
        "notes": raw.get("notes", ""),
    }
    # UI-friendly aliases.  Do not rely on them in the data pipeline itself.
    record["name"] = record["company_name"]
    record["sector"] = record["industry_track"]
    record["industry"] = record["industry_subtrack"]
    record["description"] = record["summary"]
    return record


def normalize_source(row: Mapping[str, Any]) -> dict[str, str]:
    return {clean_text(key): clean_text(value) for key, value in row.items() if key != "_line_number"}


def normalize_json_record(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if key == "_line_number":
            continue
        if key in MULTI_VALUE_FIELDS:
            normalized[key] = split_values(value)
        elif isinstance(value, str):
            normalized[key] = clean_text(value)
        else:
            normalized[key] = value
    return normalized


def load_companies(data_dir: str | Path) -> list[dict[str, Any]]:
    paths = resolve_data_paths(data_dir)
    return [normalize_company(row) for row in read_csv_rows(paths.companies)]


def load_sources(data_dir: str | Path) -> list[dict[str, str]]:
    paths = resolve_data_paths(data_dir)
    return [normalize_source(row) for row in read_csv_rows(paths.sources)]


def load_sector_profiles(data_dir: str | Path) -> list[dict[str, Any]]:
    paths = resolve_data_paths(data_dir)
    return [normalize_json_record(row) for row in read_jsonl_rows(paths.sectors)]


def load_regional_assets(data_dir: str | Path) -> list[dict[str, Any]]:
    paths = resolve_data_paths(data_dir)
    return [normalize_json_record(row) for row in read_jsonl_rows(paths.assets)]


def company_document(company: Mapping[str, Any]) -> str:
    """Produce a stable evidence document used for vectorization and citations.

    Only fields coming from the public source dataset are emitted.  We avoid
    invented labels such as a company being a supplier simply because it happens
    to be in the same industry.
    """

    field_labels = (
        ("企业名称", "company_name"),
        ("英文名称", "english_name"),
        ("别名", "aliases"),
        ("所在区", "district"),
        ("园区", "park"),
        ("地址", "address"),
        ("产业赛道", "industry_track"),
        ("细分领域", "industry_subtrack"),
        ("产业链角色", "supply_chain_role"),
        ("产品", "products"),
        ("能力/工序", "capabilities"),
        ("输入材料", "input_materials"),
        ("输出产品", "output_products"),
        ("目标客户行业", "target_customer_industries"),
        ("公开简介", "summary"),
    )
    lines: list[str] = []
    for label, key in field_labels:
        value = company.get(key, "")
        if isinstance(value, (list, tuple, set)):
            rendered = "、".join(clean_text(item) for item in value if clean_text(item))
        else:
            rendered = clean_text(value)
        if rendered:
            lines.append(f"{label}：{rendered}")
    return "\n".join(lines)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json(value: Any) -> str:
    """Canonical JSON used for data fingerprints and generated JSONL."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_fingerprint(
    companies: Iterable[Mapping[str, Any]],
    sources: Iterable[Mapping[str, Any]] | None = None,
) -> str:
    """Hash indexable evidence plus citation metadata to detect stale indexes.

    A vector index does not change when a source title changes, but the generated
    citation JSONL does. Including referenced source metadata ensures a rebuild
    refreshes both artifacts together rather than leaving a stale traceability
    record beside fresh CSV data.
    """

    pairs = [
        {
            "company_id": clean_text(company.get("company_id", "")),
            "document": company_document(company),
        }
        for company in companies
    ]
    pairs.sort(key=lambda pair: pair["company_id"])
    source_payload: list[dict[str, Any]] = []
    if sources is not None:
        for source in sources:
            source_payload.append(
                {
                    clean_text(key): value
                    for key, value in source.items()
                    if clean_text(key) and clean_text(key) != "_line_number"
                }
            )
        source_payload.sort(key=lambda source: clean_text(source.get("source_id", "")))
    return sha256_text(stable_json({"companies": pairs, "sources": source_payload}))
