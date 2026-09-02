"""Validate the public, citation-first data files used by the local demo.

The validator intentionally does *not* fetch URLs.  It checks traceability and
file consistency without turning a reproducible local build into a network
operation.

Examples
--------
    python scripts/validate_data.py --data-dir data
    python scripts/validate_data.py --data-dir data --min-companies 100 --strict
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # Supports both ``python scripts/...`` and ``import scripts...``.
    from .data_contract import (
        ASSET_REQUIRED_FIELDS,
        COMPANY_REQUIRED_FIELDS,
        SECTOR_REQUIRED_FIELDS,
        SOURCE_REQUIRED_FIELDS,
        clean_text,
        is_http_url,
        is_iso_date,
        parse_coordinate,
        read_csv_headers,
        read_csv_rows,
        read_jsonl_rows,
        resolve_data_paths,
        split_values,
    )
except ImportError:  # pragma: no cover - exercised by the CLI invocation.
    from data_contract import (  # type: ignore
        ASSET_REQUIRED_FIELDS,
        COMPANY_REQUIRED_FIELDS,
        SECTOR_REQUIRED_FIELDS,
        SOURCE_REQUIRED_FIELDS,
        clean_text,
        is_http_url,
        is_iso_date,
        parse_coordinate,
        read_csv_headers,
        read_csv_rows,
        read_jsonl_rows,
        resolve_data_paths,
        split_values,
    )


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    file: str
    row: int | None
    field: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "file": self.file,
            "row": self.row,
            "field": self.field,
            "message": self.message,
        }


class ValidationReport:
    """Small, serializable report returned to build scripts and the API."""

    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []
        self.counts: dict[str, int] = {}

    def add(
        self,
        severity: str,
        file: str,
        message: str,
        *,
        row: int | None = None,
        field: str | None = None,
    ) -> None:
        self.issues.append(ValidationIssue(severity, file, row, field, message))

    def error(self, file: str, message: str, *, row: int | None = None, field: str | None = None) -> None:
        self.add("error", file, message, row=row, field=field)

    def warning(self, file: str, message: str, *, row: int | None = None, field: str | None = None) -> None:
        self.add("warning", file, message, row=row, field=field)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "counts": dict(sorted(self.counts.items())),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _row_number(row: Mapping[str, Any], default: int) -> int:
    value = row.get("_line_number")
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _validate_required_values(
    report: ValidationReport,
    file_name: str,
    rows: Iterable[Mapping[str, Any]],
    fields: Iterable[str],
    *,
    csv_rows: bool,
) -> None:
    for index, row in enumerate(rows, start=2 if csv_rows else 1):
        row_number = _row_number(row, index)
        for field in fields:
            if not clean_text(row.get(field, "")):
                report.error(file_name, "required value is blank", row=row_number, field=field)


def _validate_duplicate_ids(
    report: ValidationReport,
    file_name: str,
    rows: Iterable[Mapping[str, Any]],
    id_field: str,
    *,
    csv_rows: bool,
) -> set[str]:
    seen: dict[str, int] = {}
    identifiers: set[str] = set()
    for index, row in enumerate(rows, start=2 if csv_rows else 1):
        row_number = _row_number(row, index)
        identifier = clean_text(row.get(id_field, ""))
        if not identifier:
            continue
        if identifier in seen:
            report.error(
                file_name,
                f"duplicate {id_field!r}; first used on row {seen[identifier]}",
                row=row_number,
                field=id_field,
            )
        else:
            seen[identifier] = row_number
            identifiers.add(identifier)
    return identifiers


def _validate_headers(
    report: ValidationReport,
    path: Path,
    required_fields: Iterable[str],
) -> bool:
    file_name = path.name
    if not path.is_file():
        report.error(file_name, "required data file is missing")
        return False
    try:
        headers = read_csv_headers(path)
    except (OSError, UnicodeError, csv.Error) as exc:  # type: ignore[name-defined]
        report.error(file_name, f"cannot read CSV header: {exc}")
        return False
    if not headers:
        report.error(file_name, "CSV has no header row")
        return False
    for field in required_fields:
        if field not in headers:
            report.error(file_name, "required column is missing", field=field)
    duplicates = sorted({header for header in headers if headers.count(header) > 1})
    for header in duplicates:
        report.error(file_name, "duplicate column name", field=header)
    return True


def _safe_csv_rows(report: ValidationReport, path: Path) -> list[dict[str, str]]:
    try:
        return read_csv_rows(path)
    except (OSError, UnicodeError, csv.Error) as exc:  # type: ignore[name-defined]
        report.error(path.name, f"cannot read CSV rows: {exc}")
        return []


def _safe_jsonl_rows(report: ValidationReport, path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        report.error(path.name, "required data file is missing")
        return []
    try:
        return read_jsonl_rows(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        report.error(path.name, f"cannot read JSONL: {exc}")
        return []


def _validate_url_fields(
    report: ValidationReport,
    file_name: str,
    rows: Iterable[Mapping[str, Any]],
    fields: Iterable[str],
    *,
    csv_rows: bool,
) -> None:
    for index, row in enumerate(rows, start=2 if csv_rows else 1):
        row_number = _row_number(row, index)
        for field in fields:
            values = split_values(row.get(field, "")) if field.endswith("urls") else [clean_text(row.get(field, ""))]
            for value in values:
                if value and not is_http_url(value):
                    report.error(
                        file_name,
                        "must be an absolute http(s) URL",
                        row=row_number,
                        field=field,
                    )


def _validate_dates(
    report: ValidationReport,
    file_name: str,
    rows: Iterable[Mapping[str, Any]],
    fields: Iterable[str],
    *,
    csv_rows: bool,
) -> None:
    for index, row in enumerate(rows, start=2 if csv_rows else 1):
        row_number = _row_number(row, index)
        for field in fields:
            value = clean_text(row.get(field, ""))
            if value and not is_iso_date(value):
                report.error(file_name, "must use ISO date YYYY-MM-DD", row=row_number, field=field)


def _validate_company_coordinates(report: ValidationReport, rows: Iterable[Mapping[str, Any]]) -> None:
    for index, row in enumerate(rows, start=2):
        latitude_text = clean_text(row.get("latitude", ""))
        longitude_text = clean_text(row.get("longitude", ""))
        if bool(latitude_text) != bool(longitude_text):
            report.error(
                "companies.csv",
                "latitude and longitude must be provided together",
                row=index,
                field="latitude/longitude",
            )
            continue
        if not latitude_text:
            continue
        latitude = parse_coordinate(latitude_text)
        longitude = parse_coordinate(longitude_text)
        if latitude is None or not -90 <= latitude <= 90:
            report.error("companies.csv", "latitude must be between -90 and 90", row=index, field="latitude")
        if longitude is None or not -180 <= longitude <= 180:
            report.error("companies.csv", "longitude must be between -180 and 180", row=index, field="longitude")


def _validate_source_references(
    report: ValidationReport,
    file_name: str,
    rows: Iterable[Mapping[str, Any]],
    known_source_ids: set[str],
    field: str = "source_id",
    *,
    csv_rows: bool,
) -> None:
    plural_field = "source_ids" if field == "source_id" else field
    for index, row in enumerate(rows, start=2 if csv_rows else 1):
        row_number = _row_number(row, index)
        raw = row.get(field, row.get(plural_field, ""))
        for source_id in split_values(raw):
            if source_id not in known_source_ids:
                report.error(
                    file_name,
                    f"references unknown source_id {source_id!r}",
                    row=row_number,
                    field=field,
                )


def validate_data(data_dir: str | Path, *, min_companies: int = 1) -> ValidationReport:
    """Validate all four source files and their cross-file citation links."""

    paths = resolve_data_paths(data_dir)
    report = ValidationReport()

    # ``csv`` is imported lazily below to keep the public import surface small.
    companies_headers_ok = _validate_headers(report, paths.companies, COMPANY_REQUIRED_FIELDS)
    sources_headers_ok = _validate_headers(report, paths.sources, SOURCE_REQUIRED_FIELDS)

    companies = _safe_csv_rows(report, paths.companies) if companies_headers_ok else []
    sources = _safe_csv_rows(report, paths.sources) if sources_headers_ok else []
    sectors = _safe_jsonl_rows(report, paths.sectors)
    assets = _safe_jsonl_rows(report, paths.assets)

    report.counts = {
        "companies": len(companies),
        "regional_assets": len(assets),
        "sector_profiles": len(sectors),
        "sources": len(sources),
    }
    if len(companies) < min_companies:
        report.error(
            "companies.csv",
            f"contains {len(companies)} companies; expected at least {min_companies}",
        )

    if companies_headers_ok:
        _validate_required_values(report, "companies.csv", companies, COMPANY_REQUIRED_FIELDS, csv_rows=True)
        _validate_duplicate_ids(report, "companies.csv", companies, "company_id", csv_rows=True)
        _validate_url_fields(report, "companies.csv", companies, ("source_url",), csv_rows=True)
        _validate_dates(report, "companies.csv", companies, ("source_date", "verified_date"), csv_rows=True)
        _validate_company_coordinates(report, companies)
        for index, row in enumerate(companies, start=2):
            confidence = clean_text(row.get("confidence", ""))
            if confidence and confidence.casefold() not in {"high", "medium", "low", "高", "中", "低"}:
                report.warning(
                    "companies.csv",
                    "confidence is best expressed as high/medium/low (or 高/中/低)",
                    row=index,
                    field="confidence",
                )

    source_ids: set[str] = set()
    if sources_headers_ok:
        _validate_required_values(report, "sources.csv", sources, SOURCE_REQUIRED_FIELDS, csv_rows=True)
        source_ids = _validate_duplicate_ids(report, "sources.csv", sources, "source_id", csv_rows=True)
        _validate_url_fields(report, "sources.csv", sources, ("url",), csv_rows=True)
        _validate_dates(report, "sources.csv", sources, ("published_date", "accessed_date"), csv_rows=True)

    if source_ids:
        _validate_source_references(report, "companies.csv", companies, source_ids, csv_rows=True)

    _validate_required_values(report, "sector_profiles.jsonl", sectors, SECTOR_REQUIRED_FIELDS, csv_rows=False)
    _validate_duplicate_ids(report, "sector_profiles.jsonl", sectors, "sector_id", csv_rows=False)
    _validate_url_fields(report, "sector_profiles.jsonl", sectors, ("source_urls",), csv_rows=False)
    _validate_dates(report, "sector_profiles.jsonl", sectors, ("verified_date",), csv_rows=False)
    _validate_required_values(report, "regional_assets.jsonl", assets, ASSET_REQUIRED_FIELDS, csv_rows=False)
    _validate_duplicate_ids(report, "regional_assets.jsonl", assets, "asset_id", csv_rows=False)
    _validate_url_fields(report, "regional_assets.jsonl", assets, ("source_urls",), csv_rows=False)
    _validate_dates(report, "regional_assets.jsonl", assets, ("published_date", "verified_date"), csv_rows=False)
    if source_ids:
        _validate_source_references(report, "sector_profiles.jsonl", sectors, source_ids, "source_ids", csv_rows=False)
        _validate_source_references(report, "regional_assets.jsonl", assets, source_ids, "source_ids", csv_rows=False)

    return report


def _format_human(report: ValidationReport) -> str:
    lines = [
        f"Validation {'passed' if report.ok else 'failed'}: "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s).",
        "Counts: " + ", ".join(f"{key}={value}" for key, value in sorted(report.counts.items())),
    ]
    for issue in report.issues:
        position = issue.file
        if issue.row is not None:
            position += f":{issue.row}"
        if issue.field:
            position += f" [{issue.field}]"
        lines.append(f"{issue.severity.upper():7} {position} — {issue.message}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the demo's local public-data files.")
    parser.add_argument("--data-dir", default="data", help="data root or data/raw directory (default: data)")
    parser.add_argument("--min-companies", type=int, default=1, help="minimum expected company records")
    parser.add_argument("--strict", action="store_true", help="treat warnings as a non-zero exit")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = parser.parse_args(argv)
    if args.min_companies < 1:
        parser.error("--min-companies must be at least 1")

    report = validate_data(args.data_dir, min_companies=args.min_companies)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(_format_human(report))
    return 0 if report.ok and (not args.strict or not report.warnings) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
