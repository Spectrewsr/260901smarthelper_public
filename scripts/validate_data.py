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


PARK_REQUIRED_FIELDS = (
    "park_id",
    "asset_id",
    "name",
    "district",
    "summary",
    "latitude",
    "longitude",
    "coordinate_source",
    "coordinate_precision",
    "confidence",
    "verified_date",
    "source_ids",
    "sector_fit",
    "metrics",
)

LANDING_SUPPORT_REQUIRED_FIELDS = (
    "service_id",
    "category",
    "name",
    "district",
    "summary",
    "source_ids",
    "confidence",
    "verified_date",
    "tags",
)

# ``source_id`` must be present as a JSON key, but an explicitly ``unknown``
# metric may leave its value blank.  The conditional rule is enforced below.
PARK_METRIC_REQUIRED_FIELDS = ("key", "label", "value", "confidence")
LANDING_SUPPORT_CATEGORIES = frozenset({"政务办事", "生活配套", "交通区位"})
CONFIDENCE_VALUES = frozenset(
    {
        "a",
        "b",
        "c",
        "high",
        "medium",
        "low",
        "unknown",
        "高",
        "中",
        "低",
        "待核验",
        "未核验",
    }
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


def _validate_confidence_values(
    report: ValidationReport,
    file_name: str,
    rows: Iterable[Mapping[str, Any]],
    field: str = "confidence",
    *,
    csv_rows: bool,
) -> None:
    """Keep confidence labels predictable without rejecting a usable record.

    This mirrors the existing company-file convention: confidence vocabulary is
    a quality warning, while a missing required value remains an error.
    """

    for index, row in enumerate(rows, start=2 if csv_rows else 1):
        row_number = _row_number(row, index)
        value = clean_text(row.get(field, ""))
        if value and value.casefold() not in CONFIDENCE_VALUES:
            report.warning(
                file_name,
                "confidence is best expressed as A/B/C, high/medium/low, or unknown",
                row=row_number,
                field=field,
            )


def _validate_nonempty_list_fields(
    report: ValidationReport,
    file_name: str,
    rows: Iterable[Mapping[str, Any]],
    fields: Iterable[str],
) -> None:
    """Require JSON arrays for multi-value JSONL fields, not string lookalikes."""

    for index, row in enumerate(rows, start=1):
        row_number = _row_number(row, index)
        for field in fields:
            value = row.get(field)
            if not isinstance(value, list):
                report.error(file_name, "must be a non-empty JSON array", row=row_number, field=field)
                continue
            if not value or any(not clean_text(item) for item in value):
                report.error(file_name, "must contain only non-blank values", row=row_number, field=field)


def _validate_jsonl_coordinates(
    report: ValidationReport,
    file_name: str,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    for index, row in enumerate(rows, start=1):
        row_number = _row_number(row, index)
        latitude = parse_coordinate(row.get("latitude"))
        longitude = parse_coordinate(row.get("longitude"))
        if latitude is None or not -90 <= latitude <= 90:
            report.error(file_name, "latitude must be between -90 and 90", row=row_number, field="latitude")
        if longitude is None or not -180 <= longitude <= 180:
            report.error(file_name, "longitude must be between -180 and 180", row=row_number, field="longitude")


def _validate_park_metrics(
    report: ValidationReport,
    parks: Iterable[Mapping[str, Any]],
    known_source_ids: set[str],
) -> None:
    """Validate park benchmark metrics and their per-metric evidence links."""

    file_name = "parks.jsonl"
    for index, park in enumerate(parks, start=1):
        row_number = _row_number(park, index)
        metrics = park.get("metrics")
        if not isinstance(metrics, list):
            # The JSON-array shape is already reported by the general helper.
            continue
        if not metrics:
            report.error(file_name, "must contain at least one benchmark metric", row=row_number, field="metrics")
            continue
        seen_keys: set[str] = set()
        for metric_index, metric in enumerate(metrics, start=1):
            field_prefix = f"metrics[{metric_index}]"
            if not isinstance(metric, Mapping):
                report.error(file_name, "metric must be a JSON object", row=row_number, field=field_prefix)
                continue
            if "source_id" not in metric:
                report.error(file_name, "required field is missing", row=row_number, field=f"{field_prefix}.source_id")
            for field in PARK_METRIC_REQUIRED_FIELDS:
                if not clean_text(metric.get(field, "")):
                    report.error(file_name, "required value is blank", row=row_number, field=f"{field_prefix}.{field}")
            key = clean_text(metric.get("key", ""))
            if key:
                if key in seen_keys:
                    report.error(file_name, "duplicate metric key within park", row=row_number, field=f"{field_prefix}.key")
                seen_keys.add(key)
            confidence = clean_text(metric.get("confidence", ""))
            if confidence and confidence.casefold() not in CONFIDENCE_VALUES:
                report.warning(
                    file_name,
                    "confidence is best expressed as A/B/C, high/medium/low, or unknown",
                    row=row_number,
                    field=f"{field_prefix}.confidence",
                )
            source_id = clean_text(metric.get("source_id", ""))
            if not source_id:
                if confidence.casefold() != "unknown":
                    report.error(
                        file_name,
                        "a metric without source_id must use confidence 'unknown'",
                        row=row_number,
                        field=f"{field_prefix}.source_id",
                    )
            elif source_id not in known_source_ids:
                report.error(
                    file_name,
                    f"references unknown source_id {source_id!r}",
                    row=row_number,
                    field=f"{field_prefix}.source_id",
                )


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


def _validate_reserved_company_ids(report: ValidationReport, rows: Iterable[Mapping[str, Any]]) -> None:
    """Keep reviewed raw IDs separate from the manual-record namespace."""

    for index, row in enumerate(rows, start=2):
        company_id = clean_text(row.get("company_id", ""))
        if company_id.upper().startswith("MAN-"):
            report.error(
                "companies.csv",
                "MAN- is reserved for reviewed CSV/manual intake records",
                row=index,
                field="company_id",
            )


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
    """Validate local source files and their cross-file citation links.

    ``parks.jsonl`` and ``landing_support.jsonl`` are validated whenever the
    full demo data bundle is present.  A deliberately small legacy fixture may
    omit both companion files, but a partially supplied bundle is always an
    error so a production import cannot silently lose either knowledge base.
    """

    paths = resolve_data_paths(data_dir)
    report = ValidationReport()

    # ``csv`` is imported lazily below to keep the public import surface small.
    companies_headers_ok = _validate_headers(report, paths.companies, COMPANY_REQUIRED_FIELDS)
    sources_headers_ok = _validate_headers(report, paths.sources, SOURCE_REQUIRED_FIELDS)

    companies = _safe_csv_rows(report, paths.companies) if companies_headers_ok else []
    sources = _safe_csv_rows(report, paths.sources) if sources_headers_ok else []
    sectors = _safe_jsonl_rows(report, paths.sectors)
    assets = _safe_jsonl_rows(report, paths.assets)
    parks_path = paths.raw / "parks.jsonl"
    landing_support_path = paths.raw / "landing_support.jsonl"
    has_parks = parks_path.is_file()
    has_landing_support = landing_support_path.is_file()
    parks = _safe_jsonl_rows(report, parks_path) if has_parks else []
    landing_support = _safe_jsonl_rows(report, landing_support_path) if has_landing_support else []
    if has_parks != has_landing_support:
        missing = landing_support_path if has_parks else parks_path
        report.error(missing.name, "required companion JSONL file is missing")

    report.counts = {
        "companies": len(companies),
        "landing_services": len(landing_support),
        "parks": len(parks),
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
        _validate_reserved_company_ids(report, companies)
        _validate_url_fields(report, "companies.csv", companies, ("source_url",), csv_rows=True)
        _validate_dates(report, "companies.csv", companies, ("source_date", "verified_date"), csv_rows=True)
        _validate_company_coordinates(report, companies)
        _validate_confidence_values(report, "companies.csv", companies, csv_rows=True)

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
    asset_ids = _validate_duplicate_ids(report, "regional_assets.jsonl", assets, "asset_id", csv_rows=False)
    _validate_url_fields(report, "regional_assets.jsonl", assets, ("source_urls",), csv_rows=False)
    _validate_dates(report, "regional_assets.jsonl", assets, ("published_date", "verified_date"), csv_rows=False)
    if source_ids:
        _validate_source_references(report, "sector_profiles.jsonl", sectors, source_ids, "source_ids", csv_rows=False)
        _validate_source_references(report, "regional_assets.jsonl", assets, source_ids, "source_ids", csv_rows=False)

    if has_parks:
        _validate_required_values(report, "parks.jsonl", parks, PARK_REQUIRED_FIELDS, csv_rows=False)
        _validate_duplicate_ids(report, "parks.jsonl", parks, "park_id", csv_rows=False)
        _validate_nonempty_list_fields(report, "parks.jsonl", parks, ("source_ids", "sector_fit", "metrics"))
        _validate_jsonl_coordinates(report, "parks.jsonl", parks)
        _validate_dates(report, "parks.jsonl", parks, ("verified_date",), csv_rows=False)
        _validate_confidence_values(report, "parks.jsonl", parks, csv_rows=False)
        if source_ids:
            _validate_source_references(report, "parks.jsonl", parks, source_ids, "source_ids", csv_rows=False)
            _validate_park_metrics(report, parks, source_ids)
        else:
            # A sources.csv error is already recorded; still validate the
            # metric structure so JSONL mistakes are visible in the same run.
            _validate_park_metrics(report, parks, set())

        for index, park in enumerate(parks, start=1):
            row_number = _row_number(park, index)
            asset_id = clean_text(park.get("asset_id", ""))
            if asset_id and asset_id not in asset_ids:
                report.error(
                    "parks.jsonl",
                    f"references unknown asset_id {asset_id!r}",
                    row=row_number,
                    field="asset_id",
                )

    if has_landing_support:
        _validate_required_values(
            report,
            "landing_support.jsonl",
            landing_support,
            LANDING_SUPPORT_REQUIRED_FIELDS,
            csv_rows=False,
        )
        _validate_duplicate_ids(report, "landing_support.jsonl", landing_support, "service_id", csv_rows=False)
        _validate_nonempty_list_fields(report, "landing_support.jsonl", landing_support, ("source_ids", "tags"))
        _validate_dates(report, "landing_support.jsonl", landing_support, ("verified_date",), csv_rows=False)
        _validate_confidence_values(report, "landing_support.jsonl", landing_support, csv_rows=False)
        for index, item in enumerate(landing_support, start=1):
            category = clean_text(item.get("category", ""))
            if category and category not in LANDING_SUPPORT_CATEGORIES:
                report.error(
                    "landing_support.jsonl",
                    "category must be one of 政务办事、生活配套、交通区位",
                    row=_row_number(item, index),
                    field="category",
                )
        if source_ids:
            _validate_source_references(
                report,
                "landing_support.jsonl",
                landing_support,
                source_ids,
                "source_ids",
                csv_rows=False,
            )

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
