"""Read-only adapter between the web app and the local data/retrieval pipeline.

The data scripts deliberately live outside ``app``.  This adapter accepts the
pipeline's CSV/JSONL artifacts when they arrive, while keeping the UI usable
when a dataset has not yet been built.
"""

from __future__ import annotations

import csv
import importlib
import json
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DataPaths:
    """Resolved locations for a data bundle in either supported layout."""

    companies: Path | None
    sources: Path | None
    assets: Path | None


class CompanyRepository:
    """Load companies and delegate semantic matching to ``scripts/retriever``.

    ``LocalRetriever`` is intentionally optional.  Until an embedding index has
    been created, a small lexical fallback lets the local demo remain truthful
    and functional rather than returning a fabricated AI result.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.project_dir = self.data_dir.parent
        self._lock = threading.RLock()
        self._fingerprint: tuple[tuple[str, int, int], ...] | None = None
        self._companies: list[dict[str, Any]] = []
        self._company_by_id: dict[str, dict[str, Any]] = {}
        self._sources_by_id: dict[str, dict[str, Any]] = {}
        self._regional_assets: list[dict[str, Any]] = []
        self._retriever: Any | None = None
        self._retriever_error: str | None = None
        self._retrieval_backend = "lexical fallback"
        self._paths = DataPaths(None, None, None)

    # ---- Public API -----------------------------------------------------

    def status(self) -> dict[str, Any]:
        self._ensure_loaded()
        ready = bool(self._companies)
        return {
            "state": "ready" if ready else "waiting",
            "company_count": len(self._companies),
            "retrieval_backend": self._retrieval_backend,
            "message": (
                f"已加载 {len(self._companies)} 家企业公开资料"
                if ready
                else "企业资料尚未接入。请先运行数据准备与索引脚本。"
            ),
            "data_path": str(self._paths.companies) if self._paths.companies else None,
        }

    def filter_options(self) -> dict[str, list[str]]:
        self._ensure_loaded()
        return {
            "districts": self._distinct("district"),
            "sectors": self._distinct("sector"),
            "roles": self._distinct_list("supply_chain_role"),
        }

    def list_companies(
        self,
        *,
        query: str = "",
        district: str | None = None,
        sector: str | None = None,
        page: int = 1,
        page_size: int = 24,
    ) -> dict[str, Any]:
        self._ensure_loaded()
        page = max(page, 1)
        page_size = min(max(page_size, 1), 100)
        filtered = self._filter(self._companies, district=district, sector=sector)

        needle = query.strip().casefold()
        if needle:
            filtered = [item for item in filtered if needle in self._searchable_text(item)]

        total = len(filtered)
        start = (page - 1) * page_size
        return {
            "items": [self._summary(item) for item in filtered[start : start + page_size]],
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": start + page_size < total,
        }

    def get_company(self, company_id: str) -> dict[str, Any] | None:
        self._ensure_loaded()
        item = self._company_by_id.get(str(company_id))
        return dict(item) if item else None

    def search(
        self,
        query: str,
        *,
        district: str | None = None,
        sector: str | None = None,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Return evidence candidates, preferring the pipeline's local index."""

        self._ensure_loaded()
        filters = {key: value for key, value in {"district": district, "sector": sector}.items() if value}
        if not self._companies:
            return []

        if self._retriever is not None:
            try:
                raw_results = self._retriever.search(query, filters=filters or None, top_k=top_k)
                normalized = [self._normalize_retrieval_result(item) for item in raw_results]
                normalized = [item for item in normalized if item.get("name")]
                if normalized:
                    return normalized[:top_k]
            except Exception as exc:  # The page must still work when an index is stale.
                self._retriever_error = type(exc).__name__
                self._retrieval_backend = "lexical fallback"

        return self._lexical_search(query, district=district, sector=sector, top_k=top_k)

    def match_business_need(
        self,
        query: str,
        *,
        district: str | None = None,
        sector: str | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Classify evidence into potential roles without asserting a transaction."""

        self._ensure_loaded()
        filters = {key: value for key, value in {"district": district, "sector": sector}.items() if value}
        if self._retriever is not None and hasattr(self._retriever, "match_business_need"):
            try:
                payload = self._retriever.match_business_need(query, filters=filters or None, top_k=top_k)
                if isinstance(payload, dict):
                    result: dict[str, Any] = {}
                    for key in ("suppliers", "customers", "partners", "all_matches"):
                        result[key] = [
                            self._normalize_retrieval_result(item)
                            for item in payload.get(key, [])
                            if isinstance(item, dict)
                        ]
                    result["rationale"] = str(payload.get("rationale", ""))
                    if any(result[key] for key in ("suppliers", "customers", "partners", "all_matches")):
                        return result
            except Exception as exc:
                self._retriever_error = type(exc).__name__
                self._retrieval_backend = "lexical fallback"

        matches = self._lexical_search(query, district=district, sector=sector, top_k=top_k)
        suppliers: list[dict[str, Any]] = []
        customers: list[dict[str, Any]] = []
        partners: list[dict[str, Any]] = []
        for item in matches:
            roles = " ".join(item.get("supply_chain_role", [])).casefold()
            if any(term in roles for term in ("供应", "材料", "零部件", "supplier", "upstream")):
                suppliers.append(item)
            elif any(term in roles for term in ("客户", "整机", "应用", "customer", "downstream")):
                customers.append(item)
            else:
                partners.append(item)

        return {
            "suppliers": suppliers[:4],
            "customers": customers[:4],
            "partners": partners[:4],
            "all_matches": matches,
            "rationale": "根据公开产品、能力与产业链角色字段进行的潜在匹配，不代表已建立业务关系。",
        }

    def regional_assets(self, district: str | None = None, limit: int = 6) -> list[dict[str, Any]]:
        self._ensure_loaded()
        if not self._regional_assets:
            return []
        district = (district or "").strip()
        direct = [asset for asset in self._regional_assets if district and district in self._asset_area(asset)]
        return (direct or self._regional_assets)[:limit]

    # ---- Data loading ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        paths = self._resolve_paths()
        fingerprint = self._make_fingerprint(paths)
        if fingerprint == self._fingerprint:
            return

        with self._lock:
            # Another request can have refreshed data while this request waited.
            paths = self._resolve_paths()
            fingerprint = self._make_fingerprint(paths)
            if fingerprint == self._fingerprint:
                return

            self._paths = paths
            self._sources_by_id = self._load_sources(paths.sources)
            self._companies = self._load_companies(paths.companies)
            self._company_by_id = {item["company_id"]: item for item in self._companies}
            self._regional_assets = self._load_assets(paths.assets)
            self._retriever = None
            self._retriever_error = None
            self._retrieval_backend = "lexical fallback"
            if self._companies:
                self._try_create_retriever()
            self._fingerprint = fingerprint

    def _resolve_paths(self) -> DataPaths:
        roots = (self.data_dir, self.data_dir / "raw")

        def first_existing(filename: str) -> Path | None:
            return next((root / filename for root in roots if (root / filename).is_file()), None)

        return DataPaths(
            companies=first_existing("companies.csv"),
            sources=first_existing("sources.csv"),
            assets=first_existing("regional_assets.jsonl"),
        )

    def _make_fingerprint(self, paths: DataPaths) -> tuple[tuple[str, int, int], ...]:
        output: list[tuple[str, int, int]] = []
        for path in (paths.companies, paths.sources, paths.assets):
            if path and path.exists():
                stat = path.stat()
                output.append((str(path), stat.st_mtime_ns, stat.st_size))
        # A freshly built vector index can appear without modifying the raw CSV.
        # Fingerprinting it lets an already-running local server switch from the
        # fallback retriever on the next request instead of requiring a restart.
        for filename in ("company_index.npz", "company_documents.jsonl"):
            path = self.data_dir / "derived" / filename
            if path.is_file():
                stat = path.stat()
                output.append((str(path), stat.st_mtime_ns, stat.st_size))
        return tuple(output)

    def _load_sources(self, path: Path | None) -> dict[str, dict[str, Any]]:
        if not path:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for row in self._read_csv(path):
            source_id = self._value(row, "source_id", "id", "来源ID")
            if not source_id:
                continue
            result[source_id] = {
                "source_id": source_id,
                "title": self._value(row, "title", "source_title", "名称", "标题") or "公开资料来源",
                "url": self._value(row, "url", "source_url", "链接"),
                "published_date": self._value(row, "published_date", "source_date", "发布日期"),
                "publisher": self._value(row, "publisher", "发布单位", "来源"),
            }
        return result

    def _load_companies(self, path: Path | None) -> list[dict[str, Any]]:
        if not path:
            return []
        records: list[dict[str, Any]] = []
        for position, row in enumerate(self._read_csv(path), start=1):
            name = self._value(row, "company_name", "name", "企业名称")
            if not name:
                continue
            source_ids = self._list_value(self._value(row, "source_id", "source_ids", "来源ID"))
            direct_url = self._value(row, "source_url", "url", "来源链接")
            sources = [self._sources_by_id[source_id] for source_id in source_ids if source_id in self._sources_by_id]
            if direct_url and not any(source.get("url") == direct_url for source in sources):
                sources.append(
                    {
                        "source_id": source_ids[0] if source_ids else f"direct-{position}",
                        "title": "企业公开资料",
                        "url": direct_url,
                        "published_date": self._value(row, "source_date", "发布日期"),
                        "publisher": "",
                    }
                )
            company_id = self._value(row, "company_id", "id", "企业ID") or f"company-{position:03d}"
            records.append(
                {
                    "company_id": str(company_id),
                    "name": name,
                    "aliases": self._list_value(self._value(row, "aliases", "alias", "别名")),
                    "english_name": self._value(row, "english_name", "英文名称"),
                    "district": self._value(row, "district", "区县", "区域"),
                    "park": self._value(row, "park", "园区"),
                    "address": self._value(row, "address", "地址"),
                    "latitude": self._value(row, "latitude", "纬度"),
                    "longitude": self._value(row, "longitude", "经度"),
                    "sector": self._value(row, "industry_track", "sector", "赛道", "产业赛道"),
                    "industry": self._value(row, "industry_subtrack", "industry", "行业", "细分行业"),
                    "products": self._list_value(self._value(row, "products", "product", "产品")),
                    "capabilities": self._list_value(self._value(row, "capabilities", "能力", "核心工序")),
                    "supply_chain_role": self._list_value(
                        self._value(row, "supply_chain_role", "supply_chain_roles", "产业链角色")
                    ),
                    "input_materials": self._list_value(self._value(row, "input_materials", "输入材料")),
                    "output_products": self._list_value(self._value(row, "output_products", "输出产品")),
                    "target_customer_industries": self._list_value(
                        self._value(row, "target_customer_industries", "目标客户行业")
                    ),
                    "summary": self._value(row, "summary", "description", "简介", "企业简介"),
                    "source_ids": source_ids,
                    "sources": sources,
                    "source_date": self._value(row, "source_date", "发布日期"),
                    "verified_date": self._value(row, "verified_date", "核验日期"),
                    "confidence": self._value(row, "confidence", "可信度"),
                    "notes": self._value(row, "notes", "备注"),
                }
            )
        return records

    @staticmethod
    def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
        if not path:
            return []
        records: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    value = json.loads(line)
                    if isinstance(value, dict):
                        records.append(value)
        except (OSError, json.JSONDecodeError):
            return []
        return records

    def _load_assets(self, path: Path | None) -> list[dict[str, Any]]:
        """Attach source cards to regional assets using the same source table."""

        records = self._load_jsonl(path)
        for asset in records:
            source_ids = self._list_value(asset.get("source_ids") or asset.get("source_id"))
            source_urls = self._list_value(asset.get("source_urls") or asset.get("source_url"))
            sources = [self._sources_by_id[source_id] for source_id in source_ids if source_id in self._sources_by_id]
            known_urls = {str(source.get("url") or "") for source in sources}
            for index, url in enumerate(source_urls, start=1):
                if url and url not in known_urls:
                    sources.append(
                        {
                            "source_id": source_ids[index - 1] if len(source_ids) >= index else f"asset-{index}",
                            "title": "区域资源公开资料",
                            "url": url,
                            "published_date": asset.get("published_date") or asset.get("source_date") or "",
                            "publisher": "",
                        }
                    )
            asset["sources"] = self._normalise_sources(sources)
            if asset["sources"]:
                asset["source_url"] = asset["sources"][0].get("url", "")
                asset["source_title"] = asset["sources"][0].get("title", "公开资料来源")
        return records

    @staticmethod
    def _read_csv(path: Path) -> Iterable[dict[str, str]]:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                yield from csv.DictReader(handle)
        except OSError:
            return

    def _try_create_retriever(self) -> None:
        scripts_dir = self.project_dir / "scripts"
        if not (scripts_dir / "retriever.py").is_file():
            return
        path_text = str(scripts_dir)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
        try:
            module = importlib.import_module("retriever")
            retriever_class = getattr(module, "LocalRetriever")
            self._retriever = retriever_class(self.data_dir)
            backend = getattr(self._retriever, "backend_name", None)
            self._retrieval_backend = str(backend or "local vector / lexical index")
        except Exception as exc:
            self._retriever = None
            self._retriever_error = type(exc).__name__

    # ---- Normalisation and fallback matching --------------------------

    def _normalize_retrieval_result(self, item: dict[str, Any]) -> dict[str, Any]:
        company_id = str(item.get("company_id") or item.get("id") or "")
        base = self._company_by_id.get(company_id, {})
        merged: dict[str, Any] = {**base, **item}
        if not merged.get("company_id"):
            merged["company_id"] = company_id
        if not merged.get("name"):
            merged["name"] = merged.get("company_name", "")
        for field in (
            "aliases",
            "products",
            "capabilities",
            "supply_chain_role",
            "input_materials",
            "output_products",
            "target_customer_industries",
            "matched_terms",
            "source_ids",
        ):
            merged[field] = self._list_value(merged.get(field))
        if not merged.get("sources"):
            merged["sources"] = [
                self._sources_by_id[source_id]
                for source_id in merged.get("source_ids", [])
                if source_id in self._sources_by_id
            ]
        merged["sources"] = self._normalise_sources(merged.get("sources", []))
        try:
            merged["score"] = round(float(merged.get("score", 0)), 3)
        except (TypeError, ValueError):
            merged["score"] = 0.0
        return self._summary(merged, include_evidence=True)

    def _lexical_search(
        self,
        query: str,
        *,
        district: str | None,
        sector: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        candidates = self._filter(self._companies, district=district, sector=sector)
        tokens = self._query_tokens(query)
        scored: list[tuple[float, dict[str, Any], list[str]]] = []
        for item in candidates:
            text = self._searchable_text(item)
            matched = [token for token in tokens if token and token in text]
            if not matched and query.strip():
                continue
            score = float(len(matched))
            if query.strip().casefold() in text:
                score += 3.0
            # Prefer an exact product/capability match to a generic description hit.
            product_text = " ".join(item.get("products", []) + item.get("capabilities", [])).casefold()
            score += sum(0.35 for token in matched if token in product_text)
            item_copy = dict(item)
            item_copy["score"] = round(score, 3)
            item_copy["matched_terms"] = matched[:8]
            scored.append((score, item_copy, matched))
        scored.sort(key=lambda entry: (-entry[0], entry[1].get("name", "")))
        return [self._summary(item, include_evidence=True) for _, item, _ in scored[:top_k]]

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        clean = query.strip().casefold()
        if not clean:
            return []
        tokens = [part for part in re.split(r"[\s,，、;；/|]+", clean) if len(part) >= 2]
        # Chinese input is often an unspaced phrase.  Two-to-four character
        # fragments give a useful, transparent fallback without pretending to be
        # a semantic model.
        compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", clean)
        if len(compact) > 3:
            tokens.extend(compact[index : index + 3] for index in range(len(compact) - 2))
        tokens.append(clean)
        return list(dict.fromkeys(token for token in tokens if len(token) >= 2))

    @staticmethod
    def _filter(
        records: Iterable[dict[str, Any]], *, district: str | None, sector: str | None
    ) -> list[dict[str, Any]]:
        result = list(records)
        if district:
            result = [item for item in result if item.get("district") == district]
        if sector:
            result = [item for item in result if item.get("sector") == sector]
        return result

    @staticmethod
    def _searchable_text(item: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in (
            "name",
            "english_name",
            "district",
            "park",
            "sector",
            "industry",
            "summary",
            "address",
        ):
            if item.get(key):
                parts.append(str(item[key]))
        for key in (
            "aliases",
            "products",
            "capabilities",
            "supply_chain_role",
            "input_materials",
            "output_products",
            "target_customer_industries",
        ):
            parts.extend(str(value) for value in item.get(key, []) if value)
        return " ".join(parts).casefold()

    def _summary(self, item: dict[str, Any], *, include_evidence: bool = False) -> dict[str, Any]:
        output = {
            "company_id": str(item.get("company_id") or item.get("id") or ""),
            "name": str(item.get("name") or item.get("company_name") or ""),
            "english_name": item.get("english_name", ""),
            "district": item.get("district", ""),
            "park": item.get("park", ""),
            "address": item.get("address", ""),
            "sector": item.get("sector") or item.get("industry_track", ""),
            "industry": item.get("industry") or item.get("industry_subtrack", ""),
            "products": self._list_value(item.get("products")),
            "capabilities": self._list_value(item.get("capabilities")),
            "supply_chain_role": self._list_value(item.get("supply_chain_role")),
            "summary": item.get("summary") or item.get("description", ""),
            "source_ids": self._list_value(item.get("source_ids")),
            "sources": self._normalise_sources(item.get("sources", [])),
            "source_date": item.get("source_date", ""),
            "verified_date": item.get("verified_date", ""),
            "confidence": item.get("confidence", ""),
        }
        if include_evidence:
            output.update(
                {
                    "matched_terms": self._list_value(item.get("matched_terms")),
                    "match_reason": item.get("match_reason", ""),
                    "match_type": item.get("match_type", ""),
                    "explicit_role": item.get("explicit_role", ""),
                    "semantic_score": item.get("semantic_score", ""),
                    "score": item.get("score", 0),
                    "input_materials": self._list_value(item.get("input_materials")),
                    "output_products": self._list_value(item.get("output_products")),
                    "target_customer_industries": self._list_value(item.get("target_customer_industries")),
                }
            )
        return output

    @staticmethod
    def _normalise_sources(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()
        for position, source in enumerate(value, start=1):
            if not isinstance(source, dict):
                continue
            source_id = str(source.get("source_id") or source.get("id") or f"source-{position}")
            identity = str(source.get("url") or source_id)
            if identity in seen:
                continue
            seen.add(identity)
            sources.append(
                {
                    "source_id": source_id,
                    "title": source.get("title") or source.get("source_title") or "公开资料来源",
                    "url": source.get("url") or source.get("source_url") or "",
                    "published_date": source.get("published_date") or source.get("source_date") or "",
                    "publisher": source.get("publisher") or "",
                }
            )
        return sources

    @staticmethod
    def _value(row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _list_value(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (tuple, list, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    pass
            return [part.strip() for part in re.split(r"[,，;；|、]", text) if part.strip()]
        return [str(value).strip()]

    def _distinct(self, field: str) -> list[str]:
        return sorted({str(item[field]).strip() for item in self._companies if item.get(field)}, key=str.casefold)

    def _distinct_list(self, field: str) -> list[str]:
        return sorted({value for item in self._companies for value in item.get(field, [])}, key=str.casefold)

    @staticmethod
    def _asset_area(asset: dict[str, Any]) -> str:
        return " ".join(str(asset.get(key, "")) for key in ("district", "area", "service_area", "name"))
