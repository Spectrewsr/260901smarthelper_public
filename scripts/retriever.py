"""Local search, evidence packaging, and conservative business matching.

``LocalRetriever`` is intentionally framework-free.  It can be imported by the
FastAPI service, used in a notebook, or exercised from a terminal without a
running database.  It reads only public local files and never sends data to an
external model by itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Supports package imports and ``python scripts/...`` style use.
    from .build_index import (
        DEFAULT_HASH_DIMENSION,
        INDEX_FORMAT,
        BgeM3EmbeddingBackend,
        HashEmbeddingBackend,
        _l2_normalize,
        _numpy,
        company_document,
    )
    from .data_contract import (
        clean_text,
        load_companies,
        load_regional_assets,
        load_sector_profiles,
        load_sources,
        normalize_json_record,
        resolve_data_paths,
        source_fingerprint,
        split_values,
    )
    from .matching import assess_company_match, general_matched_terms, infer_query_intents, normalize
except ImportError:  # pragma: no cover - direct script/module use.
    from build_index import (  # type: ignore
        DEFAULT_HASH_DIMENSION,
        INDEX_FORMAT,
        BgeM3EmbeddingBackend,
        HashEmbeddingBackend,
        _l2_normalize,
        _numpy,
        company_document,
    )
    from data_contract import (  # type: ignore
        clean_text,
        load_companies,
        load_regional_assets,
        load_sector_profiles,
        load_sources,
        normalize_json_record,
        resolve_data_paths,
        source_fingerprint,
        split_values,
    )
    from matching import assess_company_match, general_matched_terms, infer_query_intents, normalize  # type: ignore


FILTER_ALIASES = {
    "sector": "industry_track",
    "industry": "industry_subtrack",
    "role": "supply_chain_role",
    "supply_chain": "supply_chain_role",
    "area": "district",
}
SUPPORTED_FILTERS = frozenset({"district", "park", "industry_track", "industry_subtrack", "supply_chain_role"})


def _optional_load(loader: Any, data_dir: Path) -> list[dict[str, Any]]:
    """Allow the app to start while a non-company supplemental file is added."""

    try:
        return loader(data_dir)
    except FileNotFoundError:
        return []


def _copy_value(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    return value


class LocalRetriever:
    """Search a local company corpus and expose citation-ready results.

    Parameters
    ----------
    data_dir:
        Either ``demo/data`` (recommended) or ``demo/data/raw``.
    model_path:
        Optional local BAAI/bge-m3 directory. It is only needed to query a
        persisted BGE index. If unavailable, the retriever safely creates an
        in-memory deterministic hash index instead of silently comparing unlike
        vector spaces.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> None:
        self.paths = resolve_data_paths(data_dir)
        self.companies = sorted(load_companies(self.paths.raw), key=lambda item: item["company_id"])
        if not self.companies:
            raise ValueError(f"No companies found in {self.paths.companies}")
        self.companies_by_id = {company["company_id"]: company for company in self.companies}
        if len(self.companies_by_id) != len(self.companies):
            raise ValueError("companies.csv has duplicate company_id values; run validate_data.py")
        self.sources_by_id = {
            source.get("source_id", ""): source for source in _optional_load(load_sources, self.paths.raw)
        }
        self.sector_profiles = _optional_load(load_sector_profiles, self.paths.raw)
        self.regional_assets = _optional_load(load_regional_assets, self.paths.raw)
        self.documents = [company_document(company) for company in self.companies]
        self.model_path = self._resolve_model_path(model_path)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.vectors: Any
        self.embedder: Any
        self.index_status = ""
        self.index_manifest: dict[str, Any] = {}
        self._load_or_create_index()

    def _resolve_model_path(self, supplied: str | Path | None) -> Path | None:
        candidates: list[Path] = []
        if supplied:
            candidates.append(Path(supplied).expanduser())
        environment_path = os.environ.get("BGE_M3_MODEL_PATH")
        if environment_path:
            candidates.append(Path(environment_path).expanduser())
        # Standard project layout: demo/data + demo/models/bge-m3.
        candidates.extend(
            [
                self.paths.root.parent / "models" / "bge-m3",
                self.paths.root / "models" / "bge-m3",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _try_load_persisted_index(self) -> tuple[bool, str]:
        manifest_path = self.paths.derived / "index_manifest.json"
        index_path = self.paths.derived / "company_index.npz"
        manifest = self._read_json(manifest_path)
        if not manifest or not index_path.is_file():
            return False, "no generated index found"
        if manifest.get("format") != INDEX_FORMAT:
            return False, "index format is unsupported"
        if manifest.get("source_fingerprint") != source_fingerprint(self.companies, self.sources_by_id.values()):
            return False, "raw data changed after the index was built"
        np = _numpy()
        try:
            with np.load(index_path, allow_pickle=False) as saved:
                company_ids = [str(value) for value in saved["company_ids"].tolist()]
                vectors = _l2_normalize(saved["vectors"])
                hash_idf = saved["hash_idf"].copy() if "hash_idf" in saved.files else None
        except (OSError, ValueError, KeyError) as exc:
            return False, f"generated index cannot be read ({exc})"
        expected_ids = [company["company_id"] for company in self.companies]
        if company_ids != expected_ids or vectors.shape[0] != len(expected_ids):
            return False, "generated index company IDs do not match raw data"
        embedding = manifest.get("embedding") if isinstance(manifest.get("embedding"), dict) else {}
        backend_name = clean_text(embedding.get("backend", ""))
        try:
            if backend_name == "hash":
                if hash_idf is None:
                    return False, "hash index is missing its IDF weights"
                self.embedder = HashEmbeddingBackend(dimension=int(vectors.shape[1]), idf=hash_idf)
            elif backend_name == "bge-m3":
                if not self.model_path:
                    return False, "BGE-M3 model is unavailable for the generated BGE index"
                self.embedder = BgeM3EmbeddingBackend(
                    self.model_path,
                    batch_size=self.batch_size,
                    max_length=int(embedding.get("max_length", self.max_length)),
                    device=self.device,
                )
            else:
                return False, f"unknown embedding backend {backend_name!r}"
        except RuntimeError as exc:
            return False, str(exc)
        self.vectors = vectors
        self.index_manifest = manifest
        self.index_status = f"persisted:{backend_name}"
        return True, ""

    def _activate_runtime_hash(self, reason: str) -> None:
        self.embedder = HashEmbeddingBackend(dimension=DEFAULT_HASH_DIMENSION)
        self.vectors = self.embedder.fit_encode(self.documents)
        self.index_manifest = {
            "format": INDEX_FORMAT,
            "company_count": len(self.companies),
            "embedding": {
                "backend": "hash",
                "model": self.embedder.model_name,
                **self.embedder.manifest_extra(),
            },
            "source_fingerprint": source_fingerprint(self.companies, self.sources_by_id.values()),
        }
        self.index_status = f"runtime-fallback:hash ({reason})"

    def _load_or_create_index(self) -> None:
        loaded, reason = self._try_load_persisted_index()
        if not loaded:
            self._activate_runtime_hash(reason)

    def status(self) -> dict[str, Any]:
        """Expose a small diagnostics object safe to show in a local admin UI."""

        embedding = self.index_manifest.get("embedding", {})
        return {
            "company_count": len(self.companies),
            "sector_profile_count": len(self.sector_profiles),
            "regional_asset_count": len(self.regional_assets),
            "index_status": self.index_status,
            "embedding_backend": embedding.get("backend", ""),
            "embedding_model": embedding.get("model", ""),
        }

    @property
    def backend_name(self) -> str:
        """Short label used by the web adapter's health/status endpoint."""

        return str(self.index_manifest.get("embedding", {}).get("backend", "hash"))

    @staticmethod
    def _filter_values(value: Any) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            flattened: list[str] = []
            for item in value:
                flattened.extend(split_values(item))
            return flattened
        return split_values(value)

    def _matches_filters(self, company: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
        if not filters:
            return True
        for original_key, requested_value in filters.items():
            key = FILTER_ALIASES.get(original_key, original_key)
            if key not in SUPPORTED_FILTERS:
                continue
            requested = [normalize(value) for value in self._filter_values(requested_value) if normalize(value)]
            if not requested:
                continue
            candidate_value = company.get(key, "")
            if isinstance(candidate_value, list):
                candidates = [normalize(value) for value in candidate_value]
            else:
                candidates = [normalize(candidate_value)]
            # Multiple filter values are an OR within one facet and an AND
            # across facets. Short district forms such as “新北” match “新北区”.
            if not any(
                requested_value in candidate or candidate in requested_value
                for requested_value in requested
                for candidate in candidates
                if candidate
            ):
                return False
        return True

    def _citations(self, company: Mapping[str, Any]) -> list[dict[str, str]]:
        citations: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        source_ids = list(company.get("source_ids", []))
        source_urls = list(company.get("source_urls", []))
        for index, source_id in enumerate(source_ids):
            source = self.sources_by_id.get(source_id, {})
            url = source.get("url", "") or (source_urls[index] if index < len(source_urls) else "")
            citation = {
                "source_id": source_id,
                "title": source.get("title", ""),
                "publisher": source.get("publisher", ""),
                "url": url,
                "published_date": source.get("published_date", ""),
                "accessed_date": source.get("accessed_date", ""),
                "source_type": source.get("source_type", ""),
            }
            marker = (citation["source_id"], citation["url"])
            if marker not in seen:
                seen.add(marker)
                citations.append(citation)
        for url in source_urls:
            marker = ("", url)
            if url and marker not in seen and not any(item["url"] == url for item in citations):
                citations.append(
                    {
                        "source_id": "",
                        "title": "",
                        "publisher": "",
                        "url": url,
                        "published_date": "",
                        "accessed_date": "",
                        "source_type": "",
                    }
                )
        return citations

    def _result(self, company: Mapping[str, Any], *, score: float, semantic_score: float, query: str) -> dict[str, Any]:
        result = {key: _copy_value(value) for key, value in company.items()}
        result["score"] = round(float(score), 4)
        result["semantic_score"] = round(float(semantic_score), 4)
        result["matched_terms"] = general_matched_terms(company, query) if clean_text(query) else []
        result["sources"] = self._citations(company)
        # A plain list is convenient for small cards; ``sources`` has full citation
        # metadata for report generation and a source drawer in the web UI.
        result["citation_urls"] = [citation["url"] for citation in result["sources"] if citation["url"]]
        return result

    def _query_vector(self, query: str) -> Any:
        try:
            return self.embedder.encode([query])[0]
        except RuntimeError:
            # A GPU/model failure after startup must never cause hash query
            # vectors to be compared with BGE document vectors. Rebuild both in
            # memory using the same deterministic fallback space.
            self._activate_runtime_hash("embedding model unavailable while querying")
            return self.embedder.encode([query])[0]

    def search(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        *,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Return citation-ready companies ordered by local relevance.

        ``filters`` supports district, park, industry_track, industry_subtrack,
        and supply_chain_role (with sector/industry/role aliases).  It never
        relaxes an unmatched filter, so an empty result is meaningful.
        """

        if top_k < 1:
            return []
        text = clean_text(query)
        candidate_indices = [
            index
            for index, company in enumerate(self.companies)
            if self._matches_filters(company, filters)
        ]
        if not candidate_indices:
            return []
        if not text:
            return [
                self._result(self.companies[index], score=0.0, semantic_score=0.0, query="")
                for index in candidate_indices[:top_k]
            ]
        query_vector = self._query_vector(text)
        np = _numpy()
        semantic_scores = self.vectors @ np.asarray(query_vector, dtype=np.float32)
        ranked: list[tuple[float, float, int]] = []
        normalized_query = normalize(text)
        for index in candidate_indices:
            company = self.companies[index]
            semantic = float(semantic_scores[index])
            matched_terms = general_matched_terms(company, text)
            name = normalize(company.get("company_name", ""))
            aliases = [normalize(value) for value in company.get("aliases", [])]
            exact_name = bool(normalized_query and (normalized_query in name or name in normalized_query))
            alias_match = any(normalized_query and (normalized_query in alias or alias in normalized_query) for alias in aliases)
            name_boost = 0.26 if exact_name else (0.16 if alias_match else 0.0)
            evidence_boost = min(0.16, 0.025 * len(matched_terms))
            score = max(-1.0, min(1.0, semantic + name_boost + evidence_boost))
            ranked.append((score, semantic, index))
        # Name is an explicit final tie-breaker, avoiding NumPy sort instability.
        ranked.sort(key=lambda item: (-item[0], -item[1], self.companies[item[2]]["company_id"]))
        return [
            self._result(self.companies[index], score=score, semantic_score=semantic, query=text)
            for score, semantic, index in ranked[:top_k]
        ]

    def get_company(self, company_id: str) -> dict[str, Any] | None:
        company = self.companies_by_id.get(clean_text(company_id))
        if not company:
            return None
        return self._result(company, score=0.0, semantic_score=0.0, query="")

    @staticmethod
    def _record_text(record: Mapping[str, Any]) -> str:
        parts: list[str] = []
        for key in sorted(record):
            if key.startswith("_"):
                continue
            value = record[key]
            if isinstance(value, list):
                parts.extend(clean_text(item) for item in value)
            else:
                parts.append(clean_text(value))
        return " ".join(part for part in parts if part)

    @staticmethod
    def _record_match_score(record: Mapping[str, Any], query: str) -> tuple[float, list[str]]:
        if not clean_text(query):
            return 0.0, []
        # Reuse the public explainability helper by giving it compact synthetic
        # product/capability fields; no assertions about a sector's role follow.
        pseudo_company = {
            "products": [clean_text(record.get("name", ""))],
            "capabilities": [LocalRetriever._record_text(record)],
            "output_products": [],
            "input_materials": [],
            "target_customer_industries": [],
            "supply_chain_role": [],
            "summary": clean_text(record.get("overview", record.get("description", ""))),
        }
        assessment = assess_company_match(pseudo_company, query, "partner")
        return assessment.score, list(assessment.matched_terms)

    def _supplemental_records(
        self,
        records: Sequence[Mapping[str, Any]],
        query: str = "",
        *,
        district: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        requested_district = normalize(district) if district else ""
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for source_record in records:
            record = normalize_json_record(source_record)
            record_district = normalize(record.get("district", ""))
            if requested_district and record_district and requested_district not in record_district and record_district not in requested_district:
                continue
            score, terms = self._record_match_score(record, query)
            item = {key: _copy_value(value) for key, value in record.items()}
            item["score"] = round(score, 4)
            item["matched_terms"] = terms
            stable_id = clean_text(item.get("sector_id", item.get("asset_id", item.get("name", ""))))
            ranked.append((score, stable_id, item))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        limit = len(ranked) if top_k is None else max(top_k, 0)
        return [item for _, _, item in ranked[:limit]]

    def relevant_sector_profiles(self, query: str = "", *, top_k: int | None = 4) -> list[dict[str, Any]]:
        return self._supplemental_records(self.sector_profiles, query, top_k=top_k)

    def relevant_regional_assets(
        self,
        query: str = "",
        *,
        district: str | None = None,
        top_k: int | None = 8,
    ) -> list[dict[str, Any]]:
        return self._supplemental_records(self.regional_assets, query, district=district, top_k=top_k)

    def match_business_need(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Group local hits into potential suppliers/customers/partners.

        The result deliberately says "potential" and carries an evidence-based
        reason per company. It is safe to pass as context to an LLM as long as
        the LLM prompt preserves that uncertainty and its source citations.
        """

        if top_k < 1:
            return {
                "suppliers": [],
                "customers": [],
                "partners": [],
                "all_matches": [],
                "query_intents": [],
                "rationale": "未请求候选数量。",
            }
        candidates = self.search(query, filters, top_k=max(20, top_k * 4))
        groups: dict[str, list[dict[str, Any]]] = {"supplier": [], "customer": [], "partner": []}
        for candidate in candidates:
            company = self.companies_by_id[candidate["company_id"]]
            relevance = max(0.0, min(1.0, (float(candidate["semantic_score"]) + 1.0) / 2.0))
            for kind in groups:
                assessment = assess_company_match(company, query, kind)
                combined = 0.62 * relevance + 0.38 * assessment.score
                item = {key: _copy_value(value) for key, value in candidate.items()}
                item.update(assessment.to_dict())
                item["score"] = round(combined, 4)
                groups[kind].append(item)
        for items in groups.values():
            items.sort(key=lambda item: (-item["score"], item["company_id"]))
        explicit_intents = infer_query_intents(query)
        # Supply and demand are always useful in a招商 report. Partners are
        # displayed only when the user actually asks for collaboration context.
        suppliers = groups["supplier"][:top_k]
        customers = groups["customer"][:top_k]
        partners = groups["partner"][:top_k] if "partner" in explicit_intents else []
        unique: dict[str, dict[str, Any]] = {}
        for item in [*suppliers, *customers, *partners]:
            existing = unique.get(item["company_id"])
            if existing is None or item["score"] > existing["score"]:
                unique[item["company_id"]] = item
        all_matches = sorted(unique.values(), key=lambda item: (-item["score"], item["company_id"]))[: top_k * 2]
        return {
            "suppliers": suppliers,
            "customers": customers,
            "partners": partners,
            "all_matches": all_matches,
            "query_intents": sorted(explicit_intents),
            "disclaimer": "全部为基于公开资料的潜在匹配，不代表已建立商业合作，需进一步核验。",
            "rationale": "依据公开资料中的产品、能力、输入/输出与产业链角色进行潜在匹配；不代表企业间已建立商业合作。",
        }

    def build_evidence_context(
        self,
        results: Iterable[Mapping[str, Any]],
        *,
        max_chars: int = 7000,
    ) -> str:
        """Create compact, source-labelled context for a report-generation API.

        It contains only retrieved public facts and URLs, never API credentials
        or the full private corpus. The caller should still instruct its model to
        cite source IDs and say when evidence is insufficient.
        """

        remaining = max(int(max_chars), 0)
        chunks: list[str] = []
        for result in results:
            company_id = clean_text(result.get("company_id", ""))
            company = self.companies_by_id.get(company_id)
            if not company:
                continue
            lines = [f"[企业 {company_id}]", company_document(company)]
            citations = result.get("sources")
            if not isinstance(citations, list):
                citations = self._citations(company)
            for citation in citations:
                if not isinstance(citation, Mapping):
                    continue
                source_id = clean_text(citation.get("source_id", "")) or "未编号来源"
                title = clean_text(citation.get("title", ""))
                url = clean_text(citation.get("url", ""))
                lines.append(f"来源 [{source_id}] {title} {url}".rstrip())
            chunk = "\n".join(line for line in lines if line) + "\n"
            if len(chunk) > remaining:
                if remaining > 80:
                    chunks.append(chunk[: remaining - 1].rstrip() + "…")
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining <= 0:
                break
        return "\n".join(chunks).strip()

    def report_context(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        *,
        company_limit: int = 8,
    ) -> dict[str, Any]:
        """Convenience payload for the app's API service and report prompt."""

        companies = self.search(query, filters, top_k=company_limit)
        district = clean_text((filters or {}).get("district", "")) or None
        matching = self.match_business_need(query, filters, top_k=min(5, company_limit))
        return {
            "companies": companies,
            "matching": matching,
            "sector_profiles": self.relevant_sector_profiles(query),
            "regional_assets": self.relevant_regional_assets(query, district=district),
            "evidence_context": self.build_evidence_context(companies),
            "status": self.status(),
        }
