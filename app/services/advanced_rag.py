"""Advanced local retrieval: contextual chunks, hybrid RRF, cross-encoder rerank.

The implementation is intentionally small and dependency-light.  SQLite FTS5
is the sparse side; the existing BGE-M3 local index is the dense side; every
stage returns an audit trace so a caller can prove that a query did (or did
not) use RAG.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Iterable

from .knowledge import KnowledgeRepository, cjk_terms


class CrossEncoderReranker:
    """Lazy BGE reranker with a transparent deterministic fallback."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: Any | None = None
        self._load_error = ""
        self._lock = threading.Lock()
        self._attempted = False
        self.device = os.getenv("RERANK_DEVICE", "cuda").strip() or "cuda"

    def _load(self) -> None:
        if self._attempted:
            return
        with self._lock:
            if self._attempted:
                return
            self._attempted = True
            if not self.model_path.is_dir():
                self._load_error = "local BGE reranker model directory is missing"
                return
            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(str(self.model_path), use_fp16=self.device.startswith("cuda"), device=self.device)
            except Exception:  # GPU pressure can make a local demo fail gracefully.
                # Diagnostics remain local; browser-visible traces use a
                # category-only reason so a third-party exception cannot
                # disclose an absolute model path.
                self._load_error = "本地交叉重排模型初始化失败"
                self._model = None

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self._load()
        if self._model is not None:
            try:
                pairs = [[query, f"{item.get('context', '')}\n{item.get('content', '')}"] for item in candidates]
                scores = self._model.compute_score(pairs, normalize=True)
                if not isinstance(scores, list):
                    scores = [scores]
                ranked = []
                for item, score in zip(candidates, scores):
                    enriched = dict(item)
                    enriched["rerank_score"] = round(float(score), 5)
                    ranked.append(enriched)
                ranked.sort(key=lambda item: (-float(item["rerank_score"]), -float(item.get("rrf_score", 0)), item.get("chunk_id", "")))
                return ranked, {
                    "engine": "BAAI/bge-reranker-v2-m3",
                    "device": self.device,
                    "model_loaded": True,
                    "fallback": False,
                    "candidate_count": len(candidates),
                }
            except Exception:
                self._load_error = "本地交叉重排模型运行失败"
        query_terms = set(cjk_terms(query))
        ranked = []
        for item in candidates:
            text_terms = set(cjk_terms(f"{item.get('context', '')} {item.get('content', '')}"))
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            enriched = dict(item)
            enriched["rerank_score"] = round(0.68 * float(item.get("rrf_score", 0)) + 0.32 * overlap, 5)
            ranked.append(enriched)
        ranked.sort(key=lambda item: (-float(item["rerank_score"]), -float(item.get("rrf_score", 0)), item.get("chunk_id", "")))
        return ranked, {
            "engine": "contextual lexical fallback",
            "model_directory": self.model_path.name or "local-reranker",
            "model_loaded": False,
            "fallback": True,
            "reason": self._load_error or "reranker unavailable",
            "candidate_count": len(candidates),
        }


class AdvancedRAG:
    """Hybrid retrieval facade layered on the local evidence catalogue."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository
        project_dir = repository.data_dir.parent
        configured_reranker = os.getenv("RERANK_MODEL_PATH", "").strip()
        configured_path = Path(configured_reranker).expanduser() if configured_reranker else None
        reranker_path = (configured_path if configured_path and configured_path.is_absolute() else project_dir / configured_path) if configured_path else project_dir / "models" / "bge-reranker-v2-m3"
        self.reranker = CrossEncoderReranker(reranker_path)
        self._dense: Any | None = None
        self._dense_error = ""
        self._dense_lock = threading.Lock()
        self._dynamic_lock = threading.Lock()
        self._dynamic_signature = ""
        self._dynamic_records: list[dict[str, Any]] = []
        self._dynamic_vectors: Any | None = None
        self._dynamic_error = ""

    def _dense_retriever(self) -> Any | None:
        if self._dense is not None or self._dense_error:
            return self._dense
        with self._dense_lock:
            if self._dense is not None or self._dense_error:
                return self._dense
            try:
                from scripts.retriever import LocalRetriever

                self._dense = LocalRetriever(self.repository.data_dir)
            except Exception:
                self._dense_error = "本地稠密索引初始化失败"
        return self._dense

    def retrieve(
        self,
        query: str,
        *,
        collections: Iterable[str] | None = None,
        district: str | None = None,
        sector: str | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        collections = list(collections or ("enterprise_chain", "park_negotiation", "landing_support"))
        lexical = self.repository.fts_search(query, collections=collections, limit=40)
        dense_results, dense_trace = self._dense_candidates(query, district=district, sector=sector)

        chunks_by_id: dict[str, dict[str, Any]] = {item["chunk_id"]: dict(item) for item in lexical}
        # Dense results are company records, so attach their contextual company
        # chunks from the SQLite knowledge base when possible.
        for result in dense_results:
            company_id = str(result.get("company_id") or "")
            if not company_id:
                continue
            chunk_id = f"enterprise_chain:company:{company_id}"
            if chunk_id not in chunks_by_id:
                row = self.repository.fts_search(str(result.get("name") or company_id), collections=["enterprise_chain"], limit=6)
                match = next((item for item in row if item.get("chunk_id") == chunk_id), None)
                if match:
                    chunks_by_id[chunk_id] = dict(match)
            if chunk_id in chunks_by_id:
                chunks_by_id[chunk_id]["dense_score"] = float(result.get("semantic_score", result.get("score", 0)) or 0)

        lexical_rank = {item["chunk_id"]: rank for rank, item in enumerate(lexical, start=1)}
        dense_rank = {
            f"enterprise_chain:company:{result['company_id']}": rank
            for rank, result in enumerate(dense_results, start=1)
            if result.get("company_id")
        }
        rrf_constant = 60
        fused: list[dict[str, Any]] = []
        for chunk_id, candidate in chunks_by_id.items():
            lexical_position = lexical_rank.get(chunk_id)
            dense_position = dense_rank.get(chunk_id)
            score = (1 / (rrf_constant + lexical_position) if lexical_position else 0.0) + (1 / (rrf_constant + dense_position) if dense_position else 0.0)
            enriched = dict(candidate)
            enriched.update({"lexical_rank": lexical_position, "dense_rank": dense_position, "rrf_score": round(score, 7)})
            fused.append(enriched)
        fused.sort(key=lambda item: (-float(item["rrf_score"]), item["chunk_id"]))
        reranked, reranker_trace = self.reranker.rerank(query, fused[:20])
        reranked = reranked[: max(1, top_k)]

        company_ids = [item["parent_id"] for item in reranked if item.get("parent_type") == "company"]
        companies = self.repository.company_by_ids(company_ids)
        company_by_id = {item["company_id"]: item for item in companies}
        for item in reranked:
            if item.get("parent_type") == "company":
                item["company"] = company_by_id.get(item["parent_id"])
            item["sources"] = self.repository.source_cards(item.get("source_ids", []))

        return {
            "query": query,
            "chunks": reranked,
            "companies": companies,
            "trace": {
                "contextual": {
                    "strategy": "parent metadata prefix (collection, entity, district, sector, sources, confidence)",
                    "collections": collections,
                    "candidate_count": len(lexical),
                },
                "lexical": {"engine": "SQLite FTS5 / BM25", "candidate_count": len(lexical)},
                "dense": dense_trace,
                "fusion": {"algorithm": "reciprocal rank fusion", "rrf_k": rrf_constant, "candidate_count": len(fused)},
                "reranker": reranker_trace,
            },
        }

    def _dense_candidates(self, query: str, *, district: str | None, sector: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        dense = self._dense_retriever()
        if dense is None:
            return [], {"engine": "BGE-M3 unavailable", "candidate_count": 0, "fallback": True, "reason": self._dense_error}
        try:
            filters = {key: value for key, value in {"district": district, "sector": sector}.items() if value}
            results = dense.search(query, filters=filters or None, top_k=40)
            dynamic_results, dynamic_trace = self._dynamic_dense_candidates(dense, query, filters)
            merged = {str(item.get("company_id")): dict(item) for item in results if item.get("company_id")}
            for item in dynamic_results:
                company_id = str(item.get("company_id") or "")
                if company_id:
                    merged[company_id] = item
            results = sorted(
                merged.values(),
                key=lambda item: (-float(item.get("score", 0.0)), -float(item.get("semantic_score", 0.0)), str(item.get("company_id", ""))),
            )[:40]
            status = dense.status()
            backend = str(status.get("embedding_backend", "") or "")
            engine = "BAAI/bge-m3" if backend == "bge-m3" else ("deterministic hash fallback" if backend == "hash" else "local dense fallback")
            return results, {
                "engine": engine,
                "candidate_count": len(results),
                "index_status": status.get("index_status", ""),
                "fallback": backend != "bge-m3",
                "dynamic_extension": dynamic_trace,
            }
        except Exception:
            self._dense_error = "本地稠密检索运行失败"
            return [], {"engine": "BGE-M3 unavailable", "candidate_count": 0, "fallback": True, "reason": self._dense_error}

    @staticmethod
    def _dense_document(record: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Adapt a SQLite company row to the raw-index document contract."""

        raw = {
            "company_id": record.get("company_id", ""),
            "company_name": record.get("name", ""),
            "english_name": record.get("english_name", ""),
            "aliases": record.get("aliases", []),
            "district": record.get("district", ""),
            "park": record.get("park", ""),
            "address": record.get("address", ""),
            "industry_track": record.get("sector", ""),
            "industry_subtrack": record.get("industry", ""),
            "supply_chain_role": record.get("supply_chain_role", []),
            "products": record.get("products", []),
            "capabilities": record.get("capabilities", []),
            "input_materials": record.get("input_materials", []),
            "output_products": record.get("output_products", []),
            "target_customer_industries": record.get("target_customer_industries", []),
            "summary": record.get("summary", ""),
        }
        lines: list[str] = []
        labels = (
            ("企业名称", "company_name"), ("英文名称", "english_name"), ("别名", "aliases"),
            ("所在区", "district"), ("园区", "park"), ("地址", "address"),
            ("产业赛道", "industry_track"), ("细分领域", "industry_subtrack"),
            ("产业链角色", "supply_chain_role"), ("产品", "products"), ("能力/工序", "capabilities"),
            ("输入材料", "input_materials"), ("输出产品", "output_products"),
            ("目标客户行业", "target_customer_industries"), ("公开简介", "summary"),
        )
        for label, key in labels:
            value = raw[key]
            rendered = "、".join(str(item) for item in value if str(item)) if isinstance(value, list) else str(value or "")
            if rendered:
                lines.append(f"{label}：{rendered}")
        return raw, "\n".join(lines)

    def _dynamic_dense_state(self, dense: Any) -> tuple[list[dict[str, Any]], Any | None, dict[str, Any]]:
        """Build/reuse a tiny in-memory embedding delta for reviewed additions."""

        all_records = self.repository.published_companies_for_dense_extension()
        known_ids = set(getattr(dense, "companies_by_id", {}))
        dynamic_records = [record for record in all_records if str(record.get("company_id") or "") not in known_ids]
        status = dense.status()
        backend = str(status.get("embedding_backend", "") or "")
        shape = getattr(getattr(dense, "vectors", None), "shape", (0, 0))
        vector_dimension = int(shape[1]) if len(shape) > 1 else 0
        document_signature = "|".join(
            f"{record.get('company_id', '')}:{record.get('updated_at', '')}" for record in dynamic_records
        )
        # A local retriever can fall back from BGE's 1024-D space to the
        # 768-D deterministic hash space after a GPU/model failure.  Backend
        # and dimensionality are part of the cache identity, so stale vectors
        # are never multiplied by a query from another embedding space.
        signature = f"{backend}:{vector_dimension}|{document_signature}"
        with self._dynamic_lock:
            refreshed = signature != self._dynamic_signature
            if refreshed:
                self._dynamic_signature = signature
                self._dynamic_records = []
                self._dynamic_vectors = None
                self._dynamic_error = ""
                if dynamic_records:
                    try:
                        from scripts.build_index import _l2_normalize

                        adapted = [self._dense_document(record) for record in dynamic_records]
                        vectors = dense.embedder.encode([document for _, document in adapted])
                        self._dynamic_records = [
                            {"record": record, "raw": raw}
                            for record, (raw, _document) in zip(dynamic_records, adapted)
                        ]
                        self._dynamic_vectors = _l2_normalize(vectors)
                    except Exception:
                        self._dynamic_error = "本地增量嵌入失败"
            return self._dynamic_records, self._dynamic_vectors, {
                "mode": "in-memory reviewed-record delta",
                "company_count": len(dynamic_records),
                "embedded_count": len(self._dynamic_records),
                "embedding_backend": backend,
                "vector_dimension": vector_dimension,
                "refreshed": refreshed,
                "error": self._dynamic_error or None,
            }

    def _dynamic_dense_candidates(self, dense: Any, query: str, filters: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            from scripts.build_index import _numpy

            query_vector = dense._query_vector(query)
            # Query first: LocalRetriever may switch its backend to hash while
            # producing the query vector.  The subsequent state lookup then
            # refreshes any reviewed-record delta in that same vector space.
            records, vectors, trace = self._dynamic_dense_state(dense)
            if not records or vectors is None:
                return [], trace
            np = _numpy()
            if int(vectors.shape[1]) != int(np.asarray(query_vector).shape[0]):
                trace["error"] = "dynamic vector dimension differs from local query vector; delta withheld"
                trace["candidate_count"] = 0
                return [], trace
            scores = vectors @ np.asarray(query_vector, dtype=np.float32)
            results: list[dict[str, Any]] = []
            for index, item in enumerate(records):
                raw = item["raw"]
                if not dense._matches_filters(raw, filters or None):
                    continue
                record = item["record"]
                semantic = float(scores[index])
                name = str(record.get("name") or "").casefold()
                normalized_query = query.casefold()
                name_boost = 0.26 if name and (name in normalized_query or normalized_query in name) else 0.0
                score = max(-1.0, min(1.0, semantic + name_boost))
                results.append(
                    {
                        "company_id": str(record.get("company_id") or ""),
                        "name": str(record.get("name") or ""),
                        "score": round(score, 4),
                        "semantic_score": round(semantic, 4),
                        "dynamic_extension": True,
                    }
                )
            results.sort(key=lambda item: (-float(item["score"]), -float(item["semantic_score"]), item["company_id"]))
            trace["candidate_count"] = len(results)
            return results, trace
        except Exception:
            # Keep the base dense result; a transient dynamic-delta failure is
            # auditable but does not erase the persisted index result set.
            trace = {"mode": "in-memory reviewed-record delta", "candidate_count": 0}
            trace["error"] = "本地增量检索失败"
            return [], trace
