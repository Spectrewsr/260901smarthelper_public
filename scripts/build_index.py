"""Build a deterministic local company-search index.

The default ``hash`` backend is deliberately dependency-light and repeatable. It
makes the demo usable immediately and offers a graceful fallback when a local
model is unavailable.  Passing ``--embedding-backend bge-m3`` uses the intended
local BAAI/bge-m3 dense embedding model; no model is downloaded implicitly.

Examples
--------
    python scripts/build_index.py --data-dir data --min-companies 100
    python scripts/build_index.py --data-dir data --embedding-backend bge-m3 ^
        --model-path models/bge-m3 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import unicodedata
import uuid
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

try:  # Supports both ``python scripts/...`` and package imports.
    from .data_contract import (
        company_document,
        load_companies,
        load_sources,
        resolve_data_paths,
        sha256_text,
        source_fingerprint,
        stable_json,
    )
    from .validate_data import validate_data
except ImportError:  # pragma: no cover - exercised by CLI use.
    from data_contract import (  # type: ignore
        company_document,
        load_companies,
        load_sources,
        resolve_data_paths,
        sha256_text,
        source_fingerprint,
        stable_json,
    )
    from validate_data import validate_data  # type: ignore


INDEX_FORMAT = "changzhou-company-index/v1"
DEFAULT_HASH_DIMENSION = 768
# BGE-M3 itself is a large multilingual model.  These defaults keep a 6 GB
# laptop GPU responsive while remaining more than sufficient for the compact
# company profiles in this demo.
DEFAULT_BGE_BATCH_SIZE = 4
DEFAULT_BGE_MAX_LENGTH = 512
_TOKEN_RE = re.compile(r"[\u3400-\u9fff]+|[a-z0-9][a-z0-9.+#_-]*", re.IGNORECASE)


def _numpy() -> Any:
    """Import numpy only when an index is actually built or read."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on local setup.
        raise RuntimeError(
            "numpy is required to build the local index. Install the demo requirements first."
        ) from exc
    return np


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def lexical_tokens(value: str) -> list[str]:
    """Tokenize Chinese and Latin text without a language-specific tokenizer.

    Chinese words are represented both as their original run and overlapping
    two/three-character grams.  This makes a deterministic fallback useful for
    product phrases such as ``锂电池隔膜`` and for mixed Chinese/English company
    descriptions, while BGE-M3 remains the preferred semantic backend.
    """

    normalized = normalize_text(value)
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        run = match.group(0)
        if not run:
            continue
        if "\u3400" <= run[0] <= "\u9fff":
            tokens.append(run)
            # Single characters have little discriminatory value, so retain
            # n-grams where possible. The full run preserves exact phrases.
            for width in (2, 3):
                if len(run) >= width:
                    tokens.extend(run[start : start + width] for start in range(len(run) - width + 1))
        else:
            tokens.append(run)
    return tokens


def _hash_bucket(token: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(
        token.encode("utf-8"), digest_size=8, person=b"cz-rag-v1"
    ).digest()
    number = int.from_bytes(digest[:4], "little", signed=False)
    sign = 1.0 if (digest[4] & 1) else -1.0
    return number % dimension, sign


def _l2_normalize(matrix: Any) -> Any:
    np = _numpy()
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(values / norms, dtype=np.float32)


class EmbeddingBackend(Protocol):
    backend_name: str
    model_name: str

    def fit_encode(self, texts: Sequence[str]) -> Any:
        """Embed documents, learning corpus-local state if applicable."""

    def encode(self, texts: Sequence[str]) -> Any:
        """Embed queries after ``fit_encode`` has been called."""

    def manifest_extra(self) -> dict[str, Any]:
        """Return small, serializable metadata for the index manifest."""


class HashEmbeddingBackend:
    """A deterministic, local lexical vectorizer used as a robust fallback."""

    backend_name = "hash"
    model_name = "deterministic-hash-v1"

    def __init__(self, dimension: int = DEFAULT_HASH_DIMENSION, idf: Any | None = None) -> None:
        if dimension < 32:
            raise ValueError("hash embedding dimension must be at least 32")
        self.dimension = int(dimension)
        self.idf = idf

    def _vectorize(self, texts: Sequence[str]) -> Any:
        np = _numpy()
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        weights = self.idf
        if weights is None:
            weights = np.ones(self.dimension, dtype=np.float32)
        for row_index, text in enumerate(texts):
            counts = Counter(lexical_tokens(text))
            for token, count in counts.items():
                bucket, sign = _hash_bucket(token, self.dimension)
                matrix[row_index, bucket] += sign * (1.0 + math.log(float(count))) * float(weights[bucket])
        return _l2_normalize(matrix)

    def fit_encode(self, texts: Sequence[str]) -> Any:
        np = _numpy()
        document_frequency = np.zeros(self.dimension, dtype=np.int32)
        for text in texts:
            occupied = {_hash_bucket(token, self.dimension)[0] for token in set(lexical_tokens(text))}
            for bucket in occupied:
                document_frequency[bucket] += 1
        count = max(len(texts), 1)
        self.idf = np.asarray(
            [math.log((1.0 + count) / (1.0 + df)) + 1.0 for df in document_frequency],
            dtype=np.float32,
        )
        return self._vectorize(texts)

    def encode(self, texts: Sequence[str]) -> Any:
        if self.idf is None:
            # Useful for direct API callers that only need query embeddings.
            self.idf = _numpy().ones(self.dimension, dtype=_numpy().float32)
        return self._vectorize(texts)

    def manifest_extra(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "tokenizer": "cjk-ngrams+latin-words"}


class BgeM3EmbeddingBackend:
    """Thin adapter for an already-downloaded local BAAI/bge-m3 model."""

    backend_name = "bge-m3"
    model_name = "BAAI/bge-m3"

    def __init__(
        self,
        model_path: str | Path,
        *,
        batch_size: int = DEFAULT_BGE_BATCH_SIZE,
        max_length: int = DEFAULT_BGE_MAX_LENGTH,
        device: str = "auto",
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise RuntimeError(
                f"Local BGE-M3 model was not found at {path}. "
                "Download it first, or use --embedding-backend hash."
            )
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - depends on installed environment.
            raise RuntimeError(
                "FlagEmbedding is required for --embedding-backend bge-m3. "
                "Install the demo requirements first."
            ) from exc
        self.model_path = path
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device
        # FlagEmbedding chooses CUDA when available. ``use_fp16=False`` is the
        # safe choice for an explicit CPU build; RTX 3050 runs efficiently with
        # fp16 through the default/auto setting.
        use_fp16 = device != "cpu"
        devices = None if device == "auto" else device
        try:
            self.model = BGEM3FlagModel(
                str(path),
                use_fp16=use_fp16,
                devices=devices,
                passage_max_length=self.max_length,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except Exception as exc:  # pragma: no cover - hardware/model dependent.
            raise RuntimeError(f"Unable to load local BGE-M3 model at {path}: {exc}") from exc

    def _encode(self, texts: Sequence[str]) -> Any:
        if not texts:
            return _numpy().zeros((0, 0), dtype=_numpy().float32)
        try:
            encoded = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except Exception as exc:  # pragma: no cover - hardware/model dependent.
            raise RuntimeError(f"BGE-M3 embedding failed: {exc}") from exc
        dense = encoded.get("dense_vecs") if isinstance(encoded, dict) else encoded
        if dense is None:
            raise RuntimeError("BGE-M3 did not return dense_vecs")
        return _l2_normalize(dense)

    def fit_encode(self, texts: Sequence[str]) -> Any:
        return self._encode(texts)

    def encode(self, texts: Sequence[str]) -> Any:
        return self._encode(texts)

    def manifest_extra(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "requested_device": self.device,
            "local_model_directory": self.model_path.name,
        }


def create_embedding_backend(
    backend: str,
    *,
    model_path: str | Path | None = None,
    hash_dimension: int = DEFAULT_HASH_DIMENSION,
    batch_size: int = DEFAULT_BGE_BATCH_SIZE,
    max_length: int = DEFAULT_BGE_MAX_LENGTH,
    device: str = "auto",
    hash_idf: Any | None = None,
) -> EmbeddingBackend:
    """Create a backend without making an implicit network request.

    ``auto`` uses BGE-M3 only when an existing local model directory was
    explicitly supplied; otherwise it selects deterministic hash retrieval.
    """

    choice = backend.casefold()
    if choice == "auto":
        choice = "bge-m3" if model_path and Path(model_path).expanduser().exists() else "hash"
    if choice == "hash":
        return HashEmbeddingBackend(dimension=hash_dimension, idf=hash_idf)
    if choice == "bge-m3":
        if not model_path:
            raise RuntimeError("--model-path is required with --embedding-backend bge-m3")
        return BgeM3EmbeddingBackend(
            model_path,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
    raise ValueError(f"Unsupported embedding backend: {backend!r}")


def _source_records_for_company(company: dict[str, Any], sources_by_id: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    source_ids = list(company.get("source_ids", []))
    source_urls = list(company.get("source_urls", []))
    for index, source_id in enumerate(source_ids):
        source = dict(sources_by_id.get(source_id, {}))
        url = source.get("url") or (source_urls[index] if index < len(source_urls) else "")
        citation = {
            "source_id": source_id,
            "title": source.get("title", ""),
            "publisher": source.get("publisher", ""),
            "url": url,
            "published_date": source.get("published_date", ""),
            "source_type": source.get("source_type", ""),
        }
        key = (citation["source_id"], citation["url"])
        if key not in seen:
            seen.add(key)
            records.append(citation)
    # Preserve URLs even when a producer has one URL serving multiple source IDs.
    for url in source_urls:
        key = ("", url)
        if url and key not in seen and not any(item.get("url") == url for item in records):
            records.append({"source_id": "", "title": "", "publisher": "", "url": url, "published_date": "", "source_type": ""})
    return records


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write atomically without ``NamedTemporaryFile``'s Windows path probe.

    ``NamedTemporaryFile(dir=...)`` can stall on some Windows installations
    when the directory includes non-ASCII characters.  A unique sibling file
    retains the same atomic ``os.replace`` behavior and works reliably for
    this local demo workspace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def write_deterministic_npz(path: Path, arrays: dict[str, Any]) -> None:
    """Write an NPZ with fixed ZIP metadata so identical inputs stay identical."""

    np = _numpy()
    destination = BytesIO()
    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            array_buffer = BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.asanyarray(arrays[key]),
                version=None,
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_buffer.getvalue())
    _atomic_write_bytes(path, destination.getvalue())


def build_index(
    data_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    embedding_backend: str = "auto",
    model_path: str | Path | None = None,
    hash_dimension: int = DEFAULT_HASH_DIMENSION,
    batch_size: int = DEFAULT_BGE_BATCH_SIZE,
    max_length: int = DEFAULT_BGE_MAX_LENGTH,
    device: str = "auto",
    min_companies: int = 1,
) -> dict[str, Any]:
    """Validate source data and build files consumed by :class:`LocalRetriever`.

    The returned manifest is also written as ``index_manifest.json``.  Generated
    outputs never modify the human-reviewed raw data directory.
    """

    report = validate_data(data_dir, min_companies=min_companies)
    if not report.ok:
        errors = "; ".join(
            f"{issue.file}{':' + str(issue.row) if issue.row else ''} {issue.message}"
            for issue in report.errors[:8]
        )
        raise ValueError(f"Data validation failed; index was not built. {errors}")

    paths = resolve_data_paths(data_dir)
    destination = Path(output_dir).expanduser().resolve() if output_dir else paths.derived
    companies = sorted(load_companies(paths.raw), key=lambda company: company["company_id"])
    sources_by_id = {source.get("source_id", ""): source for source in load_sources(paths.raw)}
    documents = [company_document(company) for company in companies]
    backend = create_embedding_backend(
        embedding_backend,
        model_path=model_path,
        hash_dimension=hash_dimension,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    vectors = backend.fit_encode(documents)
    np = _numpy()
    vectors = _l2_normalize(vectors)
    if vectors.shape[0] != len(companies):
        raise RuntimeError("Embedding backend returned a vector count different from the company count")

    docs_payload: list[dict[str, Any]] = []
    for company, document in zip(companies, documents, strict=True):
        docs_payload.append(
            {
                "company_id": company["company_id"],
                "document": document,
                "document_sha256": sha256_text(document),
                "source_ids": company["source_ids"],
                "sources": _source_records_for_company(company, sources_by_id),
            }
        )
    docs_text = "".join(stable_json(item) + "\n" for item in docs_payload)
    docs_hash = sha256_text(docs_text)

    arrays: dict[str, Any] = {
        "company_ids": np.asarray([company["company_id"] for company in companies]),
        "vectors": np.ascontiguousarray(vectors, dtype=np.float32),
    }
    if isinstance(backend, HashEmbeddingBackend) and backend.idf is not None:
        arrays["hash_idf"] = np.ascontiguousarray(backend.idf, dtype=np.float32)
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / "company_index.npz"
    docs_path = destination / "company_documents.jsonl"
    manifest_path = destination / "index_manifest.json"
    write_deterministic_npz(index_path, arrays)
    _atomic_write_text(docs_path, docs_text)
    manifest: dict[str, Any] = {
        "format": INDEX_FORMAT,
        "company_count": len(companies),
        "company_ids_sha256": sha256_text(stable_json([company["company_id"] for company in companies])),
        "document_sha256": docs_hash,
        "source_fingerprint": source_fingerprint(companies, sources_by_id.values()),
        "embedding": {
            "backend": backend.backend_name,
            "model": backend.model_name,
            "normalize": "l2",
            **backend.manifest_extra(),
        },
        "files": {
            "documents": docs_path.name,
            "vectors": index_path.name,
        },
    }
    _atomic_write_text(manifest_path, stable_json(manifest) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the demo's deterministic local company index.")
    parser.add_argument("--data-dir", default="data", help="data root or data/raw directory (default: data)")
    parser.add_argument("--output-dir", help="generated-index directory (default: <data-dir>/derived)")
    parser.add_argument(
        "--embedding-backend",
        choices=("auto", "hash", "bge-m3"),
        default="auto",
        help="auto uses BGE-M3 only when an existing --model-path is supplied",
    )
    parser.add_argument("--model-path", help="already-downloaded local BAAI/bge-m3 directory")
    parser.add_argument("--hash-dimension", type=int, default=DEFAULT_HASH_DIMENSION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BGE_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_BGE_MAX_LENGTH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-companies", type=int, default=1)
    args = parser.parse_args(argv)
    if args.min_companies < 1:
        parser.error("--min-companies must be at least 1")
    try:
        manifest = build_index(
            args.data_dir,
            output_dir=args.output_dir,
            embedding_backend=args.embedding_backend,
            model_path=args.model_path,
            hash_dimension=args.hash_dimension,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            min_companies=args.min_companies,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Index build failed: {exc}")
        return 1
    print(stable_json(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
