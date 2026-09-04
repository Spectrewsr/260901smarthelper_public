"""SQLite knowledge base for the local Changzhou investment demo.

The raw CSV/JSONL files remain the evidence-preserving source bundle.  This
module materialises a small, inspectable SQLite catalogue for exact queries,
FTS retrieval, geo lookup, graph traversal, the lead ledger, and data intake.
It deliberately stores confidence and coordinate precision alongside facts so
the UI never mistakes an inferred/demo value for a verified business fact.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import math
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from scripts.data_contract import (
    clean_text,
    load_companies,
    load_regional_assets,
    load_sector_profiles,
    load_sources,
    read_jsonl_rows,
    split_values,
)


SCHEMA_VERSION = "2026.09.advanced-rag.1"
MAX_IMPORT_ROWS = 2_000
MAX_IMPORT_FIELD_CHARS = 1_600
MAX_IMPORT_COLUMNS = 32

# Intake records are intended to describe public corporate evidence, not a
# contact database.  These patterns remove common personal identifiers before
# a CSV/manual record reaches the searchable knowledge base.  The retained
# preview metadata records that a field was redacted without retaining the
# original sensitive value.
_MOBILE_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[\s-]?)?1[3-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![\w.-])")
_IDENTITY_NUMBER_RE = re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])")

# These are district-level *reference* points, used only where the public
# corpus lacks a verified company address/coordinate.  The generated values
# carry explicit precision/provenance labels in every API response.
DISTRICT_REFERENCE_POINTS: dict[str, tuple[float, float]] = {
    "金坛区": (31.731, 119.597),
    "武进区": (31.688, 119.946),
    "新北区": (31.859, 119.960),
    "天宁区": (31.780, 119.970),
    "钟楼区": (31.780, 119.920),
    "溧阳市": (31.416, 119.484),
    "常州经开区": (31.770, 120.080),
    "常州市": (31.781, 119.974),
}

# The park data deliberately does not invent land, plant or incentive facts.
# These dimensions turn the heterogeneous source cards into one transparent
# comparison matrix: evidence coverage is a data-completeness aid, never an
# investment-suitability or policy-preference score.
PARK_BENCHMARK_DIMENSIONS: tuple[dict[str, str], ...] = (
    {
        "key": "industry_fit",
        "label": "产业适配",
        "metric_key": "sector_fit",
        "definition": "公开资料中可见的产业话题/赛道适配；不代表准入结论。",
    },
    {
        "key": "green_transition",
        "label": "绿色低碳公开基础",
        "metric_key": "near_zero_carbon_pilot",
        "definition": "公开名单或指标中的绿色低碳线索；不代表项目可获支持。",
    },
    {
        "key": "site_supply",
        "label": "载体供给",
        "metric_key": "site_supply",
        "definition": "土地、厂房、接入等载体可用性，必须以园区实时书面核验为准。",
    },
    {
        "key": "policy_basis",
        "label": "政策与服务依据",
        "metric_key": "policy_basis",
        "definition": "公开政策或服务依据；不构成任何优惠、资金或审批承诺。",
    },
)

PROJECT_TOPIC_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("新能源汽车与新能源", ("新能源", "汽车", "电池", "锂电", "储能", "光伏", "氢能"), ("项目备案与准入", "能耗与环评", "物流组织")),
    ("新材料与化工新材料", ("新材料", "材料", "化工", "金属", "高分子"), ("项目备案与准入", "环保与安全条件", "物流组织")),
    ("智能制造与高端装备", ("装备", "制造", "自动化", "机器人", "工业软件"), ("企业开办", "人才服务", "物流组织")),
    ("生物医药与医疗器械", ("生物医药", "医疗", "器械", "药品", "诊断"), ("企业开办", "项目准入", "人才服务")),
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return split_values(value)
    return json_list(parsed)


def cjk_terms(value: str) -> list[str]:
    """Return compact FTS-friendly terms for unsegmented Chinese text."""

    text = clean_text(value).casefold()
    terms: list[str] = []
    ascii_word = ""
    cjk_run = ""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            if ascii_word:
                terms.append(ascii_word)
                ascii_word = ""
            cjk_run += char
            continue
        if cjk_run:
            if len(cjk_run) == 1:
                terms.append(cjk_run)
            else:
                terms.extend(cjk_run[index : index + 2] for index in range(len(cjk_run) - 1))
            cjk_run = ""
        if char.isalnum() or char in {"-", "_"}:
            ascii_word += char
        elif ascii_word:
            terms.append(ascii_word)
            ascii_word = ""
    if cjk_run:
        terms.extend(cjk_run[index : index + 2] for index in range(max(1, len(cjk_run) - 1)))
    if ascii_word:
        terms.append(ascii_word)
    return list(dict.fromkeys(term for term in terms if len(term) >= 2))


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius = 6371.0088
    latitude = math.radians(lat_b - lat_a)
    longitude = math.radians(lon_b - lon_a)
    a = math.sin(latitude / 2) ** 2 + math.cos(math.radians(lat_a)) * math.cos(math.radians(lat_b)) * math.sin(longitude / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def redact_intake_text(value: Any) -> tuple[str, list[str]]:
    """Remove direct personal identifiers from searchable intake text.

    The caller retains only field/type metadata in the import audit.  It never
    records the original identifier alongside the sanitized knowledge record.
    """

    text = clean_text(value)
    redactions: list[str] = []
    for pattern, label, replacement in (
        (_IDENTITY_NUMBER_RE, "身份证号", "[已脱敏身份证号]"),
        (_MOBILE_PHONE_RE, "手机号", "[已脱敏手机号]"),
        (_EMAIL_RE, "邮箱", "[已脱敏邮箱]"),
    ):
        text, count = pattern.subn(replacement, text)
        if count:
            redactions.append(label)
    return text, redactions


def intake_identity(company_name: Any, source_id: Any) -> str:
    """Normalize the minimum evidence identity used for duplicate warnings.

    A published company may legitimately have a similar name from a different
    source.  The import gate therefore only blocks an exact normalized
    ``name + source`` repeat rather than trying to guess legal-entity identity.
    """

    name = re.sub(r"\s+", "", clean_text(company_name)).casefold()
    source = clean_text(source_id).casefold()
    return f"{name}|{source}" if name and source else ""


class KnowledgeRepository:
    """Thread-safe SQLite adapter used by all exact and evidence operations."""

    # A repository can be instantiated more than once in tests, a CLI, or a
    # reload transition.  Locks therefore belong to the database path rather
    # than a single adapter instance: a seed refresh cannot take a snapshot
    # while another instance is adding a ledger and then overwrite that write.
    _database_lock_guard = threading.Lock()
    _database_locks: dict[str, threading.RLock] = {}

    @classmethod
    def _lock_for_database(cls, db_path: Path) -> threading.RLock:
        key = str(db_path.resolve()).casefold()
        with cls._database_lock_guard:
            lock = cls._database_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._database_locks[key] = lock
            return lock

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.raw_dir = self.data_dir / "raw"
        self.derived_dir = self.data_dir / "derived"
        self.db_path = self.derived_dir / "knowledge.db"
        self._lock = self._lock_for_database(self.db_path)
        self._ready = False

    # -- lifecycle -------------------------------------------------------

    def ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self.derived_dir.mkdir(parents=True, exist_ok=True)
            fingerprint = self._seed_fingerprint()
            rebuild = not self.db_path.exists()
            preserve_operations = False
            if not rebuild:
                try:
                    with self._connect() as connection:
                        metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
                    schema_changed = metadata.get("schema_version") != SCHEMA_VERSION
                    seed_changed = metadata.get("seed_fingerprint") != fingerprint
                    rebuild = schema_changed or seed_changed
                    # Raw public files may be refreshed independently of the
                    # operational catalogue.  A seed refresh therefore builds
                    # a fresh evidence cache and copies user-created companies,
                    # ledgers, import audit rows and export audit rows forward.
                    # This avoids silently deleting a招商台账 merely because a
                    # public source was corrected.
                    preserve_operations = rebuild
                except sqlite3.DatabaseError as exc:
                    # Never mistake a locked/corrupt/temporarily unavailable
                    # operational database for a disposable cache.  Failing
                    # closed keeps a ledger intact until the local operator can
                    # retry or recover it, rather than replacing it with a
                    # blank seed database.
                    raise RuntimeError("无法安全读取本地知识库；为保护台账，已拒绝自动重建。请关闭占用后重试。") from exc
            if rebuild:
                self._rebuild(fingerprint, preserve_operations=preserve_operations)
            self._ready = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived transaction and always release Windows file handles.

        ``sqlite3.Connection``'s own context manager commits/rolls back but
        does not close the handle.  Closing here matters for a local Windows
        demo: import/rebuild tests must be able to discard an isolated database
        immediately after a request finishes.
        """
        # Every repository query/mutation uses this guard.  The lifecycle path
        # holds the same re-entrant lock while snapshotting/replacing a raw-data
        # cache, so an in-process import/ledger write cannot interleave with
        # that refresh and produce orphaned audit rows.
        with self._lock:
            connection = sqlite3.connect(self.db_path, timeout=15, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA journal_mode = WAL")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def _seed_fingerprint(self) -> str:
        names = (
            "companies.csv",
            "sources.csv",
            "sector_profiles.jsonl",
            "regional_assets.jsonl",
            "parks.jsonl",
            "landing_support.jsonl",
        )
        digest = hashlib.sha256(SCHEMA_VERSION.encode("utf-8"))
        for name in names:
            path = self.raw_dir / name
            digest.update(name.encode("utf-8"))
            if path.exists():
                stat = path.stat()
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def _rebuild(self, fingerprint: str, *, preserve_operations: bool = False) -> None:
        """Build a fresh evidence cache and, when possible, retain user work.

        Public source records are reproducible cache material.  Lead ledgers,
        reviewed imports, generated-export audit entries and MAN-* companies
        are not.  Keeping the latter during an ordinary seed refresh is what
        makes the local electronic ledger safe to use beyond a single run.
        """

        operational_state = self._snapshot_operational_state() if preserve_operations else {}
        temporary = self.db_path.with_suffix(".building.db")
        if temporary.exists():
            temporary.unlink()
        connection = sqlite3.connect(temporary)
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript(self._schema_sql())
            self._seed_database(connection)
            if operational_state:
                self._restore_operational_state(connection, operational_state)
                self._rebuild_graph(connection)
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [("schema_version", SCHEMA_VERSION), ("seed_fingerprint", fingerprint), ("built_at", utc_now())],
            )
            connection.commit()
        finally:
            connection.close()
        temporary.replace(self.db_path)

    def _snapshot_operational_state(self) -> dict[str, list[dict[str, Any]]]:
        """Read only non-reproducible rows from an existing healthy database.

        This deliberately keeps all source cards as a small evidence archive:
        a manual/imported company can remain traceable even if a subsequent raw
        source bundle no longer carries its historic source row.
        """

        if not self.db_path.is_file():
            return {}
        try:
            connection = sqlite3.connect(self.db_path)
            connection.row_factory = sqlite3.Row
            try:
                # A single read transaction provides one consistent view across
                # parents and child audit rows while a normal app writer is
                # active.  Mutating paths are also serialized with _lock.
                connection.execute("BEGIN")
                table_names = {
                    str(row[0])
                    for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                required = {
                    "sources", "companies", "company_tags", "knowledge_chunks", "users", "ledgers",
                    "ledger_events", "import_jobs", "import_rows", "exports",
                }
                if not required.issubset(table_names):
                    raise RuntimeError("本地知识库结构不完整；为保护台账，已拒绝自动重建。")
                ledgers = [dict(row) for row in connection.execute("SELECT * FROM ledgers ORDER BY ledger_id")]
                ledger_company_ids = {clean_text(row.get("company_id")) for row in ledgers if clean_text(row.get("company_id"))}
                manual_company_ids = {
                    str(row[0])
                    for row in connection.execute("SELECT company_id FROM companies WHERE company_id GLOB 'MAN-*'").fetchall()
                }
                company_ids = sorted(manual_company_ids | ledger_company_ids)
                if company_ids:
                    placeholders = ",".join("?" for _ in company_ids)
                    companies = [
                        dict(row)
                        for row in connection.execute(
                            f"SELECT * FROM companies WHERE company_id IN ({placeholders}) ORDER BY company_id",
                            company_ids,
                        )
                    ]
                    company_tags = [
                        dict(row)
                        for row in connection.execute(
                            f"SELECT * FROM company_tags WHERE company_id IN ({placeholders}) ORDER BY company_id, tag_type, tag_value",
                            company_ids,
                        )
                    ]
                    chunks = [
                        dict(row)
                        for row in connection.execute(
                            f"SELECT * FROM knowledge_chunks WHERE parent_type='company' AND parent_id IN ({placeholders}) ORDER BY chunk_id",
                            company_ids,
                        )
                    ]
                else:
                    companies, company_tags, chunks = [], [], []
                return {
                    "sources": [dict(row) for row in connection.execute("SELECT * FROM sources ORDER BY source_id")],
                    "companies": companies,
                    "company_tags": company_tags,
                    "knowledge_chunks": chunks,
                    "users": [dict(row) for row in connection.execute("SELECT * FROM users ORDER BY username")],
                    "ledgers": ledgers,
                    "ledger_events": [dict(row) for row in connection.execute("SELECT * FROM ledger_events ORDER BY event_id")],
                    "import_jobs": [dict(row) for row in connection.execute("SELECT * FROM import_jobs ORDER BY created_at, import_id")],
                    "import_rows": [dict(row) for row in connection.execute("SELECT * FROM import_rows ORDER BY import_id, row_number")],
                    "exports": [dict(row) for row in connection.execute("SELECT * FROM exports ORDER BY created_at, export_id")],
                }
            finally:
                try:
                    connection.rollback()
                except sqlite3.DatabaseError:
                    pass
                connection.close()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("无法一致性快照本地台账；为避免数据丢失，已拒绝自动重建。") from exc

    @staticmethod
    def _insert_snapshot_rows(
        connection: sqlite3.Connection,
        table: str,
        rows: Iterable[Mapping[str, Any]],
        columns: tuple[str, ...],
        *,
        replace: bool = False,
    ) -> None:
        values = [tuple(row.get(column) for column in columns) for row in rows]
        if not values:
            return
        operation = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        placeholders = ", ".join("?" for _ in columns)
        connection.executemany(
            f"{operation} INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )

    def _restore_operational_state(self, connection: sqlite3.Connection, state: Mapping[str, list[dict[str, Any]]]) -> None:
        """Restore snapshot rows in foreign-key order into a fresh cache."""

        self._insert_snapshot_rows(
            connection,
            "sources",
            state.get("sources", []),
            ("source_id", "title", "publisher", "url", "published_date", "accessed_date", "source_type", "scope", "notes"),
        )
        company_columns = (
            "company_id", "name", "aliases", "english_name", "district", "park", "address", "latitude", "longitude",
            "coordinate_source", "coordinate_precision", "geo_confidence", "sector", "industry", "supply_chain_role", "products",
            "capabilities", "input_materials", "output_products", "target_customer_industries", "summary", "source_ids", "source_date",
            "verified_date", "confidence", "notes", "status", "created_at", "updated_at",
        )
        existing_company_ids = {
            str(row[0])
            for row in connection.execute("SELECT company_id FROM companies").fetchall()
        }
        manual_collisions = {
            str(row.get("company_id") or "")
            for row in state.get("companies", [])
            if str(row.get("company_id") or "").upper().startswith("MAN-")
            and str(row.get("company_id") or "") in existing_company_ids
        }
        if manual_collisions:
            joined = "、".join(sorted(manual_collisions)[:3])
            raise RuntimeError(f"原始资料与人工录入企业 ID 冲突（{joined}）；已拒绝刷新以保护台账。")
        company_rows = [row for row in state.get("companies", []) if str(row.get("company_id") or "") not in existing_company_ids]
        restored_company_ids = {str(row.get("company_id") or "") for row in company_rows}
        self._insert_snapshot_rows(connection, "companies", company_rows, company_columns)
        for row in company_rows:
            company_id = str(row.get("company_id") or "")
            location = connection.execute(
                "SELECT row_id, latitude, longitude FROM companies WHERE company_id=?", (company_id,)
            ).fetchone()
            if location and location["latitude"] is not None and location["longitude"] is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO company_geo_index VALUES (?, ?, ?, ?, ?)",
                    (
                        int(location["row_id"]), float(location["latitude"]), float(location["latitude"]),
                        float(location["longitude"]), float(location["longitude"]),
                    ),
                )
        self._insert_snapshot_rows(
            connection,
            "company_tags",
            [row for row in state.get("company_tags", []) if str(row.get("company_id") or "") in restored_company_ids],
            ("company_id", "tag_type", "tag_value"),
        )
        chunk_columns = (
            "chunk_id", "collection", "parent_type", "parent_id", "title", "context", "content", "source_ids", "confidence", "verified_date",
        )
        chunks = [row for row in state.get("knowledge_chunks", []) if str(row.get("parent_id") or "") in restored_company_ids]
        self._insert_snapshot_rows(connection, "knowledge_chunks", chunks, chunk_columns, replace=True)
        for chunk in chunks:
            full_content = str(chunk.get("content") or "")
            connection.execute("DELETE FROM knowledge_fts WHERE chunk_id=?", (chunk.get("chunk_id"),))
            connection.execute(
                "INSERT INTO knowledge_fts(chunk_id, token_text) VALUES (?, ?)",
                (chunk.get("chunk_id"), " ".join(cjk_terms(full_content))),
            )
        self._insert_snapshot_rows(
            connection,
            "users",
            state.get("users", []),
            ("username", "display_name", "role", "password_hash", "password_salt"),
            replace=True,
        )
        self._insert_snapshot_rows(
            connection,
            "ledgers",
            state.get("ledgers", []),
            ("ledger_id", "company_id", "project_name", "stage", "owner", "next_step", "contact_name", "contact_phone", "contact_email", "sensitivity", "created_by", "created_at", "updated_at"),
        )
        self._insert_snapshot_rows(
            connection,
            "ledger_events",
            state.get("ledger_events", []),
            ("event_id", "ledger_id", "event_type", "detail", "actor", "created_at"),
        )
        self._insert_snapshot_rows(
            connection,
            "import_jobs",
            state.get("import_jobs", []),
            ("import_id", "created_by", "status", "filename", "created_at", "reviewed_at"),
        )
        self._insert_snapshot_rows(
            connection,
            "import_rows",
            state.get("import_rows", []),
            ("import_id", "row_number", "payload", "valid", "errors", "published_company_id"),
        )
        self._insert_snapshot_rows(
            connection,
            "exports",
            state.get("exports", []),
            ("export_id", "export_type", "format", "file_path", "created_by", "created_at", "record_ids"),
        )

    @staticmethod
    def _schema_sql() -> str:
        return """
        PRAGMA foreign_keys = ON;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE sources (
          source_id TEXT PRIMARY KEY, title TEXT NOT NULL, publisher TEXT, url TEXT,
          published_date TEXT, accessed_date TEXT, source_type TEXT, scope TEXT, notes TEXT
        );
        CREATE TABLE companies (
          row_id INTEGER PRIMARY KEY AUTOINCREMENT, company_id TEXT UNIQUE NOT NULL,
          name TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]', english_name TEXT,
          district TEXT, park TEXT, address TEXT, latitude REAL, longitude REAL,
          coordinate_source TEXT, coordinate_precision TEXT, geo_confidence TEXT,
          sector TEXT, industry TEXT, supply_chain_role TEXT NOT NULL DEFAULT '[]',
          products TEXT NOT NULL DEFAULT '[]', capabilities TEXT NOT NULL DEFAULT '[]',
          input_materials TEXT NOT NULL DEFAULT '[]', output_products TEXT NOT NULL DEFAULT '[]',
          target_customer_industries TEXT NOT NULL DEFAULT '[]', summary TEXT,
          source_ids TEXT NOT NULL DEFAULT '[]', source_date TEXT, verified_date TEXT,
          confidence TEXT, notes TEXT, status TEXT NOT NULL DEFAULT 'published',
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE company_tags (
          company_id TEXT NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
          tag_type TEXT NOT NULL, tag_value TEXT NOT NULL,
          PRIMARY KEY (company_id, tag_type, tag_value)
        );
        CREATE TABLE parks (
          park_id TEXT PRIMARY KEY, asset_id TEXT, name TEXT NOT NULL, district TEXT,
          summary TEXT, latitude REAL, longitude REAL, coordinate_source TEXT,
          coordinate_precision TEXT, confidence TEXT, verified_date TEXT,
          source_ids TEXT NOT NULL DEFAULT '[]', sector_fit TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE park_metrics (
          metric_id INTEGER PRIMARY KEY AUTOINCREMENT, park_id TEXT NOT NULL REFERENCES parks(park_id) ON DELETE CASCADE,
          metric_key TEXT NOT NULL, label TEXT NOT NULL, value TEXT NOT NULL,
          source_id TEXT, confidence TEXT NOT NULL DEFAULT 'unknown'
        );
        CREATE TABLE landing_services (
          service_id TEXT PRIMARY KEY, category TEXT NOT NULL, name TEXT NOT NULL,
          district TEXT, summary TEXT, source_ids TEXT NOT NULL DEFAULT '[]', confidence TEXT,
          verified_date TEXT, tags TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE knowledge_chunks (
          chunk_id TEXT PRIMARY KEY, collection TEXT NOT NULL, parent_type TEXT NOT NULL,
          parent_id TEXT NOT NULL, title TEXT NOT NULL, context TEXT NOT NULL, content TEXT NOT NULL,
          source_ids TEXT NOT NULL DEFAULT '[]', confidence TEXT, verified_date TEXT
        );
        CREATE VIRTUAL TABLE knowledge_fts USING fts5(chunk_id UNINDEXED, token_text);
        CREATE VIRTUAL TABLE company_geo_index USING rtree(row_id, min_lat, max_lat, min_lon, max_lon);
        CREATE VIRTUAL TABLE park_geo_index USING rtree(row_id, min_lat, max_lat, min_lon, max_lon);
        CREATE TABLE graph_nodes (
          node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, label TEXT NOT NULL,
          source_ids TEXT NOT NULL DEFAULT '[]', confidence TEXT NOT NULL DEFAULT 'unknown'
        );
        CREATE TABLE graph_edges (
          edge_id TEXT PRIMARY KEY, from_node TEXT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
          to_node TEXT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
          relation TEXT NOT NULL, evidence TEXT NOT NULL, source_ids TEXT NOT NULL DEFAULT '[]',
          confidence TEXT NOT NULL DEFAULT 'unknown', inferred INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE users (
          username TEXT PRIMARY KEY, display_name TEXT NOT NULL, role TEXT NOT NULL,
          password_hash TEXT NOT NULL, password_salt TEXT NOT NULL
        );
        CREATE TABLE ledgers (
          ledger_id INTEGER PRIMARY KEY AUTOINCREMENT, company_id TEXT NOT NULL REFERENCES companies(company_id),
          project_name TEXT NOT NULL, stage TEXT NOT NULL, owner TEXT NOT NULL, next_step TEXT NOT NULL,
          contact_name TEXT, contact_phone TEXT, contact_email TEXT, sensitivity TEXT NOT NULL DEFAULT 'internal',
          created_by TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE ledger_events (
          event_id INTEGER PRIMARY KEY AUTOINCREMENT, ledger_id INTEGER NOT NULL REFERENCES ledgers(ledger_id) ON DELETE CASCADE,
          event_type TEXT NOT NULL, detail TEXT NOT NULL, actor TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE import_jobs (
          import_id TEXT PRIMARY KEY, created_by TEXT NOT NULL, status TEXT NOT NULL,
          filename TEXT, created_at TEXT NOT NULL, reviewed_at TEXT
        );
        CREATE TABLE import_rows (
          import_id TEXT NOT NULL REFERENCES import_jobs(import_id) ON DELETE CASCADE,
          row_number INTEGER NOT NULL, payload TEXT NOT NULL, valid INTEGER NOT NULL,
          errors TEXT NOT NULL DEFAULT '[]', published_company_id TEXT,
          PRIMARY KEY (import_id, row_number)
        );
        CREATE TABLE exports (
          export_id TEXT PRIMARY KEY, export_type TEXT NOT NULL, format TEXT NOT NULL,
          file_path TEXT NOT NULL, created_by TEXT NOT NULL, created_at TEXT NOT NULL,
          record_ids TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX company_filters ON companies(status, district, sector, company_id);
        CREATE INDEX tag_lookup ON company_tags(tag_type, tag_value);
        CREATE INDEX graph_from ON graph_edges(from_node);
        CREATE INDEX graph_to ON graph_edges(to_node);
        """

    # -- seed ------------------------------------------------------------

    def _seed_database(self, connection: sqlite3.Connection) -> None:
        source_rows = load_sources(self.data_dir)
        source_by_id = {row["source_id"]: row for row in source_rows}
        for source in source_rows:
            connection.execute(
                """INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    source["source_id"], source.get("title", "公开资料来源"), source.get("publisher", ""),
                    source.get("url", ""), source.get("published_date", ""), source.get("accessed_date", ""),
                    source.get("source_type", ""), source.get("scope", ""), source.get("notes", ""),
                ),
            )

        companies = load_companies(self.data_dir)
        for company in companies:
            if clean_text(company.get("company_id")).upper().startswith("MAN-"):
                raise ValueError("原始 companies.csv 不得使用保留的 MAN- 人工录入企业 ID 前缀")
            self._insert_company(connection, company, source_by_id=source_by_id, status="published")

        parks = read_jsonl_rows(self.raw_dir / "parks.jsonl") if (self.raw_dir / "parks.jsonl").exists() else []
        for park in parks:
            connection.execute(
                """INSERT INTO parks(park_id, asset_id, name, district, summary, latitude, longitude,
                   coordinate_source, coordinate_precision, confidence, verified_date, source_ids, sector_fit)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    clean_text(park.get("park_id")), clean_text(park.get("asset_id")), clean_text(park.get("name")),
                    clean_text(park.get("district")), clean_text(park.get("summary")), park.get("latitude"), park.get("longitude"),
                    clean_text(park.get("coordinate_source")), clean_text(park.get("coordinate_precision")),
                    clean_text(park.get("confidence")), clean_text(park.get("verified_date")),
                    json_text(json_list(park.get("source_ids"))), json_text(json_list(park.get("sector_fit"))),
                ),
            )
            row_id = int(connection.execute("SELECT rowid FROM parks WHERE park_id = ?", (clean_text(park.get("park_id")),)).fetchone()[0])
            if park.get("latitude") is not None and park.get("longitude") is not None:
                connection.execute(
                    "INSERT INTO park_geo_index VALUES (?, ?, ?, ?, ?)",
                    (row_id, float(park["latitude"]), float(park["latitude"]), float(park["longitude"]), float(park["longitude"])),
                )
            for metric in park.get("metrics", []):
                if isinstance(metric, Mapping):
                    connection.execute(
                        "INSERT INTO park_metrics(park_id, metric_key, label, value, source_id, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            clean_text(park.get("park_id")), clean_text(metric.get("key")), clean_text(metric.get("label")),
                            clean_text(metric.get("value")), clean_text(metric.get("source_id")), clean_text(metric.get("confidence", "unknown")),
                        ),
                    )
            self._add_chunk(
                connection,
                collection="park_negotiation",
                parent_type="park",
                parent_id=clean_text(park.get("park_id")),
                title=clean_text(park.get("name")),
                context=(f"知识库:园区与招商对标；园区:{clean_text(park.get('name'))}；区域:{clean_text(park.get('district'))}；"
                         f"产业适配:{'、'.join(json_list(park.get('sector_fit')))}；来源:{'、'.join(json_list(park.get('source_ids')))}；"
                         f"置信度:{clean_text(park.get('confidence'))}"),
                content=clean_text(park.get("summary")),
                source_ids=json_list(park.get("source_ids")),
                confidence=clean_text(park.get("confidence")),
                verified_date=clean_text(park.get("verified_date")),
            )

        landing_rows = read_jsonl_rows(self.raw_dir / "landing_support.jsonl") if (self.raw_dir / "landing_support.jsonl").exists() else []
        for item in landing_rows:
            connection.execute(
                """INSERT INTO landing_services(service_id, category, name, district, summary, source_ids, confidence, verified_date, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    clean_text(item.get("service_id")), clean_text(item.get("category")), clean_text(item.get("name")),
                    clean_text(item.get("district")), clean_text(item.get("summary")), json_text(json_list(item.get("source_ids"))),
                    clean_text(item.get("confidence")), clean_text(item.get("verified_date")), json_text(json_list(item.get("tags"))),
                ),
            )
            self._add_chunk(
                connection,
                collection="landing_support",
                parent_type="landing_service",
                parent_id=clean_text(item.get("service_id")),
                title=clean_text(item.get("name")),
                context=(f"知识库:本地化落地配套；类别:{clean_text(item.get('category'))}；区域:{clean_text(item.get('district'))}；"
                         f"服务:{clean_text(item.get('name'))}；标签:{'、'.join(json_list(item.get('tags')))}；"
                         f"来源:{'、'.join(json_list(item.get('source_ids')))}；置信度:{clean_text(item.get('confidence'))}"),
                content=clean_text(item.get("summary")),
                source_ids=json_list(item.get("source_ids")),
                confidence=clean_text(item.get("confidence")),
                verified_date=clean_text(item.get("verified_date")),
            )

        for profile in load_sector_profiles(self.data_dir):
            source_ids = json_list(profile.get("source_ids"))
            self._add_chunk(
                connection,
                collection="enterprise_chain",
                parent_type="sector_profile",
                parent_id=clean_text(profile.get("sector_id")),
                title=clean_text(profile.get("name")),
                context=(f"知识库:企业与产业链；产业:{clean_text(profile.get('name'))}；上游:{'、'.join(json_list(profile.get('upstream')))}；"
                         f"中游:{'、'.join(json_list(profile.get('midstream')))}；下游:{'、'.join(json_list(profile.get('downstream')))}；"
                         f"来源:{'、'.join(source_ids)}"),
                content=clean_text(profile.get("overview")),
                source_ids=source_ids,
                confidence="B",
                verified_date=clean_text(profile.get("verified_date")),
            )

        # The historical regional assets stay available as evidence chunks too.
        for asset in load_regional_assets(self.data_dir):
            source_ids = json_list(asset.get("source_ids"))
            self._add_chunk(
                connection,
                collection="landing_support",
                parent_type="regional_asset",
                parent_id=clean_text(asset.get("asset_id")),
                title=clean_text(asset.get("name")),
                context=(f"知识库:本地化落地配套；类别:{clean_text(asset.get('asset_type'))}；区域:{clean_text(asset.get('district'))}；"
                         f"服务产业:{'、'.join(json_list(asset.get('serves_sectors')))}；来源:{'、'.join(source_ids)}；"
                         f"置信度:{clean_text(asset.get('confidence'))}"),
                content=clean_text(asset.get("description")),
                source_ids=source_ids,
                confidence=clean_text(asset.get("confidence")),
                verified_date=clean_text(asset.get("verified_date")),
            )

        self._seed_users(connection)
        self._rebuild_graph(connection)

    def _insert_company(
        self,
        connection: sqlite3.Connection,
        company: Mapping[str, Any],
        *,
        source_by_id: Mapping[str, Mapping[str, Any]] | None = None,
        status: str = "published",
    ) -> str:
        company_id = clean_text(company.get("company_id")) or f"MAN-{uuid.uuid4().hex[:8].upper()}"
        district = clean_text(company.get("district")) or "常州市"
        latitude, longitude, coordinate_source, coordinate_precision, geo_confidence = self._company_coordinate(company_id, district, company)
        source_ids = json_list(company.get("source_ids") or company.get("source_id"))
        if not source_ids and clean_text(company.get("source_id")):
            source_ids = [clean_text(company.get("source_id"))]
        now = utc_now()
        values = (
            company_id, clean_text(company.get("company_name") or company.get("name")), json_text(json_list(company.get("aliases"))),
            clean_text(company.get("english_name")), district, clean_text(company.get("park")), clean_text(company.get("address")),
            latitude, longitude, coordinate_source, coordinate_precision, geo_confidence,
            clean_text(company.get("industry_track") or company.get("sector")), clean_text(company.get("industry_subtrack") or company.get("industry")),
            json_text(json_list(company.get("supply_chain_role"))), json_text(json_list(company.get("products"))),
            json_text(json_list(company.get("capabilities"))), json_text(json_list(company.get("input_materials"))),
            json_text(json_list(company.get("output_products"))), json_text(json_list(company.get("target_customer_industries"))),
            clean_text(company.get("summary") or company.get("description")), json_text(source_ids), clean_text(company.get("source_date")),
            clean_text(company.get("verified_date")), clean_text(company.get("confidence") or "medium"), clean_text(company.get("notes")), status, now, now,
        )
        connection.execute(
            """INSERT INTO companies(company_id, name, aliases, english_name, district, park, address, latitude, longitude,
               coordinate_source, coordinate_precision, geo_confidence, sector, industry, supply_chain_role, products,
               capabilities, input_materials, output_products, target_customer_industries, summary, source_ids, source_date,
               verified_date, confidence, notes, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
        row_id = int(connection.execute("SELECT row_id FROM companies WHERE company_id = ?", (company_id,)).fetchone()[0])
        if latitude is not None and longitude is not None:
            connection.execute("INSERT INTO company_geo_index VALUES (?, ?, ?, ?, ?)", (row_id, latitude, latitude, longitude, longitude))
        tags = {
            "sector": [clean_text(company.get("industry_track") or company.get("sector"))],
            "industry": [clean_text(company.get("industry_subtrack") or company.get("industry"))],
            "role": json_list(company.get("supply_chain_role")),
            "product": json_list(company.get("products")),
            "capability": json_list(company.get("capabilities")),
            "input": json_list(company.get("input_materials")),
            "output": json_list(company.get("output_products")),
            "customer": json_list(company.get("target_customer_industries")),
        }
        for tag_type, values_for_type in tags.items():
            for tag in values_for_type:
                if tag:
                    connection.execute("INSERT OR IGNORE INTO company_tags VALUES (?, ?, ?)", (company_id, tag_type, tag))
        context = (
            f"知识库:企业与产业链；企业:{clean_text(company.get('company_name') or company.get('name'))}；区域:{district}；"
            f"产业:{clean_text(company.get('industry_track') or company.get('sector'))}；细分:{clean_text(company.get('industry_subtrack') or company.get('industry'))}；"
            f"产品:{'、'.join(json_list(company.get('products')))}；能力:{'、'.join(json_list(company.get('capabilities')))}；"
            f"输入:{'、'.join(json_list(company.get('input_materials')))}；输出:{'、'.join(json_list(company.get('output_products')))}；"
            f"目标客户行业:{'、'.join(json_list(company.get('target_customer_industries')))}；"
            f"角色:{'、'.join(json_list(company.get('supply_chain_role')))}；来源:{'、'.join(source_ids)}；"
            f"置信度:{clean_text(company.get('confidence') or 'medium')}"
        )
        self._add_chunk(
            connection,
            collection="enterprise_chain",
            parent_type="company",
            parent_id=company_id,
            title=clean_text(company.get("company_name") or company.get("name")),
            context=context,
            content=clean_text(company.get("summary") or company.get("description")),
            source_ids=source_ids,
            confidence=clean_text(company.get("confidence") or "medium"),
            verified_date=clean_text(company.get("verified_date")),
        )
        return company_id

    @staticmethod
    def _company_coordinate(company_id: str, district: str, company: Mapping[str, Any]) -> tuple[float | None, float | None, str, str, str]:
        try:
            raw_lat = float(str(company.get("latitude") or ""))
            raw_lon = float(str(company.get("longitude") or ""))
            if -90 <= raw_lat <= 90 and -180 <= raw_lon <= 180:
                return raw_lat, raw_lon, "企业公开资料坐标", "verified_point", clean_text(company.get("confidence") or "medium")
        except (TypeError, ValueError):
            pass
        base = DISTRICT_REFERENCE_POINTS.get(district)
        if not base:
            return None, None, "未提供公开坐标", "unknown", "unknown"
        digest = int(hashlib.sha256(company_id.encode("utf-8")).hexdigest()[:12], 16)
        # ≤1.5 km deterministic spread only prevents every record appearing as
        # the same point; provenance remains district-reference, not address.
        lat_offset = ((digest % 1001) - 500) / 500_000
        lon_offset = (((digest // 1001) % 1001) - 500) / 450_000
        return base[0] + lat_offset, base[1] + lon_offset, "演示参考点：区县中心附近，非企业地址", "district_reference", "demo"

    @staticmethod
    def _add_chunk(
        connection: sqlite3.Connection,
        *,
        collection: str,
        parent_type: str,
        parent_id: str,
        title: str,
        context: str,
        content: str,
        source_ids: list[str],
        confidence: str,
        verified_date: str,
    ) -> None:
        chunk_id = f"{collection}:{parent_type}:{parent_id}"
        full_content = f"{context}\n{content}".strip()
        connection.execute(
            "INSERT OR REPLACE INTO knowledge_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chunk_id, collection, parent_type, parent_id, title, context, full_content, json_text(source_ids), confidence, verified_date),
        )
        connection.execute("DELETE FROM knowledge_fts WHERE chunk_id = ?", (chunk_id,))
        connection.execute("INSERT INTO knowledge_fts(chunk_id, token_text) VALUES (?, ?)", (chunk_id, " ".join(cjk_terms(full_content))))

    @staticmethod
    def _seed_users(connection: sqlite3.Connection) -> None:
        # Demo-only credentials are intentionally displayed in the local login
        # panel.  Passwords are PBKDF2 hashes, not plaintext DB values.
        for username, display_name, role, password in (
            ("admin", "演示管理员", "admin", "admin-demo-2026"),
            ("officer", "招商专员", "officer", "officer-demo-2026"),
        ):
            salt = hashlib.sha256(f"changzhou-demo:{username}".encode()).hexdigest()[:32]
            password_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 160_000).hex()
            connection.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (username, display_name, role, password_hash, salt))

    def _rebuild_graph(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM graph_edges")
        connection.execute("DELETE FROM graph_nodes")
        companies = connection.execute("SELECT * FROM companies WHERE status = 'published' ORDER BY company_id").fetchall()
        for row in companies:
            company = self._company_from_row(row, include_sources=False)
            node_id = f"company:{company['company_id']}"
            self._graph_node(connection, node_id, "company", company["name"], company["source_ids"], company["confidence"])
            sector = company.get("sector") or "未分类产业"
            sector_id = f"sector:{hashlib.sha1(sector.encode('utf-8')).hexdigest()[:12]}"
            self._graph_node(connection, sector_id, "sector", sector, company["source_ids"], "B")
            self._graph_edge(connection, node_id, sector_id, "belongs_to", "公开资料中的产业赛道字段", company["source_ids"], company["confidence"], False)
            for product in company.get("products", []):
                product_id = f"product:{hashlib.sha1(product.encode('utf-8')).hexdigest()[:12]}"
                self._graph_node(connection, product_id, "product", product, company["source_ids"], company["confidence"])
                self._graph_edge(connection, node_id, product_id, "produces", "公开资料中的产品字段", company["source_ids"], company["confidence"], False)
            stage = self._stage_for_company(company)
            stage_id = f"stage:{stage}"
            self._graph_node(connection, stage_id, "chain_stage", stage, ["S003", "S007"], "B")
            self._graph_edge(connection, node_id, stage_id, "potential_role", "根据公开产品/行业字段归纳的产业链位置，待核验", company["source_ids"], "B", True)
        # Direct potential edges make the locally scoped GraphRAG traversable in
        # two hops without ever representing an actual confirmed transaction.
        staged: dict[str, list[dict[str, Any]]] = {"upstream_material": [], "midstream_battery": [], "downstream_vehicle": []}
        for row in companies:
            company = self._company_from_row(row, include_sources=False)
            staged.setdefault(self._stage_for_company(company), []).append(company)
        for upstream in staged.get("upstream_material", []):
            for battery in staged.get("midstream_battery", []):
                self._graph_edge(
                    connection, f"company:{upstream['company_id']}", f"company:{battery['company_id']}",
                    "potential_material_supply", "产业链结构推断：材料企业与动力电池制造企业的潜在供需方向，非已证实合作",
                    sorted(set(upstream["source_ids"] + battery["source_ids"])), "B", True,
                )
        for battery in staged.get("midstream_battery", []):
            for vehicle in staged.get("downstream_vehicle", []):
                self._graph_edge(
                    connection, f"company:{battery['company_id']}", f"company:{vehicle['company_id']}",
                    "potential_battery_supply", "产业链结构推断：动力电池与整车制造企业的潜在供需方向，非已证实合作",
                    sorted(set(battery["source_ids"] + vehicle["source_ids"])), "B", True,
                )
        for park in connection.execute("SELECT * FROM parks").fetchall():
            park_id = f"park:{park['park_id']}"
            source_ids = json_list(park["source_ids"])
            self._graph_node(connection, park_id, "park", str(park["name"]), source_ids, str(park["confidence"] or "B"))
            for sector in json_list(park["sector_fit"]):
                sector_id = f"sector:{hashlib.sha1(sector.encode('utf-8')).hexdigest()[:12]}"
                self._graph_node(connection, sector_id, "sector", sector, source_ids, "B")
                self._graph_edge(connection, park_id, sector_id, "topic_fit", "公开试点园区资料中的产业话题适配，不代表招商承诺", source_ids, "B", True)

    @staticmethod
    def _stage_for_company(company: Mapping[str, Any]) -> str:
        text = " ".join(
            [company.get("industry", ""), company.get("sector", ""), *company.get("products", []), *company.get("supply_chain_role", [])]
        )
        if any(token in text for token in ("正极", "负极", "材料", "铝材", "铜材", "隔膜")):
            return "upstream_material"
        if "动力电池" in text or "电池制造" in text:
            return "midstream_battery"
        if "整车" in text or "新能源汽车制造" in text or "汽车" in text:
            return "downstream_vehicle"
        return "industry_support"

    @staticmethod
    def _graph_node(connection: sqlite3.Connection, node_id: str, node_type: str, label: str, source_ids: list[str], confidence: str) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO graph_nodes VALUES (?, ?, ?, ?, ?)",
            (node_id, node_type, label, json_text(source_ids), confidence or "unknown"),
        )

    @staticmethod
    def _graph_edge(
        connection: sqlite3.Connection,
        from_node: str,
        to_node: str,
        relation: str,
        evidence: str,
        source_ids: list[str],
        confidence: str,
        inferred: bool,
    ) -> None:
        if from_node == to_node:
            return
        marker = "|".join((from_node, to_node, relation))
        edge_id = hashlib.sha1(marker.encode("utf-8")).hexdigest()[:18]
        connection.execute(
            "INSERT OR REPLACE INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (edge_id, from_node, to_node, relation, evidence, json_text(source_ids), confidence or "unknown", int(inferred)),
        )

    # -- read APIs -------------------------------------------------------

    def status(self) -> dict[str, Any]:
        self.ensure_ready()
        with self._connect() as connection:
            counts = {
                "companies": int(connection.execute("SELECT COUNT(*) FROM companies WHERE status = 'published'").fetchone()[0]),
                "chunks": int(connection.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0]),
                "parks": int(connection.execute("SELECT COUNT(*) FROM parks").fetchone()[0]),
                "landing_services": int(connection.execute("SELECT COUNT(*) FROM landing_services").fetchone()[0]),
                "graph_edges": int(connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]),
            }
        return {
            "state": "ready",
            "company_count": counts["companies"],
            "knowledge_collections": {"enterprise_chain": counts["chunks"], "park_negotiation": counts["parks"], "landing_support": counts["landing_services"]},
            "counts": counts,
            "database": "本地 SQLite（路径不对外暴露）",
            "retrieval_backend": "SQLite FTS5 + BGE-M3 + RRF + Cross-Encoder",
            "message": f"已加载 {counts['companies']} 家企业、{counts['parks']} 个园区对标记录和 {counts['landing_services']} 条落地配套资料。",
        }

    def filter_options(self) -> dict[str, list[str]]:
        self.ensure_ready()
        with self._connect() as connection:
            return {
                "districts": [row[0] for row in connection.execute("SELECT DISTINCT district FROM companies WHERE status='published' AND district <> '' ORDER BY district")],
                # The landing-page selector must not present an enterprise
                # grouping for which the local-support corpus has no district
                # record.  General company filters retain every raw district.
                "landing_districts": ["常州市", *[row[0] for row in connection.execute("SELECT DISTINCT district FROM landing_services WHERE district <> '常州市' AND district <> '' ORDER BY district")]],
                "sectors": [row[0] for row in connection.execute("SELECT DISTINCT sector FROM companies WHERE status='published' AND sector <> '' ORDER BY sector")],
                "roles": [row[0] for row in connection.execute("SELECT DISTINCT tag_value FROM company_tags WHERE tag_type='role' ORDER BY tag_value")],
                "parks": [dict(row) for row in connection.execute("SELECT park_id, name FROM parks ORDER BY name")],
            }

    def published_companies_for_dense_extension(self) -> list[dict[str, Any]]:
        """Return current structured enterprise rows for the live dense delta.

        The persisted BGE index is intentionally built from the reviewed raw
        source bundle.  Newly reviewed CSV/manual records live in SQLite until
        the next full index build, so the Advanced-RAG facade embeds only that
        small delta in memory.  This keeps new data searchable immediately
        without mutating the reproducible raw index on each form submission.
        """

        self.ensure_ready()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM companies WHERE status='published' ORDER BY company_id"
            ).fetchall()
            return [self._company_from_row(row, connection=connection) for row in rows]

    def list_companies(
        self,
        *,
        query: str = "",
        district: str | None = None,
        sector: str | None = None,
        company_ids: Iterable[str] | None = None,
        page: int = 1,
        page_size: int = 24,
    ) -> dict[str, Any]:
        self.ensure_ready()
        clauses = ["status = 'published'"]
        params: list[Any] = []
        if district:
            clauses.append("district = ?")
            params.append(district)
        if sector:
            clauses.append("sector = ?")
            params.append(sector)
        if query:
            clauses.append("(name LIKE ? OR summary LIKE ? OR products LIKE ? OR capabilities LIKE ? OR industry LIKE ?)")
            token = f"%{query.strip()}%"
            params.extend([token] * 5)
        ids = [clean_text(item) for item in (company_ids or []) if clean_text(item)]
        if ids:
            clauses.append(f"company_id IN ({','.join('?' for _ in ids)})")
            params.extend(ids)
        where = " AND ".join(clauses)
        page = max(1, int(page))
        page_size = min(100, max(1, int(page_size)))
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM companies WHERE {where}", params).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM companies WHERE {where} ORDER BY company_id LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
            items = [self._company_from_row(row, connection=connection) for row in rows]
        return {"items": items, "total": total, "page": page, "page_size": page_size, "has_more": page * page_size < total}

    def get_company(self, company_id: str) -> dict[str, Any] | None:
        self.ensure_ready()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM companies WHERE company_id = ? AND status = 'published'", (company_id,)).fetchone()
            return self._company_from_row(row, connection=connection) if row else None

    def source_cards(self, source_ids: Iterable[str]) -> list[dict[str, str]]:
        self.ensure_ready()
        source_ids = list(dict.fromkeys(clean_text(item) for item in source_ids if clean_text(item)))
        if not source_ids:
            return []
        with self._connect() as connection:
            rows = connection.execute(f"SELECT * FROM sources WHERE source_id IN ({','.join('?' for _ in source_ids)})", source_ids).fetchall()
        by_id = {row["source_id"]: dict(row) for row in rows}
        return [by_id[source_id] for source_id in source_ids if source_id in by_id]

    def fts_search(self, query: str, *, collections: Iterable[str] | None = None, limit: int = 30) -> list[dict[str, Any]]:
        self.ensure_ready()
        terms = cjk_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:20])
        clauses = ["knowledge_fts MATCH ?"]
        params: list[Any] = [match]
        desired = [clean_text(value) for value in (collections or []) if clean_text(value)]
        if desired:
            clauses.append(f"c.collection IN ({','.join('?' for _ in desired)})")
            params.extend(desired)
        with self._connect() as connection:
            try:
                rows = connection.execute(
                    f"""SELECT c.*, bm25(knowledge_fts) AS bm25_score
                    FROM knowledge_fts JOIN knowledge_chunks c ON c.chunk_id = knowledge_fts.chunk_id
                    WHERE {' AND '.join(clauses)} ORDER BY bm25_score LIMIT ?""",
                    [*params, max(1, limit)],
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            if not rows:
                collection_clause = ""
                fallback_params: list[Any] = []
                if desired:
                    collection_clause = f" AND collection IN ({','.join('?' for _ in desired)})"
                    fallback_params.extend(desired)
                rows = connection.execute(
                    f"SELECT *, 0.0 AS bm25_score FROM knowledge_chunks WHERE ({' OR '.join('content LIKE ?' for _ in terms[:8])}){collection_clause} LIMIT ?",
                    [*(f"%{term}%" for term in terms[:8]), *fallback_params, max(1, limit)],
                ).fetchall()
        output: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            item = dict(row)
            item["source_ids"] = json_list(item.pop("source_ids", []))
            item["rank"] = rank
            item["bm25_score"] = round(float(item.get("bm25_score") or 0), 5)
            output.append(item)
        return output

    def company_by_ids(self, company_ids: Iterable[str]) -> list[dict[str, Any]]:
        wanted = [clean_text(item) for item in company_ids if clean_text(item)]
        if not wanted:
            return []
        payload = self.list_companies(company_ids=wanted, page_size=max(len(wanted), 1))
        by_id = {item["company_id"]: item for item in payload["items"]}
        return [by_id[item] for item in wanted if item in by_id]

    def geo_search(
        self,
        *,
        latitude: float,
        longitude: float,
        radius_km: float,
        entity_type: str = "company",
        limit: int = 30,
    ) -> dict[str, Any]:
        """RTree candidate filtering + Haversine, entirely outside RAG."""

        self.ensure_ready()
        radius_km = max(0.1, min(float(radius_km), 200.0))
        latitude = float(latitude)
        longitude = float(longitude)
        lat_delta = radius_km / 110.574
        lon_delta = radius_km / max(1.0, 111.320 * math.cos(math.radians(latitude)))
        table = "company_geo_index" if entity_type == "company" else "park_geo_index"
        parent = "companies" if entity_type == "company" else "parks"
        join_field = "row_id" if entity_type == "company" else "rowid"
        status_filter = "AND p.status = 'published'" if entity_type == "company" else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT p.* FROM {table} g JOIN {parent} p ON p.{join_field}=g.row_id
                WHERE g.min_lat >= ? AND g.max_lat <= ? AND g.min_lon >= ? AND g.max_lon <= ?
                {status_filter}""",
                (latitude - lat_delta, latitude + lat_delta, longitude - lon_delta, longitude + lon_delta),
            ).fetchall()
            output = []
            for row in rows:
                item = self._company_from_row(row, connection=connection) if entity_type == "company" else self._park_from_row(row, connection)
                straight = haversine_km(latitude, longitude, float(row["latitude"]), float(row["longitude"]))
                if straight > radius_km:
                    continue
                road = straight * 1.22 + 0.4
                item.update(
                    {
                        "straight_line_km": round(straight, 2),
                        "distance_km": round(road, 2),
                        "duration_min": max(1, round(road / 38 * 60)),
                        "method": "local_road_factor_estimate",
                        "method_note": "本地道路系数估算，不是实时导航或实测车程。",
                    }
                )
                output.append(item)
        output.sort(key=lambda item: (item["distance_km"], item.get("name", "")))
        return {
            "reference_point": {"latitude": latitude, "longitude": longitude, "radius_km": radius_km},
            "items": output[: max(1, int(limit))],
            "execution": {"mode": "sql_geo", "rag_called": False, "method": "RTree + Haversine + local road-factor estimate"},
        }

    def graph_expand(self, seed_company_ids: Iterable[str], *, max_hops: int = 2, limit: int = 24) -> dict[str, Any]:
        self.ensure_ready()
        seeds = [f"company:{clean_text(value)}" for value in seed_company_ids if clean_text(value)]
        if not seeds:
            return {"nodes": [], "edges": [], "max_hops": 0}
        max_hops = max(1, min(2, int(max_hops)))
        visited = set(seeds)
        frontier = set(seeds)
        edge_rows: list[sqlite3.Row] = []
        with self._connect() as connection:
            for _hop in range(max_hops):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                rows = connection.execute(
                    f"SELECT * FROM graph_edges WHERE from_node IN ({placeholders}) OR to_node IN ({placeholders})",
                    [*frontier, *frontier],
                ).fetchall()
                new_frontier: set[str] = set()
                for row in rows:
                    if row["edge_id"] in {item["edge_id"] for item in edge_rows}:
                        continue
                    edge_rows.append(row)
                    new_frontier.update({row["from_node"], row["to_node"]})
                frontier = new_frontier - visited
                visited.update(new_frontier)
                if len(edge_rows) >= limit:
                    break
            if not edge_rows:
                return {"nodes": [], "edges": [], "max_hops": max_hops}
            node_ids = sorted({node for edge in edge_rows for node in (edge["from_node"], edge["to_node"])})
            node_rows = connection.execute(f"SELECT * FROM graph_nodes WHERE node_id IN ({','.join('?' for _ in node_ids)})", node_ids).fetchall()
        nodes = [{**dict(row), "source_ids": json_list(row["source_ids"])} for row in node_rows]
        edges = [
            {
                **dict(row),
                "source_ids": json_list(row["source_ids"]),
                "inferred": bool(row["inferred"]),
                "status": "potential" if row["inferred"] else "source_fact",
            }
            for row in edge_rows[:limit]
        ]
        return {"nodes": nodes, "edges": edges, "max_hops": max_hops}

    def compare_parks(
        self,
        park_ids: Iterable[str],
        *,
        sector: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        """Compare parks through a fixed, evidence-first dimension matrix.

        The returned percentage is deliberately an *evidence-completeness*
        indicator.  It helps a招商人员 see what still needs to be verified; it
        is not an investment score, a land-supply conclusion, or a policy
        recommendation.
        """

        self.ensure_ready()
        wanted = [clean_text(value) for value in park_ids if clean_text(value)]
        if not wanted:
            with self._connect() as connection:
                wanted = [str(row[0]) for row in connection.execute("SELECT park_id FROM parks ORDER BY park_id LIMIT 3")]
        with self._connect() as connection:
            placeholders = ",".join("?" for _ in wanted)
            rows = connection.execute(
                f"SELECT * FROM parks WHERE park_id IN ({placeholders}) OR asset_id IN ({placeholders})",
                [*wanted, *wanted],
            ).fetchall()
            parks = [self._park_from_row(row, connection) for row in rows]
        by_reference = {
            reference: park
            for park in parks
            for reference in (park["park_id"], park.get("asset_id"))
            if reference
        }
        parks = [by_reference[reference] for reference in wanted if reference in by_reference]
        requested_sector = clean_text(sector)
        project_label = clean_text(project_name) or "本项目"
        matrix_rows: list[dict[str, Any]] = []
        pending_by_park: dict[str, list[str]] = {}
        for park in parks:
            dimensions: list[dict[str, Any]] = []
            metrics_by_key = {clean_text(item.get("metric_key")): item for item in park.get("metrics", [])}
            for definition in PARK_BENCHMARK_DIMENSIONS:
                key = definition["key"]
                if key == "industry_fit":
                    sector_fit = list(park.get("sector_fit", []))
                    matched = requested_sector in sector_fit if requested_sector else None
                    evidence = metrics_by_key.get("sector_fit")
                    source_ids = [clean_text(evidence.get("source_id"))] if evidence and clean_text(evidence.get("source_id")) else list(park.get("source_ids", []))
                    status = "evidenced" if sector_fit else "pending_verification"
                    value = "；".join(sector_fit) if sector_fit else "待向园区核验"
                    dimensions.append(
                        {
                            **definition,
                            "value": value,
                            "status": status,
                            "confidence": clean_text((evidence or {}).get("confidence")) or clean_text(park.get("confidence")) or "unknown",
                            "source_ids": source_ids,
                            "requested_sector_match": matched,
                        }
                    )
                    continue
                metric_keys = {
                    "green_transition": ("near_zero_carbon_pilot",),
                    "site_supply": ("site_supply", "land_factory", "plant_availability"),
                    "policy_basis": ("policy_basis", "incentives"),
                }[key]
                metric = next((metrics_by_key[item] for item in metric_keys if item in metrics_by_key), None)
                source_id = clean_text((metric or {}).get("source_id"))
                confidence = clean_text((metric or {}).get("confidence")) or "unknown"
                known = bool(metric and source_id and confidence.casefold() != "unknown")
                dimensions.append(
                    {
                        **definition,
                        "value": clean_text((metric or {}).get("value")) or "待向园区核验",
                        "status": "evidenced" if known else "pending_verification",
                        "confidence": confidence,
                        "source_ids": [source_id] if source_id else [],
                    }
                )
            evidenced = sum(1 for item in dimensions if item["status"] == "evidenced")
            pending = [item["label"] for item in dimensions if item["status"] != "evidenced"]
            pending_by_park[str(park["park_id"])] = pending
            matrix_rows.append(
                {
                    "park_id": park["park_id"],
                    "park_name": park["name"],
                    "district": park.get("district", ""),
                    "dimensions": dimensions,
                    "evidence_completeness_percent": round(evidenced / max(1, len(dimensions)) * 100),
                    "pending_dimensions": pending,
                }
            )
        sector_matched = [row["park_name"] for row in matrix_rows if any(item.get("requested_sector_match") is True for item in row["dimensions"])]
        negotiation_points = [
            f"先确认{project_label}的主体、产品范围、环评/能耗条件与实际用地或厂房可得性；公开试点名单不构成供给或优惠承诺。",
            "以项目投资强度、产能节奏、供应链协同、人才需求和物流方案形成双向信息清单，再进入条款沟通。",
            "要求园区按公开口径补充审批路径、配套责任边界和可核验的支持政策依据。",
        ]
        if requested_sector:
            if sector_matched:
                negotiation_points.insert(1, f"公开产业话题与“{requested_sector}”相符的候选为：{'、'.join(sector_matched)}；仍需单独核验准入与承载条件。")
            else:
                negotiation_points.insert(1, f"当前公开资料未显示“{requested_sector}”的明确园区适配，应将产业准入列为首项待核验。")
        pending_labels = sorted({label for labels in pending_by_park.values() for label in labels})
        if pending_labels:
            negotiation_points.append(f"本轮资料仍缺少：{'、'.join(pending_labels)}；这些维度不参与招商优先级判断。")
        return {
            "parks": parks,
            "benchmark_model": {
                "name": "公开证据多维对标矩阵",
                "requested_sector": requested_sector or None,
                "project_name": project_label,
                "dimensions": [{key: value for key, value in item.items() if key != "metric_key"} for item in PARK_BENCHMARK_DIMENSIONS],
                "rows": matrix_rows,
                "score_interpretation": "资料完备度只统计有来源且非待核验的对标维度，不是招商优先级、项目准入或政策支持评分。",
            },
            "negotiation_points": negotiation_points,
            "phased_plan": [
                {"phase": "0-30 天", "goal": "项目画像与合规前置核验", "actions": [f"确认{project_label}主体、产品与工艺", "核验准入和环保要求", "按对标矩阵建立候选园区问题清单"]},
                {"phase": "31-90 天", "goal": "选址与洽谈", "actions": ["园区现场踏勘", "核验地块/厂房和接入条件", "形成投资与配套责任边界"]},
                {"phase": "91-180 天", "goal": "落地准备", "actions": ["推进企业设立与项目备案", "对接人才、物流和生活配套", "建立台账与节点复盘"]},
            ],
            "caveat": "对标维度仅使用已入库公开资料；未知项保持“待核验”，资料完备度不等于适配度，不会生成土地、厂房或优惠承诺。",
        }

    def landing_package(self, *, project_name: str, district: str | None = None) -> dict[str, Any]:
        self.ensure_ready()
        requested_district = clean_text(district) or "常州市"
        project_label = clean_text(project_name) or "招商项目"
        with self._connect() as connection:
            # A district package may include citywide services, but it must not
            # silently present another district's local resource as applicable.
            # The filter is deliberately SQL, not RAG, because applicability is
            # a structured field rather than a semantic similarity decision.
            if requested_district == "常州市":
                rows = connection.execute("SELECT * FROM landing_services ORDER BY category, service_id").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM landing_services WHERE district IN (?, '常州市') ORDER BY category, service_id",
                    (requested_district,),
                ).fetchall()
        services: dict[str, list[dict[str, Any]]] = {"政务办事": [], "生活配套": [], "交通区位": []}
        citywide_count = 0
        local_count = 0
        for row in rows:
            item = dict(row)
            item["source_ids"] = json_list(item.pop("source_ids"))
            item["tags"] = json_list(item.pop("tags"))
            item["sources"] = self.source_cards(item["source_ids"])
            item_district = clean_text(item.get("district")) or "常州市"
            if item_district == "常州市" and requested_district != "常州市":
                citywide_count += 1
                item["scope"] = f"常州市全域通用；落地到{requested_district}前需向属地窗口核验"
            else:
                local_count += 1
                item["scope"] = f"适用区域：{item_district}"
            services.setdefault(str(item["category"]), []).append(item)
        project_lower = project_label.casefold()
        topic_hits: list[dict[str, Any]] = []
        for track, keywords, workstreams in PROJECT_TOPIC_RULES:
            hits = [keyword for keyword in keywords if keyword.casefold() in project_lower]
            if hits:
                topic_hits.append({"track": track, "matched_keywords": hits, "workstreams": list(workstreams)})
        workstreams = list(dict.fromkeys(stream for item in topic_hits for stream in item["workstreams"]))
        if not workstreams:
            workstreams = ["企业开办", "人才服务", "物流组织"]
        tag_priorities = {
            "企业开办": {"企业开办", "政务服务", "登记", "帮办代办", "行政审批"},
            "项目备案与准入": {"政务服务", "行政审批", "登记"},
            "项目准入": {"政务服务", "行政审批", "登记"},
            "能耗与环评": {"政务服务", "行政审批"},
            "环保与安全条件": {"政务服务", "行政审批"},
            "人才服务": {"人才公寓", "住房", "人才服务", "教育", "医疗"},
            "物流组织": {"航空货运", "物流", "交通"},
        }
        for category_items in services.values():
            for item in category_items:
                tags = set(item.get("tags", []))
                matched_workstreams = [
                    workstream
                    for workstream in workstreams
                    if tags & tag_priorities.get(workstream, set())
                ]
                item["project_relevance"] = {
                    "priority": "高" if matched_workstreams else "常规",
                    "matched_workstreams": matched_workstreams,
                    "reason": "项目名称画像与服务标签匹配" if matched_workstreams else "作为项目通用落地核验资料保留",
                }
            category_items.sort(key=lambda item: (item["project_relevance"]["priority"] != "高", item["name"]))
        checklist_definitions = (
            ("企业开办", "政务办事", "0-30 天", "企业/属地政务服务窗口", "核验企业设立、主题服务入口及材料清单。"),
            ("项目备案与准入", "政务办事", "0-30 天", "企业/属地主管部门", "核验项目备案、准入边界及办理条件。"),
            ("项目准入", "政务办事", "0-30 天", "企业/属地主管部门", "核验行业准入、许可和项目申报边界。"),
            ("能耗与环评", "政务办事", "0-30 天", "企业/属地主管部门", "核验环评、能耗和安全条件；本系统不作审批结论。"),
            ("环保与安全条件", "政务办事", "0-30 天", "企业/属地主管部门", "核验环保、安全及配套责任边界；本系统不作审批结论。"),
            ("人才服务", "生活配套", "31-90 天", "企业/属地人才服务窗口", "核验人才居住、教育、医疗等实际资格、房源和服务条件。"),
            ("物流组织", "交通区位", "31-90 天", "企业/物流服务商", "按货物属性核验运输方式、时效、价格与承运限制。"),
        )
        action_checklist: list[dict[str, Any]] = []
        for workstream in workstreams:
            definition = next((item for item in checklist_definitions if item[0] == workstream), None)
            if not definition:
                continue
            _, category, phase, owner, action = definition
            supporting = [
                item for item in services.get(category, [])
                if workstream in item.get("project_relevance", {}).get("matched_workstreams", [])
            ]
            action_checklist.append(
                {
                    "workstream": workstream,
                    "category": category,
                    "priority": "高",
                    "phase": phase,
                    "suggested_owner": owner,
                    "action": action,
                    "supporting_service_ids": [item["service_id"] for item in supporting],
                    "evidence_status": "有公开入口资料" if supporting else "待向属地补充公开或窗口资料",
                }
            )
        localization_note = (
            f"已按“{requested_district} + 常州市全域通用资料”筛选：{local_count} 条属地资料、"
            f"其中 {citywide_count} 条为市级通用资料。"
            if requested_district != "常州市"
            else "已按常州市全域范围组织资料；如需区县窗口、房源或具体交通节点，请在下一轮向属地主管部门核验。"
        )
        return {
            "project_name": project_label,
            "district": requested_district,
            "sections": [{"category": key, "items": value} for key, value in services.items()],
            "localization": {
                "requested_district": requested_district,
                "local_item_count": local_count,
                "citywide_item_count": citywide_count,
                "filter": "district = requested district OR 常州市",
            },
            "localization_note": localization_note,
            "execution": {
                "engine": "parameterized SQLite landing-service query",
                "rag_called": False,
                "method": "区县精确过滤 + 项目画像关键词与服务标签排序",
            },
            "customization": {
                "method": "项目名称关键词画像 + 服务标签排序；不使用 RAG，不将关键词推断当作审批或政策事实。",
                "topic_hits": topic_hits,
                "workstreams": workstreams,
            },
            "action_checklist": action_checklist,
            "note": "配套包用于下一轮核验与责任分工；项目资格、可用资源和办理条件应以属地主管部门的最新答复为准。",
        }

    # -- ledger, intake and RBAC ----------------------------------------

    def authenticate(self, username: str, password: str) -> dict[str, str] | None:
        self.ensure_ready()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return None
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), str(row["password_salt"]).encode(), 160_000).hex()
        if not hmac.compare_digest(hashed, str(row["password_hash"])):
            return None
        return {"username": str(row["username"]), "display_name": str(row["display_name"]), "role": str(row["role"])}

    def user_identity(self, username: str) -> dict[str, str] | None:
        """Return the current authorized identity for a signed local session."""

        self.ensure_ready()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT username, display_name, role FROM users WHERE username = ?",
                (clean_text(username),),
            ).fetchone()
        if not row or str(row["role"]) not in {"admin", "officer"}:
            return None
        return {"username": str(row["username"]), "display_name": str(row["display_name"]), "role": str(row["role"])}

    def add_ledger(
        self,
        *,
        company_id: str,
        project_name: str,
        stage: str,
        owner: str,
        next_step: str,
        contact_name: str = "演示联系人",
        contact_phone: str = "13800138123",
        contact_email: str = "demo.contact@example.test",
        actor: str,
        role: str,
    ) -> dict[str, Any]:
        self.ensure_ready()
        project_name, _ = redact_intake_text(project_name)
        stage, _ = redact_intake_text(stage)
        owner, _ = redact_intake_text(owner)
        next_step, _ = redact_intake_text(next_step)
        now = utc_now()
        with self._connect() as connection:
            if not connection.execute("SELECT 1 FROM companies WHERE company_id = ?", (company_id,)).fetchone():
                raise KeyError("企业不存在")
            cursor = connection.execute(
                """INSERT INTO ledgers(company_id, project_name, stage, owner, next_step, contact_name, contact_phone, contact_email,
                   sensitivity, created_by, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'internal', ?, ?, ?)""",
                (company_id, project_name, stage, owner, next_step, contact_name, contact_phone, contact_email, actor, now, now),
            )
            ledger_id = int(cursor.lastrowid)
            connection.execute("INSERT INTO ledger_events(ledger_id, event_type, detail, actor, created_at) VALUES (?, ?, ?, ?, ?)", (ledger_id, "created", "创建企业对接台账", actor, now))
            connection.commit()
        return self.get_ledger(ledger_id, role=role) or {}

    def update_ledger(self, ledger_id: int, *, stage: str, owner: str, next_step: str, actor: str, role: str) -> dict[str, Any] | None:
        self.ensure_ready()
        stage, _ = redact_intake_text(stage)
        owner, _ = redact_intake_text(owner)
        next_step, _ = redact_intake_text(next_step)
        now = utc_now()
        with self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM ledgers WHERE ledger_id=?", (ledger_id,)).fetchone()
            if not exists:
                return None
            connection.execute("UPDATE ledgers SET stage=?, owner=?, next_step=?, updated_at=? WHERE ledger_id=?", (stage, owner, next_step, now, ledger_id))
            connection.execute("INSERT INTO ledger_events(ledger_id, event_type, detail, actor, created_at) VALUES (?, ?, ?, ?, ?)", (ledger_id, "updated", f"阶段更新为：{stage}", actor, now))
            connection.commit()
        return self.get_ledger(ledger_id, role=role)

    def get_ledger(self, ledger_id: int, *, role: str) -> dict[str, Any] | None:
        self.ensure_ready()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT l.*, c.name AS company_name FROM ledgers l JOIN companies c ON c.company_id=l.company_id
                WHERE l.ledger_id=?""",
                (ledger_id,),
            ).fetchone()
            events = connection.execute("SELECT * FROM ledger_events WHERE ledger_id=? ORDER BY event_id", (ledger_id,)).fetchall()
        if not row:
            return None
        item = dict(row)
        if role != "admin":
            item["contact_phone"] = self._mask_phone(str(item.get("contact_phone") or ""))
            item["contact_email"] = self._mask_email(str(item.get("contact_email") or ""))
            item["contact_name"] = self._mask_name(str(item.get("contact_name") or ""))
            # Older entries may predate intake-wide sanitization.  Redact the
            # non-contact free-text fields again at read time for the officer
            # role so a phone/email embedded in "下一步" cannot bypass RBAC.
            for field in ("project_name", "stage", "owner", "next_step"):
                item[field], _ = redact_intake_text(item.get(field))
        event_items = [dict(event) for event in events]
        if role != "admin":
            for event in event_items:
                event["detail"], _ = redact_intake_text(event.get("detail"))
                event["actor"], _ = redact_intake_text(event.get("actor"))
        item["events"] = event_items
        return item

    def list_ledgers(self, *, role: str) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as connection:
            ids = [int(row[0]) for row in connection.execute("SELECT ledger_id FROM ledgers ORDER BY updated_at DESC")]
        return [item for item in (self.get_ledger(ledger_id, role=role) for ledger_id in ids) if item]

    @staticmethod
    def _mask_phone(value: str) -> str:
        return f"{value[:3]}****{value[-4:]}" if len(value) >= 7 else "已脱敏"

    @staticmethod
    def _mask_email(value: str) -> str:
        if "@" not in value:
            return "已脱敏"
        local, domain = value.split("@", 1)
        return f"{local[:1]}***@{domain}"

    @staticmethod
    def _mask_name(value: str) -> str:
        return f"{value[:1]}*" if len(value) > 1 else ("*" if value else "")

    def preflight_import(self, content: bytes, *, filename: str, actor: str) -> dict[str, Any]:
        self.ensure_ready()
        try:
            decoded = content.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(decoded))
            if not reader.fieldnames:
                raise ValueError("CSV 缺少表头")
            if len(reader.fieldnames) > MAX_IMPORT_COLUMNS:
                raise ValueError(f"CSV 表头超过 {MAX_IMPORT_COLUMNS} 列演示版导入上限")
            rows: list[dict[str, str]] = []
            for row in reader:
                if len(rows) >= MAX_IMPORT_ROWS:
                    raise ValueError(f"CSV 超过 {MAX_IMPORT_ROWS} 行演示版导入上限")
                rows.append(row)
        except (UnicodeDecodeError, csv.Error) as exc:
            raise ValueError(f"无法读取 UTF-8 CSV：{exc}") from exc
        if not rows:
            raise ValueError("CSV 没有数据行")
        with self._connect() as connection:
            known_sources = {str(row[0]) for row in connection.execute("SELECT source_id FROM sources")}
            existing_identities = {
                intake_identity(row["name"], source_id)
                for row in connection.execute("SELECT name, source_ids FROM companies WHERE status='published'").fetchall()
                for source_id in json_list(row["source_ids"])
            }
            batch_identities: set[str] = set()
            import_id = uuid.uuid4().hex
            connection.execute("INSERT INTO import_jobs VALUES (?, ?, 'staged', ?, ?, NULL)", (import_id, actor, filename, utc_now()))
            report_rows = []
            for index, row in enumerate(rows, start=2):
                normalized = self._normalize_import_row(row)
                errors = []
                oversized_fields = [
                    str(key or "额外列")
                    for key, value in row.items()
                    if key is None or len(clean_text(value)) > MAX_IMPORT_FIELD_CHARS
                ]
                if oversized_fields:
                    errors.append(f"字段超过 {MAX_IMPORT_FIELD_CHARS} 字符上限：{'、'.join(oversized_fields[:4])}")
                if not normalized["company_name"]:
                    errors.append("缺少 company_name")
                if not normalized["source_id"]:
                    errors.append("缺少 source_id")
                elif normalized["source_id"] not in known_sources:
                    errors.append(f"未知 source_id: {normalized['source_id']}")
                protected_identity_fields = {
                    str(item.get("field") or "")
                    for item in normalized.get("desensitization", {}).get("redacted_fields", [])
                }
                if protected_identity_fields & {"company_name", "source_id"}:
                    errors.append("企业名称和来源 ID 不得包含个人敏感信息")
                if normalized["confidence"] not in {"high", "medium", "low"}:
                    errors.append("confidence 必须为 high、medium 或 low")
                identity = intake_identity(normalized["company_name"], normalized["source_id"])
                if identity and identity in existing_identities:
                    errors.append("企业名称与来源 ID 已存在于已发布资料，请核验是否重复导入")
                elif identity and identity in batch_identities:
                    errors.append("CSV 内存在相同企业名称与来源 ID 的重复行")
                if identity:
                    batch_identities.add(identity)
                valid = not errors
                connection.execute(
                    "INSERT INTO import_rows VALUES (?, ?, ?, ?, ?, NULL)",
                    (import_id, index, json_text(normalized), int(valid), json_text(errors)),
                )
                report_rows.append({"row_number": index, "valid": valid, "errors": errors, "normalized": normalized})
            connection.commit()
        return {"import_id": import_id, "filename": filename, "rows": report_rows, "valid_count": sum(row["valid"] for row in report_rows), "invalid_count": sum(not row["valid"] for row in report_rows)}

    def publish_import(self, import_id: str, *, actor: str) -> dict[str, Any]:
        self.ensure_ready()
        with self._connect() as connection:
            job = connection.execute("SELECT * FROM import_jobs WHERE import_id=?", (import_id,)).fetchone()
            if not job:
                raise KeyError("找不到导入任务")
            rows = connection.execute("SELECT * FROM import_rows WHERE import_id=? AND valid=1 ORDER BY row_number", (import_id,)).fetchall()
            source_by_id = {row["source_id"]: dict(row) for row in connection.execute("SELECT * FROM sources")}
            published: list[str] = []
            for row in rows:
                if row["published_company_id"]:
                    published.append(str(row["published_company_id"]))
                    continue
                payload = json.loads(str(row["payload"]))
                payload["company_id"] = f"MAN-{uuid.uuid4().hex[:8].upper()}"
                payload["verified_date"] = datetime.now().date().isoformat()
                company_id = self._insert_company(connection, payload, source_by_id=source_by_id, status="published")
                connection.execute("UPDATE import_rows SET published_company_id=? WHERE import_id=? AND row_number=?", (company_id, import_id, row["row_number"]))
                published.append(company_id)
            connection.execute("UPDATE import_jobs SET status='published', reviewed_at=? WHERE import_id=?", (utc_now(), import_id))
            self._rebuild_graph(connection)
            connection.commit()
        return {"import_id": import_id, "published_company_ids": published, "status": "published"}

    def add_manual_company(self, payload: Mapping[str, Any], *, actor: str) -> dict[str, Any]:
        self.ensure_ready()
        normalized = self._normalize_import_row(payload)
        if not normalized["company_name"] or not normalized["source_id"]:
            raise ValueError("企业名称和来源 ID 为必填项")
        protected_identity_fields = {
            str(item.get("field") or "")
            for item in normalized.get("desensitization", {}).get("redacted_fields", [])
        }
        if protected_identity_fields & {"company_name", "source_id"}:
            raise ValueError("企业名称和来源 ID 不得包含个人敏感信息")
        with self._connect() as connection:
            known = connection.execute("SELECT 1 FROM sources WHERE source_id=?", (normalized["source_id"],)).fetchone()
            if not known:
                raise ValueError("来源 ID 不存在")
            identity = intake_identity(normalized["company_name"], normalized["source_id"])
            existing = connection.execute("SELECT name, source_ids FROM companies WHERE status='published'").fetchall()
            if identity and any(identity == intake_identity(row["name"], source_id) for row in existing for source_id in json_list(row["source_ids"])):
                raise ValueError("企业名称与来源 ID 已存在于已发布资料，请核验是否重复录入")
            normalized["company_id"] = f"MAN-{uuid.uuid4().hex[:8].upper()}"
            normalized["verified_date"] = datetime.now().date().isoformat()
            company_id = self._insert_company(connection, normalized, status="published")
            self._rebuild_graph(connection)
            connection.commit()
        return self.get_company(company_id) or {}

    @staticmethod
    def _normalize_import_row(row: Mapping[str, Any]) -> dict[str, Any]:
        redacted_fields: list[dict[str, Any]] = []

        def pick(field: str, *keys: str) -> str:
            for key in keys:
                value = row.get(key)
                if value is not None and clean_text(value):
                    cleaned, redactions = redact_intake_text(clean_text(value)[:MAX_IMPORT_FIELD_CHARS])
                    if redactions:
                        redacted_fields.append({"field": field, "types": redactions})
                    return cleaned
            return ""
        return {
            "company_name": pick("company_name", "company_name", "name", "企业名称"),
            "aliases": pick("aliases", "aliases", "别名"),
            "district": pick("district", "district", "区县", "区域") or "常州市",
            "park": pick("park", "park", "园区"),
            "industry_track": pick("industry_track", "industry_track", "sector", "产业赛道"),
            "industry_subtrack": pick("industry_subtrack", "industry_subtrack", "industry", "细分行业"),
            "supply_chain_role": pick("supply_chain_role", "supply_chain_role", "产业链角色"),
            "products": pick("products", "products", "产品"),
            "capabilities": pick("capabilities", "capabilities", "能力"),
            "input_materials": pick("input_materials", "input_materials", "输入材料"),
            "output_products": pick("output_products", "output_products", "输出产品"),
            "target_customer_industries": pick("target_customer_industries", "target_customer_industries", "目标客户行业"),
            "summary": pick("summary", "summary", "简介", "description"),
            "source_id": pick("source_id", "source_id", "来源ID"),
            "confidence": pick("confidence", "confidence", "可信度") or "medium",
            "notes": pick("notes", "notes", "备注"),
            "desensitization": {
                "classification": "internal" if redacted_fields else "public",
                "redacted_fields": redacted_fields,
            },
        }

    def record_export(self, *, export_type: str, fmt: str, file_path: str, actor: str, record_ids: Iterable[str]) -> str:
        self.ensure_ready()
        export_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("INSERT INTO exports VALUES (?, ?, ?, ?, ?, ?, ?)", (export_id, export_type, fmt, file_path, actor, utc_now(), json_text(list(record_ids))))
            connection.commit()
        return export_id

    # -- row adapters ----------------------------------------------------

    def _company_from_row(self, row: sqlite3.Row | None, *, connection: sqlite3.Connection | None = None, include_sources: bool = True) -> dict[str, Any]:
        if row is None:
            return {}
        item = dict(row)
        for key in ("aliases", "supply_chain_role", "products", "capabilities", "input_materials", "output_products", "target_customer_industries", "source_ids"):
            item[key] = json_list(item.get(key))
        item["company_id"] = str(item["company_id"])
        item["name"] = item.pop("name")
        if include_sources:
            item["sources"] = self._source_cards_in_connection(connection, item["source_ids"]) if connection else self.source_cards(item["source_ids"])
        return item

    def _park_from_row(self, row: sqlite3.Row, connection: sqlite3.Connection) -> dict[str, Any]:
        item = dict(row)
        item["source_ids"] = json_list(item.get("source_ids"))
        item["sector_fit"] = json_list(item.get("sector_fit"))
        metrics = connection.execute("SELECT metric_key, label, value, source_id, confidence FROM park_metrics WHERE park_id=? ORDER BY metric_id", (item["park_id"],)).fetchall()
        item["metrics"] = [dict(metric) for metric in metrics]
        item["sources"] = self._source_cards_in_connection(connection, item["source_ids"])
        for metric in item["metrics"]:
            metric["source"] = self._source_cards_in_connection(connection, [metric.get("source_id", "")])
        return item

    @staticmethod
    def _source_cards_in_connection(connection: sqlite3.Connection | None, source_ids: Iterable[str]) -> list[dict[str, str]]:
        if connection is None:
            return []
        ids = [clean_text(item) for item in source_ids if clean_text(item)]
        if not ids:
            return []
        rows = connection.execute(f"SELECT * FROM sources WHERE source_id IN ({','.join('?' for _ in ids)})", ids).fetchall()
        by_id = {str(row["source_id"]): dict(row) for row in rows}
        return [by_id[source_id] for source_id in ids if source_id in by_id]
