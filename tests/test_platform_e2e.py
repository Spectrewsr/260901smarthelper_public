"""Ten acceptance scenarios for the local investment knowledge platform.

The suite uses an isolated copy of the raw data.  It nevertheless builds a
real BGE-M3 index and loads the locally downloaded BGE cross-encoder, so the
Advanced-RAG assertion is an end-to-end check rather than a mock.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from scripts.build_index import build_index


class PlatformAcceptanceTests(unittest.TestCase):
    """Exactly ten user-facing acceptance examples from the implementation spec."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory(prefix="cz-platform-e2e-")
        cls.temporary_root = Path(cls._temporary.name)
        cls.data_dir = cls.temporary_root / "data"
        shutil.copytree(DEMO_ROOT / "data" / "raw", cls.data_dir / "raw")

        cls._environment = {key: os.environ.get(key) for key in ("APP_DATA_DIR", "BGE_M3_MODEL_PATH", "RERANK_MODEL_PATH", "RERANK_DEVICE", "TOKENIZERS_PARALLELISM")}
        os.environ["APP_DATA_DIR"] = str(cls.data_dir)
        os.environ["BGE_M3_MODEL_PATH"] = str(DEMO_ROOT / "models" / "bge-m3")
        os.environ["RERANK_MODEL_PATH"] = str(DEMO_ROOT / "models" / "bge-reranker-v2-m3")
        os.environ["RERANK_DEVICE"] = "cuda"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        build_index(
            cls.data_dir,
            embedding_backend="bge-m3",
            model_path=DEMO_ROOT / "models" / "bge-m3",
            device="cuda",
            batch_size=2,
            max_length=768,
            min_companies=100,
        )

        import app.main as main
        from fastapi.testclient import TestClient

        cls.main = importlib.reload(main)
        cls.client = TestClient(cls.main.app)
        cls.client.__enter__()
        cls.admin_headers = cls._headers_for("admin", "admin-demo-2026")
        cls.officer_headers = cls._headers_for("officer", "officer-demo-2026")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        for key, value in cls._environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        cls._temporary.cleanup()

    @classmethod
    def _headers_for(cls, username: str, password: str) -> dict[str, str]:
        response = cls.client.post("/api/auth/login", json={"username": username, "password": password})
        if response.status_code != 200:
            raise AssertionError(response.text)
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    def _query(self, payload: dict) -> dict:
        response = self.client.post("/api/query", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_01_two_roles_and_sensitive_ledger_masking(self) -> None:
        """Admin can archive a lead; officer sees only desensitized contacts."""
        no_auth = self.client.get("/api/ledgers")
        self.assertEqual(no_auth.status_code, 401)

        # A token signed by this process but naming a user absent from SQLite
        # must still be rejected; roles are not trusted from bearer claims.
        from app.services.auth import DemoUser

        forged = self.main.app.state.auth._sign(DemoUser("ghost-admin", "伪造管理员", "admin"))
        forged_response = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {forged}"})
        self.assertEqual(forged_response.status_code, 401)

        created = self.client.post(
            "/api/ledgers",
            headers=self.admin_headers,
            json={
                "company_id": "C004",
                "project_name": "新能源电池项目",
                "stage": "待联系",
                "owner": "招商管理员",
                "next_step": "核验产能与商务联系人 13900001111",
                "contact_name": "王工",
                "contact_phone": "13900001111",
                "contact_email": "wang@example.test",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        ledger_id = created.json()["ledger_id"]
        admin_detail = self.client.get(f"/api/ledgers/{ledger_id}", headers=self.admin_headers).json()
        officer_detail = self.client.get(f"/api/ledgers/{ledger_id}", headers=self.officer_headers).json()
        self.assertEqual(admin_detail["contact_phone"], "13900001111")
        self.assertEqual(officer_detail["contact_phone"], "139****1111")
        self.assertEqual(officer_detail["contact_email"], "w***@example.test")
        self.assertEqual(officer_detail["contact_name"], "王*")
        self.assertIn("[已脱敏手机号]", officer_detail["next_step"])
        self.assertNotIn("13900001111", officer_detail["next_step"])

        officer_created = self.client.post(
            "/api/ledgers",
            headers=self.officer_headers,
            json={
                "company_id": "C006",
                "project_name": "专员创建脱敏验收",
                "stage": "待联系",
                "owner": "招商专员",
                "next_step": "核验公开资料",
                "contact_name": "李工",
                "contact_phone": "13800138123",
                "contact_email": "li@example.test",
            },
        )
        self.assertEqual(officer_created.status_code, 200, officer_created.text)
        self.assertEqual(officer_created.json()["contact_phone"], "138****8123")
        self.assertEqual(officer_created.json()["contact_email"], "l***@example.test")
        missing = self.client.patch(
            "/api/ledgers/999999",
            headers=self.officer_headers,
            json={"stage": "待联系", "owner": "招商专员", "next_step": "不存在的台账不应被创建"},
        )
        self.assertEqual(missing.status_code, 404)

    def test_02_hybrid_contextual_rag_uses_real_cross_encoder(self) -> None:
        """Contextual + FTS/BM25 + BGE-M3 + RRF + BGE reranker all leave a trace."""
        result = self._query(
            {
                "query": "我想在常州布局新能源汽车项目，优先应联系哪些类型的企业？",
                "sector": "新能源汽车与新能源",
                "mode": "hybrid",
            }
        )
        trace = result["retrieval_trace"]
        self.assertTrue(result["execution"]["rag_called"])
        self.assertIn("parent metadata prefix", trace["contextual"]["strategy"])
        self.assertEqual(trace["lexical"]["engine"], "SQLite FTS5 / BM25")
        self.assertEqual(trace["dense"]["engine"], "BAAI/bge-m3")
        self.assertFalse(trace["dense"]["fallback"])
        self.assertEqual(trace["fusion"]["algorithm"], "reciprocal rank fusion")
        self.assertEqual(trace["reranker"]["engine"], "BAAI/bge-reranker-v2-m3")
        self.assertTrue(trace["reranker"]["model_loaded"])
        self.assertFalse(trace["reranker"]["fallback"])
        self.assertTrue(result["sources"])
        self.assertTrue(result["matches"]["all_matches"])

    def test_03_lightweight_agent_joins_three_knowledge_bases(self) -> None:
        """A project question uses the bounded three-tool execution plan."""
        result = self._query(
            {
                "query": "在常州独资布局新能源项目，怎样选园区并准备落地？",
                "district": "金坛区",
                "sector": "新能源汽车与新能源",
                "mode": "project",
                "park_ids": ["P001", "P002", "P003"],
                "project_name": "新能源项目",
            }
        )
        self.assertEqual(result["execution"]["mode"], "agentic_project")
        self.assertEqual([item["tool"] for item in result["execution"]["tool_calls"]], ["hybrid_rag", "park_compare", "landing_package"])
        self.assertTrue(result["execution"]["rag_called"])
        self.assertEqual(len(result["park_comparison"]["parks"]), 3)
        categories = {item["category"] for item in result["landing_package"]["sections"]}
        self.assertEqual(categories, {"政务办事", "生活配套", "交通区位"})

    def test_04_exact_structured_sql_bypasses_rag(self) -> None:
        """Exact district/track query has a stable SQL result set and no RAG trace."""
        result = self._query(
            {
                "query": "列出金坛区新能源汽车与新能源企业",
                "district": "金坛区",
                "sector": "新能源汽车与新能源",
                "mode": "sql",
            }
        )
        self.assertEqual(result["execution"]["mode"], "sql")
        self.assertFalse(result["execution"]["rag_called"])
        self.assertNotIn("retrieval_trace", result)
        self.assertEqual([item["company_id"] for item in result["companies"]], ["C004", "C006", "C018", "C037", "C044", "C046"])

    def test_05_geo_sql_bypasses_rag_and_marks_reference_precision(self) -> None:
        """Distance query uses the local spatial index, not semantic retrieval."""
        result = self._query(
            {
                "query": "金坛区附近 20 公里的新能源企业",
                "district": "金坛区",
                "radius_km": 20,
                "mode": "geo",
            }
        )
        self.assertFalse(result["execution"]["rag_called"])
        self.assertEqual(result["execution"]["mode"], "sql_geo")
        self.assertIn("RTree + Haversine", result["execution"]["method"])
        distances = [item["distance_km"] for item in result["companies"]]
        self.assertEqual(distances, sorted(distances))
        self.assertTrue(result["companies"])
        self.assertTrue(all(item["coordinate_precision"] == "district_reference" for item in result["companies"]))
        self.assertTrue(all("演示参考点" in item["coordinate_source"] for item in result["companies"]))

    def test_06_local_graphrag_expands_a_labelled_two_hop_chain(self) -> None:
        """Graph expansion shows inferred/potential edges with source IDs rather than claiming transactions."""
        result = self._query({"query": "蜂巢能源的上游材料、动力电池和整车关联路径", "mode": "graph"})
        self.assertEqual(result["execution"]["mode"], "graph_rag")
        node_ids = {node["node_id"] for node in result["graph"]["nodes"]}
        self.assertTrue({"company:C004", "company:C006", "company:C046", "company:C008"}.issubset(node_ids))
        edges = result["graph"]["edges"]
        self.assertTrue(any(edge["relation"] == "potential_material_supply" and edge["to_node"] == "company:C004" for edge in edges))
        self.assertTrue(all(edge["status"] in {"inferred", "potential", "source_fact"} for edge in edges))
        self.assertTrue(all(edge["source_ids"] for edge in edges))

    def test_07_park_comparison_supports_asset_aliases_and_phased_plan(self) -> None:
        """The park view provides comparison facts, negotiation prompts and a staged plan."""
        response = self.client.get("/api/parks/compare?park_ids=A002,A003,A005")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual([item["park_id"] for item in result["parks"]], ["P003", "P001", "P002"])
        self.assertTrue(all(item["sources"] and item["metrics"] for item in result["parks"]))
        self.assertEqual([item["phase"] for item in result["phased_plan"]], ["0-30 天", "31-90 天", "91-180 天"])
        self.assertIn("待核验", result["caveat"])

    def test_08_landing_package_has_government_life_and_transport_evidence(self) -> None:
        """Landing package keeps all three local-service categories and their public sources."""
        response = self.client.get("/api/landing/package?project_name=%E6%96%B0%E8%83%BD%E6%BA%90%E9%A1%B9%E7%9B%AE&district=%E9%87%91%E5%9D%9B%E5%8C%BA")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        sections = {section["category"]: section["items"] for section in result["sections"]}
        self.assertEqual(set(sections), {"政务办事", "生活配套", "交通区位"})
        for items in sections.values():
            self.assertTrue(items)
            self.assertTrue(all(item["sources"] and item["verified_date"] for item in items))
            self.assertTrue(all(item["district"] in {"常州市", "金坛区"} for item in items))
            self.assertTrue(all(item.get("scope") for item in items))
        self.assertEqual(result["localization"]["requested_district"], "金坛区")
        self.assertEqual(result["localization"]["filter"], "district = requested district OR 常州市")
        self.assertIn("金坛区", result["localization_note"])

    def test_09_templates_docx_pptx_and_target_list_are_real_files(self) -> None:
        """Three preset material templates, DOCX/PPTX and a standard target list are downloadable."""
        templates = self.client.get("/api/export-templates")
        self.assertEqual(templates.status_code, 200)
        self.assertEqual([item["key"] for item in templates.json()["items"]], ["chain", "park", "landing"])

        csv_response = self.client.get("/api/exports/target-list?company_ids=C004,C006", headers=self.admin_headers)
        self.assertEqual(csv_response.status_code, 200, csv_response.text)
        self.assertIn("企业名称".encode("utf-8"), csv_response.content)
        self.assertIn("C004".encode("utf-8"), csv_response.content)

        docx_response = self.client.post(
            "/api/exports/material",
            headers=self.admin_headers,
            json={"company_ids": ["C004", "C006"], "project_name": "新能源项目", "format": "docx", "template_key": "park"},
        )
        self.assertEqual(docx_response.status_code, 200, docx_response.text)
        self.assertTrue(docx_response.content.startswith(b"PK"))
        with zipfile.ZipFile(__import__("io").BytesIO(docx_response.content)) as archive:
            document_xml = archive.read("word/document.xml").decode("utf-8")
        self.assertIn("蜂巢", document_xml)
        self.assertIn("独资设立园区洽谈", document_xml)

        pptx_response = self.client.post(
            "/api/exports/material",
            headers=self.admin_headers,
            json={"company_ids": ["C004", "C006"], "project_name": "新能源项目", "format": "pptx", "template_key": "landing"},
        )
        self.assertEqual(pptx_response.status_code, 200, pptx_response.text)
        self.assertTrue(pptx_response.content.startswith(b"PK"))
        with zipfile.ZipFile(__import__("io").BytesIO(pptx_response.content)) as archive:
            texts = "".join(archive.read(name).decode("utf-8") for name in archive.namelist() if name.startswith("ppt/slides/slide"))
        self.assertIn("落地配套协同", texts)

    def test_10_csv_preflight_review_publish_and_manual_intake(self) -> None:
        """Admin-only intake validates/cleans data, updates dense retrieval, and preserves operations on raw refresh."""
        content = (
            "company_name,source_id,district,industry_track,products,confidence,summary\n"
            "验收示例材料企业,S001,金坛区,新能源汽车与新能源,动态稠密索引验收材料/13900001111,medium,联系人 13900001111；test@example.test；证件 320311199001011234\n"
            "无来源企业,UNKNOWN,金坛区,新能源汽车与新能源,材料,high,应被预检拦截\n"
        ).encode("utf-8")
        preview = self.client.post(
            "/api/admin/imports/preview",
            headers=self.admin_headers,
            files={"file": ("acceptance.csv", content, "text/csv")},
        )
        self.assertEqual(preview.status_code, 200, preview.text)
        report = preview.json()
        self.assertEqual((report["valid_count"], report["invalid_count"]), (1, 1))
        self.assertIn("未知 source_id", report["rows"][1]["errors"][0])
        sanitized = report["rows"][0]["normalized"]
        self.assertIn("[已脱敏手机号]", sanitized["summary"])
        self.assertIn("[已脱敏邮箱]", sanitized["summary"])
        self.assertIn("[已脱敏身份证号]", sanitized["summary"])
        self.assertNotIn("13900001111", sanitized["summary"])
        self.assertIn("[已脱敏手机号]", sanitized["products"])
        self.assertNotIn("13900001111", json.dumps(sanitized, ensure_ascii=False))
        self.assertEqual(sanitized["desensitization"]["classification"], "internal")

        published = self.client.post(f"/api/admin/imports/{report['import_id']}/publish", headers=self.admin_headers)
        self.assertEqual(published.status_code, 200, published.text)
        self.assertEqual(len(published.json()["published_company_ids"]), 1)

        manual = self.client.post(
            "/api/admin/companies",
            headers=self.admin_headers,
            json={
                "company_name": "人工录入验收企业",
                "source_id": "S001",
                "district": "新北区",
                "industry_track": "智能制造与高端装备",
                "industry_subtrack": "工业软件",
                "supply_chain_role": "潜在合作方",
                "products": "动态稠密索引验收材料/13800138123",
                "capabilities": "数据治理",
                "summary": "人工录入的可追溯演示资料。",
                "confidence": "low",
            },
        )
        self.assertEqual(manual.status_code, 200, manual.text)
        manual_company_id = manual.json()["company_id"]
        self.assertTrue(manual_company_id.startswith("MAN-"))
        self.assertTrue(any("[已脱敏手机号]" in product for product in manual.json()["products"]))
        self.assertNotIn("13800138123", json.dumps(manual.json(), ensure_ascii=False))
        prohibited_name = self.client.post(
            "/api/admin/companies",
            headers=self.admin_headers,
            json={"company_name": "企业联系人13900001111", "source_id": "S001"},
        )
        self.assertEqual(prohibited_name.status_code, 400)

        dense_result = self._query({"query": "动态稠密索引验收材料", "mode": "hybrid"})
        dense_delta = dense_result["retrieval_trace"]["dense"]["dynamic_extension"]
        self.assertGreaterEqual(dense_delta["company_count"], 2)
        self.assertGreaterEqual(dense_delta["embedded_count"], 2)
        self.assertFalse(dense_delta["error"])

        # Exercise the labelled fallback path after the real-BGE assertions in
        # test_02: reviewed-record delta vectors must be rebuilt in the 768-D
        # hash space and the trace must never call it BGE-M3.
        live_dense = self.main.app.state.agent.rag._dense
        self.assertIsNotNone(live_dense)
        live_dense._activate_runtime_hash("acceptance fallback simulation")
        fallback_result = self._query({"query": "动态稠密索引验收材料", "mode": "hybrid"})
        fallback_trace = fallback_result["retrieval_trace"]["dense"]
        self.assertEqual(fallback_trace["engine"], "deterministic hash fallback")
        self.assertTrue(fallback_trace["fallback"])
        self.assertEqual(fallback_trace["dynamic_extension"]["embedding_backend"], "hash")
        self.assertEqual(fallback_trace["dynamic_extension"]["vector_dimension"], 768)
        self.assertFalse(fallback_trace["dynamic_extension"]["error"])

        formula_company = self.client.post(
            "/api/admin/companies",
            headers=self.admin_headers,
            json={
                "company_name": "=CSV公式保护验收企业",
                "source_id": "S001",
                "district": "金坛区",
                "industry_track": "新能源汽车与新能源",
                "summary": "用于验证导出单元格安全处理。",
            },
        )
        self.assertEqual(formula_company.status_code, 200, formula_company.text)
        formula_company_id = formula_company.json()["company_id"]
        protected_csv = self.client.get(
            f"/api/exports/target-list?company_ids={formula_company_id}", headers=self.admin_headers
        )
        self.assertEqual(protected_csv.status_code, 200, protected_csv.text)
        self.assertIn("'=CSV公式保护验收企业", protected_csv.content.decode("utf-8-sig"))

        persistent_ledger = self.client.post(
            "/api/ledgers",
            headers=self.admin_headers,
            json={
                "company_id": manual_company_id,
                "project_name": "原始资料刷新留存验收",
                "stage": "跟进中",
                "owner": "招商管理员",
                "next_step": "确认刷新后仍可读取",
            },
        )
        self.assertEqual(persistent_ledger.status_code, 200, persistent_ledger.text)
        ledger_id = persistent_ledger.json()["ledger_id"]

        # Altering only raw-file whitespace simulates a harmless source bundle
        # refresh.  The fresh SQLite evidence cache must retain user-created
        # enterprise/ledger/import/export records.
        landing_path = self.data_dir / "raw" / "landing_support.jsonl"
        landing_path.write_text(landing_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        from app.services.knowledge import KnowledgeRepository

        refreshed = KnowledgeRepository(self.data_dir)
        refreshed.ensure_ready()
        self.assertTrue(refreshed.get_company(manual_company_id))
        self.assertTrue(refreshed.get_ledger(ledger_id, role="admin"))
        with refreshed._connect() as connection:
            self.assertGreaterEqual(int(connection.execute("SELECT COUNT(*) FROM import_jobs").fetchone()[0]), 1)
            self.assertGreaterEqual(int(connection.execute("SELECT COUNT(*) FROM exports").fetchone()[0]), 1)

        # A second repository instance shares the path-scoped lock.  While it
        # pauses after an operational snapshot, a concurrent ledger write from
        # the app repository must wait; after replacement it lands in the new
        # database rather than disappearing between snapshot and replace.
        racing = KnowledgeRepository(self.data_dir)
        original_restore = racing._restore_operational_state
        snapshot_ready = threading.Event()
        continue_refresh = threading.Event()
        refresh_errors: list[BaseException] = []
        writer_errors: list[BaseException] = []
        writer_ledgers: list[dict] = []
        writer_done = threading.Event()

        def delayed_restore(connection, state):
            snapshot_ready.set()
            if not continue_refresh.wait(timeout=5):
                raise TimeoutError("refresh test did not receive release signal")
            return original_restore(connection, state)

        racing._restore_operational_state = delayed_restore  # type: ignore[method-assign]
        landing_path.write_text(landing_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

        def refresh_database() -> None:
            try:
                racing.ensure_ready()
            except BaseException as exc:  # Test records background failure.
                refresh_errors.append(exc)

        def write_during_refresh() -> None:
            try:
                writer_ledgers.append(self.main.app.state.repository.add_ledger(
                    company_id=manual_company_id,
                    project_name="刷新并发写入验收",
                    stage="待联系",
                    owner="招商管理员",
                    next_step="验证共享锁保护写入",
                    actor="admin",
                    role="admin",
                ))
            except BaseException as exc:  # Test records background failure.
                writer_errors.append(exc)
            finally:
                writer_done.set()

        refresh_thread = threading.Thread(target=refresh_database)
        refresh_thread.start()
        self.assertTrue(snapshot_ready.wait(timeout=5))
        writer_thread = threading.Thread(target=write_during_refresh)
        writer_thread.start()
        time.sleep(0.15)
        self.assertFalse(writer_done.is_set(), "台账写入不应越过正在刷新的共享数据库锁")
        continue_refresh.set()
        refresh_thread.join(timeout=10)
        writer_thread.join(timeout=10)
        self.assertFalse(refresh_thread.is_alive())
        self.assertFalse(writer_thread.is_alive())
        self.assertFalse(refresh_errors, refresh_errors)
        self.assertFalse(writer_errors, writer_errors)
        self.assertEqual(len(writer_ledgers), 1)
        self.assertEqual(racing.get_ledger(writer_ledgers[0]["ledger_id"], role="admin")["project_name"], "刷新并发写入验收")
        denied = self.client.post("/api/admin/companies", headers=self.officer_headers, json={"company_name": "不应写入", "source_id": "S001"})
        self.assertEqual(denied.status_code, 403)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
