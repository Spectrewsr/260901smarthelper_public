"""Focused regression tests for the local, citation-first data pipeline."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from scripts.build_index import build_index
from scripts.retriever import LocalRetriever
from scripts.validate_data import validate_data


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def write_fixture(root: Path, *, broken_source_ref: bool = False) -> Path:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    companies = [
        {
            "company_id": "C001",
            "company_name": "常州示例正极材料有限公司",
            "aliases": "示例正极",
            "english_name": "Example Cathode Materials",
            "district": "金坛区",
            "park": "示例园区",
            "address": "公开地址",
            "latitude": "31.72",
            "longitude": "119.58",
            "industry_track": "新能源汽车与新能源",
            "industry_subtrack": "电池材料",
            "supply_chain_role": "上游材料供应商",
            "products": "锂电池正极材料,磷酸铁锂",
            "capabilities": "材料研发,正极材料生产",
            "input_materials": "锂盐,铁源",
            "output_products": "磷酸铁锂正极材料",
            "target_customer_industries": "动力电池,储能",
            "summary": "公开资料显示企业从事锂电池正极材料研发与生产。",
            "source_id": "UNKNOWN" if broken_source_ref else "S001",
            "source_url": "https://example.test/s001",
            "source_date": "2025-01-01",
            "verified_date": "2026-09-02",
            "confidence": "high",
            "notes": "",
        },
        {
            "company_id": "C002",
            "company_name": "常州示例电池装备有限公司",
            "aliases": "示例装备",
            "english_name": "",
            "district": "新北区",
            "park": "示例园区",
            "address": "公开地址",
            "latitude": "",
            "longitude": "",
            "industry_track": "智能制造与高端装备",
            "industry_subtrack": "智能装备",
            "supply_chain_role": "中游装备制造",
            "products": "电池装配设备,自动化产线",
            "capabilities": "自动化装配,设备集成",
            "input_materials": "传感器",
            "output_products": "电池制造装备",
            "target_customer_industries": "动力电池,新能源汽车",
            "summary": "公开资料显示企业提供电池制造自动化装备。",
            "source_id": "S001",
            "source_url": "https://example.test/s001",
            "source_date": "2025-01-01",
            "verified_date": "2026-09-02",
            "confidence": "medium",
            "notes": "",
        },
    ]
    with (raw / "companies.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(companies[0]))
        writer.writeheader()
        writer.writerows(companies)
    with (raw / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "title", "publisher", "url", "published_date", "accessed_date", "source_type", "scope", "notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "S001",
                "title": "示例政府公开名单",
                "publisher": "示例发布单位",
                "url": "https://example.test/s001",
                "published_date": "2025-01-01",
                "accessed_date": "2026-09-02",
                "source_type": "政府公开名单",
                "scope": "示例",
                "notes": "",
            }
        )
    sector = {
        "sector_id": "battery",
        "name": "新能源汽车与新能源",
        "overview": "示例产业链资料。",
        "upstream": ["电池材料"],
        "midstream": ["动力电池"],
        "downstream": ["新能源汽车"],
        "keywords": ["锂电", "正极材料"],
        "transport_modes": ["公路"],
        "source_ids": ["S001"],
        "source_urls": ["https://example.test/s001"],
        "verified_date": "2026-09-02",
    }
    asset = {
        "asset_id": "A001",
        "name": "示例物流服务",
        "asset_type": "物流",
        "district": "常州市",
        "description": "示例公开物流资源。",
        "reference_location": "常州市",
        "latitude": "",
        "longitude": "",
        "serves_sectors": ["新能源汽车与新能源"],
        "source_ids": ["S001"],
        "source_urls": ["https://example.test/s001"],
        "published_date": "2025-01-01",
        "verified_date": "2026-09-02",
        "confidence": "high",
        "notes": "",
    }
    (raw / "sector_profiles.jsonl").write_text(json.dumps(sector, ensure_ascii=False) + "\n", encoding="utf-8")
    (raw / "regional_assets.jsonl").write_text(json.dumps(asset, ensure_ascii=False) + "\n", encoding="utf-8")
    return root / "data"


class DataValidationTests(unittest.TestCase):
    def test_valid_data_has_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = write_fixture(Path(temporary))
            report = validate_data(data_dir, min_companies=2)
            self.assertTrue(report.ok, report.to_dict())
            self.assertEqual(report.counts["companies"], 2)

    def test_unknown_source_id_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = write_fixture(Path(temporary), broken_source_ref=True)
            report = validate_data(data_dir)
            self.assertFalse(report.ok)
            self.assertTrue(any("unknown source_id" in issue.message for issue in report.errors))


@unittest.skipUnless(HAS_NUMPY, "numpy is required for index tests")
class LocalIndexTests(unittest.TestCase):
    def test_rebuild_is_byte_deterministic_and_retrieves_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = write_fixture(Path(temporary))
            manifest_one = build_index(data_dir, embedding_backend="hash", hash_dimension=128, min_companies=2)
            index_path = data_dir / "derived" / "company_index.npz"
            first_bytes = index_path.read_bytes()
            manifest_two = build_index(data_dir, embedding_backend="hash", hash_dimension=128, min_companies=2)
            self.assertEqual(first_bytes, index_path.read_bytes())
            self.assertEqual(manifest_one, manifest_two)

            retriever = LocalRetriever(data_dir)
            self.assertTrue(retriever.status()["index_status"].startswith("persisted:hash"))
            results = retriever.search("锂电池正极材料", filters={"district": "金坛"}, top_k=3)
            self.assertEqual(results[0]["company_id"], "C001")
            self.assertEqual(results[0]["sources"][0]["source_id"], "S001")
            self.assertIn("锂电池正极材料", retriever.build_evidence_context(results))

            matching = retriever.match_business_need("寻找锂电池正极材料供应商", top_k=2)
            self.assertTrue(matching["suppliers"])
            self.assertIn("潜在供应商", matching["suppliers"][0]["match_reason"])

