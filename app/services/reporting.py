"""Evidence-grounded consultation reporting with an optional OpenAI-compatible LLM."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

from .repository import CompanyRepository


class ConsultationService:
    """Turn local retrieval results into an auditable招商研判 response.

    The deterministic version is always assembled first.  An optional LLM can
    polish only the narrative fields after retrieval, never replace the source
    cards or invent the list of matched companies.
    """

    def __init__(self, repository: CompanyRepository) -> None:
        self.repository = repository
        self.llm = OpenAICompatibleReporter.from_environment()

    async def answer(
        self,
        *,
        query: str,
        district: str | None = None,
        sector: str | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        matches = await asyncio.to_thread(
            self.repository.match_business_need,
            query,
            district=district,
            sector=sector,
            top_k=8,
        )
        evidence = self._unique_companies(matches)
        assets = await asyncio.to_thread(self.repository.regional_assets, district, 6)
        sources = self._unique_sources(evidence, assets)
        report = self._local_report(query, matches, assets, language)
        generation = {"mode": "local", "note": "本地检索与规则化报告"}

        if self.llm.is_configured and evidence:
            generated = await self.llm.generate(
                query=query,
                district=district,
                sector=sector,
                language=language,
                evidence=evidence,
                assets=assets,
            )
            if generated:
                report = self._merge_generated_report(report, generated)
                generation = {"mode": "llm", "note": f"已使用 {self.llm.model} 依据本次证据生成叙述"}
            else:
                generation = {"mode": "local", "note": "模型服务未返回可用结果，已保留本地可追溯报告"}

        return {
            "query": query,
            "filters": {"district": district or "", "sector": sector or ""},
            "generation": generation,
            "report": report,
            "matches": {
                "suppliers": matches.get("suppliers", [])[:4],
                "customers": matches.get("customers", [])[:4],
                "partners": matches.get("partners", [])[:4],
                "all_matches": matches.get("all_matches", [])[:8],
                "rationale": matches.get("rationale", ""),
            },
            "regional_assets": self._normalise_assets(assets),
            "sources": sources,
            "evidence_count": len(evidence),
            "data_status": self.repository.status(),
        }

    @staticmethod
    def _unique_companies(matches: dict[str, Any]) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in ("suppliers", "customers", "partners", "all_matches"):
            for company in matches.get(group, []):
                if not isinstance(company, dict):
                    continue
                identity = str(company.get("company_id") or company.get("name") or "")
                if not identity or identity in seen:
                    continue
                seen.add(identity)
                ordered.append(company)
        return ordered[:8]

    @staticmethod
    def _unique_sources(companies: list[dict[str, Any]], assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        candidates: list[Any] = []
        for company in companies:
            candidates.extend(company.get("sources", []))
        for asset in assets:
            candidates.extend(asset.get("sources", []) if isinstance(asset, dict) else [])
            if isinstance(asset, dict) and asset.get("source_url"):
                candidates.append(
                    {
                        "source_id": asset.get("source_id", "regional-asset"),
                        "title": asset.get("source_title") or asset.get("name") or "区域资源来源",
                        "url": asset.get("source_url"),
                        "published_date": asset.get("source_date", ""),
                    }
                )
        for position, source in enumerate(candidates, start=1):
            if not isinstance(source, dict):
                continue
            identity = str(source.get("url") or source.get("source_id") or f"source-{position}")
            if identity in seen:
                continue
            seen.add(identity)
            result.append(
                {
                    "source_id": str(source.get("source_id") or source.get("id") or f"S{position}"),
                    "title": str(source.get("title") or source.get("source_title") or "公开资料来源"),
                    "url": str(source.get("url") or source.get("source_url") or ""),
                    "published_date": str(source.get("published_date") or source.get("source_date") or ""),
                    "publisher": str(source.get("publisher") or ""),
                }
            )
        return result[:16]

    def _local_report(
        self,
        query: str,
        matches: dict[str, Any],
        assets: list[dict[str, Any]],
        language: str,
    ) -> dict[str, Any]:
        suppliers = matches.get("suppliers", [])
        customers = matches.get("customers", [])
        partners = matches.get("partners", [])
        all_matches = matches.get("all_matches", [])
        names = self._company_names(all_matches)
        roles = self._role_terms(all_matches)
        asset_names = [str(asset.get("name") or asset.get("asset_name") or "") for asset in assets if isinstance(asset, dict)]

        if language == "en":
            overview = (
                f"The current public-data corpus returned {len(all_matches)} evidence-backed company matches for “{query}”. "
                "The result is a lead-generation view, not a verification of an existing commercial relationship."
            )
            supply_chain = (
                f"Observed roles in the retrieved evidence: {', '.join(roles[:6]) or 'not sufficiently tagged'}. "
                "Prioritize direct confirmation of specifications, capacity, certifications, and commercial fit."
            )
            local_assets = (
                f"Available regional-resource evidence: {', '.join(asset_names[:4])}."
                if asset_names
                else "No verified regional-resource record was retrieved for this filter."
            )
            recommendations = [
                "Start with the highest-ranked companies and verify their current product scope using the linked public sources.",
                "Treat supplier/customer labels as potential matches until both sides confirm specifications and sourcing intent.",
                "Record qualification requirements and the missing information needed for the next round of matching.",
            ]
            caveats = [
                "Only public, in-corpus evidence is used; blank fields are not inferred.",
                "No route distance or real-time industrial/business registration data is claimed in this demo.",
            ]
        else:
            overview = (
                f"围绕“{query}”，当前公开资料库检出 {len(all_matches)} 家可回溯证据的潜在企业。"
                "结果用于招商线索研判，不代表企业间已经存在采购、供货或合作关系。"
            )
            supply_chain = (
                f"本次命中资料中出现的产业链角色包括：{('、'.join(roles[:6])) or '资料标签不足'}。"
                "建议以产品规格、产能、认证和实际采购意向进行下一轮核验。"
            )
            local_assets = (
                f"本次筛选可关联的区域资源包括：{'、'.join(asset_names[:4])}。"
                if asset_names
                else "当前筛选未检出已核验的交通或区域配套资料。"
            )
            recommendations = [
                "优先联系排序靠前的企业，并以页面来源链接复核其当前产品与能力范围。",
                "将“潜在供应商/客户”视为初步线索，双方确认规格与采购意向后再形成正式匹配。",
                "补录认证、产能、关键物料和目标客户行业等缺失字段，提高下一轮匹配精度。",
            ]
            caveats = [
                "仅使用已入库的公开资料；空白字段不作推断。",
                "演示版不声称驾车距离、实时工商状态或企业间已建立的商务关系。",
            ]

        return {
            "overview": overview,
            "supplier_intro": self._role_intro(suppliers, "supplier", language),
            "customer_intro": self._role_intro(customers, "customer", language),
            "partner_intro": self._role_intro(partners, "partner", language),
            "supply_chain": supply_chain,
            "local_assets": local_assets,
            "recommendations": recommendations,
            "caveats": caveats,
            "matched_company_names": names,
        }

    @staticmethod
    def _company_names(companies: list[dict[str, Any]]) -> list[str]:
        return [str(item.get("name")) for item in companies if item.get("name")][:8]

    @staticmethod
    def _role_terms(companies: list[dict[str, Any]]) -> list[str]:
        output: list[str] = []
        for company in companies:
            for role in company.get("supply_chain_role", []):
                role = str(role).strip()
                if role and role not in output:
                    output.append(role)
        return output

    @staticmethod
    def _role_intro(companies: list[dict[str, Any]], role: str, language: str) -> str:
        count = len(companies)
        if language == "en":
            label = {"supplier": "potential supplier", "customer": "potential customer", "partner": "potential partner"}[role]
            return f"{count} {label} evidence card(s) are shown below; each is a lead, not a confirmed relationship."
        label = {"supplier": "潜在供应商", "customer": "潜在客户", "partner": "潜在合作方"}[role]
        return f"下方展示 {count} 家{label}线索；均为公开资料推断，尚待业务核验。"

    @staticmethod
    def _normalise_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            normalized.append(
                {
                    "name": str(asset.get("name") or asset.get("asset_name") or "区域资源"),
                    "type": str(asset.get("type") or asset.get("asset_type") or "配套资源"),
                    "district": str(asset.get("district") or asset.get("area") or ""),
                    "summary": str(asset.get("summary") or asset.get("description") or ""),
                    "source_url": str(asset.get("source_url") or asset.get("url") or ""),
                    "source_title": str(asset.get("source_title") or asset.get("title") or "公开资料来源"),
                }
            )
        return normalized

    @staticmethod
    def _merge_generated_report(local: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
        # Keep company lists and role cards local.  The model may only improve
        # narrative fields whose source evidence stays visible beside the report.
        merged = dict(local)
        for key in ("overview", "supply_chain", "local_assets"):
            value = generated.get(key)
            if isinstance(value, str) and value.strip():
                merged[key] = value.strip()[:1800]
        for key in ("recommendations", "caveats"):
            value = generated.get(key)
            if isinstance(value, list):
                cleaned = [str(item).strip()[:420] for item in value if str(item).strip()]
                if cleaned:
                    merged[key] = cleaned[:5]
        return merged


class OpenAICompatibleReporter:
    """A small, provider-neutral Chat Completions client.

    No default URL is supplied: the app never accidentally sends enterprise
    records away merely because a key file exists elsewhere on the machine.
    """

    def __init__(self, *, api_key: str, base_url: str, model: str, timeout_seconds: float) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleReporter":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENCODEGO_API_KEY") or ""
        if not api_key:
            api_key = cls._read_key_file(os.getenv("OPENCODEGO_KEY_FILE", ""))
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENCODEGO_BASE_URL") or ""
        model = os.getenv("LLM_MODEL") or os.getenv("OPENCODEGO_MODEL") or "deepseek-v4-flash"
        try:
            timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        except ValueError:
            timeout_seconds = 45.0
        return cls(api_key=api_key.strip(), base_url=base_url.strip(), model=model.strip(), timeout_seconds=timeout_seconds)

    @staticmethod
    def _read_key_file(configured_path: str) -> str:
        """Read an explicitly configured local key file without logging its value.

        The accepted file formats are a single token, an ``API_KEY=value``
        line, or a standalone token at the beginning of a notes file.  The last
        form supports a user-maintained credential note without treating its
        explanatory prose as a credential.  It is deliberately opt-in through
        ``.env``; no workspace file is discovered automatically.
        """

        if not configured_path.strip():
            return ""
        path = Path(configured_path.strip())
        try:
            text = path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError):
            return ""
        if not text:
            return ""
        standalone_token = ""
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            if "=" in candidate:
                name, value = candidate.split("=", 1)
                if name.strip().upper() in {"API_KEY", "LLM_API_KEY", "OPENCODEGO_API_KEY"}:
                    return value.strip().strip('"').strip("'")
            # A raw token may precede human-readable usage notes.  Restrict the
            # accepted shape so ordinary text, URLs, and Markdown headings are
            # never mistaken for a credential.
            if not standalone_token and re.fullmatch(r"[A-Za-z0-9._-]{20,}", candidate):
                standalone_token = candidate
        return text if "\n" not in text else standalone_token

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    async def generate(
        self,
        *,
        query: str,
        district: str | None,
        sector: str | None,
        language: str,
        evidence: list[dict[str, Any]],
        assets: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.is_configured:
            return None
        payload = {
            "model": self.model,
            "temperature": 0.15,
            # The model only polishes the narrative.  A bounded response keeps
            # the local demo responsive; verified cards and sources are kept
            # entirely in the browser response below.
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._system_prompt(language)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_need": query,
                            "filters": {"district": district or "", "sector": sector or ""},
                            "company_evidence": [self._compact_company(item) for item in evidence[:6]],
                            "regional_asset_evidence": [self._compact_asset(item) for item in assets[:4]],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        endpoint = f"{self.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return self._parse_json_object(content)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            # Intentionally do not log request/response bodies, which can include
            # the key or enterprise data depending on a provider's error format.
            return None

    @staticmethod
    def _compact_company(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "company_id": item.get("company_id"),
            "name": item.get("name"),
            "district": item.get("district"),
            "park": item.get("park"),
            "sector": item.get("sector"),
            "industry": item.get("industry"),
            "products": item.get("products", []),
            "capabilities": item.get("capabilities", []),
            "supply_chain_role": item.get("supply_chain_role", []),
            "summary": item.get("summary", ""),
            "matched_terms": item.get("matched_terms", []),
            "source_ids": item.get("source_ids", []),
        }

    @staticmethod
    def _compact_asset(item: dict[str, Any]) -> dict[str, Any]:
        """Keep LLM context small; full linked records remain local."""
        return {
            "name": item.get("name") or item.get("asset_name"),
            "type": item.get("type") or item.get("asset_type"),
            "district": item.get("district") or item.get("area"),
            "summary": item.get("summary") or item.get("description"),
            "source_id": item.get("source_id"),
        }

    @staticmethod
    def _system_prompt(language: str) -> str:
        output_language = "English" if language == "en" else "简体中文"
        return f"""You are an evidence-grounded industrial investment analyst. Write in {output_language}.
Use only the supplied public evidence. Do not invent companies, certifications, capacity, revenue, transport times, routes, or existing commercial relationships. A supplier/customer/partner is always a potential match. If evidence is insufficient, say so clearly.
Return one strict JSON object with exactly these optional keys: overview (string), supply_chain (string), local_assets (string), recommendations (array of strings), caveats (array of strings). Keep every string concise (under 160 Chinese characters or 120 English words) and use at most 3 items in each array. Do not include Markdown or any company list; the application renders verified company cards itself."""

    @staticmethod
    def _parse_json_object(content: Any) -> dict[str, Any] | None:
        if not isinstance(content, str):
            return None
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
