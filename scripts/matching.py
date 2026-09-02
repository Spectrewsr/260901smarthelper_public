"""Explainable, conservative business-matching rules for retrieved companies.

These helpers rank *potential* matches.  They never assert that two companies
already trade with one another: every reason is tied to a supplied product,
capability, input material, target industry, or explicitly recorded role.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

try:
    from .data_contract import clean_text
except ImportError:  # pragma: no cover - direct script use.
    from data_contract import clean_text  # type: ignore


_TOKEN_RE = re.compile(r"[\u3400-\u9fff]+|[a-z0-9][a-z0-9.+#_-]*", re.IGNORECASE)

_SUPPLIER_CUES = (
    "供应",
    "上游",
    "原料",
    "材料",
    "零部件",
    "配套",
    "制造",
    "加工",
    "生产",
    "supplier",
    "upstream",
    "material",
    "component",
)
_CUSTOMER_CUES = (
    "客户",
    "下游",
    "应用",
    "整机",
    "终端",
    "系统集成",
    "装配",
    "需求",
    "customer",
    "downstream",
    "application",
    "oem",
)
_PARTNER_CUES = (
    "合作",
    "招商",
    "落地",
    "配套",
    "研发",
    "联合",
    "园区",
    "partner",
    "cooperation",
    "r&d",
)


@dataclass(frozen=True)
class MatchAssessment:
    """One transparent potential-match assessment."""

    kind: str
    score: float
    matched_terms: tuple[str, ...]
    reason: str
    explicit_role: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_type": self.kind,
            "match_score": round(float(self.score), 4),
            "matched_terms": list(self.matched_terms),
            "match_reason": self.reason,
            "explicit_role": self.explicit_role,
        }


def normalize(value: Any) -> str:
    return unicodedata.normalize("NFKC", clean_text(value)).casefold()


def _token_set(value: Any) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_RE.finditer(normalize(value)):
        run = match.group(0)
        if not run:
            continue
        if "\u3400" <= run[0] <= "\u9fff":
            # Two-character grams offer a useful compromise between generic
            # single characters and overly strict full phrase equality.
            if len(run) >= 2:
                tokens.update(run[index : index + 2] for index in range(len(run) - 1))
            else:
                tokens.add(run)
        else:
            tokens.add(run)
    return tokens


def _values(company: Mapping[str, Any], field: str) -> list[str]:
    value = company.get(field, [])
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [clean_text(item) for item in value if clean_text(item)]
    return []


def _query_field_overlap(query: str, values: Iterable[str]) -> tuple[float, list[str]]:
    """Return an explainable overlap score and values that supplied evidence."""

    normalized_query = normalize(query)
    query_tokens = _token_set(query)
    if not normalized_query:
        return 0.0, []
    evidence: list[str] = []
    best = 0.0
    for original in values:
        normalized_value = normalize(original)
        if not normalized_value:
            continue
        if len(normalized_value) >= 2 and normalized_value in normalized_query:
            score = 1.0
        elif len(normalized_query) >= 2 and normalized_query in normalized_value:
            score = 0.85
        else:
            value_tokens = _token_set(original)
            if not query_tokens or not value_tokens:
                score = 0.0
            else:
                intersection = len(query_tokens & value_tokens)
                score = intersection / max(len(query_tokens), 1)
                # A single generic two-character gram should not be called a
                # convincing match on its own.
                if intersection == 1 and len(query_tokens) > 2:
                    score *= 0.55
        if score >= 0.18:
            evidence.append(original)
        best = max(best, score)
    return best, evidence[:4]


def _has_cue(values: Iterable[str], cues: Iterable[str]) -> tuple[bool, list[str]]:
    joined = " ".join(normalize(value) for value in values)
    found = [cue for cue in cues if cue in joined]
    return bool(found), found


def infer_query_intents(query: str) -> set[str]:
    """Detect explicit request wording; both primary views remain available."""

    text = normalize(query)
    intents: set[str] = set()
    if any(cue in text for cue in ("供应商", "上游", "采购", "原料", "供应", "supplier")):
        intents.add("supplier")
    if any(cue in text for cue in ("客户", "下游", "销售", "买家", "需求", "customer")):
        intents.add("customer")
    if any(cue in text for cue in ("合作", "招商", "落地", "伙伴", "partner")):
        intents.add("partner")
    return intents


def assess_company_match(company: Mapping[str, Any], query: str, kind: str) -> MatchAssessment:
    """Assess a potential supplier/customer/partner using only shown fields."""

    if kind not in {"supplier", "customer", "partner"}:
        raise ValueError(f"Unsupported match kind: {kind}")

    role_values = _values(company, "supply_chain_role")
    if kind == "supplier":
        evidence_fields = (
            ("产品", _values(company, "products")),
            ("能力/工序", _values(company, "capabilities")),
            ("输出产品", _values(company, "output_products")),
            ("产业链角色", role_values),
        )
        role_cues = _SUPPLIER_CUES
        label = "潜在供应商"
    elif kind == "customer":
        evidence_fields = (
            ("输入材料", _values(company, "input_materials")),
            ("目标客户行业", _values(company, "target_customer_industries")),
            ("产品", _values(company, "products")),
            ("产业链角色", role_values),
        )
        role_cues = _CUSTOMER_CUES
        label = "潜在客户"
    else:
        evidence_fields = (
            ("能力/工序", _values(company, "capabilities")),
            ("产品", _values(company, "products")),
            ("产业链角色", role_values),
        )
        role_cues = _PARTNER_CUES
        label = "潜在合作方"

    evidence_bits: list[str] = []
    matched_terms: list[str] = []
    field_score = 0.0
    for label_name, values in evidence_fields:
        score, terms = _query_field_overlap(query, values)
        if terms:
            evidence_bits.append(f"{label_name}含“{'、'.join(terms[:2])}”")
            matched_terms.extend(terms)
        field_score = max(field_score, score)

    explicit_role, matched_role_cues = _has_cue(role_values, role_cues)
    role_bonus = 0.22 if explicit_role else 0.0
    if explicit_role and role_values:
        evidence_bits.insert(0, f"公开产业链角色为“{'、'.join(role_values[:2])}”")
    # A concise public summary is supporting evidence only; it should not give a
    # role claim the dataset did not contain.
    summary_score, summary_terms = _query_field_overlap(query, [clean_text(company.get("summary", ""))])
    if summary_terms and not evidence_bits:
        evidence_bits.append(f"公开简介提及“{'、'.join(summary_terms[:2])}”")
        matched_terms.extend(summary_terms)
    total = min(1.0, 0.72 * field_score + 0.28 * summary_score + role_bonus)
    if evidence_bits:
        reason = "；".join(evidence_bits[:2]) + "，因此列为" + label + "（需进一步商务核验）。"
    elif explicit_role:
        reason = f"公开产业链角色包含“{'、'.join(role_values[:2])}”，因此列为{label}（需进一步商务核验）。"
    else:
        reason = f"与查询在公开资料中存在语义相关性，暂列为{label}（缺少直接供需关系证据，需人工核验）。"
    deduped_terms: list[str] = []
    seen: set[str] = set()
    for term in matched_terms:
        key = normalize(term)
        if key and key not in seen:
            seen.add(key)
            deduped_terms.append(term)
    return MatchAssessment(
        kind=kind,
        score=total,
        matched_terms=tuple(deduped_terms[:6]),
        reason=reason,
        explicit_role=explicit_role,
    )


def general_matched_terms(company: Mapping[str, Any], query: str) -> list[str]:
    """Return public fields that explain a general retrieval result."""

    fields = (
        "company_name",
        "aliases",
        "industry_track",
        "industry_subtrack",
        "products",
        "capabilities",
        "input_materials",
        "output_products",
        "target_customer_industries",
    )
    matches: list[str] = []
    seen: set[str] = set()
    for field in fields:
        values = _values(company, field)
        if not values and isinstance(company.get(field), str):
            values = [clean_text(company[field])]
        _, evidence = _query_field_overlap(query, values)
        for item in evidence:
            key = normalize(item)
            if key and key not in seen:
                seen.add(key)
                matches.append(item)
    return matches[:8]

