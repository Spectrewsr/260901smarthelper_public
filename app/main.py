"""FastAPI entry point for the local Changzhou investment knowledge platform."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.services import AdvancedRAG, DemoUser, ExportService, InvestmentAgent, KnowledgeRepository, LocalAuthService


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MAX_IMPORT_BYTES = 5 * 1024 * 1024
load_dotenv(PROJECT_DIR / ".env")


def configured_data_dir() -> Path:
    configured = Path(os.getenv("APP_DATA_DIR", "data"))
    return configured if configured.is_absolute() else PROJECT_DIR / configured


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = KnowledgeRepository(configured_data_dir())
    await asyncio.to_thread(repository.ensure_ready)
    rag = AdvancedRAG(repository)
    app.state.repository = repository
    app.state.agent = InvestmentAgent(repository, rag)
    app.state.auth = LocalAuthService(repository)
    app.state.exports = ExportService(repository)
    yield


app = FastAPI(
    title="常州产业招商知识平台",
    summary="可追溯的 Advanced RAG、SQL 地理查询与招商协同演示版",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=4, max_length=256)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    district: str | None = Field(default=None, max_length=64)
    sector: str | None = Field(default=None, max_length=128)
    mode: Literal["auto", "sql", "geo", "graph", "park", "landing", "project", "hybrid"] = "auto"
    company_ids: list[str] = Field(default_factory=list, max_length=30)
    park_ids: list[str] = Field(default_factory=list, max_length=10)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    radius_km: float | None = Field(default=None, gt=0, le=200)
    project_name: str | None = Field(default=None, max_length=160)


class LedgerCreateRequest(BaseModel):
    company_id: str = Field(min_length=2, max_length=64)
    project_name: str = Field(min_length=2, max_length=160)
    stage: str = Field(default="待联系", min_length=2, max_length=64)
    owner: str = Field(default="招商专员", min_length=2, max_length=64)
    next_step: str = Field(default="核验公开资料并预约首次沟通", min_length=2, max_length=300)
    contact_name: str = Field(default="演示联系人", max_length=64)
    contact_phone: str = Field(default="13800138123", max_length=32)
    contact_email: str = Field(default="demo.contact@example.test", max_length=128)


class LedgerUpdateRequest(BaseModel):
    stage: str = Field(min_length=2, max_length=64)
    owner: str = Field(min_length=2, max_length=64)
    next_step: str = Field(min_length=2, max_length=300)


class ManualCompanyRequest(BaseModel):
    company_name: str = Field(min_length=2, max_length=200)
    source_id: str = Field(min_length=2, max_length=32)
    district: str = Field(default="常州市", max_length=64)
    industry_track: str = Field(default="", max_length=128)
    industry_subtrack: str = Field(default="", max_length=128)
    supply_chain_role: str = Field(default="", max_length=300)
    products: str = Field(default="", max_length=500)
    capabilities: str = Field(default="", max_length=500)
    summary: str = Field(default="", max_length=1600)
    confidence: Literal["high", "medium", "low"] = "medium"
    notes: str = Field(default="", max_length=500)


class MaterialExportRequest(BaseModel):
    company_ids: list[str] = Field(min_length=1, max_length=12)
    project_name: str = Field(min_length=2, max_length=160)
    format: Literal["docx", "pptx"]
    template_key: Literal["chain", "park", "landing"] = "chain"


def get_repository(request: Request) -> KnowledgeRepository:
    repository = getattr(request.app.state, "repository", None)
    if repository is None:
        raise HTTPException(status_code=503, detail="服务正在初始化，请稍后重试。")
    return repository


def get_agent(request: Request) -> InvestmentAgent:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="智能研判服务正在初始化，请稍后重试。")
    return agent


def get_auth(request: Request) -> LocalAuthService:
    auth = getattr(request.app.state, "auth", None)
    if auth is None:
        raise HTTPException(status_code=503, detail="认证服务正在初始化，请稍后重试。")
    return auth


def get_exports(request: Request) -> ExportService:
    export_service = getattr(request.app.state, "exports", None)
    if export_service is None:
        raise HTTPException(status_code=503, detail="导出服务正在初始化，请稍后重试。")
    return export_service


def current_user(authorization: str | None = Header(default=None), auth: LocalAuthService = Depends(get_auth)) -> DemoUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先使用本地演示账号登录。")
    user = auth.decode(authorization.split(" ", 1)[1].strip())
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已失效，请重新登录。")
    return user


def admin_user(user: DemoUser = Depends(current_user)) -> DemoUser:
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="此功能仅对管理员开放。")
    return user


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request) -> HTMLResponse:
    repository = get_repository(request)
    return templates.TemplateResponse(request=request, name="index.html", context={"data_status": repository.status()})


@app.get("/api/health")
async def health(request: Request) -> dict:
    repository = get_repository(request)
    return {"ok": True, "data": repository.status(), "architecture": "Advanced RAG / SQLite / local GraphRAG"}


@app.get("/api/data-status")
async def data_status(request: Request) -> dict:
    return get_repository(request).status()


@app.get("/api/acceptance")
async def acceptance_status() -> dict:
    return {
        "implemented": [
            "统一入口与两级权限、敏感字段脱敏",
            "产业链上下游匹配、距离估算、标准化清单",
            "园区对标、独资洽谈要点与分阶段方案",
            "政务办事、生活配套、交通区位落地包",
            "招商材料 DOCX/PPTX 与企业对接台账",
            "单一轻量 Agent 与三类知识库",
            "CSV 预检、审核发布、人工录入与标签/来源校验",
        ],
        "explicitly_out_of_scope": ["国产化基础软硬件适配与内网安全防护", "项目实施、用户培训和一年运维服务"],
        "query_architecture": "Contextual Retrieval + FTS5/BM25 + BGE-M3 + RRF + BGE Cross-Encoder; SQL/Geo bypass RAG",
    }


@app.get("/api/filter-options")
async def filter_options(request: Request) -> dict:
    return get_repository(request).filter_options()


@app.get("/api/export-templates")
async def export_templates(request: Request) -> dict:
    return {"items": get_exports(request).template_options()}


@app.get("/api/companies")
async def list_companies(
    request: Request,
    q: str = Query(default="", max_length=160),
    district: str | None = Query(default=None, max_length=64),
    sector: str | None = Query(default=None, max_length=128),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
) -> dict:
    return get_repository(request).list_companies(query=q, district=district, sector=sector, page=page, page_size=page_size)


@app.get("/api/companies/{company_id}")
async def company_detail(company_id: str, request: Request) -> dict:
    company = get_repository(request).get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="未找到该企业资料。")
    return company


@app.post("/api/auth/login")
async def login(payload: LoginRequest, request: Request) -> dict:
    result = get_auth(request).login(payload.username, payload.password)
    if not result:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="账号或密码不正确。")
    token, user = result
    return {"access_token": token, "token_type": "bearer", "user": user.to_dict()}


@app.get("/api/auth/me")
async def me(user: DemoUser = Depends(current_user)) -> dict:
    return {"user": user.to_dict()}


@app.post("/api/query")
async def query(payload: QueryRequest, request: Request) -> dict:
    agent = get_agent(request)
    return await asyncio.to_thread(
        agent.answer,
        query=payload.query,
        district=payload.district,
        sector=payload.sector,
        mode=payload.mode,
        company_ids=payload.company_ids,
        park_ids=payload.park_ids,
        latitude=payload.latitude,
        longitude=payload.longitude,
        radius_km=payload.radius_km,
        project_name=payload.project_name,
    )


@app.post("/api/consult")
async def consult_compat(payload: QueryRequest, request: Request) -> dict:
    """Compatibility path retained for existing local bookmarks."""
    return await query(payload, request)


@app.get("/api/parks/compare")
async def compare_parks(request: Request, park_ids: str = Query(default="")) -> dict:
    ids = [item.strip() for item in park_ids.split(",") if item.strip()]
    return get_repository(request).compare_parks(ids)


@app.get("/api/landing/package")
async def landing_package(request: Request, project_name: str = Query(min_length=2, max_length=160), district: str | None = Query(default=None)) -> dict:
    return get_repository(request).landing_package(project_name=project_name, district=district)


@app.get("/api/ledgers")
async def ledgers(request: Request, user: DemoUser = Depends(current_user)) -> dict:
    return {"items": get_repository(request).list_ledgers(role=user.role)}


@app.post("/api/ledgers")
async def create_ledger(payload: LedgerCreateRequest, request: Request, user: DemoUser = Depends(current_user)) -> dict:
    try:
        return get_repository(request).add_ledger(actor=user.username, role=user.role, **payload.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/ledgers/{ledger_id}")
async def ledger_detail(ledger_id: int, request: Request, user: DemoUser = Depends(current_user)) -> dict:
    item = get_repository(request).get_ledger(ledger_id, role=user.role)
    if not item:
        raise HTTPException(status_code=404, detail="未找到台账。")
    return item


@app.patch("/api/ledgers/{ledger_id}")
async def update_ledger(ledger_id: int, payload: LedgerUpdateRequest, request: Request, user: DemoUser = Depends(current_user)) -> dict:
    item = get_repository(request).update_ledger(ledger_id, actor=user.username, role=user.role, **payload.model_dump())
    if not item:
        raise HTTPException(status_code=404, detail="未找到台账。")
    return item


@app.post("/api/admin/imports/preview")
async def preview_import(request: Request, file: UploadFile = File(...), user: DemoUser = Depends(admin_user)) -> dict:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="演示版目前仅接受 UTF-8 CSV。")
    content = await file.read(MAX_IMPORT_BYTES + 1)
    if len(content) > MAX_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail="CSV 文件超过 5 MB 演示版导入上限。")
    try:
        return get_repository(request).preflight_import(content, filename=file.filename or "upload.csv", actor=user.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/imports/{import_id}/publish")
async def publish_import(import_id: str, request: Request, user: DemoUser = Depends(admin_user)) -> dict:
    try:
        return get_repository(request).publish_import(import_id, actor=user.username)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/admin/companies")
async def add_manual_company(payload: ManualCompanyRequest, request: Request, user: DemoUser = Depends(admin_user)) -> dict:
    try:
        return get_repository(request).add_manual_company(payload.model_dump(), actor=user.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/exports/target-list")
async def export_target_list(request: Request, company_ids: str = Query(min_length=2), user: DemoUser = Depends(current_user)) -> FileResponse:
    ids = [item.strip() for item in company_ids.split(",") if item.strip()]
    try:
        path, export_id = get_exports(request).target_list_csv(ids, actor=user.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, filename=path.name, media_type="text/csv; charset=utf-8", headers={"X-Export-Id": export_id})


@app.post("/api/exports/material")
async def export_material(payload: MaterialExportRequest, request: Request, user: DemoUser = Depends(current_user)) -> FileResponse:
    try:
        path, export_id = await asyncio.to_thread(
            get_exports(request).material,
            payload.company_ids,
            project_name=payload.project_name,
            fmt=payload.format,
            actor=user.username,
            template_key=payload.template_key,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if payload.format == "docx" else "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    return FileResponse(path, filename=path.name, media_type=media_type, headers={"X-Export-Id": export_id})
