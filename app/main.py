"""FastAPI entry point for the local Changzhou investment-matching demo."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.services import CompanyRepository, ConsultationService


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

# Interpret a credential-file path from demo/.env relative to this project, not
# relative to whichever directory happened to launch Uvicorn.  The value stays
# in process memory only and is never exposed by the API.
_configured_key_file = os.getenv("OPENCODEGO_KEY_FILE", "").strip()
if _configured_key_file:
    _key_file_path = Path(_configured_key_file).expanduser()
    if not _key_file_path.is_absolute():
        os.environ["OPENCODEGO_KEY_FILE"] = str((PROJECT_DIR / _key_file_path).resolve())


def configured_data_dir() -> Path:
    configured = Path(os.getenv("APP_DATA_DIR", "data"))
    return configured if configured.is_absolute() else PROJECT_DIR / configured


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = CompanyRepository(configured_data_dir())
    app.state.repository = repository
    app.state.consultation_service = ConsultationService(repository)
    yield


app = FastAPI(
    title="常州产业链招商助手",
    summary="公开资料可追溯的企业检索与招商线索演示版",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


class ConsultationRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000, description="企业、行业或招商需求")
    district: str | None = Field(default=None, max_length=64)
    sector: str | None = Field(default=None, max_length=128)
    language: Literal["zh", "en"] = "zh"


def get_repository(request: Request) -> CompanyRepository:
    repository = getattr(request.app.state, "repository", None)
    if repository is None:
        raise HTTPException(status_code=503, detail="服务正在初始化，请稍后重试。")
    return repository


def get_consultation_service(request: Request) -> ConsultationService:
    service = getattr(request.app.state, "consultation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="服务正在初始化，请稍后重试。")
    return service


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request) -> HTMLResponse:
    repository = get_repository(request)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"data_status": repository.status()},
    )


@app.get("/api/health")
async def health(request: Request) -> dict:
    repository = get_repository(request)
    return {"ok": True, "data": repository.status()}


@app.get("/api/data-status")
async def data_status(request: Request) -> dict:
    return get_repository(request).status()


@app.get("/api/filter-options")
async def filter_options(request: Request) -> dict:
    return get_repository(request).filter_options()


@app.get("/api/companies")
async def list_companies(
    request: Request,
    q: str = Query(default="", max_length=160),
    district: str | None = Query(default=None, max_length=64),
    sector: str | None = Query(default=None, max_length=128),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
) -> dict:
    return get_repository(request).list_companies(
        query=q,
        district=district,
        sector=sector,
        page=page,
        page_size=page_size,
    )


@app.get("/api/companies/{company_id}")
async def company_detail(company_id: str, request: Request) -> dict:
    company = get_repository(request).get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="未找到该企业资料。")
    return company


@app.post("/api/consult")
async def consult(payload: ConsultationRequest, request: Request) -> dict:
    service = get_consultation_service(request)
    return await service.answer(
        query=payload.query.strip(),
        district=(payload.district or "").strip() or None,
        sector=(payload.sector or "").strip() or None,
        language=payload.language,
    )
