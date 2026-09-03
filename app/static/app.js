/* Same-origin browser client for the local investment knowledge platform. */

const state = {
  token: sessionStorage.getItem("cz_demo_token") || "",
  user: null,
  options: { districts: [], sectors: [], parks: [] },
  directory: { page: 1, hasMore: false },
  selectedCompanyIds: new Set(),
  currentCompanyIds: [],
  currentResult: null,
  importId: "",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const elements = {
  queryForm: $("#queryForm"), queryInput: $("#queryInput"), queryDistrict: $("#queryDistrict"), querySector: $("#querySector"), queryMode: $("#queryMode"), projectName: $("#projectName"), geoFields: $("#geoFields"), latitude: $("#latitude"), longitude: $("#longitude"), radiusKm: $("#radiusKm"), querySubmit: $("#querySubmit"), queryHint: $("#queryHint"), queryResults: $("#queryResults"),
  dataBadge: $("#dataBadge"), companyCount: $("#companyCount"), parkCount: $("#parkCount"), supportCount: $("#supportCount"), backendStatus: $("#backendStatus"),
  directoryForm: $("#directoryForm"), directoryQuery: $("#directoryQuery"), directoryDistrict: $("#directoryDistrict"), directorySector: $("#directorySector"), directoryTotal: $("#directoryTotal"), companyGrid: $("#companyGrid"), loadMore: $("#loadMoreCompanies"),
  parkIds: $("#parkIds"), compareParks: $("#compareParks"), parkResults: $("#parkResults"),
  landingProjectName: $("#landingProjectName"), landingDistrict: $("#landingDistrict"), buildLanding: $("#buildLanding"), landingResults: $("#landingResults"),
  ledgerGate: $("#ledgerGate"), ledgerWorkspace: $("#ledgerWorkspace"), ledgerForm: $("#ledgerForm"), ledgerResults: $("#ledgerResults"),
  adminGate: $("#adminGate"), adminWorkspace: $("#adminWorkspace"), importFile: $("#importFile"), previewImport: $("#previewImport"), importResults: $("#importResults"), manualCompanyForm: $("#manualCompanyForm"),
  loginButton: $("#loginButton"), logoutButton: $("#logoutButton"), loginDialog: $("#loginDialog"), loginForm: $("#loginForm"), loginUsername: $("#loginUsername"), loginPassword: $("#loginPassword"), loginFeedback: $("#loginFeedback"), closeLogin: $("#closeLogin"),
  companyDialog: $("#companyDialog"), companyDialogBody: $("#companyDialogBody"), closeCompany: $("#closeCompany"), acceptanceDialog: $("#acceptanceDialog"), acceptanceBody: $("#acceptanceBody"), closeAcceptance: $("#closeAcceptance"), acceptanceButton: $("#acceptanceButton"), toast: $("#toast"),
};

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function safeUrl(value) {
  try { const url = new URL(String(value)); return ["https:", "http:"].includes(url.protocol) ? url.href : ""; } catch { return ""; }
}

function toArray(value) { return Array.isArray(value) ? value : []; }

async function api(url, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  if (state.token && options.auth !== false) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(url, { ...options, headers });
  if (!response.ok) {
    let message = "本地服务暂时无法完成请求。";
    try { message = (await response.json()).detail || message; } catch { /* Keep a useful fallback. */ }
    if (response.status === 401) updateSession(null);
    throw new Error(message);
  }
  return response.json();
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { elements.toast.hidden = true; }, 3200);
}

function setOptions(select, values, placeholder) {
  const selected = select.value;
  select.innerHTML = [`<option value="">${escapeHtml(placeholder)}</option>`, ...values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`)].join("");
  if (values.includes(selected)) select.value = selected;
}

function updateSession(user) {
  state.user = user;
  if (!user) { state.token = ""; sessionStorage.removeItem("cz_demo_token"); }
  elements.loginButton.hidden = Boolean(user);
  elements.logoutButton.hidden = !user;
  elements.logoutButton.textContent = user ? `${user.display_name} · 退出` : "退出";
  $$(".auth-only").forEach((item) => { item.hidden = false; });
  $$(".admin-only").forEach((item) => { item.hidden = false; });
  elements.ledgerGate.hidden = Boolean(user);
  elements.ledgerWorkspace.hidden = !user;
  const admin = user?.role === "admin";
  elements.adminGate.hidden = admin;
  elements.adminWorkspace.hidden = !admin;
  if (user) loadLedgers();
}

async function initialize() {
  wireEvents();
  await Promise.all([loadDataStatus(), loadFilterOptions(), restoreSession()]);
  await loadCompanies();
}

async function restoreSession() {
  if (!state.token) { updateSession(null); return; }
  try { const result = await api("/api/auth/me"); updateSession(result.user); } catch { updateSession(null); }
}

async function loadDataStatus() {
  try {
    const data = await api("/api/data-status", { auth: false });
    elements.dataBadge.textContent = `${data.company_count || 0} 家企业资料`;
    elements.companyCount.textContent = data.company_count || 0;
    elements.parkCount.textContent = data.counts?.parks ?? "—";
    elements.supportCount.textContent = data.counts?.landing_services ?? "—";
    elements.backendStatus.textContent = data.retrieval_backend || "本地知识底座";
  } catch (error) { showToast(error.message); }
}

async function loadFilterOptions() {
  try {
    state.options = await api("/api/filter-options", { auth: false });
    const districts = state.options.districts || [];
    const sectors = state.options.sectors || [];
    setOptions(elements.queryDistrict, districts, "常州全域");
    setOptions(elements.querySector, sectors, "全部赛道");
    setOptions(elements.directoryDistrict, districts, "全部区域");
    setOptions(elements.directorySector, sectors, "全部赛道");
    setOptions(elements.landingDistrict, districts, "常州市");
  } catch (error) { showToast(`筛选项加载失败：${error.message}`); }
}

function switchView(name) {
  $$(".view").forEach((view) => { const active = view.id === name; view.hidden = !active; view.classList.toggle("is-active", active); });
  $$(".nav-item").forEach((item) => item.classList.toggle("is-active", item.dataset.view === name));
  const label = { workspace: "智能招商工作台", directory: "企业与产业链", parks: "园区对标", landing: "落地配套", ledger: "企业对接台账", data: "知识与数据管理" }[name] || "招商知识工作空间";
  $("#viewTitle").textContent = label;
  if (name === "parks" && !elements.parkResults.children.length) loadParkComparison();
  if (name === "landing" && !elements.landingResults.children.length) loadLandingPackage();
  if (name === "ledger" && state.user) loadLedgers();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function modeChanged() {
  elements.geoFields.hidden = elements.queryMode.value !== "geo";
}

async function submitQuery(event) {
  event.preventDefault();
  const query = elements.queryInput.value.trim();
  if (!query) { elements.queryInput.focus(); elements.queryHint.textContent = "请输入需要研判的项目问题。"; return; }
  const payload = {
    query, district: elements.queryDistrict.value || null, sector: elements.querySector.value || null, mode: elements.queryMode.value,
    project_name: elements.projectName.value.trim() || null,
    latitude: numericOrNull(elements.latitude.value), longitude: numericOrNull(elements.longitude.value), radius_km: numericOrNull(elements.radiusKm.value),
  };
  elements.querySubmit.disabled = true;
  elements.querySubmit.textContent = "研判中…";
  elements.queryHint.textContent = "正在执行受限工具计划并整理来源证据。";
  try {
    const result = await api("/api/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), auth: false });
    renderQueryResult(result);
    elements.queryHint.textContent = result.execution?.rag_called ? "已执行本地 Advanced RAG，并显示每一阶段的检索轨迹。" : "本次为 SQL/地理/资料工具查询，未调用 RAG。";
  } catch (error) {
    elements.queryResults.innerHTML = `<div class="empty-state"><h2>本次研判未完成</h2><p>${escapeHtml(error.message)}</p></div>`;
    elements.queryHint.textContent = "请检查输入和本地服务状态后再试。";
  } finally {
    elements.querySubmit.disabled = false;
    elements.querySubmit.innerHTML = "开始研判 <span>→</span>";
  }
}

function numericOrNull(value) { const number = Number(value); return value === "" || Number.isNaN(number) ? null : number; }

function renderQueryResult(result) {
  state.currentResult = result;
  const execution = result.execution || {};
  const report = result.report || {};
  const companies = uniqueCompanies([...(result.companies || []), ...toArray(result.matches?.all_matches)]);
  state.currentCompanyIds = companies.map((company) => company.company_id);
  const tools = toArray(execution.tool_calls).map((tool) => `<span>${escapeHtml(tool.tool)}：${escapeHtml(tool.reason)}</span>`).join("<br>");
  const header = `<div class="results-header"><div><p class="eyebrow">研判结果</p><h2>“${escapeHtml(result.query || "本次问题")}”</h2></div><div class="execution-card"><strong>${escapeHtml(execution.mode || "local")}</strong><span>${execution.rag_called ? "已调用本地 RAG" : "未调用 RAG"}</span><span>${tools}</span></div></div>`;
  const overview = `<section class="result-overview panel"><p>${escapeHtml(report.overview || "当前资料不足以形成结论。")}</p></section>`;
  const supplyChain = report.supply_chain ? `<section class="result-overview panel"><h3>产业链研判</h3><p>${escapeHtml(report.supply_chain)}</p></section>` : "";
  const trace = result.retrieval_trace ? renderTrace(result.retrieval_trace) : "";
  const geo = result.geo ? renderGeoSummary(result.geo) : "";
  const matches = result.matches ? renderMatchGroups(result.matches) : (companies.length ? `<section class="panel match-group"><h3>查询结果</h3>${companies.map((company) => companyCard(company)).join("")}</section>` : "");
  const graph = result.graph ? renderGraph(result.graph) : "";
  const park = result.park_comparison ? renderParkResult(result.park_comparison) : "";
  const landing = result.landing_package ? renderLandingResult(result.landing_package) : "";
  const exports = companies.length ? renderExportBar(companies) : "";
  const advice = report.recommendations?.length ? `<section class="content-card panel"><h3>下一步建议</h3><ul>${report.recommendations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></section>` : "";
  const sources = renderSources(result.sources || []);
  elements.queryResults.innerHTML = `${header}${overview}${supplyChain}${trace}${geo}${matches}${graph}${park}${landing}${exports}${advice}${sources}`;
}

function renderTrace(trace) {
  const labels = { contextual: "Contextual Retrieval", lexical: "稀疏召回", dense: "稠密召回", fusion: "RRF 融合", reranker: "Cross-Encoder 重排" };
  return `<section class="trace-panel panel"><h3>可审计检索轨迹</h3><div class="trace-list">${Object.entries(trace).map(([key, value]) => {
    const descriptor = [value.engine, value.strategy, value.algorithm].filter(Boolean).join(" · ");
    const count = value.candidate_count ?? "—";
    const extra = value.fallback ? "（后备状态）" : "";
    return `<div class="trace-item"><b>${escapeHtml(labels[key] || key)}</b><span>${escapeHtml(descriptor || "本地阶段")}<br>候选：${escapeHtml(count)} ${escapeHtml(extra)}</span></div>`;
  }).join("")}</div></section>`;
}

function renderMatchGroups(matches) {
  const groups = [["潜在供应商", matches.suppliers || []], ["潜在客户", matches.customers || []], ["潜在合作方", matches.partners || []]];
  return `<section class="match-columns">${groups.map(([title, companies]) => `<article class="match-group panel"><h3>${title}</h3><p>${escapeHtml(matches.rationale || "全部为公开资料推断的潜在线索。")}</p>${companies.length ? companies.map((company) => companyCard(company)).join("") : `<p>暂无足够证据。</p>`}</article>`).join("")}</section>`;
}

function displayNumber(value, maximumFractionDigits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("zh-CN", { maximumFractionDigits }) : "";
}

function geoEstimate(company) {
  const distance = displayNumber(company.distance_km, 1);
  const duration = displayNumber(company.duration_min, 0);
  if (!distance && !duration) return "";
  const straight = displayNumber(company.straight_line_km, 1);
  const method = company.method === "local_road_factor_estimate" ? "RTree 初筛 + Haversine + 本地道路系数" : (company.method || "本地道路估算");
  const note = company.method_note || "道路距离为本地估算，不是实时导航或实测车程。";
  const facts = [distance ? `道路估算 ${distance} km` : "道路距离待补充", duration ? `预计车程约 ${duration} 分钟` : "车程待补充", straight ? `直线距离 ${straight} km` : ""].filter(Boolean).join(" · ");
  return `<div class="company-meta"><b>地理匹配：</b>${escapeHtml(facts)}<br><span>方法：${escapeHtml(method)}；${escapeHtml(note)}</span></div>`;
}

function renderGeoSummary(geo) {
  const point = geo.reference_point || {};
  const radius = displayNumber(point.radius_km, 1);
  const method = geo.execution?.method || "RTree + Haversine + local road-factor estimate";
  const center = Number.isFinite(Number(point.latitude)) && Number.isFinite(Number(point.longitude))
    ? `${displayNumber(point.latitude, 4)}, ${displayNumber(point.longitude, 4)}` : "当前查询参考点";
  return `<section class="result-overview panel"><h3>地理匹配口径</h3><p>参考点：${escapeHtml(center)}${radius ? `；检索半径：${escapeHtml(radius)} km` : ""}。道路距离和预计车程使用 ${escapeHtml(method)}，仅供线索初筛；不是实时导航或实测车程。</p></section>`;
}

function companyCard(company, compact = false) {
  const source = toArray(company.sources)[0] || {};
  const tags = [...toArray(company.products), ...toArray(company.capabilities), ...toArray(company.supply_chain_role)].slice(0, 4);
  const id = escapeHtml(company.company_id || "");
  const selected = state.selectedCompanyIds.has(company.company_id);
  const sourceUrl = safeUrl(source.url);
  return `<article class="${compact ? "directory-card" : "company-card"}"><h4><button type="button" data-company="${id}">${escapeHtml(company.name || "未命名企业")}</button></h4><div class="company-meta">${escapeHtml([company.district, company.sector, company.industry].filter(Boolean).join(" · ") || "资料待补充")}</div>${company.summary ? `<p>${escapeHtml(company.summary)}</p>` : ""}${company.match_reason ? `<p><b>匹配依据：</b>${escapeHtml(company.match_reason)}</p>` : ""}${geoEstimate(company)}<div class="tag-row">${tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>${sourceUrl ? `<a class="source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">[${escapeHtml(source.source_id || "来源")}] ${escapeHtml(source.title || "公开来源")} ↗</a>` : ""}${!compact ? `<div class="card-actions"><button class="mini-button" type="button" data-select="${id}">${selected ? "已加入导出" : "加入导出"}</button><button class="mini-button" type="button" data-ledger="${id}">加入台账</button></div>` : ""}</article>`;
}

function renderGraph(graph) {
  const nodes = Object.fromEntries(toArray(graph.nodes).map((node) => [node.node_id, node.label]));
  const edges = toArray(graph.edges);
  return `<section class="graph-panel panel"><h3>局部产业关系图谱（最多 ${escapeHtml(graph.max_hops || 0)} 跳）</h3><p class="company-meta">所有 inferred / potential 边为待核验关联，不代表已确认合作。</p><div class="graph-path">${edges.length ? edges.map((edge) => `<div class="graph-edge"><b>${escapeHtml(nodes[edge.from_node] || edge.from_node)}</b> → ${escapeHtml(edge.relation)} → <b>${escapeHtml(nodes[edge.to_node] || edge.to_node)}</b><span>${escapeHtml(edge.evidence)}｜${escapeHtml(edge.status)}｜来源：${escapeHtml(toArray(edge.source_ids).join("、") || "待核验")}</span></div>`).join("") : "<p>当前没有可展开的图谱边。</p>"}</div></section>`;
}

function renderParkResult(comparison) {
  const cards = toArray(comparison.parks).map((park) => `<article class="content-card panel"><h3>${escapeHtml(park.name)}</h3><p>${escapeHtml(park.summary || "")}</p><div class="metric-list">${toArray(park.metrics).map((metric) => `<div class="metric"><b>${escapeHtml(metric.label)}</b><span>${escapeHtml(metric.value)} · ${escapeHtml(metric.confidence || "unknown")}</span></div>`).join("")}</div>${renderSourceLinks(park.sources || [])}</article>`).join("");
  const phases = toArray(comparison.phased_plan).map((phase) => `<div class="phase"><b>${escapeHtml(phase.phase)} · ${escapeHtml(phase.goal)}</b><span>${escapeHtml(toArray(phase.actions).join("；"))}</span></div>`).join("");
  return `<section class="content-grid">${cards}<article class="content-card panel"><h3>独资设立洽谈要点</h3><ul>${toArray(comparison.negotiation_points).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul><div class="phase-list">${phases}</div></article></section>`;
}

function renderLandingResult(package) {
  const localization = package.localization && typeof package.localization === "object" ? package.localization : {};
  const district = package.district || localization.district || package.applicable_district || "";
  const scope = package.scope || localization.scope || package.applicable_scope || "";
  const note = package.localization_note || localization.note || package.note || "";
  const project = package.project_name || localization.project_name || "招商项目";
  const context = `<article class="content-card panel"><p class="eyebrow">定制化落地配套包</p><h3>${escapeHtml(project)}</h3>${district ? `<div class="metric"><b>适用区域</b><span>${escapeHtml(district)}</span></div>` : ""}${scope ? `<div class="metric"><b>适用范围</b><span>${escapeHtml(scope)}</span></div>` : ""}${note ? `<p><b>定制与核验说明：</b>${escapeHtml(note)}</p>` : ""}</article>`;
  const sections = toArray(package.sections).map((section) => `<article class="content-card panel"><h3>${escapeHtml(section.category)}</h3>${toArray(section.items).map((item) => {
    const itemScope = item.scope || item.district || item.applicable_scope || item.localization?.scope || "";
    const itemNote = item.localization_note || item.note || "";
    return `<div class="landing-item"><strong>${escapeHtml(item.name)}</strong><p>${escapeHtml(item.summary)}</p><span class="company-meta">${itemScope ? `适用：${escapeHtml(itemScope)} · ` : ""}置信度：${escapeHtml(item.confidence || "unknown")} · 核验：${escapeHtml(item.verified_date || "待补充")}</span>${itemNote ? `<p class="company-meta">${escapeHtml(itemNote)}</p>` : ""}${renderSourceLinks(item.sources || [])}</div>`;
  }).join("")}</article>`).join("");
  return `<section class="content-grid">${context}${sections}</section>`;
}

function renderExportBar(companies) {
  const selected = [...state.selectedCompanyIds].filter((id) => companies.some((company) => company.company_id === id));
  return `<section class="content-card panel"><h3>靶向招商清单与材料</h3><p>已选 ${selected.length} 家企业。未选择时，将使用当前结果中的前 3 家候选。</p><label class="inline-field">材料场景 <select id="materialTemplate"><option value="chain">产业链靶向招商</option><option value="park">独资设立园区洽谈</option><option value="landing">落地配套协同</option></select></label><div class="card-actions"><button class="secondary-button" data-export="csv" type="button">导出标准化清单 CSV</button><button class="secondary-button" data-export="docx" type="button">导出招商简报 DOCX</button><button class="secondary-button" data-export="pptx" type="button">导出招商简报 PPTX</button></div></section>`;
}

function renderSources(sources) {
  if (!toArray(sources).length) return "";
  return `<section class="source-panel panel"><h3>本次使用的公开来源</h3><div class="source-list">${sources.map((source) => { const url = safeUrl(source.url); return `<div><a href="${escapeHtml(url || "#")}" ${url ? "target=\"_blank\" rel=\"noreferrer\"" : ""}>[${escapeHtml(source.source_id || "来源")}] ${escapeHtml(source.title || "公开资料来源")}</a><span>${escapeHtml([source.publisher, source.published_date].filter(Boolean).join(" · "))}</span></div>`; }).join("")}</div></section>`;
}

function renderSourceLinks(sources) { return toArray(sources).slice(0, 2).map((source) => { const url = safeUrl(source.url); return url ? `<a class="source-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">[${escapeHtml(source.source_id || "来源")}] ${escapeHtml(source.title || "公开来源")} ↗</a>` : ""; }).join(""); }
function uniqueCompanies(companies) { const map = new Map(); companies.forEach((company) => { if (company?.company_id && !map.has(company.company_id)) map.set(company.company_id, company); }); return [...map.values()]; }

async function loadCompanies({ append = false } = {}) {
  if (!append) state.directory.page = 1;
  const params = new URLSearchParams({ q: elements.directoryQuery.value.trim(), district: elements.directoryDistrict.value, sector: elements.directorySector.value, page: String(state.directory.page), page_size: "18" });
  try {
    const data = await api(`/api/companies?${params}` , { auth: false });
    const cards = toArray(data.items).map((company) => companyCard(company, true)).join("") || `<div class="empty-state"><h2>没有符合条件的企业资料</h2></div>`;
    if (append) elements.companyGrid.insertAdjacentHTML("beforeend", cards); else elements.companyGrid.innerHTML = cards;
    state.directory.hasMore = Boolean(data.has_more);
    elements.loadMore.hidden = !state.directory.hasMore;
    elements.directoryTotal.textContent = `${data.total || 0} 家`;
  } catch (error) { elements.companyGrid.innerHTML = `<div class="empty-state"><h2>无法读取目录</h2><p>${escapeHtml(error.message)}</p></div>`; }
}

async function loadParkComparison() {
  const ids = elements.parkIds.value.trim();
  elements.parkResults.innerHTML = `<div class="empty-state">正在生成园区对标…</div>`;
  try { const data = await api(`/api/parks/compare?park_ids=${encodeURIComponent(ids)}`, { auth: false }); elements.parkResults.innerHTML = renderParkResult(data); } catch (error) { elements.parkResults.innerHTML = `<div class="empty-state"><p>${escapeHtml(error.message)}</p></div>`; }
}

async function loadLandingPackage() {
  const project = elements.landingProjectName.value.trim() || "招商项目";
  const district = elements.landingDistrict.value;
  elements.landingResults.innerHTML = `<div class="empty-state">正在生成落地配套包…</div>`;
  try { const data = await api(`/api/landing/package?project_name=${encodeURIComponent(project)}&district=${encodeURIComponent(district)}`, { auth: false }); elements.landingResults.innerHTML = renderLandingResult(data); } catch (error) { elements.landingResults.innerHTML = `<div class="empty-state"><p>${escapeHtml(error.message)}</p></div>`; }
}

async function loadLedgers() {
  if (!state.user) return;
  try {
    const data = await api("/api/ledgers");
    elements.ledgerResults.innerHTML = toArray(data.items).length ? data.items.map((item) => `<article class="ledger-card"><header><h3>${escapeHtml(item.company_name)} · ${escapeHtml(item.project_name)}</h3><span class="count-chip">${escapeHtml(item.stage)}</span></header><p>${escapeHtml(item.next_step)}</p><div class="ledger-meta"><span>负责人：${escapeHtml(item.owner)}</span><span>联系人：${escapeHtml(item.contact_name || "")}</span><span>${escapeHtml(item.contact_phone || "")}</span><span>${escapeHtml(item.contact_email || "")}</span></div></article>`).join("") : `<div class="empty-state"><h2>尚无台账</h2><p>可从企业卡片加入首条对接记录。</p></div>`;
  } catch (error) { elements.ledgerResults.innerHTML = `<div class="empty-state"><p>${escapeHtml(error.message)}</p></div>`; }
}

async function createLedger(companyId) {
  if (!state.user) { openLogin(); return; }
  const payload = { company_id: companyId, project_name: elements.projectName.value.trim() || "招商项目", stage: "待联系", owner: state.user.display_name, next_step: "核验公开资料并预约首次沟通" };
  try { await api("/api/ledgers", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); showToast("已加入企业对接台账。"); loadLedgers(); } catch (error) { showToast(error.message); }
}

async function submitLedger(event) {
  event.preventDefault();
  if (!state.user) { openLogin(); return; }
  const payload = { company_id: $("#ledgerCompanyId").value.trim(), project_name: $("#ledgerProjectName").value.trim(), stage: $("#ledgerStage").value, owner: $("#ledgerOwner").value.trim(), next_step: $("#ledgerNextStep").value.trim() };
  try { await api("/api/ledgers", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); showToast("台账已保存。"); elements.ledgerForm.reset(); loadLedgers(); } catch (error) { showToast(error.message); }
}

function toggleSelection(companyId) {
  if (!companyId) return;
  if (state.selectedCompanyIds.has(companyId)) state.selectedCompanyIds.delete(companyId); else state.selectedCompanyIds.add(companyId);
  const last = state.currentResult;
  if (last) renderQueryResult(last);
  else showToast(state.selectedCompanyIds.has(companyId) ? "已加入导出清单。" : "已从导出清单移除。");
}

async function downloadExport(format) {
  if (!state.user) { openLogin(); return; }
  let ids = [...state.selectedCompanyIds].filter((id) => state.currentCompanyIds.includes(id));
  if (!ids.length) ids = state.currentCompanyIds.slice(0, 3);
  if (!ids.length) { showToast("请先执行企业研判或选择企业。" ); return; }
  try {
    if (format === "csv") {
      await downloadResponse(`/api/exports/target-list?company_ids=${encodeURIComponent(ids.join(","))}`, { method: "GET" });
    } else {
      const template_key = $("#materialTemplate")?.value || "chain";
      await downloadResponse("/api/exports/material", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ company_ids: ids, project_name: elements.projectName.value.trim() || "招商项目", format, template_key }) });
    }
    showToast("材料已生成并开始下载。");
  } catch (error) { showToast(error.message); }
}

async function downloadResponse(url, options) {
  const headers = { ...(options.headers || {}), Authorization: `Bearer ${state.token}` };
  const response = await fetch(url, { ...options, headers });
  if (!response.ok) { let message = "导出失败"; try { message = (await response.json()).detail || message; } catch {} throw new Error(message); }
  const disposition = response.headers.get("content-disposition") || "";
  const match = /filename="?([^";]+)"?/i.exec(disposition);
  const name = match?.[1] || `export.${url.includes("target-list") ? "csv" : "bin"}`;
  const blob = await response.blob();
  const anchor = document.createElement("a"); anchor.href = URL.createObjectURL(blob); anchor.download = name; document.body.append(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(anchor.href);
}

async function openCompany(companyId) {
  try {
    const company = await api(`/api/companies/${encodeURIComponent(companyId)}`, { auth: false });
    elements.companyDialogBody.innerHTML = `<div class="company-detail"><p class="eyebrow">企业资料与来源</p><h2>${escapeHtml(company.name)}</h2><p>${escapeHtml(company.summary || "公开简介待补录")}</p><section class="detail-section"><div class="detail-grid"><div><b>区域 / 园区</b><span>${escapeHtml([company.district, company.park].filter(Boolean).join(" · ") || "待补充")}</span></div><div><b>产业赛道</b><span>${escapeHtml([company.sector, company.industry].filter(Boolean).join(" · ") || "待补充")}</span></div><div><b>产业链角色</b><span>${escapeHtml(toArray(company.supply_chain_role).join("、") || "待补充")}</span></div><div><b>坐标精度</b><span>${escapeHtml(company.coordinate_precision || "unknown")} · ${escapeHtml(company.coordinate_source || "")}</span></div></div></section><section class="detail-section"><h3>产品与能力</h3><div class="tag-row">${[...toArray(company.products), ...toArray(company.capabilities)].map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join("") || "暂无"}</div></section><section class="detail-section"><h3>公开来源</h3>${renderSources(company.sources || [])}</section></div>`;
    elements.companyDialog.showModal();
  } catch (error) { showToast(error.message); }
}

function openLogin() { elements.loginFeedback.textContent = ""; elements.loginDialog.showModal(); }
async function submitLogin(event) {
  event.preventDefault();
  try {
    const data = await api("/api/auth/login", { method: "POST", auth: false, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username: elements.loginUsername.value.trim(), password: elements.loginPassword.value }) });
    state.token = data.access_token; sessionStorage.setItem("cz_demo_token", state.token); updateSession(data.user); elements.loginDialog.close(); showToast(`已以${data.user.display_name}登录。`);
  } catch (error) { elements.loginFeedback.textContent = error.message; }
}

async function previewImport() {
  if (!state.user || state.user.role !== "admin") { openLogin(); return; }
  const file = elements.importFile.files[0];
  if (!file) { showToast("请选择 CSV 文件。" ); return; }
  const form = new FormData(); form.append("file", file);
  try {
    const response = await fetch("/api/admin/imports/preview", { method: "POST", headers: { Authorization: `Bearer ${state.token}` }, body: form });
    if (!response.ok) { const payload = await response.json(); throw new Error(payload.detail || "预检失败"); }
    const data = await response.json(); state.importId = data.import_id;
    elements.importResults.innerHTML = `<div class="import-report"><b>有效 ${data.valid_count} 行；无效 ${data.invalid_count} 行</b>${toArray(data.rows).map((row) => `<div class="import-row ${row.valid ? "is-valid" : "is-invalid"}">第 ${escapeHtml(row.row_number)} 行：${row.valid ? "通过" : escapeHtml(toArray(row.errors).join("；"))}</div>`).join("")}${data.valid_count ? `<button id="publishImport" class="primary-button" type="button">审核并发布有效行</button>` : ""}</div>`;
  } catch (error) { showToast(error.message); }
}

async function publishImport() {
  if (!state.importId) return;
  try { const data = await api(`/api/admin/imports/${encodeURIComponent(state.importId)}/publish`, { method: "POST" }); showToast(`已发布 ${toArray(data.published_company_ids).length} 条有效资料。`); await Promise.all([loadDataStatus(), loadFilterOptions(), loadCompanies()]); } catch (error) { showToast(error.message); }
}

async function submitManualCompany(event) {
  event.preventDefault();
  if (!state.user || state.user.role !== "admin") { openLogin(); return; }
  const payload = Object.fromEntries(new FormData(elements.manualCompanyForm).entries());
  try { const company = await api("/api/admin/companies", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); showToast(`已发布 ${company.name}。`); elements.manualCompanyForm.reset(); await Promise.all([loadDataStatus(), loadFilterOptions(), loadCompanies()]); } catch (error) { showToast(error.message); }
}

async function showAcceptance() {
  try {
    const data = await api("/api/acceptance", { auth: false });
    elements.acceptanceBody.innerHTML = `<div class="acceptance-content"><p class="eyebrow">知识库 PDF 验收范围</p><h2>本版已覆盖的模块</h2><ul>${toArray(data.implemented).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul><h3>查询架构</h3><p>${escapeHtml(data.query_architecture)}</p><p class="excluded"><b>明确排除：</b>${escapeHtml(toArray(data.explicitly_out_of_scope).join("；"))}</p></div>`;
    elements.acceptanceDialog.showModal();
  } catch (error) { showToast(error.message); }
}

function wireEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  elements.queryForm.addEventListener("submit", submitQuery);
  elements.queryMode.addEventListener("change", modeChanged);
  $$("[data-prompt]").forEach((button) => button.addEventListener("click", () => { elements.queryInput.value = button.dataset.prompt || ""; elements.queryInput.focus(); }));
  elements.directoryForm.addEventListener("submit", (event) => { event.preventDefault(); loadCompanies(); });
  elements.loadMore.addEventListener("click", () => { state.directory.page += 1; loadCompanies({ append: true }); });
  elements.compareParks.addEventListener("click", loadParkComparison);
  elements.buildLanding.addEventListener("click", loadLandingPackage);
  elements.ledgerForm.addEventListener("submit", submitLedger);
  elements.loginButton.addEventListener("click", openLogin); elements.logoutButton.addEventListener("click", () => { updateSession(null); showToast("已退出本地会话。"); });
  elements.loginForm.addEventListener("submit", submitLogin); elements.closeLogin.addEventListener("click", () => elements.loginDialog.close());
  elements.closeCompany.addEventListener("click", () => elements.companyDialog.close()); elements.closeAcceptance.addEventListener("click", () => elements.acceptanceDialog.close()); elements.acceptanceButton.addEventListener("click", (event) => { event.preventDefault(); showAcceptance(); });
  elements.previewImport.addEventListener("click", previewImport); elements.manualCompanyForm.addEventListener("submit", submitManualCompany);
  document.addEventListener("click", (event) => {
    const company = event.target.closest("[data-company]"); if (company) { openCompany(company.dataset.company); return; }
    const select = event.target.closest("[data-select]"); if (select) { toggleSelection(select.dataset.select); return; }
    const ledger = event.target.closest("[data-ledger]"); if (ledger) { createLedger(ledger.dataset.ledger); return; }
    const exportButton = event.target.closest("[data-export]"); if (exportButton) { downloadExport(exportButton.dataset.export); return; }
    if (event.target.closest("#publishImport")) publishImport();
  });
}

initialize();
