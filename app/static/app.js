/* Global fetch is intentionally limited to this same-origin local FastAPI app. */

const state = {
  options: { districts: [], sectors: [], roles: [] },
  directory: { page: 1, hasMore: false, loaded: false },
};

const elements = {
  consultForm: document.querySelector("#consultForm"),
  consultQuery: document.querySelector("#consultQuery"),
  consultSubmit: document.querySelector("#consultSubmit"),
  consultHint: document.querySelector("#consultHint"),
  districtFilter: document.querySelector("#districtFilter"),
  sectorFilter: document.querySelector("#sectorFilter"),
  languageSelect: document.querySelector("#languageSelect"),
  emptyResults: document.querySelector("#emptyResults"),
  reportResults: document.querySelector("#reportResults"),
  results: document.querySelector("#results"),
  dataStateBadge: document.querySelector("#dataStateBadge"),
  dataNotice: document.querySelector("#dataNotice"),
  dataNoticeText: document.querySelector("#dataNoticeText"),
  retrievalBackend: document.querySelector("#retrievalBackend"),
  directoryForm: document.querySelector("#directoryForm"),
  directoryQuery: document.querySelector("#directoryQuery"),
  directoryDistrict: document.querySelector("#directoryDistrict"),
  directorySector: document.querySelector("#directorySector"),
  directoryFeedback: document.querySelector("#directoryFeedback"),
  companyGrid: document.querySelector("#companyGrid"),
  directoryCount: document.querySelector("#directoryCount"),
  loadMoreCompanies: document.querySelector("#loadMoreCompanies"),
  companyDialog: document.querySelector("#companyDialog"),
  companyDialogContent: document.querySelector("#companyDialogContent"),
  closeCompanyDialog: document.querySelector("#closeCompanyDialog"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function safeUrl(value) {
  try {
    const url = new URL(String(value));
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { Accept: "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = "本地服务暂时无法完成请求。";
    try {
      detail = (await response.json()).detail || detail;
    } catch { /* The default detail is enough for a local UI. */ }
    throw new Error(detail);
  }
  return response.json();
}

function setOptions(select, entries, placeholder, preserveValue = true) {
  const previous = preserveValue ? select.value : "";
  const options = [`<option value="">${escapeHtml(placeholder)}</option>`]
    .concat(entries.map((entry) => `<option value="${escapeHtml(entry)}">${escapeHtml(entry)}</option>`));
  select.innerHTML = options.join("");
  if (previous && entries.includes(previous)) select.value = previous;
}

async function loadFilterOptions() {
  try {
    const options = await requestJson("/api/filter-options");
    state.options = options;
    setOptions(elements.districtFilter, options.districts || [], "常州全域");
    setOptions(elements.sectorFilter, options.sectors || [], "全部赛道");
    setOptions(elements.directoryDistrict, options.districts || [], "全部区域");
    setOptions(elements.directorySector, options.sectors || [], "全部赛道");
  } catch (error) {
    elements.consultHint.textContent = `筛选项暂不可用：${error.message}`;
  }
}

async function refreshDataStatus() {
  try {
    const data = await requestJson("/api/data-status");
    elements.dataStateBadge.textContent = `${data.company_count || 0} 家资料`;
    elements.dataStateBadge.className = `data-state data-state-${data.state || "waiting"}`;
    elements.dataNotice.dataset.state = data.state || "waiting";
    elements.dataNoticeText.textContent = data.message || "正在读取资料状态。";
    elements.retrievalBackend.textContent = data.retrieval_backend || "";
    return data;
  } catch (error) {
    elements.dataNoticeText.textContent = `无法读取本地资料状态：${error.message}`;
    return null;
  }
}

function setConsultLoading(isLoading) {
  elements.consultSubmit.disabled = isLoading;
  elements.consultSubmit.classList.toggle("is-loading", isLoading);
  elements.consultSubmit.innerHTML = isLoading
    ? "<span>正在检索公开资料…</span>"
    : "<span>生成招商研判</span><span aria-hidden=\"true\">→</span>";
  elements.results.setAttribute("aria-busy", String(isLoading));
}

function renderTags(values, className = "") {
  return (values || []).slice(0, 5).map((value) => `<span class="tag ${className}">${escapeHtml(value)}</span>`).join("");
}

function sourceLink(source, label = "查看来源") {
  const url = safeUrl(source?.url);
  const title = escapeHtml(source?.title || label);
  if (!url) return `<span class="company-source-link" aria-label="${title}">⌁ ${title}</span>`;
  return `<a class="company-source-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">↗ ${title}</a>`;
}

function companyCard(company, { compact = false } = {}) {
  const location = [company.district, company.park].filter(Boolean).join(" · ") || "区域资料待补充";
  const source = (company.sources || [])[0];
  const tags = company.products?.length ? company.products : (company.capabilities || company.supply_chain_role || []);
  const score = Number(company.score || 0);
  const evidenceTags = (company.matched_terms || []).slice(0, 2);
  const control = compact
    ? `<div class="card-footer"><span class="source-count">${(company.sources || []).length} 条来源</span><button class="detail-button" type="button" data-company-id="${escapeHtml(company.company_id)}">查看资料 →</button></div>`
    : sourceLink(source, "公开来源");
  return `
    <article class="${compact ? "directory-company" : "match-company"}">
      ${compact ? `<p class="company-classification">${escapeHtml(company.sector || company.industry || "产业资料")}</p>` : ""}
      <div class="company-topline">
        <button class="company-name-button" type="button" data-company-id="${escapeHtml(company.company_id)}">${escapeHtml(company.name || "未命名企业")}</button>
        ${!compact && score ? `<span class="company-score">相关度 ${escapeHtml(score.toFixed(2))}</span>` : ""}
      </div>
      <p class="company-location">${escapeHtml(location)}</p>
      ${company.summary ? `<p class="company-summary">${escapeHtml(company.summary)}</p>` : ""}
      ${!compact && company.match_reason ? `<p class="match-reason">匹配依据：${escapeHtml(company.match_reason)}</p>` : ""}
      <div class="tag-row">${renderTags(tags)}${renderTags(evidenceTags, "match-tag")}</div>
      ${control}
    </article>`;
}

function dedupeCompanyGroups(matches) {
  const seen = new Set();
  const unique = (items) => (items || []).filter((company) => {
    const id = company.company_id || company.name;
    if (!id || seen.has(id)) return false;
    seen.add(id);
    return true;
  });
  const suppliers = unique(matches.suppliers);
  const customers = unique(matches.customers);
  const partners = unique([...(matches.partners || []), ...(matches.all_matches || [])]);
  return { suppliers, customers, partners };
}

function matchColumn(title, number, intro, companies) {
  return `<div class="match-column">
    <p class="match-label"><span>${number}</span>${escapeHtml(title)}</p>
    <div class="match-stack">
      ${companies.length ? companies.map((company) => companyCard(company)).join("") : `<div class="match-empty">${escapeHtml(intro)}</div>`}
    </div>
  </div>`;
}

function renderAssets(assets) {
  if (!assets?.length) return `<div class="match-empty">当前筛选未检出已核验的区域配套资料。</div>`;
  return `<div class="asset-list">${assets.map((asset) => {
    const link = safeUrl(asset.source_url);
    return `<article class="asset-item">
      <span class="asset-type">${escapeHtml(asset.type || "区域资源")}</span>
      <strong>${escapeHtml(asset.name)}</strong>
      ${asset.summary ? `<p>${escapeHtml(asset.summary)}</p>` : ""}
      ${link ? `<a class="company-source-link" href="${escapeHtml(link)}" target="_blank" rel="noreferrer">↗ ${escapeHtml(asset.source_title || "公开来源")}</a>` : ""}
    </article>`;
  }).join("")}</div>`;
}

function renderSources(sources) {
  if (!sources?.length) return `<div class="match-empty">本次未附带可访问的来源链接；请在企业资料库中补录来源。</div>`;
  return `<div class="source-list">${sources.map((source) => {
    const link = safeUrl(source.url);
    return `<article class="source-item">
      <strong>${escapeHtml(source.title || "公开资料来源")}</strong>
      ${link ? `<a href="${escapeHtml(link)}" target="_blank" rel="noreferrer">${escapeHtml(link)}</a>` : "<span class=\"source-meta\">暂未提供访问链接</span>"}
      ${(source.publisher || source.published_date) ? `<p class="source-meta">${escapeHtml([source.publisher, source.published_date].filter(Boolean).join(" · "))}</p>` : ""}
    </article>`;
  }).join("")}</div>`;
}

function reportList(items) {
  return `<ul class="report-list">${(items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function renderReport(payload) {
  const report = payload.report || {};
  const groups = dedupeCompanyGroups(payload.matches || {});
  const query = payload.query || "本次需求";
  elements.reportResults.innerHTML = `
    <div class="report-header">
      <div>
        <p class="eyebrow">招商研判结果</p>
        <h2 class="report-title">“${escapeHtml(query)}”</h2>
        <p class="report-subtitle">已基于 ${escapeHtml(payload.evidence_count || 0)} 家命中企业的公开资料整理。企业卡片和来源链接可供逐条复核。</p>
      </div>
      <span class="generation-note">${escapeHtml(payload.generation?.note || "本地可追溯报告")}</span>
    </div>
    <div class="report-grid">
      <article class="report-card overview-card"><h3>行业概况</h3><p>${escapeHtml(report.overview || "当前资料不足以生成行业概况。")}</p></article>
      <article class="report-card chain-card"><h3>产业链与本地配套</h3><p>${escapeHtml(report.supply_chain || "当前资料不足以形成产业链判断。")}</p></article>
      <article class="report-card advice-card"><h3>招商建议</h3>${reportList(report.recommendations || [])}</article>
      <article class="report-card caveat-card"><h3>使用边界</h3>${reportList(report.caveats || [])}</article>
    </div>
    <section class="match-section" aria-labelledby="matchTitle">
      <div class="match-section-heading"><h2 id="matchTitle">潜在企业匹配</h2><p>${escapeHtml(payload.matches?.rationale || "企业角色仅为公开资料基础上的潜在匹配。")}</p></div>
      <div class="match-columns">
        ${matchColumn("潜在供应商", "1", report.supplier_intro || "暂无足够证据将企业归为潜在供应商。", groups.suppliers)}
        ${matchColumn("潜在客户", "2", report.customer_intro || "暂无足够证据将企业归为潜在客户。", groups.customers)}
        ${matchColumn("潜在合作方", "3", report.partner_intro || "暂无更多可归类的潜在合作企业。", groups.partners)}
      </div>
    </section>
    <section class="asset-source-layout">
      <article class="report-card"><h3>区域配套与交通资源</h3><p style="margin-bottom:12px">${escapeHtml(report.local_assets || "仅展示已核验的区域资源资料。")}</p>${renderAssets(payload.regional_assets)}</article>
      <article class="report-card"><h3>本次使用的公开来源</h3>${renderSources(payload.sources)}</article>
    </section>`;
  elements.emptyResults.hidden = true;
  elements.reportResults.hidden = false;
}

function showConsultError(message) {
  elements.emptyResults.hidden = true;
  elements.reportResults.hidden = false;
  elements.reportResults.innerHTML = `<div class="empty-results"><p class="eyebrow">请求未完成</p><h2>暂时无法生成研判</h2><p>${escapeHtml(message)} 请确认本地服务仍在运行后重试。</p></div>`;
}

async function submitConsultation(event) {
  event.preventDefault();
  const query = elements.consultQuery.value.trim();
  if (query.length < 2) {
    elements.consultHint.textContent = "请至少输入两个字的企业、行业或招商需求。";
    elements.consultQuery.focus();
    return;
  }
  setConsultLoading(true);
  elements.consultHint.textContent = "正在从本地资料中检索，并整理可回溯的产业链线索。";
  try {
    const payload = await requestJson("/api/consult", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        district: elements.districtFilter.value || null,
        sector: elements.sectorFilter.value || null,
        language: elements.languageSelect.value,
      }),
    });
    renderReport(payload);
    await refreshDataStatus();
    elements.consultHint.textContent = payload.generation?.mode === "llm"
      ? "已将本次问题和少量命中公开摘要交由已配置模型生成叙述；完整资料库与密钥未外发。"
      : "本次仅使用本地检索与规则化报告；每项匹配仍应通过公开来源进一步确认。";
    document.querySelector("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showConsultError(error.message);
    elements.consultHint.textContent = "未能完成本次研判，请稍后重试。";
  } finally {
    setConsultLoading(false);
  }
}

async function loadCompanies({ append = false } = {}) {
  if (!append) state.directory.page = 1;
  const params = new URLSearchParams({
    q: elements.directoryQuery.value.trim(),
    district: elements.directoryDistrict.value,
    sector: elements.directorySector.value,
    page: String(state.directory.page),
    page_size: "18",
  });
  elements.directoryFeedback.textContent = "正在读取企业资料…";
  try {
    const payload = await requestJson(`/api/companies?${params.toString()}`);
    const cards = (payload.items || []).map((company) => companyCard(company, { compact: true })).join("");
    if (append) elements.companyGrid.insertAdjacentHTML("beforeend", cards);
    else elements.companyGrid.innerHTML = cards || `<div class="match-empty">当前条件下没有可展示的企业资料。请调整关键词或筛选范围。</div>`;
    state.directory.hasMore = Boolean(payload.has_more);
    state.directory.loaded = true;
    elements.loadMoreCompanies.hidden = !state.directory.hasMore;
    elements.directoryCount.textContent = payload.total || 0;
    elements.directoryFeedback.textContent = payload.total
      ? `显示 ${Math.min((payload.page || 1) * (payload.page_size || 18), payload.total)} / ${payload.total} 家企业`
      : "尚无企业资料。数据导入后会自动显示在这里。";
  } catch (error) {
    elements.companyGrid.innerHTML = `<div class="match-empty">无法读取企业资料：${escapeHtml(error.message)}</div>`;
    elements.directoryFeedback.textContent = "";
    elements.loadMoreCompanies.hidden = true;
  }
}

async function openCompany(companyId) {
  if (!companyId) return;
  try {
    const company = await requestJson(`/api/companies/${encodeURIComponent(companyId)}`);
    const location = [company.district, company.park, company.address].filter(Boolean).join(" · ") || "暂未提供";
    const metaItem = (title, value) => `<div><dt>${escapeHtml(title)}</dt><dd>${escapeHtml(value || "暂未提供")}</dd></div>`;
    const sourceHtml = (company.sources || []).length ? company.sources.map((source) => {
      const link = safeUrl(source.url);
      return `<article class="dialog-source">${link ? `<a href="${escapeHtml(link)}" target="_blank" rel="noreferrer">${escapeHtml(source.title || "公开资料来源")} ↗</a>` : `<strong>${escapeHtml(source.title || "公开资料来源")}</strong>`}<p>${escapeHtml([source.publisher, source.published_date].filter(Boolean).join(" · ") || "来源日期待补充")}</p></article>`;
    }).join("") : "<div class=\"match-empty\">当前企业资料尚未附带来源链接。</div>";
    elements.companyDialogContent.innerHTML = `<div class="dialog-content">
      <h2 id="companyDialogTitle">${escapeHtml(company.name)}</h2>
      ${company.english_name ? `<p class="dialog-subtitle">${escapeHtml(company.english_name)}</p>` : ""}
      <section class="detail-section"><dl class="detail-meta">
        ${metaItem("区域 / 园区", location)}
        ${metaItem("产业赛道", [company.sector, company.industry].filter(Boolean).join(" · "))}
        ${metaItem("产业链角色", (company.supply_chain_role || []).join("、"))}
        ${metaItem("核验日期", company.verified_date)}
      </dl></section>
      ${company.summary ? `<section class="detail-section"><h3>公开简介</h3><p>${escapeHtml(company.summary)}</p></section>` : ""}
      <section class="detail-section"><h3>产品与能力</h3><div class="tag-row">${renderTags(company.products)}${renderTags(company.capabilities)}</div></section>
      ${(company.input_materials || company.output_products || company.target_customer_industries) ? `<section class="detail-section"><h3>产业链信息</h3><dl class="detail-meta">${metaItem("输入材料", (company.input_materials || []).join("、"))}${metaItem("输出产品", (company.output_products || []).join("、"))}${metaItem("目标客户行业", (company.target_customer_industries || []).join("、"))}${metaItem("资料可信度", company.confidence)}</dl></section>` : ""}
      <section class="detail-section"><h3>公开来源</h3><div class="dialog-sources">${sourceHtml}</div></section>
    </div>`;
    if (typeof elements.companyDialog.showModal === "function") elements.companyDialog.showModal();
  } catch (error) {
    elements.directoryFeedback.textContent = `无法打开企业资料：${error.message}`;
  }
}

function switchView(viewName) {
  document.querySelectorAll(".view").forEach((view) => {
    const active = view.id === viewName;
    view.hidden = !active;
    view.classList.toggle("is-active", active);
  });
  document.querySelectorAll("[data-view-target]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.viewTarget === viewName);
  });
  if (viewName === "directory" && !state.directory.loaded) loadCompanies();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function wireEvents() {
  elements.consultForm.addEventListener("submit", submitConsultation);
  document.querySelectorAll(".prompt-chip").forEach((button) => button.addEventListener("click", () => {
    elements.consultQuery.value = button.dataset.prompt || "";
    elements.consultQuery.focus();
  }));
  document.querySelectorAll("[data-view-target]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.viewTarget)));
  elements.directoryForm.addEventListener("submit", (event) => { event.preventDefault(); loadCompanies(); });
  elements.loadMoreCompanies.addEventListener("click", () => {
    state.directory.page += 1;
    loadCompanies({ append: true });
  });
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-company-id]");
    if (button) openCompany(button.dataset.companyId);
  });
  elements.closeCompanyDialog.addEventListener("click", () => elements.companyDialog.close());
  elements.companyDialog.addEventListener("click", (event) => {
    const bounds = elements.companyDialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
      elements.companyDialog.close();
    }
  });
}

async function initialize() {
  wireEvents();
  await Promise.all([loadFilterOptions(), refreshDataStatus()]);
  // Render the directory once in the background so its navigation is instant.
  loadCompanies();
}

initialize();
