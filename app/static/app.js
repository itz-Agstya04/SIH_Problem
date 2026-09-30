const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = { summary: null, reports: [], filters: null, mlStatus: null, currentReport: null, currentView: "overview" };
const viewCopy = {
  overview: ["Overview", "SAFETY SIGNALS / FIELD INTELLIGENCE", "A reviewable view of the conditions and controls behind reported events."],
  reports: ["Report explorer", "SOURCE REPORTS / EVIDENCE", "Search the original narratives and inspect the evidence behind each screening result."],
  patterns: ["Precursor patterns", "RECURRING CONDITIONS / BARRIER FAILURES", "Repeated extracted barrier failures, grouped for expert review."],
  rules: ["Life-Saving Rules", "CONTROL MAPPING / PROVISIONAL CATALOGUE", "Rule matches are suggestions based on phrases in the report, not approved classifications."],
  review: ["Review queue", "HSE REVIEW / HUMAN DECISION", "Review flagged reports, record a decision, and retain the evidence trail."]
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
}
function scoreCell(score) {
  const percent = Math.round(Number(score || 0) * 100);
  return `<div class="score-cell"><div class="score-label"><span>Demo score</span><span>${percent}</span></div><div class="score-track"><span class="${percent >= 55 ? "high" : ""}" style="width:${Math.min(percent, 100)}%"></span></div></div>`;
}
function statusPill(status) {
  const label = { needs_review: "Needs review", unreviewed: "Unreviewed", reviewed: "Reviewed" }[status] || status || "Unknown";
  return `<span class="status-pill ${escapeHtml(status)}">${escapeHtml(label)}</span>`;
}
function rowMarkup(report, mode = "full") {
  const summary = `<span class="report-id">${escapeHtml(report.report_id)}</span><span class="report-snippet">${escapeHtml(report.raw_description)}</span>`;
  if (mode === "latest") return `<tr data-report-id="${escapeHtml(report.report_id)}"><td>${summary}</td><td>${escapeHtml(report.timestamp || "—")}</td><td>${escapeHtml(report.site)} / ${escapeHtml(report.activity)}</td><td>${escapeHtml(report.report_type)}</td><td>${scoreCell(report.sif_score)}</td><td>${statusPill(report.review_status)}</td></tr>`;
  if (mode === "review") return `<tr data-report-id="${escapeHtml(report.report_id)}"><td>${summary}</td><td>${escapeHtml(report.timestamp || "—")}</td><td>${escapeHtml(report.site)}</td><td>${scoreCell(report.sif_score)}</td><td>${statusPill(report.review_status)}</td></tr>`;
  return `<tr data-report-id="${escapeHtml(report.report_id)}"><td>${summary}</td><td>${escapeHtml(report.timestamp || "—")}</td><td>${escapeHtml(report.site)}</td><td>${escapeHtml(report.activity)}</td><td>${scoreCell(report.sif_score)}</td><td>${statusPill(report.review_status)}</td></tr>`;
}
function emptyRow(message, columns) { return `<tr><td colspan="${columns}" class="empty-state">${escapeHtml(message)}</td></tr>`; }
function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("visible"), 2800);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}
function meanText(score) { return `${Math.round((score || 0) * 100)} / 100`; }

async function loadData() {
  const [summary, reports, filters, mlStatus] = await Promise.all([api("/api/summary"), api("/api/reports"), api("/api/filters"), api("/api/ml/status")]);
  state.summary = summary;
  state.reports = reports.items;
  state.filters = filters;
  state.mlStatus = mlStatus;
  renderAll();
  $("#updated-at").textContent = `Updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}
function renderMetrics() {
  const data = state.summary;
  const items = [
    ["REPORTS IN VIEW", data.total_reports, "Demo and imported records"],
    ["HIGH SCREENING", data.high_screening_count, "Prototype score at or above 50"],
    ["MEAN DEMO SCORE", `${Math.round(data.mean_screening_score * 100)}%`, "Uncalibrated screening index"],
    ["AWAITING REVIEW", data.needs_review, "Flagged for closer attention"]
  ];
  $("#metric-grid").innerHTML = items.map(([label, value, foot]) => `<article class="metric"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(value)}</div><div class="metric-foot">${escapeHtml(foot)}</div></article>`).join("");
  $("#queue-count").textContent = data.needs_review;
}
function renderMlStatus() {
  const model = state.mlStatus;
  const meta = model.trained
    ? `Version ${escapeHtml(model.model_version)} · CV F1 ${model.metrics.f1} · ROC AUC ${model.metrics.roc_auc} · ${model.eligible_labeled_reports} eligible labels in this database`
    : `${model.eligible_labeled_reports} / ${model.required_total} eligible reports · minimum ${model.required_per_class} per class · current labels: ${model.class_counts.sif_potential} SIF / ${model.class_counts.not_sif_potential} non-SIF`;
  const heading = model.trained ? (model.prediction_ready ? "Reviewed-label SIF model has basic validation signal" : "Saved SIF model is blocked") : "Supervised SIF model is not trained yet";
  const message = model.trained ? model.message : "The model trains only from real reports with expert-reviewed labels; synthetic demo rows are excluded.";
  $("#ml-status-panel").innerHTML = `<div class="ml-status-copy"><strong>${escapeHtml(heading)}</strong><span>${escapeHtml(message)} ${model.trained ? "Even a passing baseline still needs calibration and HSE validation." : ""}</span></div><div class="ml-status-meta">${meta}</div><button class="button button-primary" id="train-model-button" ${model.ready_to_train ? "" : "disabled"}>${model.trained ? "Retrain model" : "Train model"}</button>`;
  $("#train-model-button").addEventListener("click", trainModel);
}
async function trainModel() {
  const button = $("#train-model-button");
  button.disabled = true;
  button.textContent = "Training…";
  try {
    const result = await api("/api/ml/train", { method: "POST" });
    await loadData();
    showToast(`Model trained. Cross-validation F1: ${result.metrics.f1}.`);
  } catch (error) {
    showToast(error.message);
    await loadData();
  }
}
function renderTrend() {
  const items = state.summary.monthly.slice(-8);
  const max = Math.max(.25, ...items.map((item) => item.score));
  $("#trend-chart").innerHTML = items.length ? items.map((item) => {
    const height = Math.max(4, (item.score / max) * 100);
    const month = item.month === "Unknown" ? "—" : item.month.slice(5);
    return `<div class="trend-item" title="${escapeHtml(item.month)}: ${Math.round(item.score * 100)} demo score, ${item.count} reports"><span class="trend-value">${Math.round(item.score * 100)}</span><div class="trend-bar-wrap"><div class="trend-bar ${item.score >= .55 ? "high" : ""}" style="height:${height}%"></div></div><span class="trend-label">${escapeHtml(month)}</span></div>`;
  }).join("") : `<p class="empty-state">No dated reports yet.</p>`;
}
function renderRankList() {
  const rules = state.summary.life_saving_rules.slice(0, 5);
  const max = Math.max(1, ...rules.map((item) => item.count));
  $("#rule-list").innerHTML = rules.length ? rules.map((item) => `<div class="rank-row"><span class="rank-name">${escapeHtml(item.name)}</span><span class="rank-count">${item.count}</span><div class="rank-track"><span style="width:${(item.count / max) * 100}%"></span></div></div>`).join("") : `<div class="empty-state">No rule evidence found.</div>`;
}
function renderBars(selector, rows) {
  const max = Math.max(.1, ...rows.map((row) => row.score));
  $(selector).innerHTML = rows.slice(0, 6).map((row) => `<div class="hbar-row" title="${escapeHtml(row.count)} reports"><span class="hbar-name">${escapeHtml(row.name)}</span><div class="hbar-track"><div class="hbar-fill ${row.score >= .55 ? "high" : ""}" style="width:${Math.max(3, row.score / max * 100)}%"></div></div><span class="hbar-value">${Math.round(row.score * 100)}</span></div>`).join("") || `<div class="empty-state">No data available.</div>`;
}
function renderTables() {
  const reports = state.reports;
  $("#latest-rows").innerHTML = reports.slice(0, 5).map((report) => rowMarkup(report, "latest")).join("") || emptyRow("No reports have been added.", 6);
  const search = $("#search-input").value.trim().toLocaleLowerCase();
  const site = $("#site-filter").value;
  const activity = $("#activity-filter").value;
  const type = $("#type-filter").value;
  const filtered = reports.filter((report) => (!site || report.site === site) && (!activity || report.activity === activity) && (!type || report.report_type === type) && (!search || `${report.report_id} ${report.raw_description}`.toLocaleLowerCase().includes(search)));
  $("#report-total").textContent = `${filtered.length} report${filtered.length === 1 ? "" : "s"}`;
  $("#report-rows").innerHTML = filtered.map((report) => rowMarkup(report)).join("") || emptyRow("No reports match these filters.", 6);
  const queue = reports.filter((report) => report.review_status === "needs_review").sort((a, b) => b.sif_score - a.sif_score);
  $("#review-rows").innerHTML = queue.map((report) => rowMarkup(report, "review")).join("") || emptyRow("The review queue is clear.", 5);
}
async function renderPatterns() {
  const result = await api("/api/patterns");
  const items = result.items;
  $("#pattern-grid").innerHTML = items.length ? items.map((item) => `<article class="pattern-item"><div class="pattern-top"><h2>${escapeHtml(item.name)}</h2><span class="pattern-count">${item.count}</span></div><div class="pattern-meta">${item.sites.map((site) => `<span class="tag">${escapeHtml(site)}</span>`).join("")}</div><div class="pattern-meta">${item.activities.map((activity) => `<span class="tag">${escapeHtml(activity)}</span>`).join("")}</div><div class="pattern-score">Mean demo screening score: ${Math.round(item.mean_screening_score * 100)} / 100</div><button class="text-button pattern-reports" data-report-ids="${escapeHtml(item.report_ids.join(","))}">Open ${item.count} reports →</button></article>`).join("") : `<div class="empty-state">No recurring extracted barrier-failure groups yet.</div>`;
  $$(".pattern-reports").forEach((button) => button.addEventListener("click", () => {
    switchView("reports");
    const ids = new Set(button.dataset.reportIds.split(","));
    $$("#report-rows tr[data-report-id]").forEach((row) => { row.hidden = !ids.has(row.dataset.reportId); });
    $("#report-total").textContent = `${ids.size} pattern reports`;
  }));
}
function renderRuleTable() {
  const rules = state.summary.life_saving_rules;
  const total = Math.max(1, state.summary.total_reports);
  $("#rule-rows").innerHTML = rules.map((item) => `<tr><td><strong>${escapeHtml(item.name)}</strong></td><td>${item.count}</td><td>${Math.round(item.count / total * 100)}%</td></tr>`).join("") || emptyRow("No matched rule evidence.", 3);
}
function fillFilters() {
  const options = [["#site-filter", state.filters.sites], ["#activity-filter", state.filters.activities], ["#type-filter", state.filters.report_types]];
  for (const [selector, values] of options) {
    const select = $(selector);
    const first = select.options[0].outerHTML;
    select.innerHTML = first + values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
  }
}
function renderAll() {
  renderMlStatus();
  renderMetrics();
  renderTrend();
  renderRankList();
  renderBars("#site-list", state.summary.site_density);
  renderBars("#activity-list", state.summary.activity_density);
  renderTables();
  renderRuleTable();
  fillFilters();
  renderPatterns().catch((error) => showToast(error.message));
}
function switchView(view) {
  state.currentView = view;
  $$(".view-panel").forEach((panel) => panel.classList.add("hidden"));
  $(`#${view}-view`).classList.remove("hidden");
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  const [title, kicker, description] = viewCopy[view];
  $("#current-section").textContent = title;
  $("#page-title").textContent = title;
  $("#page-kicker").textContent = kicker;
  $("#page-description").textContent = description;
  if (view === "reports") renderTables();
}
async function openReport(report) {
  try {
    report = await api(`/api/reports/${encodeURIComponent(report.report_id)}`);
  } catch (error) {
    showToast(error.message);
    return;
  }
  state.currentReport = report;
  $("#dialog-title").textContent = report.report_id;
  const factors = Object.entries(report.factors || {}).flatMap(([category, values]) => values.map((factor) => `<div class="factor-row"><span class="factor-category">${escapeHtml(category.replaceAll("_", " "))}</span><div><span class="factor-concept">${escapeHtml(factor.concept)}</span><span class="evidence-quote">“${escapeHtml(factor.evidence)}”</span></div></div>`)).join("");
  const rules = (report.rule_scores || []).filter((item) => item.evidence).map((item) => `<span class="rule-chip">${escapeHtml(item.rule)} · “${escapeHtml(item.evidence)}”</span>`).join("") || `<span class="empty-state">No rule phrase matched.</span>`;
  const history = (report.review_history || []).map((event) => `<div class="factor-row"><span class="factor-category">${escapeHtml(event.created_at || "Review")}</span><div><span class="factor-concept">${escapeHtml(event.reviewer)} · ${escapeHtml(event.new_label.replaceAll("_", " "))}</span><span class="evidence-quote">${escapeHtml(event.review_note || "No reviewer note")}</span></div></div>`).join("");
  $("#dialog-content").innerHTML = `
    <div class="detail-meta"><span>${escapeHtml(report.timestamp || "Date unknown")}</span><span>${escapeHtml(report.site)}</span><span>${escapeHtml(report.location)}</span><span>${escapeHtml(report.activity)}</span><span>${escapeHtml(report.report_type)}</span>${report.is_demo ? "<span class=\"status-pill unreviewed\">Demo record</span>" : ""}</div>
    <div class="detail-score"><strong>${Math.round(report.sif_score * 100)}<small>/100</small></strong><span><b>Illustrative screening score</b><br>${escapeHtml(report.score_notice)}</span></div>
    <h3 class="detail-section-title">Original narrative</h3><blockquote class="detail-narrative">${escapeHtml(report.raw_description)}</blockquote>
    <h3 class="detail-section-title">Extracted factors and source evidence</h3><div class="factor-list">${factors || `<div class="empty-state">No safety factors matched the current prototype vocabulary.</div>`}</div>
    <h3 class="detail-section-title">Life-Saving Rule suggestions · provisional</h3><div>${rules}</div>
    <h3 class="detail-section-title">Reviewed-label SIF model</h3><div class="ml-inference-result"><button class="button button-primary" id="run-model-button" ${state.mlStatus.prediction_ready ? "" : "disabled"}>Run trained model</button><span>${state.mlStatus.prediction_ready ? "Separate from the demo screening score above." : escapeHtml(state.mlStatus.message)}</span></div>
    ${history ? `<h3 class="detail-section-title">Review history</h3><div class="factor-list">${history}</div>` : ""}
    <h3 class="detail-section-title">HSE review</h3>
    <form class="review-form" id="review-form">
      <label>Decision<select name="human_label" required><option value="uncertain">Uncertain / needs discussion</option><option value="sif_potential">SIF potential</option><option value="not_sif_potential">Not SIF potential</option></select></label>
      <label>Reviewer note<textarea name="review_note" maxlength="2000" placeholder="Add a brief reason or correction"></textarea></label>
      <button class="button button-primary" type="submit">Save expert review</button>
    </form>`;
  if (report.human_label) $("select[name=human_label]").value = report.human_label;
  $("textarea[name=review_note]").value = report.review_note || "";
  $("#review-form").addEventListener("submit", saveReview);
  $("#run-model-button").addEventListener("click", runModelPrediction);
  $("#report-dialog").showModal();
}
async function runModelPrediction() {
  const button = $("#run-model-button");
  button.disabled = true;
  button.textContent = "Scoring…";
  try {
    const result = await api(`/api/ml/predict/${encodeURIComponent(state.currentReport.report_id)}`);
    const output = document.createElement("div");
    output.className = "trained-model-output";
    output.innerHTML = `<strong>${escapeHtml(result.prediction.replaceAll("_", " "))} · ${Math.round(result.model_score * 100)} / 100</strong><span>${escapeHtml(result.notice)}</span><small>${escapeHtml(result.model_version)}</small>`;
    button.closest(".ml-inference-result").replaceWith(output);
  } catch (error) {
    showToast(error.message);
    button.disabled = false;
    button.textContent = "Run trained model";
  }
}
async function saveReview(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  try {
    const updated = await api(`/api/reports/${encodeURIComponent(state.currentReport.report_id)}/review`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_status: "reviewed", human_label: form.get("human_label"), review_note: form.get("review_note"), reviewer: "local-user" })
    });
    state.currentReport = updated;
    $("#report-dialog").close();
    await loadData();
    showToast("HSE review saved with the original report retained.");
  } catch (error) { showToast(error.message); }
}
function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  $$('[data-navigate]').forEach((button) => button.addEventListener("click", () => switchView(button.dataset.navigate)));
  $(".notice-dismiss").addEventListener("click", () => $(".notice-banner").remove());
  $("#dialog-close").addEventListener("click", () => $("#report-dialog").close());
  $("#report-dialog").addEventListener("click", (event) => { if (event.target === $("#report-dialog")) $("#report-dialog").close(); });
  for (const selector of ["#search-input", "#site-filter", "#activity-filter", "#type-filter"]) $(selector).addEventListener("input", renderTables);
  document.body.addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-report-id]");
    if (!row || row.hidden) return;
    const report = state.reports.find((item) => item.report_id === row.dataset.reportId);
    if (report) openReport(report);
  });
  $("#csv-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      const result = await api("/api/import", { method: "POST", body: form });
      await loadData();
      showToast(`Imported ${result.created} reports; skipped ${result.skipped}.`);
    } catch (error) { showToast(error.message); }
    event.target.value = "";
  });
}

bindEvents();
loadData().catch((error) => {
  $("#updated-at").textContent = "API unavailable";
  $("#metric-grid").innerHTML = `<div class="empty-state">Could not load local API: ${escapeHtml(error.message)}. Start the server with the command in README.md.</div>`;
});
