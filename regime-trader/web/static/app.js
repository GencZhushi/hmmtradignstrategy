// Minimal dashboard client for Regime Trader.
// Communicates with the FastAPI platform over REST + SSE.
const API_BASE = "";

const state = {
  token: sessionStorage.getItem("rt_token") || null,
  role: sessionStorage.getItem("rt_role") || null,
  username: sessionStorage.getItem("rt_username") || null,
  activeTab: "overview",
  eventSource: null,
};

function authHeaders(extra = {}) {
  const headers = { "Content-Type": "application/json", ...extra };
  if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
  return headers;
}

async function api(path, { method = "GET", body = undefined, headers = {} } = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: authHeaders(headers),
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  if (response.status === 204) return null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json();
  return response.text();
}

function setText(selector, value) {
  const el = document.querySelector(selector);
  if (el) el.textContent = value;
}

function renderOverview(data) {
  const { health, portfolio, regime, freshness } = data;
  setText('[data-role="engine-status"]', `${health.trading_mode.toUpperCase()} - ${health.dry_run ? "dry-run" : "live-exec"} - model ${health.active_model || "unset"}`);
  setText('[data-role="regime-name"]', regime.regime_name || "-");
  setText('[data-role="regime-probability"]', regime.regime_name ? `p=${(regime.probability || 0).toFixed(2)} - ${regime.consecutive_bars} bars` : "waiting for regime");
  setText('[data-role="equity"]', portfolio.equity.toLocaleString(undefined, { style: "currency", currency: "USD" }));
  setText('[data-role="daily-pnl"]', `Daily PnL ${portfolio.daily_pnl >= 0 ? "+" : ""}${portfolio.daily_pnl.toFixed(2)}`);
  setText('[data-role="exposure"]', `${(portfolio.total_exposure_pct * 100).toFixed(1)}%`);
  setText('[data-role="leverage"]', `peak eq ${portfolio.peak_equity.toLocaleString()}`);
  setText('[data-role="breaker"]', portfolio.breaker_state.toUpperCase());
  setText('[data-role="stale-flag"]', freshness.stale_data_blocked ? "DATA STALE" : "fresh");

  const sectorList = document.getElementById("sector-list");
  sectorList.innerHTML = "";
  const entries = Object.entries(portfolio.sector_exposure || {});
  if (!entries.length) {
    sectorList.innerHTML = "<li>No open positions</li>";
  } else {
    for (const [sector, weight] of entries) {
      const li = document.createElement("li");
      li.textContent = `${sector}: ${(weight * 100).toFixed(1)}%`;
      sectorList.appendChild(li);
    }
  }
}

function renderPositions(portfolio) {
  const tbody = document.getElementById("positions-body");
  tbody.innerHTML = "";
  if (!portfolio.positions.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="8" class="py-3 text-slate-500">No open positions</td>';
    tbody.appendChild(row);
    return;
  }
  for (const pos of portfolio.positions) {
    const row = document.createElement("tr");
    row.className = "border-t border-slate-800";
    row.innerHTML = `
      <td class="py-2 font-semibold">${pos.symbol}</td>
      <td>${pos.quantity.toFixed(2)}</td>
      <td>${pos.avg_entry_price.toFixed(2)}</td>
      <td>${pos.current_price.toFixed(2)}</td>
      <td>${pos.stop_price ?? "-"}</td>
      <td>${pos.regime_at_entry ?? "-"}</td>
      <td class="${pos.unrealized_pnl >= 0 ? "text-emerald-400" : "text-rose-400"}">${pos.unrealized_pnl.toFixed(2)}</td>
      <td><button data-close-symbol="${pos.symbol}" class="text-xs text-rose-400 hover:text-rose-300">close</button></td>
    `;
    tbody.appendChild(row);
  }
}

function renderSignals(signals) {
  const list = document.getElementById("signals-list");
  list.innerHTML = "";
  if (!signals.length) {
    list.innerHTML = "<li class=\"text-slate-500\">No recent signals</li>";
    return;
  }
  for (const signal of signals.slice().reverse()) {
    const li = document.createElement("li");
    li.className = "p-3 bg-slate-800/60 rounded";
    li.innerHTML = `<div class="flex justify-between"><span class="font-semibold">${signal.symbol}</span><span class="text-xs text-slate-400">${signal.strategy_name || "-"}</span></div>
      <div class="text-xs text-slate-300">${signal.direction || "?"} alloc=${((signal.target_allocation_pct || 0) * 100).toFixed(0)}% lev=${(signal.leverage || 1).toFixed(2)}x</div>
      <div class="text-xs text-slate-400">${(signal.reasoning || []).join(" - ")}</div>`;
    list.appendChild(li);
  }
}

function renderApprovals(approvals) {
  const list = document.getElementById("approvals-list");
  list.innerHTML = "";
  if (!approvals.length) {
    list.innerHTML = "<li>No pending approvals</li>";
    return;
  }
  for (const approval of approvals) {
    const li = document.createElement("li");
    li.className = "p-3 bg-slate-800/60 rounded flex justify-between items-center";
    li.innerHTML = `<div>
        <div class="font-semibold">${approval.intent_id}</div>
        <div class="text-xs text-slate-400">requested_by=${approval.requested_by} (${approval.requested_by_type})</div>
      </div>
      <div class="flex gap-2 text-xs">
        <button data-action="approve" data-id="${approval.approval_id}" class="btn-primary">Approve</button>
        <button data-action="reject" data-id="${approval.approval_id}" class="px-3 py-1 rounded bg-rose-500 text-slate-950 font-semibold">Reject</button>
      </div>`;
    list.appendChild(li);
  }
}

function renderAudit(events) {
  const list = document.getElementById("audit-list");
  list.innerHTML = "";
  for (const event of events.slice(0, 100)) {
    const li = document.createElement("li");
    li.textContent = `${event.timestamp} ${event.actor}/${event.actor_type} ${event.action} -> ${event.resource_type}:${event.resource_id} (${event.reason || ""})`;
    list.appendChild(li);
  }
}

function renderSettings(info, governance) {
  const grid = document.getElementById("settings-grid");
  grid.innerHTML = "";
  const entries = [
    ["Trading mode", info.trading_mode],
    ["Execution enabled", info.execution_enabled],
    ["Approval mode", info.approval_mode],
    ["Role", state.role || "anonymous"],
  ];
  for (const [key, value] of entries) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    dt.className = "text-slate-400";
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    grid.appendChild(dt);
    grid.appendChild(dd);
  }

  const governanceBox = document.getElementById("governance");
  governanceBox.innerHTML = "";
  if (!governance.active_model_version) {
    governanceBox.textContent = "No active model promoted yet";
  } else {
    governanceBox.innerHTML = `
      <p>Active: <strong>${governance.active_model_version}</strong></p>
      <p>Fallback: ${governance.fallback_model_version || "-"}</p>
      <p>Candidates: ${governance.candidates.length}</p>
    `;
  }
}

async function refreshOverview() {
  try {
    const [health, portfolio, regime, freshness, signals, approvals, audit, info, governance] = await Promise.all([
      api("/health"),
      api("/portfolio"),
      api("/regime/current"),
      api("/freshness"),
      api("/signals/latest").catch(() => []),
      api("/approvals/pending").catch(() => []),
      api("/audit/logs?limit=50").catch(() => []),
      api("/info"),
      api("/regime/model"),
    ]);
    renderOverview({ health, portfolio, regime, freshness });
    renderPositions(portfolio);
    renderSignals(signals || []);
    renderApprovals(approvals || []);
    renderAudit(audit || []);
    renderSettings(info, governance);
  } catch (err) {
    console.warn("dashboard refresh failed", err);
  }
}

function activateTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.panel !== tab);
  });
}

function connectEvents() {
  if (state.eventSource) state.eventSource.close();
  const source = new EventSource(`${API_BASE}/events/stream`);
  source.onmessage = (msg) => {
    if (!msg.data) return;
    try {
      const payload = JSON.parse(msg.data);
      console.debug("event", payload);
      refreshOverview();
    } catch (err) {
      // ignore
    }
  };
  source.onerror = () => {
    console.warn("SSE connection lost - retrying in 5s");
    source.close();
    setTimeout(connectEvents, 5000);
  };
  state.eventSource = source;
}

function setAuthUi() {
  const loginForm = document.getElementById("login-form");
  const authIdentity = document.getElementById("auth-identity");
  if (state.token) {
    loginForm.classList.add("hidden");
    authIdentity.classList.remove("hidden");
    authIdentity.textContent = `${state.username} (${state.role})`;
  } else {
    loginForm.classList.remove("hidden");
    authIdentity.classList.add("hidden");
  }
}

async function handleLogin(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  try {
    const data = await api("/auth/login", {
      method: "POST",
      body: { username: form.get("username"), password: form.get("password") },
    });
    state.token = data.access_token;
    state.role = data.role;
    state.username = data.username;
    sessionStorage.setItem("rt_token", data.access_token);
    sessionStorage.setItem("rt_role", data.role);
    sessionStorage.setItem("rt_username", data.username);
    setAuthUi();
    refreshOverview();
  } catch (err) {
    alert(`Login failed: ${err.message}`);
  }
}

async function handleApprovalClick(event) {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const approvalId = target.dataset.id;
  const action = target.dataset.action;
  const endpoint = action === "approve" ? "/approvals/approve" : "/approvals/reject";
  try {
    await api(endpoint, { method: "POST", body: { approval_id: approvalId, reason: "dashboard" } });
    refreshOverview();
  } catch (err) {
    alert(`Approval action failed: ${err.message}`);
  }
}

async function handlePreviewSubmit(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = {
    symbol: form.get("symbol"),
    direction: form.get("direction"),
    allocation_pct: Number(form.get("allocation_pct")),
    thesis: form.get("thesis") || "",
  };
  try {
    const plan = await api("/orders/preview", { method: "POST", body: payload });
    alert(`Preview ${plan.status}: ${plan.rejection_reason || "ok"}`);
    refreshOverview();
  } catch (err) {
    alert(`Preview failed: ${err.message}`);
  }
}

async function handleCloseClick(event) {
  const target = event.target.closest("[data-close-symbol]");
  if (!target) return;
  const symbol = target.dataset.closeSymbol;
  if (!confirm(`Close position in ${symbol}?`)) return;
  try {
    await api(`/positions/close?symbol=${encodeURIComponent(symbol)}`, { method: "POST" });
    refreshOverview();
  } catch (err) {
    alert(`Close failed: ${err.message}`);
  }
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });
  activateTab("overview");
}

document.addEventListener("DOMContentLoaded", () => {
  setAuthUi();
  initTabs();
  refreshOverview();
  connectEvents();
  document.getElementById("login-form").addEventListener("submit", handleLogin);
  document.addEventListener("click", handleApprovalClick);
  document.getElementById("preview-form").addEventListener("submit", handlePreviewSubmit);
  document.addEventListener("click", handleCloseClick);
  setInterval(refreshOverview, 15000);
});
