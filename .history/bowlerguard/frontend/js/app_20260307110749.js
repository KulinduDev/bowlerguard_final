const API = "http://localhost:5000";

const featureCols = [
  "deliveries_match","deliveries_7d","deliveries_28d","acwr_std","days_since_prev",
  "match_day","avg_temp_c","avg_humidity_pct","precip_mm","esi_norm",
  "age_years","inferred_fielding_time_minutes"
];

const defaults = {
  deliveries_match: 72,
  deliveries_7d: 240,
  deliveries_28d: 700,
  acwr_std: 1.37,
  days_since_prev: 1,
  match_day: 3,
  avg_temp_c: 31,
  avg_humidity_pct: 78,
  precip_mm: 3,
  esi_norm: 0.65,
  age_years: 28,
  inferred_fielding_time_minutes: 160
};

const pretty = {
  deliveries_match: "Deliveries (match)",
  deliveries_7d: "Deliveries (last 7 days)",
  deliveries_28d: "Deliveries (last 28 days)",
  acwr_std: "ACWR (standardised)",
  days_since_prev: "Days since previous match",
  match_day: "Match day (1–5)",
  avg_temp_c: "Average temperature (°C)",
  avg_humidity_pct: "Average humidity (%)",
  precip_mm: "Precipitation (mm)",
  esi_norm: "ESI (0–1)",
  age_years: "Age (years)",
  inferred_fielding_time_minutes: "Fielding time (minutes)"
};

const hints = {
  deliveries_match: "Total deliveries bowled in the match.",
  deliveries_7d: "Last 7 days total workload (deliveries).",
  deliveries_28d: "Last 28 days total workload (deliveries).",
  acwr_std: "Acute:chronic workload ratio.",
  days_since_prev: "Rest between matches (days).",
  match_day: "Test match day number (1–5).",
  esi_norm: "Environmental Stress Index (0–1).",
  inferred_fielding_time_minutes: "Approx time fielding (minutes)."
};

function createForm() {
  const container = document.getElementById("form");
  container.innerHTML = "";

  featureCols.forEach((c) => {
    const div = document.createElement("div");
    div.className = "row";
    div.innerHTML = `
      <label for="${c}">${pretty[c] ?? c}</label>
      <input id="${c}" value="${defaults[c] ?? 0}" inputmode="decimal">
      <div class="hint">${hints[c] ?? ""}</div>
    `;
    container.appendChild(div);
  });
}

function buildPayload() {
  const payload = {};
  featureCols.forEach((c) => {
    payload[c] = Number(document.getElementById(c).value);
  });
  return payload;
}

function showRaw(obj) {
  document.getElementById("out").textContent = JSON.stringify(obj, null, 2);
}

function showLoginView() {
  document.getElementById("loginView").classList.remove("hidden");
  document.getElementById("appView").classList.add("hidden");
}

function showAppView(username, role) {
  document.getElementById("loginView").classList.add("hidden");
  document.getElementById("appView").classList.remove("hidden");
  document.getElementById("welcomeText").innerText = `Welcome, ${username}`;
  document.getElementById("roleText").innerText = `Role: ${role}`;

  const explainBtn = document.getElementById("explainBtn");
  if (["admin", "coach", "analyst"].includes(role)) {
    explainBtn.classList.remove("hidden");
  } else {
    explainBtn.classList.add("hidden");
  }
}

function renderStandardOutput(data) {
  const body = document.getElementById("standardOutputBody");
  body.innerHTML = "";

  Object.entries(data).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "json-row";
    row.innerHTML = `
      <span class="json-key">${key}</span>
      <span class="json-value">${value}</span>
    `;
    body.appendChild(row);
  });
}

function setRiskBadge(label) {
  const badge = document.getElementById("riskBadge");
  badge.classList.remove("low", "med", "high");

  const lower = (label || "").toLowerCase();
  if (lower === "low") badge.classList.add("low");
  if (lower === "medium") badge.classList.add("med");
  if (lower === "high") badge.classList.add("high");
}

function resetOutputs() {
  document.getElementById("riskLabelText").innerText = "—";
  document.getElementById("fatigueText").innerText = "—";

  document.getElementById("pLowTxt").innerText = "—";
  document.getElementById("pMedTxt").innerText = "—";
  document.getElementById("pHighTxt").innerText = "—";

  document.getElementById("pLow").style.width = "0%";
  document.getElementById("pMed").style.width = "0%";
  document.getElementById("pHigh").style.width = "0%";

  document.getElementById("riskBadge").classList.remove("low", "med", "high");
  document.getElementById("contributorsCard").classList.add("hidden");
  document.getElementById("contributorsBody").innerHTML = "";

  renderStandardOutput({ Status: "Awaiting action" });
  showRaw({});
}

function renderPredictOutput(obj) {
  showRaw(obj);

  if (obj.error) {
    resetOutputs();
    renderStandardOutput({
      Status: "Request failed",
      Error: obj.error
    });
    return;
  }

  const label = obj.predicted_label || "—";
  const fatigue = Number(obj.fatigue_score);

  document.getElementById("riskLabelText").innerText = label;
  setRiskBadge(label);

  document.getElementById("fatigueText").innerText =
    Number.isFinite(fatigue) ? fatigue.toFixed(1) : "—";

  const probs = obj.probabilities || {};
  const pLow = (probs.Low ?? 0) * 100;
  const pMed = (probs.Medium ?? 0) * 100;
  const pHigh = (probs.High ?? 0) * 100;

  document.getElementById("pLow").style.width = `${pLow.toFixed(0)}%`;
  document.getElementById("pMed").style.width = `${pMed.toFixed(0)}%`;
  document.getElementById("pHigh").style.width = `${pHigh.toFixed(0)}%`;

  document.getElementById("pLowTxt").innerText = `${pLow.toFixed(0)}%`;
  document.getElementById("pMedTxt").innerText = `${pMed.toFixed(0)}%`;
  document.getElementById("pHighTxt").innerText = `${pHigh.toFixed(0)}%`;

  renderStandardOutput({
    "Prediction Type": "Risk Prediction",
    "Predicted Label": label,
    "Fatigue Score": Number.isFinite(fatigue) ? fatigue.toFixed(2) : "—",
    "Low Probability": `${pLow.toFixed(2)}%`,
    "Medium Probability": `${pMed.toFixed(2)}%`,
    "High Probability": `${pHigh.toFixed(2)}%`
  });

  document.getElementById("contributorsCard").classList.add("hidden");
  document.getElementById("contributorsBody").innerHTML = "";
}

function renderExplainOutput(obj) {
  showRaw(obj);

  if (obj.error) {
    renderStandardOutput({
      Status: "Explanation failed",
      Error: obj.error
    });
    document.getElementById("contributorsCard").classList.add("hidden");
    document.getElementById("contributorsBody").innerHTML = "";
    return;
  }

  const label = obj.predicted_label || "—";
  document.getElementById("riskLabelText").innerText = label;
  setRiskBadge(label);

  const probs = obj.predicted_probabilities || {};
  const pLow = (probs.Low ?? 0) * 100;
  const pMed = (probs.Medium ?? 0) * 100;
  const pHigh = (probs.High ?? 0) * 100;

  document.getElementById("pLow").style.width = `${pLow.toFixed(0)}%`;
  document.getElementById("pMed").style.width = `${pMed.toFixed(0)}%`;
  document.getElementById("pHigh").style.width = `${pHigh.toFixed(0)}%`;

  document.getElementById("pLowTxt").innerText = `${pLow.toFixed(0)}%`;
  document.getElementById("pMedTxt").innerText = `${pMed.toFixed(0)}%`;
  document.getElementById("pHighTxt").innerText = `${pHigh.toFixed(0)}%`;

  renderStandardOutput({
    "Prediction Type": "SHAP Explanation",
    "Predicted Label": label,
    "Low Probability": `${pLow.toFixed(2)}%`,
    "Medium Probability": `${pMed.toFixed(2)}%`,
    "High Probability": `${pHigh.toFixed(2)}%`,
    "Top Contributors Returned": `${(obj.top_contributors || []).length}`
  });

  const tbody = document.getElementById("contributorsBody");
  tbody.innerHTML = "";

  (obj.top_contributors || []).forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.feature}</td>
      <td>${Number(item.shap_value).toFixed(4)}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById("contributorsCard").classList.remove("hidden");
}

async function checkSession() {
  try {
    const r = await fetch(`${API}/session-status`, {
      credentials: "include"
    });
    const data = await r.json();

    if (data.logged_in) {
      showAppView(data.username, data.role);
    } else {
      showLoginView();
    }
  } catch (error) {
    showLoginView();
    const status = document.getElementById("loginStatus");
    status.innerText = "Unable to connect to backend.";
    status.className = "status error-text";
  }
}

async function login() {
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value.trim();

  const status = document.getElementById("loginStatus");
  status.className = "status";
  status.innerText = "";

  try {
    const r = await fetch(`${API}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ username, password })
    });

    const data = await r.json();

    if (!r.ok) {
      status.innerText = data.error || "Login failed";
      status.className = "status error-text";
      return;
    }

    status.innerText = `${data.message} | Role: ${data.role}`;
    status.className = "status success-text";

    showAppView(data.username, data.role);
    resetOutputs();
  } catch (error) {
    status.innerText = "Unable to connect to backend.";
    status.className = "status error-text";
  }
}

async function logout() {
  try {
    await fetch(`${API}/logout`, {
      method: "POST",
      credentials: "include"
    });
  } catch (error) {}

  showLoginView();
  resetOutputs();

  const status = document.getElementById("loginStatus");
  status.innerText = "Logged out successfully.";
  status.className = "status success-text";

  document.getElementById("username").value = "";
  document.getElementById("password").value = "";
}

async function health() {
  const r = await fetch(`${API}/health`, {
    credentials: "include"
  });
  const data = await r.json();

  showRaw(data);
  renderStandardOutput({
    "API Status": data.status || "unknown"
  });
}

async function predict() {
  const payload = buildPayload();

  const r = await fetch(`${API}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(payload)
  });

  const data = await r.json();
  renderPredictOutput(data);
}

async function explain() {
  const payload = buildPayload();

  const r = await fetch(`${API}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(payload)
  });

  const data = await r.json();
  renderExplainOutput(data);
}

function bindEvents() {
  document.getElementById("loginBtn").addEventListener("click", login);
  document.getElementById("logoutBtn").addEventListener("click", logout);
  document.getElementById("healthBtn").addEventListener("click", health);
  document.getElementById("predictBtn").addEventListener("click", predict);
  document.getElementById("explainBtn").addEventListener("click", explain);

  document.getElementById("password").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      login();
    }
  });
}

function init() {
  createForm();
  bindEvents();
  resetOutputs();
  checkSession();
}

document.addEventListener("DOMContentLoaded", init);