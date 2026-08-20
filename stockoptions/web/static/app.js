// stockoptions dashboard frontend. Plain JS, no build step, no framework --
// every number rendered here comes straight from the /api/* endpoints,
// which are themselves thin wrappers around the same tested Python
// modules the CLI uses.

const state = { ticker: "AAPL", period: "1y" };

// ---------------- navigation ----------------

document.querySelectorAll(".side-link").forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    document.querySelectorAll(".side-link").forEach((l) => l.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
    link.classList.add("active");
    document.getElementById(link.dataset.section).classList.remove("hidden");
  });
});

// ---------------- fetch helper ----------------

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function apiPost(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function setStatus(message, isError = false) {
  const el = document.getElementById("topbar-status");
  el.textContent = message;
  el.classList.toggle("error", isError);
}

const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
const money = (x) => `$${x.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}`;

// Escapes any string interpolated into an innerHTML template below. Most
// strings rendered here are our own formatted numbers, but chart series
// names originate from the user-typed ticker box (see sanitizeTicker),
// so nothing gets into innerHTML unescaped even if a "ticker" ever
// contained HTML special characters.
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Tickers are letters/digits plus '.' and '-' (share classes like BRK.B,
// BF-B), uppercased, capped at a sane length. Applied once at the only
// place free text enters this app, so everything downstream (URLs built
// from it, chart series named after it) is already-clean.
function sanitizeTicker(raw) {
  return raw.toUpperCase().replace(/[^A-Z0-9.\-]/g, "").slice(0, 10);
}

// ---------------- SVG line chart (reusable, no dependencies) ----------------

function renderLineChart(container, series, opts = {}) {
  const width = container.clientWidth || 600;
  const height = container.clientHeight || 260;
  const padding = { top: 10, right: 14, bottom: 24, left: 54 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const allYs = series.flatMap((s) => s.points.map((p) => p.y));
  const yMin = Math.min(...allYs);
  const yMax = Math.max(...allYs);
  const yPad = (yMax - yMin) * 0.1 || Math.abs(yMax || 1) * 0.1 || 1;
  const y0 = yMin - yPad;
  const y1 = yMax + yPad;

  const n = series[0]?.points.length || 0;
  const xScale = (i) => padding.left + (n <= 1 ? innerW / 2 : (i / (n - 1)) * innerW);
  const yScale = (v) => padding.top + innerH - ((v - y0) / (y1 - y0)) * innerH;
  const yFmt = opts.yFormat || ((v) => v.toFixed(2));

  const parts = [];
  const gridLines = 4;
  for (let g = 0; g <= gridLines; g++) {
    const v = y0 + (g / gridLines) * (y1 - y0);
    const y = yScale(v);
    parts.push(`<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#232b25" stroke-width="1" />`);
    parts.push(`<text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#8b978f">${yFmt(v)}</text>`);
  }

  if (opts.zeroLine !== undefined && opts.zeroLine >= y0 && opts.zeroLine <= y1) {
    const y = yScale(opts.zeroLine);
    parts.push(`<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#8b978f" stroke-width="1.5" stroke-dasharray="4 3" />`);
  }

  for (const s of series) {
    const d = s.points.map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(i).toFixed(2)} ${yScale(p.y).toFixed(2)}`).join(" ");
    parts.push(`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.25" stroke-linejoin="round" stroke-linecap="round" />`);
  }

  if (opts.xLabels && opts.xLabels.length && n > 0) {
    const idxs = [0, Math.floor((n - 1) / 2), n - 1];
    for (const i of idxs) {
      if (i < 0 || i >= opts.xLabels.length) continue;
      const anchor = i === 0 ? "start" : i === n - 1 ? "end" : "middle";
      parts.push(`<text x="${xScale(i)}" y="${height - 6}" text-anchor="${anchor}" font-size="10" fill="#8b978f">${opts.xLabels[i]}</text>`);
    }
  }

  const hoverLineId = `hover-${Math.random().toString(36).slice(2)}`;
  parts.push(`<line id="${hoverLineId}" x1="0" y1="${padding.top}" x2="0" y2="${padding.top + innerH}" stroke="#eef2ef" stroke-width="1" opacity="0" />`);

  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${parts.join("")}</svg>`;

  const tooltip = document.createElement("div");
  tooltip.style.cssText =
    "position:absolute;pointer-events:none;background:#171d18;border:1px solid #232b25;border-radius:8px;padding:8px 10px;font-size:12px;display:none;white-space:nowrap;z-index:5;";
  container.style.position = "relative";
  container.appendChild(tooltip);
  const hoverLine = container.querySelector(`#${hoverLineId}`);

  container.addEventListener("mousemove", (e) => {
    if (n === 0) return;
    const rect = container.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * width;
    const i = Math.max(0, Math.min(n - 1, Math.round(((relX - padding.left) / innerW) * (n - 1))));
    hoverLine.setAttribute("x1", xScale(i));
    hoverLine.setAttribute("x2", xScale(i));
    hoverLine.setAttribute("opacity", "1");
    const lines = series.map((s) => `<div><span style="color:${s.color}">&#9679;</span> ${escapeHtml(s.name)}: ${escapeHtml(yFmt(s.points[i].y))}</div>`);
    const label = opts.xLabels ? opts.xLabels[i] : i;
    tooltip.innerHTML = `<div style="color:#8b978f;margin-bottom:4px;">${escapeHtml(label)}</div>${lines.join("")}`;
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(relX - padding.left > innerW / 2 ? relX - 130 : relX + 12, width - 140)}px`;
    tooltip.style.top = "8px";
  });
  container.addEventListener("mouseleave", () => {
    hoverLine.setAttribute("opacity", "0");
    tooltip.style.display = "none";
  });
}

// ---------------- overview ----------------

async function loadOverview(ticker) {
  const overview = await apiGet(`/api/overview/${ticker}`);
  document.getElementById("overview-title").textContent = `${overview.ticker} overview`;
  document.getElementById("stat-price").textContent = money(overview.price);
  document.getElementById("stat-iv").textContent = pct(overview.atmIv);
  document.getElementById("stat-hv").textContent = pct(overview.realizedVol30d);

  const readEl = document.getElementById("stat-read");
  const readCard = document.getElementById("stat-read-card");
  readCard.classList.remove("good", "bad", "neutral");
  readEl.textContent = overview.read;
  if (overview.read === "cheap") readCard.classList.add("neutral");
  else if (overview.read === "rich") readCard.classList.add("good");
  document.getElementById("stat-ratio").textContent = `IV/HV ${overview.ivHvRatio.toFixed(2)}`;

  document.getElementById("meta-rate").textContent = pct(overview.riskFreeRate, 2);
  document.getElementById("meta-div").textContent = pct(overview.dividendYield, 2);
  document.getElementById("meta-exp").textContent = overview.nearestExpiration;
}

async function loadPriceChart(ticker, period) {
  const rows = await apiGet(`/api/history/${ticker}?period=${period}`);
  renderLineChart(
    document.getElementById("price-chart"),
    [{ name: ticker, color: "#34d399", points: rows.map((r, i) => ({ x: i, y: r.close })) }],
    { xLabels: rows.map((r) => r.date), yFormat: (v) => `$${v.toFixed(0)}` }
  );
}

document.querySelectorAll("#period-chips .chip").forEach((chip) => {
  chip.addEventListener("click", async () => {
    document.querySelectorAll("#period-chips .chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    state.period = chip.dataset.period;
    await loadPriceChart(state.ticker, state.period);
  });
});

// ---------------- option chain ----------------

async function loadChainExpirations(ticker) {
  const expirations = await apiGet(`/api/expirations/${ticker}`);
  const select = document.getElementById("chain-expiration");
  select.innerHTML = expirations.map((e) => `<option value="${e}">${e}</option>`).join("");
  return expirations;
}

async function loadChainTable() {
  const expiration = document.getElementById("chain-expiration").value;
  if (!expiration) return;
  const type = document.querySelector("#chain [data-type].active")?.dataset.type || "call";
  const data = await apiGet(`/api/chain/${state.ticker}?expiration=${expiration}&type=${type}`);
  const tbody = document.getElementById("chain-tbody");
  const closestStrike = data.rows.reduce(
    (best, r) => (Math.abs(r.strike - data.spot) < Math.abs(best - data.spot) ? r.strike : best),
    data.rows[0]?.strike
  );
  tbody.innerHTML = data.rows
    .map(
      (r) => `<tr class="${r.strike === closestStrike ? "atm" : ""}">
      <td>${r.strike.toFixed(2)}</td><td>${r.lastPrice.toFixed(2)}</td><td>${pct(r.iv)}</td>
      <td>${r.delta.toFixed(3)}</td><td>${r.gamma.toFixed(4)}</td><td>${r.vega.toFixed(3)}</td><td>${r.theta.toFixed(2)}</td>
    </tr>`
    )
    .join("");
}

document.getElementById("chain-expiration").addEventListener("change", () => loadChainTable().catch((e) => setStatus(e.message, true)));
document.querySelectorAll("#chain [data-type]").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll("#chain [data-type]").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    loadChainTable().catch((e) => setStatus(e.message, true));
  });
});

// ---------------- strategy builder ----------------

const STRATEGY_FIELDS = {
  "iron-condor": [
    ["putLongStrike", "Put long strike", 90],
    ["putShortStrike", "Put short strike", 95],
    ["callShortStrike", "Call short strike", 105],
    ["callLongStrike", "Call long strike", 110],
    ["putLongPremium", "Put long premium", 1],
    ["putShortPremium", "Put short premium", 2],
    ["callShortPremium", "Call short premium", 2],
    ["callLongPremium", "Call long premium", 1],
  ],
  vertical: [
    ["longStrike", "Long strike", 100],
    ["shortStrike", "Short strike", 110],
    ["longPremium", "Long premium", 8],
    ["shortPremium", "Short premium", 3],
  ],
  strangle: [
    ["callStrike", "Call strike", 110],
    ["putStrike", "Put strike", 90],
    ["callPremium", "Call premium", 2],
    ["putPremium", "Put premium", 2],
  ],
};

let strategyKind = "iron-condor";

function renderStrategyForm() {
  const form = document.getElementById("strategy-form");
  const fields = STRATEGY_FIELDS[strategyKind];
  form.innerHTML =
    fields.map(([name, label, def]) => `<label>${label}<input name="${name}" type="number" step="0.01" value="${def}" required /></label>`).join("") +
    `<div class="form-actions"><button type="submit" class="btn btn-primary">Compute payoff</button></div>`;
}

document.querySelectorAll("#strategy-kind-chips .chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll("#strategy-kind-chips .chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    strategyKind = chip.dataset.kind;
    renderStrategyForm();
  });
});

document.getElementById("strategy-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const payload = { kind: strategyKind };
  if (strategyKind === "vertical") payload.optionType = "call";
  for (const [key, value] of formData.entries()) payload[key] = parseFloat(value);

  try {
    const result = await apiPost("/api/strategy", payload);
    document.getElementById("strat-premium").textContent = `${money(Math.abs(result.netPremium))} ${result.netPremium >= 0 ? "(debit)" : "(credit)"}`;
    document.getElementById("strat-profit").textContent = result.maxProfitUnlimited ? "Unlimited" : money(result.maxProfit);
    document.getElementById("strat-loss").textContent = result.maxLossUnlimited ? "Unlimited" : money(result.maxLoss);
    document.getElementById("strat-breakeven").textContent = result.breakevens.length ? result.breakevens.map((b) => b.toFixed(2)).join(", ") : "none";

    renderLineChart(document.getElementById("strategy-chart"), [{ name: "P&L", color: "#34d399", points: result.curve.map((p) => ({ x: p.s, y: p.pnl })) }], {
      xLabels: result.curve.map((p) => p.s.toFixed(0)),
      yFormat: (v) => `$${v.toFixed(0)}`,
      zeroLine: 0,
    });
  } catch (err) {
    setStatus(err.message, true);
  }
});

// ---------------- backtest ----------------

document.getElementById("backtest-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const period = document.getElementById("backtest-period").value;
  const horizon = document.getElementById("backtest-horizon").value;
  try {
    setStatus(`Running backtest for ${state.ticker}...`);
    const result = await apiGet(`/api/backtest/${state.ticker}?period=${period}&horizon=${horizon}`);
    setStatus("");
    document.getElementById("backtest-results").classList.remove("hidden");
    document.getElementById("bt-accuracy").textContent = pct(result.accuracy);
    document.getElementById("bt-baseline").textContent = pct(result.majorityBaseline);
    const beatsCard = document.getElementById("bt-beats-card");
    beatsCard.classList.remove("good", "bad");
    beatsCard.classList.add(result.beatsBaseline ? "good" : "bad");
    document.getElementById("bt-beats").textContent = result.beatsBaseline ? "Yes" : "No";
    document.getElementById("bt-samples").textContent = `${result.nTrain} / ${result.nTest}`;

    renderLineChart(
      document.getElementById("backtest-chart"),
      [
        { name: "Signal strategy", color: "#34d399", points: result.strategyCurve.map((v, i) => ({ x: i, y: v })) },
        { name: "Buy & hold", color: "#60a5fa", points: result.buyHoldCurve.map((v, i) => ({ x: i, y: v })) },
      ],
      { xLabels: result.dates, yFormat: (v) => v.toFixed(2) }
    );
  } catch (err) {
    setStatus(err.message, true);
  }
});

// ---------------- ticker load ----------------

document.getElementById("ticker-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const ticker = sanitizeTicker(document.getElementById("ticker-input").value.trim());
  if (!ticker) return;
  document.getElementById("ticker-input").value = ticker;
  await loadTicker(ticker);
});

async function loadTicker(ticker) {
  state.ticker = ticker;
  const btn = document.querySelector("#ticker-form button");
  btn.disabled = true;
  setStatus(`Loading ${ticker}...`);
  try {
    await Promise.all([loadOverview(ticker), loadPriceChart(ticker, state.period)]);
    await loadChainExpirations(ticker);
    await loadChainTable();
    setStatus(`Loaded ${ticker}`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    btn.disabled = false;
  }
}

// ---------------- init ----------------

renderStrategyForm();
loadTicker(state.ticker);
