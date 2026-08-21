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

const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Animates a stat value counting from its current displayed number up (or
// down) to a new one -- feedback that this specific number just changed,
// not a decorative flourish. Skips straight to the final text if the user
// has asked for reduced motion, or if the value isn't actually numeric
// (e.g. "Yes"/"No" or a strike list), so it's a safe drop-in for setting
// any stat-value's text.
function setStatAnimated(el, targetText, { from = null, duration = 500 } = {}) {
  const numeric = targetText.match(/^(\$?)(-?[\d,]+\.?\d*)(%?)$/);
  if (!numeric || prefersReducedMotion()) {
    el.textContent = targetText;
    return;
  }
  const [, prefix, numStr, suffix] = numeric;
  const to = parseFloat(numStr.replace(/,/g, ""));
  const startValue = from !== null && Number.isFinite(from) ? from : 0;
  const decimals = (numStr.split(".")[1] || "").length;
  const start = performance.now();

  function tick(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3); // ease-out
    const value = startValue + (to - startValue) * eased;
    el.textContent = `${prefix}${value.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`;
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// Escapes any string interpolated into an innerHTML template below. Most
// strings rendered here are our own formatted numbers, but chart series
// names originate from the user-typed ticker box (see sanitizeTicker),
// so nothing gets into innerHTML unescaped even if a "ticker" ever
// contained HTML special characters.
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Guards every href built from externally-sourced text (news article
// links, social post links) before it reaches the DOM -- unlike the
// ticker box, this text comes from third-party feeds/unofficial scrapers
// we don't control, so a "javascript:" or "data:" URL slipping through
// isn't a hypothetical. escapeHtml() alone only protects against markup
// injection, not a malicious href scheme, so both are needed here.
function safeUrl(url) {
  try {
    const parsed = new URL(url, window.location.href);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : "#";
  } catch {
    return "#";
  }
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

  // Draw-in: the data line sweeps from left to right instead of just
  // appearing -- orients the eye to "this is a series over time," and
  // doubles as free confirmation that new data actually replaced the old
  // chart rather than a stale render sitting there. Off entirely under
  // reduced-motion; each series is staggered slightly so overlapping
  // lines (strategy vs. buy & hold) read as distinct sweeps.
  if (!prefersReducedMotion()) {
    container.querySelectorAll("svg path").forEach((path, idx) => {
      const length = path.getTotalLength();
      path.style.strokeDasharray = `${length}`;
      path.style.strokeDashoffset = `${length}`;
      path.getBoundingClientRect(); // force reflow so the transition below actually animates
      path.style.transition = `stroke-dashoffset 0.7s ease-out ${idx * 0.1}s`;
      requestAnimationFrame(() => {
        path.style.strokeDashoffset = "0";
      });
    });
  }

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

// ---------------- forecast cone chart ----------------
// A dedicated renderer (not renderLineChart above) because the shape is
// different in kind, not just data: a trailing price line meeting a
// widening probability band at "today," rather than N parallel series
// over the same x-range. The band is the whole point -- it's what makes
// this a "here's a plausible range, with explicitly shown uncertainty"
// chart instead of a single confident-looking line pretending to know
// the future.

function renderForecastChart(container, historyRows, cone, S) {
  const width = container.clientWidth || 600;
  const height = container.clientHeight || 280;
  const padding = { top: 10, right: 14, bottom: 24, left: 54 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const histN = historyRows.length;
  const todayIdx = histN - 1;
  const totalN = histN + cone.length;

  const allYs = [...historyRows.map((r) => r.close), ...cone.map((p) => p.lower1), ...cone.map((p) => p.upper1)];
  const yMin = Math.min(...allYs);
  const yMax = Math.max(...allYs);
  const yPad = (yMax - yMin) * 0.08 || Math.abs(S) * 0.05 || 1;
  const y0 = yMin - yPad;
  const y1 = yMax + yPad;

  const xScale = (i) => padding.left + (totalN <= 1 ? innerW / 2 : (i / (totalN - 1)) * innerW);
  const yScale = (v) => padding.top + innerH - ((v - y0) / (y1 - y0)) * innerH;
  const coneX = (day) => xScale(todayIdx + day);

  const parts = [];
  const gridLines = 4;
  for (let g = 0; g <= gridLines; g++) {
    const v = y0 + (g / gridLines) * (y1 - y0);
    const y = yScale(v);
    parts.push(`<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#232b25" stroke-width="1" />`);
    parts.push(`<text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#8b978f">$${v.toFixed(0)}</text>`);
  }

  function band(upperKey, lowerKey) {
    const start = `${xScale(todayIdx).toFixed(2)},${yScale(S).toFixed(2)}`;
    const upper = cone.map((p) => `${coneX(p.day).toFixed(2)},${yScale(p[upperKey]).toFixed(2)}`).join(" ");
    const lower = cone
      .slice()
      .reverse()
      .map((p) => `${coneX(p.day).toFixed(2)},${yScale(p[lowerKey]).toFixed(2)}`)
      .join(" ");
    return `${start} ${upper} ${lower} ${start}`;
  }
  parts.push(`<polygon points="${band("upper1", "lower1")}" fill="#34d399" opacity="0.2" />`);

  parts.push(
    `<line x1="${xScale(todayIdx)}" y1="${yScale(S)}" x2="${coneX(cone[cone.length - 1].day)}" y2="${yScale(S)}" stroke="#8b978f" stroke-width="1.25" stroke-dasharray="4 3" />`
  );
  parts.push(
    `<line x1="${xScale(todayIdx)}" y1="${padding.top}" x2="${xScale(todayIdx)}" y2="${padding.top + innerH}" stroke="#eef2ef" stroke-width="1" stroke-dasharray="2 2" opacity="0.35" />`
  );
  parts.push(`<text x="${xScale(todayIdx) + 4}" y="${padding.top + 12}" font-size="10" fill="#8b978f">today</text>`);

  const histPath = historyRows.map((r, i) => `${i === 0 ? "M" : "L"} ${xScale(i).toFixed(2)} ${yScale(r.close).toFixed(2)}`).join(" ");
  parts.push(`<path d="${histPath}" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />`);

  const hoverLineId = `hover-${Math.random().toString(36).slice(2)}`;
  parts.push(`<line id="${hoverLineId}" x1="0" y1="${padding.top}" x2="0" y2="${padding.top + innerH}" stroke="#eef2ef" stroke-width="1" opacity="0" />`);

  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${parts.join("")}</svg>`;

  if (!prefersReducedMotion()) {
    const historyPath = container.querySelector("svg path");
    if (historyPath) {
      const length = historyPath.getTotalLength();
      historyPath.style.strokeDasharray = `${length}`;
      historyPath.style.strokeDashoffset = `${length}`;
      historyPath.getBoundingClientRect();
      historyPath.style.transition = "stroke-dashoffset 0.7s ease-out";
      requestAnimationFrame(() => {
        historyPath.style.strokeDashoffset = "0";
      });
    }
    const band = container.querySelector("svg polygon");
    if (band) {
      band.style.opacity = "0";
      band.style.transition = "opacity 0.4s ease-out 0.5s";
      requestAnimationFrame(() => {
        band.style.opacity = "0.2";
      });
    }
  }

  const tooltip = document.createElement("div");
  tooltip.style.cssText =
    "position:absolute;pointer-events:none;background:#171d18;border:1px solid #232b25;border-radius:8px;padding:8px 10px;font-size:12px;display:none;white-space:nowrap;z-index:5;";
  container.style.position = "relative";
  container.appendChild(tooltip);
  const hoverLine = container.querySelector(`#${hoverLineId}`);

  container.addEventListener("mousemove", (e) => {
    const rect = container.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * width;
    const i = Math.max(0, Math.min(totalN - 1, Math.round(((relX - padding.left) / innerW) * (totalN - 1))));
    hoverLine.setAttribute("x1", xScale(i));
    hoverLine.setAttribute("x2", xScale(i));
    hoverLine.setAttribute("opacity", "1");

    let label, lines;
    if (i <= todayIdx) {
      label = historyRows[i].date;
      lines = [`<div><span style="color:#60a5fa">&#9679;</span> Close: $${historyRows[i].close.toFixed(2)}</div>`];
    } else {
      const p = cone[i - todayIdx - 1];
      label = `${p.day} day${p.day === 1 ? "" : "s"} from now`;
      lines = [`<div><span style="color:#34d399">&#9679;</span> Likely range: $${p.lower1.toFixed(2)} &ndash; $${p.upper1.toFixed(2)}</div>`];
    }
    tooltip.innerHTML = `<div style="color:#8b978f;margin-bottom:4px;">${escapeHtml(label)}</div>${lines.join("")}`;
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(relX - padding.left > innerW / 2 ? relX - 150 : relX + 12, width - 160)}px`;
    tooltip.style.top = "8px";
  });
  container.addEventListener("mouseleave", () => {
    hoverLine.setAttribute("opacity", "0");
    tooltip.style.display = "none";
  });
}

// Plain-language reading of a probabilistic model output, alongside the
// raw number -- a bare "57.6%" doesn't on its own convey whether that's
// a strong signal or barely above a coin flip. Thresholds are informed
// by this project's own backtest results (accuracy rarely clears the
// high 60s%, and often sits close to 50%), not an arbitrary scale.
function confidenceLabel(probability) {
  if (probability >= 0.65) return { text: "High confidence", cls: "confidence-high" };
  if (probability >= 0.55) return { text: "Moderate confidence", cls: "confidence-moderate" };
  return { text: "Low confidence", cls: "confidence-low" };
}

// ---------------- predict ----------------

async function loadPredict(ticker) {
  const accountSize = parseFloat(document.getElementById("predict-account").value) || 10000;
  const riskPct = (parseFloat(document.getElementById("predict-risk").value) || 2) / 100;
  const horizon = parseInt(document.getElementById("predict-horizon").value, 10) || 35;
  const targetDelta = parseFloat(document.getElementById("predict-delta").value) || 0.35;

  const btn = document.querySelector("#predict-form button");
  btn.disabled = true;
  setStatus(`Building a trade recommendation for ${ticker}...`);
  try {
    const [rec, historyRows] = await Promise.all([
      apiGet(`/api/predict/${ticker}?account_size=${accountSize}&risk_pct=${riskPct}&horizon=${horizon}&delta=${targetDelta}`),
      apiGet(`/api/history/${ticker}?period=3mo`),
    ]);
    setStatus("");
    renderPredictResult(rec, historyRows);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    btn.disabled = false;
  }
}

function renderPredictResult(rec, historyRows) {
  document.getElementById("predict-results").classList.remove("hidden");

  const warningsEl = document.getElementById("predict-warnings");
  warningsEl.innerHTML = rec.warnings.length
    ? rec.warnings.map((w) => `<div class="predict-warning"><span class="predict-warning-icon">!</span><span>${escapeHtml(w)}</span></div>`).join("")
    : `<div class="predict-none-card">No sizing warnings triggered for this ticker/horizon -- still not a guarantee, just no flagged red light.</div>`;

  const isUp = rec.direction === "up";
  const conf = confidenceLabel(rec.liveProbability);
  const verdictCard = document.getElementById("verdict-card");
  verdictCard.classList.remove("bullish", "bearish");
  verdictCard.classList.add(isUp ? "bullish" : "bearish");
  document.getElementById("verdict-arrow").textContent = isUp ? "↑" : "↓";
  document.getElementById("verdict-headline").textContent = isUp ? "Bullish" : "Bearish";
  document.getElementById("verdict-confidence").innerHTML =
    `${(rec.liveProbability * 100).toFixed(1)}% model confidence<span class="confidence-chip ${conf.cls}">${escapeHtml(conf.text)}</span>`;

  const c = rec.contract;
  document.getElementById("verdict-plain").textContent =
    rec.sizing.contracts > 0
      ? `Suggested: buy ${rec.sizing.contracts} ${rec.ticker} $${c.strike.toFixed(0)} ${c.optionType}${rec.sizing.contracts === 1 ? "" : "s"}, expiring ${c.expiration} (~${money(rec.sizing.actualDollarRisk)} total).`
      : `The math doesn't support risking real money on this trade right now -- recommended size is 0 contracts. See "Show the math" below for why.`;

  document.getElementById("predict-accuracy").textContent = `${(rec.backtest.accuracy * 100).toFixed(1)}% / ${(rec.backtest.baseline * 100).toFixed(1)}%`;

  document.getElementById("predict-kelly").textContent = pct(rec.edge.kellyFraction);
  document.getElementById("predict-kelly-sub").textContent = `win rate ${pct(rec.edge.winRate)}, n=${rec.edge.sampleSize}`;

  document.getElementById("predict-size").textContent = `${rec.sizing.contracts} contract${rec.sizing.contracts === 1 ? "" : "s"}`;
  document.getElementById("predict-size-sub").textContent = `${money(rec.sizing.actualDollarRisk)} of ${money(rec.sizing.dollarBudget)} budget`;
  const sizeCard = document.getElementById("predict-size-card");
  sizeCard.classList.remove("good", "neutral");
  sizeCard.classList.add(rec.sizing.contracts > 0 ? "good" : "neutral");

  document.getElementById("predict-contract").innerHTML = `
    <div><span class="meta-label">Contract</span><span class="meta-value">${escapeHtml(rec.ticker)} ${escapeHtml(c.optionType)} $${c.strike.toFixed(2)}</span></div>
    <div><span class="meta-label">Expiration</span><span class="meta-value">${escapeHtml(c.expiration)} (${c.dte}d)</span></div>
    <div><span class="meta-label">Premium</span><span class="meta-value">$${c.premium.toFixed(2)} ($${(c.premium * 100).toFixed(2)}/contract)</span></div>
    <div><span class="meta-label">IV / delta</span><span class="meta-value">${pct(c.iv)} / ${c.delta.toFixed(3)}</span></div>
  `;

  renderForecastChart(document.getElementById("predict-chart"), historyRows, rec.cone, rec.price);
  document.getElementById("predict-asof").textContent = `Fetched ${new Date().toLocaleTimeString()} -- prices/quotes can be cached up to 15 minutes`;
}

document.getElementById("predict-form").addEventListener("submit", (e) => {
  e.preventDefault();
  loadPredict(state.ticker);
});

// ---------------- overview ----------------

async function loadOverview(ticker) {
  const statPrice = document.getElementById("stat-price");
  const statIv = document.getElementById("stat-iv");
  const statHv = document.getElementById("stat-hv");
  const loadingEls = [statPrice, statIv, statHv];
  loadingEls.forEach((el) => el.classList.add("is-loading"));

  let overview;
  try {
    overview = await apiGet(`/api/overview/${ticker}`);
  } finally {
    loadingEls.forEach((el) => el.classList.remove("is-loading"));
  }

  document.getElementById("overview-title").textContent = `${overview.ticker} overview`;
  setStatAnimated(statPrice, money(overview.price));
  document.getElementById("stat-price-sub").textContent = `as of ${new Date().toLocaleTimeString()}`;
  setStatAnimated(statIv, pct(overview.atmIv));
  setStatAnimated(statHv, pct(overview.realizedVol30d));

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

// ---------------- news & influencer watch ----------------

function formatPostTime(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

async function loadNews(ticker) {
  document.getElementById("news-ticker-title").textContent = `Recent news: ${ticker}`;
  const list = document.getElementById("news-list");
  try {
    const items = await apiGet(`/api/news/${ticker}?limit=8`);
    if (!items.length) {
      list.innerHTML = `<p class="fine-print">No recent news found for ${escapeHtml(ticker)}.</p>`;
      return;
    }
    list.innerHTML = items
      .map(
        (item) => `<a class="news-item" href="${safeUrl(item.url)}" target="_blank" rel="noopener">
          <div class="news-item-head"><span>${escapeHtml(item.publisher)}</span><span>${escapeHtml(formatPostTime(item.publishedAt))}</span></div>
          <div class="news-item-title">${escapeHtml(item.title)}</div>
          ${item.summary ? `<div class="news-item-summary">${escapeHtml(item.summary)}</div>` : ""}
        </a>`
      )
      .join("");
  } catch (err) {
    list.innerHTML = `<p class="fine-print">Couldn't load news: ${escapeHtml(err.message)}</p>`;
  }
}

function renderTopComments(comments) {
  if (!comments || !comments.length) return "";
  const rows = comments
    .map(
      (c) => `<div class="influencer-comment">
        <span class="influencer-comment-meta">@${escapeHtml(c.author)} &middot; ${c.favouritesCount.toLocaleString()} likes</span>
        <a href="${safeUrl(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.text || "(no text)")}</a>
      </div>`
    )
    .join("");
  return `<div class="influencer-comments"><div class="influencer-comments-label">Top comments</div>${rows}</div>`;
}

async function loadInfluencers() {
  const container = document.getElementById("influencer-list");
  try {
    const entries = await apiGet(`/api/influencers?limit=12`);
    container.innerHTML = entries
      .map((entry) => {
        const platformLabel = entry.platform === "truth" ? "Truth Social" : "X";
        const head = `<div class="influencer-card-head"><span class="influencer-name">${escapeHtml(entry.label)}</span><span class="influencer-platform">${escapeHtml(platformLabel)}</span></div>`;
        if (!entry.available) {
          return `<div class="influencer-card">${head}<div class="influencer-unavailable">Unavailable right now: ${escapeHtml(entry.error)}</div></div>`;
        }
        const posts = entry.posts.length
          ? entry.posts
              .map(
                (p) => `<div class="influencer-post">
                  <a href="${safeUrl(p.url)}" target="_blank" rel="noopener">
                    <span class="influencer-post-time">${escapeHtml(formatPostTime(p.postedAt))}</span>${escapeHtml(p.text || "(no text)")}
                  </a>
                  ${renderTopComments(p.topComments)}
                </div>`
              )
              .join("")
          : `<div class="influencer-unavailable">No recent posts found.</div>`;
        return `<div class="influencer-card">${head}${posts}</div>`;
      })
      .join("");
  } catch (err) {
    container.innerHTML = `<p class="fine-print">Couldn't load influencer posts: ${escapeHtml(err.message)}</p>`;
  }
}

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

function renderImportanceBars(container, importance) {
  const ranked = Object.entries(importance).sort((a, b) => b[1] - a[1]);
  const max = ranked.length ? ranked[0][1] : 1;
  container.innerHTML = ranked
    .map(
      ([name, weight]) => `<div class="importance-row">
        <span class="importance-label">${escapeHtml(name)}</span>
        <span class="importance-track"><span class="importance-fill" data-target="${((weight / max) * 100).toFixed(1)}"></span></span>
        <span class="importance-value">${pct(weight)}</span>
      </div>`
    )
    .join("");
  // Set width after the bar exists in the DOM (not inline in the markup
  // above) so the CSS transition actually animates from 0 instead of
  // snapping straight to its final width.
  requestAnimationFrame(() => {
    container.querySelectorAll(".importance-fill").forEach((el) => {
      el.style.width = `${el.dataset.target}%`;
    });
  });
}

document.getElementById("backtest-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const period = document.getElementById("backtest-period").value;
  const horizon = document.getElementById("backtest-horizon").value;
  const model = document.getElementById("backtest-model").value;
  try {
    setStatus(`Running backtest for ${state.ticker}...`);
    const result = await apiGet(`/api/backtest/${state.ticker}?period=${period}&horizon=${horizon}&model=${model}`);
    setStatus("");
    document.getElementById("backtest-results").classList.remove("hidden");
    document.getElementById("bt-accuracy").textContent = pct(result.accuracy);
    document.getElementById("bt-baseline").textContent = pct(result.majorityBaseline);
    const beatsCard = document.getElementById("bt-beats-card");
    beatsCard.classList.remove("good", "bad");
    beatsCard.classList.add(result.beatsBaseline ? "good" : "bad");
    document.getElementById("bt-beats").textContent = result.beatsBaseline ? "Yes" : "No";
    document.getElementById("bt-samples").textContent = `${result.nTrain} / ${result.nTest}`;
    document.getElementById("backtest-asof").textContent = `Fetched ${new Date().toLocaleTimeString()} · ${model.replace("_", " ")} model`;

    renderLineChart(
      document.getElementById("backtest-chart"),
      [
        { name: "Signal strategy", color: "#34d399", points: result.strategyCurve.map((v, i) => ({ x: i, y: v })) },
        { name: "Buy & hold", color: "#60a5fa", points: result.buyHoldCurve.map((v, i) => ({ x: i, y: v })) },
      ],
      { xLabels: result.dates, yFormat: (v) => v.toFixed(2) }
    );

    renderImportanceBars(document.getElementById("bt-importance"), result.featureImportance);
    document.getElementById("walkforward-results").classList.add("hidden"); // stale from a previous ticker/settings until re-run
  } catch (err) {
    setStatus(err.message, true);
  }
});

document.getElementById("run-walkforward").addEventListener("click", async () => {
  const period = document.getElementById("backtest-period").value;
  const horizon = document.getElementById("backtest-horizon").value;
  const model = document.getElementById("backtest-model").value;
  const btn = document.getElementById("run-walkforward");
  btn.disabled = true;
  setStatus(`Running walk-forward validation for ${state.ticker} (retrains ${5}x)...`);
  try {
    const result = await apiGet(`/api/walkforward/${state.ticker}?period=${period}&horizon=${horizon}&model=${model}&folds=5`);
    setStatus("");
    document.getElementById("walkforward-results").classList.remove("hidden");
    document.getElementById("walkforward-tbody").innerHTML = result.folds
      .map(
        (f) => `<tr>
          <td>${f.fold}</td>
          <td>${f.nTrain} / ${f.nTest}</td>
          <td>${pct(f.accuracy)}</td>
          <td>${pct(f.baseline)}</td>
          <td class="${f.beatsBaseline ? "cell-good" : "cell-bad"}">${f.beatsBaseline ? "yes" : "no"}</td>
        </tr>`
      )
      .join("");
    document.getElementById("walkforward-summary").textContent =
      `Across folds: mean accuracy ${pct(result.meanAccuracy)} (std ${pct(result.stdAccuracy)}), ` +
      `mean baseline ${pct(result.meanBaseline)}, beat baseline in ${pct(result.fractionBeatingBaseline, 0)} of folds.`;
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    btn.disabled = false;
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
    await Promise.all([loadOverview(ticker), loadPriceChart(ticker, state.period), loadNews(ticker)]);
    await loadChainExpirations(ticker);
    await loadChainTable();
    setStatus(`Loaded ${ticker}`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    btn.disabled = false;
  }
}

// ---------------- scan ----------------

function renderScanResults(results, sortBy) {
  const tbody = document.getElementById("scan-tbody");
  const ok = results.filter((r) => !r.error);
  const failed = results.filter((r) => r.error);

  const sortKey = { volume: (r) => -(r.volumeRatio ?? 0), iv: (r) => -(r.ivHvRatio ?? 0), accuracy: (r) => -(r.backtestAccuracy ?? 0) }[sortBy];
  ok.sort((a, b) => sortKey(a) - sortKey(b));

  const rows = ok.map((r) => {
    const volStr = r.volumeRatio != null ? `${r.volumeRatio >= 0 ? "+" : ""}${(r.volumeRatio * 100).toFixed(0)}%` : "?";
    const modelStr = `${escapeHtml(r.direction)} (${pct(r.liveProbability, 0)})`;
    const beatsClass = r.beatsBaseline ? "cell-good" : "cell-bad";
    const btStr = `${pct(r.backtestAccuracy)} vs ${pct(r.backtestBaseline)}`;
    return `<tr data-ticker="${escapeHtml(r.ticker)}">
      <td>${escapeHtml(r.ticker)}</td>
      <td>$${r.price.toFixed(2)}</td>
      <td>${volStr}</td>
      <td>${r.ivHvRatio.toFixed(2)}</td>
      <td>${escapeHtml(r.read)}</td>
      <td>${modelStr}</td>
      <td class="${beatsClass}">${btStr}</td>
    </tr>`;
  });

  const errorRows = failed.map(
    (r) => `<tr class="error-row"><td>${escapeHtml(r.ticker)}</td><td colspan="6">Skipped: ${escapeHtml(r.error)}</td></tr>`
  );

  tbody.innerHTML = rows.concat(errorRows).join("") || `<tr><td colspan="7" class="fine-print">No results.</td></tr>`;

  tbody.querySelectorAll("tr[data-ticker]").forEach((row) => {
    row.addEventListener("click", () => {
      const ticker = row.dataset.ticker;
      document.getElementById("ticker-input").value = ticker;
      loadTicker(ticker);
      document.querySelector('.side-link[data-section="overview"]').click();
    });
  });
}

document.getElementById("scan-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const raw = document.getElementById("scan-tickers").value.trim();
  const sortBy = document.getElementById("scan-sort").value;
  const btn = document.querySelector("#scan-form button");
  btn.disabled = true;
  document.getElementById("scan-tbody").innerHTML =
    `<tr><td colspan="7" class="fine-print">Scanning${raw ? "" : " the curated watchlist"}... this trains a model per ticker, give it a moment.</td></tr>`;
  try {
    const params = new URLSearchParams({ horizon: "5", model: "logistic" });
    if (raw) params.set("tickers", raw);
    const data = await apiGet(`/api/scan?${params.toString()}`);
    renderScanResults(data.results, sortBy);
  } catch (err) {
    document.getElementById("scan-tbody").innerHTML = `<tr><td colspan="7" class="fine-print">Couldn't load scan: ${escapeHtml(err.message)}</td></tr>`;
  } finally {
    btn.disabled = false;
  }
});

// ---------------- init ----------------

renderStrategyForm();
loadTicker(state.ticker);
loadInfluencers(); // fixed watchlist, independent of the searched ticker -- load once
