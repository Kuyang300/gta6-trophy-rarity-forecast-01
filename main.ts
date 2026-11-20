// main.ts — GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).
//
// Reads one case per *.in file (schema makefun/gta6-trophies-case/v1) from
// $MAKEFUN_INPUT_DIR (else /input) and writes /output/<id>.out with five
// quantiles (5/25/50/75/95%) of each trophy's Exophase earn rate at the
// case's target. Node 22, standard library only, no network, deterministic:
// the same case always yields the same bytes.
//
// Model (per trophy, all in logit space, rates clamped to [0.01, 99.99]):
//   * A direct trend: weighted least squares of logit(rate) on
//     ln(1 + hours since release), recency-weighted (24 h half-life), so
//     launch-night behaviour matters less than the last day's.
//   * A counts decomposition: the earn rate is earned/tracked, so fit
//     ln(earned) and ln(players_tracked) separately on the same clock and
//     take their ratio at the target. This sees dilution: when Exophase
//     refreshes and newly tracked players arrive holding few trophies,
//     rates fall even while earned counts keep rising.
//   * Tier shrinkage: each trophy's slope is shrunk toward its tier's
//     median slope (platinum/gold/silver/bronze), which steadies secret
//     and late-game trophies whose own histories are thin or flat.
//   * Median: a blend of the two fits and persistence (the last read).
//   * Spread: a random-walk term growing with the horizon plus the fit's
//     residual scale, the slope's standard error projected to the target,
//     and a refresh-jump term from the largest single-read logit drop seen.
//     Wide where the history is jumpy, narrow where it is steady.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z");
const HOUR_MS = 3600000;
const HALF_LIFE_HOURS = 24;

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function logit(pct: number): number {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}

function invLogit(l: number): number {
  const v = 1 / (1 + Math.exp(-clamp(l, -12, 12)));
  return clamp(v * 100, 0.01, 99.99);
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

type Wls = { a: number; b: number; xbar: number; sxx: number; resid: number; seB: number; wsum: number };

// Weighted least squares of y on x. Returns intercept, slope, residual
// scale and the slope's standard error; degenerate inputs fall back to a
// flat line at the weighted mean of y.
function wls(pts: { x: number; y: number; w: number }[]): Wls {
  let wsum = 0;
  let xbar = 0;
  let ybar = 0;
  for (const p of pts) {
    wsum += p.w;
    xbar += p.w * p.x;
    ybar += p.w * p.y;
  }
  if (wsum <= 0 || pts.length === 0) return { a: 0, b: 0, xbar: 0, sxx: 0, resid: 0, seB: 0, wsum: 0 };
  xbar /= wsum;
  ybar /= wsum;
  let sxx = 0;
  let sxy = 0;
  for (const p of pts) {
    sxx += p.w * (p.x - xbar) * (p.x - xbar);
    sxy += p.w * (p.x - xbar) * (p.y - ybar);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = ybar - b * xbar;
  let rss = 0;
  for (const p of pts) {
    const r = p.y - (a + b * p.x);
    rss += p.w * r * r;
  }
  const dof = Math.max(pts.length - 2, 1);
  const resid = Math.sqrt(rss / (wsum * dof / Math.max(pts.length, 1)) || 0);
  const seB = sxx > 1e-12 ? resid / Math.sqrt(sxx) : 0;
  return { a, b, xbar, sxx, resid, seB, wsum };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir).filter((f) => f.endsWith(".in")).sort();
  for (const f of files) {
    const kase = JSON.parse(readFileSync(join(inputDir, f), "utf8"));
    const out = forecastCase(kase);
    writeFileSync(join(outputDir, String(kase.id) + ".out"), JSON.stringify(out));
  }
}

main();

// ---- history preparation -------------------------------------------------

type Read = { h: number; x: number; n: number; rates: (number | null)[] };

// Compress the case history to its change points (a read whose players
// tracked and every rate equal the previous kept read adds no signal, and
// would inflate the effective sample size), and express each read on the
// log-time clock x = ln(1 + hours since release).
function prepReads(kase: any): Read[] {
  const out: Read[] = [];
  const hist: any[] = Array.isArray(kase.history) ? kase.history : [];
  for (const r of hist) {
    if (!r || typeof r.at !== "string" || !Array.isArray(r.rates)) continue;
    const ms = Date.parse(r.at);
    if (!Number.isFinite(ms)) continue;
    const h = Math.max(0, (ms - RELEASE_MS) / HOUR_MS);
    const n = typeof r.players_tracked === "number" && r.players_tracked > 0 ? r.players_tracked : 0;
    const rates = r.rates.map((v: any) => (typeof v === "number" && Number.isFinite(v) ? v : null));
    const prev = out[out.length - 1];
    if (prev && prev.n === n && prev.rates.length === rates.length) {
      let same = true;
      for (let i = 0; i < rates.length; i++) {
        if (rates[i] !== prev.rates[i]) { same = false; break; }
      }
      if (same) continue;
    }
    out.push({ h, x: Math.log1p(h), n, rates });
  }
  return out;
}

// Recency weight of a read at hour h against the last read's hour.
function recency(h: number, hLast: number): number {
  return Math.pow(0.5, Math.max(0, hLast - h) / HALF_LIFE_HOURS);
}

type TrophyFit = {
  last: number; // last observed rate, clamped
  slope: number; // shrunk direct-trend slope
  medDirect: number; // median logit from the direct fit at the target
  medCounts: number; // median logit from the counts fit at the target
  resid: number; // direct fit residual scale
  seProj: number; // slope standard error projected to the target
  jump: number; // largest single-read logit drop observed
  nPts: number;
};

// Fit one trophy. `tierSlope` is its tier's median raw slope, used for
// shrinkage; `nFit` is the shared log-players fit for the counts model.
function fitTrophy(idx: number, reads: Read[], xTarget: number, tierSlope: number, nFit: Wls): TrophyFit {
  const lastRead = reads[reads.length - 1];
  const hLast = lastRead ? lastRead.h : 0;
  const pts: { x: number; y: number; w: number }[] = [];
  const ePts: { x: number; y: number; w: number }[] = [];
  let last: number | null = null;
  let jump = 0;
  let prevLogit: number | null = null;
  for (const r of reads) {
    const v = idx < r.rates.length ? r.rates[idx] : null;
    if (v === null) continue;
    const y = logit(v);
    if (prevLogit !== null) jump = Math.max(jump, prevLogit - y);
    prevLogit = y;
    last = v;
    const w = recency(r.h, hLast);
    pts.push({ x: r.x, y, w });
    if (r.n > 0 && v > 0) ePts.push({ x: r.x, y: Math.log(Math.max(1, (v / 100) * r.n)), w });
  }
  const lastRate = last === null ? NaN : clamp(last, 0.01, 99.99);
  const direct = wls(pts);
  const eFit = wls(ePts);
  // Shrink the direct slope toward the tier median; the weight on the
  // trophy's own slope grows with its effective sample size.
  const k = 8;
  const slope = (direct.wsum * direct.b + k * tierSlope) / (direct.wsum + k);
  const aShrunk = direct.a + (slope - direct.b) * direct.xbar;
  const medDirect = aShrunk + slope * xTarget;
  let medCounts = medDirect;
  if (ePts.length >= 3 && nFit.wsum > 0) {
    const lnE = eFit.a + eFit.b * xTarget;
    const lnN = nFit.a + nFit.b * xTarget;
    medCounts = logit(clamp(Math.exp(lnE - lnN) * 100, 0.01, 99.99));
  }
  const seProj = direct.seB * Math.abs(xTarget - direct.xbar);
  return {
    last: lastRate, slope, medDirect, medCounts,
    resid: direct.resid, seProj, jump, nPts: pts.length,
  };
}

// ---- forecast ------------------------------------------------------------

// Tier priors for trophies with no usable history at all (median rate in
// percent, wide spread): the platinum needs every other trophy, golds are
// late-game, bronzes are the broad base.
const TIER_PRIOR: Record<string, number> = { platinum: 0.5, gold: 2, silver: 5, bronze: 10 };

function forecastCase(kase: any): { trophies: { id: number; q: number[] }[] } {
  const trophies: any[] = Array.isArray(kase.trophies) ? kase.trophies : [];
  const reads = prepReads(kase);
  const targetMs = Date.parse(typeof kase.target === "string" ? kase.target : "") || 0;
  const horizonDays = Math.max(0, (targetMs - (reads.length ? Date.parse(kase.at) : targetMs)) / 86400000);
  const hTarget = Math.max(0, (targetMs - RELEASE_MS) / HOUR_MS);
  const xTarget = Math.log1p(hTarget);
  const hLast = reads.length ? reads[reads.length - 1].h : 0;

  // Shared log-players fit (counts model denominator), recency-weighted.
  const nPts = reads.filter((r) => r.n > 0).map((r) => ({ x: r.x, y: Math.log(r.n), w: recency(r.h, hLast) }));
  const nFit = wls(nPts);

  // First pass: raw direct slopes, to take each tier's median for shrinkage.
  const rawSlopes: Record<string, number[]> = {};
  const rawByIdx: (Wls | null)[] = [];
  for (let i = 0; i < trophies.length; i++) {
    const pts: { x: number; y: number; w: number }[] = [];
    for (const r of reads) {
      const v = i < r.rates.length ? r.rates[i] : null;
      if (v === null) continue;
      pts.push({ x: r.x, y: logit(v), w: recency(r.h, hLast) });
    }
    const f = pts.length >= 2 ? wls(pts) : null;
    rawByIdx.push(f);
    const tier = typeof trophies[i].tier === "string" ? trophies[i].tier : "bronze";
    if (f) (rawSlopes[tier] = rawSlopes[tier] || []).push(f.b);
  }
  const tierSlope: Record<string, number> = {};
  for (const t of Object.keys(rawSlopes)) tierSlope[t] = median(rawSlopes[t]);

  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < trophies.length; i++) {
    const t = trophies[i];
    const id = typeof t.id === "number" ? t.id : i + 1;
    const tier = typeof t.tier === "string" ? t.tier : "bronze";
    const fit = fitTrophy(i, reads, xTarget, tierSlope[tier] ?? 0, nFit);
    let medLogit: number;
    let sigma: number;
    if (!Number.isFinite(fit.last) || fit.nPts === 0) {
      // No history for this trophy: tier prior, wide spread.
      medLogit = logit(TIER_PRIOR[tier] ?? 5);
      sigma = 1.2;
    } else if (fit.nPts < 3) {
      medLogit = logit(fit.last);
      sigma = 0.6;
    } else {
      // Blend: direct trend, counts decomposition, persistence. The trend
      // earns its weight from the number of distinct reads; with a flat
      // history all three agree and the blend is persistence.
      const wTrend = clamp(fit.nPts / 24, 0.25, 0.8);
      const trend = (fit.medDirect + fit.medCounts) / 2;
      medLogit = wTrend * trend + (1 - wTrend) * logit(fit.last);
      // Never let the extrapolation run away from the last read.
      medLogit = clamp(medLogit, logit(fit.last) - 3.5, logit(fit.last) + 3.5);
      const base = 0.45 * Math.sqrt(Math.max(horizonDays, 0.05));
      const resid = clamp(fit.resid, 0.05, 0.8);
      const se = clamp(fit.seProj, 0, 1.0);
      const jumpTerm = 0.35 * clamp(fit.jump, 0, 2);
      sigma = Math.sqrt(base * base + resid * resid + se * se + jumpTerm * jumpTerm);
      sigma = clamp(sigma, 0.4, 1.5);
    }
    const q = Z.map((z) => {
      const v = invLogit(medLogit + z * sigma);
      return Math.round(v * 10000) / 10000;
    });
    // Guarantee non-decreasing after rounding.
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    rows.push({ id, q });
  }
  return { trophies: rows };
}
