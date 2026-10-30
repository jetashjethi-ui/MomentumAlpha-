import type { CsvRecord, RunSummary } from "./api-types"
import { clamp, dateLabel, decimal, finite } from "./formatters"

export function phase(summary: RunSummary) {
  const perf = summary.performance || {}
  const dev = perf.development || perf
  return {
    perf,
    dev,
    strategy: dev.strategy_after_costs || perf.strategy_after_costs || {},
    benchmarks: dev.benchmarks_after_costs || perf.benchmarks_after_costs || {},
    classification: dev.classification || perf.classification || {},
    config: summary.configuration || {},
  }
}

export function holdoutState(summary: RunSummary) {
  const { perf, config } = phase(summary)
  const evaluated = Number(config.evaluated_final_holdout_weeks || 0)
  const reserved = Number(config.reserved_final_holdout_weeks || 0)
  if (evaluated > 0) {
    return {
      consumed: true,
      title: `Final holdout consumed · ${evaluated} weeks`,
      copy: `Window ${dateLabel(config.final_holdout_start)} to ${dateLabel(perf.final_holdout?.period_end)} has been inspected. Do not tune against it.`,
    }
  }
  return {
    consumed: false,
    title: reserved > 0 ? `${reserved}-week final holdout reserved · not evaluated` : "No final holdout reserved",
    copy: config.final_holdout_start
      ? `Reserved window begins ${dateLabel(config.final_holdout_start)}. This run scores development data only.`
      : "This run scores development data only. A fresh holdout is required for independent validation.",
  }
}

export function evidenceReadout(summary: RunSummary) {
  const { perf, classification, config } = phase(summary)
  const evaluated = Number(config.evaluated_final_holdout_weeks || 0)
  const spyCi = perf.annualized_excess_return_95pct_ci?.spy
  const crossesZero = Array.isArray(spyCi) && spyCi.length >= 2 && spyCi[0] <= 0 && spyCi[1] >= 0
  if (evaluated > 0) {
    return {
      headline: "Holdout already inspected",
      tone: "warn" as const,
      statement: `This run includes a ${evaluated}-week final period. Treat it as consumed evidence; tuning against it would invalidate an independent claim.`,
    }
  }
  if (finite(classification.roc_auc) && Number(classification.roc_auc) < 0.52) {
    let statement = `Development AUC is ${decimal(classification.roc_auc, 3)}, close to chance. `
    if (crossesZero) statement += "The 95% excess-return interval versus SPY crosses zero. "
    statement += "The current result is a research lead, not proof of a durable edge."
    return { headline: "Predictive edge unconfirmed", tone: "warn" as const, statement }
  }
  return {
    headline: "Development evidence only",
    tone: "neutral" as const,
    statement:
      "The strategy has not passed an untouched final holdout. Review uncertainty, costs, and risk before drawing conclusions.",
  }
}

export interface EquityPoint {
  date: string
  equity: number
  ret: number
  drawdown: number
}

export function equitySeries(records: CsvRecord[] | null): EquityPoint[] {
  if (!records) return []
  const usable = records.filter(
    (row) => row.evaluation_phase === "development" && finite(row.net_return) && row.signal_week,
  )
  let equity = 1
  let peak = 1
  return usable.map((row) => {
    const ret = Number(row.net_return)
    equity *= 1 + ret
    peak = Math.max(peak, equity)
    return { date: row.signal_week, equity, ret, drawdown: equity / peak - 1 }
  })
}

export function costCurve(rows: CsvRecord[] | null) {
  if (!rows) return []
  const byCost = new Map<number, CsvRecord>()
  rows
    .filter(
      (row) =>
        row.evaluation_phase === "development" && (row.portfolio === "MomentumAlpha" || row.portfolio === "strategy"),
    )
    .forEach((row) => {
      const bps = Number(row.transaction_cost_bps)
      if (Number.isFinite(bps)) byCost.set(bps, row)
    })
  return Array.from(byCost.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([bps, row]) => ({
      bps,
      annual: finite(row["Annualized Return"]) ? Number(row["Annualized Return"]) : null,
      sharpe: finite(row["Sharpe Ratio"]) ? Number(row["Sharpe Ratio"]) : null,
    }))
}

export function featureStability(rows: CsvRecord[] | null) {
  if (!rows) return []
  const groups = new Map<string, { rank: number; stability: number; folds: number; count: number }>()
  rows.forEach((row) => {
    if (!row.feature || !finite(row.mean_rank)) return
    const item = groups.get(row.feature) || { rank: 0, stability: 0, folds: 0, count: 0 }
    item.rank += Number(row.mean_rank)
    item.stability += Number(row.nonzero_fold_fraction || 0)
    item.folds += Number(row.folds_observed || 0)
    item.count += 1
    groups.set(row.feature, item)
  })
  return Array.from(groups.entries())
    .map(([feature, v]) => ({
      feature,
      rank: v.rank / v.count,
      stability: v.stability / v.count,
      folds: v.folds / v.count,
    }))
    .sort((a, b) => a.rank - b.rank)
    .slice(0, 8)
}

export function factorAlpha(rows: CsvRecord[] | null) {
  if (!rows) return null
  const row = rows.find((item) => item.phase === "all" && item.term === "alpha")
  if (!row || !finite(row.estimate)) return null
  const low = Number(row.ci95_low)
  const high = Number(row.ci95_high)
  return {
    estimate: Number(row.estimate),
    low: finite(row.ci95_low) ? low : null,
    high: finite(row.ci95_high) ? high : null,
    p: finite(row.nominal_p_value) ? Number(row.nominal_p_value) : null,
    crossesZero: finite(row.ci95_low) && finite(row.ci95_high) ? low <= 0 && high >= 0 : null,
  }
}

export function calibrationPoints(summary: RunSummary) {
  const curve = summary.performance?.probability_calibration_curve || []
  const valid = curve.filter(
    (point) => finite(point.mean_predicted_probability) && finite(point.observed_positive_rate),
  )
  const calibrated = valid.filter((point) => point.calibration_type === "calibrated_ensemble")
  const raw = valid.filter((point) => point.calibration_type !== "calibrated_ensemble")
  const points = calibrated.length ? calibrated : raw
  return {
    kind: calibrated.length ? "Calibrated ensemble" : raw.length ? "Raw ensemble" : null,
    points: points.map((point) => ({
      x: clamp(Number(point.mean_predicted_probability), 0, 1),
      y: clamp(Number(point.observed_positive_rate), 0, 1),
      n: point.observations,
    })),
  }
}
