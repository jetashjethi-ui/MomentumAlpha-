"use client"

import { Chapter, Label, Reveal, Unavailable } from "./chapter"
import type { RunBundle } from "@/lib/api-types"
import { calibrationPoints, costCurve, evidenceReadout, factorAlpha, featureStability, phase } from "@/lib/derive"
import { clamp, decimal, finite, pct, signedPct } from "@/lib/formatters"
import { cn } from "@/lib/utils"

export function EvidenceChapter({ bundle }: { bundle: RunBundle | undefined }) {
  return (
    <Chapter
      id="evidence"
      index="03"
      label="Evidence"
      title="Model evidence"
      aside={
        <p className="font-mono text-xs leading-relaxed text-muted-foreground">
          Predictive quality is judged separately from portfolio returns. Missing artifacts are reported as unavailable,
          never as zero.
        </p>
      }
    >
      {!bundle ? <Unavailable>Evidence appears once a completed run is selected.</Unavailable> : <EvidenceBody bundle={bundle} />}
    </Chapter>
  )
}

function EvidenceBody({ bundle }: { bundle: RunBundle }) {
  const { summary } = bundle
  const { perf, classification } = phase(summary)
  const readout = evidenceReadout(summary)
  const calibration = calibrationPoints(summary)
  const costs = costCurve(bundle.costs)
  const features = featureStability(bundle.features)
  const alpha = factorAlpha(bundle.factors)
  const auc = classification.roc_auc

  const quality = [
    { label: "ROC AUC", value: decimal(auc, 3) },
    { label: "Brier score", value: decimal(classification.brier_score, 3) },
    { label: "Mean IC", value: decimal(perf.information_coefficient_mean, 3) },
    { label: "Random-pick p", value: decimal(perf.random_pick_cost_matched_p_value, 3) },
  ]

  return (
    <div className="space-y-32">
      <Reveal className="max-w-3xl">
        <Label className={readout.tone === "warn" ? "text-accent" : undefined}>{readout.headline}</Label>
        <p className="mt-5 text-pretty text-2xl leading-snug text-foreground/90 md:text-3xl">{readout.statement}</p>
      </Reveal>

      <div className="grid gap-16 lg:grid-cols-12">
        <div className="lg:col-span-6">
          <Reveal from="left">
            <Label>Predictive quality · development</Label>
          </Reveal>
          <dl className="mt-8 grid grid-cols-2 gap-x-8 gap-y-10">
            {quality.map((item, index) => (
              <Reveal key={item.label} from="left" delay={index * 70}>
                <dt>
                  <Label>{item.label}</Label>
                </dt>
                <dd className="mt-2 font-display text-5xl tabular-nums leading-none">{item.value}</dd>
              </Reveal>
            ))}
          </dl>
          <Reveal className="mt-12" delay={120}>
            <AucScale auc={finite(auc) ? Number(auc) : null} />
          </Reveal>
        </div>

        <Reveal from="right" delay={120} className="lg:col-span-5 lg:col-start-8">
          <Label>Probability calibration</Label>
          {calibration.points.length ? (
            <>
              <CalibrationPlot points={calibration.points} />
              <p className="mt-4 font-mono text-xs leading-relaxed text-muted-foreground">
                {calibration.kind} · ECE {decimal(classification.expected_calibration_error_10_bins, 3)} across 10 bins.
                Points near the diagonal are better aligned.
              </p>
            </>
          ) : (
            <Unavailable>No probability bins were recorded for this run.</Unavailable>
          )}
        </Reveal>
      </div>

      <div className="grid gap-16 lg:grid-cols-12">
        <Reveal from="left" className="lg:col-span-6">
          <Label>Feature stability · top 8 by mean rank</Label>
          {features.length ? (
            <ul className="mt-8 space-y-4">
              {features.map((item) => (
                <li key={item.feature} className="grid grid-cols-[minmax(0,1fr)_120px_48px] items-center gap-4">
                  <span className="truncate font-mono text-xs text-foreground" title={item.feature}>
                    {item.feature}
                  </span>
                  <span className="relative h-px bg-foreground/15" aria-hidden="true">
                    <span
                      className="absolute inset-y-[-1px] left-0 bg-accent"
                      style={{ width: `${clamp(item.stability, 0, 1) * 100}%` }}
                    />
                  </span>
                  <span className="text-right font-mono text-[11px] tabular-nums text-muted-foreground">
                    {pct(item.stability, 0)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <Unavailable>feature_importance_stability.csv unavailable for this run.</Unavailable>
          )}
          {features.length ? (
            <p className="mt-6 font-mono text-[11px] text-muted-foreground">Share of folds with non-zero importance.</p>
          ) : null}
        </Reveal>

        <Reveal from="right" delay={120} className="lg:col-span-5 lg:col-start-8">
          <Label>Transaction-cost sensitivity</Label>
          {costs.length ? <CostLadder costs={costs} /> : <Unavailable>cost_sensitivity.csv unavailable for this run.</Unavailable>}
        </Reveal>
      </div>

      <Reveal className="max-w-4xl">
        <Label>Factor attribution · alpha</Label>
        {alpha ? (
          <div className="mt-8">
            <p className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
              <span className="font-display text-7xl tabular-nums leading-none md:text-8xl">{signedPct(alpha.estimate)}</span>
              <span className="font-mono text-xs text-muted-foreground">annualized · p = {decimal(alpha.p, 3)}</span>
            </p>
            <AlphaInterval estimate={alpha.estimate} low={alpha.low} high={alpha.high} />
            <p className="mt-6 max-w-2xl text-pretty font-mono text-xs leading-relaxed text-muted-foreground">
              {alpha.crossesZero == null
                ? "The confidence interval was not recorded."
                : alpha.crossesZero
                  ? "The 95% confidence interval crosses zero. Alpha is not statistically distinguishable from zero in this development sample."
                  : "The interval does not cross zero in this sample; it still requires a fresh holdout before it can be treated as independent evidence."}
            </p>
          </div>
        ) : (
          <Unavailable>Factor attribution was not available for this run.</Unavailable>
        )}
      </Reveal>
    </div>
  )
}

function AucScale({ auc }: { auc: number | null }) {
  const position = auc == null ? null : clamp((auc - 0.3) / 0.4, 0, 1) * 100
  return (
    <div>
      <div className="relative h-8">
        <div className="absolute inset-x-0 top-1/2 h-px bg-foreground/15" />
        <div className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-foreground/40" style={{ left: "50%" }} />
        {position != null ? (
          <div
            className="absolute top-1/2 h-4 w-1 -translate-x-1/2 -translate-y-1/2 bg-accent"
            style={{ left: `${position}%` }}
          />
        ) : null}
      </div>
      <div className="flex justify-between font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
        <span>0.30</span>
        <span>0.50 · chance</span>
        <span>0.70</span>
      </div>
      <p className="mt-4 font-mono text-xs leading-relaxed text-muted-foreground">
        {auc == null
          ? "ROC AUC is unavailable for this run."
          : auc < 0.52
            ? "Near-chance separation: do not read return performance as strong classification skill."
            : "Ranking skill should still be judged against uncertainty and costs."}
      </p>
    </div>
  )
}

function CalibrationPlot({ points }: { points: { x: number; y: number; n?: number }[] }) {
  const size = 260
  const pad = 28
  const inner = size - pad * 2
  return (
    <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Calibration plot of predicted versus observed positive rate" className="mt-6 w-full max-w-sm">
      <line x1={pad} y1={size - pad} x2={size - pad} y2={pad} stroke="currentColor" strokeDasharray="3 5" className="text-foreground/30" />
      <line x1={pad} y1={size - pad} x2={size - pad} y2={size - pad} stroke="currentColor" className="text-foreground/15" />
      <line x1={pad} y1={pad} x2={pad} y2={size - pad} stroke="currentColor" className="text-foreground/15" />
      <text x={size - pad} y={size - 8} textAnchor="end" className="fill-muted-foreground font-mono text-[9px]">
        predicted
      </text>
      <text x={pad - 6} y={pad - 10} className="fill-muted-foreground font-mono text-[9px]">
        observed
      </text>
      {points.map((point, index) => (
        <circle key={index} cx={pad + point.x * inner} cy={size - pad - point.y * inner} r={3.5} fill="#ff6b1a">
          <title>
            Predicted {pct(point.x)} · observed {pct(point.y)}
            {point.n != null ? ` · n=${point.n}` : ""}
          </title>
        </circle>
      ))}
    </svg>
  )
}

function CostLadder({ costs }: { costs: { bps: number; annual: number | null; sharpe: number | null }[] }) {
  const max = Math.max(0.0001, ...costs.map((row) => Math.abs(row.annual ?? 0)))
  return (
    <ul className="mt-8 space-y-4">
      {costs.map((row) => (
        <li key={row.bps} className="grid grid-cols-[64px_minmax(0,1fr)_64px] items-center gap-4">
          <span className="font-mono text-xs text-muted-foreground">{row.bps} bps</span>
          <span className="relative h-px bg-foreground/15" aria-hidden="true">
            {row.annual != null ? (
              <span
                className={cn("absolute inset-y-[-1px] left-0", row.annual >= 0 ? "bg-foreground/70" : "bg-accent")}
                style={{ width: `${clamp((Math.abs(row.annual) / max) * 100, 2, 100)}%` }}
              />
            ) : null}
          </span>
          <span className="text-right font-mono text-xs tabular-nums text-foreground">{pct(row.annual)}</span>
        </li>
      ))}
    </ul>
  )
}

function AlphaInterval({ estimate, low, high }: { estimate: number; low: number | null; high: number | null }) {
  const pos = (value: number) => (clamp(value, -0.5, 0.5) + 0.5) * 100
  return (
    <div className="mt-10">
      <div className="relative h-10">
        <div className="absolute inset-x-0 top-1/2 h-px bg-foreground/15" />
        <div className="absolute top-1/2 h-5 w-px -translate-y-1/2 bg-foreground/50" style={{ left: "50%" }} />
        {low != null && high != null ? (
          <div
            className="absolute top-1/2 h-1 -translate-y-1/2 bg-accent/40"
            style={{ left: `${pos(low)}%`, width: `${Math.max(0, pos(high) - pos(low))}%` }}
          />
        ) : null}
        <div className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent" style={{ left: `${pos(estimate)}%` }} />
      </div>
      <div className="flex justify-between font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
        <span>−50%</span>
        <span>
          95% CI {signedPct(low)} to {signedPct(high)}
        </span>
        <span>+50%</span>
      </div>
    </div>
  )
}
