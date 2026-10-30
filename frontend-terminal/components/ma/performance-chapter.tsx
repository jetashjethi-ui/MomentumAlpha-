"use client"

import { useMemo } from "react"
import { Chapter, Label, Reveal, Unavailable } from "./chapter"
import { EquityChart } from "./equity-chart"
import type { MetricBlock, RunBundle } from "@/lib/api-types"
import { equitySeries, phase } from "@/lib/derive"
import { dateLabel, decimal, pct } from "@/lib/formatters"
import { cn } from "@/lib/utils"

export function PerformanceChapter({ bundle }: { bundle: RunBundle | undefined }) {
  const series = useMemo(() => equitySeries(bundle?.returns ?? null), [bundle?.returns])

  return (
    <Chapter
      id="performance"
      index="02"
      label="Performance"
      title="The equity curve"
      aside={
        <p className="font-mono text-xs leading-relaxed text-muted-foreground">
          Net of modeled transaction costs. Development period only; the final holdout is never blended into this curve.
        </p>
      }
    >
      {!bundle ? (
        <Unavailable>Performance appears once a completed run is selected.</Unavailable>
      ) : (
        <PerformanceBody bundle={bundle} series={series} />
      )}
    </Chapter>
  )
}

function PerformanceBody({ bundle, series }: { bundle: RunBundle; series: ReturnType<typeof equitySeries> }) {
  const { perf, dev, strategy, benchmarks } = phase(bundle.summary)
  const sharpeCi = perf.sharpe_95pct_block_bootstrap_ci
  const metrics = [
    { label: "CAGR · net", value: pct(strategy["Annualized Return"]) },
    { label: "Sharpe", value: decimal(strategy["Sharpe Ratio"]) },
    { label: "Max drawdown", value: pct(strategy["Max Drawdown"]) },
    { label: "Volatility", value: pct(strategy["Annualized Volatility"]) },
    { label: "CVaR 95%", value: pct(strategy["CVaR (95%)"]) },
    { label: "Mean turnover", value: pct(perf.mean_turnover) },
  ]

  const comparison: { name: string; value?: MetricBlock; selected?: boolean }[] = [
    { name: "MomentumAlpha", value: strategy, selected: true },
    { name: "Equal weight", value: benchmarks.equal_weight },
    { name: "SPY", value: benchmarks.spy },
    { name: "Raw momentum", value: benchmarks.raw_momentum },
  ]

  return (
    <div>
      <Reveal>
        <EquityChart series={series} />
      </Reveal>

      <dl className="mt-24 grid grid-cols-2 gap-x-8 gap-y-12 md:grid-cols-3 xl:grid-cols-6">
        {metrics.map((metric, index) => (
          <Reveal key={metric.label} from={index % 2 ? "right" : "left"} delay={(index % 3) * 70}>
            <dt>
              <Label>{metric.label}</Label>
            </dt>
            <dd className="mt-3 font-display text-5xl tabular-nums leading-none">{metric.value}</dd>
          </Reveal>
        ))}
      </dl>
      <Reveal className="mt-8">
        <p className="font-mono text-xs text-muted-foreground">
          Sharpe 95% block-bootstrap interval:{" "}
          <span className="text-foreground">
            {Array.isArray(sharpeCi) && sharpeCi.length >= 2 ? `${decimal(sharpeCi[0])} to ${decimal(sharpeCi[1])}` : "not available"}
          </span>{" "}
          · {dateLabel(dev.period_start)} — {dateLabel(dev.period_end)}
        </p>
      </Reveal>

      <div className="mt-24 grid gap-12 lg:grid-cols-12">
        <Reveal from="left" className="lg:col-span-4">
          <Label>Benchmark comparison</Label>
          <p className="mt-4 max-w-sm text-pretty text-xl leading-snug text-foreground/85">
            The same universe, the same costs. Benchmarks without recorded values are shown as unavailable.
          </p>
        </Reveal>
        <Reveal from="right" delay={120} className="overflow-x-auto lg:col-span-8">
          <table className="w-full min-w-[480px] text-left">
            <caption className="sr-only">Annualized return, Sharpe ratio and max drawdown by portfolio, after costs</caption>
            <thead>
              <tr>
                {["Portfolio", "Annualized", "Sharpe", "Max DD"].map((head, index) => (
                  <th
                    key={head}
                    scope="col"
                    className={cn(
                      "pb-4 font-mono text-[10px] font-normal uppercase tracking-[0.24em] text-muted-foreground",
                      index > 0 && "text-right",
                    )}
                  >
                    {head}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparison.map((row) => (
                <tr key={row.name} className="group">
                  <th scope="row" className={cn("py-3 text-base font-normal", row.selected ? "text-accent" : "text-foreground")}>
                    {row.name}
                  </th>
                  <td className="py-3 text-right font-mono text-sm tabular-nums">{pct(row.value?.["Annualized Return"])}</td>
                  <td className="py-3 text-right font-mono text-sm tabular-nums">{decimal(row.value?.["Sharpe Ratio"])}</td>
                  <td className="py-3 text-right font-mono text-sm tabular-nums">{pct(row.value?.["Max Drawdown"])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Reveal>
      </div>
    </div>
  )
}
