"use client"

import { ArrowDownRight, ChevronDown } from "lucide-react"
import { Label, Reveal } from "./chapter"
import type { RunBundle, RunRow } from "@/lib/api-types"
import { runMode } from "@/lib/api-client"
import { evidenceReadout, holdoutState, phase } from "@/lib/derive"
import { dateLabel, decimal, integer, modeLabel, pct, shortDate } from "@/lib/formatters"
import { cn } from "@/lib/utils"

interface Props {
  online: boolean
  apiLoading: boolean
  rows: RunRow[]
  selectedId: string | null
  onSelect: (id: string) => void
  bundle: RunBundle | undefined
  bundleLoading: boolean
  bundleError: Error | undefined
}

export function OverviewChapter({ online, apiLoading, rows, selectedId, onSelect, bundle, bundleLoading, bundleError }: Props) {
  return (
    <section id="overview" aria-labelledby="overview-title" className="relative px-5 pb-24 pt-36 md:px-12 md:pt-44 lg:pl-28 lg:pr-16">
      <Reveal>
        <p className="font-mono text-[11px] uppercase tracking-[0.32em] text-accent">01 / Overview</p>
      </Reveal>
      <Reveal delay={80}>
        <h1
          id="overview-title"
          className="mt-6 max-w-[14ch] text-balance font-display text-[18vw] leading-[0.84] tracking-tight md:text-[10.5rem] lg:text-[12rem]"
        >
          Read the <span className="text-accent">signal</span>. Respect the noise.
        </h1>
      </Reveal>

      <div className="mt-14 grid gap-10 md:grid-cols-12">
        <Reveal from="left" delay={160} className="md:col-span-6 lg:col-span-5">
          <p className="max-w-lg font-mono text-sm leading-relaxed text-muted-foreground">
            MomentumAlpha is a local research terminal for systematic equity momentum. Every figure on this page is a
            historical backtest read from saved run artifacts. It is not a live trading system and makes no forecast.
          </p>
          <a
            href="#performance"
            className="group mt-10 inline-flex items-center gap-3 font-mono text-[11px] uppercase tracking-[0.26em] text-foreground"
          >
            <span className="border-b border-foreground/20 pb-1 transition-colors group-hover:border-accent group-hover:text-accent">
              Enter the evidence
            </span>
            <ArrowDownRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:translate-y-0.5" />
          </a>
        </Reveal>

        <Reveal from="right" delay={220} className="md:col-span-6 md:col-start-7 lg:col-span-5 lg:col-start-8">
          <RunSelector rows={rows} selectedId={selectedId} onSelect={onSelect} online={online} />
        </Reveal>
      </div>

      <div className="mt-24">
        {apiLoading ? (
          <Label>Connecting to the local research API…</Label>
        ) : !online ? (
          <OfflineState />
        ) : !rows.length ? (
          <EmptyState />
        ) : bundleError && !bundle ? (
          <Reveal>
            <Label className="text-accent">Run result not ready</Label>
            <p className="mt-4 max-w-xl font-mono text-sm leading-relaxed text-muted-foreground">{bundleError.message}</p>
          </Reveal>
        ) : bundle ? (
          <Snapshot bundle={bundle} loading={bundleLoading} />
        ) : (
          <Label>Loading selected run…</Label>
        )}
      </div>
    </section>
  )
}

function RunSelector({
  rows,
  selectedId,
  onSelect,
  online,
}: {
  rows: RunRow[]
  selectedId: string | null
  onSelect: (id: string) => void
  online: boolean
}) {
  return (
    <div>
      <label htmlFor="run-selector">
        <Label>Selected run</Label>
      </label>
      <div className="relative mt-3">
        <select
          id="run-selector"
          value={selectedId ?? ""}
          disabled={!online || !rows.length}
          onChange={(event) => onSelect(event.target.value)}
          className="w-full appearance-none border-b border-foreground/20 bg-transparent py-3 pr-8 font-mono text-sm text-foreground transition-colors hover:border-foreground/50 focus:border-accent focus:outline-none disabled:text-muted-foreground"
        >
          {!rows.length ? <option value="">{online ? "No runs yet" : "Runs unavailable"}</option> : null}
          {rows.map((run) => (
            <option key={run.run_id} value={run.run_id} className="bg-background">
              {run.run_id} · {modeLabel(runMode(run))} · {shortDate(run.created_at_utc)}
              {run.status && run.status !== "completed" ? ` · ${run.status}` : ""}
            </option>
          ))}
        </select>
        <ChevronDown aria-hidden="true" className="pointer-events-none absolute right-1 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
      </div>
    </div>
  )
}

function Snapshot({ bundle, loading }: { bundle: RunBundle; loading: boolean }) {
  const { summary, verification } = bundle
  const { strategy, dev, config } = phase(summary)
  const readout = evidenceReadout(summary)
  const holdout = holdoutState(summary)
  const verified = verification?.valid
  const cost = config.transaction_cost_bps_per_traded_notional ?? config.transaction_cost_bps

  const metrics = [
    { label: "Annualized return · net", value: pct(strategy["Annualized Return"]), note: cost != null ? `${cost} bps modeled cost` : "Cost not recorded" },
    { label: "Sharpe ratio", value: decimal(strategy["Sharpe Ratio"]), note: "After costs" },
    { label: "Max drawdown", value: pct(strategy["Max Drawdown"]), note: "Worst peak to trough" },
    { label: "Development weeks", value: integer(dev.weeks ?? strategy["Num Weeks"]), note: `${dateLabel(dev.period_start)} — ${dateLabel(dev.period_end)}` },
  ]

  return (
    <div className={cn("transition-opacity", loading && "opacity-60")} aria-busy={loading}>
      <div className="grid gap-12 md:grid-cols-12">
        <Reveal from="left" className="md:col-span-7">
          <Label className={readout.tone === "warn" ? "text-accent" : undefined}>{readout.headline}</Label>
          <p className="mt-5 max-w-2xl text-pretty text-2xl leading-snug text-foreground/90 md:text-3xl">{readout.statement}</p>
        </Reveal>
        <Reveal from="right" delay={120} className="space-y-8 md:col-span-4 md:col-start-9">
          <div>
            <Label>Run</Label>
            <p className="mt-2 font-mono text-sm text-foreground">{summary.run_id}</p>
            <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
              {modeLabel(config.evaluation_mode)}
            </p>
          </div>
          <div>
            <Label>Artifact verification</Label>
            <p className="mt-2 flex items-center gap-2 font-mono text-sm">
              <span
                aria-hidden="true"
                className={cn("h-1.5 w-1.5 rounded-full", verified ? "bg-accent" : verification ? "bg-foreground/60" : "bg-foreground/20")}
              />
              {verification == null
                ? "Unavailable"
                : verified
                  ? `Verified · ${verification.checked_artifacts ?? "—"} artifacts`
                  : "Needs review"}
            </p>
          </div>
          <div>
            <Label>Final holdout</Label>
            <p className="mt-2 text-sm text-foreground">{holdout.title}</p>
            <p className="mt-1 font-mono text-xs leading-relaxed text-muted-foreground">{holdout.copy}</p>
          </div>
        </Reveal>
      </div>

      <dl className="mt-20 grid grid-cols-2 gap-x-8 gap-y-12 lg:grid-cols-4">
        {metrics.map((metric, index) => (
          <Reveal key={metric.label} from={index % 2 ? "right" : "left"} delay={index * 80}>
            <dt>
              <Label>{metric.label}</Label>
            </dt>
            <dd className="mt-3 font-display text-6xl tabular-nums leading-none md:text-7xl">{metric.value}</dd>
            <dd className="mt-3 font-mono text-[11px] text-muted-foreground">{metric.note}</dd>
          </Reveal>
        ))}
      </dl>
    </div>
  )
}

function OfflineState() {
  return (
    <Reveal className="max-w-2xl">
      <Label className="text-accent">Research API offline</Label>
      <p className="mt-5 text-pretty text-2xl leading-snug text-foreground/90 md:text-3xl">
        Start the local MomentumAlpha API to load saved runs, metrics, and evidence.
      </p>
      <p className="mt-6 font-mono text-xs leading-relaxed text-muted-foreground">
        The interface reads from the same origin as FastAPI (default http://127.0.0.1:8000). No sample numbers are shown
        while the API is unavailable.
      </p>
    </Reveal>
  )
}

function EmptyState() {
  return (
    <Reveal className="max-w-2xl">
      <Label className="text-accent">No research runs yet</Label>
      <p className="mt-5 text-pretty text-2xl leading-snug text-foreground/90 md:text-3xl">
        Launch a first development run to populate this terminal. The final holdout stays untouched until you choose to
        spend it.
      </p>
      <a
        href="#experiments"
        className="mt-8 inline-block border-b border-accent pb-1 font-mono text-[11px] uppercase tracking-[0.26em] text-accent"
      >
        Go to experiments
      </a>
    </Reveal>
  )
}
