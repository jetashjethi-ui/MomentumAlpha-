"use client"

import { useState, type FormEvent } from "react"
import { ArrowUpRight } from "lucide-react"
import { Chapter, Label, Reveal, Unavailable } from "./chapter"
import type { LaunchPayload, RunRow } from "@/lib/api-types"
import { api, isActive, runMode, runSource } from "@/lib/api-client"
import { dateLabel, decimal, modeLabel, pct } from "@/lib/formatters"
import { cn } from "@/lib/utils"

const DEFAULTS: LaunchPayload = {
  start_date: "2012-01-01",
  data_source: "dolt",
  initial_train_end: "2022-12-31",
  final_holdout_weeks: 52,
  evaluation_mode: "development",
  retrain_freq: 13,
  top_n: 5,
  transaction_cost_bps: 10,
  n_jobs: 4,
  random_simulations: 2000,
  bootstrap_samples: 2000,
  universe_size: 50,
  min_price: 5,
  min_dollar_volume: 5000000,
  min_history_bars: 252,
  ticker_gap_days: 180,
  allow_consumed_holdout: false,
  include_plots: false,
}

type NumberKey = {
  [K in keyof LaunchPayload]: LaunchPayload[K] extends number ? K : never
}[keyof LaunchPayload]

const NUMERIC: { key: NumberKey; label: string; min: number; max: number; step?: number; advanced?: boolean }[] = [
  { key: "top_n", label: "Holdings", min: 1, max: 50 },
  { key: "transaction_cost_bps", label: "Cost · bps", min: 0, max: 1000, step: 0.5 },
  { key: "retrain_freq", label: "Retrain · weeks", min: 1, max: 260 },
  { key: "final_holdout_weeks", label: "Reserved holdout · weeks", min: 1, max: 260 },
  { key: "universe_size", label: "Universe size", min: 2, max: 500, advanced: true },
  { key: "min_price", label: "Min price", min: 0.01, max: 1000000000, step: 0.5, advanced: true },
  { key: "min_dollar_volume", label: "Min dollar volume", min: 0, max: 1e12, step: 100000, advanced: true },
  { key: "min_history_bars", label: "Min history bars", min: 60, max: 5000, advanced: true },
  { key: "ticker_gap_days", label: "Max ticker gap · days", min: 30, max: 3650, advanced: true },
  { key: "random_simulations", label: "Random simulations", min: 100, max: 100000, advanced: true },
  { key: "bootstrap_samples", label: "Bootstrap samples", min: 100, max: 100000, advanced: true },
  { key: "n_jobs", label: "Parallel jobs", min: 1, max: 256, advanced: true },
]

interface Props {
  online: boolean
  rows: RunRow[]
  selectedId: string | null
  onSelect: (id: string) => void
  onLaunched: (id: string) => void
}

export function ExperimentsChapter({ online, rows, selectedId, onSelect, onLaunched }: Props) {
  const active = rows.find(isActive)
  return (
    <Chapter
      id="experiments"
      index="05"
      label="Experiments"
      title="Run the next study"
      aside={
        <p className="font-mono text-xs leading-relaxed text-muted-foreground">
          Development runs are cheap and repeatable. The final holdout can be spent once; it is kept behind a separate,
          explicit confirmation.
        </p>
      }
    >
      <div className="grid gap-20 lg:grid-cols-12">
        <div className="lg:col-span-7">
          <LaunchForm online={online} busy={Boolean(active)} onLaunched={onLaunched} />
        </div>
        <div className="lg:col-span-4 lg:col-start-9">
          <JobStatus job={active ?? rows.find((run) => run.run_id === selectedId)} />
        </div>
      </div>
      <RunHistory rows={rows} selectedId={selectedId} onSelect={onSelect} online={online} />
    </Chapter>
  )
}

function LaunchForm({ online, busy, onLaunched }: { online: boolean; busy: boolean; onLaunched: (id: string) => void }) {
  const [form, setForm] = useState<LaunchPayload>(DEFAULTS)
  const [holdoutConfirmed, setHoldoutConfirmed] = useState(false)
  const [submitting, setSubmitting] = useState<"development" | "final_holdout" | null>(null)
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null)

  const set = <K extends keyof LaunchPayload>(key: K, value: LaunchPayload[K]) => setForm((prev) => ({ ...prev, [key]: value }))

  const validate = () => {
    const start = Date.parse(form.start_date)
    const train = Date.parse(form.initial_train_end)
    if (!Number.isFinite(start) || !Number.isFinite(train)) return "Both dates are required."
    if (train <= start) return "Initial training end must be after the start date."
    for (const field of NUMERIC) {
      const value = form[field.key]
      if (!Number.isFinite(value) || value < field.min || value > field.max) {
        return `${field.label} must be between ${field.min} and ${field.max}.`
      }
    }
    const portfolioCapacity = form.data_source === "dolt" ? form.universe_size : 10
    if (form.top_n > portfolioCapacity) {
      return `Holdings cannot exceed the ${portfolioCapacity}-security capacity of this data source.`
    }
    return null
  }

  const submit = async (mode: "development" | "final_holdout") => {
    const problem = validate()
    if (problem) {
      setMessage({ tone: "error", text: problem })
      return
    }
    if (mode === "final_holdout" && !holdoutConfirmed) {
      setMessage({ tone: "error", text: "Confirm that you intend to spend the final holdout." })
      return
    }
    setSubmitting(mode)
    setMessage(null)
    try {
      const response = await api.launch({ ...form, evaluation_mode: mode, allow_consumed_holdout: false })
      setMessage({ tone: "ok", text: `Run ${response.run_id} queued · ${modeLabel(mode)}.` })
      setHoldoutConfirmed(false)
      onLaunched(response.run_id)
    } catch (error) {
      setMessage({ tone: "error", text: error instanceof Error ? error.message : "Launch failed." })
    } finally {
      setSubmitting(null)
    }
  }

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    submit("development")
  }

  const disabled = !online || busy || submitting != null

  return (
    <Reveal from="left">
      <form onSubmit={onSubmit} noValidate aria-describedby="launch-message">
        <fieldset disabled={!online} className="space-y-10">
          <legend className="sr-only">Development run configuration</legend>
          <div className="grid gap-x-8 gap-y-8 sm:grid-cols-2">
            <Field label="Start date" id="start_date">
              <input id="start_date" type="date" value={form.start_date} onChange={(e) => set("start_date", e.target.value)} className="ma-input" />
            </Field>
            <Field label="Initial training end" id="initial_train_end">
              <input
                id="initial_train_end"
                type="date"
                value={form.initial_train_end}
                onChange={(e) => set("initial_train_end", e.target.value)}
                className="ma-input"
              />
            </Field>
            <Field label="Data source" id="data_source">
              <select id="data_source" value={form.data_source} onChange={(e) => set("data_source", e.target.value)} className="ma-input">
                <option value="dolt" className="bg-background">Dolt · stocks</option>
                <option value="yahoo" className="bg-background">Yahoo Finance</option>
              </select>
            </Field>
            {NUMERIC.filter((field) => !field.advanced).map((field) => {
              const supportedField = field.key === "top_n"
                ? { ...field, max: Math.min(field.max, form.data_source === "dolt" ? form.universe_size : 10) }
                : field
              return <NumberField key={field.key} field={supportedField} value={form[field.key]} onChange={(value) => set(field.key, value)} />
            })}
          </div>

          <details className="group">
            <summary className="cursor-pointer list-none font-mono text-[10px] uppercase tracking-[0.26em] text-muted-foreground transition-colors hover:text-foreground">
              <span className="mr-2 inline-block transition-transform group-open:rotate-90">›</span>
              Universe, statistics and execution
            </summary>
            <div className="mt-8 grid gap-x-8 gap-y-8 sm:grid-cols-2">
              {NUMERIC.filter((field) => field.advanced).map((field) => (
                <NumberField key={field.key} field={field} value={form[field.key]} onChange={(value) => set(field.key, value)} />
              ))}
              <label className="flex items-center gap-3 font-mono text-xs text-foreground/80">
                <input
                  type="checkbox"
                  checked={form.include_plots}
                  onChange={(e) => set("include_plots", e.target.checked)}
                  className="h-4 w-4 accent-[#ff6b1a]"
                />
                Render diagnostic plots
              </label>
            </div>
          </details>

          <div className="flex flex-wrap items-center gap-6">
            <button
              type="submit"
              disabled={disabled}
              className="group inline-flex items-center gap-3 border border-foreground/25 px-6 py-3.5 font-mono text-[11px] uppercase tracking-[0.26em] transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-foreground/25 disabled:hover:text-foreground"
            >
              {submitting === "development" ? "Queuing…" : "Launch development run"}
              <ArrowUpRight className="h-4 w-4 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
            </button>
            <button
              type="button"
              disabled={!online}
              onClick={() => {
                setForm(DEFAULTS)
                setMessage(null)
              }}
              className="font-mono text-[10px] uppercase tracking-[0.24em] text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40"
            >
              Reset defaults
            </button>
          </div>

          <details className="group pt-6">
            <summary className="cursor-pointer list-none font-mono text-[10px] uppercase tracking-[0.26em] text-accent">
              <span className="mr-2 inline-block transition-transform group-open:rotate-90">›</span>
              Final holdout evaluation · irreversible
            </summary>
            <div className="mt-6 max-w-xl space-y-5">
              <p className="text-pretty text-sm leading-relaxed text-foreground/80">
                Scores the reserved {form.final_holdout_weeks}-week window with the configuration above. Once inspected, the
                window is consumed: results can no longer count as independent validation, and further tuning against it
                is not allowed.
              </p>
              <label className="flex items-start gap-3 font-mono text-xs leading-relaxed text-foreground/80">
                <input
                  type="checkbox"
                  checked={holdoutConfirmed}
                  onChange={(e) => setHoldoutConfirmed(e.target.checked)}
                  className="mt-0.5 h-4 w-4 accent-[#ff6b1a]"
                />
                I have frozen this configuration and intend to spend the final holdout.
              </label>
              <button
                type="button"
                disabled={disabled || !holdoutConfirmed}
                onClick={() => submit("final_holdout")}
                className="inline-flex items-center gap-3 border border-accent/60 px-6 py-3.5 font-mono text-[11px] uppercase tracking-[0.26em] text-accent transition-colors hover:bg-accent hover:text-accent-foreground disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-accent"
              >
                {submitting === "final_holdout" ? "Queuing…" : "Run final holdout"}
              </button>
            </div>
          </details>
        </fieldset>

        <p id="launch-message" role="status" aria-live="polite" className="mt-8 min-h-5 font-mono text-xs">
          {!online ? (
            <span className="text-muted-foreground">Launching is disabled while the research API is offline.</span>
          ) : busy ? (
            <span className="text-muted-foreground">A run is already in progress; launch the next one when it completes.</span>
          ) : message ? (
            <span className={message.tone === "error" ? "text-accent" : "text-foreground"}>{message.text}</span>
          ) : null}
        </p>
      </form>
    </Reveal>
  )
}

function Field({ label, id, children }: { label: string; id: string; children: React.ReactNode }) {
  return (
    <div>
      <label htmlFor={id}>
        <Label>{label}</Label>
      </label>
      <div className="mt-2">{children}</div>
    </div>
  )
}

function NumberField({
  field,
  value,
  onChange,
}: {
  field: (typeof NUMERIC)[number]
  value: number
  onChange: (value: number) => void
}) {
  return (
    <Field label={field.label} id={field.key}>
      <input
        id={field.key}
        type="number"
        inputMode="decimal"
        min={field.min}
        max={field.max}
        step={field.step ?? 1}
        value={Number.isFinite(value) ? value : ""}
        onChange={(e) => onChange(e.target.value === "" ? Number.NaN : Number(e.target.value))}
        className="ma-input tabular-nums"
      />
    </Field>
  )
}

function JobStatus({ job }: { job: RunRow | undefined }) {
  return (
    <Reveal from="right" delay={120}>
      <Label>Job status</Label>
      {!job ? (
        <Unavailable>No job selected. Launch or select a run to see its status.</Unavailable>
      ) : (
        <div className="mt-4" aria-live="polite">
          <p className="flex items-center gap-3">
            <span
              aria-hidden="true"
              className={cn(
                "h-2 w-2 rounded-full",
                isActive(job) ? "animate-pulse bg-accent" : job.status === "failed" ? "bg-accent" : "bg-foreground/50",
              )}
            />
            <span className="font-display text-5xl uppercase leading-none">{job.status || "completed"}</span>
          </p>
          <p className="mt-3 font-mono text-xs text-muted-foreground">
            {job.run_id} · {modeLabel(runMode(job))} · updated {dateLabel(job.updated_at_utc || job.created_at_utc)}
          </p>
          {job.error ? <p className="mt-4 font-mono text-xs leading-relaxed text-accent">{job.error}</p> : null}
          {job.log_tail ? (
            <pre className="mt-6 max-h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-foreground/70">
              {job.log_tail.split("\n").slice(-20).join("\n")}
            </pre>
          ) : null}
        </div>
      )}
    </Reveal>
  )
}

function RunHistory({
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
    <Reveal className="mt-32">
      <div className="flex items-baseline justify-between gap-4">
        <Label>Run history</Label>
        <span className="font-mono text-[10px] uppercase tracking-[0.24em] text-muted-foreground">{rows.length} runs</span>
      </div>
      {!rows.length ? (
        <div className="mt-6">
          <Unavailable>{online ? "No runs recorded yet." : "Run history is unavailable while the API is offline."}</Unavailable>
        </div>
      ) : (
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[720px] text-left">
            <caption className="sr-only">Saved research runs; select a row to load it</caption>
            <thead>
              <tr>
                {["Run", "Mode", "Source", "Status", "Annualized", "Sharpe", "Created"].map((head, index) => (
                  <th
                    key={head}
                    scope="col"
                    className={cn(
                      "pb-4 font-mono text-[10px] font-normal uppercase tracking-[0.24em] text-muted-foreground",
                      index >= 4 && index <= 5 && "text-right",
                    )}
                  >
                    {head}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 30).map((run) => {
                const metrics = run.strategy_summary?.strategy_after_costs || {}
                const selected = run.run_id === selectedId
                return (
                  <tr key={run.run_id} className={cn("transition-colors hover:bg-foreground/[0.03]", selected && "text-accent")}>
                    <th scope="row" className="py-2.5 pr-4 font-normal">
                      <button
                        type="button"
                        onClick={() => {
                          onSelect(run.run_id)
                          document.getElementById("overview")?.scrollIntoView({ behavior: "smooth" })
                        }}
                        aria-current={selected ? "true" : undefined}
                        className="font-mono text-xs underline-offset-4 hover:underline"
                      >
                        {run.run_id}
                      </button>
                    </th>
                    <td className="py-2.5 pr-4 font-mono text-xs capitalize">{modeLabel(runMode(run))}</td>
                    <td className="py-2.5 pr-4 font-mono text-xs">{runSource(run)}</td>
                    <td className="py-2.5 pr-4 font-mono text-xs">{run.status || "completed"}</td>
                    <td className="py-2.5 pr-4 text-right font-mono text-xs tabular-nums">{pct(metrics["Annualized Return"])}</td>
                    <td className="py-2.5 pr-4 text-right font-mono text-xs tabular-nums">{decimal(metrics["Sharpe Ratio"])}</td>
                    <td className="py-2.5 font-mono text-xs text-muted-foreground">{dateLabel(run.created_at_utc)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Reveal>
  )
}
