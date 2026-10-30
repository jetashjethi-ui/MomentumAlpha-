"use client"

import { useState } from "react"
import { Check, Copy } from "lucide-react"
import { Chapter, Label, Reveal, Unavailable } from "./chapter"
import type { RunBundle } from "@/lib/api-types"
import { artifactUrl } from "@/lib/api-client"
import { dateLabel, integer } from "@/lib/formatters"
import { cn } from "@/lib/utils"

export function DataChapter({ bundle }: { bundle: RunBundle | undefined }) {
  return (
    <Chapter
      id="data"
      index="04"
      label="Data"
      title="Provenance"
      aside={
        <p className="font-mono text-xs leading-relaxed text-muted-foreground">
          Where the prices came from, which securities were eligible, and whether the saved artifacts still match.
        </p>
      }
    >
      {!bundle ? <Unavailable>Provenance appears once a completed run is selected.</Unavailable> : <DataBody bundle={bundle} />}
    </Chapter>
  )
}

function DataBody({ bundle }: { bundle: RunBundle }) {
  const { summary, verification, artifacts } = bundle
  const data = summary.data || {}
  const quality = data.quality_summary || {}
  const panel = data.panel_validation || {}
  const flags =
    Number(panel.weeks_with_misaligned_execution_dates || 0) +
    Number(panel.weeks_with_incomplete_universe || 0) +
    Number(quality.symbols_with_invalid_prices || 0)

  const universe =
    panel.minimum_tickers_per_week == null
      ? "—"
      : panel.minimum_tickers_per_week === panel.maximum_tickers_per_week
        ? `${panel.minimum_tickers_per_week} / week`
        : `${panel.minimum_tickers_per_week}–${panel.maximum_tickers_per_week ?? "—"} / week`

  const facts = [
    { label: "Source", value: data.provider || "Not recorded" },
    { label: "Observed", value: `${dateLabel(data.observed_start)} — ${dateLabel(data.observed_end)}` },
    { label: "Symbols", value: integer(quality.symbols) },
    { label: "Universe", value: universe },
  ]
  const checks = [
    { label: "Invalid price symbols", value: quality.symbols_with_invalid_prices },
    { label: "OHLC inconsistencies", value: quality.symbols_with_ohlc_inconsistencies, unit: "symbols" },
    { label: "Incomplete universe", value: panel.weeks_with_incomplete_universe, unit: "weeks" },
    { label: "Misaligned execution", value: panel.weeks_with_misaligned_execution_dates, unit: "weeks" },
  ]

  const limits = [...(Array.isArray(data.known_data_and_execution_limits) ? data.known_data_and_execution_limits.slice(0, 6) : [])]
  if (panel.weeks_with_misaligned_execution_dates) {
    limits.unshift(
      `${panel.weeks_with_misaligned_execution_dates} weeks have execution-date alignment flags; inspect panel validation before interpreting results.`,
    )
  }

  return (
    <div className="space-y-28">
      <dl className="grid gap-x-8 gap-y-12 sm:grid-cols-2 xl:grid-cols-4">
        {facts.map((fact, index) => (
          <Reveal key={fact.label} from={index % 2 ? "right" : "left"} delay={index * 70}>
            <dt>
              <Label>{fact.label}</Label>
            </dt>
            <dd className="mt-3 text-pretty text-2xl leading-tight text-foreground">{fact.value}</dd>
          </Reveal>
        ))}
      </dl>

      <div className="grid gap-16 lg:grid-cols-12">
        <Reveal from="left" className="lg:col-span-5">
          <Label>Universe and liquidity rules</Label>
          <p className="mt-5 text-pretty text-xl leading-snug text-foreground/85">
            {data.universe_policy || "Universe policy not recorded."}
          </p>
        </Reveal>
        <Reveal from="right" delay={120} className="lg:col-span-6 lg:col-start-7">
          <div className="flex items-baseline justify-between gap-4">
            <Label>Validation checks</Label>
            <span className={cn("font-mono text-[10px] uppercase tracking-[0.24em]", flags > 0 ? "text-accent" : "text-foreground/70")}>
              {flags > 0 ? "Review flags" : "Checks clear"}
            </span>
          </div>
          <dl className="mt-6">
            {checks.map((check) => (
              <div key={check.label} className="flex items-baseline justify-between gap-4 py-2.5">
                <dt className="text-sm text-muted-foreground">{check.label}</dt>
                <dd
                  className={cn(
                    "font-mono text-sm tabular-nums",
                    Number(check.value || 0) > 0 ? "text-accent" : "text-foreground",
                  )}
                >
                  {check.value == null ? "—" : `${check.value}${check.unit ? ` ${check.unit}` : ""}`}
                </dd>
              </div>
            ))}
          </dl>
        </Reveal>
      </div>

      <div className="grid gap-16 lg:grid-cols-12">
        <Reveal from="left" className="lg:col-span-7">
          <Label>Data fingerprint · SHA-256</Label>
          <Fingerprint value={data.fingerprint_sha256} />
          <div className="mt-12">
            <Label>Run-artifact verification</Label>
            <p className="mt-3 text-xl text-foreground">
              {verification == null
                ? "Verification unavailable"
                : verification.valid
                  ? `Verified · ${verification.checked_artifacts ?? "—"} artifacts checked`
                  : "Verification needs review"}
            </p>
            {verification && !verification.valid && Array.isArray(verification.errors) && verification.errors.length ? (
              <ul className="mt-3 space-y-1 font-mono text-xs text-accent">
                {verification.errors.slice(0, 5).map((error) => (
                  <li key={error}>{error}</li>
                ))}
              </ul>
            ) : null}
          </div>
        </Reveal>

        <Reveal from="right" delay={120} className="lg:col-span-4 lg:col-start-9">
          <Label>Artifacts</Label>
          {artifacts && artifacts.length ? (
            <ul className="mt-5 max-h-72 space-y-1.5 overflow-y-auto pr-2">
              {artifacts.map((artifact) =>
                artifact.name ? (
                  <li key={artifact.name}>
                    <a
                      href={artifactUrl(summary.run_id, artifact.name)}
                      target="_blank"
                      rel="noreferrer"
                      className="font-mono text-xs text-foreground/80 underline-offset-4 transition-colors hover:text-accent hover:underline"
                    >
                      {artifact.name}
                    </a>
                  </li>
                ) : null,
              )}
            </ul>
          ) : (
            <Unavailable>Artifact inventory unavailable.</Unavailable>
          )}
          <a
            href={artifactUrl(summary.run_id, "run_manifest.json")}
            target="_blank"
            rel="noreferrer"
            className="mt-6 inline-block border-b border-foreground/30 pb-1 font-mono text-[10px] uppercase tracking-[0.24em] transition-colors hover:border-accent hover:text-accent"
          >
            Open run manifest
          </a>
        </Reveal>
      </div>

      <Reveal className="max-w-3xl">
        <Label>Known data and execution limits</Label>
        <ul className="mt-6 space-y-4">
          {(limits.length ? limits : ["No run-specific caveats were recorded in the summary."]).map((item) => (
            <li key={item} className="flex gap-4 text-pretty text-base leading-relaxed text-foreground/80">
              <span aria-hidden="true" className="mt-2.5 h-px w-4 shrink-0 bg-accent" />
              {item}
            </li>
          ))}
        </ul>
      </Reveal>
    </div>
  )
}

function Fingerprint({ value }: { value?: string }) {
  const [copied, setCopied] = useState(false)
  if (!value) return <p className="mt-3 font-mono text-sm text-muted-foreground">Not recorded</p>
  return (
    <div className="mt-3 flex items-start gap-4">
      <code className="break-all font-mono text-sm leading-relaxed text-foreground">{value}</code>
      <button
        type="button"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(value)
            setCopied(true)
            window.setTimeout(() => setCopied(false), 1800)
          } catch {
            setCopied(false)
          }
        }}
        className="shrink-0 p-1 text-muted-foreground transition-colors hover:text-accent"
      >
        {copied ? <Check className="h-4 w-4 text-accent" /> : <Copy className="h-4 w-4" />}
        <span className="sr-only">{copied ? "Fingerprint copied" : "Copy fingerprint"}</span>
      </button>
    </div>
  )
}
