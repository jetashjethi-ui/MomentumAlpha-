"use client"

import { useState } from "react"
import { SiteHeader } from "./site-header"
import { ParticleWave } from "./particle-wave"
import { PointerOrb } from "./pointer-orb"
import { OverviewChapter } from "./overview-chapter"
import { PerformanceChapter } from "./performance-chapter"
import { EvidenceChapter } from "./evidence-chapter"
import { DataChapter } from "./data-chapter"
import { ExperimentsChapter } from "./experiments-chapter"
import { useApiStatus, useRunBundle, useRuns } from "@/hooks/use-research"

export function ResearchTerminal() {
  const status = useApiStatus()
  const runs = useRuns()
  const [chosenId, setChosenId] = useState<string | null>(null)

  const selectedId = chosenId ?? runs.latestCompleted?.run_id ?? runs.rows[0]?.run_id ?? null
  const selectedRun = runs.rows.find((run) => run.run_id === selectedId)
  const bundle = useRunBundle(status.online ? selectedId : null, selectedRun?.status)
  const data = status.online ? bundle.data : undefined

  return (
    <>
      <a
        href="#main"
        className="sr-only z-[60] bg-accent px-4 py-2 font-mono text-xs text-accent-foreground focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
      >
        Skip to content
      </a>
      <ParticleWave />
      <PointerOrb />
      <SiteHeader online={status.online} loading={status.loading} />
      <main id="main" className="relative z-10">
        <OverviewChapter
          online={status.online}
          apiLoading={status.loading}
          rows={runs.rows}
          selectedId={selectedId}
          onSelect={setChosenId}
          bundle={data}
          bundleLoading={bundle.isValidating}
          bundleError={bundle.error as Error | undefined}
        />
        <PerformanceChapter bundle={data} />
        <EvidenceChapter bundle={data} />
        <DataChapter bundle={data} />
        <ExperimentsChapter
          online={status.online}
          rows={runs.rows}
          selectedId={selectedId}
          onSelect={setChosenId}
          onLaunched={(id) => {
            setChosenId(id)
            runs.refresh()
          }}
        />
      </main>
      <footer className="relative z-10 px-5 pb-16 pt-8 md:px-12 lg:pl-28 lg:pr-16">
        <p className="max-w-3xl font-mono text-[11px] leading-relaxed text-muted-foreground">
          MomentumAlpha · local research terminal. Historical backtests only, not investment advice. Past performance
          under modeled costs does not predict future results.
          {status.version ? ` API ${status.version}.` : ""}
        </p>
      </footer>
    </>
  )
}
