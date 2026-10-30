"use client"

import useSWR from "swr"
import { api, isActive, isCompleted, loadRunBundle, normalizeRuns } from "@/lib/api-client"
import type { RunRow } from "@/lib/api-types"

export function useApiStatus() {
  const health = useSWR("health", api.health, { refreshInterval: 30000, shouldRetryOnError: false })
  const ready = useSWR("ready", api.ready, { refreshInterval: 30000, shouldRetryOnError: false })
  const online = Boolean(health.data) && !health.error
  const loading = health.isLoading
  const isReady = online && !ready.error && Boolean(ready.data) && ready.data?.ready !== false
  return { online, loading, isReady, version: health.data?.version, error: health.error as Error | undefined }
}

export function useRuns() {
  const swr = useSWR("runs", api.runs, {
    shouldRetryOnError: false,
    keepPreviousData: true,
    refreshInterval: (latest) => (normalizeRuns(latest).some(isActive) ? 3000 : 15000),
  })
  const rows: RunRow[] = normalizeRuns(swr.data)
  return {
    rows,
    activeJob: rows.find(isActive),
    latestCompleted: rows.find((run) => isCompleted(run)),
    isLoading: swr.isLoading,
    error: swr.error as Error | undefined,
    refresh: swr.mutate,
  }
}

export function useRunBundle(runId: string | null, statusKey?: string) {
  return useSWR(runId ? ["bundle", runId, statusKey ?? ""] : null, ([, id]) => loadRunBundle(id), {
    keepPreviousData: true,
    shouldRetryOnError: false,
    revalidateOnFocus: false,
  })
}
