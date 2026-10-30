import type {
  ArtifactEntry,
  CsvRecord,
  HealthPayload,
  LaunchPayload,
  LaunchResponse,
  ReadyPayload,
  RunBundle,
  RunRow,
  RunsPayload,
  RunSummary,
  VerificationReport,
} from "./api-types"
import { parseCsv } from "./csv"

// Same-origin by default. Set NEXT_PUBLIC_MOMENTUM_API_BASE only when the UI is
// served from a different origin than FastAPI.
const BASE = (process.env.NEXT_PUBLIC_MOMENTUM_API_BASE ?? "").replace(/\/$/, "")

const RUN_ID = /^[A-Za-z0-9-]{1,64}$/

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

function safeRunId(runId: string) {
  if (!RUN_ID.test(runId)) throw new ApiError("Invalid run identifier.", 400)
  return encodeURIComponent(runId)
}

function detailMessage(detail: unknown, status: number): string {
  if (typeof detail === "string") return detail
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          const loc = Array.isArray((item as { loc?: unknown[] }).loc)
            ? (item as { loc: unknown[] }).loc.slice(1).join(".")
            : ""
          return (loc ? loc + ": " : "") + String((item as { msg: unknown }).msg)
        }
        return JSON.stringify(item)
      })
      .join(" · ")
  }
  if (detail) return JSON.stringify(detail)
  return "Request failed with status " + status
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(BASE + path, { cache: "no-store", ...init })
  } catch {
    throw new ApiError("The local research API is unreachable.", 0)
  }
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new ApiError(detailMessage((body as { detail?: unknown }).detail, response.status), response.status)
  }
  return body as T
}

async function artifactText(runId: string, name: string): Promise<string> {
  const response = await fetch(BASE + "/runs/" + safeRunId(runId) + "/artifacts/" + encodeURIComponent(name), {
    cache: "no-store",
  })
  if (!response.ok) throw new ApiError("Artifact unavailable: " + name, response.status)
  return response.text()
}

export function artifactUrl(runId: string, name: string) {
  return BASE + "/runs/" + safeRunId(runId) + "/artifacts/" + encodeURIComponent(name)
}

export const api = {
  health: () => request<HealthPayload>("/health"),
  ready: () => request<ReadyPayload>("/ready"),
  runs: () => request<RunsPayload>("/runs?limit=100"),
  run: (runId: string) => request<RunRow>("/runs/" + safeRunId(runId)),
  result: (runId: string) => request<RunSummary>("/runs/" + safeRunId(runId) + "/result"),
  verification: (runId: string) => request<VerificationReport>("/runs/" + safeRunId(runId) + "/verification"),
  artifacts: async (runId: string): Promise<ArtifactEntry[]> => {
    const body = await request<unknown>("/runs/" + safeRunId(runId) + "/artifacts")
    if (Array.isArray(body)) return body.map((item) => (typeof item === "string" ? { name: item } : item))
    const list = (body as { artifacts?: unknown[] }).artifacts
    if (Array.isArray(list)) return list.map((item) => (typeof item === "string" ? { name: item } : (item as ArtifactEntry)))
    return []
  },
  launch: (payload: LaunchPayload) =>
    request<LaunchResponse>("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
}

async function optionalCsv(runId: string, name: string): Promise<CsvRecord[] | null> {
  try {
    return parseCsv(await artifactText(runId, name))
  } catch {
    return null
  }
}

async function optional<T>(promise: Promise<T>): Promise<T | null> {
  try {
    return await promise
  } catch {
    return null
  }
}

export async function loadRunBundle(runId: string): Promise<RunBundle> {
  const summary = await api.result(runId)
  const [verification, artifacts, returns, costs, features, factors] = await Promise.all([
    optional(api.verification(runId)),
    optional(api.artifacts(runId)),
    optionalCsv(runId, "portfolio_returns.csv"),
    optionalCsv(runId, "cost_sensitivity.csv"),
    optionalCsv(runId, "feature_importance_stability.csv"),
    optionalCsv(runId, "factor_attribution.csv"),
  ])
  return { summary, verification, artifacts, returns, costs, features, factors }
}

export function normalizeRuns(payload: RunsPayload | undefined): RunRow[] {
  if (!payload) return []
  const jobs = Array.isArray(payload.jobs) ? payload.jobs : []
  const completed = Array.isArray(payload.completed_runs) ? payload.completed_runs : []
  const ids = new Set(jobs.map((run) => run.run_id))
  const merged = jobs.concat(completed.filter((run) => !ids.has(run.run_id)))
  return merged.sort((a, b) => {
    const da = Date.parse(a.created_at_utc || a.updated_at_utc || "") || 0
    const db = Date.parse(b.created_at_utc || b.updated_at_utc || "") || 0
    return db - da
  })
}

export function runMode(run: RunRow): string {
  return run.request?.evaluation_mode || run.evaluation_mode || "development"
}

export function runSource(run: RunRow): string {
  const source = String(run.request?.data_source || run.data_source || run.strategy_summary?.data_source || "—")
  const lower = source.toLowerCase()
  if (lower.includes("dolt")) return "Dolt"
  if (lower.includes("yahoo")) return "Yahoo"
  return source
}

export function isCompleted(run: RunRow) {
  return run.status === "completed" || (!run.status && (Boolean(run.strategy_summary) || Boolean(run.run_dir)))
}

export function isActive(run: RunRow) {
  return run.status === "queued" || run.status === "running"
}
