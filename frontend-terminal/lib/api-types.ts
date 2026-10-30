// Shapes mirror the fields the existing MomentumAlpha dashboard reads.
// `/openapi.json` on the FastAPI backend remains the source of truth.

export type RunStatus = "queued" | "running" | "completed" | "failed" | "interrupted" | string

export type EvaluationMode = "development" | "final_holdout" | string

export interface MetricBlock {
  "Annualized Return"?: number
  "Sharpe Ratio"?: number
  "Max Drawdown"?: number
  "Annualized Volatility"?: number
  "CVaR (95%)"?: number
  "Num Weeks"?: number
  [key: string]: number | undefined
}

export interface RunRequest {
  evaluation_mode?: EvaluationMode
  data_source?: string
  [key: string]: unknown
}

export interface RunRow {
  run_id: string
  status?: RunStatus
  created_at_utc?: string
  updated_at_utc?: string
  evaluation_mode?: EvaluationMode
  data_source?: string
  request?: RunRequest
  strategy_summary?: {
    strategy_after_costs?: MetricBlock
    data_source?: string
    [key: string]: unknown
  }
  run_dir?: string
  log_tail?: string
  error?: string | null
}

export interface RunsPayload {
  jobs?: RunRow[]
  completed_runs?: RunRow[]
}

export interface CalibrationPoint {
  calibration_type?: string
  mean_predicted_probability?: number
  observed_positive_rate?: number
  observations?: number
}

export interface PhasePerformance {
  weeks?: number
  period_start?: string
  period_end?: string
  strategy_after_costs?: MetricBlock
  benchmarks_after_costs?: {
    equal_weight?: MetricBlock
    spy?: MetricBlock
    raw_momentum?: MetricBlock
    [key: string]: MetricBlock | undefined
  }
  classification?: {
    roc_auc?: number
    brier_score?: number
    expected_calibration_error_10_bins?: number
    [key: string]: number | undefined
  }
}

export interface RunSummary {
  run_id: string
  configuration?: {
    evaluation_mode?: EvaluationMode
    transaction_cost_bps?: number
    transaction_cost_bps_per_traded_notional?: number
    evaluated_final_holdout_weeks?: number
    reserved_final_holdout_weeks?: number
    final_holdout_start?: string
    [key: string]: unknown
  }
  performance?: PhasePerformance & {
    development?: PhasePerformance
    final_holdout?: PhasePerformance
    mean_turnover?: number
    sharpe_95pct_block_bootstrap_ci?: number[]
    annualized_excess_return_95pct_ci?: Record<string, number[] | undefined>
    information_coefficient_mean?: number
    random_pick_cost_matched_p_value?: number
    probability_calibration_curve?: CalibrationPoint[]
  }
  data?: {
    provider?: string
    observed_start?: string
    observed_end?: string
    fingerprint_sha256?: string
    universe_policy?: string
    known_data_and_execution_limits?: string[]
    quality_summary?: {
      symbols?: number
      symbols_with_invalid_prices?: number
      symbols_with_ohlc_inconsistencies?: number
    }
    panel_validation?: {
      minimum_tickers_per_week?: number
      maximum_tickers_per_week?: number
      weeks_with_incomplete_universe?: number
      weeks_with_misaligned_execution_dates?: number
    }
  }
}

export interface VerificationReport {
  valid?: boolean
  checked_artifacts?: number
  errors?: string[]
  [key: string]: unknown
}

export interface ArtifactEntry {
  name?: string
  size_bytes?: number
  [key: string]: unknown
}

export interface ReadyPayload {
  ready?: boolean
  status?: string
  [key: string]: unknown
}

export interface HealthPayload {
  status?: string
  version?: string
  [key: string]: unknown
}

export interface LaunchPayload {
  start_date: string
  data_source: string
  initial_train_end: string
  final_holdout_weeks: number
  evaluation_mode: "development" | "final_holdout"
  retrain_freq: number
  top_n: number
  transaction_cost_bps: number
  n_jobs: number
  random_simulations: number
  bootstrap_samples: number
  universe_size: number
  min_price: number
  min_dollar_volume: number
  min_history_bars: number
  ticker_gap_days: number
  allow_consumed_holdout: boolean
  include_plots: boolean
}

export interface LaunchResponse {
  run_id: string
  status?: RunStatus
  [key: string]: unknown
}

export type CsvRecord = Record<string, string>

export interface RunBundle {
  summary: RunSummary
  verification: VerificationReport | null
  artifacts: ArtifactEntry[] | null
  returns: CsvRecord[] | null
  costs: CsvRecord[] | null
  features: CsvRecord[] | null
  factors: CsvRecord[] | null
}
