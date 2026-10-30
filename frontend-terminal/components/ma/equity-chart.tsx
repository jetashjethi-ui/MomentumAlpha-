"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import type { EquityPoint } from "@/lib/derive"
import { clamp, dateLabel, monthLabel, pct, signedPct } from "@/lib/formatters"

const RANGES = [
  { key: "52", label: "1Y", weeks: 52 },
  { key: "156", label: "3Y", weeks: 156 },
  { key: "260", label: "5Y", weeks: 260 },
  { key: "all", label: "All", weeks: Number.POSITIVE_INFINITY },
]

const HEIGHT = 380
const DD_HEIGHT = 90
const PAD = { left: 52, right: 12, top: 16, bottom: 28 }

export function EquityChart({ series }: { series: EquityPoint[] }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(960)
  const [range, setRange] = useState("all")
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    const node = wrapRef.current
    if (!node) return
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)))
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  const available = RANGES.filter((item) => item.key === "all" || series.length > item.weeks)

  const data = useMemo(() => {
    const weeks = RANGES.find((item) => item.key === range)?.weeks ?? Number.POSITIVE_INFINITY
    const slice = Number.isFinite(weeks) ? series.slice(-weeks) : series
    if (!slice.length) return []
    // Rebase the visible window to 1.0 so range views read as growth of one unit.
    const base = slice[0].equity / (1 + slice[0].ret)
    let peak = 1
    return slice.map((point) => {
      const equity = point.equity / base
      peak = Math.max(peak, equity)
      return { ...point, equity, drawdown: equity / peak - 1 }
    })
  }, [series, range])

  const geometry = useMemo(() => {
    if (!data.length) return null
    const values = data.map((point) => point.equity)
    let min = Math.min(1, ...values)
    let max = Math.max(1, ...values)
    const pad = Math.max((max - min) * 0.1, 0.03)
    min = Math.max(0, min - pad)
    max += pad
    const innerW = width - PAD.left - PAD.right
    const innerH = HEIGHT - PAD.top - PAD.bottom
    const x = (i: number) => PAD.left + (data.length === 1 ? 0 : (i / (data.length - 1)) * innerW)
    const y = (v: number) => PAD.top + (1 - (v - min) / (max - min || 1)) * innerH
    const minDd = Math.min(-0.01, ...data.map((point) => point.drawdown))
    const yDd = (v: number) => 6 + (v / minDd) * (DD_HEIGHT - 12)
    const line = data.map((point, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(point.equity).toFixed(1)}`).join(" ")
    const area = `${line} L${x(data.length - 1).toFixed(1)} ${HEIGHT - PAD.bottom} L${x(0).toFixed(1)} ${HEIGHT - PAD.bottom} Z`
    const ddArea =
      data.map((point, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${yDd(point.drawdown).toFixed(1)}`).join(" ") +
      ` L${x(data.length - 1).toFixed(1)} 6 L${x(0).toFixed(1)} 6 Z`
    const ticks = Array.from({ length: 5 }, (_, i) => min + ((max - min) * i) / 4)
    const xTicks = [0, Math.floor((data.length - 1) / 3), Math.floor(((data.length - 1) * 2) / 3), data.length - 1]
    return { x, y, yDd, line, area, ddArea, ticks, xTicks, minDd, baseline: y(1) }
  }, [data, width])

  if (!series.length || !geometry) {
    return (
      <div className="flex h-[320px] items-center justify-center">
        <p className="font-mono text-xs uppercase tracking-[0.24em] text-muted-foreground">
          portfolio_returns.csv unavailable for this run
        </p>
      </div>
    )
  }

  const onMove = (clientX: number) => {
    const box = wrapRef.current?.getBoundingClientRect()
    if (!box) return
    const local = clamp(clientX - box.left, PAD.left, width - PAD.right)
    const index = Math.round(((local - PAD.left) / (width - PAD.left - PAD.right)) * (data.length - 1))
    setHover(clamp(index, 0, data.length - 1))
  }

  const point = hover != null ? data[hover] : null
  const last = data[data.length - 1]
  const shown = point ?? last

  return (
    <div>
      <div className="mb-8 flex flex-wrap items-end justify-between gap-6">
        <div aria-live="polite">
          <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted-foreground">
            {point ? dateLabel(point.date) : `Growth of 1.00 · ${dateLabel(data[0].date)} → ${dateLabel(last.date)}`}
          </p>
          <p className="mt-2 flex items-baseline gap-4">
            <span className="font-display text-6xl tabular-nums md:text-7xl">{shown.equity.toFixed(3)}</span>
            <span className={cn("font-mono text-sm tabular-nums", shown.equity >= 1 ? "text-accent" : "text-foreground/70")}>
              {signedPct(shown.equity - 1)}
            </span>
            {point ? (
              <span className="font-mono text-xs text-muted-foreground">
                week {signedPct(point.ret, 2)} · dd {pct(point.drawdown)}
              </span>
            ) : null}
          </p>
        </div>
        <div role="group" aria-label="Chart range" className="flex gap-1">
          {available.map((item) => (
            <button
              key={item.key}
              type="button"
              aria-pressed={range === item.key}
              onClick={() => {
                setRange(item.key)
                setHover(null)
              }}
              className={cn(
                "px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.22em] transition-colors",
                range === item.key ? "text-accent" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div
        ref={wrapRef}
        className="relative touch-pan-y"
        onPointerMove={(event) => onMove(event.clientX)}
        onPointerLeave={() => setHover(null)}
        data-interactive
      >
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Net equity curve from ${dateLabel(data[0].date)} to ${dateLabel(last.date)}, ending at ${last.equity.toFixed(3)}`}
          className="block overflow-visible"
        >
          <defs>
            <linearGradient id="equity-fill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#ff6b1a" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#ff6b1a" stopOpacity="0" />
            </linearGradient>
          </defs>
          {geometry.ticks.map((tick) => (
            <g key={tick}>
              <line
                x1={PAD.left}
                x2={width - PAD.right}
                y1={geometry.y(tick)}
                y2={geometry.y(tick)}
                stroke="currentColor"
                className="text-foreground/[0.06]"
              />
              <text
                x={PAD.left - 10}
                y={geometry.y(tick) + 3}
                textAnchor="end"
                className="fill-muted-foreground font-mono text-[10px]"
              >
                {tick.toFixed(2)}
              </text>
            </g>
          ))}
          <line
            x1={PAD.left}
            x2={width - PAD.right}
            y1={geometry.baseline}
            y2={geometry.baseline}
            stroke="currentColor"
            strokeDasharray="2 4"
            className="text-foreground/25"
          />
          {geometry.xTicks.map((index, i) => (
            <text
              key={`${index}-${i}`}
              x={geometry.x(index)}
              y={HEIGHT - 6}
              textAnchor={i === 0 ? "start" : i === geometry.xTicks.length - 1 ? "end" : "middle"}
              className="fill-muted-foreground font-mono text-[10px]"
            >
              {monthLabel(data[index].date)}
            </text>
          ))}
          <path d={geometry.area} fill="url(#equity-fill)" />
          <path d={geometry.line} fill="none" stroke="#ff6b1a" strokeWidth={1.6} className="chart-draw" />
          {point && hover != null ? (
            <g>
              <line
                x1={geometry.x(hover)}
                x2={geometry.x(hover)}
                y1={PAD.top}
                y2={HEIGHT - PAD.bottom}
                stroke="currentColor"
                className="text-foreground/40"
              />
              <circle cx={geometry.x(hover)} cy={geometry.y(point.equity)} r={4} fill="#0a0a0a" stroke="#ff6b1a" strokeWidth={1.5} />
            </g>
          ) : null}
        </svg>

        <div className="mt-6 flex items-center justify-between">
          <span className="font-mono text-[10px] uppercase tracking-[0.26em] text-muted-foreground">Drawdown</span>
          <span className="font-mono text-[10px] tabular-nums text-muted-foreground">worst {pct(geometry.minDd)}</span>
        </div>
        <svg width={width} height={DD_HEIGHT} viewBox={`0 0 ${width} ${DD_HEIGHT}`} aria-hidden="true" className="block">
          <path d={geometry.ddArea} className="fill-foreground/[0.12]" />
          {hover != null ? (
            <line
              x1={geometry.x(hover)}
              x2={geometry.x(hover)}
              y1={0}
              y2={DD_HEIGHT}
              stroke="currentColor"
              className="text-foreground/40"
            />
          ) : null}
        </svg>
      </div>
    </div>
  )
}
