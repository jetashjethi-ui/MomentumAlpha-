"use client"

import { useEffect, useRef } from "react"

const COLS = 84
const ROWS = 44

export function ParticleWave() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches
    const coarse = window.matchMedia("(pointer: coarse)").matches
    const cols = coarse ? Math.round(COLS * 0.55) : COLS
    const rows = coarse ? Math.round(ROWS * 0.6) : ROWS

    let width = 0
    let height = 0
    let frame = 0
    let running = true
    let scrollShift = 0
    const pointer = { x: -9999, y: -9999, active: false }

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5)
      width = window.innerWidth
      height = window.innerHeight
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      canvas.style.width = width + "px"
      canvas.style.height = height + "px"
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    const draw = (time: number) => {
      const t = reduce ? 0 : time * 0.00022
      ctx.clearRect(0, 0, width, height)
      const horizon = height * 0.42
      const focal = height * 0.9
      const cameraHeight = 150
      const spanX = width * 2.4

      for (let r = 0; r < rows; r += 1) {
        const depth = 1 - r / rows
        const z = 60 + r * 26
        const scale = focal / z
        for (let c = 0; c < cols; c += 1) {
          const u = c / (cols - 1) - 0.5
          const wx = u * spanX
          const wave =
            Math.sin(u * 7 + t * 3 + scrollShift * 0.9) * 18 +
            Math.cos(r * 0.32 - t * 2.2 + scrollShift) * 14 +
            Math.sin((u * 3 + r * 0.08) * 2 + t) * 8
          let sx = width / 2 + wx * scale * 0.35
          let sy = horizon + (cameraHeight - wave) * scale * 0.55

          if (pointer.active) {
            const dx = sx - pointer.x
            const dy = sy - pointer.y
            const dist2 = dx * dx + dy * dy
            if (dist2 < 22000) {
              const lift = (1 - dist2 / 22000) * 14
              sy -= lift
              sx += (dx / 140) * lift * 0.3
            }
          }

          if (sx < -10 || sx > width + 10 || sy < -10 || sy > height + 10) continue
          const alpha = Math.min(0.55, 0.06 + depth * 0.5) * (0.65 + Math.sin(u * 9 + t * 2) * 0.12)
          const size = Math.max(0.6, scale * 0.9)
          const warm = wave > 26
          ctx.fillStyle = warm ? `rgba(255,107,26,${alpha * 0.9})` : `rgba(236,232,224,${alpha * 0.5})`
          ctx.fillRect(sx, sy, size, size)
        }
      }
    }

    const loop = (time: number) => {
      if (!running) return
      draw(time)
      frame = requestAnimationFrame(loop)
    }

    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight
      scrollShift = max > 0 ? (window.scrollY / max) * 2.4 : 0
      if (reduce) draw(0)
    }
    const onPointer = (event: PointerEvent) => {
      if (event.pointerType !== "mouse") return
      pointer.x = event.clientX
      pointer.y = event.clientY
      pointer.active = true
    }
    const onLeave = () => {
      pointer.active = false
    }
    const onVisibility = () => {
      if (document.hidden) {
        running = false
        cancelAnimationFrame(frame)
      } else if (!reduce && !running) {
        running = true
        frame = requestAnimationFrame(loop)
      }
    }

    resize()
    onScroll()
    window.addEventListener("resize", resize)
    window.addEventListener("scroll", onScroll, { passive: true })
    document.addEventListener("visibilitychange", onVisibility)
    if (!reduce) {
      window.addEventListener("pointermove", onPointer, { passive: true })
      document.documentElement.addEventListener("pointerleave", onLeave)
      frame = requestAnimationFrame(loop)
    } else {
      draw(0)
    }

    return () => {
      running = false
      cancelAnimationFrame(frame)
      window.removeEventListener("resize", resize)
      window.removeEventListener("scroll", onScroll)
      window.removeEventListener("pointermove", onPointer)
      document.removeEventListener("visibilitychange", onVisibility)
      document.documentElement.removeEventListener("pointerleave", onLeave)
    }
  }, [])

  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 z-0">
      <canvas ref={canvasRef} className="h-full w-full opacity-70" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_20%,transparent_0%,var(--background)_78%)]" />
    </div>
  )
}
