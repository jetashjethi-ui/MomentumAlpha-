"use client"

import { useEffect, useRef } from "react"

const INTERACTIVE = "a, button, select, summary, input, label, textarea, [data-interactive]"

export function PointerOrb() {
  const dotRef = useRef<HTMLDivElement>(null)
  const ringRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const fine = window.matchMedia("(hover: hover) and (pointer: fine)").matches
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches
    const dot = dotRef.current
    const ring = ringRef.current
    if (!fine || !dot || !ring) return

    let tx = -100
    let ty = -100
    let rx = tx
    let ry = ty
    let frame = 0

    const animate = () => {
      rx += (tx - rx) * (reduce ? 1 : 0.16)
      ry += (ty - ry) * (reduce ? 1 : 0.16)
      dot.style.transform = `translate3d(${tx}px, ${ty}px, 0) translate(-50%, -50%)`
      ring.style.transform = `translate3d(${rx}px, ${ry}px, 0) translate(-50%, -50%)`
      frame = Math.abs(tx - rx) > 0.1 || Math.abs(ty - ry) > 0.1 ? requestAnimationFrame(animate) : 0
    }

    const onMove = (event: PointerEvent) => {
      if (event.pointerType !== "mouse") return
      tx = event.clientX
      ty = event.clientY
      const hovering = event.target instanceof Element && Boolean(event.target.closest(INTERACTIVE))
      dot.dataset.state = hovering ? "hover" : "idle"
      ring.dataset.state = hovering ? "hover" : "idle"
      dot.dataset.visible = "true"
      ring.dataset.visible = "true"
      if (!frame) frame = requestAnimationFrame(animate)
    }
    const hide = () => {
      dot.dataset.visible = "false"
      ring.dataset.visible = "false"
    }

    window.addEventListener("pointermove", onMove, { passive: true })
    document.documentElement.addEventListener("pointerleave", hide)
    window.addEventListener("blur", hide)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener("pointermove", onMove)
      document.documentElement.removeEventListener("pointerleave", hide)
      window.removeEventListener("blur", hide)
    }
  }, [])

  return (
    <>
      <div ref={ringRef} aria-hidden="true" className="pointer-ring" data-visible="false" />
      <div ref={dotRef} aria-hidden="true" className="pointer-dot" data-visible="false" />
    </>
  )
}
