"use client"

import { useEffect, useRef, type CSSProperties, type ElementType, type ReactNode } from "react"
import { motion, useScroll, useTransform, useReducedMotion } from "framer-motion"
import { cn } from "@/lib/utils"

type RevealFrom = "up" | "left" | "right"

// Content is visible in the server HTML. Hiding only starts after mount
// (html.motion-ready), so the page stays readable without JavaScript.
export function Reveal({
  children,
  from = "up",
  delay = 0,
  as: Tag = "div",
  className,
}: {
  children: ReactNode
  from?: RevealFrom
  delay?: number
  as?: ElementType
  className?: string
}) {
  const ref = useRef<HTMLElement>(null)

  useEffect(() => {
    const node = ref.current
    if (!node) return
    document.documentElement.classList.add("motion-ready")
    const observer = new IntersectionObserver(
      ([entry]) => {
        node.dataset.visible = entry.isIntersecting ? "true" : "false"
      },
      { threshold: 0.02, rootMargin: "-8% 0px -8% 0px" },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  return (
    <Tag
      ref={ref}
      data-reveal={from}
      className={className}
      style={{ "--reveal-delay": `${delay}ms` } as CSSProperties}
    >
      {children}
    </Tag>
  )
}

export function Chapter({
  id,
  index,
  label,
  title,
  aside,
  children,
  className,
}: {
  id: string
  index: string
  label: string
  title: ReactNode
  aside?: ReactNode
  children: ReactNode
  className?: string
}) {
  const ref = useRef<HTMLElement>(null)
  const reduce = useReducedMotion()
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end start"] })
  const drift = useTransform(scrollYProgress, [0, 0.5, 1], reduce ? [0, 0, 0] : [36, 0, -36])

  return (
    <section
      ref={ref}
      id={id}
      aria-labelledby={`${id}-title`}
      className={cn("relative scroll-mt-20 px-5 py-24 md:px-12 md:py-32 lg:pl-28 lg:pr-16", className)}
    >
      <header className="mb-14 flex flex-col gap-6 md:mb-20 md:flex-row md:items-end md:justify-between">
        <div>
          <Reveal>
            <p className="font-mono text-[11px] uppercase tracking-[0.32em] text-accent">
              {index} / {label}
            </p>
          </Reveal>
          <motion.div style={{ y: drift }}>
            <Reveal delay={80}>
              <h2
                id={`${id}-title`}
                className="mt-5 text-balance font-display text-6xl leading-[0.88] tracking-tight md:text-8xl lg:text-[8.5rem]"
              >
                {title}
              </h2>
            </Reveal>
          </motion.div>
        </div>
        {aside ? (
          <Reveal from="right" delay={160} className="max-w-sm md:text-right">
            {aside}
          </Reveal>
        ) : null}
      </header>
      {children}
    </section>
  )
}

export function Label({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cn("font-mono text-[10px] uppercase tracking-[0.28em] text-muted-foreground", className)}>
      {children}
    </span>
  )
}

export function Unavailable({ children }: { children: ReactNode }) {
  return <p className="font-mono text-xs leading-relaxed text-muted-foreground">{children}</p>
}
