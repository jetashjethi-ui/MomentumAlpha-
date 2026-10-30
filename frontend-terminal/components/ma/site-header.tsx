"use client"

import { useState } from "react"
import { motion, useScroll, useSpring } from "framer-motion"
import { Menu, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { useActiveSection } from "@/hooks/use-active-section"

export const CHAPTERS = [
  { id: "overview", label: "Overview" },
  { id: "performance", label: "Performance" },
  { id: "evidence", label: "Evidence" },
  { id: "data", label: "Data" },
  { id: "experiments", label: "Experiments" },
]

const IDS = CHAPTERS.map((chapter) => chapter.id)

export function SiteHeader({ online, loading }: { online: boolean; loading: boolean }) {
  const active = useActiveSection(IDS)
  const [open, setOpen] = useState(false)
  const { scrollYProgress } = useScroll()
  const progress = useSpring(scrollYProgress, { stiffness: 140, damping: 30, mass: 0.3 })

  const status = loading ? "Connecting" : online ? "API online" : "API offline"

  return (
    <>
      <header className="fixed inset-x-0 top-0 z-50 border-b border-foreground/[0.06] bg-background/70 backdrop-blur-md">
        <div className="flex h-14 items-center justify-between gap-6 px-5 md:px-12 lg:pl-28 lg:pr-16">
          <a href="#overview" className="flex items-baseline gap-1 font-mono text-xs uppercase tracking-[0.3em]">
            <span className="text-foreground">Momentum</span>
            <span className="text-accent">/</span>
            <span className="text-muted-foreground">Alpha</span>
          </a>

          <nav aria-label="Chapters" className="hidden items-center gap-7 md:flex">
            {CHAPTERS.map((chapter) => (
              <a
                key={chapter.id}
                href={`#${chapter.id}`}
                aria-current={active === chapter.id ? "true" : undefined}
                className={cn(
                  "font-mono text-[10px] uppercase tracking-[0.26em] transition-colors hover:text-foreground",
                  active === chapter.id ? "text-accent" : "text-muted-foreground",
                )}
              >
                {chapter.label}
              </a>
            ))}
          </nav>

          <div className="flex items-center gap-4">
            <span className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              <span
                aria-hidden="true"
                className={cn(
                  "h-1.5 w-1.5 rounded-full",
                  loading ? "animate-pulse bg-muted-foreground" : online ? "bg-accent" : "bg-foreground/30",
                )}
              />
              <span className="hidden sm:inline">{status}</span>
              <span className="sr-only sm:hidden">{status}</span>
            </span>
            <button
              type="button"
              onClick={() => setOpen((value) => !value)}
              aria-expanded={open}
              aria-controls="mobile-nav"
              className="p-1 text-foreground md:hidden"
            >
              {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
              <span className="sr-only">{open ? "Close menu" : "Open menu"}</span>
            </button>
          </div>
        </div>
        <motion.div
          aria-hidden="true"
          style={{ scaleX: progress }}
          className="absolute inset-x-0 bottom-0 h-px origin-left bg-accent"
        />
        {open ? (
          <nav id="mobile-nav" aria-label="Chapters" className="border-t border-foreground/[0.06] px-5 pb-6 pt-2 md:hidden">
            {CHAPTERS.map((chapter, index) => (
              <a
                key={chapter.id}
                href={`#${chapter.id}`}
                onClick={() => setOpen(false)}
                className={cn(
                  "flex items-baseline gap-4 py-3 font-display text-3xl",
                  active === chapter.id ? "text-accent" : "text-foreground",
                )}
              >
                <span className="font-mono text-[10px] tracking-[0.2em] text-muted-foreground">0{index + 1}</span>
                {chapter.label}
              </a>
            ))}
          </nav>
        ) : null}
      </header>

      <nav aria-label="Chapter rail" className="fixed left-0 top-1/2 z-40 hidden -translate-y-1/2 pl-8 lg:block">
        <ol className="flex flex-col gap-5">
          {CHAPTERS.map((chapter) => {
            const isActive = active === chapter.id
            return (
              <li key={chapter.id}>
                <a href={`#${chapter.id}`} className="group flex items-center gap-3" aria-label={chapter.label}>
                  <span
                    aria-hidden="true"
                    className={cn(
                      "block h-1.5 w-1.5 rounded-full transition-all duration-300",
                      isActive ? "scale-125 bg-accent" : "bg-muted-foreground/40 group-hover:bg-foreground/70",
                    )}
                  />
                  <span
                    className={cn(
                      "font-mono text-[9px] uppercase tracking-[0.24em] opacity-0 transition-opacity duration-300 group-hover:opacity-100 group-focus-visible:opacity-100",
                      isActive ? "text-accent" : "text-muted-foreground",
                    )}
                  >
                    {chapter.label}
                  </span>
                </a>
              </li>
            )
          })}
        </ol>
      </nav>
    </>
  )
}
