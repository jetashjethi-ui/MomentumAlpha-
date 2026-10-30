"use client"

import { useEffect, useState } from "react"

export function useActiveSection(ids: string[]) {
  const [active, setActive] = useState(ids[0])

  useEffect(() => {
    let frame = 0
    const update = () => {
      const marker = window.scrollY + Math.min(window.innerHeight * 0.4, 320)
      let current = ids[0]
      for (const id of ids) {
        const node = document.getElementById(id)
        if (node && node.offsetTop <= marker) current = id
      }
      setActive(current)
      frame = 0
    }
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(update)
    }
    update()
    window.addEventListener("scroll", onScroll, { passive: true })
    window.addEventListener("resize", onScroll)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener("scroll", onScroll)
      window.removeEventListener("resize", onScroll)
    }
  }, [ids])

  return active
}
