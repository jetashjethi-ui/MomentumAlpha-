// When MOMENTUM_API_URL is set (e.g. http://127.0.0.1:8000), API paths are
// proxied so the UI and FastAPI share an origin during `next dev`.
const apiUrl = process.env.MOMENTUM_API_URL?.replace(/\/$/, "")

/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    unoptimized: true,
  },
  async rewrites() {
    if (!apiUrl) return []
    return ["/health", "/ready", "/openapi.json", "/runs", "/runs/:path*"].map((source) => ({
      source,
      destination: apiUrl + source,
    }))
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "SAMEORIGIN" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ]
  },
}

export default nextConfig
