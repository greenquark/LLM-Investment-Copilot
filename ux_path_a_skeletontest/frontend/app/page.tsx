'use client'

import { useEffect, useState } from 'react'
import { apiFetch } from '../lib/api'

type Health = { status: string; service: string; timestamp: string }

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true
    apiFetch('/api/health', { cache: 'no-store' })
      .then((data) => {
        if (!mounted) return
        setHealth(data as Health)
      })
      .catch((e) => {
        if (!mounted) return
        setError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      mounted = false
    }
  }, [])

  return (
    <main style={{ padding: 24, maxWidth: 860, margin: '0 auto' }}>
      <h1 style={{ marginTop: 0 }}>UX Path A Skeleton Test</h1>
      <p>
        This frontend calls <code>/api/health</code> and relies on a rewrite to proxy to the backend.
      </p>

      <div style={{ padding: 16, border: '1px solid #ddd', borderRadius: 8 }}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>Backend health</div>
        {error && <pre style={{ color: '#b91c1c', whiteSpace: 'pre-wrap' }}>{error}</pre>}
        {!error && !health && <div>Loading...</div>}
        {health && (
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(health, null, 2)}</pre>
        )}
      </div>

      <div style={{ marginTop: 16, fontSize: 12, opacity: 0.8 }}>
        Tip: set <code>NEXT_PUBLIC_BACKEND_URL</code> in Vercel to your Railway backend base URL.
      </div>
    </main>
  )
}

