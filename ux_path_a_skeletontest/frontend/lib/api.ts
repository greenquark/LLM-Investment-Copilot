export async function apiFetch(path: string, init?: RequestInit) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  const res = await fetch(normalizedPath, init) // call "/api/..." and let rewrites proxy it
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  const contentType = res.headers.get('content-type') || ''
  return contentType.includes('application/json') ? res.json() : res.text()
}

