/**
 * API client for Path A backend.
 */

// Simple fetch helper for calling the backend via Vercel rewrites.
// Usage: apiFetch("/api/health") or apiFetch("/api/chat/messages", { method: "POST", body: ... })
export async function apiFetch(path: string, init?: RequestInit) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  const res = await fetch(normalizedPath, init) // call "/api/..." and let Vercel rewrite it
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  const contentType = res.headers.get('content-type') || ''
  return contentType.includes('application/json') ? res.json() : res.text()
}

// In production (Vercel), prefer relative URLs so `/api/*` can be rewritten/proxied.
// In local dev, default to the local backend unless explicitly overridden.
const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000')

export interface ApiResponse<T> {
  data?: T
  error?: string
}

class ApiClient {
  private baseUrl: string
  private token: string | null = null

  constructor(baseUrl: string = API_URL) {
    // Normalize the base URL - remove trailing slash
    this.baseUrl = baseUrl.replace(/\/$/, '')
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('auth_token')
    }
  }
  
  /**
   * Test if backend is accessible
   */
  async testConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      return response.ok
    } catch {
      return false
    }
  }

  setToken(token: string | null) {
    this.token = token
    if (token && typeof window !== 'undefined') {
      localStorage.setItem('auth_token', token)
    } else if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token')
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    // Ensure endpoint starts with /
    const normalizedEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`
    const url = `${this.baseUrl}${normalizedEndpoint}`
    const headers = new Headers(options.headers)

    // Only set default Content-Type if not already specified
    if (!headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json')
    }

    if (this.token) {
      headers.set('Authorization', `Bearer ${this.token}`)
    }

    // DEBUG: Check if debug logging is enabled
    const isDebug = typeof window !== 'undefined' && (
      process.env.NODE_ENV === 'development' || 
      (window as any).__DEBUG_API__ === true
    )

    if (isDebug) {
      console.group(`🔵 API Request: ${options.method || 'GET'} ${normalizedEndpoint}`)
      console.log('URL:', url)
      const headersObj = Object.fromEntries(headers.entries())
      if (headersObj.Authorization) headersObj.Authorization = 'Bearer ***'
      console.log('Headers:', headersObj)
      if (options.body) {
        try {
          const bodyObj = typeof options.body === 'string' ? JSON.parse(options.body) : options.body
          console.log('Body:', bodyObj)
        } catch {
          console.log('Body:', options.body)
        }
      }
    }

    const startTime = Date.now()

    try {
      // Ensure URL is properly formatted
      const fullUrl = url.startsWith('http') ? url : `${this.baseUrl}${url.startsWith('/') ? url : `/${url}`}`
      
      const response = await fetch(fullUrl, {
        ...options,
        headers,
        // Add credentials for CORS
        credentials: 'include',
      })

      const duration = Date.now() - startTime

      if (isDebug) {
        console.log(`✅ Response (${duration}ms):`, {
          status: response.status,
          statusText: response.statusText,
          ok: response.ok,
        })
      }

      if (!response.ok) {
        // Handle 401 Unauthorized - clear invalid token
        if (response.status === 401) {
          this.setToken(null)
          // Trigger auth check by dispatching a custom event
          if (typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent('auth:unauthorized'))
          }
        }
        
        try {
          const errorData = await response.json()
          if (isDebug) {
            console.error('❌ Error Response:', errorData)
          }
          // Handle FastAPI validation errors
          if (Array.isArray(errorData.detail)) {
            const errorMessages = errorData.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
            const result = { error: errorMessages || errorData.detail || response.statusText }
            if (isDebug) console.groupEnd()
            return result
          }
          const result = { error: errorData.error || errorData.detail || errorData.message || response.statusText }
          if (isDebug) console.groupEnd()
          return result
        } catch {
          const result = { error: response.statusText || 'Request failed' }
          if (isDebug) {
            console.error('❌ Error Response (non-JSON):', response.statusText)
            console.groupEnd()
          }
          return result
        }
      }

      const data = await response.json()
      if (isDebug) {
        console.log('📦 Response Data:', data)
        console.groupEnd()
      }
      return { data }
    } catch (error) {
      const duration = Date.now() - startTime
      // Handle network errors (CORS, connection refused, etc.)
      const errorMessage = error instanceof Error ? error.message : 'Network error'
      
      if (isDebug) {
        console.error('💥 Request Failed:', {
          error: errorMessage,
          duration: `${duration}ms`,
          url,
        })
        console.groupEnd()
      }
      
      // Provide more helpful error messages
      if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError') || errorMessage.includes('Network request failed')) {
        const backendUrl = this.baseUrl || 'http://localhost:8000'
        return { 
          error: `Cannot connect to backend at ${backendUrl}. Please ensure:\n1. Backend server is running (cd ux_path_a/backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000)\n2. Backend is accessible at ${backendUrl}\n3. No firewall is blocking the connection\n4. CORS is properly configured` 
        }
      }
      
      return { error: errorMessage }
    }
  }

  // Auth endpoints
  async login(username: string, password: string) {
    // OAuth2PasswordRequestForm expects application/x-www-form-urlencoded
    const params = new URLSearchParams()
    params.append('username', username)
    params.append('password', password)
    
    try {
      const response = await this.request<{ access_token: string; token_type: string }>(
        '/api/auth/token',
        {
          method: 'POST',
          body: params.toString(),
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
        }
      )

      if (response.data) {
        this.setToken(response.data.access_token)
      }

      return response
    } catch (error) {
      return { 
        error: error instanceof Error ? error.message : 'Login failed' 
      }
    }
  }

  async register(email: string, username: string, password: string) {
    return this.request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, username, password }),
    })
  }

  // Chat endpoints
  async createSession(title?: string) {
    return this.request<{ id: string; title: string; created_at: string; updated_at: string; message_count: number }>(
      '/api/chat/sessions',
      {
        method: 'POST',
        body: JSON.stringify({ title }),
      }
    )
  }

  async listSessions() {
    return this.request<Array<{ id: string; title: string; created_at: string; updated_at: string; message_count: number }>>(
      '/api/chat/sessions'
    )
  }

  async sendMessage(content: string, sessionId?: string) {
    return this.request<{
      message: {
        role: string
        content: string
        timestamp: string
        tool_calls?: any
        tool_results?: any
      }
      session_id: string
      token_usage?: {
        prompt_tokens: number
        completion_tokens: number
        total_tokens: number
      }
    }>('/api/chat/messages', {
      method: 'POST',
      body: JSON.stringify({ content, session_id: sessionId }),
    })
  }

  async getMessages(sessionId: string) {
    return this.request<Array<{
      role: string
      content: string
      timestamp: string
      tool_calls?: any
      tool_results?: any
    }>>(`/api/chat/sessions/${sessionId}/messages`)
  }

  // Config endpoints
  async getConfig() {
    return this.request<{
      llm_model: string
    }>('/api/health/config')
  }
}

export const apiClient = new ApiClient()

// Expose debug utilities in development
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  (window as any).__API_DEBUG__ = {
    testConnection: () => apiClient.testConnection(),
    getBaseUrl: () => (apiClient as any).baseUrl,
    setBaseUrl: (url: string) => { (apiClient as any).baseUrl = url.replace(/\/$/, '') },
    enableLogging: () => { (window as any).__DEBUG_API__ = true; console.log('✅ API logging enabled') },
    disableLogging: () => { (window as any).__DEBUG_API__ = false; console.log('❌ API logging disabled') },
    getToken: () => (apiClient as any).token ? 'Bearer ***' : null,
  }
  
  console.log('%c🔧 API Debug utilities available:', 'color: #4CAF50; font-weight: bold')
  console.log('   window.__API_DEBUG__.testConnection() - Test backend connection')
  console.log('   window.__API_DEBUG__.enableLogging() - Enable request/response logging')
  console.log('   window.__API_DEBUG__.getBaseUrl() - Get current API base URL')
}
