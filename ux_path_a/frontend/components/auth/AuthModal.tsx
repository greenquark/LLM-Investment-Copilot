'use client'

import { useState } from 'react'
import { useAuth } from '@/hooks/useAuth'

interface AuthModalProps {
  isOpen: boolean
  onClose: () => void
  onLoginSuccess?: () => void
  login?: (username: string, password: string) => Promise<{ success: boolean; error?: string }>
}

export default function AuthModal({ isOpen, onClose, onLoginSuccess, login: loginProp }: AuthModalProps) {
  const [isLogin, setIsLogin] = useState(true)
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const authHook = useAuth()
  const login = loginProp || authHook.login

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    // Client-side validation
    if (isLogin) {
      if (!username.trim()) {
        setError('Username is required')
        return
      }
      if (!password.trim()) {
        setError('Password is required')
        return
      }
    } else {
      if (!email.trim()) {
        setError('Email is required')
        return
      }
      if (!username.trim()) {
        setError('Username is required')
        return
      }
      if (!password.trim()) {
        setError('Password is required')
        return
      }
    }

    setLoading(true)

    try {
      if (isLogin) {
        const result = await login(username.trim(), password)
        if (!result.success) {
          // Ensure error is a string
          const errorMsg = typeof result.error === 'string' 
            ? result.error 
            : JSON.stringify(result.error)
          setError(errorMsg || 'Login failed')
        } else {
          onClose()
          if (onLoginSuccess) {
            onLoginSuccess()
          }
        }
      } else {
        // TODO: Implement registration
        setError('Registration not yet implemented')
      }
    } catch (err) {
      const errorMsg = err instanceof Error 
        ? err.message 
        : (typeof err === 'string' ? err : JSON.stringify(err))
      setError(errorMsg || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-8 w-full max-w-md">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">
          {isLogin ? 'Login' : 'Register'}
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4" noValidate>
          {!isLogin && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                placeholder="Enter email"
                autoComplete="email"
              />
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Username
            </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                placeholder="Enter username"
                autoComplete="username"
              />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              placeholder="Enter password"
              autoComplete={isLogin ? "current-password" : "new-password"}
            />
          </div>

          {error && (
            <div className="text-red-600 dark:text-red-400 text-sm">{error}</div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? 'Loading...' : isLogin ? 'Login' : 'Register'}
          </button>
        </form>

        <div className="mt-4 text-center">
          <button
            onClick={() => {
              setIsLogin(!isLogin)
              setError('')
            }}
            className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            {isLogin ? "Don't have an account? Register" : 'Already have an account? Login'}
          </button>
        </div>

        <div className="mt-6 text-xs text-gray-500 dark:text-gray-400 text-center">
          <p>For MVP, any credentials will work.</p>
          <p className="mt-2">This tool is for educational purposes only.</p>
        </div>
      </div>
    </div>
  )
}
