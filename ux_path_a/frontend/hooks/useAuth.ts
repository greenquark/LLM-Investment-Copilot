'use client'

import { useState, useEffect } from 'react'
import { apiClient } from '@/lib/api'

export interface User {
  id: number
  username: string
  email: string
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  const checkAuth = async () => {
    // Check if user is authenticated
    const token = localStorage.getItem('auth_token')
    if (token) {
      // Verify token by making a lightweight API call
      // If it fails with 401, the token is invalid and will be cleared by API client
      try {
        const response = await apiClient.listSessions()
        if (response.data) {
          // Token is valid, set user
          setUser({ id: 1, username: 'user', email: 'user@example.com' })
        } else if (response.error) {
          // Only clear token if it's an auth error, not a network error
          // Network errors should not clear the token
          if (response.error.includes('401') || response.error.includes('Unauthorized') || response.error.includes('Could not validate credentials')) {
            setUser(null)
            apiClient.setToken(null)
          } else {
            // Network error - keep token but don't set user (will show login)
            console.warn('Network error during auth check:', response.error)
            setUser(null)
          }
        }
      } catch (error) {
        // Network error - don't clear token, just don't set user
        console.warn('Error verifying token (network issue):', error)
        setUser(null)
      }
    } else {
      setUser(null)
    }
    setLoading(false)
  }

  useEffect(() => {
    // Call async checkAuth function
    checkAuth().catch((error) => {
      console.error('Error checking auth:', error)
      setUser(null)
      setLoading(false)
    })
    
    // Listen for unauthorized events from API client
    const handleUnauthorized = () => {
      setUser(null)
      setLoading(false)
    }
    
    window.addEventListener('auth:unauthorized', handleUnauthorized)
    
    return () => {
      window.removeEventListener('auth:unauthorized', handleUnauthorized)
    }
  }, [])

  const login = async (username: string, password: string) => {
    setLoading(true)
    const response = await apiClient.login(username, password)
    if (response.data) {
      // Set user state immediately after successful login
      const userData = { id: 1, username, email: '' }
      setUser(userData)
      setLoading(false)
      return { success: true }
    }
    // Ensure error is a string
    const errorMsg = typeof response.error === 'string' 
      ? response.error 
      : JSON.stringify(response.error)
    setLoading(false)
    return { success: false, error: errorMsg }
  }

  const logout = () => {
    apiClient.setToken(null)
    setUser(null)
  }

  return { user, loading, login, logout, checkAuth }
}
