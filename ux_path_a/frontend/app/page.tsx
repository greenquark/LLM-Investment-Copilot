'use client'

import { useState, useEffect } from 'react'
import ChatInterface from '@/components/chat/ChatInterface'
import AuthModal from '@/components/auth/AuthModal'
import { useAuth } from '@/hooks/useAuth'

export default function Home() {
  const { user, loading, login, logout } = useAuth()
  const [showAuth, setShowAuth] = useState(false)

  useEffect(() => {
    // Show auth modal if not loading and no user
    if (!loading && !user) {
      setShowAuth(true)
    } else if (user) {
      setShowAuth(false)
    }
  }, [user, loading])
  
  // Also listen for unauthorized events to show login
  useEffect(() => {
    const handleUnauthorized = () => {
      setShowAuth(true)
    }
    
    window.addEventListener('auth:unauthorized', handleUnauthorized)
    
    return () => {
      window.removeEventListener('auth:unauthorized', handleUnauthorized)
    }
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="text-lg text-gray-700 dark:text-gray-300">Loading...</div>
      </div>
    )
  }

  if (!user) {
    return (
      <AuthModal 
        isOpen={showAuth} 
        onClose={() => {}}
        login={login}
      />
    )
  }

  return (
    <main className="flex flex-col h-screen">
      <ChatInterface onLogout={logout} />
    </main>
  )
}
