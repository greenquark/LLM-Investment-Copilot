'use client'

import { useState, useEffect, useRef } from 'react'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import SessionSidebar from './SessionSidebar'
import { apiClient } from '@/lib/api'

interface ChatInterfaceProps {
  onLogout?: () => void
}

export default function ChatInterface({ onLogout }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Array<{
    role: string
    content: string
    timestamp: string
    tool_calls?: any
    tool_results?: any
  }>>([])
  const [sessions, setSessions] = useState<Array<{
    id: string
    title: string
    created_at: string
    updated_at: string
    message_count: number
  }>>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [llmModel, setLlmModel] = useState<string | null>(null)
  const [showMessageMeta, setShowMessageMeta] = useState(false)

  const scrollContainerRef = useRef<HTMLDivElement | null>(null)
  const isNearBottomRef = useRef(true)
  const pendingScrollModeRef = useRef<'none' | 'instant' | 'smooth'>('instant')

  useEffect(() => {
    loadSessions()
    loadConfig()
  }, [])

  const loadConfig = async () => {
    try {
      const response = await apiClient.getConfig()
      if (response.data) {
        setLlmModel(response.data.llm_model)
      } else if (response.error) {
        console.warn('Failed to load LLM model config:', response.error)
        // Set a default or fallback value
        setLlmModel('Unknown')
        // Don't show alert for network errors during initial load
        if (!response.error.includes('Cannot connect to backend')) {
          console.warn('Config load error:', response.error)
        }
      }
    } catch (error) {
      console.error('Error loading config:', error)
      // Set a default or fallback value
      setLlmModel('Unknown')
    }
  }

  useEffect(() => {
    if (currentSessionId) {
      // When switching sessions (especially legacy/history), jump to bottom after load.
      pendingScrollModeRef.current = 'instant'
      loadMessages(currentSessionId)
    } else {
      setMessages([])
    }
  }, [currentSessionId])

  useEffect(() => {
    const mode = pendingScrollModeRef.current
    if (mode === 'none') return
    const el = scrollContainerRef.current
    if (!el) return
    if (messages.length === 0) return

    const scrollToBottom = () => {
      const top = el.scrollHeight
      if (mode === 'smooth') {
        el.scrollTo({ top, behavior: 'smooth' })
      } else {
        el.scrollTop = top
      }
      pendingScrollModeRef.current = 'none'
    }

    // Wait for DOM to paint the new messages before scrolling.
    requestAnimationFrame(scrollToBottom)
  }, [messages.length])

  const loadSessions = async () => {
    try {
      const response = await apiClient.listSessions()
      if (response.data) {
        setSessions(response.data)
        if (response.data.length > 0 && !currentSessionId) {
          setShowMessageMeta(false) // legacy chat load
          setCurrentSessionId(response.data[0].id)
        }
      } else if (response.error) {
        console.error('Failed to load sessions:', response.error)
        // Don't show alert for auth errors (401) - handled by useAuth
        if (!response.error.includes('401') && !response.error.includes('Unauthorized')) {
          console.warn('Error loading sessions:', response.error)
        }
      }
    } catch (error) {
      console.error('Error loading sessions:', error)
    }
  }

  const loadMessages = async (sessionId: string) => {
    try {
      // On legacy chat load, ensure we end up at the bottom.
      pendingScrollModeRef.current = 'instant'
      const response = await apiClient.getMessages(sessionId)
      if (response.data) {
        setMessages(response.data)
      } else if (response.error) {
        console.error('Failed to load messages:', response.error)
        // Don't show alert for auth errors (401) - handled by useAuth
        if (!response.error.includes('401') && !response.error.includes('Unauthorized')) {
          console.warn('Error loading messages:', response.error)
        }
      }
    } catch (error) {
      console.error('Error loading messages:', error)
    }
  }

  const handleNewSession = async () => {
    try {
      setLoading(true)
      const response = await apiClient.createSession()
      if (response.data) {
        setShowMessageMeta(true) // show meta/debug only for newly created chats
        await loadSessions()
        setCurrentSessionId(response.data.id)
        setMessages([]) // Clear messages for new session
      } else if (response.error) {
        // Show error message
        const errorMsg = typeof response.error === 'string' 
          ? response.error 
          : JSON.stringify(response.error)
        console.error('Failed to create session:', errorMsg)
        // Optionally show error to user
        alert(`Failed to create new chat: ${errorMsg}`)
      }
    } catch (error) {
      const errorMsg = error instanceof Error 
        ? error.message 
        : (typeof error === 'string' ? error : JSON.stringify(error))
      console.error('Error creating session:', errorMsg)
      alert(`Error creating new chat: ${errorMsg}`)
    } finally {
      setLoading(false)
    }
  }

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return

    // Add user message immediately
    const userMessage = {
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    }
    if (isNearBottomRef.current) pendingScrollModeRef.current = 'smooth'
    setMessages(prev => [...prev, userMessage])
    setLoading(true)

    try {
      const response = await apiClient.sendMessage(content, currentSessionId || undefined)
      if (response.data) {
        // Update session ID if new session was created
        if (response.data.session_id !== currentSessionId) {
          setShowMessageMeta(true) // new session created implicitly by backend
          setCurrentSessionId(response.data.session_id)
          await loadSessions()
        }

        // Add assistant message
        if (isNearBottomRef.current) pendingScrollModeRef.current = 'smooth'
        setMessages(prev => [...prev, response.data!.message])
      } else if (response.error) {
        // Add error message - ensure it's a string
        const errorMsg = typeof response.error === 'string' 
          ? response.error 
          : JSON.stringify(response.error)
        if (isNearBottomRef.current) pendingScrollModeRef.current = 'smooth'
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Error: ${errorMsg}`,
          timestamp: new Date().toISOString(),
        }])
      }
    } catch (error) {
      const errorMsg = error instanceof Error 
        ? error.message 
        : (typeof error === 'string' ? error : JSON.stringify(error))
      if (isNearBottomRef.current) pendingScrollModeRef.current = 'smooth'
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${errorMsg}`,
        timestamp: new Date().toISOString(),
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <SessionSidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={(id) => {
          setShowMessageMeta(false) // legacy session view
          setCurrentSessionId(id)
        }}
        onNewSession={handleNewSession}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                Smart Trading Copilot
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Educational market analysis assistant
              </p>
            </div>
            <div className="flex items-center gap-4">
              {llmModel && (
                <div className="text-right">
                  <p className="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">
                    Model
                  </p>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {llmModel}
                  </p>
                </div>
              )}
              {onLogout && (
                <button
                  onClick={onLogout}
                  className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors"
                  title="Logout"
                >
                  Logout
                </button>
              )}
            </div>
          </div>
        </header>

        {/* Messages */}
        <div
          ref={scrollContainerRef}
          className="flex-1 overflow-y-auto"
          onScroll={() => {
            const el = scrollContainerRef.current
            if (!el) return
            const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
            isNearBottomRef.current = distanceFromBottom < 120
          }}
        >
          <MessageList messages={messages} loading={loading} showMessageMeta={showMessageMeta} />
        </div>

        {/* Input */}
        <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <ChatInput onSend={handleSendMessage} disabled={loading} />
        </div>
      </div>
    </div>
  )
}
