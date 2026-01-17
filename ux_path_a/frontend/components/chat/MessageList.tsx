'use client'

import MessageBubble from './MessageBubble'

interface Message {
  role: string
  content: string
  timestamp: string
  thinking_content?: string
  tool_calls?: any
  tool_results?: any
}

interface MessageListProps {
  messages: Message[]
  loading: boolean
  showMessageMeta?: boolean
}

export default function MessageList({ messages, loading, showMessageMeta = false }: MessageListProps) {
  return (
    <div className="flex flex-col gap-4 p-6">
      {messages.length === 0 && (
        <div className="text-center text-gray-500 dark:text-gray-400 mt-12">
          <p className="text-lg mb-2">Welcome to Smart Trading Copilot</p>
          <p className="text-sm">
            Ask me about market analysis, trends, or technical indicators.
          </p>
          <p className="text-xs mt-4 text-gray-400">
            This tool is for educational purposes only. Not financial advice.
          </p>
        </div>
      )}

      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} showMeta={showMessageMeta} />
      ))}

      {loading && (
        <div className="flex items-center gap-2 text-gray-500">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-500"></div>
          <span className="text-sm">Thinking...</span>
        </div>
      )}
    </div>
  )
}
