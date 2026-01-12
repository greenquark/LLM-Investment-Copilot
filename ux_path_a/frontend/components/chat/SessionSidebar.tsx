'use client'

interface Session {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

interface SessionSidebarProps {
  sessions: Session[]
  currentSessionId: string | null
  onSelectSession: (sessionId: string) => void
  onNewSession: () => void
}

export default function SessionSidebar({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
}: SessionSidebarProps) {
  return (
    <div className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={onNewSession}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          + New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {sessions.length === 0 ? (
          <div className="p-4 text-center text-gray-500 dark:text-gray-400 text-sm">
            No sessions yet. Create one to get started.
          </div>
        ) : (
          <div className="p-2">
            {sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => onSelectSession(session.id)}
                className={`w-full text-left px-3 py-2 rounded-lg mb-1 transition-colors ${
                  currentSessionId === session.id
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                <div className="font-medium text-sm truncate">{session.title}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {session.message_count} messages
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
