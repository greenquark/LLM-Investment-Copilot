'use client'

import { useState } from 'react'
import { apiClient } from '@/lib/api'

export default function ConnectionTest() {
  const [results, setResults] = useState<Array<{
    name: string
    success: boolean
    duration?: number
    result?: any
    error?: string
  }>>([])
  const [testing, setTesting] = useState(false)

  const runTests = async () => {
    setTesting(true)
    setResults([])
    
    const tests = [
      {
        name: 'Health Check',
        test: async () => {
          const response = await (apiClient as any).request('/api/health')
          return response
        }
      },
      {
        name: 'Config Check',
        test: async () => {
          const response = await apiClient.getConfig()
          return response
        }
      },
      {
        name: 'Connection Test',
        test: async () => {
          const connected = await apiClient.testConnection()
          return { data: connected, error: connected ? null : 'Connection failed' }
        }
      },
      {
        name: 'Debug Info',
        test: async () => {
          const response = await (apiClient as any).request('/api/health/debug')
          return response
        }
      },
    ]
    
    for (const test of tests) {
      try {
        const start = Date.now()
        const result = await test.test()
        const duration = Date.now() - start
        
        setResults(prev => [...prev, {
          name: test.name,
          success: !result.error,
          duration,
          result: result.data || result,
        }])
      } catch (error) {
        setResults(prev => [...prev, {
          name: test.name,
          success: false,
          error: error instanceof Error ? error.message : String(error),
        }])
      }
    }
    
    setTesting(false)
  }

  return (
    <div className="p-4 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800">
      <h2 className="text-lg font-bold mb-2 text-gray-900 dark:text-white">Connection Diagnostics</h2>
      <button
        onClick={runTests}
        disabled={testing}
        className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 hover:bg-blue-600 transition-colors"
      >
        {testing ? 'Testing...' : 'Run Tests'}
      </button>
      
      {results.length > 0 && (
        <div className="mt-4 space-y-2">
          {results.map((result, i) => (
            <div key={i} className={`p-3 rounded ${result.success ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800' : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'}`}>
              <div className="font-semibold text-gray-900 dark:text-white">
                {result.success ? '✅' : '❌'} {result.name}
                {result.duration && (
                  <span className="text-sm font-normal text-gray-600 dark:text-gray-400 ml-2">
                    ({result.duration}ms)
                  </span>
                )}
              </div>
              {result.error && (
                <div className="text-sm text-red-600 dark:text-red-400 mt-1">{result.error}</div>
              )}
              {result.result && (
                <pre className="text-xs mt-2 overflow-auto bg-gray-100 dark:bg-gray-900 p-2 rounded">
                  {JSON.stringify(result.result, null, 2)}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
