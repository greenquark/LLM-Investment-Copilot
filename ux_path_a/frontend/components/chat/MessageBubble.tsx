'use client'

import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import ChartRenderer, { ChartData } from './ChartRenderer'

// Allow relative links like "/disclaimer" through sanitize so markdown links work.
const sanitizeSchema: any = {
  ...defaultSchema,
  attributes: {
    ...(defaultSchema as any).attributes,
    a: [
      ...((((defaultSchema as any).attributes || {}).a || [])),
      ['href'],
      ['target'],
      ['rel'],
    ],
  },
}

interface Message {
  role: string
  content: string
  timestamp: string
  thinking_content?: string
  tool_calls?: any
  tool_results?: any
}

interface MessageBubbleProps {
  message: Message
  showMeta?: boolean
}

export default function MessageBubble({ message, showMeta = false }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const [showThinking, setShowThinking] = React.useState(false)

  const toolNames = React.useMemo(() => {
    if (!message.tool_calls) return []
    const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [message.tool_calls]
    return calls
      .map((t: any) => t?.name || t?.function?.name || t?.id)
      .filter(Boolean)
  }, [message.tool_calls])

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-3xl rounded-lg px-4 py-3 ${
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-700'
        }`}
      >
        {/* Thinking/Reasoning Process */}
        {!isUser && message.thinking_content && (
          <div className="mb-3 pb-3 border-b border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setShowThinking(!showThinking)}
              className="flex items-center gap-2 text-xs font-medium text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
            >
              <span>{showThinking ? '▼' : '▶'}</span>
              <span>Thinking Process</span>
            </button>
            {showThinking && (
              <div className="mt-2 p-3 bg-gray-50 dark:bg-gray-900 rounded-md border border-gray-200 dark:border-gray-700">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkBreaks]}
                  rehypePlugins={[rehypeRaw, [rehypeSanitize, sanitizeSchema]]}
                  className="prose prose-xs dark:prose-invert max-w-none text-gray-700 dark:text-gray-300"
                >
                  {message.thinking_content}
                </ReactMarkdown>
              </div>
            )}
          </div>
        )}

        {/* Main Content with Enhanced Markdown */}
        <div className="prose prose-sm dark:prose-invert max-w-none
                        prose-headings:font-semibold prose-headings:tracking-tight
                        prose-p:leading-relaxed prose-p:my-3
                        prose-li:my-1
                        prose-ul:pl-5 prose-ol:pl-5
                        prose-a:text-blue-600 dark:prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline
                        prose-strong:font-semibold
                        prose-code:bg-gray-100 dark:prose-code:bg-gray-800 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-sm
                        prose-pre:bg-transparent prose-pre:p-0
                        prose-blockquote:border-l-4 prose-blockquote:border-gray-300 dark:prose-blockquote:border-gray-600 prose-blockquote:pl-4 prose-blockquote:italic
                        prose-ul:list-disc prose-ol:list-decimal
                        prose-table:w-full prose-table:border-collapse prose-table:border prose-table:border-gray-300 dark:prose-table:border-gray-700 prose-table:my-4
                        prose-th:border prose-th:border-gray-300 dark:prose-th:border-gray-700 prose-th:bg-gray-100 dark:prose-th:bg-gray-800 prose-th:p-2 prose-th:font-semibold prose-th:text-left
                        prose-td:border prose-td:border-gray-300 dark:prose-td:border-gray-700 prose-td:p-2">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkBreaks]}
            rehypePlugins={[rehypeRaw, [rehypeSanitize, sanitizeSchema]]}
            components={{
              code({ node, inline, className, children, ...props }: any) {
                const match = /language-(\w+)/.exec(className || '')
                const language = match ? match[1] : ''
                const codeContent = String(children).replace(/\n$/, '')
                
                // Check if this is a chart code block (language is 'chart' or 'json:chart')
                if (!inline && (language === 'chart' || language === 'json:chart')) {
                  try {
                    const chartData: ChartData = JSON.parse(codeContent)
                    // Validate chart data structure
                    if (chartData.type && chartData.data && Array.isArray(chartData.data)) {
                      return <ChartRenderer chartData={chartData} />
                    }
                  } catch (e) {
                    // If parsing fails, fall through to regular code display
                    console.warn('Failed to parse chart data:', e)
                  }
                }
                
                // Check if this is JSON that might be chart data (heuristic detection)
                if (!inline && language === 'json') {
                  try {
                    const parsed = JSON.parse(codeContent)
                    // Check if it looks like chart data
                    if (
                      parsed.type &&
                      ['line', 'candlestick', 'bar', 'scatter', 'area'].includes(parsed.type) &&
                      parsed.data &&
                      Array.isArray(parsed.data)
                    ) {
                      return <ChartRenderer chartData={parsed as ChartData} />
                    }
                  } catch (e) {
                    // Not valid JSON or not chart data, continue to syntax highlighting
                  }
                }
                
                return !inline && match ? (
                  <div className="relative my-4">
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={language}
                      PreTag="div"
                      className="rounded-md !m-0"
                      customStyle={{
                        margin: 0,
                        borderRadius: '0.375rem',
                      }}
                      {...props}
                    >
                      {codeContent}
                    </SyntaxHighlighter>
                  </div>
                ) : (
                  <code className={`${className} bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-sm`} {...props}>
                    {children}
                  </code>
                )
              },
              table({ children }: any) {
                return (
                  <div className="overflow-x-auto my-4">
                    <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-700">
                      {children}
                    </table>
                  </div>
                )
              },
              a({ href, children }: any) {
                return (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    {children}
                  </a>
                )
              },
            }}
          >
            {typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)}
          </ReactMarkdown>
        </div>

        {/* Tool meta (compact) */}
        {showMeta && !isUser && (toolNames.length > 0 || (message.tool_results && message.tool_results.length > 0)) && (
          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700 space-y-2 text-xs">
            {toolNames.length > 0 && (
              <div className="bg-gray-50 dark:bg-gray-900 px-2 py-1 rounded border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300">
                <span className="font-semibold">Tool used:</span>{' '}
                {toolNames.map((n: string, idx: number) => (
                  <span key={`${n}-${idx}`}>
                    <code className="bg-white/60 dark:bg-gray-800 px-1 py-0.5 rounded">{n}</code>
                    {idx < toolNames.length - 1 ? ', ' : ''}
                  </span>
                ))}
              </div>
            )}

            {message.tool_results && message.tool_results.length > 0 && (
              <details className="bg-gray-50 dark:bg-gray-900 px-2 py-1 rounded border border-gray-200 dark:border-gray-700">
                <summary className="cursor-pointer text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white font-semibold">
                  Tool response{message.tool_results.length ? ` (${message.tool_results.length})` : ''}
                </summary>
                <div className="mt-2 space-y-2">
                  {message.tool_results.map((result: any, idx: number) => (
                    <div key={idx} className="bg-white dark:bg-gray-950 p-2 rounded border border-gray-200 dark:border-gray-700">
                      <div className="font-semibold text-gray-700 dark:text-gray-300 mb-1">
                        {result.name || 'Result'}
                      </div>
                      <pre className="text-xs mt-1 overflow-x-auto text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
                        {typeof result.result === 'string'
                          ? result.result
                          : JSON.stringify(result.result, null, 2)}
                      </pre>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}

        {showMeta && (
          <div className={`text-xs mt-2 ${isUser ? 'text-blue-100' : 'text-gray-500'}`}>
            {new Date(message.timestamp).toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  )
}
