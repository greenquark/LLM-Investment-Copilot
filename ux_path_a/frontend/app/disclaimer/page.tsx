"use client"

import Link from "next/link"

export default function DisclaimerPage() {
  return (
    <main className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="mx-auto max-w-3xl px-6 py-10">
        <div className="mb-6">
          <Link href="/" className="text-sm text-blue-600 dark:text-blue-400 hover:underline">
            ← Back to chat
          </Link>
        </div>

        <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">Disclaimer</h1>

        <div className="mt-6 space-y-4 text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          <p>
            <strong>Educational and research purposes only.</strong> Smart Trading Copilot is provided for education and
            research. It is not financial, investment, legal, or tax advice.
          </p>
          <p>
            <strong>No recommendations.</strong> The app does not recommend specific trades or provide personalized
            investment advice. You are responsible for your own decisions.
          </p>
          <p>
            <strong>Market risk.</strong> Trading and investing involve risk, including the possible loss of principal.
            Past performance does not guarantee future results.
          </p>
          <p>
            <strong>Data limitations.</strong> Market data may be delayed, incomplete, or unavailable. Always verify
            information with official sources before acting.
          </p>
        </div>
      </div>
    </main>
  )
}

