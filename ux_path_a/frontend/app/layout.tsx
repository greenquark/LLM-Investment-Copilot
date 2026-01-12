import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'UX Path A - Smart Trading Copilot',
  description: 'ChatGPT-style conversational market analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
