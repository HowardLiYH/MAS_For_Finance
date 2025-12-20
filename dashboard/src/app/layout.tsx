import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'PopAgent Dashboard',
  description: 'Multi-Agent LLM Trading Visualization',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        <div className="min-h-screen grid-pattern">
          {children}
        </div>
      </body>
    </html>
  )
}

