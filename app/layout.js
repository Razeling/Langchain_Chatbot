import './globals.css'

export const metadata = {
  title: 'Car Troubleshoot Europe - Automotive Diagnostic Assistant',
  description: 'AI-powered car troubleshooting and diagnostic assistance for European markets, specializing in Lithuania and surrounding regions.',
  keywords: 'car repair, automotive diagnostic, Europe, Lithuania, car troubleshooting',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="description" content={metadata.description} />
        <meta name="keywords" content={metadata.keywords} />
        <title>{metadata.title}</title>
      </head>
      <body className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
        <div id="root">{children}</div>
      </body>
    </html>
  )
} 