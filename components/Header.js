'use client'

import React from 'react'

export default function Header() {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-green-600 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-xl">🚗</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Car Troubleshoot Europe
              </h1>
              <p className="text-sm text-gray-600">
                AI-powered automotive diagnostic assistant for European markets
              </p>
            </div>
          </div>

          {/* European Union Flag */}
          <div className="hidden md:flex items-center space-x-2">
            <div className="text-3xl">🇪🇺</div>
            <div className="text-sm text-gray-600">
              <div>European Market</div>
              <div className="text-xs">Specialized for EU regulations</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
} 