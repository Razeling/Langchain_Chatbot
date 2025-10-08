'use client'

import React, { useState, useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble'
import TypingIndicator from './TypingIndicator'
import FunctionCallDisplay from './FunctionCallDisplay'
import SourceCitations from './SourceCitations'
import Spinner from './Spinner' // Import Spinner component

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || (typeof window !== 'undefined' && window.location.hostname === 'localhost' ? 'http://localhost:8000' : '/api')

const welcomeMessage = `🚗 Welcome to Car Troubleshoot Europe! I'm your automotive diagnostic assistant specialized in European markets.

I can help you with:
• 🔧 Diagnosing car problems and symptoms
• 💰 Calculating real repair costs in EUR (with VAT by country)
• 📅 Creating maintenance schedules  
• 🏛️ European regulations & compliance (MOT, TÜV, Euro 6/7)
• 🌍 Country-specific inspection requirements
• 🚫 Low emission zone restrictions (LEZ/ULEZ)
• ⚡ AdBlue, DPF, and SCR system guidance

Feel free to describe any car issues you're experiencing, and I'll provide expert guidance with accurate European market information!`

export default function ChatInterface({ selectedCountry, vehicleInfo }) {
  const [messages, setMessages] = useState([
    {
      id: '1',
      role: 'assistant',
      content: welcomeMessage,
      timestamp: new Date()
    }
  ])
  
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [error, setError] = useState(null)
  const [functionCalls, setFunctionCalls] = useState([])
  const [allConversationSources, setAllConversationSources] = useState({
    knowledge: [],
    web: [],
    learned: []
  })
  const [newInfoLearned, setNewInfoLearned] = useState(false) // State to track new information learning
  
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Helper function to deduplicate sources by source content
  const deduplicateSources = (sources) => {
    const seen = new Set()
    return sources.filter(source => {
      const key = source.source || source.content || source.title
      if (seen.has(key)) {
        return false
      }
      seen.add(key)
      return true
    })
  }

  // Helper function to update conversation sources
  const updateConversationSources = (newSources) => {
    setAllConversationSources(prev => ({
      knowledge: deduplicateSources([
        ...prev.knowledge,
        ...(newSources.knowledge || [])
      ]),
      web: deduplicateSources([
        ...prev.web,
        ...(newSources.web || [])
      ]),
      learned: deduplicateSources([
        ...prev.learned,
        ...(newSources.learned || [])
      ])
    }))
  }

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)
    setIsTyping(true)
    setError(null)
    setFunctionCalls([])

    try {
      // Create AbortController for timeout handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 45000); // 45 second timeout for Lithuanian/multilingual queries
      
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage.trim(),
          messages: messages.slice(-5).map(msg => ({
            id: msg.id,
            role: msg.role,
            content: msg.content,
            timestamp: new Date(msg.timestamp).toISOString()
          })), // Last 5 messages for context with proper timestamp format
          vehicle_info: {
            make: vehicleInfo.make || null,
            model: vehicleInfo.model || null,
            year: vehicleInfo.year ? parseInt(vehicleInfo.year) : null,
            mileage: vehicleInfo.mileage ? parseInt(vehicleInfo.mileage) : null,
            engine_type: vehicleInfo.engine_type || null,
            transmission: vehicleInfo.transmission || null
          },
          country: selectedCountry,
          include_sources: true,
          stream: false
        }),
        signal: controller.signal
      })
      
      clearTimeout(timeoutId); // Clear timeout if request completes

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      // Parse function calls if they exist
      if (data.function_calls && data.function_calls.length > 0) {
        setFunctionCalls(data.function_calls)
      }

      // Update conversation sources if they exist
      if (data.knowledge_sources || data.web_sources || data.learned_documents || data.pure_knowledge_sources || data.web_learned_sources) {
        updateConversationSources({
          knowledge: [
            ...(data.pure_knowledge_sources || []),
            ...(data.knowledge_sources || []) // Legacy fallback
          ],
          web: [
            ...(data.web_sources || []),
            ...(data.web_learned_sources || [])
          ],
          learned: data.learned_documents || []
        })
        
        // Only show notification if there are actually NEW documents learned
        if (data.learned_documents && data.learned_documents.length > 0) {
          setNewInfoLearned(true) // Set new information learned to true
        }
      }

      const assistantMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response || data.message || 'I apologize, but I encountered an issue processing your request.',
        timestamp: new Date(),
        // Categorized source fields from backend
        pure_knowledge_sources: data.pure_knowledge_sources || [],
        web_learned_sources: data.web_learned_sources || [],
        web_sources: data.web_sources || [],
        learnedDocuments: data.learned_documents || [],
        // Legacy fallback fields
        sources: data.sources || data.knowledge_sources || [],
        webSources: data.web_sources || []
      }

      setMessages(prev => [...prev, assistantMessage])

    } catch (error) {
      console.error('Chat error:', error)
      
      let errorMessage = 'Sorry, I encountered an error. Please try again.';
      let chatErrorContent = 'I apologize, but I encountered an error processing your request.';
      
      // Handle specific error types
      if (error.name === 'AbortError') {
        errorMessage = 'Request timed out. Please try a shorter query or check your connection.';
        chatErrorContent = 'Your query is taking longer than expected. This can happen with complex multilingual queries. Please try again or use a simpler question.';
      } else if (error.message.includes('fetch')) {
        errorMessage = 'Connection error. Please check your internet connection.';
        chatErrorContent = 'I apologize, but I encountered a connection error. Please check your internet connection and try again.';
      } else if (error.message.includes('HTTP error')) {
        errorMessage = `Server error: ${error.message}`;
        chatErrorContent = 'The server encountered an error. Please try again in a moment.';
      }
      
      setError(errorMessage)
      
      const chatErrorMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: chatErrorContent,
        timestamp: new Date(),
        isError: true
      }
      
      setMessages(prev => [...prev, chatErrorMessage])
    } finally {
      setIsLoading(false)
      setIsTyping(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => {
    setMessages([
      {
        id: '1',
        role: 'assistant',
        content: welcomeMessage,
        timestamp: new Date()
      }
    ])
    setAllConversationSources({
      knowledge: [],
      web: [],
      learned: []
    })
    setFunctionCalls([])
    setError(null)
  }

  // Handle new info learned notification
  useEffect(() => {
    if (newInfoLearned) {
      const timer = setTimeout(() => setNewInfoLearned(false), 5000)
      return () => clearTimeout(timer)
    }
  }, [newInfoLearned])

  return (
    <div className="flex flex-col h-[700px] max-h-[80vh]">
      {isLoading && <Spinner />} {/* Show spinner when loading */}
      {newInfoLearned && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
          New information has been learned and integrated!
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-green-50">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-green-500 rounded-full flex items-center justify-center">
            <span className="text-white text-lg font-bold">🚗</span>
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              Car Troubleshoot Europe
            </h2>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span>Market Assistant</span>
              <span className="text-xs text-gray-500">Online</span>
            </div>
          </div>
        </div>
        
        <button
          onClick={clearChat}
          title="Clear"
          className="px-3 py-1 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors"
        >
          Clear
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        
        {/* Function Calls Display */}
        {functionCalls.length > 0 && (
          <div className="space-y-2">
            {functionCalls.map((call, index) => (
              <FunctionCallDisplay key={index} functionCall={call} />
            ))}
          </div>
        )}
        
        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <TypingIndicator />
          </div>
        )}
        
        {/* Error Display */}
        {error && (
          <div className="flex justify-center">
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-800">
              {error}
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Sources Summary (if any exist) */}
      {(allConversationSources.knowledge.length > 0 || 
        allConversationSources.web.length > 0 || 
        allConversationSources.learned.length > 0) && (
        <div className="border-t border-gray-200 p-4 bg-blue-50">
          <SourceCitations 
            pureKnowledgeSources={allConversationSources.knowledge.filter(s => !s.source || (!s.source.includes('Web Search') && !s.title.includes('Web Learned')))}
            webLearnedSources={allConversationSources.knowledge.filter(s => s.source && (s.source.includes('Web Search') || s.title.includes('Web Learned')))}
            webSources={allConversationSources.web}
            learnedDocuments={allConversationSources.learned}
            // Legacy fallback
            knowledgeSources={allConversationSources.knowledge}
            legacyWebSources={allConversationSources.web}
            showTitle={true}
            variant="conversation-summary"
          />
        </div>
      )}

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4 bg-white">
        {/* Quick Actions */}
        <div className="flex flex-wrap gap-2 mb-4">
          <QuickActionButton
            text="Check engine symptoms"
            onClick={() => setInputMessage("My engine has been making strange noises and running rough")}
          />
          <QuickActionButton
            text="Estimate repair cost"
            onClick={() => setInputMessage("How much would it cost to replace brake pads?")}
          />
          <QuickActionButton
            text="Maintenance schedule"
            onClick={() => setInputMessage("When should I service my car next?")}
          />
          <QuickActionButton
            text="Winter preparation"
            onClick={() => setInputMessage("How should I prepare my car for Baltic winter?")}
          />
        </div>
        
        {/* Input Field */}
        <div className="flex items-end space-x-3">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Describe your car problem or ask for advice..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
              rows="2"
              maxLength={500}
              disabled={isLoading}
            />
            <div className="text-xs text-gray-500 mt-1 flex justify-between">
              <span>Press Enter to send, Shift+Enter for new line</span>
              <span>{inputMessage.length}/500</span>
            </div>
          </div>
          
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center min-w-[80px] h-12"
          >
            {isLoading ? (
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span className="text-sm">Sending</span>
              </div>
            ) : (
              <span className="text-sm font-medium">Send</span>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

// Quick Action Button Component
function QuickActionButton({ text, onClick }) {
  return (
    <button
      onClick={onClick}
      className="px-4 py-2 text-sm bg-gradient-to-r from-blue-50 to-indigo-50 hover:from-blue-100 hover:to-indigo-100 text-blue-700 rounded-lg border border-blue-200 transition-all duration-200 hover:shadow-sm"
    >
      {text}
    </button>
  )
}

 