'use client'

import React, { useState } from 'react'

export default function SourceCitations({ 
  pureKnowledgeSources = [], 
  webLearnedSources = [], 
  webSources = [], 
  learnedDocuments = [],
  // Legacy parameters for backward compatibility
  knowledgeSources = [], 
  legacyWebSources = []
}) {
  const [isExpanded, setIsExpanded] = useState(false)

  // Simple inline deduplication function
  const deduplicateSources = (sources) => {
    const seen = new Set()
    return sources.filter(source => {
      const key = source.content || source.title || source.source
      if (seen.has(key)) {
        return false
      }
      seen.add(key)
      return true
    })
  }

  // Log the data structure for debugging
  console.log('Pure Knowledge Sources:', pureKnowledgeSources)
  console.log('Web Learned Sources:', webLearnedSources)
  console.log('Web Sources:', webSources)
  console.log('Learned Documents:', learnedDocuments)
  console.log('Legacy Knowledge Sources:', knowledgeSources)
  console.log('Legacy Web Sources:', legacyWebSources)

  let uniquePureKnowledgeSources = []
  let uniqueWebLearnedSources = []
  let uniqueWebSources = []
  let uniqueLearnedDocuments = []

  try {
    // Use new separate category fields if available, otherwise fall back to legacy
    uniquePureKnowledgeSources = deduplicateSources(pureKnowledgeSources.length > 0 ? pureKnowledgeSources : knowledgeSources)
    uniqueWebLearnedSources = deduplicateSources(webLearnedSources)
    uniqueWebSources = deduplicateSources(webSources.length > 0 ? webSources : legacyWebSources)
    uniqueLearnedDocuments = deduplicateSources(learnedDocuments)
  } catch (error) {
    console.error('Error during deduplication:', error)
  }

  const totalSources = uniquePureKnowledgeSources.length + uniqueWebLearnedSources.length + uniqueWebSources.length + uniqueLearnedDocuments.length
  
  if (totalSources === 0) {
    return null
  }

  return (
    <div className="mt-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center space-x-2 text-sm text-blue-600 hover:text-blue-800 transition-colors"
      >
        <span>
          📚 Knowledge Sources ({uniquePureKnowledgeSources.length})
          {uniqueWebLearnedSources.length > 0 && ` + 🧠 Web Learned (${uniqueWebLearnedSources.length})`}
          {uniqueWebSources.length > 0 && ` + 🌐 Web Sources (${uniqueWebSources.length})`}
          {uniqueLearnedDocuments.length > 0 && ` + 🆕 New Learning (${uniqueLearnedDocuments.length})`}
        </span>
        <span className="text-xs">{isExpanded ? '▼' : '▶'}</span>
      </button>

      {isExpanded && (
        <div className="mt-2 space-y-3">
          {/* Pure Knowledge Base Sources */}
          {uniquePureKnowledgeSources.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-700 mb-2">📚 Internal Knowledge Base</h4>
              <div className="space-y-2">
                {uniquePureKnowledgeSources.map((source, index) => (
                  <SourceCitation key={`kb-${index}`} source={source} index={index} type="knowledge" />
                ))}
              </div>
            </div>
          )}
          
          {/* Web-Learned Sources */}
          {uniqueWebLearnedSources.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-700 mb-2">🧠 Previously Learned from Web</h4>
              <div className="space-y-2">
                {uniqueWebLearnedSources.map((source, index) => (
                  <SourceCitation key={`web-learned-${index}`} source={source} index={index} type="web-learned" />
                ))}
              </div>
            </div>
          )}
          
          {/* Fresh Web Search Sources */}
          {uniqueWebSources.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-700 mb-2">🌐 Web Sources</h4>
              <div className="space-y-2">
                {uniqueWebSources.map((source, index) => (
                  <SourceCitation key={`web-${index}`} source={source} index={index} type="web" />
                ))}
              </div>
            </div>
          )}

          {/* New Learning Documents */}
          {uniqueLearnedDocuments.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-700 mb-2">🆕 New Learning</h4>
              <div className="space-y-2">
                {uniqueLearnedDocuments.map((source, index) => (
                  <SourceCitation key={`learned-${index}`} source={source} index={index} type="learned" />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function SourceCitation({ source, index, type }) {
  const [isExpanded, setIsExpanded] = useState(false)

  const getSimilarityColor = (similarity) => {
    if (similarity > 0.8) return 'text-green-600'
    if (similarity > 0.6) return 'text-yellow-600'
    return 'text-gray-600'
  }

  const getTypeConfig = (type) => {
    switch (type) {
      case 'web':
        return {
          icon: '🌐',
          borderColor: 'border-orange-500',
          bgColor: 'bg-orange-50',
          badge: { text: 'Fresh', color: 'bg-orange-100 text-orange-800' }
        }
      case 'web-learned':
        return {
          icon: '🧠',
          borderColor: 'border-purple-500',
          bgColor: 'bg-purple-50',
          badge: { text: 'Learned', color: 'bg-purple-100 text-purple-800' }
        }
      default:
        return {
          icon: '📚',
          borderColor: 'border-blue-500',
          bgColor: 'bg-gray-50',
          badge: null
        }
    }
  }

  const typeConfig = getTypeConfig(type)

  return (
    <div className={`source-citation ${typeConfig.bgColor} border-l-4 ${typeConfig.borderColor} p-3 rounded-r`}>
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex-1">
          <div className="text-sm font-medium text-gray-900 flex items-center">
            <span className="mr-2">{typeConfig.icon}</span>
            {source.title || `Source ${index + 1}`}
            {typeConfig.badge && (
              <span className={`ml-2 px-2 py-1 ${typeConfig.badge.color} text-xs rounded-full`}>
                {typeConfig.badge.text}
              </span>
            )}
          </div>
          <div className="text-xs text-gray-600 mt-1">
            {type === 'web' || type === 'web-learned' ? (
              <>
                {type === 'web-learned' ? 'Previously learned from: ' : 'Domain: '}
                {type === 'web-learned' 
                  ? (source.source?.replace('Web Search - ', '') || 'Web Source')
                  : (source.metadata?.domain || 'Unknown')
                }
                {source.similarity && (
                  <span className={`ml-2 source-similarity ${getSimilarityColor(source.similarity)}`}>
                    Relevance: {Math.round(source.similarity * 100)}%
                  </span>
                )}
              </>
            ) : (
              <>
                Category: {source.category || 'Unknown'}
                {source.similarity && (
                  <span className={`ml-2 source-similarity ${getSimilarityColor(source.similarity)}`}>
                    Relevance: {Math.round(source.similarity * 100)}%
                  </span>
                )}
              </>
            )}
          </div>
        </div>
        
        <button className="text-gray-500 hover:text-gray-700 transition-colors ml-2">
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>

      {isExpanded && (
        <div className="mt-3 space-y-2">
          {/* Source Content */}
          <div>
            <h4 className="text-xs font-medium text-gray-700 mb-1">Content:</h4>
            <div className="text-xs text-gray-800 bg-white bg-opacity-60 rounded border p-2">
              <div className="whitespace-pre-wrap">
                {source.content ? source.content.substring(0, 500) : 'No content available'}
                {source.content && source.content.length > 500 && '...'}
              </div>
            </div>
          </div>

          {/* Tags */}
          {source.tags && source.tags.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-gray-700 mb-1">Tags:</h4>
              <div className="flex flex-wrap gap-1">
                {source.tags.map((tag, tagIndex) => (
                  <span
                    key={tagIndex}
                    className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Source Origin */}
          {source.source && (
            <div>
              <h4 className="text-xs font-medium text-gray-700 mb-1">Source:</h4>
              <div className="text-xs text-gray-600">
                {source.source}
              </div>
            </div>
          )}

          {/* Web Source URL */}
          {(type === 'web' || type === 'web-learned') && source.metadata?.url && (
            <div>
              <h4 className="text-xs font-medium text-gray-700 mb-1">URL:</h4>
              <div className="text-xs">
                <a 
                  href={source.metadata.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 underline break-all"
                >
                  {source.metadata.url}
                </a>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function LearnedDocument({ document, index }) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div className="source-citation bg-green-50 border-l-4 border-green-500 p-3 rounded-r">
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex-1">
          <div className="text-sm font-medium text-gray-900 flex items-center">
            <span className="mr-2">🧠</span>
            {document.title || `Learned Document ${index + 1}`}
            <span className="ml-2 px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
              New Knowledge
            </span>
          </div>
          <div className="text-xs text-gray-600 mt-1">
            Category: {document.category || 'Unknown'}
            <span className="ml-2 text-green-600">
              ⏰ Learned: {new Date(document.learned_at || document.last_updated).toLocaleTimeString()}
            </span>
          </div>
        </div>
        
        <button className="text-gray-500 hover:text-gray-700 transition-colors ml-2">
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>

      {isExpanded && (
        <div className="mt-3 space-y-2">
          {/* Document Content */}
          <div>
            <h4 className="text-xs font-medium text-gray-700 mb-1">Content:</h4>
            <div className="text-xs text-gray-800 bg-white bg-opacity-60 rounded border p-2">
              <div className="whitespace-pre-wrap">
                {document.content ? document.content.substring(0, 500) : 'No content available'}
                {document.content && document.content.length > 500 && '...'}
              </div>
            </div>
          </div>

          {/* Tags */}
          {document.tags && document.tags.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-gray-700 mb-1">Tags:</h4>
              <div className="flex flex-wrap gap-1">
                {document.tags.map((tag, tagIndex) => (
                  <span
                    key={tagIndex}
                    className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Source Origin */}
          {document.source && (
            <div>
              <h4 className="text-xs font-medium text-gray-700 mb-1">Source:</h4>
              <div className="text-xs text-gray-600">
                {document.source}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
} 