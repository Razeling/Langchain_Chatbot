'use client'

import React, { useState } from 'react'

export default function FunctionCallDisplay({ functionCalls }) {
  const [expandedCalls, setExpandedCalls] = useState(new Set())

  const toggleExpanded = (callId) => {
    const newExpanded = new Set(expandedCalls)
    if (newExpanded.has(callId)) {
      newExpanded.delete(callId)
    } else {
      newExpanded.add(callId)
    }
    setExpandedCalls(newExpanded)
  }

  if (!functionCalls || functionCalls.length === 0) {
    return null
  }

  return (
    <div className="space-y-2">
      {functionCalls.map((call, index) => (
        <FunctionCallItem
          key={`${call.name}-${index}`}
          call={call}
          index={index}
          isExpanded={expandedCalls.has(`${call.name}-${index}`)}
          onToggle={() => toggleExpanded(`${call.name}-${index}`)}
        />
      ))}
    </div>
  )
}

function FunctionCallItem({ call, index, isExpanded, onToggle }) {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return '✅'
      case 'failed':
        return '❌'
      case 'pending':
        return '⏳'
      default:
        return '🔧'
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'border-green-200 bg-green-50'
      case 'failed':
        return 'border-red-200 bg-red-50'
      case 'pending':
        return 'border-yellow-200 bg-yellow-50'
      default:
        return 'border-blue-200 bg-blue-50'
    }
  }

  const getFunctionDisplayName = (name) => {
    switch (name) {
      case 'diagnoseProblem':
        return '🔍 Diagnostic Analysis'
      case 'estimateRepairCost':
        return '💰 Cost Estimation'
      case 'generateMaintenanceSchedule':
        return '📅 Maintenance Schedule'
      default:
        return `🔧 ${name}`
    }
  }

  return (
    <div className={`function-call border rounded-lg p-4 ${getStatusColor(call.status)}`}>
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={onToggle}
      >
        <div className="flex items-center space-x-2">
          <span className="text-lg">{getStatusIcon(call.status)}</span>
          <span className="font-medium text-gray-900">
            {getFunctionDisplayName(call.name)}
          </span>
          {call.status && (
            <span className="text-xs text-gray-500 capitalize">
              ({call.status})
            </span>
          )}
        </div>
        
        <button className="text-gray-500 hover:text-gray-700 transition-colors">
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>

      {isExpanded && (
        <div className="mt-3 space-y-3">
          {/* Function Arguments */}
          {call.arguments && Object.keys(call.arguments).length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-1">Parameters:</h4>
              <div className="bg-white bg-opacity-60 rounded border p-2 text-xs">
                <pre className="whitespace-pre-wrap">
                  {JSON.stringify(call.arguments, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Function Result */}
          {call.result && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-1">Result:</h4>
              <div className="bg-white bg-opacity-60 rounded border p-3">
                <FunctionResult result={call.result} functionName={call.name} />
              </div>
            </div>
          )}

          {/* Error Message */}
          {call.error && (
            <div>
              <h4 className="text-sm font-medium text-red-700 mb-1">Error:</h4>
              <div className="bg-red-100 rounded border border-red-200 p-2 text-xs text-red-800">
                {call.error}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function FunctionResult({ result, functionName }) {
  if (!result) return null

  // Handle different result types
  if (typeof result === 'string') {
    return <div className="text-sm">{result}</div>
  }

  if (typeof result === 'object') {
    return (
      <div className="space-y-2">
        {functionName === 'diagnoseProblem' && (
          <DiagnosticResult result={result} />
        )}
        {functionName === 'estimateRepairCost' && (
          <CostEstimateResult result={result} />
        )}
        {functionName === 'generateMaintenanceSchedule' && (
          <MaintenanceResult result={result} />
        )}
        {!['diagnoseProblem', 'estimateRepairCost', 'generateMaintenanceSchedule'].includes(functionName) && (
          <pre className="text-xs whitespace-pre-wrap">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </div>
    )
  }

  return <div className="text-sm">{String(result)}</div>
}

function DiagnosticResult({ result }) {
  return (
    <div className="space-y-2 text-sm">
      {result.problem && (
        <div>
          <span className="font-medium text-gray-700">Problem:</span>
          <span className="ml-2">{result.problem}</span>
        </div>
      )}
      {result.severity && (
        <div>
          <span className="font-medium text-gray-700">Severity:</span>
          <span className={`ml-2 px-2 py-1 rounded text-xs ${
            result.severity === 'high' || result.severity === 'critical' ? 'bg-red-100 text-red-800' :
            result.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
            'bg-green-100 text-green-800'
          }`}>
            {result.severity}
          </span>
        </div>
      )}
      {result.description && (
        <div>
          <span className="font-medium text-gray-700">Description:</span>
          <span className="ml-2">{result.description}</span>
        </div>
      )}
      {result.possible_causes && result.possible_causes.length > 0 && (
        <div>
          <span className="font-medium text-gray-700">Possible Causes:</span>
          <ul className="ml-4 mt-1 list-disc text-xs">
            {result.possible_causes.map((cause, idx) => (
              <li key={idx}>{cause}</li>
            ))}
          </ul>
        </div>
      )}
      {result.recommended_actions && result.recommended_actions.length > 0 && (
        <div>
          <span className="font-medium text-gray-700">Recommended Actions:</span>
          <ul className="ml-4 mt-1 list-disc text-xs">
            {result.recommended_actions.map((action, idx) => (
              <li key={idx}>{action}</li>
            ))}
          </ul>
        </div>
      )}
      {result.urgency && (
        <div>
          <span className="font-medium text-gray-700">Urgency:</span>
          <span className="ml-2">{result.urgency}</span>
        </div>
      )}
      {result.safety_notes && result.safety_notes.length > 0 && (
        <div>
          <span className="font-medium text-gray-700">Safety Notes:</span>
          <ul className="ml-4 mt-1 list-disc text-xs text-red-700">
            {result.safety_notes.map((note, idx) => (
              <li key={idx}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function CostEstimateResult({ result }) {
  return (
    <div className="space-y-2 text-sm">
      {result.total_cost && (
        <div className="text-lg font-bold text-green-700">
          Total: {result.total_cost}
        </div>
      )}
      {result.labor_cost && (
        <div>
          <span className="font-medium text-gray-700">Labor:</span>
          <span className="ml-2">{result.labor_cost}</span>
        </div>
      )}
      {result.parts_cost && (
        <div>
          <span className="font-medium text-gray-700">Parts:</span>
          <span className="ml-2">{result.parts_cost}</span>
        </div>
      )}
      {result.breakdown && result.breakdown.length > 0 && (
        <div>
          <span className="font-medium text-gray-700">Cost Breakdown:</span>
          <ul className="ml-4 mt-1 text-xs">
            {result.breakdown.map((item, idx) => (
              <li key={idx} className="flex justify-between">
                <span>{item.description}</span>
                <span className="font-medium">{item.cost}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function MaintenanceResult({ result }) {
  return (
    <div className="space-y-2 text-sm">
      {result.next_service && (
        <div>
          <span className="font-medium text-gray-700">Next Service:</span>
          <span className="ml-2">{result.next_service}</span>
        </div>
      )}
      {result.upcoming_items && result.upcoming_items.length > 0 && (
        <div>
          <span className="font-medium text-gray-700">Upcoming Maintenance:</span>
          <ul className="ml-4 mt-1 list-disc text-xs">
            {result.upcoming_items.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
} 