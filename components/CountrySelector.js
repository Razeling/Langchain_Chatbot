'use client'

import React, { useState, useRef, useEffect } from 'react'

// European countries data
const europeanCountries = [
  { code: 'LT', name: 'Lithuania', currency: 'EUR', language: 'Lithuanian' },
  { code: 'LV', name: 'Latvia', currency: 'EUR', language: 'Latvian' },
  { code: 'EE', name: 'Estonia', currency: 'EUR', language: 'Estonian' },
  { code: 'PL', name: 'Poland', currency: 'PLN', language: 'Polish' },
  { code: 'DE', name: 'Germany', currency: 'EUR', language: 'German' },
  { code: 'FR', name: 'France', currency: 'EUR', language: 'French' },
  { code: 'IT', name: 'Italy', currency: 'EUR', language: 'Italian' },
  { code: 'ES', name: 'Spain', currency: 'EUR', language: 'Spanish' },
  { code: 'NL', name: 'Netherlands', currency: 'EUR', language: 'Dutch' },
  { code: 'BE', name: 'Belgium', currency: 'EUR', language: 'Dutch/French' },
  { code: 'AT', name: 'Austria', currency: 'EUR', language: 'German' },
  { code: 'CH', name: 'Switzerland', currency: 'CHF', language: 'German/French' },
  { code: 'CZ', name: 'Czech Republic', currency: 'CZK', language: 'Czech' },
  { code: 'SK', name: 'Slovakia', currency: 'EUR', language: 'Slovak' },
  { code: 'HU', name: 'Hungary', currency: 'HUF', language: 'Hungarian' },
  { code: 'SI', name: 'Slovenia', currency: 'EUR', language: 'Slovenian' },
  { code: 'HR', name: 'Croatia', currency: 'EUR', language: 'Croatian' },
  { code: 'FI', name: 'Finland', currency: 'EUR', language: 'Finnish' },
  { code: 'SE', name: 'Sweden', currency: 'SEK', language: 'Swedish' },
  { code: 'NO', name: 'Norway', currency: 'NOK', language: 'Norwegian' },
  { code: 'DK', name: 'Denmark', currency: 'DKK', language: 'Danish' },
  { code: 'IE', name: 'Ireland', currency: 'EUR', language: 'English' },
  { code: 'GB', name: 'United Kingdom', currency: 'GBP', language: 'English' },
  { code: 'PT', name: 'Portugal', currency: 'EUR', language: 'Portuguese' },
  { code: 'GR', name: 'Greece', currency: 'EUR', language: 'Greek' },
  { code: 'BG', name: 'Bulgaria', currency: 'BGN', language: 'Bulgarian' },
  { code: 'RO', name: 'Romania', currency: 'RON', language: 'Romanian' }
]

// Flag Icon Component
function FlagIcon({ countryCode, size = "w-6 h-4" }) {
  const [imgError, setImgError] = useState(false)

  if (imgError || !countryCode) {
    return (
      <div className={`${size} bg-gray-200 rounded-sm flex items-center justify-center border border-gray-300`}>
        <span className="text-xs text-gray-500 font-semibold">{countryCode}</span>
      </div>
    )
  }

  return (
    <div className={`${size} relative rounded-sm overflow-hidden border border-gray-200`}>
      <img
        src={`https://flagcdn.com/${countryCode.toLowerCase()}.svg`}
        alt={`${countryCode} flag`}
        className={`${size} object-cover`}
        onError={() => setImgError(true)}
        loading="eager"
      />
    </div>
  )
}

export default function CountrySelector({ selectedCountry, onCountryChange }) {
  const [isOpen, setIsOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const dropdownRef = useRef(null)
  const currentCountry = europeanCountries.find(country => country.code === selectedCountry) || europeanCountries[0]

  // Filter countries based on search term
  const filteredCountries = europeanCountries.filter(country =>
    country.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    country.code.toLowerCase().includes(searchTerm.toLowerCase())
  )

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false)
        setSearchTerm('')
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [])

  const handleCountrySelect = (countryCode) => {
    onCountryChange(countryCode)
    setIsOpen(false)
    setSearchTerm('')
  }

  return (
    <div className="country-selector">
      <label className="block text-sm font-medium text-gray-700 mb-3">
        Select your country for localized pricing and regulations
      </label>
      
      {/* Custom Dropdown */}
      <div className="relative" ref={dropdownRef}>
        {/* Dropdown Button */}
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="w-full px-4 py-3 text-left border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white hover:bg-gray-50 transition-colors"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <FlagIcon countryCode={currentCountry.code} size="w-8 h-5" />
              <span className="text-gray-900 font-medium">{currentCountry.name}</span>
            </div>
            <svg
              className={`w-5 h-5 text-gray-400 transition-transform ${isOpen ? 'transform rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </button>

        {/* Dropdown Menu */}
        {isOpen && (
          <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
            {/* Search Input */}
            <div className="p-3 border-b border-gray-200">
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search countries..."
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                autoFocus
              />
            </div>
            
            {/* Countries List */}
            <div className="py-1">
              {filteredCountries.length > 0 ? (
                filteredCountries.map((country) => (
                  <button
                    key={country.code}
                    onClick={() => handleCountrySelect(country.code)}
                    className={`w-full px-4 py-3 text-left hover:bg-gray-50 focus:outline-none focus:bg-gray-50 flex items-center space-x-3 transition-colors ${
                      country.code === selectedCountry ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-500' : 'text-gray-900'
                    }`}
                  >
                    <FlagIcon countryCode={country.code} size="w-6 h-4" />
                    <span className="flex-1 font-medium">{country.name}</span>
                    <span className="text-xs text-gray-500 font-semibold">{country.code}</span>
                  </button>
                ))
              ) : (
                <div className="px-4 py-3 text-sm text-gray-500 text-center">
                  No countries found
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      
      {/* Selected Country Info */}
      <div className="mt-4 p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg border border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <FlagIcon countryCode={currentCountry.code} size="w-10 h-6" />
          <span className="text-sm font-semibold text-gray-900">{currentCountry.name}</span>
        </div>
        
        <div className="text-xs text-gray-600 space-y-2">
          <div className="flex justify-between">
            <span className="font-medium">Currency:</span>
            <span className="font-semibold">{currentCountry.currency}</span>
          </div>
          <div className="flex justify-between">
            <span className="font-medium">Language:</span>
            <span className="font-semibold">{currentCountry.language}</span>
          </div>
        </div>
      </div>
      
      {/* Quick Market Info */}
      <div className="mt-4 text-xs text-gray-500 bg-blue-50 p-3 rounded-lg border border-blue-200">
        <p className="flex items-start">
          <span className="mr-2">💡</span>
          <span>Pricing, regulations, and available services are customized for your selected country.</span>
        </p>
      </div>
    </div>
  )
} 