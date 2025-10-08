'use client'

import React, { useState, useEffect } from 'react'
import ChatInterface from '../components/ChatInterface'
import CountrySelector from '../components/CountrySelector'
import Header from '../components/Header'

export default function HomePage() {
  const [selectedCountry, setSelectedCountry] = useState('LT') // Default to Lithuania
  const [vehicleInfo, setVehicleInfo] = useState({
    make: '',
    model: '',
    year: '',
    mileage: '',
    engine_type: '',
    transmission: ''
  })

  // Load saved preferences
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const savedCountry = localStorage.getItem('selectedCountry')
      const savedVehicle = localStorage.getItem('vehicleInfo')
      
      if (savedCountry) {
        setSelectedCountry(savedCountry)
      }
      
      if (savedVehicle) {
        try {
          setVehicleInfo(JSON.parse(savedVehicle))
        } catch (e) {
          console.error('Error parsing saved vehicle info:', e)
        }
      }
    }
  }, [])

  // Save preferences
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('selectedCountry', selectedCountry)
      localStorage.setItem('vehicleInfo', JSON.stringify(vehicleInfo))
    }
  }, [selectedCountry, vehicleInfo])

  const handleCountryChange = (newCountry) => {
    setSelectedCountry(newCountry)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          {/* Welcome Section */}
          <div className="text-center mb-10">
            <h1 className="text-5xl font-bold text-gray-900 mb-6">
              Car Troubleshoot Europe
            </h1>
            <div className="max-w-4xl mx-auto">
              <p className="text-2xl text-gray-700 leading-relaxed mb-8">
                AI-powered automotive diagnostic assistant for European markets
              </p>
              <p className="text-lg text-gray-600 leading-relaxed">
                Get expert diagnosis, repair estimates, and maintenance guidance tailored to your country's regulations and pricing
              </p>
            </div>
            <div className="flex justify-center mt-8 mb-6">
              <div className="bg-gradient-to-r from-blue-500 to-green-500 h-1 w-32 rounded-full"></div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            {/* Sidebar with Settings */}
            <div className="lg:col-span-1">
              <div className="space-y-6">
                {/* Country Selection */}
                <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    🌍 <span className="ml-2">Your Location</span>
                  </h3>
                  <CountrySelector
                    selectedCountry={selectedCountry}
                    onCountryChange={handleCountryChange}
                  />
                </div>

                {/* Vehicle Information */}
                <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    🚙 <span className="ml-2">Vehicle Information</span>
                  </h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Make
                      </label>
                      <select
                        value={vehicleInfo.make}
                        onChange={(e) => setVehicleInfo(prev => ({ ...prev, make: e.target.value }))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="">Select Make</option>
                        <option value="BMW">BMW</option>
                        <option value="Mercedes-Benz">Mercedes-Benz</option>
                        <option value="Audi">Audi</option>
                        <option value="Volkswagen">Volkswagen</option>
                        <option value="Volvo">Volvo</option>
                        <option value="Saab">Saab</option>
                        <option value="Peugeot">Peugeot</option>
                        <option value="Renault">Renault</option>
                        <option value="Fiat">Fiat</option>
                        <option value="Opel">Opel</option>
                        <option value="Škoda">Škoda</option>
                        <option value="SEAT">SEAT</option>
                        <option value="Citroën">Citroën</option>
                        <option value="Alfa Romeo">Alfa Romeo</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Model
                      </label>
                      <input
                        type="text"
                        value={vehicleInfo.model}
                        onChange={(e) => setVehicleInfo(prev => ({ ...prev, model: e.target.value }))}
                        placeholder="e.g., Golf, A4, 320i"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Year
                      </label>
                      <input
                        type="number"
                        min="1990"
                        max="2025"
                        value={vehicleInfo.year}
                        onChange={(e) => setVehicleInfo(prev => ({ ...prev, year: e.target.value }))}
                        placeholder="2020"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Mileage (km)
                      </label>
                      <input
                        type="number"
                        value={vehicleInfo.mileage}
                        onChange={(e) => setVehicleInfo(prev => ({ ...prev, mileage: e.target.value }))}
                        placeholder="150000"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                  </div>
                </div>

                {/* Quick Help */}
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
                  <h3 className="text-lg font-semibold text-blue-900 mb-3 flex items-center">
                    <span className="ml-2">Quick Help</span>
                  </h3>
                  <ul className="text-sm text-blue-800 space-y-2">
                    <li>• Describe your car's symptoms</li>
                    <li>• Ask for repair cost estimates</li>
                    <li>• Get maintenance schedules</li>
                    <li>• European market pricing</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Main Chat Area */}
            <div className="lg:col-span-3">
              <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-100">
                <ChatInterface
                  selectedCountry={selectedCountry}
                  vehicleInfo={vehicleInfo}
                />
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
} 