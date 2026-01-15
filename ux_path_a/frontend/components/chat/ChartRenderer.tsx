'use client'

import React, { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-96 bg-gray-50 dark:bg-gray-800 rounded-lg"><div className="text-gray-500">Loading chart...</div></div>
})

export interface ChartData {
  type: 'line' | 'candlestick' | 'bar' | 'scatter' | 'area'
  data: {
    x?: (string | number)[]
    y?: (string | number)[]
    open?: (string | number)[]
    high?: (string | number)[]
    low?: (string | number)[]
    close?: (string | number)[]
    name?: string
    mode?: string
    line?: {
      color?: string
      width?: number
    }
    marker?: {
      color?: string
      size?: number
    }
    fill?: string
  }[]
  layout?: {
    title?: string
    xaxis?: {
      title?: string
      type?: string
      rangeslider?: { visible: boolean }
    }
    yaxis?: {
      title?: string
    }
    height?: number
    showlegend?: boolean
    hovermode?: string
    template?: string
  }
  config?: {
    displayModeBar?: boolean
    responsive?: boolean
  }
  maxHeight?: number  // Maximum height in pixels for vertical scrolling
}

interface ChartRendererProps {
  chartData: ChartData
  className?: string
}

export default function ChartRenderer({ chartData, className = '' }: ChartRendererProps) {

  // Default layout with dark mode support
  const defaultLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      color: '#374151', // gray-700
      family: 'Inter, system-ui, sans-serif',
    },
    xaxis: {
      gridcolor: '#E5E7EB', // gray-200
      ...chartData.layout?.xaxis,
    },
    yaxis: {
      gridcolor: '#E5E7EB', // gray-200
      ...chartData.layout?.yaxis,
    },
    ...chartData.layout,
  }

  // Dark mode layout
  const darkLayout = {
    ...defaultLayout,
    font: {
      color: '#D1D5DB', // gray-300
      family: 'Inter, system-ui, sans-serif',
    },
    xaxis: {
      ...defaultLayout.xaxis,
      gridcolor: '#374151', // gray-700
    },
    yaxis: {
      ...defaultLayout.yaxis,
      gridcolor: '#374151', // gray-700
    },
  }

  // Detect dark mode
  const [isDark, setIsDark] = React.useState(false)

  useEffect(() => {
    // Check for dark mode
    const checkDarkMode = () => {
      if (typeof window !== 'undefined') {
        const isDarkMode = document.documentElement.classList.contains('dark') ||
          window.matchMedia('(prefers-color-scheme: dark)').matches
        setIsDark(isDarkMode)
      }
    }

    checkDarkMode()

    // Watch for dark mode changes
    const observer = new MutationObserver(checkDarkMode)
    if (document.documentElement) {
      observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['class'],
      })
    }

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    mediaQuery.addEventListener('change', checkDarkMode)

    return () => {
      observer.disconnect()
      mediaQuery.removeEventListener('change', checkDarkMode)
    }
  }, [])

  const layout = isDark ? darkLayout : defaultLayout

  // Default config
  const config = {
    displayModeBar: true,
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    ...chartData.config,
  }

  // Process data for different chart types
  const processedData = chartData.data.map((trace) => {
    if (chartData.type === 'candlestick') {
      return {
        type: 'candlestick',
        x: trace.x,
        open: trace.open,
        high: trace.high,
        low: trace.low,
        close: trace.close,
        name: trace.name || 'Price',
        increasing: { line: { color: '#10B981' } }, // green-500
        decreasing: { line: { color: '#EF4444' } }, // red-500
      }
    }

    if (chartData.type === 'area') {
      return {
        type: 'scatter',
        x: trace.x,
        y: trace.y,
        mode: 'lines',
        name: trace.name,
        fill: 'tozeroy',
        line: trace.line || { color: '#3B82F6', width: 2 }, // blue-500
        ...trace,
      }
    }

    return {
      type: chartData.type || 'scatter',
      x: trace.x,
      y: trace.y,
      mode: trace.mode || (chartData.type === 'line' ? 'lines' : 'markers'),
      name: trace.name,
      line: trace.line,
      marker: trace.marker,
      ...trace,
    }
  })

  const maxHeight = chartData.maxHeight || 600  // Default max height for vertical scrolling
  const chartHeight = chartData.layout?.height || 400  // Use layout height or default 400px

  return (
    <div className={`my-4 ${className}`}>
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <div 
          className="overflow-x-auto overflow-y-auto"
          style={{ maxHeight: `${maxHeight}px` }}
        >
          <div style={{ minWidth: '600px' }}>
            <Plot
              data={processedData as any}
              layout={layout as any}
              config={config as any}
              style={{ width: '100%', minWidth: '600px', height: `${chartHeight}px` }}
              useResizeHandler={true}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
