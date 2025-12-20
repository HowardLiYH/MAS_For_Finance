/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Dark theme inspired by trading terminals
        background: '#0a0e17',
        surface: '#111827',
        surfaceLight: '#1f2937',
        primary: '#22d3ee',  // Cyan
        secondary: '#a78bfa', // Purple
        success: '#10b981',  // Green
        danger: '#ef4444',   // Red
        warning: '#f59e0b',  // Amber
        muted: '#6b7280',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Outfit', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'flow': 'flow 2s ease-in-out infinite',
      },
      keyframes: {
        flow: {
          '0%, 100%': { transform: 'translateX(0)', opacity: 0.5 },
          '50%': { transform: 'translateX(10px)', opacity: 1 },
        },
      },
    },
  },
  plugins: [],
}
