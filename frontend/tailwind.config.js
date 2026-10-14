/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          900: '#0F172A',
          800: '#1E293B',
          700: '#334155',
        },
        civic: {
          green: '#16A34A',
          bg: '#F8FAFC',
          high: '#DC2626',
          medium: '#EAB308',
          low: '#16A34A',
        },
      },
    },
  },
  plugins: [],
};
