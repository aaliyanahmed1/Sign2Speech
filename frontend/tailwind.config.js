/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "#0A0A0F",
                primary: "#00E5CC",
                secondary: "#6C3FC8",
            },
            fontFamily: {
                syne: ["Syne", "sans-serif"],
                mono: ["IBM Plex Mono", "monospace"],
                sans: ["Lato", "sans-serif"],
            },
            backgroundImage: {
                'grid-pattern': "url('/grid.svg')"
            }
        },
    },
    plugins: [],
}
