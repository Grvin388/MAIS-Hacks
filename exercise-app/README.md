# Exercise App (Vite + React)

Minimal wrapper to run your `ExerciseFormCorrector` component.

Prerequisites

- Node.js (>=16) and npm

Setup

```powershell
cd "c:\Users\chock\OneDrive\Desktop\MAIS hacks\exercise-app"
npm install
```

Run dev server

```powershell
npm run dev
```

Open the URL printed by Vite (usually http://localhost:5173).

Notes

- The component posts to `http://localhost:5000/api/analyze`. Ensure your backend is running and CORS is configured, or add a proxy in `vite.config.js`.
- To get Tailwind styling, install and configure Tailwind in this project (optional).
