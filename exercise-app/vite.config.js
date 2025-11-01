import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // proxy can be added here if you want to forward /api to your backend
    // proxy: { '/api': 'http://localhost:5000' }
  },
});
