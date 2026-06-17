import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Tauri convention: dev server on 1420, HMR over a fixed port, strict so
// the wrapper sees a hard failure (not a port walk) if the slot is taken.
// `TAURI_DEV_HOST` is set by `tauri dev` on mobile; harmless on desktop.
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  // Tauri rebuilds whenever a file changes; tell Vite to ignore src-tauri so
  // the frontend isn't churned by Rust rebuilds.
  clearScreen: false,
  server: {
    host: host || false,
    port: 1420,
    strictPort: true,
    hmr: host
      ? { protocol: 'ws', host, port: 1421 }
      : undefined,
    watch: {
      ignored: ['**/src-tauri/**'],
    },
    proxy: {
      '/ws': {
        target: 'ws://localhost:3000',
        ws: true,
      },
      '/health': 'http://localhost:3000',
    },
  },
});
