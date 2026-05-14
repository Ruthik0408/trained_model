import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  server: {
    host: "127.0.0.1",
    port: 5173,
    open: true,
    strictPort: true,
  },

  build: {
    minify: "esbuild",
    sourcemap: false,
    cssCodeSplit: true,

    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
        },
      },
    },
  },

  esbuild: {
    drop: ["console", "debugger"],
  },
});
