import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  server: {
    host: process.env.VITE_DEV_HOST || "0.0.0.0",
    port: Number(process.env.VITE_DEV_PORT || 5173),
    open: false,
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
