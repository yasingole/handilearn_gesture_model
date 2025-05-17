import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    open: true
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
    // Optimize for M1 Mac
    target: 'esnext',
    minify: 'esbuild'
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      '@tensorflow/tfjs',
      '@mediapipe/hands'
    ]
  }
});
