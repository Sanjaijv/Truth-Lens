import { defineConfig } from 'vite';
import path from 'path';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [react()],
    base: './', // Use relative paths for extension compatibility
    build: {
        minify: false, // Easier debugging
        rollupOptions: {
            input: {
                content: path.resolve(__dirname, 'content.js'),
                popup: path.resolve(__dirname, 'popup.html')
            },
            output: {
                entryFileNames: '[name].js',
                chunkFileNames: '[name].js',
                assetFileNames: '[name].[ext]',
                // format: 'iife', // REMOVED: Cannot use IIFE with code splitting (React)
                // name: 'TruthLensContent' // REMOVED
            }
        },
        outDir: 'dist',
        emptyOutDir: true
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './')
        }
    }
});
