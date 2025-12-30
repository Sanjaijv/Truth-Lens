import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
    build: {
        minify: false,
        emptyOutDir: false,
        outDir: 'dist',
        rollupOptions: {
            input: {
                'analysis.worker': path.resolve(__dirname, 'analysis.worker.js')
            },
            output: {
                entryFileNames: '[name].js',
                format: 'iife', // IIFE for classic worker support
                inlineDynamicImports: true // FORCE everything into one file
            }
        }
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './')
        }
    }
});
