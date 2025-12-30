import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
    build: {
        minify: false,
        emptyOutDir: false, // Do NOT clear dist (keep popup)
        rollupOptions: {
            input: {
                content: path.resolve(__dirname, 'content.js')
            },
            output: {
                entryFileNames: '[name].js',
                assetFileNames: '[name].[ext]',
                format: 'es', // Use ES modules
                name: 'TruthLensContent'
            },
        },
        outDir: 'dist'
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './')
        }
    }
});
