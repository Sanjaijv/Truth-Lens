import { defineConfig } from 'vite';
import path from 'path';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [react()],
    base: './',
    build: {
        minify: false,
        emptyOutDir: true, // Clear dist on first build
        rollupOptions: {
            input: {
                popup: path.resolve(__dirname, 'popup.html'),
                frame: path.resolve(__dirname, 'frame.html')
            },
            output: {
                entryFileNames: '[name].js',
                chunkFileNames: '[name].js',
                assetFileNames: '[name].[ext]'
            }
        },
        outDir: 'dist'
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './')
        }
    }
});
