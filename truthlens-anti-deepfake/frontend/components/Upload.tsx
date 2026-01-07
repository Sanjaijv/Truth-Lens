'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload as UploadIcon, FileVideo, Shield, CheckCircle } from 'lucide-react';

interface UploadProps {
    onUpload: (file: File) => void;
    isLoading: boolean;
    onInteraction?: () => void;
}

const diagnosticMessages = [
    "Initializing Forensic Engine...",
    "Scanning Light Intensity...",
    "Analyzing Pixel Vectors...",
    "Detecting Motion Anomalies...",
    "Validating Sensor Noise...",
    "Finalizing Score..."
];

const Upload: React.FC<UploadProps> = ({ onUpload, isLoading, onInteraction }) => {
    const [file, setFile] = useState<File | null>(null);
    const [progress, setProgress] = useState(0);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles.length > 0) {
            setFile(acceptedFiles[0]);
            onInteraction?.();
        }
    }, [onInteraction]);

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isLoading) {
            interval = setInterval(() => {
                setProgress(prev => {
                    if (prev >= 100) return 100;
                    const next = prev + (100 / (diagnosticMessages.length * 10));
                    return next;
                });
            }, 100);
        }
        return () => clearInterval(interval);
    }, [isLoading]);

    const diagnosticIndex = Math.min(
        Math.floor((progress / 100) * diagnosticMessages.length),
        diagnosticMessages.length - 1
    );

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'video/*': ['.mp4', '.avi', '.mov', '.mkv'] },
        multiple: false,
        disabled: isLoading,
    });

    return (
        <div className="w-full max-w-4xl mx-auto">
            <div {...getRootProps()} className="relative group">
                <input {...getInputProps()} />
                <motion.div
                    whileHover={!isLoading ? { scale: 1.01 } : {}}
                    whileTap={!isLoading ? { scale: 0.99 } : {}}
                    className={`relative cursor-pointer transition-all duration-500 rounded-[48px] p-10 md:p-16 flex flex-col items-center justify-center text-center overflow-hidden border-2
                    ${isDragActive
                            ? 'bg-indigo-500/20 border-indigo-400 border-dashed'
                            : file
                                ? 'bg-white/10 border-indigo-500 shadow-glow'
                                : 'bg-white/5 border-white/10 hover:bg-white/10 backdrop-blur-xl'}
                    ${isLoading ? 'opacity-80 cursor-not-allowed' : ''}
                    `}
                >
                    <motion.div
                        initial={false}
                        animate={{
                            scale: file ? 1.1 : 1,
                            rotate: isDragActive ? 10 : 0
                        }}
                        className={`mb-10 p-8 rounded-[32px] shadow-lg transition-colors
                            ${file ? 'bg-indigo-600 text-white' : 'bg-white text-indigo-600 shadow-indigo-100'}
                        `}
                    >
                        {file ? <FileVideo size={40} /> : <UploadIcon size={40} />}
                    </motion.div>

                    <div className="space-y-4 max-w-lg">
                        <h3 className="text-3xl md:text-4xl font-black text-white tracking-tighter">
                            {file ? file.name : 'Intake Video Data'}
                        </h3>
                        <p className="text-lg text-slate-400 font-medium tracking-tight">
                            {file
                                ? `${(file.size / (1024 * 1024)).toFixed(2)} MB • Ready for Physics Validation`
                                : "HEVC, H.264, or ProRes (max 500MB)"
                            }
                        </p>
                    </div>

                    {!file && (
                        <div className="mt-12 flex items-center space-x-3 text-[11px] font-black uppercase tracking-[0.3em] text-slate-300">
                            <Shield size={16} />
                            <span>Secure Forensic-Grade Pipeline</span>
                        </div>
                    )}

                    {file && !isLoading && (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="mt-8 flex items-center space-x-3 text-indigo-400 font-bold text-base"
                        >
                            <CheckCircle size={24} />
                            <span>File successfully staged</span>
                        </motion.div>
                    )}
                </motion.div>
            </div>

            <AnimatePresence>
                {file && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                        className="mt-12"
                    >
                        <button
                            onClick={() => {
                                if (!isLoading) {
                                    setProgress(0);
                                    onUpload(file);
                                }
                            }}
                            className={`relative w-full py-7 px-16 rounded-[40px] font-black text-xl uppercase tracking-[0.4em] transition-all duration-500 overflow-hidden shadow-2xl
                                ${isLoading
                                    ? 'bg-slate-100 text-slate-900 cursor-not-allowed shadow-none'
                                    : 'bg-slate-950 text-white hover:bg-indigo-600 active:scale-95 shadow-slate-200'}
                            `}
                        >
                            {isLoading && (
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${progress}%` }}
                                    className="absolute inset-0 bg-indigo-600 opacity-10"
                                />
                            )}

                            <div className="relative z-10 flex items-center justify-center space-x-4">
                                {isLoading ? (
                                    <>
                                        <div className="flex flex-col items-center">
                                            <span className="text-indigo-600 text-[10px] tracking-[0.6em] mb-1 font-black">
                                                {Math.round(progress)}% COMPLETE
                                            </span>
                                            <span className="text-slate-950">
                                                {diagnosticMessages[diagnosticIndex]}
                                            </span>
                                        </div>
                                    </>
                                ) : (
                                    <span>Initialize Forensic Scan</span>
                                )}
                            </div>
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default Upload;
