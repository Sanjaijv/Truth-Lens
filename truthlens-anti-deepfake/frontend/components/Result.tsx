'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, XCircle, Camera, Share2, Download, ArrowLeft } from 'lucide-react';
import Score from './Score';
import SignatureCube from './SignatureCube';

interface Marker {
    name: string;
    score: number;
    status: 'pass' | 'fail';
    description: string;
}

interface ResultProps {
    filename: string;
    scanType: string;
    aiLikelihood: number;
    physicsMarkers: Marker[];
    onReset: () => void;
}

const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: {
            staggerChildren: 0.1
        }
    }
};

const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
        opacity: 1,
        y: 0,
        transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] as [number, number, number, number] }
    }
};

const Result: React.FC<ResultProps> = ({ filename, scanType, aiLikelihood, physicsMarkers, onReset }) => {
    const isLikelyAI = aiLikelihood > 0.5;

    const handleShare = async () => {
        const shareText = `Analysis Report: ${filename}\nAI Likelihood: ${(aiLikelihood * 100).toFixed(0)}%\nStatus: ${aiLikelihood > 0.5 ? 'Suspicious' : 'Authentic'}`;
        if (navigator.share) {
            try {
                await navigator.share({
                    title: 'Detectify Analysis Report',
                    text: shareText,
                    url: window.location.href,
                });
            } catch (err) {
                console.error('Error sharing:', err);
            }
        } else {
            await navigator.clipboard.writeText(`${shareText}\nLink: ${window.location.href}`);
            alert('Report summary copied to clipboard!');
        }
    };

    const handleDownload = () => {
        const data = {
            filename,
            scanType,
            aiLikelihood,
            physicsMarkers,
            timestamp: new Date().toISOString()
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `analysis_report_${filename.split('.')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="w-full max-w-5xl mx-auto space-y-12">
            <div className="flex items-center justify-between pb-8 border-b border-slate-100/10">
                <div className="space-y-1">
                    <button
                        onClick={onReset}
                        className="flex items-center text-[10px] font-black uppercase tracking-[0.2em] text-slate-400 hover:text-indigo-600 transition-colors mb-4 group"
                    >
                        <ArrowLeft size={14} className="mr-2 transition-transform group-hover:-translate-x-1" />
                        Back to Analysis
                    </button>
                    <h2 className="text-4xl font-black tracking-tighter text-white">Analysis Report</h2>
                    <div className="flex items-center space-x-4 text-xs text-slate-400 font-bold font-mono">
                        <span className="flex items-center">
                            <Camera className="w-4 h-4 mr-2" />
                            {filename}
                        </span>
                        <span>•</span>
                        <span className="uppercase text-indigo-600 tracking-[0.2em]">{scanType}</span>
                    </div>
                </div>
                <div className="flex items-center space-x-6">
                    <button
                        onClick={handleShare}
                        className="p-5 text-slate-300 bg-white/5 backdrop-blur-xl rounded-[28px] border border-white/10 hover:text-indigo-400 transition-all hover:bg-white/10 active:scale-95 group relative shadow-premium"
                        title="Share Report"
                    >
                        <Share2 size={24} className="transition-transform group-hover:scale-110" />
                    </button>
                    <button
                        onClick={handleDownload}
                        className="p-5 text-slate-300 bg-white/5 backdrop-blur-xl rounded-[28px] border border-white/10 hover:text-indigo-400 transition-all hover:bg-white/10 active:scale-95 group relative shadow-premium"
                        title="Download JSON Report"
                    >
                        <Download size={24} className="transition-transform group-hover:scale-110" />
                    </button>
                </div>
            </div>

            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className={`p-12 md:p-14 rounded-[56px] border backdrop-blur-3xl transition-all flex flex-col md:flex-row items-center gap-14 shadow-premium
                    ${isLikelyAI ? 'bg-red-500/10 border-red-500/20 shadow-red-500/10' : 'bg-emerald-500/10 border-emerald-500/20 shadow-emerald-500/10'}
                `}
            >
                <div className="shrink-0">
                    <Score score={Math.round(aiLikelihood * 100)} label="Detection" type="likelihood" />
                </div>
                <div className="text-center md:text-left flex-1 space-y-6">
                    <h3 className={`text-4xl font-black tracking-tight ${isLikelyAI ? 'text-red-400' : 'text-emerald-400'}`}>
                        {isLikelyAI ? 'Potential AI Generation Detected' : 'Content Appears Authentic'}
                    </h3>
                    <p className="text-xl text-slate-400 leading-relaxed font-medium max-w-2xl">
                        {isLikelyAI
                            ? "Significant anomalies found in light consistency and Bayer noise patterns. Physics signatures do not match typical camera hardware profiles."
                            : "Physics signatures match hardware sensor profiles perfectly. The temporal consistency matches natural recording patterns."}
                    </p>
                </div>
            </motion.div>

            <motion.div
                variants={containerVariants}
                initial="hidden"
                animate="visible"
                className="space-y-8 pt-6"
            >
                <h4 className="text-2xl font-black ml-4 tracking-tight text-white">Forensic Physics Breakdown</h4>
                <div className="grid grid-cols-1 gap-4">
                    {physicsMarkers.map((marker, idx) => (
                        <motion.div
                            key={idx}
                            variants={itemVariants}
                            whileHover={{ x: 8 }}
                            className="bg-white/5 backdrop-blur-xl p-8 rounded-[36px] border border-white/10 flex items-center justify-between shadow-glow hover:border-indigo-500/50 transition-all"
                        >
                            <div className="flex items-center space-x-6">
                                <div className={`p-4 rounded-2xl ${marker.status === 'pass' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                                    {marker.status === 'pass' ? <CheckCircle2 size={24} /> : <XCircle size={24} />}
                                </div>
                                <div className="space-y-1">
                                    <h5 className="text-lg font-bold text-white leading-tight">{marker.name}</h5>
                                    <p className="text-sm text-slate-400 font-medium">{marker.description}</p>
                                </div>
                            </div>
                            <div className="text-right space-y-1">
                                <div className="text-2xl font-black text-white">{(marker.score * 100).toFixed(0)}%</div>
                                <div className="text-[10px] uppercase font-bold text-slate-500 tracking-[0.2em]">Confidence</div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </motion.div>

            <motion.div
                variants={itemVariants}
                className="space-y-4"
            >
                <h4 className="text-2xl font-black ml-4 tracking-tight text-white">Hardware Signature Cluster</h4>
                <SignatureCube isLikelyAI={isLikelyAI} />
            </motion.div>

            <button
                onClick={onReset}
                className="w-full py-8 text-slate-400 hover:text-indigo-600 font-bold uppercase text-[11px] tracking-[0.4em] transition-colors"
            >
                Analyze Another Video
            </button>
        </div>
    );
};

export default Result;