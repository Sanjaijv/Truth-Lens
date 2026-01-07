'use client';
import React from 'react';

interface ScoreProps {
    score: number;
    label: string;
    type: 'physics' | 'likelihood';
}

const Score: React.FC<ScoreProps> = ({ score, label, type }) => {
    const isLikelihood = type === 'likelihood';
    const circumference = 2 * Math.PI * 45;
    const offset = circumference - (score / 100) * circumference;

    return (
        <div className="flex flex-col items-center space-y-6 p-10 rounded-[40px] bg-white shadow-2xl border border-gray-100 transition-all duration-500 hover:shadow-[0_40px_80px_-15px_rgba(0,0,0,0.1)] hover:-translate-y-2 group">
            <div className="relative w-48 h-48">
                <svg className="w-full h-full transform -rotate-90">
                    <circle cx="96" cy="96" r="45" stroke="currentColor" strokeWidth="8" fill="transparent" className="text-gray-50" />
                    <circle
                        cx="96" cy="96" r="45"
                        stroke="url(#scoreGradient)"
                        strokeWidth="12"
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                        strokeLinecap="round"
                        fill="transparent"
                        className="transition-all duration-1000 ease-out"
                    />
                    <defs>
                        <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" style={{ stopColor: isLikelihood && score > 50 ? '#ef4444' : '#6366f1' }} />
                            <stop offset="100%" style={{ stopColor: isLikelihood && score > 50 ? '#f97316' : '#4f46e5' }} />
                        </linearGradient>
                    </defs>
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-5xl font-black text-gray-900 tracking-tighter">{score}%</span>
                    <span className="text-[11px] uppercase tracking-[0.3em] font-bold text-gray-400 mt-1">Score</span>
                </div>
            </div>
            <div className="text-center">
                <h4 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-1">{label}</h4>
                <p className={`text-xl font-black ${isLikelihood ? (score > 50 ? 'text-red-500' : 'text-emerald-500') : 'text-indigo-600'}`}>
                    {isLikelihood ? (score > 50 ? 'Suspicious' : 'Authentic') : 'Analyzing...'}
                </p>
            </div>
        </div>
    );
};
export default Score;