'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Zap, Database, Activity, Shield } from 'lucide-react';

const loadingStates = [
    { text: "Validating Light Consistency Patterns", icon: Zap },
    { text: "Analyzing Sensor Noise (Bayer Pattern)", icon: Cpu },
    { text: "Checking Temporal Frame Stability", icon: Activity },
    { text: "Cross-referencing Hardware Profiles", icon: Database },
    { text: "Final Physics Signature Audit", icon: Shield },
];

const Loader = () => {
    const [stateIndex, setStateIndex] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setStateIndex((prev) => (prev + 1) % loadingStates.length);
        }, 2500);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="flex flex-col items-center justify-center space-y-12 py-20">
            <div className="relative">
                <motion.div
                    animate={{
                        scale: [1, 1.1, 1],
                        opacity: [0.3, 0.6, 0.3]
                    }}
                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                    className="absolute inset-0 bg-indigo-500/20 blur-3xl rounded-full"
                />

                <motion.div
                    initial={{ rotate: -10 }}
                    animate={{ rotate: 0 }}
                    className="relative bg-white p-12 rounded-[40px] border border-indigo-100 shadow-2xl shadow-indigo-100/50"
                >
                    <Cpu size={72} className="text-indigo-600" />

                    {[...Array(4)].map((_, i) => (
                        <motion.div
                            key={i}
                            animate={{
                                x: [0, (i % 2 === 0 ? 40 : -40)],
                                y: [0, (i < 2 ? 40 : -40)],
                                opacity: [0, 1, 0]
                            }}
                            transition={{
                                duration: 2,
                                repeat: Infinity,
                                delay: i * 0.5,
                                ease: "circOut"
                            }}
                            className="absolute top-1/2 left-1/2 w-1.5 h-1.5 bg-indigo-400 rounded-full"
                        />
                    ))}
                </motion.div>

                <motion.div
                    animate={{ y: [0, -5, 0] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="absolute -top-4 -right-4 bg-slate-950 text-white p-3 rounded-2xl shadow-xl"
                >
                    <Zap size={18} />
                </motion.div>
            </div>

            <div className="text-center space-y-4 min-h-[80px]">
                <h3 className="text-3xl font-black text-slate-950 tracking-tighter">
                    Running Forensic Scan
                </h3>

                <AnimatePresence mode="wait">
                    <motion.p
                        key={stateIndex}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="text-slate-400 font-bold uppercase text-[10px] tracking-[0.4em] flex items-center justify-center gap-3"
                    >
                        <span className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_10px_rgba(79,70,229,0.5)]" />
                        {loadingStates[stateIndex].text}
                    </motion.p>
                </AnimatePresence>
            </div>

            <div className="w-72 h-2 bg-slate-100 rounded-full overflow-hidden relative border border-slate-50">
                <motion.div
                    initial={{ x: "-100%" }}
                    animate={{ x: "100%" }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    className="absolute inset-0 w-full h-full bg-linear-to-r from-transparent via-indigo-500 to-transparent"
                />
            </div>
        </div>
    );
};

export default Loader;
