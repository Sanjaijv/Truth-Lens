'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Image from 'next/image';
import { Scan, ShieldCheck, Microscope, Layers } from 'lucide-react';

const features = [
    {
        title: "Optical Flow Analysis",
        description: "Scanning pixel vectors to detect unnatural movement patterns that betray AI temporal stitching.",
        icon: Scan,
        image: "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=800",
    },
    {
        title: "Sensor Noise Fingerprinting",
        description: "Every camera sensor has a unique noise 'DNA'. We detect if a video lacks this authentic hardware signature.",
        icon: Microscope,
        image: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=800",
    },
    {
        title: "Temporal Consistency",
        description: "AI often struggles with physics consistency over time. Our engine validates light bounce and shadow logic.",
        icon: Layers,
        image: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80&w=800",
    }
];

export default function ForensicShowcase() {
    return (
        <section className="w-full max-w-7xl mx-auto py-32 px-6">
            <div className="flex flex-col items-center text-center mb-20">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="flex items-center space-x-2 text-[10px] font-black uppercase tracking-[0.4em] text-indigo-600 mb-4 bg-indigo-50 px-4 py-2 rounded-full"
                >
                    <ShieldCheck size={16} />
                    <span>Lab-Grade Technology</span>
                </motion.div>
                <h2 className="text-5xl md:text-7xl font-black tracking-tighter text-white mb-6">
                    How the Scanner <br /> Sees the World
                </h2>
                <p className="max-w-2xl text-lg text-slate-400 font-medium leading-relaxed">
                    Our multi-layer forensic engine doesn&apos;t just look for &quot;AI signs&quot;&mdash;it validates the fundamental physics of light and motion.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                {features.map((feature, index) => {
                    const Icon = feature.icon;
                    return (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 40 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.2 }}
                            className="group relative"
                        >
                            <div className="relative aspect-video w-full rounded-[40px] overflow-hidden bg-white/5 backdrop-blur-xl shadow-premium border border-white/10 transition-all duration-700 group-hover:shadow-indigo-500/30">
                                <Image
                                    src={feature.image}
                                    alt={feature.title}
                                    fill
                                    className="object-cover transition-transform duration-1000 group-hover:scale-110 opacity-80"
                                />
                                <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent opacity-60" />
                                <div className="absolute bottom-8 left-8 right-8">
                                    <div className="flex items-center space-x-4">
                                        <div className="p-3 bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 text-white">
                                            <Icon size={24} />
                                        </div>
                                        <h3 className="text-xl font-bold text-white tracking-tight">
                                            {feature.title}
                                        </h3>
                                    </div>
                                </div>
                            </div>
                            <div className="mt-8 px-4">
                                <p className="text-slate-400 font-medium leading-relaxed">
                                    {feature.description}
                                </p>
                            </div>
                        </motion.div>
                    );
                })}
            </div>
        </section>
    );
}