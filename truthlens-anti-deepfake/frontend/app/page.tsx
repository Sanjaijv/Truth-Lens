'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Upload from '@/components/Upload';
import Result from '@/components/Result';
import Loader from '@/components/Loader';
import ForensicShowcase from '@/components/ForensicShowcase';
import Background3D from '@/components/Background3D';
import { analyzeVideo, AnalysisResult } from '@/lib/api';
import {
  Sparkles, Zap, Brain, Shield
} from 'lucide-react';

type ScanType = 'quick' | 'deep' | 'forensic';

export default function HomePage() {
  const [selectedScanType, setSelectedScanType] = useState<ScanType>('quick');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const scanTypes = [
    { id: 'quick' as ScanType, name: 'Quick Scan', icon: Zap, description: 'Basic forensic check', duration: '~30s' },
    { id: 'deep' as ScanType, name: 'Deep Scan', icon: Brain, description: 'Sensor noise analysis', duration: '~2m' },
    { id: 'forensic' as ScanType, name: 'Forensic Scan', icon: Shield, description: 'Full physics breakdown', duration: '~5m' },
  ];

  const handleUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await analyzeVideo(file, selectedScanType);
      setResult(data);
    } catch (err) {
      setError('Analysis failed. Please ensure the backend is running.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <main className="min-h-screen bg-transparent text-white overflow-x-hidden relative selection:bg-indigo-500/30 selection:text-white">
      <Background3D />

      <AnimatePresence mode="wait">
        {result ? (
          <motion.div
            key="result"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="min-h-screen flex items-center justify-center py-20 px-6"
          >
            <Result
              filename={result.filename}
              scanType={result.scanType}
              aiLikelihood={result.aiLikelihood}
              physicsMarkers={result.physicsMarkers}
              onReset={handleReset}
            />
          </motion.div>
        ) : isLoading ? (
          <motion.div
            key="loader"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            className="min-h-screen flex items-center justify-center"
          >
            <Loader />
          </motion.div>
        ) : (
          <motion.div
            key="home"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="max-w-5xl w-full mx-auto px-6 min-h-screen flex flex-col items-center justify-start md:translate-x-60"
          >
            <section className="flex flex-col items-center text-center mb-0 w-full shrink-0">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                className="inline-flex items-center space-x-3 px-8 py-4 bg-white/5 backdrop-blur-xl text-indigo-400 rounded-full text-[11px] font-bold uppercase tracking-[0.3em] shadow-glow border border-white/10 mb-12"
              >
                <Sparkles size={16} className="animate-pulse" />
                <span>Forensic Physics Validation</span>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                className="w-full max-w-6xl"
              >
                <h1 className="text-7xl md:text-[7.5rem] font-black tracking-tighter leading-[0.8]">
                  Verify video
                  <br />
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 via-blue-600 to-indigo-500">
                    Authenticity
                  </span>
                </h1>
              </motion.div>
            </section>

            <div className="flex flex-col justify-center w-full max-w-6xl mx-auto py-0 -mt-20">
              <div className="w-full mb-16 relative z-10">
                <motion.h2
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  className="text-[10px] font-black text-slate-300 uppercase tracking-[0.4em] text-center mb-10"
                >
                  Select Analysis Depth
                </motion.h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-10">
                  {scanTypes.map((scan, index) => {
                    const Icon = scan.icon;
                    const isSelected = selectedScanType === scan.id;
                    return (
                      <motion.button
                        key={scan.id}
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5 + index * 0.1, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                        whileHover={{ y: -8 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setSelectedScanType(scan.id)}
                        className={`group relative p-10 md:p-12 rounded-[48px] border transition-all duration-700 text-left w-full h-full
                          ${isSelected
                            ? 'border-indigo-500 bg-white/10 backdrop-blur-2xl shadow-[0_60px_100px_-20px_rgba(79,70,229,0.3)]'
                            : 'border-white/5 bg-white/5 hover:bg-white/10 hover:border-white/20 backdrop-blur-md'}
                        `}
                      >
                        <div className="flex flex-col items-center text-center space-y-6 w-full">
                          <div className={`p-5 rounded-[24px] transition-all duration-700 ${isSelected ? 'bg-indigo-600 text-white shadow-lg scale-110' : 'bg-white shadow-sm text-slate-400 group-hover:text-indigo-400'}`}>
                            <Icon size={28} />
                          </div>
                          <div className="w-full">
                            <h3 className="text-xl font-black text-white mb-1 tracking-tight">{scan.name}</h3>
                            <p className="text-xs text-slate-400 font-medium leading-relaxed mb-6 px-4">{scan.description}</p>
                            <div className={`mx-auto flex items-center justify-center text-[10px] font-black uppercase tracking-[0.3em] px-4 py-2 rounded-full w-fit ${isSelected ? 'text-white bg-indigo-600/50' : 'text-slate-400 bg-white/5'}`}>
                              <Zap size={10} className="mr-2" /> {scan.duration}
                            </div>
                          </div>
                        </div>
                      </motion.button>
                    );
                  })}
                </div>
              </div>

              <motion.div
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                className="w-full relative z-20"
              >
                <div className="bg-slate-900/40 rounded-[64px] p-16 md:p-20 flex flex-col items-center border border-white/10 shadow-glow backdrop-blur-3xl">
                  <Upload onUpload={handleUpload} isLoading={isLoading} />
                </div>
              </motion.div>
            </div>

            <div className="w-full mt-20 relative z-10">
              <ForensicShowcase />
            </div>

            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="mt-12 w-full max-w-2xl mx-auto p-6 bg-red-50 border border-red-100 rounded-[40px] flex items-center justify-center space-x-5 text-red-600 shadow-sm"
              >
                <Shield size={20} className="animate-pulse" />
                <p className="font-bold text-sm tracking-tight">{error}</p>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
}