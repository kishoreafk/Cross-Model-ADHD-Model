import React, { useState } from "react";
import { Brain3D } from "./components/Brain3D";
import { ModalityForm } from "./components/ModalityForm";
import { ExplainabilityLayer } from "./components/ExplainabilityLayer";
import { Activity, Brain, ClipboardList, Zap, ShieldCheck, Database, Cpu } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

const LoadingStep = ({ text, delay }: { text: string; delay: number }) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.5 }}
    className="flex items-center gap-3 text-[10px] font-mono uppercase tracking-widest text-white/40"
  >
    <div className="w-1 h-1 bg-white/20 rounded-full" />
    <span>{text}</span>
  </motion.div>
);

export default function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handlePredict = async (data: any) => {
    setLoading(true);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const res = await response.json();
      setResult(res);
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white selection:bg-white selection:text-black font-sans">
      {/* Background Grid Effect */}
      <div className="fixed inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 border-b border-white/5 backdrop-blur-md sticky top-0">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-white rounded-xl flex items-center justify-center">
              <Brain size={24} className="text-black" />
            </div>
            <div>
              <h1 className="text-sm font-bold uppercase tracking-widest">ADHD Multimodal AI</h1>
              <p className="text-[10px] font-mono text-white/40 uppercase tracking-wider">Explainable Inference Dashboard v2.5</p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="hidden md:flex items-center gap-4 text-[10px] font-mono uppercase tracking-widest text-white/30">
              <div className="flex items-center gap-2">
                <Database size={12} />
                <span>HYPERAKTIV | ADHD-200</span>
              </div>
              <div className="w-px h-4 bg-white/10" />
              <div className="flex items-center gap-2">
                <Cpu size={12} />
                <span>MoE Fusion Core</span>
              </div>
            </div>
            <div className="px-4 py-2 rounded-full bg-white/5 border border-white/10 flex items-center gap-2">
              <ShieldCheck size={14} className="text-green-400" />
              <span className="text-[10px] font-mono uppercase tracking-widest text-white/70">Secure Node</span>
            </div>
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
          {/* Left Column: Input & 3D Brain */}
          <div className="lg:col-span-5 space-y-8">
            <section>
              <h2 className="text-xs font-mono uppercase tracking-widest text-white/50 mb-6 flex items-center gap-2">
                <div className="w-1 h-1 bg-white rounded-full" />
                Input Parameters
              </h2>
              <ModalityForm onPredict={handlePredict} loading={loading} />
            </section>

            <section>
              <h2 className="text-xs font-mono uppercase tracking-widest text-white/50 mb-6 flex items-center gap-2">
                <div className="w-1 h-1 bg-white rounded-full" />
                Neural Visualization
              </h2>
              <Brain3D 
                isADHD={result?.isADHD ?? false} 
                confidence={result?.confidence ?? 0.5} 
                shap={result?.shap}
              />
            </section>
          </div>

          {/* Right Column: Explainability & Results */}
          <div className="lg:col-span-7">
            <AnimatePresence mode="wait">
              {loading ? (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full min-h-[600px] flex flex-col items-center justify-center text-center p-12 rounded-3xl border border-white/10 bg-white/[0.02] relative overflow-hidden"
                >
                  {/* Progress Ring Animation */}
                  <div className="relative w-40 h-40 mb-12 group">
                    {/* Outer Glow */}
                    <div className="absolute inset-0 bg-white/5 rounded-full blur-2xl animate-pulse" />
                    
                    <svg className="w-full h-full -rotate-90 relative z-10">
                      <circle
                        cx="80"
                        cy="80"
                        r="76"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1"
                        className="text-white/5"
                      />
                      <motion.circle
                        cx="80"
                        cy="80"
                        r="76"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeDasharray="477"
                        initial={{ strokeDashoffset: 477 }}
                        animate={{ strokeDashoffset: 0 }}
                        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                        className="text-white/40"
                        strokeLinecap="round"
                      />
                      
                      {/* Secondary Orbiting Ring */}
                      <motion.circle
                        cx="80"
                        cy="80"
                        r="60"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1"
                        strokeDasharray="10 30"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                        className="text-white/10"
                      />
                    </svg>
                    
                    <div className="absolute inset-0 flex items-center justify-center z-20">
                      <div className="relative">
                        <motion.div
                          animate={{ 
                            scale: [1, 1.1, 1],
                            opacity: [0.2, 0.5, 0.2]
                          }}
                          transition={{ duration: 2, repeat: Infinity }}
                          className="absolute inset-0 bg-white/20 blur-xl rounded-full"
                        />
                        <Brain size={48} className="text-white/40 relative z-10" />
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4 max-w-sm">
                    <h2 className="text-xl font-medium text-white tracking-tight">Inference in Progress</h2>
                    <div className="flex flex-col gap-2">
                      <LoadingStep text="Synchronizing Multimodal Streams" delay={0} />
                      <LoadingStep text="Extracting EEG Spectral Features" delay={0.5} />
                      <LoadingStep text="Gating Mixture of Experts" delay={1.0} />
                      <LoadingStep text="Calculating SHAP Importance" delay={1.5} />
                    </div>
                  </div>

                  {/* Scanning Line Effect */}
                  <motion.div 
                    className="absolute inset-x-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent"
                    animate={{ top: ["0%", "100%", "0%"] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                  />
                </motion.div>
              ) : result ? (
                <ExplainabilityLayer key="results" result={result} />
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full min-h-[600px] flex flex-col items-center justify-center text-center p-12 rounded-3xl border border-dashed border-white/10 bg-white/[0.02]"
                >
                  <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-8 animate-pulse">
                    <Zap size={32} className="text-white/20" />
                  </div>
                  <h2 className="text-xl font-medium text-white/60 mb-4 tracking-tight">Awaiting Multimodal Input</h2>
                  <p className="text-sm text-white/30 max-w-md leading-relaxed font-mono uppercase tracking-wider">
                    Configure Clinical, Actigraphy, and EEG parameters on the left to initiate the Mixture of Experts inference engine.
                  </p>
                  
                  <div className="grid grid-cols-2 gap-4 mt-12 w-full max-w-sm">
                    {['Clinical', 'HRV', 'EEG', 'Brain FC'].map(m => (
                      <div key={m} className="px-4 py-3 rounded-xl border border-white/5 bg-white/5 flex items-center gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-white/20" />
                        <span className="text-[10px] font-mono uppercase tracking-widest text-white/40">{m}</span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 py-12 mt-12">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex items-center gap-8">
            <div className="flex flex-col">
              <span className="text-[10px] font-mono text-white/30 uppercase tracking-widest mb-1">Architecture</span>
              <span className="text-xs font-medium text-white/70">MoE Fusion Network</span>
            </div>
            <div className="w-px h-8 bg-white/10" />
            <div className="flex flex-col">
              <span className="text-[10px] font-mono text-white/30 uppercase tracking-widest mb-1">Explainability</span>
              <span className="text-xs font-medium text-white/70">SHAP + DSM-5 Mapping</span>
            </div>
          </div>
          <div className="text-right">
            <p className="text-[10px] font-mono text-white/20 uppercase tracking-[0.2em]">
              Research-Grade Diagnostic Support System
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
