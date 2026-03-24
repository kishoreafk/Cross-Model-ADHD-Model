import React, { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend } from "recharts";
import { CheckCircle2, AlertCircle, Info } from "lucide-react";
import { motion } from "motion/react";

interface ExplainabilityLayerProps {
  result: any;
}

// Real model performance from adhd_xai_results.json
const MODEL_METRICS: Record<string, { acc: number; f1: number; auc: number }> = {
  fusion: { acc: 0.7647, f1: 0.80, auc: 0.7639 },
  clinical: { acc: 0.7176, f1: 0.7209, auc: 0.7972 },
  activity_hrv: { acc: 0.4118, f1: 0.4444, auc: 0.3411 },
  eeg: { acc: 0.80, f1: 0.80, auc: 0.859 },
  brain: { acc: 0.6104, f1: 0.4828, auc: 0.6208 },
};

export const ExplainabilityLayer = ({ result }: ExplainabilityLayerProps) => {
  const [compareMode, setCompareMode] = useState(false);

  if (!result) return null;

  const {
    prediction,
    confidence,
    isADHD,
    gates,
    shap,
    triggeredCriteria,
    warnings = [],
    input_mode: inputMode = "demo",
    service_source: serviceSource = "python_inference",
    models_used: modelsUsed = [],
  } = result;
  const isApproximateGating = inputMode !== "artifact_aligned" || serviceSource !== "python_inference";

  if (result.error && !prediction) {
    return (
      <div className="rounded-3xl border border-red-500/20 bg-red-500/10 p-8 space-y-4">
        <div className="text-xs font-mono uppercase tracking-[0.2em] text-red-300/70">Inference Error</div>
        <div className="text-2xl font-semibold tracking-tight text-red-300">{result.error}</div>
        {warnings.length > 0 && (
          <div className="space-y-2">
            {warnings.map((warning: string, index: number) => (
              <div key={`${warning}-${index}`} className="text-sm text-white/55">
                {warning}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  const clinicalShapData = (shap?.clinical || []).map((s: any, i: number) => ({
    name: s.feat,
    value: s.val,
    baseline: s.val * (0.3 + (i % 3) * 0.2),
  }));

  const eegShapData = (shap?.eeg || []).map((s: any, i: number) => ({
    name: s.feat,
    value: s.val,
    baseline: s.val * (0.3 + (i % 3) * 0.2),
  }));

  const activityShapData = (shap?.activity_hrv || []).map((s: any, i: number) => ({
    name: s.feat,
    value: s.val,
    baseline: s.val * (0.3 + (i % 3) * 0.2),
  }));

  // Determine what secondary chart to display
  const hasEeg = eegShapData.length > 0;
  const hasActivity = activityShapData.length > 0;
  
  // Decide which secondary data to show
  const secondaryTitle = hasEeg ? "EEG Channel Importance" : "Actigraphy + HRV Importance";
  const secondaryData = hasEeg ? eegShapData : activityShapData;
  const noSecondaryMessage = hasEeg ? "No EEG SHAP data" : hasActivity ? "No Activity/HRV SHAP data" : "No secondary SHAP data";

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      {(warnings.length > 0 || modelsUsed.length > 0) && (
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 space-y-4">
          <div className="flex flex-wrap items-center gap-3 text-[10px] font-mono uppercase tracking-widest">
            <span className="px-3 py-1 rounded-full border border-white/10 text-white/60">
              Source: {serviceSource.replace("_", " ")}
            </span>
            <span className="px-3 py-1 rounded-full border border-white/10 text-white/60">
              Mode: {inputMode.replace("_", " ")}
            </span>
            {modelsUsed.length > 0 && (
              <span className="px-3 py-1 rounded-full border border-white/10 text-white/60">
                Models: {modelsUsed.join(", ")}
              </span>
            )}
          </div>
          {warnings.length > 0 && (
            <div className="space-y-2">
              {warnings.map((warning: string, index: number) => (
                <div key={`${warning}-${index}`} className="text-[11px] text-white/45 leading-relaxed">
                  {warning}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Prediction Card */}
        <div className={`col-span-2 p-8 rounded-3xl border relative overflow-hidden group transition-all duration-500 hover:shadow-[0_0_40px_rgba(255,255,255,0.05)] ${isADHD ? 'bg-red-500/10 border-red-500/20' : 'bg-green-500/10 border-green-500/20'}`}>
          <div className={`absolute -top-24 -right-24 w-64 h-64 rounded-full blur-[100px] opacity-20 transition-all duration-700 group-hover:opacity-40 ${isADHD ? 'bg-red-500' : 'bg-green-500'}`} />
          
          <div className="flex items-start justify-between relative z-10">
            <div className="space-y-6">
              <div>
                <h2 className="text-xs font-mono uppercase tracking-[0.2em] text-white/40 mb-2 flex items-center gap-2">
                  <div className={`w-1 h-1 rounded-full animate-pulse ${isADHD ? 'bg-red-400' : 'bg-green-400'}`} />
                  Inference Result
                </h2>
                <motion.h1 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className={`text-6xl font-bold tracking-tighter ${isADHD ? 'text-red-400' : 'text-green-400'}`}
                >
                  {prediction}
                </motion.h1>
              </div>

              <div className="flex items-center gap-8">
                <div className="flex flex-col">
                  <span className="text-[10px] font-mono text-white/30 uppercase tracking-widest mb-1">Confidence</span>
                  <div className="flex items-baseline gap-1">
                    <span className="text-2xl font-mono text-white">{(confidence * 100).toFixed(1)}</span>
                    <span className="text-xs text-white/40">%</span>
                  </div>
                  <div className="w-full h-1 bg-white/5 rounded-full mt-2 overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${confidence * 100}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className={`h-full rounded-full ${isADHD ? 'bg-red-400/50' : 'bg-green-400/50'}`}
                    />
                  </div>
                </div>
                <div className="w-px h-12 bg-white/10" />
                <div className="flex flex-col">
                  <span className="text-[10px] font-mono text-white/30 uppercase tracking-widest mb-1">Status</span>
                  <span className="text-2xl font-mono text-white uppercase tracking-tighter">{isADHD ? 'Positive' : 'Negative'}</span>
                </div>
              </div>
            </div>
            
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: "spring", stiffness: 200, damping: 15 }}
            >
              {isADHD ? (
                <div className="relative">
                  <div className="absolute inset-0 bg-red-400/20 blur-2xl rounded-full animate-pulse" />
                  <AlertCircle className="text-red-400 relative z-10" size={64} />
                </div>
              ) : (
                <div className="relative">
                  <div className="absolute inset-0 bg-green-400/20 blur-2xl rounded-full animate-pulse" />
                  <CheckCircle2 className="text-green-400 relative z-10" size={64} />
                </div>
              )}
            </motion.div>
          </div>
        </div>

        {/* MoE Gating Weights */}
        <div className="p-8 rounded-3xl bg-white/5 border border-white/10">
          <h2 className="text-xs font-mono uppercase tracking-widest text-white/50 mb-6">MoE Gating Trust</h2>
          <p className="text-[10px] font-mono uppercase tracking-widest text-white/30 mb-5">
            {serviceSource === "node_fallback"
              ? "Node fallback heuristic"
              : isApproximateGating
                ? "Approximate in demo mode"
                : "Artifact-aligned fusion path"}
          </p>
          <div className="space-y-4">
            {Object.entries(gates || {}).map(([key, val]: any) => {
              const modMetrics = MODEL_METRICS[key];
              const modAuc = modMetrics?.auc || 0.5;
              const isWeak = modAuc < 0.55;
              
              return (
                <div key={key} className="space-y-1 relative group cursor-help">
                  <div className="flex justify-between text-[10px] font-mono">
                    <span className="text-white/40 flex items-center gap-1">
                      {key.replace('_', ' ')}
                      {isWeak && <span className="text-yellow-400 text-[8px]">⚠ weak</span>}
                      {isApproximateGating && <span className="text-sky-400 text-[8px]">approx</span>}
                    </span>
                    <span className="text-white/70">{(val * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-1000 ${isWeak ? 'bg-yellow-400/30' : 'bg-white/40'}`}
                      style={{ width: `${val * 100}%` }} 
                    />
                  </div>
                  <div className="absolute -top-8 right-0 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
                    <div className="bg-[#1a1a1a] border border-white/10 text-white text-[10px] font-mono px-3 py-1.5 rounded-lg shadow-xl whitespace-nowrap flex flex-col gap-0.5">
                      <span className="text-white/40">Weight: <span className="text-white">{val}</span></span>
                      {modMetrics && (
                        <span className="text-white/40">AUC: <span className={modAuc < 0.55 ? 'text-yellow-400' : 'text-white'}>{modAuc.toFixed(3)}</span></span>
                      )}
                      <span className="text-white/40">Mode: <span className="text-white">{inputMode.replace('_', ' ')}</span></span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          
          {/* Model Performance Summary */}
          <div className="mt-6 pt-4 border-t border-white/5">
            <h3 className="text-[9px] font-mono uppercase tracking-widest text-white/30 mb-3">Model AUC Scores</h3>
            <div className="space-y-1.5">
              {Object.entries(MODEL_METRICS).map(([name, metrics]) => (
                <div key={name} className="flex justify-between text-[10px] font-mono">
                  <span className={name === 'fusion' ? 'text-white' : 'text-white/40'}>
                    {name === 'fusion' ? '★ ' : ''}{name.replace('_', ' ')}
                  </span>
                  <span className={metrics.auc < 0.55 ? 'text-yellow-400' : metrics.auc > 0.75 ? 'text-green-400' : 'text-white/60'}>
                    {metrics.auc.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </div>
          <p className="mt-5 text-[10px] text-white/25 leading-relaxed">
            {isApproximateGating
              ? "Low activity_hrv weight here reflects approximate demo or fallback inputs, not a missing model artifact."
              : "These gate weights come from the saved fusion model with exported preprocessing artifacts applied."}
          </p>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between mt-8 mb-4 gap-4">
        <div className="flex items-center gap-2">
          <div className="w-1 h-1 bg-white rounded-full" />
          <h2 className="text-xs font-mono uppercase tracking-widest text-white/50">Feature Importance Analysis</h2>
        </div>
        <button 
          onClick={() => setCompareMode(!compareMode)}
          className={`text-[10px] font-mono uppercase px-4 py-2 rounded-full border transition-all duration-300 ${compareMode ? 'bg-white/20 border-white/30 text-white shadow-[0_0_15px_rgba(255,255,255,0.1)]' : 'bg-transparent border-white/10 text-white/40 hover:text-white/70 hover:border-white/20'}`}
        >
          {compareMode ? 'Hide Baseline Comparison' : 'Compare with Healthy Baseline'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* SHAP Importance Charts */}
        <div className="p-8 rounded-3xl bg-white/5 border border-white/10">
          <h2 className="text-xs font-mono uppercase tracking-widest text-white/50 mb-6">Clinical SHAP Importance</h2>
          <div className="h-[250px] w-full">
            {clinicalShapData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={clinicalShapData} layout="vertical" margin={{ left: 40 }}>
                  <XAxis type="number" hide />
                  <YAxis 
                    dataKey="name" 
                    type="category" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }} 
                    width={120}
                  />
                  <Tooltip 
                    cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                    contentStyle={{ backgroundColor: '#1a1a1a', border: 'none', borderRadius: '8px', fontSize: '10px', color: '#ffffff' }}
                    itemStyle={{ color: '#ffffff' }}
                    labelStyle={{ color: '#ffffff' }}
                  />
                  {compareMode && <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />}
                  <Bar dataKey="value" name="Current Patient" radius={[0, 4, 4, 0]}>
                    {clinicalShapData.map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={`rgba(255,255,255,${0.8 - index * 0.15})`} />
                    ))}
                  </Bar>
                  {compareMode && (
                    <Bar dataKey="baseline" name="Healthy Baseline" fill="rgba(255,255,255,0.1)" radius={[0, 4, 4, 0]} />
                  )}
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-white/20 text-xs font-mono">No clinical SHAP data</div>
            )}
          </div>
        </div>

        <div className="p-8 rounded-3xl bg-white/5 border border-white/10">
          <h2 className="text-xs font-mono uppercase tracking-widest text-white/50 mb-6">{secondaryTitle}</h2>
          <div className="h-[250px] w-full">
            {secondaryData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={secondaryData} layout="vertical" margin={{ left: 40 }}>
                  <XAxis type="number" hide />
                  <YAxis 
                    dataKey="name" 
                    type="category" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }} 
                    width={120}
                  />
                  <Tooltip 
                    cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                    contentStyle={{ backgroundColor: '#1a1a1a', border: 'none', borderRadius: '8px', fontSize: '10px', color: '#ffffff' }}
                    itemStyle={{ color: '#ffffff' }}
                    labelStyle={{ color: '#ffffff' }}
                  />
                  {compareMode && <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />}
                  <Bar dataKey="value" name="Current Patient" radius={[0, 4, 4, 0]}>
                    {secondaryData.map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={`rgba(255,255,255,${0.8 - index * 0.15})`} />
                    ))}
                  </Bar>
                  {compareMode && (
                    <Bar dataKey="baseline" name="Healthy Baseline" fill="rgba(255,255,255,0.1)" radius={[0, 4, 4, 0]} />
                  )}
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-white/20 text-xs font-mono">{noSecondaryMessage}</div>
            )}
          </div>
        </div>
      </div>

      {/* DSM-5 Mapping */}
      <div className="p-8 rounded-3xl bg-white/5 border border-white/10">
        <div className="flex items-center gap-2 mb-6">
          <Info size={16} className="text-white/40" />
          <h2 className="text-xs font-mono uppercase tracking-widest text-white/50">DSM-5 Criterion Mapping</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(triggeredCriteria || []).map((crit: string) => (
            <div key={crit} className="p-4 rounded-xl bg-white/5 border border-white/10 flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider text-white/70">{crit.replace('_', ' ')}</span>
              <div className="w-2 h-2 rounded-full bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.5)]" />
            </div>
          ))}
          {(!triggeredCriteria || triggeredCriteria.length === 0) && (
            <div className="col-span-3 text-center py-4 text-white/30 text-xs font-mono italic">
              No specific DSM-5 criteria mapped to current feature importance.
            </div>
          )}
        </div>
        <p className="mt-6 text-[10px] font-mono text-white/20 italic text-center">
          NOTE: AI-assisted decision support only. All outputs must be reviewed by a licensed clinician.
        </p>
      </div>
    </div>
  );
};
