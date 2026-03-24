import React from "react";
import { Activity, Brain, ClipboardList, FileUp, Zap } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

interface ModalityFormProps {
  onPredict: (data: any) => void;
  loading: boolean;
}

type InputMode = "demo" | "artifact_aligned";
type UploadSlot = "clinical" | "activity_hrv";
type FeatureMap = Record<string, number>;
type UploadDescriptor = {
  name: string;
  data: FeatureMap | null;
  error: string | null;
  count: number;
};

const modalities = [
  { id: "clinical", name: "Clinical", icon: ClipboardList, color: "text-blue-400" },
  { id: "activity", name: "Actigraphy + HRV", icon: Activity, color: "text-green-400" },
  { id: "eeg", name: "EEG", icon: Zap, color: "text-orange-400" },
  { id: "brain", name: "Brain FC", icon: Brain, color: "text-pink-400" },
];

const emptyUploadState = (): Record<UploadSlot, UploadDescriptor> => ({
  clinical: { name: "", data: null, error: null, count: 0 },
  activity_hrv: { name: "", data: null, error: null, count: 0 },
});

const parseCsvLine = (line: string) => {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === "\"") {
      if (inQuotes && line[i + 1] === "\"") {
        current += "\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  values.push(current);
  return values.map((value) => value.trim());
};

const coerceFeatureObject = (value: unknown) => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Expected an object of feature names to numeric values.");
  }

  const features: FeatureMap = {};
  Object.entries(value as Record<string, unknown>).forEach(([key, raw]) => {
    if (typeof raw === "number" && Number.isFinite(raw)) {
      features[key] = raw;
      return;
    }
    if (typeof raw === "string" && raw.trim() !== "") {
      const numeric = Number(raw);
      if (!Number.isNaN(numeric)) {
        features[key] = numeric;
      }
    }
  });

  if (!Object.keys(features).length) {
    throw new Error("No numeric feature values found in upload.");
  }
  return features;
};

const parseJsonFeatures = (text: string) => {
  const parsed = JSON.parse(text);

  if (Array.isArray(parsed)) {
    if (!parsed.length) throw new Error("JSON array upload is empty.");
    return coerceFeatureObject(parsed[0]);
  }

  if (
    parsed &&
    typeof parsed === "object" &&
    Array.isArray((parsed as any).feature_names) &&
    Array.isArray((parsed as any).values)
  ) {
    const featureNames = (parsed as any).feature_names as string[];
    const values = (parsed as any).values as Array<number | string>;
    if (featureNames.length !== values.length) {
      throw new Error("feature_names and values length mismatch.");
    }
    return Object.fromEntries(
      featureNames.map((name, index) => [name, Number(values[index])]),
    ) as FeatureMap;
  }

  return coerceFeatureObject(parsed);
};

const parseCsvFeatures = (text: string) => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    throw new Error("CSV upload needs a header row and one data row.");
  }

  const headers = parseCsvLine(lines[0]);
  const values = parseCsvLine(lines[1]);
  if (headers.length !== values.length) {
    throw new Error("CSV header/value length mismatch.");
  }

  const features: FeatureMap = {};
  headers.forEach((header, index) => {
    const numeric = Number(values[index]);
    if (Number.isNaN(numeric)) {
      throw new Error(`Non-numeric value for ${header}.`);
    }
    features[header] = numeric;
  });
  return features;
};

const parseFeatureFile = async (file: File) => {
  const text = await file.text();
  const lowerName = file.name.toLowerCase();

  if (lowerName.endsWith(".json")) {
    return parseJsonFeatures(text);
  }
  if (lowerName.endsWith(".csv")) {
    return parseCsvFeatures(text);
  }

  try {
    return parseJsonFeatures(text);
  } catch {
    return parseCsvFeatures(text);
  }
};

export const ModalityForm = ({ onPredict, loading }: ModalityFormProps) => {
  const [activeTab, setActiveTab] = React.useState("clinical");
  const [inputMode, setInputMode] = React.useState<InputMode>("demo");
  const [uploads, setUploads] = React.useState<Record<UploadSlot, UploadDescriptor>>(emptyUploadState);
  const [formData, setFormData] = React.useState({
    clinical: {
      tscore: 65,
      hitrt: 0.35,
      commissions: 12,
      cpt_rt_mean: 420,
      cpt_rt_std: 85,
    },
    activity: {
      act_mean: 450,
      act_std: 120,
      act_median: 430,
      act_range: 800,
      act_skew: 0.3,
      act_kurtosis: 2.1,
      hr_mean: 75,
      hr_std: 12,
      hr_median: 74,
      hr_range: 45,
      hr_skew: 0.2,
      hr_kurtosis: 2.5,
      hr_rmssd: 65,
    },
    eeg: {
      theta_beta: 2.4,
      frontal_power: 0.8,
    },
    brain: {
      pca_1: 0.12,
      pca_2: -0.05,
      pca_3: 0.08,
      pca_4: -0.03,
      pca_5: 0.01,
    },
  });

  const handleInputChange = (modality: string, field: string, value: number) => {
    setFormData((prev) => ({
      ...prev,
      [modality]: { ...prev[modality as keyof typeof prev], [field]: value },
    }));
  };

  const handleUpload = async (slot: UploadSlot, file?: File) => {
    if (!file) {
      setUploads((prev) => ({
        ...prev,
        [slot]: { name: "", data: null, error: null, count: 0 },
      }));
      return;
    }

    try {
      const parsed = await parseFeatureFile(file);
      setUploads((prev) => ({
        ...prev,
        [slot]: {
          name: file.name,
          data: parsed,
          error: null,
          count: Object.keys(parsed).length,
        },
      }));
    } catch (error: any) {
      setUploads((prev) => ({
        ...prev,
        [slot]: {
          name: file.name,
          data: null,
          error: error?.message ?? "Failed to parse upload.",
          count: 0,
        },
      }));
    }
  };

  const artifactReady =
    Boolean(uploads.clinical.data) &&
    Boolean(uploads.activity_hrv.data) &&
    !uploads.clinical.error &&
    !uploads.activity_hrv.error;

  const handleSubmit = () => {
    if (inputMode === "artifact_aligned") {
      onPredict({
        mode: "artifact_aligned",
        clinical_selected: uploads.clinical.data,
        activity_hrv_features: uploads.activity_hrv.data,
      });
      return;
    }

    onPredict({
      mode: "demo",
      ...formData,
    });
  };

  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-xl">
      <div className="flex gap-2 mb-8">
        {[
          { id: "demo", label: "Demo Sliders" },
          { id: "artifact_aligned", label: "Artifact Upload" },
        ].map((mode) => (
          <button
            key={mode.id}
            onClick={() => setInputMode(mode.id as InputMode)}
            className={`px-4 py-2 rounded-xl text-[10px] font-mono uppercase tracking-wider transition-all ${
              inputMode === mode.id
                ? "bg-white/10 text-white border border-white/20"
                : "text-white/40 hover:text-white/70 border border-transparent"
            }`}
          >
            {mode.label}
          </button>
        ))}
      </div>

      {inputMode === "demo" ? (
        <>
          <div className="flex gap-2 mb-8 overflow-x-auto pb-2 scrollbar-hide">
            {modalities.map((m) => (
              <button
                key={m.id}
                onClick={() => setActiveTab(m.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all whitespace-nowrap ${
                  activeTab === m.id
                    ? "bg-white/10 text-white border border-white/20"
                    : "text-white/40 hover:text-white/60"
                }`}
              >
                <m.icon size={16} className={m.color} />
                <span className="text-xs font-medium uppercase tracking-wider">{m.name}</span>
              </button>
            ))}
          </div>

          <div className="min-h-[200px]">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                {activeTab === "clinical" && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <InputRange label="Neuro TScore VarSE" value={formData.clinical.tscore} min={30} max={100} onChange={(v) => handleInputChange("clinical", "tscore", v)} />
                    <InputRange label="Adhd TScore HitRTIsi" value={formData.clinical.hitrt} min={0.1} max={1.0} step={0.01} onChange={(v) => handleInputChange("clinical", "hitrt", v)} />
                    <InputRange label="General TScore Commissions" value={formData.clinical.commissions} min={0} max={50} onChange={(v) => handleInputChange("clinical", "commissions", v)} />
                    <InputRange label="CPT RT Mean (ms)" value={formData.clinical.cpt_rt_mean} min={200} max={800} onChange={(v) => handleInputChange("clinical", "cpt_rt_mean", v)} />
                    <InputRange label="CPT RT Std (ms)" value={formData.clinical.cpt_rt_std} min={20} max={200} onChange={(v) => handleInputChange("clinical", "cpt_rt_std", v)} />
                  </div>
                )}

                {activeTab === "activity" && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <InputRange label="Activity Mean" value={formData.activity.act_mean} min={0} max={1000} onChange={(v) => handleInputChange("activity", "act_mean", v)} />
                    <InputRange label="Activity Std" value={formData.activity.act_std} min={0} max={500} onChange={(v) => handleInputChange("activity", "act_std", v)} />
                    <InputRange label="Activity Median" value={formData.activity.act_median} min={0} max={1000} onChange={(v) => handleInputChange("activity", "act_median", v)} />
                    <InputRange label="Activity Range" value={formData.activity.act_range} min={0} max={2000} onChange={(v) => handleInputChange("activity", "act_range", v)} />
                    <InputRange label="HRV RMSSD (ms)" value={formData.activity.hr_rmssd} min={10} max={150} onChange={(v) => handleInputChange("activity", "hr_rmssd", v)} />
                    <InputRange label="HR Mean (bpm)" value={formData.activity.hr_mean} min={40} max={120} onChange={(v) => handleInputChange("activity", "hr_mean", v)} />
                    <InputRange label="HR Std" value={formData.activity.hr_std} min={1} max={40} onChange={(v) => handleInputChange("activity", "hr_std", v)} />
                    <InputRange label="Activity Skewness" value={formData.activity.act_skew} min={-2} max={3} step={0.1} onChange={(v) => handleInputChange("activity", "act_skew", v)} />
                    <InputRange label="Activity Kurtosis" value={formData.activity.act_kurtosis} min={0} max={6} step={0.1} onChange={(v) => handleInputChange("activity", "act_kurtosis", v)} />
                  </div>
                )}

                {activeTab === "eeg" && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <InputRange label="Theta/Beta Ratio" value={formData.eeg.theta_beta} min={0.5} max={5.0} step={0.1} onChange={(v) => handleInputChange("eeg", "theta_beta", v)} />
                    <InputRange label="Frontal Power (F3/F4/Fz)" value={formData.eeg.frontal_power} min={0} max={2.0} step={0.1} onChange={(v) => handleInputChange("eeg", "frontal_power", v)} />
                  </div>
                )}

                {activeTab === "brain" && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <InputRange label="PCA Component 1" value={formData.brain.pca_1} min={-1} max={1} step={0.01} onChange={(v) => handleInputChange("brain", "pca_1", v)} />
                    <InputRange label="PCA Component 2" value={formData.brain.pca_2} min={-1} max={1} step={0.01} onChange={(v) => handleInputChange("brain", "pca_2", v)} />
                    <InputRange label="PCA Component 3" value={formData.brain.pca_3} min={-1} max={1} step={0.01} onChange={(v) => handleInputChange("brain", "pca_3", v)} />
                    <InputRange label="PCA Component 4" value={formData.brain.pca_4} min={-1} max={1} step={0.01} onChange={(v) => handleInputChange("brain", "pca_4", v)} />
                    <InputRange label="PCA Component 5" value={formData.brain.pca_5} min={-1} max={1} step={0.01} onChange={(v) => handleInputChange("brain", "pca_5", v)} />
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </div>
        </>
      ) : (
        <div className="space-y-6">
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
            <div className="flex items-start gap-3 mb-4">
              <FileUp size={18} className="text-white/60 mt-0.5" />
              <div>
                <h3 className="text-xs font-mono uppercase tracking-widest text-white/70 mb-1">
                  Artifact-Aligned Upload
                </h3>
                <p className="text-[11px] text-white/40 leading-relaxed">
                  Upload the exported `clinical_selected` and `activity_hrv` JSON or CSV templates from `models/preprocessing`.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <UploadCard
                title="Clinical Selected Features"
                description="Upload the exact selected clinical features exported from the notebook-aligned bundle."
                state={uploads.clinical}
                onFile={(file) => handleUpload("clinical", file)}
              />
              <UploadCard
                title="Activity HRV Features"
                description="Upload the exact 29-feature activity_hrv vector exported from the notebook-aligned bundle."
                state={uploads.activity_hrv}
                onFile={(file) => handleUpload("activity_hrv", file)}
              />
            </div>
          </div>
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={loading || (inputMode === "artifact_aligned" && !artifactReady)}
        className="w-full mt-12 py-4 bg-white text-black rounded-xl font-bold uppercase tracking-widest hover:bg-white/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
      >
        {loading ? (
          <div className="w-5 h-5 border-2 border-black/20 border-t-black rounded-full animate-spin" />
        ) : inputMode === "artifact_aligned" ? (
          "Run Artifact-Aligned Inference"
        ) : (
          "Run Multimodal Inference"
        )}
      </button>

      <p className="mt-4 text-[9px] font-mono text-white/20 text-center uppercase tracking-wider">
        {inputMode === "artifact_aligned"
          ? "Artifact mode requires exported preprocessing bundles and exact feature uploads"
          : "Demo mode uses approximate inputs mapped into the multimodal feature spaces"}
      </p>
    </div>
  );
};

const UploadCard = ({
  title,
  description,
  state,
  onFile,
}: {
  title: string;
  description: string;
  state: UploadDescriptor;
  onFile: (file?: File) => void;
}) => (
  <div className="rounded-xl border border-white/10 bg-black/20 p-4 space-y-3">
    <div>
      <h4 className="text-[10px] font-mono uppercase tracking-widest text-white/70">{title}</h4>
      <p className="text-[11px] text-white/35 mt-2 leading-relaxed">{description}</p>
    </div>
    <label className="flex items-center justify-between gap-4 rounded-lg border border-dashed border-white/20 px-4 py-3 cursor-pointer hover:border-white/30 transition-colors">
      <span className="text-[10px] font-mono uppercase tracking-wider text-white/50">
        {state.name || "Choose JSON or CSV"}
      </span>
      <span className="text-[10px] font-mono uppercase tracking-wider text-white/70">Upload</span>
      <input
        type="file"
        accept=".json,.csv,application/json,text/csv"
        className="hidden"
        onChange={(event) => onFile(event.target.files?.[0])}
      />
    </label>
    {state.count > 0 && !state.error && (
      <div className="text-[10px] font-mono uppercase tracking-widest text-green-400/80">
        Parsed {state.count} features
      </div>
    )}
    {state.error && (
      <div className="text-[10px] font-mono uppercase tracking-widest text-red-400/80">
        {state.error}
      </div>
    )}
  </div>
);

const InputRange = ({ label, value, min, max, step = 1, onChange }: any) => (
  <div className="space-y-2">
    <div className="flex justify-between items-center">
      <label className="text-[10px] font-mono uppercase tracking-wider text-white/50">{label}</label>
      <span className="text-xs font-mono text-white">{value}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-white"
    />
  </div>
);
