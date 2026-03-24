import express from "express";
import fs from "fs";
import path from "path";
import { spawn, type ChildProcess } from "child_process";
import { createServer as createViteServer } from "vite";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL ?? "http://127.0.0.1:5000";
const APP_PORT = Number(process.env.PORT ?? 3000);
const PYTHON_BIN = process.env.PYTHON_BIN ?? "python";
const PYTHON_SCRIPT = path.join(__dirname, "inference_service.py");
const HEALTH_TIMEOUT_MS = 20000;
const PYTHON_PORT = (() => {
  try {
    return new URL(PYTHON_SERVICE_URL).port || "5000";
  } catch {
    return "5000";
  }
})();

type PythonState = {
  ready: boolean;
  autoStarted: boolean;
  lastError: string | null;
};

let pythonProcess: ChildProcess | null = null;
let pythonReadyPromise: Promise<boolean> | null = null;
const pythonState: PythonState = {
  ready: false,
  autoStarted: false,
  lastError: null,
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchPythonHealth() {
  try {
    const response = await fetch(`${PYTHON_SERVICE_URL}/health`);
    if (!response.ok) {
      throw new Error(`health returned ${response.status}`);
    }
    const payload = await response.json();
    pythonState.ready = true;
    pythonState.lastError = null;
    return payload;
  } catch (error: any) {
    pythonState.ready = false;
    pythonState.lastError = error?.message ?? "unknown python health error";
    return null;
  }
}

function startPythonService() {
  if (pythonProcess || process.env.NODE_ENV === "production") {
    return;
  }
  if (!fs.existsSync(PYTHON_SCRIPT)) {
    pythonState.lastError = `missing python script: ${PYTHON_SCRIPT}`;
    return;
  }

  pythonState.autoStarted = true;
  pythonProcess = spawn(PYTHON_BIN, [PYTHON_SCRIPT], {
    cwd: __dirname,
    env: { ...process.env, PYTHONIOENCODING: "utf-8", INFERENCE_PORT: process.env.INFERENCE_PORT ?? PYTHON_PORT },
    stdio: ["ignore", "pipe", "pipe"],
  });

  pythonProcess.stdout?.on("data", (chunk) => {
    process.stdout.write(`[python] ${chunk.toString()}`);
  });
  pythonProcess.stderr?.on("data", (chunk) => {
    process.stderr.write(`[python] ${chunk.toString()}`);
  });
  pythonProcess.on("error", (error) => {
    pythonState.ready = false;
    pythonState.lastError = error.message;
    pythonProcess = null;
    pythonReadyPromise = null;
  });
  pythonProcess.on("exit", (code, signal) => {
    pythonState.ready = false;
    pythonState.lastError = `python exited (code=${code}, signal=${signal})`;
    pythonProcess = null;
    pythonReadyPromise = null;
  });
}

async function waitForPythonHealth(timeoutMs = HEALTH_TIMEOUT_MS) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const payload = await fetchPythonHealth();
    if (payload) {
      return true;
    }
    await sleep(500);
  }
  pythonState.lastError = pythonState.lastError ?? "timed out waiting for python health";
  return false;
}

async function ensurePythonReady() {
  const existing = await fetchPythonHealth();
  if (existing) {
    return true;
  }

  if (process.env.NODE_ENV === "production") {
    return false;
  }

  if (!pythonProcess) {
    startPythonService();
  }
  if (!pythonReadyPromise) {
    pythonReadyPromise = waitForPythonHealth().finally(() => {
      pythonReadyPromise = null;
    });
  }
  return pythonReadyPromise;
}

function stopPythonService() {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
    pythonReadyPromise = null;
  }
}

process.on("exit", stopPythonService);
process.on("SIGINT", () => {
  stopPythonService();
  process.exit(0);
});
process.on("SIGTERM", () => {
  stopPythonService();
  process.exit(0);
});

function buildFallbackPrediction(body: any, reason: string) {
  const resultsPath = path.join(__dirname, "models", "adhd_xai_results.json");
  const modelResults = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
  const { clinical, eeg } = body ?? {};

  const gates = modelResults.fusion_gate_weights || {
    clinical: 0.535,
    activity_hrv: 0.465,
  };
  const clinicalAuc = modelResults.results?.clinical?.auc || 0.797;
  const eegAuc = modelResults.results?.eeg?.auc || 0.859;

  let confidence = 0.5;
  if (clinical) confidence += (clinicalAuc - 0.5) * 0.3;
  if (eeg) confidence += (eegAuc - 0.5) * 0.4;
  const isADHD = confidence > 0.55;

  const shap = {
    clinical: [
      { feat: "Neuro TScore VarSE", val: 0.3384 },
      { feat: "ACC__fft_coeff_real_53", val: 0.2873 },
      { feat: "General TScore Commissions", val: 0.21 },
      { feat: "Adhd TScore HitRTIsi", val: 0.18 },
      { feat: "Age", val: 0.05 },
    ],
    activity_hrv: [],
    eeg: [
      { feat: "F3", val: 0.003 },
      { feat: "F4", val: 0.0028 },
      { feat: "O2", val: 0.0026 },
      { feat: "P7", val: 0.0025 },
      { feat: "P3", val: 0.0024 },
    ],
  };

  const dsm5Mapping: Record<string, string[]> = {
    inattention: ["rt", "cpt", "attention", "error", "commission", "omission", "theta", "frontal", "F3", "F4", "Fz"],
    hyperactivity: ["activity", "steps", "movement", "motor", "act_mean", "act_std"],
    impulsivity: ["rt_std", "variability", "reaction", "beta"],
    executive: ["working", "memory", "switch", "inhibit", "stroop"],
    sleep_arousal: ["sleep", "hrv", "rmssd", "hr_", "heart"],
  };
  const triggeredCriteria: string[] = [];
  const allFeats = [...shap.clinical, ...shap.eeg].map((entry) => entry.feat.toLowerCase());
  Object.entries(dsm5Mapping).forEach(([criterion, keywords]) => {
    if (keywords.some((keyword) => allFeats.some((feat) => feat.includes(keyword.toLowerCase())))) {
      triggeredCriteria.push(criterion);
    }
  });

  return {
    prediction: isADHD ? "ADHD" : "Non-ADHD (likely HC)",
    confidence: Math.min(confidence, 0.99),
    isADHD,
    gates,
    shap,
    triggeredCriteria,
    timestamp: new Date().toISOString(),
    fallback: true,
    degraded: true,
    service_source: "node_fallback",
    input_mode: "demo",
    warnings: [
      "Python inference service unavailable; using node fallback heuristics.",
      reason,
      "Fallback output does not confirm whether activity_hrv loaded in Python.",
    ],
    models_used: [],
  };
}

async function startServer() {
  const app = express();

  app.use(express.json({ limit: "2mb" }));

  if (process.env.NODE_ENV !== "production") {
    void ensurePythonReady();
  }

  app.post("/api/predict", async (req, res) => {
    const requestedMode = req.body?.mode === "artifact_aligned" ? "artifact_aligned" : "demo";
    await ensurePythonReady();

    try {
      const response = await fetch(`${PYTHON_SERVICE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });

      const result = await response.json();
      res.status(response.status).json(result);
    } catch (error: any) {
      const reason = error?.message ?? pythonState.lastError ?? "python service unavailable";
      console.error("Python service error:", reason);

      if (requestedMode === "artifact_aligned") {
        res.status(503).json({
          error: "Artifact-aligned inference requires the Python service.",
          service_source: "node_proxy",
          input_mode: "artifact_aligned",
          warnings: [
            "Python inference service is unavailable, so artifact-aligned inference cannot run.",
            reason,
          ],
          models_used: [],
        });
        return;
      }

      res.json(buildFallbackPrediction(req.body, reason));
    }
  });

  app.get("/api/health", async (req, res) => {
    const ready = await ensurePythonReady();
    if (!ready) {
      res.json({
        status: "python_service_offline",
        service_source: "node_proxy",
        modalities_loaded: [],
        models_loaded: [],
        artifact_status: {},
        ready: {
          demo: { ready: false, required_models_loaded: [] },
          artifact_aligned: { ready: false, required_artifacts_loaded: {} },
        },
        warnings: [pythonState.lastError ?? "python service unavailable"],
      });
      return;
    }

    try {
      const response = await fetch(`${PYTHON_SERVICE_URL}/health`);
      const result = await response.json();
      res.status(response.status).json(result);
    } catch (error: any) {
      res.json({
        status: "python_service_offline",
        service_source: "node_proxy",
        modalities_loaded: [],
        models_loaded: [],
        artifact_status: {},
        ready: {
          demo: { ready: false, required_models_loaded: [] },
          artifact_aligned: { ready: false, required_artifacts_loaded: {} },
        },
        warnings: [error?.message ?? pythonState.lastError ?? "python service unavailable"],
      });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(APP_PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${APP_PORT}`);
    console.log(`Python inference expected at ${PYTHON_SERVICE_URL}`);
    if (process.env.NODE_ENV !== "production") {
      console.log(`Python auto-start command: ${PYTHON_BIN} ${PYTHON_SCRIPT}`);
    }
  });
}

startServer();
