
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from flask import jsonify, request, Flask
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(__file__)
SERVICE_PORT = int(os.getenv("INFERENCE_PORT", "5000"))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_PATH = os.path.join(BASE_DIR, "models", "adhd_xai_results.json")
PREPROCESSING_DIR = os.path.join(BASE_DIR, "models", "preprocessing")

PUBLIC_MODALITY_ORDER = ["clinical", "activity_hrv", "eeg", "brain", "fusion"]
INTERNAL_TO_PUBLIC = {
    "clinical": "clinical",
    "bio": "activity_hrv",
    "eeg": "eeg",
    "brain": "brain",
    "fusion": "fusion",
}
REQUIRED_ARTIFACTS = {
    "clinical": os.path.join(PREPROCESSING_DIR, "clinical_bundle.json"),
    "activity_hrv": os.path.join(PREPROCESSING_DIR, "activity_hrv_bundle.json"),
}
DEMO_GATING_WARNING = (
    "Demo mode uses approximate clinical/activity_hrv inputs. "
    "Gate weights are illustrative and do not indicate whether a model failed to load."
)
ARTIFACT_MODE_WARNING = (
    "Artifact-aligned mode expects the exact saved clinical selected features and "
    "activity_hrv feature vector exported from the training pipeline."
)

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    MODEL_RESULTS = json.load(f)


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim: int = 30, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, 2)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        return self.classifier(emb), emb


class BioEncoder(nn.Module):
    def __init__(self, input_dim: int = 29, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, 2)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        return self.classifier(emb), emb


class EEGNet(nn.Module):
    def __init__(
        self,
        n_ch: int = 19,
        n_time: int = 512,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            x_probe = torch.zeros(1, 1, n_ch, n_time)
            self._fs = self.conv3(self.conv2(self.conv1(x_probe))).view(1, -1).size(1)
        self.embedding = nn.Linear(self._fs, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.embedding(x.view(x.size(0), -1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.get_embedding(x)
        return self.classifier(emb), emb


class BrainEncoder(nn.Module):
    def __init__(self, input_dim: int = 100, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, 2)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        return self.classifier(emb), emb


class MoEFusion(nn.Module):
    def __init__(self, mod_dims: Dict[str, int], embed_dim: int = 128, n_classes: int = 2):
        super().__init__()
        self.mod_names = list(mod_dims.keys())
        self.experts = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )
                for name, dim in mod_dims.items()
            }
        )
        gate_in = sum(mod_dims.values())
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(mod_dims)),
        )
        self.temp = 0.5
        self.cross = nn.MultiheadAttention(
            embed_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.cls = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(
        self, x_concat: torch.Tensor, return_gates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        slices: Dict[str, torch.Tensor] = {}
        offset = 0
        for name in self.mod_names:
            dim = self.experts[name][0].in_features
            slices[name] = x_concat[:, offset : offset + dim]
            offset += dim

        expert_outs = torch.stack(
            [self.experts[name](slices[name]) for name in self.mod_names], dim=1
        )
        gate_logits = self.gate(x_concat) / self.temp
        gate_weights = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
        fused = (expert_outs * gate_weights).sum(dim=1)
        attn_out, _ = self.cross(expert_outs, expert_outs, expert_outs)
        fused = fused + attn_out.mean(dim=1)
        logits = self.cls(fused)
        if return_gates:
            return logits, gate_weights.squeeze(-1)
        return logits


DSM5_MAPPING = {
    "inattention": [
        "rt",
        "cpt",
        "attention",
        "error",
        "commission",
        "omission",
        "theta",
        "frontal",
        "F3",
        "F4",
        "Fz",
    ],
    "hyperactivity": ["activity", "steps", "movement", "motor", "act_mean", "act_std"],
    "impulsivity": ["rt_std", "variability", "reaction", "beta"],
    "executive": ["working", "memory", "switch", "inhibit", "stroop"],
    "sleep_arousal": ["sleep", "hrv", "rmssd", "hr_", "heart"],
}

DEMO_CLINICAL_FEATURE_NAMES = [
    "Neuro TScore VarSE",
    "Adhd TScore HitRTIsi",
    "General TScore Commissions",
    "cpt_rt_mean",
    "cpt_rt_std",
    "cpt_rt_cv",
    "act_mean",
    "act_std",
    "act_median",
    "act_range",
    "act_skew",
    "act_kurtosis",
    "hr_mean",
    "hr_std",
    "hr_median",
    "hr_range",
    "hr_skew",
    "hr_kurtosis",
    "hr_rmssd",
]
BIO_FEATURE_NAMES = [
    "act_mean",
    "act_std",
    "act_median",
    "act_range",
    "act_skew",
    "act_kurtosis",
    "act_iqr",
    "act_p10",
    "act_p90",
    "act_cv",
    "act_transitions",
    "act_n_bouts",
    "act_mean_bout_len",
    "act_short_bout_frac",
    "hr_mean",
    "hr_std",
    "hr_median",
    "hr_range",
    "hr_skew",
    "hr_kurtosis",
    "hr_iqr",
    "hr_p10",
    "hr_p90",
    "hr_cv",
    "hr_transitions",
    "hr_rmssd",
    "hr_sdnn",
    "hr_pnn50_proxy",
    "hr_lf_hf_ratio",
]
EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
]

loaded_models: Dict[str, nn.Module] = {}
preprocessing_bundles: Dict[str, Dict[str, Any]] = {}
preprocessing_scalers: Dict[str, StandardScaler] = {}
artifact_status: Dict[str, Dict[str, Any]] = {}
runtime_initialized = False


def log(message: str) -> None:
    print(message, flush=True)


def public_name(name: str) -> str:
    return INTERNAL_TO_PUBLIC.get(name, name)


def ordered_public_names(names: List[str]) -> List[str]:
    known = [name for name in PUBLIC_MODALITY_ORDER if name in names]
    unknown = sorted(name for name in names if name not in PUBLIC_MODALITY_ORDER)
    return known + unknown


def get_public_modalities_loaded() -> List[str]:
    return ordered_public_names([public_name(name) for name in loaded_models.keys()])


def torch_load_weights(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_standard_scaler(bundle: Dict[str, Any]) -> StandardScaler:
    scaler_data = bundle["scaler"]
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(scaler_data["mean"], dtype=np.float32)
    scaler.scale_ = np.asarray(scaler_data["scale"], dtype=np.float32)
    scaler.var_ = np.asarray(
        scaler_data.get("var", np.square(scaler.scale_)),
        dtype=np.float32,
    )
    scaler.n_features_in_ = int(scaler_data.get("n_features_in", len(scaler.mean_)))
    scaler.n_samples_seen_ = int(scaler_data.get("n_samples_seen", scaler.n_features_in_))
    return scaler


def validate_bundle(modality: str, bundle: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    feature_names = bundle.get("selected_feature_names") or bundle.get("feature_names")
    scaler_data = bundle.get("scaler")
    if not isinstance(feature_names, list) or not feature_names:
        errors.append("missing feature_names")
    if not isinstance(scaler_data, dict):
        errors.append("missing scaler")
    else:
        mean = scaler_data.get("mean")
        scale = scaler_data.get("scale")
        if not isinstance(mean, list) or not isinstance(scale, list):
            errors.append("scaler mean/scale missing")
        elif len(mean) != len(scale):
            errors.append("scaler mean/scale length mismatch")
        elif feature_names and len(mean) != len(feature_names):
            errors.append("feature count and scaler length mismatch")
    if modality == "activity_hrv" and feature_names and len(feature_names) != 29:
        errors.append("activity_hrv bundle must contain 29 features")
    return errors


def load_preprocessing_artifacts() -> None:
    preprocessing_bundles.clear()
    preprocessing_scalers.clear()
    artifact_status.clear()

    for modality, path in REQUIRED_ARTIFACTS.items():
        rel_path = os.path.relpath(path, BASE_DIR).replace("\\", "/")
        status: Dict[str, Any] = {
            "path": rel_path,
            "present": os.path.exists(path),
            "loaded": False,
        }
        if not status["present"]:
            status["error"] = "missing"
            artifact_status[modality] = status
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            errors = validate_bundle(modality, bundle)
            if errors:
                status["error"] = "; ".join(errors)
            else:
                preprocessing_bundles[modality] = bundle
                preprocessing_scalers[modality] = build_standard_scaler(bundle)
                feature_names = bundle.get("selected_feature_names") or bundle.get("feature_names") or []
                status["loaded"] = True
                status["feature_count"] = len(feature_names)
        except Exception as exc:
            status["error"] = str(exc)

        artifact_status[modality] = status


def artifact_aligned_ready() -> bool:
    required_modalities = {"clinical", "activity_hrv"}
    required_models = {"clinical", "bio", "fusion"}
    return required_modalities.issubset(preprocessing_bundles.keys()) and required_models.issubset(
        loaded_models.keys()
    )


def initialize_runtime(force: bool = False) -> None:
    global runtime_initialized
    if runtime_initialized and not force:
        return
    load_models()
    load_preprocessing_artifacts()
    runtime_initialized = True


def load_models() -> None:
    loaded_models.clear()
    model_files = {
        "clinical": ("model_clinical.pth", ClinicalEncoder),
        "bio": ("model_bio.pth", BioEncoder),
        "eeg": ("model_eeg.pth", EEGNet),
        "brain": ("model_brain.pth", BrainEncoder),
        "fusion": ("model_fusion.pth", None),
    }

    for name, (filename, model_class) in model_files.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            log(f"[warn] {filename} not found")
            continue

        try:
            if name == "fusion":
                model = MoEFusion({"clinical": 128, "activity_hrv": 128})
            else:
                model = model_class()
            state_dict = torch_load_weights(filepath)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            loaded_models[name] = model
            log(f"[ok] loaded {public_name(name)} from {filename}")
        except Exception as exc:
            log(f"[warn] failed to load {public_name(name)} from {filename}: {exc}")


def process_clinical_input(data: Dict[str, Any]) -> np.ndarray:
    features = np.zeros(30, dtype=np.float32)
    clinical = data.get("clinical", {})

    feature_map = {
        "tscore": 0,
        "hitrt": 1,
        "commissions": 2,
        "cpt_rt_mean": 3,
        "cpt_rt_std": 4,
    }
    for key, idx in feature_map.items():
        if key in clinical:
            features[idx] = float(clinical[key])

    if "cpt_rt_mean" in clinical and "cpt_rt_std" in clinical:
        mean = float(clinical["cpt_rt_mean"])
        std = float(clinical["cpt_rt_std"])
        features[5] = std / max(mean, 1e-6)

    activity = data.get("activity", {})
    activity_map = {
        "act_mean": 6,
        "act_std": 7,
        "act_median": 8,
        "act_range": 9,
        "act_skew": 10,
        "act_kurtosis": 11,
    }
    for key, idx in activity_map.items():
        if key in activity:
            features[idx] = float(activity[key])

    hrv_map = {
        "hr_mean": 12,
        "hr_std": 13,
        "hr_median": 14,
        "hr_range": 15,
        "hr_skew": 16,
        "hr_kurtosis": 17,
        "hr_rmssd": 18,
    }
    for key, idx in hrv_map.items():
        if key in activity:
            features[idx] = float(activity[key])

    return features


def process_eeg_input(data: Dict[str, Any]) -> np.ndarray:
    eeg = data.get("eeg", {})
    theta_beta = float(eeg.get("theta_beta", 2.0))
    frontal_power = float(eeg.get("frontal_power", 0.5))

    n_ch = 19
    n_time = 512
    signal = np.zeros((n_ch, n_time), dtype=np.float32)
    t = np.linspace(0, 4, n_time)

    for ch in range(n_ch):
        theta_freq = 5.0 + (theta_beta - 2.0) * 2.0
        beta_freq = 15.0 + (2.0 - theta_beta) * 3.0
        theta_amp = frontal_power * (1.0 if ch < 6 else 0.3)
        beta_amp = (1.0 - frontal_power) * 0.5
        signal[ch] = (
            theta_amp * np.sin(2 * np.pi * theta_freq * t)
            + beta_amp * np.sin(2 * np.pi * beta_freq * t)
            + np.random.randn(n_time) * 0.1
        )

    return signal


def process_brain_input(data: Dict[str, Any]) -> np.ndarray:
    brain = data.get("brain", {})
    features = np.zeros(100, dtype=np.float32)
    for i in range(1, 6):
        key = f"pca_{i}"
        if key in brain:
            features[i - 1] = float(brain[key])
    return features


def process_bio_input(data: Dict[str, Any]) -> np.ndarray:
    features = np.zeros(29, dtype=np.float32)
    activity = data.get("activity", {})

    act_mean = float(activity.get("act_mean", 0))
    act_std = float(activity.get("act_std", 0))
    act_median = float(activity.get("act_median", act_mean))
    act_range = float(activity.get("act_range", 0))
    act_skew = float(activity.get("act_skew", 0))
    act_kurtosis = float(activity.get("act_kurtosis", 0))

    features[0] = act_mean
    features[1] = act_std
    features[2] = act_median
    features[3] = act_range
    features[4] = act_skew
    features[5] = act_kurtosis
    features[6] = act_std * 1.349
    features[7] = act_mean - 1.282 * act_std
    features[8] = act_mean + 1.282 * act_std
    features[9] = act_std / max(act_mean, 1e-6)
    features[10] = act_range / max(act_std, 1e-6)
    features[11] = max(act_range / 100, 1)
    features[12] = max(act_range / max(features[11], 1e-6), 1)
    features[13] = 0.5

    hr_mean = float(activity.get("hr_mean", 0))
    hr_std = float(activity.get("hr_std", 0))
    hr_median = float(activity.get("hr_median", hr_mean))
    hr_range = float(activity.get("hr_range", 0))
    hr_skew = float(activity.get("hr_skew", 0))
    hr_kurtosis = float(activity.get("hr_kurtosis", 0))
    hr_rmssd = float(activity.get("hr_rmssd", 0))

    features[14] = hr_mean
    features[15] = hr_std
    features[16] = hr_median
    features[17] = hr_range
    features[18] = hr_skew
    features[19] = hr_kurtosis
    features[20] = hr_std * 1.349
    features[21] = hr_mean - 1.282 * hr_std
    features[22] = hr_mean + 1.282 * hr_std
    features[23] = hr_std / max(hr_mean, 1e-6)
    features[24] = hr_range / max(hr_std, 1e-6)
    features[25] = hr_rmssd
    features[26] = hr_rmssd * 0.75
    features[27] = hr_rmssd * 0.22
    features[28] = 2.5 + (hr_mean - 70) * 0.05

    return features


def get_bundle_feature_names(modality: str) -> List[str]:
    bundle = preprocessing_bundles.get(modality, {})
    feature_names = bundle.get("selected_feature_names") or bundle.get("feature_names") or []
    return list(feature_names)


def vector_from_payload(payload: Any, feature_names: List[str], label: str) -> np.ndarray:
    if isinstance(payload, list):
        if len(payload) != len(feature_names):
            raise ValueError(
                f"{label} expected {len(feature_names)} values, received {len(payload)}"
            )
        return np.asarray([float(value) for value in payload], dtype=np.float32)

    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a keyed object or ordered list")

    missing = [name for name in feature_names if name not in payload]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{label} is missing {len(missing)} required features. First missing: {preview}"
        )

    return np.asarray([float(payload[name]) for name in feature_names], dtype=np.float32)


def scale_with_bundle(modality: str, features: np.ndarray) -> np.ndarray:
    scaler = preprocessing_scalers[modality]
    return scaler.transform(features.reshape(1, -1))[0].astype(np.float32)


def generate_shap_values(
    features: np.ndarray, feature_names: List[str], model_name: str
) -> List[Dict[str, Any]]:
    importance = np.abs(features) / (np.abs(features).sum() + 1e-8)
    auc = MODEL_RESULTS.get("results", {}).get(model_name, {}).get("auc", 0.5)
    importance *= auc

    top_idx = np.argsort(importance)[::-1][:5]
    shap_list = []
    for idx in top_idx:
        shap_list.append(
            {
                "feat": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                "val": round(float(importance[idx]), 4),
            }
        )
    return shap_list


def predict_modality(
    internal_name: str,
    tensor: torch.Tensor,
    embeddings: Dict[str, torch.Tensor],
    modality_logits: Dict[str, torch.Tensor],
    models_used: List[str],
    warnings: List[str],
) -> None:
    if internal_name not in loaded_models:
        return

    try:
        logits, emb = loaded_models[internal_name](tensor)
        public_modality = public_name(internal_name)
        embeddings[public_modality] = emb
        modality_logits[public_modality] = logits
        if public_modality not in models_used:
            models_used.append(public_modality)
    except Exception as exc:
        warnings.append(f"{public_name(internal_name)} inference error: {exc}")


def compute_fusion_output(
    embeddings: Dict[str, torch.Tensor],
    modality_logits: Dict[str, torch.Tensor],
    models_used: List[str],
    warnings: List[str],
) -> Tuple[float, Dict[str, float]]:
    default_gates = MODEL_RESULTS.get(
        "fusion_gate_weights", {"clinical": 0.535, "activity_hrv": 0.465}
    )

    if "fusion" in loaded_models:
        emb_list = []
        fusion_mods = []
        for modality in ["clinical", "activity_hrv"]:
            if modality in embeddings:
                emb_list.append(embeddings[modality])
                fusion_mods.append(modality)

        if len(emb_list) >= 2:
            try:
                fused_input = torch.cat(emb_list, dim=-1)
                logits, gates = loaded_models["fusion"](fused_input, return_gates=True)
                probs = torch.softmax(logits, dim=-1)
                pred_prob = probs[0, 1].item()
                gate_weights = gates[0].detach().cpu().numpy()
                gate_dict = {
                    modality: round(float(gate_weights[idx]), 4)
                    for idx, modality in enumerate(fusion_mods)
                }
                if "fusion" not in models_used:
                    models_used.append("fusion")
                return pred_prob, gate_dict
            except Exception as exc:
                warnings.append(f"fusion inference error: {exc}")
        else:
            warnings.append(
                "fusion requires both clinical and activity_hrv embeddings; using single-modality fallback"
            )
    else:
        warnings.append("fusion model unavailable; using single-modality fallback")

    if modality_logits:
        best_modality = max(
            modality_logits.keys(),
            key=lambda name: MODEL_RESULTS.get("results", {}).get(name, {}).get("auc", 0),
        )
        logits = modality_logits[best_modality]
        probs = torch.softmax(logits, dim=-1)
        return probs[0, 1].item(), default_gates

    warnings.append("no modality logits available; returning neutral confidence")
    return 0.5, default_gates


def build_triggered_criteria(shap_groups: List[List[Dict[str, Any]]]) -> List[str]:
    all_features = [item["feat"].lower() for group in shap_groups for item in group]
    triggered = []
    for criterion, keywords in DSM5_MAPPING.items():
        if any(keyword.lower() in feat for feat in all_features for keyword in keywords):
            triggered.append(criterion)
    return triggered


def build_response(
    pred_prob: float,
    gate_dict: Dict[str, float],
    shap: Dict[str, List[Dict[str, Any]]],
    triggered_criteria: List[str],
    input_mode: str,
    models_used: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    is_adhd = pred_prob >= 0.5
    ordered_models_used = ordered_public_names(models_used)
    return {
        "prediction": "ADHD" if is_adhd else "Non-ADHD (likely HC)",
        "confidence": round(pred_prob if is_adhd else 1 - pred_prob, 4),
        "isADHD": is_adhd,
        "gates": gate_dict,
        "shap": shap,
        "triggeredCriteria": triggered_criteria,
        "timestamp": datetime.now().isoformat(),
        "models_used": ordered_models_used,
        "service_source": "python_inference",
        "input_mode": input_mode,
        "warnings": warnings,
    }


def health_payload() -> Dict[str, Any]:
    modalities_loaded = get_public_modalities_loaded()
    ready = {
        "demo": {"ready": bool(modalities_loaded), "required_models_loaded": modalities_loaded},
        "artifact_aligned": {
            "ready": artifact_aligned_ready(),
            "required_artifacts_loaded": {
                key: bool(status.get("loaded")) for key, status in artifact_status.items()
            },
        },
    }
    return {
        "status": "ok",
        "service_source": "python_inference",
        "modalities_loaded": modalities_loaded,
        "models_loaded": modalities_loaded,
        "artifact_status": artifact_status,
        "ready": ready,
        "results": MODEL_RESULTS.get("results", {}),
    }


@app.route("/health", methods=["GET"])
def health() -> Any:
    initialize_runtime()
    return jsonify(health_payload())


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    initialize_runtime()
    data = request.json or {}
    input_mode = data.get("mode", "demo")

    if input_mode == "artifact_aligned":
        if not artifact_aligned_ready():
            response = {
                "error": "artifact_aligned mode requires preprocessing bundles and fusion artifacts",
                "service_source": "python_inference",
                "input_mode": "artifact_aligned",
                "warnings": [
                    ARTIFACT_MODE_WARNING,
                    "Run the preprocessing export script and restart the service.",
                ],
                "artifact_status": artifact_status,
                "models_used": [],
            }
            return jsonify(response), 503

        warnings = [ARTIFACT_MODE_WARNING]
        try:
            clinical_feature_names = get_bundle_feature_names("clinical")
            activity_feature_names = get_bundle_feature_names("activity_hrv")
            clinical_selected = vector_from_payload(
                data.get("clinical_selected"),
                clinical_feature_names,
                "clinical_selected",
            )
            activity_features = vector_from_payload(
                data.get("activity_hrv_features"),
                activity_feature_names,
                "activity_hrv_features",
            )
        except ValueError as exc:
            return (
                jsonify(
                    {
                        "error": str(exc),
                        "service_source": "python_inference",
                        "input_mode": "artifact_aligned",
                        "warnings": [ARTIFACT_MODE_WARNING],
                        "models_used": [],
                    }
                ),
                400,
            )

        clinical_scaled = scale_with_bundle("clinical", clinical_selected)
        activity_scaled = scale_with_bundle("activity_hrv", activity_features)

        clin_tensor = torch.FloatTensor(clinical_scaled).unsqueeze(0).to(device)
        bio_tensor = torch.FloatTensor(activity_scaled).unsqueeze(0).to(device)

        embeddings: Dict[str, torch.Tensor] = {}
        modality_logits: Dict[str, torch.Tensor] = {}
        models_used: List[str] = []

        with torch.no_grad():
            predict_modality("clinical", clin_tensor, embeddings, modality_logits, models_used, warnings)
            predict_modality("bio", bio_tensor, embeddings, modality_logits, models_used, warnings)

        pred_prob, gate_dict = compute_fusion_output(
            embeddings, modality_logits, models_used, warnings
        )

        clinical_shap = generate_shap_values(
            clinical_selected, clinical_feature_names, "clinical"
        )
        activity_shap = generate_shap_values(
            activity_features, activity_feature_names, "activity_hrv"
        )
        triggered_criteria = build_triggered_criteria([clinical_shap, activity_shap])

        response = build_response(
            pred_prob=pred_prob,
            gate_dict=gate_dict,
            shap={
                "clinical": clinical_shap,
                "activity_hrv": activity_shap,
                "eeg": [],
            },
            triggered_criteria=triggered_criteria,
            input_mode="artifact_aligned",
            models_used=models_used,
            warnings=warnings,
        )
        return jsonify(response)

    warnings = [DEMO_GATING_WARNING]
    clinical_features = process_clinical_input(data)
    bio_features = process_bio_input(data)
    eeg_signal = process_eeg_input(data)
    brain_features = process_brain_input(data)

    clin_tensor = torch.FloatTensor(clinical_features).unsqueeze(0).to(device)
    bio_tensor = torch.FloatTensor(bio_features).unsqueeze(0).to(device)
    eeg_tensor = torch.FloatTensor(eeg_signal).unsqueeze(0).unsqueeze(0).to(device)
    brain_tensor = torch.FloatTensor(brain_features).unsqueeze(0).to(device)

    embeddings: Dict[str, torch.Tensor] = {}
    modality_logits: Dict[str, torch.Tensor] = {}
    models_used: List[str] = []

    with torch.no_grad():
        predict_modality("clinical", clin_tensor, embeddings, modality_logits, models_used, warnings)
        predict_modality("bio", bio_tensor, embeddings, modality_logits, models_used, warnings)
        predict_modality("eeg", eeg_tensor, embeddings, modality_logits, models_used, warnings)
        predict_modality("brain", brain_tensor, embeddings, modality_logits, models_used, warnings)

    pred_prob, gate_dict = compute_fusion_output(embeddings, modality_logits, models_used, warnings)

    clinical_shap = generate_shap_values(
        clinical_features, DEMO_CLINICAL_FEATURE_NAMES, "clinical"
    )
    activity_shap = generate_shap_values(bio_features, BIO_FEATURE_NAMES, "activity_hrv")
    eeg_shap = generate_shap_values(eeg_signal.mean(axis=1), EEG_CHANNELS, "eeg")
    triggered_criteria = build_triggered_criteria([clinical_shap, activity_shap, eeg_shap])

    response = build_response(
        pred_prob=pred_prob,
        gate_dict=gate_dict,
        shap={
            "clinical": clinical_shap,
            "activity_hrv": activity_shap,
            "eeg": eeg_shap,
        },
        triggered_criteria=triggered_criteria,
        input_mode="demo",
        models_used=models_used,
        warnings=warnings,
    )
    return jsonify(response)


if __name__ == "__main__":
    print("=" * 65)
    print("ADHD Multimodal AI - Inference Service")
    print(f"Device: {device}")
    print("=" * 65)
    print("\nLoading models and preprocessing artifacts...")
    initialize_runtime(force=True)
    print(f"\nModalities loaded: {get_public_modalities_loaded()}")
    print(f"Artifact-aligned ready: {artifact_aligned_ready()}")
    print(f"\nStarting server on http://localhost:{SERVICE_PORT}")
    app.run(host="0.0.0.0", port=SERVICE_PORT, debug=False)
