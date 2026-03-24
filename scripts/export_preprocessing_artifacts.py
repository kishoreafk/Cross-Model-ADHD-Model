

import argparse
import csv
import json
import os
import re
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def serialize_scaler(scaler: StandardScaler) -> Dict[str, List[float] | int]:
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": int(scaler.n_samples_seen_),
    }


def safe_float(value: float) -> float:
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def ts_stats(path: str, prefix: str) -> Dict[int, Dict[str, float]]:
    feats: Dict[int, Dict[str, float]] = {}
    if not os.path.exists(path):
        return feats

    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".csv"):
            continue
        match = re.search(r"(\d+)", filename)
        if not match:
            continue
        patient_id = int(match.group(1))
        try:
            df_ts = None
            for separator in [";", ","]:
                try:
                    df_ts = pd.read_csv(
                        os.path.join(path, filename),
                        sep=separator,
                        skiprows=2,
                    )
                    if df_ts.shape[1] > 1:
                        break
                except Exception:
                    pass
            if df_ts is None:
                continue

            numeric_columns = df_ts.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                continue

            signal = df_ts[numeric_columns[-1]].dropna().values.astype(float)
            if len(signal) < 10:
                continue

            feats[patient_id] = {
                f"{prefix}mean": safe_float(np.mean(signal)),
                f"{prefix}std": safe_float(np.std(signal)),
                f"{prefix}median": safe_float(np.median(signal)),
                f"{prefix}range": safe_float(np.ptp(signal)),
                f"{prefix}skew": safe_float(skew(signal)),
                f"{prefix}kurtosis": safe_float(kurtosis(signal)),
            }
            if "hr" in prefix:
                feats[patient_id][f"{prefix}rmssd"] = safe_float(
                    np.sqrt(np.mean(np.diff(signal) ** 2))
                )
        except Exception:
            continue

    return feats


def extract_rich_ts_features(filepath: str, prefix: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    try:
        df_ts = None
        for separator in [";", ","]:
            try:
                df_ts = pd.read_csv(filepath, sep=separator, skiprows=2)
                if df_ts.shape[1] > 1:
                    break
            except Exception:
                pass
        if df_ts is None:
            return feats

        numeric_columns = df_ts.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return feats

        signal = df_ts[numeric_columns[-1]].dropna().values.astype(float)
        if len(signal) < 20:
            return feats

        feats[f"{prefix}mean"] = safe_float(np.mean(signal))
        feats[f"{prefix}std"] = safe_float(np.std(signal))
        feats[f"{prefix}median"] = safe_float(np.median(signal))
        feats[f"{prefix}range"] = safe_float(np.ptp(signal))
        feats[f"{prefix}skew"] = safe_float(skew(signal))
        feats[f"{prefix}kurtosis"] = safe_float(kurtosis(signal))
        feats[f"{prefix}iqr"] = safe_float(np.percentile(signal, 75) - np.percentile(signal, 25))
        feats[f"{prefix}p10"] = safe_float(np.percentile(signal, 10))
        feats[f"{prefix}p90"] = safe_float(np.percentile(signal, 90))
        feats[f"{prefix}cv"] = safe_float(np.std(signal) / (np.mean(signal) + 1e-8))

        above = (signal > np.mean(signal)).astype(int)
        feats[f"{prefix}transitions"] = safe_float(np.sum(np.diff(above) != 0))

        if "act" in prefix:
            threshold = np.percentile(signal, 25)
            active = (signal > threshold).astype(int)
            starts = np.where(np.diff(np.concatenate([[0], active])) == 1)[0]
            ends = np.where(np.diff(np.concatenate([active, [0]])) == -1)[0]
            bouts = ends - starts + 1
            if len(bouts) > 0:
                feats[f"{prefix}n_bouts"] = safe_float(len(bouts))
                feats[f"{prefix}mean_bout_len"] = safe_float(np.mean(bouts))
                feats[f"{prefix}short_bout_frac"] = safe_float(
                    np.sum(bouts <= 3) / (len(bouts) + 1e-8)
                )

        if "hr" in prefix:
            diffs = np.diff(signal)
            feats[f"{prefix}rmssd"] = safe_float(np.sqrt(np.mean(diffs**2)))
            feats[f"{prefix}sdnn"] = safe_float(np.std(signal))
            feats[f"{prefix}pnn50_proxy"] = safe_float(
                np.sum(np.abs(diffs) > np.percentile(np.abs(diffs), 50)) / (len(diffs) + 1e-8)
            )
            feats[f"{prefix}lf_hf_ratio"] = 0.0
    except Exception:
        return {}

    return feats


def build_clinical_bundle(hyperaktiv_dir: str) -> Dict[str, object]:
    patient_info = pd.read_csv(os.path.join(hyperaktiv_dir, "patient_info.csv"), sep=";")
    features_df = pd.read_csv(os.path.join(hyperaktiv_dir, "features.csv"), sep=";")
    cpt_path = os.path.join(hyperaktiv_dir, "CPT_II_ConnersContinuousPerformanceTest.csv")
    cpt_df = pd.read_csv(cpt_path, sep=";") if os.path.exists(cpt_path) else None

    for dataframe in [patient_info, features_df]:
        dataframe["ID"] = dataframe["ID"].astype(int)

    merged = pd.merge(
        features_df,
        patient_info[["ID", "ADHD", "AGE", "SEX"]],
        on="ID",
        how="inner",
    )

    if cpt_df is not None:
        cpt_df["ID"] = cpt_df["ID"].astype(int)
        score_cols = [
            col
            for col in cpt_df.columns
            if any(key in col for key in ["TScore", "Score", "Percent", "Index", "Confidence"])
        ]
        trial_cols = [col for col in cpt_df.columns if col.startswith("Trial")]
        if trial_cols:
            rt = cpt_df[trial_cols].apply(pd.to_numeric, errors="coerce")
            cpt_feats = pd.DataFrame({"ID": cpt_df["ID"]})
            cpt_feats["cpt_rt_mean"] = rt.mean(axis=1)
            cpt_feats["cpt_rt_std"] = rt.std(axis=1)
            cpt_feats["cpt_rt_cv"] = rt.std(axis=1) / (rt.mean(axis=1) + 1e-8)
            n_trials = len(trial_cols)
            if n_trials >= 6:
                third = n_trials // 3
                cpt_feats["cpt_vigilance_decrement"] = rt.iloc[:, -third:].mean(axis=1) - rt.iloc[
                    :, :third
                ].mean(axis=1)
            if score_cols:
                cpt_feats = pd.concat([cpt_feats, cpt_df[score_cols]], axis=1)
            merged = pd.merge(merged, cpt_feats, on="ID", how="left")

    for feat_dict, prefix in [
        (ts_stats(os.path.join(hyperaktiv_dir, "activity_data"), "act_"), "act_"),
        (ts_stats(os.path.join(hyperaktiv_dir, "hrv_data"), "hr_"), "hr_"),
    ]:
        if feat_dict:
            df_feat = pd.DataFrame.from_dict(feat_dict, orient="index").reset_index()
            df_feat.rename(columns={"index": "ID"}, inplace=True)
            df_feat["ID"] = df_feat["ID"].astype(int)
            merged = pd.merge(merged, df_feat, on="ID", how="left")

    exclude = {"ID", "ADHD", "AGE", "SEX", "EDUCATION", "MEDICATION", "study_id"}
    feat_cols = [
        col
        for col in merged.columns
        if col not in exclude
        and merged[col].dtype in ["float64", "int64", "float32", "int32"]
        and merged[col].nunique() > 1
    ]

    X_clin_raw = merged[feat_cols].values.astype(float)
    y_clinical = merged["ADHD"].values.astype(int)
    X_clin_imp = SimpleImputer(strategy="median").fit_transform(X_clin_raw)
    X_clin_imp = np.nan_to_num(X_clin_imp, nan=0, posinf=0, neginf=0)

    k_feats = min(30, X_clin_imp.shape[1])
    selector = SelectKBest(f_classif, k=k_feats)
    X_clin_sel = selector.fit_transform(X_clin_imp, y_clinical)
    selected_feature_names = [feat_cols[i] for i in selector.get_support(indices=True)]

    X_tr_c, _, y_tr_c, _ = train_test_split(
        X_clin_sel,
        y_clinical,
        test_size=0.2,
        stratify=y_clinical,
        random_state=42,
    )
    sc_c = StandardScaler()
    sc_c.fit(X_tr_c)

    return {
        "bundle_type": "clinical_selected_features",
        "source_notebook": "adha-new-method (2).ipynb",
        "input_key": "clinical_selected",
        "selected_feature_names": selected_feature_names,
        "feature_names": selected_feature_names,
        "scaler": serialize_scaler(sc_c),
        "metadata": {
            "subject_count": int(len(y_clinical)),
            "selected_feature_count": int(len(selected_feature_names)),
            "random_state": 42,
            "generated_at": datetime.now().isoformat(),
        },
    }


def build_activity_hrv_bundle(hyperaktiv_dir: str) -> Dict[str, object]:
    patient_info = pd.read_csv(os.path.join(hyperaktiv_dir, "patient_info.csv"), sep=";")
    patient_info["ID"] = patient_info["ID"].astype(int)
    y_bio_all = patient_info.set_index("ID")["ADHD"].to_dict()

    all_bio: Dict[int, Dict[str, float]] = {}
    for directory, prefix in [
        (os.path.join(hyperaktiv_dir, "activity_data"), "act_"),
        (os.path.join(hyperaktiv_dir, "hrv_data"), "hr_"),
    ]:
        if not os.path.exists(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".csv"):
                continue
            match = re.search(r"(\d+)", filename)
            if not match:
                continue
            patient_id = int(match.group(1))
            feats = extract_rich_ts_features(os.path.join(directory, filename), prefix)
            if feats:
                all_bio.setdefault(patient_id, {}).update(feats)

    bio_rows = []
    for patient_id, feats in all_bio.items():
        if patient_id in y_bio_all:
            row = dict(feats)
            row["ID"] = patient_id
            row["ADHD"] = y_bio_all[patient_id]
            bio_rows.append(row)

    bio_df = pd.DataFrame(bio_rows).fillna(0)
    if bio_df.empty:
        raise RuntimeError("No activity_hrv rows were extracted from the dataset")

    bio_cols = [col for col in bio_df.columns if col not in ["ID", "ADHD"]]
    X_bio_raw = bio_df[bio_cols].values.astype(float)
    X_bio_raw = np.nan_to_num(X_bio_raw, nan=0, posinf=0, neginf=0)
    y_bio = bio_df["ADHD"].values.astype(int)

    X_tr_b, _, y_tr_b, _ = train_test_split(
        X_bio_raw,
        y_bio,
        test_size=0.2,
        stratify=y_bio,
        random_state=42,
    )
    sc_b = StandardScaler()
    sc_b.fit(X_tr_b)

    return {
        "bundle_type": "activity_hrv_features",
        "source_notebook": "adha-new-method (2).ipynb",
        "input_key": "activity_hrv_features",
        "feature_names": bio_cols,
        "scaler": serialize_scaler(sc_b),
        "metadata": {
            "subject_count": int(len(y_bio)),
            "feature_count": int(len(bio_cols)),
            "random_state": 42,
            "generated_at": datetime.now().isoformat(),
        },
    }


def write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_template_csv(path: str, feature_names: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feature_names)
        writer.writeheader()
        writer.writerow({name: 0 for name in feature_names})


def write_template_json(path: str, feature_names: List[str]) -> None:
    payload = {name: 0 for name in feature_names}
    write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export preprocessing bundles for inference.")
    parser.add_argument(
        "--hyperaktiv-dir",
        required=True,
        help="Path to the HYPERAKTIV dataset root used by the notebook.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "preprocessing"),
        help="Directory where preprocessing bundles will be written.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    clinical_bundle = build_clinical_bundle(args.hyperaktiv_dir)
    activity_bundle = build_activity_hrv_bundle(args.hyperaktiv_dir)

    clinical_bundle_path = os.path.join(args.output_dir, "clinical_bundle.json")
    activity_bundle_path = os.path.join(args.output_dir, "activity_hrv_bundle.json")
    write_json(clinical_bundle_path, clinical_bundle)
    write_json(activity_bundle_path, activity_bundle)

    clinical_feature_names = clinical_bundle["selected_feature_names"]
    activity_feature_names = activity_bundle["feature_names"]
    write_template_csv(
        os.path.join(args.output_dir, "clinical_selected_template.csv"),
        clinical_feature_names,
    )
    write_template_json(
        os.path.join(args.output_dir, "clinical_selected_template.json"),
        clinical_feature_names,
    )
    write_template_csv(
        os.path.join(args.output_dir, "activity_hrv_template.csv"),
        activity_feature_names,
    )
    write_template_json(
        os.path.join(args.output_dir, "activity_hrv_template.json"),
        activity_feature_names,
    )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "hyperaktiv_dir": os.path.abspath(args.hyperaktiv_dir),
        "files": {
            "clinical_bundle": os.path.basename(clinical_bundle_path),
            "activity_hrv_bundle": os.path.basename(activity_bundle_path),
            "clinical_selected_template_csv": "clinical_selected_template.csv",
            "clinical_selected_template_json": "clinical_selected_template.json",
            "activity_hrv_template_csv": "activity_hrv_template.csv",
            "activity_hrv_template_json": "activity_hrv_template.json",
        },
    }
    write_json(os.path.join(args.output_dir, "manifest.json"), manifest)

    print(f"Saved clinical bundle: {clinical_bundle_path}")
    print(f"Saved activity_hrv bundle: {activity_bundle_path}")
    print(f"Templates and manifest written to: {args.output_dir}")


if __name__ == "__main__":
    main()
