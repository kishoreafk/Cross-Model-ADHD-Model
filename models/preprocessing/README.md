Artifact-aligned inference expects these files in this directory:

- `clinical_bundle.json`
- `activity_hrv_bundle.json`

Generate them with:

```powershell
python .\scripts\export_preprocessing_artifacts.py --hyperaktiv-dir <path-to-hyperaktiv>
```

The script also writes JSON/CSV templates for the advanced upload mode.
