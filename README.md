# 🧠 ETDD70 Feature Engineering Pipeline

> **Webcam-Based Dyslexia Detection System — Phase 1: Feature Engineering**  
> Dataset: [ETDD70 — Eye-Tracking Dyslexia Dataset](https://zenodo.org/records/13332134) · 70 subjects · 3 tasks · SMI RED-m @ 120 Hz

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Background](#dataset-background)
- [Package Contents](#package-contents)
- [Quick Start](#quick-start)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Feature Engineering Details](#feature-engineering-details)
- [Output Files Reference](#output-files-reference)
- [Using Real ETDD70 Data](#using-real-etdd70-data)
- [Connecting to Webcam Pipeline](#connecting-to-webcam-pipeline)
- [Dependencies](#dependencies)

---

## Project Overview

This package is **Phase 1** of a webcam-based dyslexia detection system. The work done here takes raw eye-tracking gaze data from the ETDD70 dataset and transforms it into a clean, normalized, model-ready feature matrix.

```
Raw gaze CSVs (x, y, timestamp)
         ↓
  I2MC Fixation & Saccade Detection
         ↓
  40 whole-task features per subject-task
  12 ROI spatial features per subject-task
         ↓
  Cross-task aggregation (mean, std, delta)
         ↓
  22 derived composite features
         ↓
  211 subject-level features  →  clean  →  RobustScaled
         ↓
  Feature selection (MI + F-test + RF)
         ↓
  30 robust features  →  X_selected.csv  →  ready for Phase 2
```

**What was built:** A fully automated pipeline that processes all 70 subjects across 3 tasks, detects fixations and saccades using an I2MC-inspired algorithm, engineers 211 features capturing reading behaviour at multiple levels of granularity, and selects the 30 most discriminative features using three independent methods.

---

## Dataset Background

| Property | Value |
|----------|-------|
| Source | ETDD70 — Zenodo DOI: [10.5281/zenodo.13332134](https://doi.org/10.5281/zenodo.13332134) |
| Subjects | 70 children (35 dyslexic, 35 non-dyslexic), ages 9–10 |
| Eye-tracker | SMI RED-m @ **120 Hz**, binocular |
| Screen | 1920 × 1080 px, viewing distance 60 cm |
| Tasks | **Syllables** · **MeaningfulText** · **PseudoText** |
| Fixation algorithm | **I2MC** — min fixation duration 40 ms |
| Language | Czech (generalisable reading patterns) |

### The Three Reading Tasks

| Task | Stimuli | Rows | Dyslexia Signal |
|------|---------|------|----------------|
| **Syllables** | 9×10 grid of Czech syllables | 10 rows | Decoding speed, letter-level fixations |
| **MeaningfulText** | 7-line passage, squirrel story | 7 lines | Comprehension-driven regressions |
| **PseudoText** | 7 lines of fictional words | 7 lines | Phonological decoding without semantics |

### Raw Gaze File Format

Each subject-task produces a CSV with columns:

| Column | Unit | Description |
|--------|------|-------------|
| `timestamp` | ms | Recording time from trial start |
| `x` | px | Gaze x-coordinate (screen space) |
| `y` | px | Gaze y-coordinate (screen space, top=0) |
| `pupil_size` | mm | Pupil diameter |
| `validity` | 0/1 | 1 = valid sample, 0 = blink/loss-of-tracking |

---

## Package Contents

```
ETDD70_Feature_Engineering_Package.zip
│
├── ETDD70_Feature_Engineering.ipynb   ← Main notebook (17 sections, 42 cells)
│
├── outputs/
│   ├── features/
│   │   ├── X_full_scaled.csv          ← 70 × 211 features (RobustScaled)
│   │   ├── X_selected.csv             ← 70 × 30 features (best subset)
│   │   ├── y_labels.csv               ← Ground-truth class labels
│   │   ├── feature_importance_report.csv  ← 211 features ranked by 3 methods
│   │   └── subject_features_raw.csv   ← Unscaled subject-level matrix
│   │
│   └── plots/
│       ├── 01_scanpaths.png           ← Raw gaze scan-path comparison
│       ├── 02_fixation_maps.png       ← Fixation bubble maps
│       ├── 03_feature_distributions.png  ← Class-wise feature histograms
│       ├── 04_effect_sizes.png        ← Cohen's d effect sizes
│       ├── 05_correlation_heatmap.png ← Feature correlation matrix
│       ├── 06_feature_importance.png  ← 3-method importance comparison
│       └── 07_pca.png                 ← PCA scatter + scree plot
│
└── README.md                          ← This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn jupyter
```

### 2. Open the notebook

```bash
jupyter notebook ETDD70_Feature_Engineering.ipynb
```

### 3. Run all cells

The notebook runs end-to-end without any real data (uses faithful synthetic data by default). To use real ETDD70 data, see [Using Real ETDD70 Data](#using-real-etdd70-data).

### 4. Use the output features for modeling

```python
import pandas as pd

X = pd.read_csv('outputs/features/X_selected.csv').drop(columns=['subject_id'])
y = pd.read_csv('outputs/features/y_labels.csv')['class_label']

# 70 subjects × 30 features, ready to go
print(X.shape, y.value_counts())
```

---

## Notebook Walkthrough

| Section | Title | Key Output |
|---------|-------|-----------|
| 1 | Imports & Configuration | Eye-tracker specs, I2MC parameters |
| 2 | Subject Labels | `dyslexia_class_label.csv` parsed (35 + 35) |
| 3 | ETDD70 File Structure | Data layout documentation |
| 4 | Gaze Simulation | Per-subject synthetic raw gaze DataFrames |
| 5 | I2MC Fixation Detection | Fixation + saccade event tables |
| 6 | Raw Gaze Visualisation | Scan-path plots `01_scanpaths.png`, `02_fixation_maps.png` |
| 7 | Whole-Task Feature Extraction | 40 features per subject-task |
| 8 | ROI Feature Extraction | 12 spatial features per subject-task |
| 9 | Full Dataset Processing | 210 rows (70 subjects × 3 tasks) |
| 10 | Cross-Task Aggregation | 191-column subject-level matrix |
| 11 | Derived Feature Engineering | +22 composite features → 211 total |
| 12 | Exploratory Data Analysis | `03_distributions.png`, `04_effect_sizes.png`, `05_heatmap.png` |
| 13 | Data Cleaning & Imputation | Missing values handled, constants removed |
| 14 | Normalization | RobustScaler applied |
| 15 | Feature Selection | 30 robust features via MI + F-test + RF |
| 16 | PCA Visualisation | `07_pca.png` |
| 17 | Final Output + Baseline CV | All CSVs saved, 5-fold ROC-AUC reported |

---

## Feature Engineering Details

### Feature Categories

#### A — Fixation Features (per task + cross-task)

| Feature | Formula | Dyslexia Direction |
|---------|---------|-------------------|
| `avg_fix_dur_ms` | mean(fixation durations) | ↑ Higher in dyslexics |
| `std_fix_dur_ms` | std(fixation durations) | ↑ More variable |
| `median_fix_dur_ms` | median(fixation durations) | ↑ |
| `n_fixations` | count of fixation events | ↑ More fixations |
| `fix_rate_per_sec` | n_fixations / trial_duration | ↑ |
| `fix_dur_skewness` | skewness of duration distribution | varies |
| `fix_dur_kurtosis` | kurtosis of duration distribution | ↑ heavier tail |
| `first_fix_dur_ms` | duration of very first fixation | ↑ |
| `total_fixation_time_ms` | sum of all fixation durations | ↑ |
| `fixation_time_ratio` | total_fix_time / total_duration | ↑ |
| `fix_x_std_px` | horizontal spread of fixations | ↓ Less coverage |
| `avg_fix_dispersion_deg` | avg within-fixation gaze spread | ↑ Noisier fixations |

**Why fixation duration matters:** Dyslexics require longer fixations because orthographic decoding is slower. They revisit characters and words more often, accumulating more and longer fixations per unit of text.

#### B — Saccade Features (per task + cross-task)

| Feature | Formula | Dyslexia Direction |
|---------|---------|-------------------|
| `n_saccades` | count of saccade events | ↑ |
| `avg_sacc_amplitude_deg` | mean(saccade amplitude in °) | ↓ Shorter saccades |
| `std_sacc_amplitude_deg` | std(saccade amplitudes) | ↑ More variable |
| `avg_sacc_velocity_deg_s` | mean(peak saccade velocity °/s) | ↓ Slower saccades |
| `n_regressions` | count(saccades moving leftward) | ↑ More re-reading |
| `regression_ratio` | n_regressions / n_saccades | ↑ |
| `progressive_ratio` | n_progressive / n_saccades | ↓ |
| `avg_prog_sacc_amp_deg` | amplitude of forward saccades | ↓ |
| `avg_regr_sacc_amp_deg` | amplitude of backward saccades | varies |
| `prog_regr_amp_ratio` | forward_amp / backward_amp | ↓ |
| `pct_horizontal_sacc` | % of saccades that are horizontal | varies |

**Why saccade amplitude matters:** Non-dyslexic readers make large forward saccades (8–10° ≈ 7–9 words skipped). Dyslexics make shorter saccades, processing fewer characters at once — a hallmark of phonological deficit.

#### C — Reading Behaviour Features

| Feature | Description | Dyslexia Direction |
|---------|-------------|-------------------|
| `total_reading_dur_ms` | Total trial time | ↑ Reads more slowly |
| `n_blink_intervals` | Number of blink events | ↑ |
| `valid_sample_ratio` | Fraction of valid gaze samples | ↓ |

#### D — ROI (Region-of-Interest) Features

| Feature | Description | Dyslexia Direction |
|---------|-------------|-------------------|
| `roi_n_visited` | Number of ROIs visited | varies |
| `roi_pct_visited` | Coverage of text regions | ↓ Skips ROIs |
| `roi_skip_rate` | 1 − pct_visited | ↑ |
| `roi_n_revisits_total` | Total re-entries into ROIs | ↑ Re-reads more |
| `roi_avg_fix_dur_mean` | Mean fixation duration across ROIs | ↑ |
| `roi_first_fix_dur_mean` | Duration of landing fixation per ROI | ↑ |
| `roi_landing_x_mean` | Where in each ROI gaze first lands | varies |

#### E — Derived Composite Features

| Feature | Formula | Dyslexia Direction |
|---------|---------|-------------------|
| `reading_efficiency` | fix_x_range / n_fixations | ↓ Less coverage per fixation |
| `cognitive_load` | avg_fix_dur / avg_sacc_amplitude | ↑ High fixation cost per saccade |
| `regression_burden` | (n_regressions × avg_fix_dur) / total_dur | ↑ |
| `fix_economy` | total_fix_time / total_dur | ↑ |
| `sacc_main_seq_residual` | log(vel) − (1 + 0.9×log(amp)) | varies (motor control proxy) |
| `complexity_effect` | feature_MeaningfulText − feature_Syllables | ↑ More difficulty scaling |
| `pseudotext_effect` | feature_PseudoText − feature_MeaningfulText | ↑ Phonological load effect |

#### F — Cross-Task Aggregates

For each of the 11 key features, three cross-task summaries are computed:

| Suffix | Description |
|--------|-------------|
| `_cross_mean` | Average across all 3 tasks |
| `_cross_std` | Variability across tasks (consistency measure) |
| `_cross_delta` | PseudoText − Syllables (complexity gradient) |

The `_cross_delta` features are especially powerful: dyslexics show disproportionately larger performance degradation as text complexity increases.

### Feature Selection Pipeline

Three independent methods are applied, and features appearing in **≥2 methods** are retained:

```
Method 1: Mutual Information (MI)     → top 30
Method 2: ANOVA F-test                → top 30    ──→  Union = 51
Method 3: Random Forest Importance    → top 30          Robust (≥2) = 30
```

The 30 robust features are saved in `X_selected.csv` and are recommended for all downstream modeling.

### Normalization

**RobustScaler** is used (not StandardScaler) because:
- Eye-tracking data contains outliers from blinks and tracking loss
- With only N=70 subjects, outliers are highly influential
- RobustScaler uses median and IQR — resistant to extremes

```python
X_scaled = (X - median(X)) / IQR(X)
```

---

## Output Files Reference

### `X_selected.csv` — Primary Model Input

- Shape: **70 rows × 31 columns** (including `subject_id`)
- 30 features selected as most discriminative across all 3 selection methods
- RobustScaled (mean ≈ 0, IQR ≈ 1)
- No missing values

### `X_full_scaled.csv` — Full Feature Matrix

- Shape: **70 rows × 212 columns** (including `subject_id`)
- All 211 engineered features, scaled
- Use for ablation studies or alternative selection strategies

### `y_labels.csv`

```
subject_id, class_label, label
1003, 0, non-dyslexic
1009, 1, dyslexic
...
```

### `feature_importance_report.csv`

Columns: `feature`, `MI_score`, `F_score`, `p_value`, `RF_importance`, `in_top_MI`, `in_top_F`, `in_top_RF`, `selection_votes`

Sorted by `selection_votes` (3 = top in all methods → most reliable).

### `subject_features_raw.csv`

Unscaled subject-level feature matrix — useful for interpretability and domain adaptation (fit scaler on this, apply to webcam features).

---

## Using Real ETDD70 Data

1. Download `data.zip` from [Zenodo](https://zenodo.org/records/13332134)

2. Extract it:
   ```bash
   unzip data.zip -d etdd70_data/
   ```

3. Your folder should look like:
   ```
   etdd70_data/
   ├── Syllables/
   │   ├── 1003/raw_gaze.csv
   │   ├── 1009/raw_gaze.csv
   │   └── ...
   ├── MeaningfulText/
   └── PseudoText/
   ```

4. In the notebook, update cell 1:
   ```python
   DATA_ROOT = Path("./etdd70_data")   # ← set this to your extraction path
   ```

5. Run all cells. The `load_or_simulate_gaze()` function will automatically load real files where they exist and fall back to simulation only for any missing ones.

> **No other code changes needed.** The entire downstream pipeline (fixation detection, feature extraction, scaling, selection) runs identically on real and simulated data.

---

## Connecting to Webcam Pipeline

This feature matrix is designed as the **source domain** for domain adaptation to webcam-based gaze estimation.

### Domain Adaptation Strategy

```python
# Step 1: Fit scaler on ETDD70 features (done in this notebook)
scaler = RobustScaler()
X_etdd70_scaled = scaler.fit_transform(X_etdd70_raw)

# Step 2: Extract same features from webcam (MediaPipe landmarks)
X_webcam_raw = extract_webcam_features(video_stream)

# Step 3: Apply same scaler to webcam features
X_webcam_scaled = scaler.transform(X_webcam_raw)

# Step 4: Optional CORAL alignment
X_webcam_adapted = coral_align(X_webcam_scaled, X_etdd70_scaled)

# Step 5: Predict
y_pred = trained_model.predict(X_webcam_adapted)
```

### Webcam-to-ETDD70 Feature Mapping

| ETDD70 Feature | Webcam Approximation |
|----------------|---------------------|
| `avg_fix_dur_ms` | Cluster low-velocity gaze points (MediaPipe iris landmarks) |
| `n_fixations` | Count stable gaze clusters per session |
| `regression_ratio` | Track leftward head/gaze direction reversals |
| `avg_sacc_amplitude_deg` | Angular displacement between fixation centroids |
| `roi_n_revisits` | Count gaze re-entries into screen regions |
| `total_reading_dur_ms` | Session duration |

> **Note:** Webcam gaze is noisy (±2–5° vs ±0.1° for SMI RED-m). Fixation detection thresholds should be relaxed: min fixation duration 80–120 ms, velocity threshold 15–20°/s.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Numerical computation |
| `pandas` | ≥2.0 | DataFrame operations |
| `scipy` | ≥1.10 | Statistics, signal processing |
| `matplotlib` | ≥3.7 | Plotting |
| `seaborn` | ≥0.12 | Statistical visualisation |
| `scikit-learn` | ≥1.3 | ML, feature selection, scaling |

Install all at once:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn jupyter
```

---

## Project Roadmap

```
✅ Phase 1 — Feature Engineering (this package)
   ETDD70 gaze data → 211 features → 30 selected → model-ready CSVs

⬜ Phase 2 — Classification
   Random Forest / XGBoost on X_selected.csv
   BiLSTM on raw gaze sequences (per-task)
   5-fold stratified CV, ROC-AUC target >0.85

⬜ Phase 3 — Webcam Eye Tracking
   MediaPipe FaceMesh iris landmark extraction
   Gaze estimation from 2D landmarks
   Approximate fixation/saccade detection

⬜ Phase 4 — Domain Adaptation
   CORAL / MMD alignment: webcam → ETDD70 feature space
   Fine-tuning with small webcam dyslexia samples

⬜ Phase 5 — Deployment
   FastAPI backend for real-time inference
   React frontend with reading task interface
   5-minute screening session → prediction
```


