# BehaviorShield

**Real-Time Privacy Guardianship in XR Metaverses via AI-Driven Behavioral Analytics**

BehaviorShield is a behavioral-analytics pipeline that detects privacy-invasive
behavior in VR/XR sessions directly from head-pose motion — without relying on
encryption-breakable identifiers or synthetic-only training data. It is trained
and evaluated on **real VR motion capture** (WhoIsAlyx) and cross-domain
augmented with **real-world egocentric head trajectories** (Project Aria
Everyday Activities), then validated with rigorous **Leave-One-Subject-Out
(LOSO)** cross-validation and explained with **SHAP**.

> **Conference Presentation (e-Poster):** *Real-Time Privacy Guardianship in
> XR Metaverses via AI-Driven Behavioral Analytics.* Presented at the **IEEE
> 5th International Conference on Intelligent Reality (ICIR 2026)**,
> University of Pisa, Italy (June 2026). Detects 3 VR attack types (tracking,
> impersonation, spatial profiling) using 19 engineered motion features,
> achieving 92% accuracy and 0.976 AUC on 993,636 frames.
>

> The e-poster (`BehaviorShield_ePoster.pptx`)
> is available in this repo.*

---

## The problem

Embodied avatars turn continuous motion into a new attack surface. Encryption
and access control can't stop an adversary from *inferring* things purely from
observable kinematics. BehaviorShield targets three concrete threat classes:

| Threat | Attacker behavior | Objective |
|---|---|---|
| **Avatar Tracking** | Blends own position toward a lagged copy of the target's trajectory, with burst pursuit intervals | Infer real-world location/routine |
| **Behavioral Impersonation** | Interpolates head rotation toward a victim's signature, fidelity degrading over time | Deceive social/security systems that trust motion signatures |
| **Spatial Profiling** | Systematic grid traversal with dwell pauses and backtracking | Build a behavioral dossier of avatar habits |

## Results at a glance

| Metric | Value |
|---|---|
| Best-fold ROC-AUC (LOSO) | **0.976** |
| Player-level LOSO (9 folds) | AUC 0.967 ± 0.009, F1 0.885 ± 0.017 |
| Date-level LOSO (4 folds) | AUC 0.963 ± 0.007, F1 0.836 ± 0.042 |
| Threat recall / precision | 91.1% / 88.6% |
| Accuracy / F1 | 92.0% / 89.9% |
| False positive rate | 7.5% |
| LightGBM inference latency | 0.85 ms (P99 = 1.2 ms) — fits a 90 fps VR frame budget |
| Annotated frames | 993,636 (60% benign / 40% threat) |

Full per-fold tables, ablations, and baseline comparisons are summarized in
the e-poster (`BehaviorShield_ePoster.pptx`) and reproduced by the
pipeline's output figures.

## How it works

```
WhoIsAlyx (HuggingFace)          Project Aria AEA
9 players, 596k frames    143 sequences, 1.8M frames
HTC Vive @ 90fps           6-DoF SLAM head pose
        │                            │
        └───────────► Feature Extraction ◄───────────┘
   motion_speed · head_rotation_rate · pose_variance
   spatial_proximity · interaction_freq · movement_entropy
   + 12 rolling 3s mean/std statistics  (19 features)
                        │
        ┌───────────────┴───────────────┐
    Benign pool                     Threat pool
  (596k XR + 119k AEA)      (Tracking / Impersonation / Profiling,
                              ~40% of final dataset)
                        │
        Correlation filter (ρ < 0.9) → SMOTE (k=5) → Random Forest
                        │
   Player-level LOSO (9 folds) · Date-level LOSO (4 folds)
   GridSearchCV hyperparameter tuning on the best fold
                        │
        Threshold selection · SHAP explainability
        Baseline comparison · Ablation study
        Adaptive-adversary (mimicry) evasion test
```

### Feature engineering

| Feature | Description | Window |
|---|---|---|
| `motion_speed` | Normalized Euclidean displacement per frame | frame |
| `head_rotation_rate` | Angular velocity from quaternion change | frame |
| `pose_variance` | Std. dev. of position magnitude | 30 frames |
| `spatial_proximity` | Inverse distance from session centroid | frame |
| `interaction_freq` | Zero-crossing rate of motion speed | 90 frames |
| `movement_entropy` | Spatial entropy over a discretized grid | 270 frames |
| `time_since_last` | Inter-frame timing | frame |

Each of the 6 base features also gets a rolling 3-second mean and standard
deviation, for 19 total features (18 after correlation filtering at ρ > 0.9).

### Evaluation protocol

- **Player-level LOSO**: each of the 9 real sessions is held out in turn.
- **Date-level LOSO**: all players recorded on a given date are held out
  together (4 folds) — a stricter test for recording-session confounds.
- Threat frames are split by `primary_sid` (the session an attack was
  *generated from*), not just `session_id`, so no threat sample derived from
  a held-out session leaks into training.
- Cross-domain AEA frames are injected only into training folds, capped at
  20% of the benign count, as out-of-distribution augmentation — a KS-test
  confirms AEA is domain-distinct from VR motion (p < 0.001), which is the
  expected and desired outcome, not a validation failure.

### Model comparison

| Model | AUC | F1 | Recall | Latency (ms) |
|---|---|---|---|---|
| Random Forest ★ (interpretable, SHAP) | 0.976 | 0.899 | 0.911 | 61.1 |
| **LightGBM** (deployed) | 0.982 | 0.909 | 0.941 | **0.85** |
| XGBoost | 0.982 | 0.897 | 0.964 | 0.67 |
| MLP | 0.947 | 0.848 | 0.907 | 0.27 |

Random Forest is used for training/explainability; LightGBM is recommended
for real-time deployment since Random Forest's latency exceeds the ~10 ms
budget for 90 fps VR.

### Explainability (SHAP)

`movement_entropy_std3s` (the rolling 3-second standard deviation of spatial
entropy) is the top feature across **all three threat types**, and the top-3
rolling features account for 58.7% of total mean |SHAP|. An ablation study
confirms rolling temporal statistics are essential — AUC rises from 0.70
(instantaneous features only) to 0.98 once they're added.

---

## Repository structure

```
.
├── main.py                      # end-to-end training + evaluation script
├── requirement.txt              # Python dependencies
├── BehaviorShield_ePoster.pptx  # ICIR 2026 e-poster; full manuscript unpublished
└── outputs/                     # generated figures (ROC, SHAP, ablation, ...)
```

## Setup

```bash
pip install -r requirement.txt
```

Requires a HuggingFace account/token for the WhoIsAlyx dataset:

```bash
huggingface-cli login
# or: export HF_TOKEN=your_token_here
```

Project Aria AEA sequences are expected under `AEA_ROOT` (see below), each
containing `mps/slam/closed_loop_trajectory.csv`. Download instructions are
in the [AEA dataset repo](https://github.com/facebookresearch/Aria-Everyday-Activities-Dataset).

## Usage

```bash
export OUTPUT_DIR=./outputs
export AEA_ROOT=./data/aea_data
python main.py
```

This will:
1. Load WhoIsAlyx sessions and AEA sequences.
2. Build the labeled benign/threat dataset with rolling features.
3. Run player-level and date-level LOSO cross-validation.
4. Tune a Random Forest via grid search on the best fold.
5. Produce evaluation figures (ROC curve, correlation matrix, SHAP summary,
   baseline comparison, ablation study, evasion-robustness curve) in
   `OUTPUT_DIR`.

If no AEA data is available locally, the pipeline still runs — AEA
augmentation and cross-domain validation are automatically skipped.

## Datasets

| Dataset | Role | Source |
|---|---|---|
| **WhoIsAlyx** | Real VR benign motion (9 players, 4 sessions, 596,182 frames @ 90 fps, HTC Vive) | [HuggingFace: cschell/xr-motion-dataset-catalogue](https://huggingface.co/datasets/cschell/xr-motion-dataset-catalogue) |
| **Project Aria Everyday Activities (AEA)** | Real-world out-of-domain benign augmentation (143 sequences, 6-DoF SLAM head pose) | [Aria Everyday Activities](https://arxiv.org/abs/2402.13349) |

Threat data (Tracking, Impersonation, Profiling) is synthetically generated
from the real WhoIsAlyx sessions using the attacker models described above —
no separate threat dataset is required.

## Limitations

- **Session confound**: date-level LOSO F1 drops for the cohort where all
  four same-date players are withheld simultaneously, indicating some
  reliance on session-specific patterns.
- **Mild overfitting**: train AUC ≈ 1.0 vs. test AUC ≈ 0.976 on Random Forest.
- **Simulated threats**: attack patterns are algorithmically generated and
  may not fully capture real adversarial behavior.
- **Deployment**: Random Forest inference latency (~61 ms) exceeds the VR
  frame budget; use LightGBM for real-time deployment.

## Future work

- Red-team adversarial data collection (real attackers, not simulated).
- Live deployment with streaming inference.
- Federated learning for cross-platform generalization.
- Transformer-based temporal models for longer-range dependencies.


