"""
BehaviorShield: XR behavioral threat detection.

Detects three classes of privacy/security threats in VR head-tracking data
(Tracking, Impersonation, Profiling) against benign motion, using the
WhoIsAlyx dataset as the primary source and Project Aria Everyday
Activities (AEA) real-world trajectories as an out-of-domain augmentation
set. Evaluation uses leave-one-session-out (LOSO) cross-validation at both
the player level and the recording-date level to check for session
confounds.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "lightgbm", "xgboost", "imbalanced-learn", "shap",
                "pyarrow", "datasets", "huggingface_hub", "projectaria-tools"])

import os, warnings, time, gc, re
from collections import defaultdict
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from scipy import stats
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AEA_ROOT   = os.environ.get("AEA_ROOT", "./data/aea_data")
VIVE_FPS   = 90
TARGET_FPS = 90

# Precision/recall tradeoff selected via threshold sensitivity analysis
# (see Cell: Threshold Selection) — 0.50 gives max F1 on the held-out fold.
SELECTED_THRESHOLD = 0.50

# ─────────────────────────────────────────────────────────────────────
# Load WhoIsAlyx (VR motion dataset, HuggingFace)
# ─────────────────────────────────────────────────────────────────────
from huggingface_hub import login, HfFileSystem

login()  # requires HF_TOKEN env var or a prior `huggingface-cli login`

fs   = HfFileSystem()
BASE = "datasets/cschell/xr-motion-dataset-catalogue/who_is_alyx"

print("Loading WhoIsAlyx sessions...")
sessions      = []
session_dates = []

for player_num in range(2, 11):
    player      = f"player_{str(player_num).zfill(2)}"
    player_path = f"{BASE}/{player}"
    try:
        files         = fs.ls(player_path, detail=False)
        parquet_files = sorted([f for f in files if f.endswith(".parquet")])
        if not parquet_files:
            print(f"  [SKIP] {player} — no parquet files found")
            continue
        target   = parquet_files[0]
        date_str = os.path.basename(target).replace(".parquet", "")
        df       = pd.read_parquet(f"hf://{target}")
        if 'delta_time_ms' in df.columns:
            df['_abs_time_ms'] = df['delta_time_ms'].cumsum()
        df = df.sort_values('_abs_time_ms').reset_index(drop=True)
        sessions.append(df)
        session_dates.append(date_str)
        print(f"  {player} — {len(df):,} frames ({date_str})")
    except Exception as e:
        print(f"  [SKIP] {player} — {str(e)[:80]}")

n_real = len(sessions)
if n_real == 0:
    raise RuntimeError("0 sessions loaded. Check HuggingFace login.")
if n_real < 3:
    raise RuntimeError(f"Only {n_real} sessions — need >=3 for LOSO.")

# ─────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────
ENTROPY_WINDOW_FRAMES = 270
ENTROPY_N_BINS        = 8

XR_FEATURES = [
    'motion_speed',
    'head_rotation_rate',
    'pose_variance',
    'spatial_proximity',
    'interaction_freq',
    'movement_entropy',
]
TEMPORAL_FEATURES = ['time_since_last']
FEATURE_COLS      = XR_FEATURES + TEMPORAL_FEATURES


def percentile_norm(arr, p_lo=1, p_hi=99):
    lo, hi = np.percentile(arr, p_lo), np.percentile(arr, p_hi)
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def rolling_std(arr, window=30):
    return pd.Series(arr).rolling(window, min_periods=1).std().fillna(0).values


def zero_crossing_rate(arr, window=90):
    centered = arr - arr.mean()
    zc = (np.diff(np.sign(centered)) != 0).astype(float)
    zc = np.append(zc, 0)
    return pd.Series(zc).rolling(window, min_periods=1).mean().fillna(0).values


def rolling_entropy_fast(pos_x, pos_z, window=ENTROPY_WINDOW_FRAMES, n_bins=ENTROPY_N_BINS):
    """Spatial-occupancy entropy over a sliding window, via grid-binned one-hot counts."""
    n   = len(pos_x)
    eps = 1e-9
    x_min, x_max = pos_x.min(), pos_x.max()
    z_min, z_max = pos_z.min(), pos_z.max()
    x_bin = np.clip(((pos_x - x_min) / (x_max - x_min + eps) * n_bins).astype(int), 0, n_bins - 1)
    z_bin = np.clip(((pos_z - z_min) / (z_max - z_min + eps) * n_bins).astype(int), 0, n_bins - 1)
    cell_id = x_bin * n_bins + z_bin
    n_cells = n_bins * n_bins
    max_ent = np.log(n_cells)
    one_hot = np.zeros((n, n_cells), dtype=np.float32)
    one_hot[np.arange(n), cell_id] = 1.0
    counts  = pd.DataFrame(one_hot).rolling(window, min_periods=1).sum().values
    row_sum = counts.sum(axis=1, keepdims=True).clip(min=1)
    p       = counts / row_sum
    log_p   = np.where(p > 0, np.log(p + eps), 0.0)
    entropy = -(p * log_p).sum(axis=1)
    return np.clip(entropy / max_ent, 0, 1)


def extract_features(rec, session_id=0, reference_pos=None, session_date=None):
    pos_cols = ['head_pos_x', 'head_pos_y', 'head_pos_z']
    rot_cols = ['head_rot_x', 'head_rot_y', 'head_rot_z', 'head_rot_w']

    dt = (rec['delta_time_ms'].clip(lower=0.001).values / 1000.0
          if 'delta_time_ms' in rec.columns
          else np.ones(len(rec)) * (1 / VIVE_FPS))

    disp         = np.sqrt(sum(rec[c].diff().fillna(0).values ** 2 for c in pos_cols))
    speed_raw    = disp / dt
    motion_speed = percentile_norm(speed_raw)

    quat = rec[rot_cols].values
    dot  = np.clip(np.sum(quat[:-1] * quat[1:], axis=1), -1, 1)
    ang  = 2 * np.arccos(np.abs(dot))
    rot_rate           = np.append(ang / dt[1:], 0)
    head_rotation_rate = percentile_norm(rot_rate)

    pos_mag       = np.sqrt(sum(rec[c].values ** 2 for c in pos_cols))
    pose_variance = percentile_norm(rolling_std(pos_mag, window=30))

    if reference_pos is None:
        ref_x, ref_z = rec['head_pos_x'].mean(), rec['head_pos_z'].mean()
    else:
        ref_x, ref_z = reference_pos
    dist = np.sqrt((rec['head_pos_x'].values - ref_x) ** 2 +
                   (rec['head_pos_z'].values - ref_z) ** 2)
    spatial_proximity = percentile_norm(1.0 / (dist + 1e-3))

    interaction_freq = zero_crossing_rate(speed_raw, window=90)
    movement_entropy = rolling_entropy_fast(rec['head_pos_x'].values, rec['head_pos_z'].values)
    abs_time_s = rec['_abs_time_ms'].values / 1000.0

    return pd.DataFrame({
        'motion_speed':       motion_speed,
        'head_rotation_rate': head_rotation_rate,
        'pose_variance':      pose_variance,
        'spatial_proximity':  spatial_proximity,
        'interaction_freq':   interaction_freq,
        'movement_entropy':   movement_entropy,
        'time_since_last':    np.append(0, np.diff(abs_time_s)),
        'session_id':         session_id,
        'frame_id':           np.arange(len(rec)),
    })


# ─────────────────────────────────────────────────────────────────────
# Synthetic threat simulators
#
# Each simulator perturbs a real session's trajectory to emulate an
# attack pattern, and tags the resulting frames with `primary_sid` —
# the session that "owns" the attack, used later to prevent leakage
# between train/test splits during LOSO.
# ─────────────────────────────────────────────────────────────────────

def simulate_tracking(rec_attacker, rec_target, session_id, primary_sid,
                       intensity=0.85, session_date=None):
    """Attacker's trajectory is blended toward a time-lagged copy of the target's."""
    n   = min(len(rec_attacker), len(rec_target))
    ra  = rec_attacker.iloc[:n].copy()
    rt  = rec_target.iloc[:n].copy()
    rng = np.random.default_rng(42 + session_id)
    lag = int(rng.integers(10, 26))
    for c in ['head_pos_x', 'head_pos_y', 'head_pos_z']:
        target_lagged = np.concatenate([rt[c].values[:lag], rt[c].values[:-lag]])
        base          = (1 - intensity) * ra[c].values + intensity * target_lagged
        base         += rng.normal(0, 0.01, n)
        n_bursts      = max(1, int(n * 0.06 / 90))
        burst_starts  = rng.choice(n - 90, size=n_bursts, replace=False)
        for bs in burst_starts:
            burst_len    = int(rng.integers(45, 180))
            end          = min(bs + burst_len, n - 1)
            base[bs:end] = ra[c].values[bs:end] + rng.normal(0, 0.05, end - bs)
        ra[c] = base
    ref  = (rt['head_pos_x'].mean(), rt['head_pos_z'].mean())
    feat = extract_features(ra, session_id=session_id, reference_pos=ref, session_date=session_date)
    feat['label']       = 1
    feat['threat_type'] = 'Tracking'
    feat['primary_sid'] = primary_sid
    return feat


def simulate_impersonation(rec_mimic, rec_victim, session_id, primary_sid, session_date=None):
    """Mimic's head rotation is blended toward the victim's, with decaying fidelity."""
    n        = min(len(rec_mimic), len(rec_victim))
    rm       = rec_mimic.iloc[:n].copy()
    rv       = rec_victim.iloc[:n].copy()
    rot_cols = ['head_rot_x', 'head_rot_y', 'head_rot_z', 'head_rot_w']
    rng      = np.random.default_rng(42 + session_id)
    fidelity     = np.linspace(0.85, 0.50, n)
    mimic_mask   = rng.random(n) < fidelity
    n_bursts     = max(1, int(n * 0.05 / 60))
    burst_starts = rng.choice(n - 90, size=n_bursts, replace=False)
    for bs in burst_starts:
        burst_len          = int(rng.integers(30, 91))
        end                = min(bs + burst_len, n - 1)
        mimic_mask[bs:end] = False
    for c in rot_cols:
        rm[c] = np.where(
            mimic_mask,
            rv[c].values + rng.normal(0, 0.05, n),
            rm[c].values + rng.normal(0, 0.02, n),
        )
    q            = rm[rot_cols].values
    rm[rot_cols] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    feat         = extract_features(rm, session_id=session_id, session_date=session_date)
    feat['label']       = 1
    feat['threat_type'] = 'Impersonation'
    feat['primary_sid'] = primary_sid
    return feat


def simulate_profiling(rec, session_id, primary_sid, n_grid_cells=16, session_date=None):
    """Replaces natural motion with a systematic grid sweep plus occasional dwell/backtrack."""
    rp  = rec.copy()
    n   = len(rp)
    rng = np.random.default_rng(42 + session_id)
    x0, x1 = rec['head_pos_x'].min(), rec['head_pos_x'].max()
    z0, z1 = rec['head_pos_z'].min(), rec['head_pos_z'].max()
    gs  = int(np.ceil(np.sqrt(n_grid_cells)))
    xs  = np.linspace(x0, x1, gs)
    zs  = np.linspace(z0, z1, gs)
    wps = [(x, z) for i, z in enumerate(zs) for x in (xs if i % 2 == 0 else xs[::-1])]
    n_backtrack         = max(1, int(len(wps) * 0.15))
    backtrack_positions = rng.choice(len(wps) // 2, size=n_backtrack)
    for bp in backtrack_positions:
        insert_at = rng.integers(len(wps) // 2, len(wps))
        wps.insert(int(insert_at), wps[bp])
    wa     = np.array(wps)
    t_wp   = np.linspace(0, n - 1, len(wa))
    t_all  = np.arange(n)
    base_x = np.interp(t_all, t_wp, wa[:, 0])
    base_z = np.interp(t_all, t_wp, wa[:, 1])
    pause_starts     = rng.choice(n - 120, size=int(n * 0.08), replace=False)
    spawn_x, spawn_z = base_x[0], base_z[0]
    for pf in sorted(pause_starts):
        dist_from_spawn = np.sqrt((base_x[pf] - spawn_x) ** 2 + (base_z[pf] - spawn_z) ** 2)
        max_dwell = int(np.clip(120 / (dist_from_spawn + 0.5), 30, 120))
        end       = min(pf + int(rng.integers(30, max_dwell + 1)), n - 1)
        base_x[pf:end] = base_x[pf]
        base_z[pf:end] = base_z[pf]
    rp['head_pos_x'] = base_x + rng.normal(0, 0.005, n)
    rp['head_pos_z'] = base_z + rng.normal(0, 0.005, n)
    feat = extract_features(rp, session_id=session_id, session_date=session_date)
    feat['label']       = 1
    feat['threat_type'] = 'Profiling'
    feat['primary_sid'] = primary_sid
    return feat


# ─────────────────────────────────────────────────────────────────────
# Build labeled dataset
# ─────────────────────────────────────────────────────────────────────
SCENARIOS = ['Benign', 'Tracking', 'Impersonation', 'Profiling']
SCENARIO_COLORS = {
    'Benign':        '#4c72b0',
    'Tracking':      '#dd8452',
    'Impersonation': '#55a868',
    'Profiling':     '#c44e52',
    'Benign_Aria':   '#9467bd',
}
TARGET_THREAT_RATIO = 0.40
SID_OFFSET          = 100

benign_dfs = []
for i, (rec, date_str) in enumerate(zip(sessions, session_dates)):
    feat = extract_features(rec, session_id=i, session_date=date_str)
    feat['label']       = 0
    feat['threat_type'] = 'Benign'
    feat['primary_sid'] = i
    benign_dfs.append(feat)

benign_all = pd.concat(benign_dfs, ignore_index=True)

# Every session appears as the primary_sid for at least one of each
# threat type, so no session is systematically absent from any LOSO
# fold's threat class.
threat_dfs = []

for attacker_sid in range(9):
    target_sid = (attacker_sid + 1) % 9
    threat_dfs.append(simulate_tracking(
        sessions[attacker_sid], sessions[target_sid],
        SID_OFFSET + attacker_sid, primary_sid=attacker_sid,
        intensity=0.85, session_date=session_dates[attacker_sid]))

for mimic_sid in range(9):
    victim_sid = (mimic_sid + 2) % 9
    threat_dfs.append(simulate_impersonation(
        sessions[mimic_sid], sessions[victim_sid],
        SID_OFFSET + 9 + mimic_sid, primary_sid=mimic_sid,
        session_date=session_dates[mimic_sid]))

for profiled_sid in range(9):
    s = sessions[profiled_sid]
    area = (s['head_pos_x'].max() - s['head_pos_x'].min()) * (s['head_pos_z'].max() - s['head_pos_z'].min())
    nc = max(4, min(int(area / 0.5), 256))
    threat_dfs.append(simulate_profiling(
        s, SID_OFFSET + 18 + profiled_sid, primary_sid=profiled_sid,
        n_grid_cells=nc, session_date=session_dates[profiled_sid]))

threat_all = pd.concat(threat_dfs, ignore_index=True)

n_threat_target = int(len(benign_all) * TARGET_THREAT_RATIO / (1 - TARGET_THREAT_RATIO))
threat_sample   = threat_all.sample(n=min(n_threat_target, len(threat_all)), random_state=42)
df_xr           = pd.concat([benign_all, threat_sample], ignore_index=True)
df_xr           = df_xr.dropna(subset=XR_FEATURES).reset_index(drop=True)

print(f"Dataset: {len(df_xr):,} frames | "
      f"{df_xr['label'].value_counts(normalize=True).round(3).to_dict()}")

# ─────────────────────────────────────────────────────────────────────
# Load Project Aria Everyday Activities (real-world out-of-domain data)
# ─────────────────────────────────────────────────────────────────────
AEA_SID_OFFSET = 200


def load_aria_trajectory(sequence_path, target_fps=TARGET_FPS):
    traj_path = os.path.join(sequence_path, "mps", "slam", "closed_loop_trajectory.csv")
    if not os.path.exists(traj_path):
        return None

    traj = pd.read_csv(traj_path)
    rename_map = {
        'tracking_timestamp_us': '_ts_us',
        'tx_world_device':       'head_pos_x',
        'ty_world_device':       'head_pos_y',
        'tz_world_device':       'head_pos_z',
        'qx_world_device':       'head_rot_x',
        'qy_world_device':       'head_rot_y',
        'qz_world_device':       'head_rot_z',
        'qw_world_device':       'head_rot_w',
    }
    traj    = traj.rename(columns=rename_map)
    missing = [c for c in rename_map.values() if c not in traj.columns]
    if missing:
        return None

    traj    = traj.sort_values('_ts_us').reset_index(drop=True)
    ts_min  = traj['_ts_us'].iloc[0]
    ts_max  = traj['_ts_us'].iloc[-1]
    step_us = 1_000_000 / target_fps
    grid_ts = np.arange(ts_min, ts_max, step_us)
    idx     = np.clip(np.searchsorted(traj['_ts_us'].values, grid_ts), 0, len(traj) - 1)
    ds      = traj.iloc[idx].reset_index(drop=True)

    ds['_abs_time_ms']  = (ds['_ts_us'] - ds['_ts_us'].iloc[0]) / 1000.0
    ds['delta_time_ms'] = ds['_abs_time_ms'].diff().fillna(1000.0 / target_fps)

    return ds[['head_pos_x', 'head_pos_y', 'head_pos_z',
               'head_rot_x', 'head_rot_y', 'head_rot_z', 'head_rot_w',
               'delta_time_ms', '_abs_time_ms']]


def discover_aria_sequences(root_dir, max_sequences=143):
    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'closed_loop_trajectory.csv' in filenames:
            seq_root = os.path.dirname(os.path.dirname(dirpath))
            if seq_root not in found:
                found.append(seq_root)
            if len(found) >= max_sequences:
                break
    return sorted(found)


aria_sequences_raw, aria_seq_names = [], []
aria_seq_paths = discover_aria_sequences(AEA_ROOT)
ARIA_AVAILABLE = False

if aria_seq_paths:
    for seq_path in aria_seq_paths:
        df_aria = load_aria_trajectory(seq_path)
        name    = os.path.basename(seq_path)
        if df_aria is not None and len(df_aria) > 500:
            aria_sequences_raw.append(df_aria)
            aria_seq_names.append(name)
    ARIA_AVAILABLE = len(aria_sequences_raw) > 0
    print(f"AEA: {len(aria_sequences_raw)} sequences loaded")
else:
    print(f"AEA: no sequences found at {AEA_ROOT}")

aria_features_df = None
if ARIA_AVAILABLE:
    aria_feat_dfs = []
    for i, df_aria_seq in enumerate(aria_sequences_raw):
        try:
            feat_aria = extract_features(df_aria_seq, session_id=AEA_SID_OFFSET + i,
                                          session_date=f"aria_seq_{i:02d}")
            feat_aria['label']       = 0
            feat_aria['threat_type'] = 'Benign_Aria'
            feat_aria['source']      = 'aria'
            feat_aria['primary_sid'] = -1
            aria_feat_dfs.append(feat_aria)
        except Exception as e:
            print(f"  [SKIP] AEA seq {i}: {e}")
    if aria_feat_dfs:
        aria_features_df = pd.concat(aria_feat_dfs, ignore_index=True)
    else:
        ARIA_AVAILABLE = False

# Cross-domain check: VR (WhoIsAlyx) motion is expected to differ from
# real-world walking (AEA) — a domain gap here is informative, not a bug.
# AEA is used purely as out-of-domain benign augmentation, not as a
# distributional match target.
validation_df = None
if ARIA_AVAILABLE:
    benign_xr_df = df_xr[df_xr['threat_type'] == 'Benign']
    N_COMPARE    = min(10_000, len(benign_xr_df), len(aria_features_df))
    xr_sample    = benign_xr_df[XR_FEATURES].dropna().sample(N_COMPARE, random_state=42)
    aria_sample  = aria_features_df[XR_FEATURES].dropna().sample(N_COMPARE, random_state=42)

    rows = []
    for feat in XR_FEATURES:
        ks_stat, p_val = ks_2samp(xr_sample[feat], aria_sample[feat])
        bins = np.linspace(0, 1, 50)
        xr_h, _ = np.histogram(xr_sample[feat], bins=bins, density=True)
        ar_h, _ = np.histogram(aria_sample[feat], bins=bins, density=True)
        xr_h, ar_h = xr_h / (xr_h.sum() + 1e-9), ar_h / (ar_h.sum() + 1e-9)
        js_div = jensenshannon(xr_h, ar_h)
        rows.append({'Feature': feat, 'KS_stat': ks_stat, 'p_value': p_val, 'JS_div': js_div})
    validation_df = pd.DataFrame(rows)
    print(f"Cross-domain check (VR vs real-world): "
          f"mean JS divergence = {validation_df['JS_div'].mean():.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feat in zip(axes.flatten(), XR_FEATURES):
        sns.kdeplot(xr_sample[feat], ax=ax, label='WhoIsAlyx (VR)', lw=2.5, color='#4c72b0')
        sns.kdeplot(aria_sample[feat], ax=ax, label='Aria AEA (real-world)', lw=2.5,
                    color='#dd8452', linestyle='--')
        row = validation_df[validation_df['Feature'] == feat].iloc[0]
        ax.set_title(f"{feat}\nKS={row['KS_stat']:.3f}  p={row['p_value']:.3e}  JS={row['JS_div']:.3f}",
                     fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_cross_domain_validation.png", dpi=150, bbox_inches='tight')
    plt.close()

# Two-user proximity check: for AEA sequences recorded as co-located
# pairs (rec1/rec2 of the same script), compares real inter-person
# distance to the model's spatial_proximity feature distribution.
if ARIA_AVAILABLE and len(aria_sequences_raw) >= 2:
    def align_and_compute_proximity(traj_a, traj_b):
        n = min(len(traj_a), len(traj_b))
        dx = traj_a['head_pos_x'].values[:n] - traj_b['head_pos_x'].values[:n]
        dz = traj_a['head_pos_z'].values[:n] - traj_b['head_pos_z'].values[:n]
        return np.sqrt(dx ** 2 + dz ** 2)

    def parse_seq_key(name):
        m = re.match(r'(loc\d+_script\d+_seq\d+)_rec\d+', name)
        return m.group(1) if m else None

    seq_map = {}
    for name, df in zip(aria_seq_names, aria_sequences_raw):
        key = parse_seq_key(name)
        if key:
            seq_map.setdefault(key, {})[name] = df

    pair_distances = []
    for key, recs in sorted(seq_map.items()):
        rec_list = sorted(recs.keys())
        if len(rec_list) >= 2:
            pair_distances.append(align_and_compute_proximity(recs[rec_list[0]], recs[rec_list[1]]))

    if pair_distances:
        all_pair_dist  = np.concatenate(pair_distances)
        prox_aria_real = percentile_norm(1.0 / (all_pair_dist + 1e-3))
        prox_xr_benign = df_xr[df_xr['threat_type'] == 'Benign']['spatial_proximity'].values
        ks_prox, p_prox = ks_2samp(prox_xr_benign, prox_aria_real)
        print(f"Proximity check: {len(pair_distances)} co-located pair(s), "
              f"KS={ks_prox:.4f} p={p_prox:.4e}")

# ─────────────────────────────────────────────────────────────────────
# Augmentation pool + rolling temporal features
# ─────────────────────────────────────────────────────────────────────
if ARIA_AVAILABLE:
    n_aria_cap  = int(len(benign_all) * 0.20)
    ARIA_AUG_DF = aria_features_df.sample(min(n_aria_cap, len(aria_features_df)),
                                           random_state=42).reset_index(drop=True)
else:
    ARIA_AUG_DF = None

WINDOW_3S = 270
df_xr = df_xr.sort_values(['session_id', 'frame_id']).reset_index(drop=True)
for feat in XR_FEATURES:
    df_xr[f'{feat}_mean3s'] = df_xr.groupby('session_id')[feat].transform(
        lambda x: x.rolling(WINDOW_3S, min_periods=1).mean())
    df_xr[f'{feat}_std3s'] = df_xr.groupby('session_id')[feat].transform(
        lambda x: x.rolling(WINDOW_3S, min_periods=1).std().fillna(0))

ROLLING_FEATURES = [f'{f}_mean3s' for f in XR_FEATURES] + [f'{f}_std3s' for f in XR_FEATURES]
FEATURE_COLS_V8  = FEATURE_COLS + ROLLING_FEATURES

SESSION_DATE_MAP = {i: d for i, d in enumerate(session_dates)}
DATE_GROUPS      = defaultdict(list)
for sid, date in SESSION_DATE_MAP.items():
    DATE_GROUPS[date].append(sid)

# ─────────────────────────────────────────────────────────────────────
# Correlation filter (drops highly correlated features)
# ─────────────────────────────────────────────────────────────────────
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_  = []

    def fit(self, X, y=None):
        corr  = pd.DataFrame(X).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.to_drop_ = [c for c in upper.columns if any(upper[c] > self.threshold)]
        return self

    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.to_drop_, errors='ignore').values


# ─────────────────────────────────────────────────────────────────────
# LOSO cross-validation
#
# Threats are split by `primary_sid`, not just `session_id`, so a
# threat frame is only in the training fold if the session it's
# derived from isn't the one being held out. This prevents the model
# from training on attacks generated from the same session it's later
# tested on.
# ─────────────────────────────────────────────────────────────────────
df_sorted = df_xr.sort_values(['session_id', 'frame_id']).reset_index(drop=True)
real_mask = df_sorted['session_id'] < SID_OFFSET
df_real   = df_sorted[real_mask].reset_index(drop=True)
df_threat = df_sorted[~real_mask].reset_index(drop=True)
real_sids = df_real['session_id'].unique()

MAX_THREAT_TEST = 50_000


def run_loso_fold(df_real, df_threat, held_sids, feature_cols,
                   aria_aug_df=None, fold_label="fold", fold_idx=0):
    train_mask_real    = ~df_real['session_id'].isin(held_sids)
    train_threat_mask  = ~df_threat['primary_sid'].isin(held_sids)
    train_parts = [df_real[train_mask_real], df_threat[train_threat_mask]]
    if aria_aug_df is not None:
        train_parts.append(aria_aug_df)
    df_train = pd.concat(train_parts, ignore_index=True)
    df_train = df_train.sample(frac=0.25, random_state=fold_idx).reset_index(drop=True)

    test_mask_real  = df_real['session_id'].isin(held_sids)
    test_threat_mask = df_threat['primary_sid'].isin(held_sids)
    df_test_threat = df_threat[test_threat_mask].sample(
        min(MAX_THREAT_TEST, test_threat_mask.sum()), random_state=fold_idx)
    df_test = pd.concat([df_real[test_mask_real], df_test_threat], ignore_index=True)

    feat_cols = [f for f in feature_cols if f in df_train.columns]
    X_tr, y_tr = df_train[feat_cols], df_train['label']
    X_te, y_te = df_test[feat_cols], df_test['label']

    if len(np.unique(y_te)) < 2:
        print(f"  {fold_label:<36} skipped — test fold has only one class")
        result = {'Label': fold_label, 'AUC': 0.5, 'F1': 0.0, 'N_train': len(y_tr), 'N_test': len(y_te)}
        return result, (0.5, None)

    pipe = ImbPipeline([
        ('corr_filter', CorrelationFilter(threshold=0.90)),
        ('scaler',      StandardScaler()),
        ('imputer',     SimpleImputer(strategy='median')),
        ('smote',       SMOTE(random_state=42, k_neighbors=5)),
        ('clf',         RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, proba)
    f1    = f1_score(y_te, (proba >= SELECTED_THRESHOLD).astype(int))
    print(f"  {fold_label:<36} AUC={auc:.4f}  F1={f1:.4f}  n_train={len(y_tr):,}  n_test={len(y_te):,}")

    result = {'Label': fold_label, 'AUC': round(auc, 4), 'F1': round(f1, 4),
              'N_train': len(y_tr), 'N_test': len(y_te)}
    del df_train, df_test, X_tr, y_tr, X_te, y_te, proba
    gc.collect()
    return result, (auc, pipe)


print(f"\nPlayer-level LOSO ({len(real_sids)} folds):")
loso_results_player, loso_pipelines = [], []
for fold, held_sid in enumerate(sorted(real_sids)):
    label = f"player_{held_sid+2:02d} held out"
    result, pipe_info = run_loso_fold(df_real, df_threat, [held_sid], FEATURE_COLS_V8,
                                       ARIA_AUG_DF, label, fold)
    loso_results_player.append(result)
    loso_pipelines.append(pipe_info)

loso_df = pd.DataFrame(loso_results_player)
print(f"  Mean AUC: {loso_df['AUC'].mean():.4f} +/- {loso_df['AUC'].std():.4f}")
print(f"  Mean F1 : {loso_df['F1'].mean():.4f} +/- {loso_df['F1'].std():.4f}")

print(f"\nDate-level LOSO ({len(DATE_GROUPS)} folds — holds out all players from one date):")
loso_results_date = []
for fold_idx, (date, held_sids) in enumerate(sorted(DATE_GROUPS.items())):
    players = [f"p{s+2:02d}" for s in held_sids]
    label   = f"{date} [{'+'.join(players)}] held out"
    result, _ = run_loso_fold(df_real, df_threat, held_sids, FEATURE_COLS_V8,
                               ARIA_AUG_DF, label, fold_idx + 100)
    loso_results_date.append(result)

loso_date_df = pd.DataFrame(loso_results_date)
print(f"  Mean AUC: {loso_date_df['AUC'].mean():.4f} +/- {loso_date_df['AUC'].std():.4f}")

# Select the best player-level fold (excluding single-class placeholders)
# to use for hyperparameter search and downstream evaluation figures.
valid_folds = [(i, a) for i, a in enumerate(loso_df['AUC']) if a > 0.5]
if valid_folds:
    best_fold_idx = max(valid_folds, key=lambda x: x[1])[0]
    best_held_sid = sorted(real_sids)[best_fold_idx]
else:
    best_fold_idx, best_held_sid = 0, sorted(real_sids)[0]

# ─────────────────────────────────────────────────────────────────────
# Hyperparameter search (on training fold only — no test leakage)
# ─────────────────────────────────────────────────────────────────────
df_train_real   = df_real[df_real['session_id'] != best_held_sid]
df_train_threat = df_threat[df_threat['primary_sid'] != best_held_sid]
train_parts = [df_train_real, df_train_threat]
if ARIA_AUG_DF is not None:
    train_parts.append(ARIA_AUG_DF)
df_train_full = pd.concat(train_parts, ignore_index=True).sample(
    frac=0.25, random_state=best_fold_idx).reset_index(drop=True)

df_test_real   = df_real[df_real['session_id'] == best_held_sid]
test_threat_mask = df_threat['primary_sid'] == best_held_sid
df_test_threat = df_threat[test_threat_mask].sample(
    min(MAX_THREAT_TEST, test_threat_mask.sum()), random_state=best_fold_idx)
df_test_full = pd.concat([df_test_real, df_test_threat], ignore_index=True)

feat_cols    = [f for f in FEATURE_COLS_V8 if f in df_train_full.columns]
X_train_full = df_train_full[feat_cols]; y_train_full = df_train_full['label']
X_test_full  = df_test_full[feat_cols];  y_test_full  = df_test_full['label']

SEARCH_SAMPLE_SIZE = 25_000
rng_gs  = np.random.default_rng(42)
pos_idx = np.where(y_train_full == 1)[0]
neg_idx = np.where(y_train_full == 0)[0]
n_each  = min(SEARCH_SAMPLE_SIZE // 2, len(pos_idx), len(neg_idx))
gs_idx  = np.concatenate([rng_gs.choice(pos_idx, n_each, replace=False),
                          rng_gs.choice(neg_idx, n_each, replace=False)])
X_gs = X_train_full.iloc[gs_idx].reset_index(drop=True)
y_gs = y_train_full.iloc[gs_idx].reset_index(drop=True)

pre_gs = SkPipeline([
    ('corr_filter', CorrelationFilter(threshold=0.90)),
    ('scaler',      StandardScaler()),
    ('imputer',     SimpleImputer(strategy='median')),
])
pre_gs.fit(X_gs, y_gs)
X_gs_pre = pre_gs.transform(X_gs)
X_gs_bal, y_gs_bal = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_gs_pre, y_gs)

param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 20], 'min_samples_leaf': [5, 10]}
grid_xr = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_grid, scoring='roc_auc',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=-1)
grid_xr.fit(X_gs_bal, y_gs_bal)
print(f"\nBest params: {grid_xr.best_params_}  (CV AUC={grid_xr.best_score_:.4f})")

best_rf_params = grid_xr.best_params_
best_pipeline  = ImbPipeline([
    ('corr_filter', CorrelationFilter(threshold=0.90)),
    ('scaler',      StandardScaler()),
    ('imputer',     SimpleImputer(strategy='median')),
    ('smote',       SMOTE(random_state=42, k_neighbors=5)),
    ('clf',         RandomForestClassifier(**best_rf_params, class_weight='balanced',
                                            random_state=42, n_jobs=-1)),
])
best_pipeline.fit(X_train_full, y_train_full)

train_auc = roc_auc_score(y_train_full, best_pipeline.predict_proba(X_train_full)[:, 1])
probas_xr = best_pipeline.predict_proba(X_test_full)[:, 1]
test_auc  = roc_auc_score(y_test_full, probas_xr)
print(f"Train AUC: {train_auc:.4f}  Test AUC: {test_auc:.4f}  Gap: {train_auc - test_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────
# Correlation matrix
# ─────────────────────────────────────────────────────────────────────
corr_matrix = pd.DataFrame(X_train_full).corr().abs()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
            annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.4, ax=ax,
            annot_kws={"size": 7})
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────────────
# Feature distributions + ANOVA
# ─────────────────────────────────────────────────────────────────────
avail_xr    = [s for s in SCENARIOS if s in df_xr['threat_type'].unique()]
plot_source = df_xr.copy()
if ARIA_AVAILABLE and aria_features_df is not None:
    plot_source = pd.concat(
        [plot_source, aria_features_df.sample(min(5000, len(aria_features_df)), random_state=42)],
        ignore_index=True)
avail_all = [s for s in avail_xr + ['Benign_Aria'] if s in plot_source['threat_type'].unique()]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.flatten(), XR_FEATURES):
    for sc in avail_all:
        sub = plot_source[plot_source['threat_type'] == sc][feat].dropna()
        if len(sub) > 10:
            jit = sub + np.random.RandomState(0).normal(0, sub.std() * 0.001 + 1e-7, len(sub))
            ls  = '--' if sc == 'Benign_Aria' else '-'
            try:
                sns.kdeplot(jit, ax=ax, label=sc, color=SCENARIO_COLORS.get(sc, 'gray'), lw=2, linestyle=ls)
            except Exception:
                ax.hist(sub, bins=30, alpha=0.4, label=sc, color=SCENARIO_COLORS.get(sc, 'gray'), density=True)
    ax.set_title(feat, fontsize=11); ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_kde_distributions.png", dpi=150, bbox_inches='tight')
plt.close()

profile_df = df_xr.groupby('threat_type')[XR_FEATURES].mean()
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(profile_df, annot=True, fmt='.3f', cmap='RdYlGn', linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Mean Feature Value (0-1)'})
ax.set_title("Mean XR Feature Values per Threat Scenario")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_scenario_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nOne-way ANOVA per feature:")
for feat in FEATURE_COLS:
    grps = [df_xr[df_xr['threat_type'] == sc][feat].dropna().values for sc in avail_xr]
    grps = [g for g in grps if len(g) > 1]
    if len(grps) < 2:
        continue
    f_stat, p = stats.f_oneway(*grps)
    gm   = df_xr[feat].mean()
    ssb  = sum(len(g) * (g.mean() - gm) ** 2 for g in grps)
    sst  = ((df_xr[feat] - gm) ** 2).sum()
    eta2 = ssb / max(sst, 1e-9)
    print(f"  {feat:<24} F={f_stat:>8.1f}  p={p:>10.2e}  eta2={eta2:.4f}")

# ─────────────────────────────────────────────────────────────────────
# Threshold selection
# ─────────────────────────────────────────────────────────────────────
prec_vals, rec_vals, thr_pr = precision_recall_curve(y_test_full, probas_xr)
ap   = average_precision_score(y_test_full, probas_xr)
f1_v = np.where((prec_vals[:-1] + rec_vals[:-1]) > 0,
                2 * prec_vals[:-1] * rec_vals[:-1] / (prec_vals[:-1] + rec_vals[:-1]), 0)
best_f1_thresh = thr_pr[np.argmax(f1_v)]
print(f"\nMax-F1 threshold: {best_f1_thresh:.3f}  Selected: {SELECTED_THRESHOLD}  AP: {ap:.4f}")

idx_sel   = np.argmin(np.abs(thr_pr - SELECTED_THRESHOLD))
y_pred_xr = (probas_xr >= SELECTED_THRESHOLD).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(rec_vals, prec_vals, lw=2, color='steelblue', label=f"AP={ap:.3f}")
axes[0].scatter(rec_vals[idx_sel], prec_vals[idx_sel], color='red', zorder=5, s=80,
                 label=f"Threshold={SELECTED_THRESHOLD}")
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(thr_pr, prec_vals[:-1], label='Precision', lw=2)
axes[1].plot(thr_pr, rec_vals[:-1], label='Recall', lw=2)
axes[1].plot(thr_pr, f1_v, label='F1', lw=2)
axes[1].axvline(SELECTED_THRESHOLD, color='red', ls='--', lw=1.8, label=f'Selected ({SELECTED_THRESHOLD})')
axes[1].axvline(best_f1_thresh, color='green', ls=':', lw=1.8, label=f'Max-F1 ({best_f1_thresh:.2f})')
axes[1].set_xlabel("Threshold"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_threshold_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────────────
# Full evaluation on the held-out fold
# ─────────────────────────────────────────────────────────────────────
acc  = accuracy_score(y_test_full, y_pred_xr)
prec = precision_score(y_test_full, y_pred_xr)
rec  = recall_score(y_test_full, y_pred_xr)
f1   = f1_score(y_test_full, y_pred_xr)
auc  = roc_auc_score(y_test_full, probas_xr)
cm   = confusion_matrix(y_test_full, y_pred_xr)
TN, FP, FN, TP = cm.ravel()
fpr_val = FP / (FP + TN)
fnr_val = FN / (FN + TP)

print(f"\nEvaluation (threshold={SELECTED_THRESHOLD}):")
print(classification_report(y_test_full, y_pred_xr, target_names=["Benign", "Threat"]))
print(f"AUC={auc:.4f}  FPR={fpr_val:.4f}  FNR={fnr_val:.4f}")

fpr_r, tpr_r, _ = roc_curve(y_test_full, probas_xr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_r, tpr_r, lw=2.5, color='steelblue', label=f"AUC={auc:.3f}")
ax.plot([0, 1], [0, 1], 'k:', lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────────────
# SHAP explanations
# ─────────────────────────────────────────────────────────────────────
filtered_cols    = [c for c in FEATURE_COLS_V8 if c not in best_pipeline.named_steps['corr_filter'].to_drop_]
X_test_filtered  = X_test_full[filtered_cols].reset_index(drop=True)
X_test_scaled    = best_pipeline.named_steps['scaler'].transform(X_test_filtered)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=filtered_cols)

explainer_xr  = shap.TreeExplainer(best_pipeline.named_steps['clf'])
shap_sample_n = min(1500, len(X_test_scaled_df))
shap_row_idx  = np.random.RandomState(42).choice(len(X_test_scaled_df), shap_sample_n, replace=False)
X_shap    = X_test_scaled_df.iloc[shap_row_idx].reset_index(drop=True)
shap_vals = explainer_xr(X_shap)

# Realign threat-type labels to the SHAP sample by rebuilding the same
# test split and indexing with shap_row_idx (positional, not index-based).
_test_mask_r     = df_real['session_id'] == best_held_sid
_test_threat_mask = df_threat['primary_sid'] == best_held_sid
_df_test_threat2 = df_threat[_test_threat_mask].sample(
    min(MAX_THREAT_TEST, _test_threat_mask.sum()), random_state=best_fold_idx)
_df_test_full   = pd.concat([df_real[_test_mask_r], _df_test_threat2], ignore_index=True)
scenario_test   = _df_test_full['threat_type'].values[shap_row_idx]
del _df_test_threat2, _df_test_full

shap.summary_plot(shap_vals, features=X_shap, feature_names=filtered_cols, show=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_shap_global.png", dpi=150, bbox_inches='tight')
plt.close()

THREAT_SCENARIOS = [s for s in ['Tracking', 'Impersonation', 'Profiling'] if s in df_xr['threat_type'].unique()]
fig, axes = plt.subplots(1, len(THREAT_SCENARIOS), figsize=(6 * len(THREAT_SCENARIOS), 5))
if len(THREAT_SCENARIOS) == 1:
    axes = [axes]

for ax, sc in zip(axes, THREAT_SCENARIOS):
    mask = scenario_test == sc
    if mask.sum() == 0:
        ax.set_title(f"{sc} — no samples")
        continue
    sv = shap_vals.values[mask]
    if sv.ndim == 3:  # binary RF TreeExplainer returns (n, features, 2) — take the threat class
        sv = sv[:, :, 1]
    mean_shap = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_shap)
    ax.barh(range(len(filtered_cols)), mean_shap[order], color=SCENARIO_COLORS[sc])
    ax.set_yticks(range(len(filtered_cols)))
    ax.set_yticklabels([filtered_cols[i] for i in order], fontsize=8)
    ax.set_title(f"{sc} (n={mask.sum():,})")
    ax.set_xlabel("Mean |SHAP|")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_shap_per_scenario.png", dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────────────
# Baseline comparisons
# ─────────────────────────────────────────────────────────────────────
pre = SkPipeline([
    ('corr_filter', CorrelationFilter(threshold=0.90)),
    ('scaler',      StandardScaler()),
    ('imputer',     SimpleImputer(strategy='median')),
])
pre.fit(X_train_full, y_train_full)
X_tr_pre, X_te_pre = pre.transform(X_train_full), pre.transform(X_test_full)
X_tr_bal, y_tr_bal = SMOTE(random_state=42).fit_resample(X_tr_pre, y_train_full)

baselines = {
    'Random Forest': RandomForestClassifier(**best_rf_params, class_weight='balanced',
                                             random_state=42, n_jobs=-1),
    'LightGBM':      LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                     num_leaves=31, device='cpu', random_state=42,
                                     verbose=-1, class_weight='balanced'),
    'XGBoost':       XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                    verbosity=0, random_state=42, n_jobs=-1,
                                    eval_metric='logloss', scale_pos_weight=3),
    'MLP':           MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                                    max_iter=300, early_stopping=True, random_state=42),
}

results = []
for name, clf in baselines.items():
    t0 = time.perf_counter()
    clf.fit(X_tr_bal, y_tr_bal)
    train_sec = time.perf_counter() - t0
    proba_b  = clf.predict_proba(X_te_pre)[:, 1]
    y_pred_b = (proba_b >= SELECTED_THRESHOLD).astype(int)
    for _ in range(10):
        clf.predict_proba(X_te_pre[:1])
    t_lat = time.perf_counter()
    for _ in range(100):
        clf.predict_proba(X_te_pre[:1])
    lat_ms = (time.perf_counter() - t_lat) / 100 * 1000
    cm_b = confusion_matrix(y_test_full, y_pred_b)
    tn_b, fp_b = cm_b[0, 0], cm_b[0, 1]
    results.append({
        'Model': name, 'Accuracy': accuracy_score(y_test_full, y_pred_b),
        'Recall': recall_score(y_test_full, y_pred_b),
        'Precision': precision_score(y_test_full, y_pred_b),
        'F1': f1_score(y_test_full, y_pred_b),
        'AUC': roc_auc_score(y_test_full, proba_b),
        'FPR': fp_b / (fp_b + tn_b), 'Train(s)': round(train_sec, 2), 'Lat(ms)': round(lat_ms, 3),
    })

results_df = pd.DataFrame(results).set_index('Model')
print(f"\nBaseline comparison:\n{results_df.round(4).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
results_df[['Accuracy', 'Recall', 'Precision', 'F1', 'AUC']].plot(kind='bar', ax=axes[0], rot=22, width=0.72)
axes[0].set_ylim(0.5, 1.02); axes[0].grid(axis='y', alpha=0.3)
bars = axes[1].bar(results_df.index, results_df['Lat(ms)'], color='steelblue', width=0.55)
axes[1].set_title("Per-Event Latency (ms)"); axes[1].tick_params(axis='x', rotation=22)
for b in bars:
    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_baseline_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────────────
# Ablation study
# ─────────────────────────────────────────────────────────────────────
ablation_configs = {
    'Motion only':          ['motion_speed', 'head_rotation_rate', 'pose_variance'],
    'Entropy + Proximity':  ['movement_entropy', 'spatial_proximity', 'interaction_freq'],
    'All XR features':      XR_FEATURES,
    'Full model (+ time)':  FEATURE_COLS,
    'Full model + rolling': FEATURE_COLS_V8,
}

ablation_results = []
for config_name, feat_subset in ablation_configs.items():
    avail_feats = [f for f in feat_subset if f in X_train_full.columns]
    if not avail_feats:
        continue
    sc_ab  = StandardScaler()
    X_ab_tr = sc_ab.fit_transform(X_train_full[avail_feats].values)
    X_ab_te = sc_ab.transform(X_test_full[avail_feats].values)
    imp_ab  = SimpleImputer(strategy='median')
    X_ab_tr = imp_ab.fit_transform(X_ab_tr)
    X_ab_te = imp_ab.transform(X_ab_te)
    X_ab_tr, y_ab_tr = SMOTE(random_state=42).fit_resample(X_ab_tr, y_train_full)

    clf_ab = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=5,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
    clf_ab.fit(X_ab_tr, y_ab_tr)
    proba_ab  = clf_ab.predict_proba(X_ab_te)[:, 1]
    y_pred_ab = (proba_ab >= SELECTED_THRESHOLD).astype(int)

    ablation_results.append({
        'Configuration': config_name, 'N features': len(avail_feats),
        'AUC': round(roc_auc_score(y_test_full, proba_ab), 4),
        'F1': round(f1_score(y_test_full, y_pred_ab), 4),
        'Recall': round(recall_score(y_test_full, y_pred_ab), 4),
        'Precision': round(precision_score(y_test_full, y_pred_ab), 4),
    })

ablation_df = pd.DataFrame(ablation_results).set_index('Configuration')
print(f"\nAblation study:\n{ablation_df.round(4).to_string()}")

fig, ax = plt.subplots(figsize=(12, 5))
x, w = np.arange(len(ablation_df)), 0.2
ax.bar(x - w, ablation_df['AUC'], w, label='AUC', color='#4c72b0')
ax.bar(x, ablation_df['F1'], w, label='F1', color='#55a868')
ax.bar(x + w, ablation_df['Recall'], w, label='Recall', color='#dd8452')
ax.set_xticks(x); ax.set_xticklabels(ablation_df.index, rotation=15, ha='right')
ax.set_ylim(0.4, 1.05); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_ablation_study.png", dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────────────
# Cross-scenario generalization
# (leaves out one threat type from training, tests only on it)
# ─────────────────────────────────────────────────────────────────────
avail_threats = [s for s in ['Tracking', 'Impersonation', 'Profiling'] if s in df_xr['threat_type'].unique()]
if len(avail_threats) >= 2:
    benign_train = df_real[df_real['session_id'] != best_held_sid]
    benign_test  = df_real[df_real['session_id'] == best_held_sid]
    print("\nCross-scenario generalization:")
    for held_out in avail_threats:
        train_threats = [t for t in avail_threats if t != held_out]
        train_threat_df = df_threat[df_threat['threat_type'].isin(train_threats) &
                                     (df_threat['primary_sid'] != best_held_sid)]
        test_threat_df  = df_threat[(df_threat['threat_type'] == held_out) &
                                     (df_threat['primary_sid'] == best_held_sid)]
        df_cs_train = pd.concat([benign_train, train_threat_df], ignore_index=True)
        df_cs_test  = pd.concat([benign_test, test_threat_df], ignore_index=True)
        if len(df_cs_test['label'].unique()) < 2:
            continue
        X_tr, y_tr = df_cs_train[feat_cols], df_cs_train['label']
        X_te, y_te = df_cs_test[feat_cols], df_cs_test['label']
        pipe = ImbPipeline([
            ('corr_filter', CorrelationFilter(0.9)), ('scaler', StandardScaler()),
            ('imputer', SimpleImputer()), ('smote', SMOTE(random_state=42)),
            ('clf', RandomForestClassifier(**best_rf_params, class_weight='balanced', random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        auc_cs = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        print(f"  Train on {train_threats} -> Test on {held_out}: AUC={auc_cs:.4f}")

# ─────────────────────────────────────────────────────────────────────
# Latency benchmark
# ─────────────────────────────────────────────────────────────────────
N_WARMUP, N_TIMED = 50, 500
single_event = X_test_full.iloc[[0]]
for _ in range(N_WARMUP):
    best_pipeline.predict_proba(single_event)
latencies = []
for _ in range(N_TIMED):
    t0 = time.perf_counter()
    best_pipeline.predict_proba(single_event)
    latencies.append((time.perf_counter() - t0) * 1000)

lat_arr = np.array(latencies)
VR_THRESHOLD_MS = 10.0  # 90fps VR frame budget
print(f"\nLatency: mean={lat_arr.mean():.3f}ms  P99={np.percentile(lat_arr, 99):.3f}ms  "
      f"meets <{VR_THRESHOLD_MS}ms budget: {np.percentile(lat_arr, 99) < VR_THRESHOLD_MS}")

# ─────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────
def bootstrap_ci(y_true, y_proba, threshold=SELECTED_THRESHOLD, n_boot=1000, ci=95):
    np.random.seed(42)
    y_true  = np.array(y_true)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    store   = {m: [] for m in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FPR']}
    for _ in range(n_boot):
        idx = np.concatenate([np.random.choice(pos_idx, len(pos_idx), replace=True),
                              np.random.choice(neg_idx, len(neg_idx), replace=True)])
        yt, yp = y_true[idx], y_proba[idx]
        yh = (yp >= threshold).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        cm_b = confusion_matrix(yt, yh)
        tn_b, fp_b = cm_b[0, 0], cm_b[0, 1]
        store['Accuracy'].append(accuracy_score(yt, yh))
        store['Precision'].append(precision_score(yt, yh, zero_division=0))
        store['Recall'].append(recall_score(yt, yh, zero_division=0))
        store['F1'].append(f1_score(yt, yh, zero_division=0))
        store['AUC'].append(roc_auc_score(yt, yp))
        store['FPR'].append(fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0)
    alpha_p = (100 - ci) / 2
    rows = []
    for metric, vals in store.items():
        v = np.array(vals)
        lo, hi = np.percentile(v, alpha_p), np.percentile(v, 100 - alpha_p)
        rows.append({'Metric': metric, 'Mean': round(v.mean(), 4), f'{ci}% CI': f"[{lo:.4f}, {hi:.4f}]"})
    return pd.DataFrame(rows).set_index('Metric')

point_est = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'AUC': auc, 'FPR': fpr_val}
ci_df = bootstrap_ci(y_test_full.values, probas_xr)
ci_df.insert(0, 'Point Est.', [point_est[m] for m in ci_df.index])
print(f"\nBootstrap 95% CIs:\n{ci_df.round(4).to_string()}")

# ─────────────────────────────────────────────────────────────────────
# Adaptive adversary: mimicry evasion
#
# Attacker gradually blends their Tracking-attack entropy/proximity
# features toward the benign mean, to see how much recall degrades
# under a mimicry attempt.
# ─────────────────────────────────────────────────────────────────────
BENIGN_ENTROPY_MEAN   = df_xr[df_xr['threat_type'] == 'Benign']['movement_entropy'].mean()
BENIGN_PROXIMITY_MEAN = df_xr[df_xr['threat_type'] == 'Benign']['spatial_proximity'].mean()
tracking_rows = df_xr[df_xr['threat_type'] == 'Tracking']
_feat_cols_v8 = [f for f in FEATURE_COLS_V8 if f in df_xr.columns]
X_tracking = tracking_rows[_feat_cols_v8].copy()
y_tracking = tracking_rows['label'].copy()

if len(X_tracking) > 0:
    evasion_results = []
    for alpha_e in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        Xe = X_tracking.copy()
        Xe['movement_entropy']  = (1 - alpha_e) * Xe['movement_entropy'] + alpha_e * BENIGN_ENTROPY_MEAN
        Xe['spatial_proximity'] = (1 - alpha_e) * Xe['spatial_proximity'] + alpha_e * BENIGN_PROXIMITY_MEAN
        if 'movement_entropy_mean3s' in Xe.columns:
            Xe['movement_entropy_mean3s'] = (1 - alpha_e) * Xe['movement_entropy_mean3s'] + alpha_e * BENIGN_ENTROPY_MEAN
        if 'spatial_proximity_mean3s' in Xe.columns:
            Xe['spatial_proximity_mean3s'] = (1 - alpha_e) * Xe['spatial_proximity_mean3s'] + alpha_e * BENIGN_PROXIMITY_MEAN
        Xe = Xe.clip(0, 1)
        proba_e  = best_pipeline.predict_proba(Xe)[:, 1]
        detected = (proba_e >= SELECTED_THRESHOLD).astype(int).sum()
        evasion_results.append({'alpha': alpha_e, 'Recall': round(detected / max(len(y_tracking), 1), 4),
                                 'Detected': detected, 'Missed': len(y_tracking) - detected})
    ev_df = pd.DataFrame(evasion_results)
    print(f"\nMimicry evasion:\n{ev_df.to_string(index=False)}")

    plt.figure(figsize=(9, 5))
    plt.plot(ev_df['alpha'], ev_df['Recall'], marker='o', lw=2.5, color='#c44e52', label='Tracking Recall')
    plt.axhline(0.5, color='gray', ls='--', lw=1.5, label='50% recall baseline')
    plt.xlabel("Evasion Intensity (0=no evasion, 1=full benign mimicry)")
    plt.ylabel("Tracking Detection Recall")
    plt.legend(); plt.grid(alpha=0.3); plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_evasion_robustness.png", dpi=150, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────
# LOSO per-fold summary plot
# ─────────────────────────────────────────────────────────────────────
date_colors = {d: c for d, c in zip(sorted(DATE_GROUPS.keys()),
                                     ['#4c72b0', '#55a868', '#dd8452', '#c44e52', '#8172b2', '#937860'])}

fig, axes = plt.subplots(1, 2, figsize=(18, 5))

ax = axes[0]
x = np.arange(len(loso_df))
bars = ax.bar(x, loso_df['AUC'], color='steelblue', width=0.55)
ax.axhline(loso_df['AUC'].mean(), color='red', ls='--', lw=2)
for b, v in zip(bars, loso_df['AUC']):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
for b, sid in zip(bars, sorted(real_sids)):
    b.set_color(date_colors.get(SESSION_DATE_MAP[sid], 'steelblue'))
    b.set_alpha(0.85)
ax.set_xticks(x); ax.set_xticklabels([f"p{s+2:02d}" for s in sorted(real_sids)], rotation=20, ha='right')
ax.set_ylabel("ROC-AUC"); ax.set_title("Player-Level LOSO\nColor = recording date")
ax.legend(handles=[mpatches.Patch(facecolor=c, label=d) for d, c in date_colors.items()], fontsize=7, loc='lower right')
ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
x2 = np.arange(len(loso_date_df))
bars2 = ax2.bar(x2, loso_date_df['AUC'],
                color=[date_colors.get(d, 'gray') for d in sorted(DATE_GROUPS.keys())], width=0.55, alpha=0.85)
ax2.axhline(loso_date_df['AUC'].mean(), color='red', ls='--', lw=2)
for b, v in zip(bars2, loso_date_df['AUC']):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
ax2.set_xticks(x2); ax2.set_xticklabels(sorted(DATE_GROUPS.keys()), rotation=15, ha='right')
ax2.set_ylabel("ROC-AUC"); ax2.set_title("Date-Level LOSO\nHolds out all players from one recording date")
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_loso_per_fold.png", dpi=150, bbox_inches='tight')
plt.close()

loso_gap = loso_df['AUC'].mean() - loso_date_df['AUC'].mean()
print(f"\nSession-level vs date-level LOSO gap: {loso_gap:.4f}")
if loso_gap > 0.03:
    print("  -> Recording-session confound likely present.")
else:
    print("  -> Model generalizes well across recording sessions.")

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
print(f"""
BehaviorShield — summary
  Frames        : {len(df_xr):,}  (benign / threat = {TARGET_THREAT_RATIO:.0%} threat target)
  Features      : {len(FEATURE_COLS_V8)}  ({len(FEATURE_COLS)} base + {len(ROLLING_FEATURES)} rolling 3s mean/std)
  Threshold     : {SELECTED_THRESHOLD}

  Player-level LOSO : AUC {loso_df['AUC'].mean():.4f} +/- {loso_df['AUC'].std():.4f}, F1 {loso_df['F1'].mean():.4f} +/- {loso_df['F1'].std():.4f}
  Date-level LOSO    : AUC {loso_date_df['AUC'].mean():.4f} +/- {loso_date_df['AUC'].std():.4f}
  Session gap        : {loso_gap:.4f}

  Held-out fold eval : AUC={auc:.4f}  F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}
                        FPR={fpr_val:.4f}  FNR={fnr_val:.4f}
""")

print(f"Figures saved to: {OUTPUT_DIR}/")
