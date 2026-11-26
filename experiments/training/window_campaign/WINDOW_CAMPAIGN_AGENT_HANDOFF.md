# Window campaign — self-contained handoff for analysis agents

**Language:** English  
**Repo path:** `IAModel/experiments/training/window_campaign/`  
**Status (2026-11):** R2 implementation in active use; overnight benchmark runs on remote server (`data_root` ≈ 23 589 clips). Multiple training bugs were found and partially fixed; **sampler scaling bug** was the main explanation for “smoke works, full run `rec_robo=0%`”.  
**Audience:** Another agent with **no** prior chat context — use this file + referenced paths only.

---

## 1. What we are trying to achieve

### 1.1 Product goal

Reduce **motion-model false positives** in production by training binary robbery detectors in a setup that **matches how motion is scored in the field**, instead of training one resampled vector per labeled action clip.

Production (approximate):

- Pose frames at **12 Hz** (target; cameras may be 15 fps with hold-last in prod — separate parity topic).
- Motion model sees **8 s buffers = 96 consecutive pose rows**.
- Chunks are evaluated with **50% overlap** between consecutive 8 s windows (stride 48 frames on a dense grid).
- Multiple windows per customer session → session-level FP arithmetic differs from single-clip val.

### 1.2 Training goal (window campaign)

| Legacy `campaign/` | Window campaign |
|------------------|-----------------|
| One sample per `poses_full.npy` (whole clip → `temporal_resize` to `seq_len`) | Many **sliding 96-frame windows** per clip, sliced **in memory** from the same `.npy` |
| Label = folder 6 or 14 vs rest | Same **binary** positives `{6, 14}`, but **core-aware** labels per window (R2) |
| Split + augment at **clip** level | Split at **clip UID**; augment at **window** level |

**Non-goals (current sweep):** multiclass training; full production replay automation (planned B9); merged session chunks in trainer (planned B8).

### 1.3 Default benchmark cell

- **Cell ID:** `win_bin_full_baseline` (`window_config.json` → `default_cell_id`)
- **Benchmark architectures (6):** experiment IDs **48, 49, 53, 57, 13, 11** (`experiment_benchmark`)
- **Production reference models:** **48, 49** (`tcn_gru`)

---

## 2. Data source and layout

### 2.1 `data_root`

Typical path on training server:

```text
/media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result
```

Structure (as used by `train_model_operations.collect_examples`):

```text
data_result/
  <category_id>/          # e.g. 3, 5, 6, 14, …
    <clip_name>/
      user_0/
        poses_full.npy    # or poses_filtered.npy if pose_source=filtered
        valid_mask.npy    # optional
        meta.json
```

- **`pose_source: full`** in current config → `poses_full.npy`.
- **`single_user_only: true`** → only `user_0` (curated single-person clips).
- Corpus scale (recent run): **~23 589** valid clip examples after filters.

### 2.2 Clip-level filters (`clip_filters` in `window_config.json`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_clip_seconds` | **4.0** | Reject clip if `meta.json` `clip_duration` &lt; 4 s |
| `min_valid_frames` / `min_valid_pct` | 0 | Off in current config |
| `max_occlusion_ratio` | 90% | Quality meta cap |

**Note:** Short **2–4 s** segments are **not** used if `clip_duration` in meta is &lt; 4 s. Windowing can still **pad** `T < 96` rows to 8 s inside a clip that passed the 4 s filter. Pre-splitting corpus into physical 8 s `.npy` files on disk is **future/optional** (`materialize_windows.py`); default train **does not** require new files on disk.

### 2.3 Train / val / test split

- **80 / 10 / 10** by **`clip_uid`** (not by window).
- **Frozen manifest:** `artifacts/plans/win_bin_full_baseline_split_manifest.json`  
  (`use_frozen_manifest: true`) — do **not** delete between runs if you need comparable val/test.
- Recent verified split: **18 864 / 2 354 / 2 371** clips (train / val / test), disjoint UIDs.

---

## 3. Eight-second windows and overlap (core design)

### 3.1 Definitions

- **`window_len` = 96** frames → **8.0 s at 12 Hz** (not 9 s).
- **`stride` = 48** → **50% overlap** between consecutive starts on the same clip.
- **`pad_mode` = `repeat_last`** → if `T < 96`, one window `[0:T)` padded to 96.

Implementation: `window_index.iter_window_slices()` + `window_build_loaders.build_windows_from_examples()`.

### 3.2 Slicing rules (all categories, including 6 and 14)

| Clip length `T` | Windows produced |
|-----------------|------------------|
| `T ≤ 0` | none |
| `T < 96` | **one** window `[0:T)` then pad to 96 |
| `T ≥ 96` | starts `0, 48, 96, …` while `start + 96 ≤ T` |
| Uncovered tail ≥ **`min_tail_for_final_window` (24)** | add aligned window `[T-96:T)` if not already present |

**Important:** Binary folder label does **not** change slicing. Positives are windowed like negatives.

### 3.3 Worked examples

**16 s @ 12 Hz → T = 192**

| Window | Row range | Approx. time | Overlap |
|--------|-----------|--------------|---------|
| 1 | `[0:96)` | 0–8 s | — |
| 2 | `[48:144)` | 4–12 s | rows 48–95 shared with w1 |
| 3 | `[96:192)` | 8–16 s | rows 96–143 shared with w2 |

**~6 s staged positive → T ≈ 72**

- One padded window `[0:72)` + 24 repeat-last frames → single 96-frame training unit.

### 3.4 Production overlap vs training overlap

| Concept | Training (today) | Production (target parity) |
|---------|------------------|----------------------------|
| Regular 8 s chunk | Sliding windows inside each **action clip** `.npy` | Fixed chunks along session timeline |
| **Overlap window** | Neighbor windows share 48 frames (stride 48) | Often **concat** of second half of chunk *i* + first half of chunk *i+1* (Session A / ses2 probes in `MOTION_FP_INVESTIGATION_BRIEF.md`) |
| Overlap in index | Implicit via stride | Explicit `production_overlap` type — **not fully in trainer yet** (FINAL_PLAN Stage 2) |

**Risk called out in docs:** velocity spikes when `valid_mask` drops frames differently on chunks vs overlaps — `mask_policy: keep_grid_v2` (B5).

---

## 4. Labels, quality, and generated logical rows (R2)

### 4.1 Core-aware positive labeling (B1)

Config: `positive_labeling.mode = core_aware`, `artifacts/core_index.json`.

Per window on robbery clips (`label=1` from binary mapping):

| `label_state` | `train_label` | Typical meaning |
|---------------|---------------|-----------------|
| `pos_core` | 1 | Overlap with gesture **core** ≥ threshold |
| `pos_partial` | 1 (lower `sample_weight`) | Partial core overlap |
| `ignore` | -1 (excluded from train/eval rows) | Standing prefix/suffix only |
| `neg` | 0 | Non-robbery folders or non-core robbery windows |

Val/test eval uses windows with `label_state != ignore` and `train_label >= 0`.

Example counts after full pipeline (one real run):

```text
train windows: 20444 | label_states: ignore 265, neg 17044, pos_core 2882, pos_partial 253
val windows:   2549  | val: ignore 20, neg 2155, pos_core 343, pos_partial 31  → n_pos 374 in metrics
```

### 4.2 Quality gate (`window_quality.py`)

Applied **per window** after slicing (can drop many windows):

- Min content duration (2 s full / 1 s padded tail).
- Teleport / centroid jump / low visibility / zero-pose ratios.

One run logged: **27 906 dropped**, **20 444 kept** (reasons: centroid_jump, teleport_jump_ratio, etc.).

### 4.3 Synthetic / manifest expansion (logical rows, not new `.npy` on disk)

- **`validate_npy` manifest cache:** `experiments/training/operations_npy/manifest_cache/`
- **`extra_manifest_views_per_clip: 1`** → at most one extra deterministic variant per window when manifest exists.
- **`window_synthetic.expand_windows_with_manifest_extra_views`:** replaces a window with identity + optional ops row(s); **same** `(pose_path, start, end)` slice.

Recent run: `[WINDOW-EXPAND] ventanas: 20444 -> 20444 | clips sin manifest~9090`

### 4.4 Padding negatives (B2)

`padded_negative_prob: 0.15` — random truncate negatives to simulate short padded tails (anti shortcut “padding ⇒ positive”).

### 4.5 Preprocessing per window (`window_dataset.py`)

Order: load full `.npy` → `extract_window` → optional augment → `normalize_sequence` → `add_velocity_v2` if `keep_grid_v2` → tensor `[96, J*4]`.

---

## 5. Training stack (how a run works)

### 5.1 Entry points

| Script | Purpose |
|--------|---------|
| `preflight_window_campaign.py` | Scan corpus, stats, write plan |
| `train_window_campaign.py` | Train experiment ID(s); checkpoints `artifacts/models/<cell_id>/modelo_XX.pt` |
| `run_overnight_batch.py` | preflight (optional) → train benchmark → `summarize_window_runs.py` → `verify_split_disjoint.py` |
| `build_core_index.py` | Build `artifacts/core_index.json` |

### 5.2 Loss and metrics

- Config name: `weighted_bce` → implementation uses **`CrossEntropyLoss` on both classes** (`window_loss.py`) + per-sample `sample_weight`.
- **Class weights** optional: `auto_pos_class_weight` from neg/pos window counts.
- **Checkpoint selection (fixed):** `train_defaults.checkpoint_metric: macro_f1` (not raw `val_acc` — avoids “always negative” checkpoint).
- Metrics line: `rec_robo`, `FP`, `n_pos` (`train_model_operations.format_binary_metrics_line`).

### 5.3 Sampler (B3) — critical for correctness

`HierarchicalClipCapSampler`:

- Target fraction of positive windows per epoch: `sampler.positive_share` (now **0.4**).
- Cap windows per clip per epoch: `max_windows_per_clip_per_epoch: 4`.
- **Fixed samples per epoch:** `samples_per_epoch: 3200` (not `len(dataset)`).

---

## 6. Incidents observed on real runs (Nov 2026)

### 6.1 Symptom: trivial classifier on full corpus

| Metric | Broken overnight run | Small smoke (200 clips) |
|--------|----------------------|-------------------------|
| `val_acc` | **0.8521** frozen every epoch | ~34–42% (model predicts some positives) |
| `rec_robo` | **0%** | **75%** ep 1 (debug) |
| `train_loss` ep 1 | ~0.24 (suspiciously low) | ~1.47 |

Val had **`n_pos=374`** → labels present; failure was **learning / sampling**, not empty val positives.

### 6.2 Bug A — Loss only trained logit for class 1 (fixed earlier)

Old `weighted_bce` used BCE on `logits[:,1]` only → argmax evaluation always class 0.

**Fix:** `F.cross_entropy(logits, y, ...)` on both classes (`window_loss.py`).

**Verified on server** before later incidents.

### 6.3 Bug B — Manifest window copy dropped labels (fixed)

`window_synthetic._window_copy` recreated `WindowExample` without `train_label` / `label_state` → defaults **`train_label=0`**, **`neg`**.

**Fix:** `dataclasses.replace(w, ...)` to preserve metadata.

### 6.4 Bug C — Checkpoint by `val_acc` (fixed)

85% negatives → “never predict robbery” maximizes accuracy.

**Fix:** `checkpoint_metric: macro_f1`.

### 6.5 Bug D — **Sampler `num_samples` scaled with dataset size (root cause smoke vs full)**

**Old code** (`window_build_loaders.py`):

```python
n_train = max(batch_size * 100, len(train_ds))
```

| Mode | `len(train_ds)` | `num_samples` / epoch | `pos_target` @ 35% |
|------|-----------------|----------------------|---------------------|
| Smoke ~400 windows | 400 | 3200 | ~1120 |
| Full ~20 444 windows | 20444 | **20444** | **~7155** |

Only **~3135** positive **windows** exist in the pool. Sampler could not satisfy `pos_target`; inner loop **broke early** and the fallback filled with **mostly negatives** → model collapsed to all-negative despite CE fix and manifest fix.

**Fix:**

- `sampler.samples_per_epoch: 3200` (config).
- Cap `pos_target` to feasible pool.
- If primary pool exhausted, try alternate pool before stopping.
- Log `[WINDOW-SAMPLER] ...` at loader build.

### 6.6 Operational notes

- Full corpus: **~1 h** before first epoch (collect 23k clips + quality gate + manifest map) with `parallel 1`.
- Each of 6 benchmark models **re-runs** full data prep in serial mode (inefficient; future: cache loaders).
- Progress: `artifacts/logs/train_progress.json` + `[PROGRESS]` heartbeat; overnight log path depends on **cwd** when launching `nohup` (must run from `window_campaign/`).

---

## 7. Quick validation tooling (after `git pull`)

| Script | Time | Purpose |
|--------|------|---------|
| `quick_sampler_sanity.py` | ~2 s | Math + mini simulation of sampler fix |
| `audit_train_loader.py --debug-clips 60 --skip-quality` | few min | `frac_pos` in train batches, `num_samples=3200` |
| `quick_train_signal.py --clips 100 --epochs 1` | few min | One real epoch; need `rec_robo > 0` |

Recommended overnight relaunch only after smoke/signal shows non-zero robbery recall.

---

## 8. Artifacts to preserve / not delete

| Path | Role |
|------|------|
| `artifacts/plans/win_bin_full_baseline_split_manifest.json` | Frozen split |
| `artifacts/core_index.json` | Core-aware labels |
| `artifacts/logs/overnight_benchmark.log` | Run log |
| `artifacts/models/win_bin_full_baseline/modelo_*.pt` | Checkpoints |

Delete stale `.pt` before a **new** full retrain to avoid confusion.

---

## 9. Config snapshot (authoritative: file on disk)

See `window_config.json`. Key fields:

```json
{
  "window_len": 96,
  "stride": 48,
  "mask_policy": "keep_grid_v2",
  "sampler": {
    "samples_per_epoch": 3200,
    "positive_share": 0.4
  },
  "train_defaults": { "checkpoint_metric": "macro_f1", "epochs": 25 },
  "split": { "use_frozen_manifest": true }
}
```

Experiment epochs still read from `model_config.EXPERIMENTS[id-1].epochs` unless `debug_epochs` override.

---

## 10. Roadmap gaps (for analyst context)

From `WINDOW_CAMPAIGN_FINAL_PLAN.md` (not blocking pilot on action clips):

- **B8:** session / production overlap chunks in training index.
- **B9:** automated `replay_eval.py` (ses2, Session A overlap, benign held-out).
- **Four binary cells** (`filtered`, `fp_hardened` variants) — only baseline cell wired.
- **Parity:** train vs prod mask path, FPS, tracker — see `MOTION_FP_INVESTIGATION_BRIEF.md`, `KANVIS_POSE_SHARED_PROJECT_SPEC.md`.
- **Physical 8 s `.npy` chunks on disk:** optional via `materialize_windows.py`; default is in-memory slice.

---

## 11. Open questions for the reviewing agent

1. Is **`samples_per_epoch=3200`** enough for 20k+ windows, or should it scale mildly with GPU time budget while **capping** `pos_target`?
2. Should **quality gate** asymmetry be tuned so robbery cores are not disproportionately dropped?
3. When moving to **on-disk 8 s chunks**, reconcile **`min_clip_seconds: 4`** with 2–4 s motion segments and production variable-length chunks.
4. Should **manifest expansion** append rows instead of replace-window to avoid losing non-manifest identity row (currently 20444→20444)?
5. Best **early-stop rule** for binary FP reduction: macro_f1 vs recall@fixed FP on replay (not yet wired).
6. **Parallel=1** retrains rebuild loaders six times — propose caching strategy?

---

## 12. Related documents in repo

| File | Content |
|------|---------|
| `WINDOW_CAMPAIGN_TRIAL_PLAN.md` | Trial design, chunk math, aug policy |
| `WINDOW_CAMPAIGN_FINAL_PLAN.md` | R2 expert plan, gates, replay requirements |
| `MOTION_FP_INVESTIGATION_BRIEF.md` | Session A, ses2 overlap FP analysis |
| `COMANDOS_PASOS.md` | Operator commands (Spanish) |
| `README.md` | Short operator overview |

---

## 13. One-paragraph executive summary

We train binary motion models on **96-frame (8 s) sliding windows** with **stride 48 (50% overlap)** from existing `data_result` pose clips, with **core-aware robbery labels**, **keep_grid_v2** preprocessing, and a **6-architecture benchmark**. Production scores overlapping 8 s buffers; training mirrors overlap via stride, not yet via explicit session overlap files. A series of bugs caused **zero robbery recall** on full runs while small smokes passed: **wrong CE head**, **manifest copy stripping labels**, **val_acc checkpointing**, and finally **sampler requesting more positives per epoch than exist** when `num_samples` grew with dataset size. Fixes are in `window_sampler.py`, `window_build_loaders.py`, `window_synthetic.py`, `window_loss.py`, `train_window_campaign.py`, and `window_config.json`. Analyst should verify post-fix runs show **`[WINDOW-SAMPLER] num_samples=3200`**, train batch **`frac_pos ≈ 35–45%`**, and **`rec_robo > 0`** by epoch 1–3 on full or medium-sized subsets before trusting overnight benchmarks.