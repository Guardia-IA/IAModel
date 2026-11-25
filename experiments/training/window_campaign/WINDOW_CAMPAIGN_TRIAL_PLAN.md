# Window campaign trial plan (binary only)

**Audience:** IAModel / motion-FP investigation agents (same thread as `MOTION_FP_INVESTIGATION_BRIEF.md`, `MOTION_FP_RETRAIN_PLAN.md`).  
**Language:** English.  
**Status:** Implementation **v1** in `experiments/training/window_campaign/`; training sweep **not started** until preflight on real `data_result` is approved.

---

## 1. Why we are doing this

Production motion uses **8 s chunks (96 frames @ 12 Hz)** with **50% overlap** between consecutive chunks. The legacy campaign trained **one sample per action clip** and then **temporal_resize’d** the whole sequence to 96 frames. That mismatches how the model is scored in the field and is a plausible contributor to **shopping / walking false positives** (e.g. session `ses2`, chunk overlap artefacts).

**Hypothesis:** Train the **same binary robbery detectors** (positive = folders **6** and **14**, negative = everything else) on **sliding 96-frame windows** sliced from existing `poses_full.npy` clips, with **synthetic augmentation applied per window** (not only per full clip), and compare architectures + epoch budgets under val/test.

**Explicit non-goal for this trial:** **No multiclass training.** Multiclass cells have historically underperformed for deployment; we only run **binary** variants analogous to the campaign’s `bin_*` cells.

---

## 2. What is already implemented (v1)

All code lives under:

`IAModel/experiments/training/window_campaign/`

| Component | Role |
|-----------|------|
| `window_config.json` | Window length 96, stride 48, `pose_source=full`, binary positives `[6,14]`, augment profile `fp_hardened` (inherited from `campaign/campaign_config.json`) |
| `preflight_window_campaign.py` | Scan `--data-root`, count clips/windows, optional `artifacts/plans/win_bin_full/window_training_plan.json` |
| `window_index.py` | Slide windows over each clip (`iter_window_slices`) |
| `window_preprocess.py` | Crop + pad short tails (`repeat_last`); `mask_policy=keep_grid` (dense grid, no per-frame drop before slice) |
| `window_dataset.py` | Load slice → optional deterministic + on-the-fly augment → normalize → velocity → resize to 96 only if speed-augment changed length |
| `window_synthetic.py` | Replicate `validate_npy` manifest variants **per window** (same clip-level manifest, multiple window rows with `forced_ops`) |
| `window_build_loaders.py` | Clip-level stratified split, then windows; weighted sampler for binary balance |
| `train_window_campaign.py` | Train one or more `experiment-id` from `model_config.EXPERIMENTS`; checkpoints under `artifacts/models/<cell_id>/` |
| `materialize_windows.py` | Optional: write each window to disk + `windows_index.jsonl` |

**Data input:** Same as today: `--data-root` pointing at `data_result/` (or `GUADIA_DATA_RESULT_ROOT` / `experiments/config.py` `OUTPUT_BASE`).

**Default training cell (config):** `win_bin_full` — binary, `poses_full`, `fp_hardened`-style synthetic settings in JSON (on-the-fly augment + manifest extra views).

### 2.1 Not in v1 yet (please flag if blocking)

| Gap | Notes |
|-----|--------|
| **Four binary “cells”** | Campaign has `bin_full`, `bin_filtered`, `bin_full_hardened`, `bin_filtered_hardened`. v1 config only wires **`win_bin_full`** (full poses + hardened aug profile in JSON). Filtered / baseline aug cells need extra `cell_id` + small config clones — **planned**, not separate scripts yet. |
| **Training parity with `train_campaign.py`** | v1 uses `train_one_epoch` + `CrossEntropyLoss`. Full campaign `fp_hardened` uses `train_one_epoch_advanced` (asymmetric focal, hard-negative mining, etc.). **We should confirm** whether v1 must match before a large sweep. |
| **Mass augment / category aug** | Campaign mass expansion is clip-based; window trial relies on **window count × per-window synth**. May need caps or hard-negative manifests later. |
| **Continuous session / overlap negatives** | No `merged/` session chunks in the trainer yet; only `data_result` action clips. Overlap-hard negatives remain on the roadmap (`MOTION_FP_RETRAIN_PLAN.md`). |
| **`seq_len` mismatch for some experiment IDs** | Experiments with `seq_len: 64` in `model_config` are **forced to `window_len=96`** in `train_window_campaign.py` (architecture still runs at 96). Worth validating that each arch behaves well at 96-only training. |

---

## 3. Binary task definitions we intend to test

Mirror the **campaign binary family** (multiclass **excluded**):

| Planned cell ID | Task | `pose_source` | Augment profile | Positive classes | Purpose |
|-----------------|------|---------------|-----------------|------------------|---------|
| `win_bin_full` | binary | `full` | `baseline` (TBD config) | 6, 14 | Closest to historical `bin_full` but windowed |
| `win_bin_filtered` | binary | `filtered` | `baseline` | 6, 14 | Smoother poses, less NaN leakage |
| `win_bin_full_hardened` | binary | `full` | `fp_hardened` | 6, 14 | **Default in current `window_config.json`** |
| `win_bin_filtered_hardened` | binary | `filtered` | `fp_hardened` | 6, 14 | Hardened + filtered |

**v1 default run:** start with **`win_bin_full_hardened`** (already aligned with `aug_profile: fp_hardened`). Add the other three cells once training-loop parity with campaign is agreed.

---

## 4. Architecture sweep (experiment IDs)

We plan to run the **same reduced architecture battery** already used for binary campaign (`campaign_config.json` → `experiment_ids`), not the full 50+ entries in `model_config.py`.

| Exp ID | Architecture | `seq_len` in config | Epochs in config | Notes for window trial |
|--------|--------------|---------------------|------------------|-------------------------|
| **05** | `tcn` | 96 | 50 | Baseline temporal conv |
| **11** | `dilated_tcn` | 96 | 50 | |
| **12** | `dilated_tcn` | 64 | 40 | Trained at **96** in window trainer |
| **13** | `res_tcn` | 96 | 50 | |
| **37** | `joint_attn` | 64 | 40 | Trained at **96** |
| **40** | `tcn_lstm` | 96 | 50 | |
| **41** | `tcn_lstm` | 64 | 40 | Trained at **96** |
| **48** | `tcn_gru` | 96 | 45 | **In production ensemble today** |
| **49** | `tcn_gru` | 96 | 60 | **In production ensemble today** |
| **50** | `tcn_gru` | 64 | 50 | Trained at **96** |
| **53** | `conformer_lite` | 96 | 50 | Strong val in past bin_full campaign |
| **57** | `ms_tcn` | 96 | 70 | Longest budget |

**Excluded from this trial (campaign already dropped):** shallow TCNs 4/6/8, dilated 10, transformer 25/26, conformer 54, ms_tcn 56, etc. (see `_experiment_ids_comment` in `campaign_config.json`).

### 4.1 Epoch strategy

- **Primary:** Use each experiment’s **`epochs` from `model_config.py`** (table above) unless GPU time forces a cut.
- **Optional follow-up:** For top 3–4 models by val F1 / operational FP@recall, **re-run with +20% epochs** or early-stop on plateau (not automated in v1).
- **`window_config.json` `train_defaults.epochs`: 25** applies only if we later override per-run; current trainer reads **`EXPERIMENTS[exp_id-1].epochs`**.

### 4.2 Architectures *not* in the sweep (open question)

The repo still defines LSTM, GRU, transformer, ST-GCN, PoseCNN2D, GAT-TCN, etc. Campaign **already pruned** them for binary val queue time.

**Question for reviewers:** Should window trial **stick to the 12 IDs above**, or add **one** extra family (e.g. `gat_tcn` or `gru_attn` at `seq_len=96`) in case windowed data benefits from graph/attention?

---

## 5. Ensembles and deployment alignment

After single-model val/test:

1. Reuse campaign-style **ensemble rules** (`mean`, `and`, `cascade`, `majority`) on the **same 12 IDs** or a subset.
2. Prior deployment reference: **48 + 49** (`tcn_gru`, mean @ 0.50); alternate campaign pick **49 + 53**.
3. **Question:** For window-trained checkpoints, should ensemble search prioritize **mean(48,49)** again, or re-grid all pairs on **window val** only?

Threshold sweep: same grids as `campaign_config.json` → `threshold_sweep` (softmax 0.50–0.98, logit margin).

---

## 6. Training data: chunking and augmentation (read this carefully)

### 6.1 Time axis and chunk size

- Each row in `poses_full.npy` is one pose frame at **12 Hz**.
- **One chunk = 96 consecutive rows = 8.0 seconds** of motion (not 9 s). This matches production motion buffers.
- We **do not write** new `.npy` files on disk by default: the trainer **cuts slices in memory** from the original clip file. Optional `materialize_windows.py` can export chunks if needed for QA.

### 6.2 Chunking rules (every category, no exceptions)

**All clips in `data_result` are windowed the same way**, including:

- **Negative folders** (e.g. shopping category 3, walking, etc.) — often longer, mixed actions in one file.
- **Positive staged robberies categories 6 and 14** — short, “clean” clips you recorded (stand → gesture → stand). **They are still sliced.** There is **no** “leave the full `.npy` as one training sample” path in `window_campaign`.

Rationale:

- Deployment always sees **96-frame windows** (with overlap), so positives must be learned **inside that geometry**, including partial gestures at chunk boundaries.
- Skipping chunking only for 6/14 would reintroduce train/serve skew.

**Algorithm** (`window_index.iter_window_slices`, defaults `window_len=96`, `stride=48`):

| Case | Behaviour |
|------|-----------|
| `T < 96` rows | **One** training window: rows `[0:T)` then **pad** to 96 (`repeat_last` by default). Typical for **~6 s** staged clips (≈72 frames). |
| `T ≥ 96` | Sliding windows: start `0, 48, 96, …` while `start + 96 ≤ T`. Each window is **exactly** 96 rows (no padding). |
| Last partial tail | If the clip end is **not** covered by the sliding grid, v1 may add one **final** aligned window `[T-96:T)` when stride does not land on the end (see code). Shorter tails **smaller than 96** that are never covered become **only** the padded short-window case above. |

**Important:** Chunking is **independent of binary label**. Folder 6/14 only sets `label=1`; the slicer does not treat them differently.

### 6.3 Worked example: 16 s clip at 12 Hz

Assume **16 s × 12 Hz = 192 rows** in one `.npy` (any category).

With `window_len=96` and `stride=48` we get **three** full chunks:

| Window index | Row range (half-open) | Time (approx.) | Overlap with previous |
|--------------|----------------------|----------------|---------------------|
| 1 | `[0:96)` | 0–8 s | — |
| 2 | `[48:144)` | 4–12 s | 50% overlap with window 1 (rows 48–95 shared) |
| 3 | `[96:192)` | 8–16 s | 50% overlap with window 2 (rows 96–143 shared) |

So from **one** 16 s `.npy` we derive **three** base training windows (same file path, different `start`/`end`). This mirrors production’s overlapping 8 s buffers.

**Shorter positive example:** 14 s → 168 rows → **two** full windows (`[0:96)`, `[48:144)`); the last 24 rows (14–16 s) are only seen inside window 2, not as a separate padded tail unless `T < 96`.

### 6.4 Data augmentation: legacy clip vs window trial

#### Legacy clip-level campaign (what we did before)

- **One training row per `.npy` per user** (whole clip resampled to `seq_len` with `temporal_resize`).
- Synthetic data: **`validate_npy` manifests** at **clip** granularity — typically **one identity view + several deterministic variants** per clip (often discussed informally as **~5 synthetic views per `.npy`** when manifest extra views + category/mass augment are enabled). Those variants warp **the entire clip**, then resize to 96.

#### Window campaign (what we do now)

| Concept | Change |
|---------|--------|
| **Unit of augmentation** | **Each 96-frame chunk** (each `(pose_path, start, end)` window), **not** the whole `.npy`. |
| **Deterministic synth** | The same clip-level `validate_npy` JSON is reused, but **every window** from that clip can be expanded into **multiple dataset rows** with different `forced_ops` (see `window_synthetic.expand_windows_with_manifest_extra_views`). |
| **On-the-fly synth** | Still applied in `WindowPoseDataset` **after** the chunk is cut — random mirror/rotate/speed/etc. **per sampled window** (stochastic; not a fixed file count). |

**Mapping “~5 variants per clip” → “~5 variants per chunk”:**

| Stage | Legacy (1 clip) | Window trial (per chunk) |
|-------|-----------------|---------------------------|
| Base physical windows from 16 s file | 1 clip → 1 resampled sequence | 1 clip → **3** chunks (example above) |
| Deterministic manifest views | ~**5** rows **per clip** (1 identity + 4 ops — exact number depended on manifest / `extra_manifest_views`) | ~**5** rows **per chunk** (1 identity + 4 ops on **that slice only**) |
| **Order-of-magnitude training rows** (deterministic only, 16 s example) | ≈ **5** | ≈ **3 × 5 = 15** from the same original `.npy` |

**Config (locked for this trial):** `window_config.json` sets `extra_manifest_views_per_clip: 4` → **1 identity + 4 manifest = 5 deterministic train rows per window** when a `validate_npy` manifest exists for that clip. On-the-fly augment is **additional**, not counted in that “5”. See **§6.7** for who gets augment.

**Val / test:** No on-the-fly random augment; one row per window (deterministic policy only).

We **do not** duplicate `.npy` files on disk for each variant unless you run `materialize_windows.py`; expansion is **logical rows** in the dataset index.

### 6.5 Summary table

| Setting | Value | Rationale |
|---------|-------|-----------|
| Window length | **96 frames (8 s @ 12 Hz)** | Match prod chunk |
| Stride | **48** (default) | 50% overlap between consecutive chunks |
| Categories **6 / 14** | **Same chunking as all other categories** | Align staged positives with deployment windows |
| Split | 80/10/10 by **clip UID** | All windows from one clip stay in one split |
| Binary labels | 1 if folder ∈ {6, 14} | Same as `bin_full` |
| Deterministic synth | `validate_npy` per chunk | Target **~5 views/chunk** (config: `extra_manifest_views_per_clip: 4`) |
| Stochastic synth | On-the-fly in train | Extra diversity beyond the fixed 5 |
| Materialize to disk | Optional | QA only; default train is in-memory |

### 6.7 Augmentation policy — **team decision** (please review)

This is the **default policy** encoded in `window_config.json` → `augmentation_policy` and `synthetic_variants`. Other agents should say if any rule should change before GPU sweep.

| ID | Decision | Choice |
|----|----------|--------|
| **A1** | Which windows get augment in **train**? | **Every** sliding window: first chunk, **overlap** chunks, and final aligned chunk — **no** “augment only non-overlap” shortcut. |
| **A2** | Categories **6 and 14** (staged positives)? | **Yes — same augment as negatives.** Positives are scarce; synth on each 8 s view (and each overlap view) improves recall and boundary robustness. We do **not** disable manifest/on-the-fly for 6/14. |
| **A3** | Categories **3, 4, 5**, etc. (negatives)? | **Yes — full augment** on every train window. Hard-FP reduction comes from **fp_hardened** sampling/loss (once ported), not from skipping augment on shopping-like folders. |
| **A4** | Deterministic views | **5 per window** (identity + 4 manifest ops) when JSON exists; global deterministic grid still sampled with `train_deterministic_prob` for windows **without** manifest. |
| **A5** | Stochastic on-the-fly | **On** in train (`augment_on_the_fly: true`, profile `fp_hardened`); **off** in val/test. |
| **A6** | Overlap-specific rule? | **None.** Overlap windows are **first-class** training units because **production scores every overlapping buffer**; FPs often appear on overlap boundaries (ses2). Training overlap chunks **with** augment matches that risk. |
| **A7** | Short 6/14 (`T < 96`, padded window) | **Same 5 + on-the-fly** on the single padded window. Acceptable trade-off: maximize recall on short staged clips; optional follow-up is **downweight** padded windows in the sampler (not implemented in v1). |

**Example (16 s negative clip, with manifest):**

- **3** physical windows (§6.3) × **5** deterministic rows ≈ **15** indexed train rows, **before** random on-the-fly resampling each epoch.
- **Same math** if the clip is folder **6** or **14** (only the binary label changes).

**What we explicitly rejected (for now):**

| Alternative | Why not |
|-------------|---------|
| Augment only the first window per clip | Under-trains overlap regions where prod fires motion. |
| No augment on 6/14 | Train/serve mismatch; too few positive examples after windowing. |
| No augment on “middle overlap” windows | Those are exactly the tensors prod double-counts through overlap. |
| Clip-level augment only (legacy) | Whole-clip warp + resize ≠ prod chunk geometry. |

### 6.8 Open point for reviewers

Should **very short** 6/14 clips (`T < 96`, heavy padding) be **downweighted** in the `WeightedRandomSampler` (not excluded)? v1 keeps full augment and **no** downweight.

---

## 7. Proposed execution checklist

1. **Preflight** on production `data_result`:  
   `python3 preflight_window_campaign.py --data-root <path> --write-plan`
2. Confirm window counts per label (expect many more negatives than positives → sampler on).
3. **Pilot:** `train_window_campaign.py --experiment-id 48 49` on one GPU.
4. **Full sweep:** all 12 IDs × default cell `win_bin_full_hardened` (and later the other three binary cells).
5. Export val metrics + threshold grid; compare to **legacy clip-level `bin_full`** campaign run.
6. Offline replay: `score_pose_ensemble.py` / session chunks (`ses2`, merged overlaps) — **outside** this folder, same as investigation brief.

---

## 8. Success criteria (draft)

| Metric | Target (draft) | Comment |
|--------|----------------|---------|
| Val **F1** (positive class) | ≥ legacy `bin_full` best single model | Not necessarily ≥ ensemble 49+53 yet |
| Val **recall** @ operational point | ≥ 60% | Align with campaign `target_recall_pct` |
| **FP rate** on held-out negatives | ↓ vs clip-trained 48/49 | Especially cat 3 / shopping-like |
| **ses2 / overlap replay** | ↓ motion FP on chunk_002-like segments | Qualitative + scored NPys |

---

## 9. Questions for other agents

Please comment on whether this plan is **complete**, **missing risks**, or **should be narrowed/expanded**:

1. **Cells:** Is running all **four** binary variants necessary in v1, or start only `win_bin_full_hardened`?
2. **Loss / mining:** Must v1 **port `fp_hardened` train_opts** (asymmetric loss, hard-negative mining) into `train_window_campaign.py` before any sweep?
3. **Stride:** Default **48** vs ablation **96** — which should be primary for matching prod overlap FP?
4. **Architectures:** Keep exactly **12 campaign IDs**, or add 1–2 graph/attention variants?
5. **Ensembles:** Fixed pairs (48+49, 49+53) vs full pair grid on window val?
6. **Data gaps:** Without **continuous session windows** in v1, is clip-only window training **still worth a full 12-arch sweep**, or wait until session NPys are in the index?
7. **Epochs:** Train all models to config epochs, or cap at e.g. **50** for first pass and only extend winners?
8. **Anything else** from `MOTION_FP_RETRAIN_PLAN.md` Phase 0–1 (preprocess v2, valid_mask policy) that must land **before** this trial?
9. **Augment policy §6.7:** Agree with **A1–A7** (augment **all** train windows including overlap + **6/14**, 5 deterministic + on-the-fly)? If not, which windows or categories should be excluded?

---

## 10. References

| Document / path | Content |
|-----------------|--------|
| `window_campaign/README.md` | CLI and artefact layout |
| `campaign/campaign_config.json` | Binary cells, experiment ID list, threshold / ensemble defaults |
| `training/model_config.py` | `EXPERIMENTS` definitions (arch, epochs, lr, `seq_len`) |
| `campaign/MOTION_FP_INVESTIGATION_BRIEF.md` | Motion FP context, prod 48+49, ses2 overlap |
| `campaign/MOTION_FP_RETRAIN_PLAN.md` | Longer train-first roadmap |

---

*GuadIA IAModel — window campaign binary trial plan. Please reply with additions or blockers before we launch the full GPU sweep.*