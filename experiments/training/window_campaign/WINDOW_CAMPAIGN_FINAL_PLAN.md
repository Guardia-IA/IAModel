# Window Campaign — Final Plan (definitive decisions for the implementation agent)

**Audience:** the agent that implemented `IAModel/experiments/training/window_campaign/` (v1).
**Reviews this resolves:** `WINDOW_CAMPAIGN_TRIAL_PLAN.md` (spec) + three independent technical reviews of it.
**Related:** `MOTION_FP_INVESTIGATION_BRIEF.md`, `MOTION_FP_RETRAIN_PLAN.md`. Where this document disagrees with the earlier retrain plan, **this document wins** (see §10 for the corrections).
**Language:** English.

---

## 0. Verdict

- **Approved:** the window-training design as the base for a **pilot** (96 frames @ 12 Hz, stride 48, same chunking for every category, binary 6|14 vs rest, window-level augmentation, split before expansion, preflight, optional materialization, 48/49 as baseline).
- **Not approved yet:** the full 12-architecture sweep. As built, v1 would most likely produce good-looking clip/window validation numbers **without fixing ses2**, and could even lose recall on true split robberies, because of the issues in §2.
- **Core diagnosis of the risk:** windowing fixes the *geometry* of the training samples but not their *labels*. With the current rules the trainer would replicate wrong positive labels (stand-only windows labeled robbery), plus a padding cue that only positives have, across many more windows and with more precision.
- **Priority order** (all reviews agree on this, the arch battery is last): (1) core-aware positive labels, (2) padding handled so it carries no class information, (3) clip/session-balanced sampling with limited synthetic expansion, (4) loss that is audited and is what its name says, (5) hard negatives from real continuous sessions, (6) replay-based selection with split-overlap recall as a hard constraint, (7) only then architectures/features.

**Sequence in one line:** Stage 0 code fixes + audits → Stage 1 pilot (clips only, 48/49) → Stage 2 add real sessions as negatives → Stage 3 reduced sweep → Stage 4 conditional features/architectures → Stage 5 release retrain.

---

## 1. Decision log (where the reviews disagreed)

| Topic | Positions in the reviews | **Decision** and reason |
|---|---|---|
| Label all windows of 6/14 as positive | Spec: yes. Reviews: no (core-aware / ignore) | **No.** Core-aware labels with an `ignore` state. Blocking (B1). |
| Stand-only windows of positives | drop / weight 0.2 / ignore | **Ignore (weight 0)**, not negative: they can contain partial gestures (B1). |
| Padding of short 6/14 clips (`repeat_last`) | weight 0.5 / `is_padded` channel / monitor | **Remove the class confound first** (add padded negatives so padding is not a positive-only cue), record `pad_frac`, down-weight; `is_pad` channel + pad-aware pooling as a Stage 3 ablation (B2). |
| Mask policy | mix 50% mask-filtered + 50% keep_grid with Δt velocity / keep_grid only / ablate | **keep_grid is the canonical training policy, but only with validity gating** (B5). Do **not** mix 50/50 as the primary design. Legacy mask-drop path only as a fallback fraction if production cannot switch. |
| Loss | port asymmetric focal first / weighted BCE first | **Weighted BCE baseline first**, focal as a controlled ablation. `fp_hardened` may not be named in any run unless the effective loss/mining matches (B4). |
| Where ses2 goes | train as high-weight hard negative / held out as acceptance | **Held out** of training and mining until gates pass; trained only in the final release retrain (B8). |
| Continuous session negatives | block the sweep / pilot first | **Pilot on clips is fine; the sweep is blocked** until session chunks + production overlaps are in the index (Stage 2). |
| Architectures | keep 12 / drop seq_len=64 and add GAT, attention / only 48,49,53,57 | **Start with 48, 49, 53, 57.** Drop the four IDs that are forced from 64 to 96 (12, 37, 41, 50). Add at most one attention-pooling variant. No GAT/ST-GCN in the first sweep (§4). |
| Extra input features | arch more valuable / features more valuable / later | **Features before new architectures, but Stage 4 only**, after labels/negatives are fixed (§4). |
| Four binary cells | all four / only `win_bin_full_hardened` / reduced factorial | **Start with one cell** (full/keep_grid). `filtered` is a later ablation (§3). |
| Synthetic variants | 5 deterministic per window / start smaller | **Start with identity + 1 deterministic** (+ moderate on-the-fly); test 4 later. |
| Epochs | per `model_config` / cap 50 / 20–30 pilot | **Pilot 25 epochs, sweep cap 50**, early stopping on replay metrics, not clip F1. |
| Ensembles | fixed pairs / full grid | **≤ 5 candidate ensembles**, evaluated on replay (§5). |
| Per-window-type thresholds | proposed by one earlier plan | **Not until the recall set has ≥ 20 positive overlaps.** One global threshold, chosen on replay. |

---

## 2. Stage 0 — blocking changes before any GPU time

### B1. Core-aware labels for staged 6/14 (BLOCKING)

**Problem.** `chunking is independent of label`: a window sliced from a 6/14 clip gets `label=1` even if it only covers the stand prefix/suffix. Staged clips are `stand ~1 s → core → stand ~1 s`; for clips longer than 8 s some windows can be almost all standing. The model would learn "standing before/after a gesture = robbery".

**Change.**
1. Add a per-clip `core_index` (file `core_index.json` next to the clip manifest): `{clip_uid: {core_start, core_end, method, confidence}}` in frames.
2. Compute it automatically (no micro-gesture labeling):

```python
def detect_core(poses, hz=12):
    # poses [T, 8, 2] raw, NaN = missing. Joint order: verify in code
    # (brief: wrists = 4,5; hips = 6,7; shoulders/elbows = 0..3).
    torso = np.nanmedian(np.linalg.norm(shoulder_mid(poses) - hip_mid(poses), axis=-1))
    wrist_speed = np.nanmax(np.linalg.norm(np.diff(poses[:, [4, 5]], axis=0), axis=-1), axis=1) * hz / torso
    wrist_speed = smooth(wrist_speed, k=5)
    active = wrist_speed > 0.25 * np.nanpercentile(wrist_speed, 95)
    segs = merge_close(runs(active), max_gap=6)          # frames
    core = largest_segment(segs)                          # or span of the segments carrying 80% of the energy
    return dilate(core, 6)                                # +-0.5 s context
    # fallback if no clear segment: trim 12 frames (1 s) from each end
```
   Use **wrist speed only**. Do not use wrist–hip distance as the activity criterion: standing with hands at the sides is also "wrist near hip".
3. Plot ~20 clips once (poses + speed + detected core) and spot-check; store `confidence` (ratio of peak to median speed). Clips with low confidence use the fallback and are logged. One manual correction pass is acceptable; no further labeling.
4. Window label from the intersection (`core_len = core_end − core_start`, `ov = |core ∩ window|`):

| Condition | Target | Weight |
|---|---|---|
| `ov ≥ min(0.6·core_len, 36)` | 1 | 1.0 |
| `0.2·core_len ≤ ov < min(0.6·core_len, 36)` | 1 (partial) | 0.3 |
| `ov < 0.2·core_len` (stand-only / context) | **ignore** | 0 (excluded from sampler and loss) |

   For `T < 96` clips (single padded window) the whole core is inside → positive.
5. Negatives from other categories keep `label=0`. Do not turn ignored windows into negatives.
6. Persist `core_overlap_frames`, `label_state ∈ {pos, partial, ignore, neg}` per window in the index.

**Test.** For 10 hand-checked clips, the label states of every window match the plot; count of `ignore` windows per class is logged in the preflight plan.

### B2. Padding must not be a class cue (BLOCKING for the baseline)

**Problem.** Short staged positives (~6 s ≈ 72 frames) become one window padded with `repeat_last` (≈ 24 static frames). Negatives are mostly long clips → full 96-frame windows with **no** padding. So "padded static tail" is correlated with the positive class — the same shortcut hypothesis (H1) that already explains the 4 s probe (48 frames + 48 static) scoring 0.948 while the unpadded first 16 frames score ~0.

**Change (pilot-level, keeps the 32-channel input).**
1. Record `pad_frac = pad_rows / 96` for every window; log its distribution **per class** in the preflight.
2. **Balance padding across classes:** with probability `padded_negative_prob` (start 0.15) turn a negative window into a truncated-and-padded one: keep `T ∈ [48, 95]` rows, pad with the same rule as positives. After this, `pad_frac` distributions of positives and negatives must be similar (report two-sample KS statistic; target `< 0.1`).
3. **Down-weight heavily padded windows:** `pad_frac > 0.25 → weight × 0.5` (starting value, ablate; applies to both classes).
4. Require that active motion (the detected core) occurs **before** the padded tail in padded positives (otherwise the repeated frames imitate a "braking" robbery).
5. **Audit the model heads:** find out how `tcn_gru` pools (last GRU state? mean?). If it uses the last state, repeated frames dominate the output. Report this in the Stage 0 notes.
6. Stage 3 ablation: `is_pad` input channel + pad-masked pooling (attention or mean over non-pad frames) instead of `repeat_last` influence.
7. Stage 2 alternative for short positives: **context fill** — instead of repeating the last frame, fill the missing frames with benign frames from real sessions (blend 3 frames at the seam), so no padded windows exist at all.

**Test.** Before/after table: `pad_frac` by class; `p_robo` vs truncation length `T ∈ {24,32,48,64,80,96}` on benign windows for the trained checkpoint (must be flat, `|Δp| < 0.10`).

### B3. Clip/session-balanced sampling and limited synthetic expansion (BLOCKING)

**Problem.** `5 deterministic rows × N windows × on-the-fly augment` multiplies a long clip many times more than a short one. A class-balanced `WeightedRandomSampler` does not balance *clips*, so one long recording can dominate the epoch.

**Change.** Hierarchical sampling per batch item:
1. sample class/source (positive core, positive partial, cat 3/4, other negatives, session negatives, hard negatives);
2. sample a **clip or session** uniformly within that source;
3. sample a window of that clip (respect `label_state != ignore`);
4. sample a variant (identity or augment).

- Cap windows per clip per epoch (`max_windows_per_clip_per_epoch`, start 4) and per session.
- `extra_manifest_views_per_clip: 1` (identity + 1) for the pilot; test 4 in Stage 3. On-the-fly augment: moderate (mirror, small noise, speed ×[0.85, 1.15], keypoint dropout), label-preserving.
- Positive share per batch: start at **35–40%** (v1 uses 50/50). With a 50/50 sampler the model is calibrated for a 50% prior while production sees a tiny prior, which pushes confidences up on ambiguous gestures. The threshold is chosen on replay (§5), but a smaller positive share is also an ablation: `{0.5, 0.35}`.

**Test.** Log effective samples per clip/session per epoch; max/median ratio must be `< 3`.

### B4. Loss audit and honest naming (BLOCKING)

**Problem.** The spec says the default cell is `win_bin_full_hardened`, but the trainer uses `train_one_epoch` + `CrossEntropyLoss`; `train_one_epoch_advanced` (asymmetric focal, hard-negative mining) is not wired. The name does not guarantee behavior.

**Change.**
1. Every run writes `run_config.json` **and** stores it in the checkpoint with: `experiment_id, architecture, seq_len, loss_type, positive_weight, negative_weight, focal_gamma_pos, focal_gamma_neg, asymmetric_loss_params, hard_negative_mining_enabled, hard_negative_weight, sampler_type, positive_share, windows_per_clip_cap, augmentation_profile, extra_manifest_views, mask_policy, padding_policy, pad_weight, label_mode (core_aware | all_windows), preprocess_version, git_commit`.
2. Rename cells by what they do. Until the advanced loss is truly active the run is **"window-distribution baseline"**, not "hardened".
3. Baseline loss: per-example **weighted BCE** (or CE with per-sample weights), normalized by `Σ w`. Starting weights (to be ablated, not optimal values):

| Example | Weight |
|---|---|
| ordinary negative | 1.0 |
| cat 3/4 negative | 1.5 |
| mined session hard negative | 2–3 (capped, see Stage 2) |
| positive core window | 2.0 |
| partial-core positive | 0.3 × 2.0 |
| positive split-overlap (real or synthetic) | 3.0 |
| ignored | 0 |
| heavily padded | × 0.5 |

4. **Saturation control** (logits of +22 were observed): label smoothing ε = 0.02 (start) or a small logit penalty `β·mean(logit²)`, β ≈ 1e-3. Ablate.
5. **Focal/asymmetric loss** (γ₋ = 2.5, γ₊ = 1.0 from the old campaign) is a **Stage 3 ablation** on top of the weighted-BCE baseline. Do not combine strong focal modulation with large hard-negative weights at the same time. Port it adapted to window-level mining, not copied blindly.

### B5. keep_grid parity and validity handling (BLOCKING)

**Why.** `keep_grid` (no per-frame drop before slicing) is the right canonical policy, but it is **not** automatically parity-safe: (a) production applies the `valid_mask` filter on regular chunks and not on overlaps; dropping frames then `add_velocity` creates teleport velocities (ses2 `chunk_002` 0% → 99%, Session A); (b) if invalid frames stay in the grid as zero-filled poses, the same spikes appear as pose→0 jumps and the per-frame normalization (joint mean/scale) is contaminated by zeros.

**Change (preprocess "keep_grid_v2").**
1. Normalize each frame using **visible joints only**; frames with `< 5` visible joints get `frame_valid = 0` and zeroed pose.
2. Velocity `v_t = x_t − x_{t−1}` **only if both frames are valid**, else 0. Clip `|v|` to the train-set 99th percentile.
3. Optional short-gap fill: linear interpolation for gaps `≤ 3` frames (ablate).
4. Augmentation: random `frame_valid` dropout up to 20% with the same rules (grid preserved), random keypoint dropout.
5. Do **not** use "50% mask-drop + Δt-scaled velocity / 50% keep_grid" as the main design. Two policies in one model hides which one production will use. Fallback only: if the orchestrator cannot switch to keep_grid_v2 soon, add a 20–25% fraction of legacy-path windows (mask-drop with velocity divided by the real elapsed Δt) so the model is not brittle on the deployed path.
6. **Deployment contract:** checkpoint metadata carries `preprocess_version` and `mask_policy`; the orchestrator must apply the same function (single shared module used by training, offline scripts and Kanvis-Orchestrator); log `preprocess_version` in `p1.json`. A keep_grid-trained model must **never** be deployed into the mask-dropping path without replaying under that path.
7. Replay harness reports every checkpoint under both `prod_legacy` (mask-drop for chunks, no filter for overlaps) and `keep_grid_v2`.
8. Verify first: what `window_preprocess.py` currently does with NaN and invalid frames, and whether `normalize_sequence` is computed with or without them.

**Test.** `|Δp| < 0.10` on benign windows between mask on / mask off under `keep_grid_v2`; golden tensors PyTorch vs Triton/TensorRT `|Δp| < 1e-3` on ≥ 20 real production tensors (resolve `chunk_006`, `chunk_004`, `chunk_003_004` mismatches before trusting offline numbers). Verify SHA/campaign id of `modelo_48` and `modelo_49`.

### B6. Windowing index audit (unit tests)

The spec's 14 s example is inconsistent with the stated rule: for T = 168 with stride 48 the sliding starts are 0 and 48 (`48 + 96 = 144 ≤ 168`), the tail of 24 frames is uncovered, so the "final aligned window" rule adds `[72:168)`. Add a `min_tail_for_final_window` (start 24 frames) so near-duplicates are not created (e.g. T = 97 would produce `[0:96)` and `[1:97)`), and write tests:

| T | Expected windows (intended rule) |
|---|---|
| 48 | `[0:48)` padded to 96 (`pad_frac 0.50`) |
| 72 | `[0:72)` padded (`0.25`) |
| 95 | `[0:95)` padded (1 row) |
| 96 | `[0:96)` |
| 97 | `[0:96)` only (tail 1 < `min_tail`) |
| 144 | `[0:96)`, `[48:144)` |
| 168 | `[0:96)`, `[48:144)`, `[72:168)` (tail 24 ≥ `min_tail`) |
| 192 | `[0:96)`, `[48:144)`, `[96:192)` |
| 240 | `[0:96)`, `[48:144)`, `[96:192)`, `[144:240)` |

Also verify: no accidental duplicates, the final aligned window is not over-weighted by overlap (down-weight by duplicated fraction), original timestamps are kept, and `core_start/core_end` are translated correctly into each window.

### B7. Split hygiene

- Split **before** expansion/augment and group by recording unit: for staged 6/14 by person/recording day; for session data by session; keep all variants and all windows of a clip in one split.
- Do not use the same sessions for mining and for the independent test.
- Locked sets (never used for training, mining, threshold choice or early stopping): see §6.

### B8. Session negatives index (required before the sweep, not before the pilot)

Add a second data source next to `data_result`: continuous sessions from `merged/` exports (`chunk_00k/poses_full.npy`, `valid_mask.npy`, `p1.json`).

- `window_type ∈ {action_clip_window, production_chunk, production_overlap, session_crop}`; overlap = `concat(a[48:96], b[0:48])` built exactly as in production.
- Negatives = sessions with no confirmed event. Because real robberies are rare, a session with no confirmed/suspected event is overwhelmingly benign: use **many** sessions (target ≥ 50; the 15–20 suggested in one review is a minimum, not a goal). Exclude sessions with confirmed/suspected robbery or an unreviewed stage-2/3 alarm.
- **Hard negatives** (windows that fire) need a **session-level yes/no confirmation** by a human; no micro-gesture labeling.
- **ses2 is held out** (not in training, not in mining, not in threshold selection). Train on other known FP sessions (`p1.json` alarms on benign visits). Final release retrain may include ses2 only after the gates pass on the held-out sets.
- Mining rule: replay current checkpoints over train-split benign sessions; collect windows with `p > 0.2` (not only `≥ 0.5`); de-duplicate neighbors (stride ≥ 12 frames); group by session/pattern; weight `w = 1 + α·min(p_prev^γ, c)` with α = 2, γ = 1, c = 1 (so 1–3); cap per session.

### B9. Replay harness before the pilot

`replay_eval.py` (or an extension of `score_pose_ensemble.py`) must run any new checkpoint on: ses2 (11 windows), the 4 s probe, Session A overlap, `09e7c4…` overlap, and the held-out benign session set — reporting **chunk and overlap separately**, per model and ensemble, with and without the mask path, plus max score per session. It must reproduce the production numbers of Appendix A for 48/49 where parity holds.

**Gate G0 (exit of Stage 0):** B1–B7 and B9 implemented with tests; `run_config.json` emitted; padding-sensitivity sweep on the existing 48/49 checkpoints reported (this shows how much of ses2 is padding/mask artifact vs content); replay harness reproduces Appendix A.

---

## 3. Staged execution

### Stage 1 — Pilot: "window-distribution baseline" (clips only)

- Models: **48 and 49** (identical architecture to production). 25 epochs, checkpoint every epoch, early stop patience 8 on the selection metric below.
- Ablation ladder on model 48 first (then repeat the best on 49):

| Run | Content | Question |
|---|---|---|
| R0 | legacy clip-level 48/49 (already trained) | reference |
| R1 (optional, only if GPU is free) | v1 as built: all-window positive labels, 5 variants, CE | quantifies what the fixes buy |
| **R2** | B1–B6 applied: core-aware labels, padding balanced, hierarchical sampler, identity+1 variants, weighted BCE, keep_grid_v2 | does windowing + correct supervision help? |
| R2f | R2 + asymmetric focal | does focal add anything over weighted BCE? |

- Do **not** call this FP-hardened. Expectations: it validates the pipeline and the distribution/label fix; it will not by itself close ses2 because there are no real session negatives yet.
- **Gate G1:** on replay (ses2 held out), R2 vs R0/R1: fewer ses2 windows ≥ 0.5, probe lower, **no** recall loss on split-overlap (`09e7c4…`) and on held-out staged clips; padding invariance test passes; per-class `pad_frac` distributions matched; sampler balance test passes.

### Stage 2 — Real session negatives + overlaps (this is the "1.5" phase)

- Add B8 data: session chunks, production overlaps, session crops, mined hard negatives (capped weights).
- Add **synthetic split-overlap positives** (small share, ~5%): cut a staged core so it straddles the frame-48 seam of an overlap window, with benign context on both sides (blend 3 frames). Apply the **same seam jitter/jumps symmetrically to negative overlaps** (including track-switch-like jumps and visibility drops) so "seam" never predicts a class.
- Optional: context-fill for short positives instead of `repeat_last` (B2.7).
- Starting composition per epoch (ablate): real benign chunks/crops 30%, real benign overlaps 10%, cat 3/4 slices 15%, other categories 10%, mined hard negatives 10%, positive cores 15%, synthetic split positives 5%, real split/in-store positives 5%.
- Models: 48, 49, **53, 57**. 
- **Gate G2:** ses2 (held out) windows `< 0.5` on at least the winning checkpoint, held-out benign-session any-alarm rate clearly reduced vs 48+49 baseline, recall constraints (§6) satisfied. If not satisfied, do **not** start the sweep; diagnose with the Phase-0 style sweeps.

### Stage 3 — Reduced sweep and controlled ablations (one change per run)

Order: (1) `extra_manifest_views` 1 → 4; (2) positive share 0.5 vs 0.35; (3) focal vs weighted BCE; (4) label smoothing / logit penalty; (5) `is_pad` + `frame_valid` channels with pad-masked pooling; (6) frame-dropout augmentation strength; (7) `filtered` pose source (the `filtered` cells only now); (8) padded-sample weight 0.5 vs 1.0 vs excluded. Architectures per §4. Epoch cap 50, no 70-epoch runs unless validation by replay keeps improving.

### Stage 4 — Conditional upgrades (only if gates are still missed)

Kinematic input channels, auxiliary head, graph architecture, HAR-teacher soft targets — see §4. Stop condition: if Stage 3 already meets the gates, do **not** add features.

### Stage 5 — Release retrain

Retrain the chosen configuration with ses2 and the remaining FP sessions included, re-run every locked set except ses2, and run the parity gate on the final engine.

### Checkpoint selection (all stages)

Lexicographic, never by clip/window F1 alone:
1. Hard constraints: split-overlap recall (`09e7c4…` and recorded positives) and held-out staged recall within the agreed tolerance of the 48+49 baseline (start: within 3 points); partial-core recall not collapsed.
2. Minimize benign-session any-alarm rate on the **held-out benign sessions**.
3. Tie-break: window FP rate at 0.5, then max-score-per-session distribution, then val loss.
- Stop early if val loss improves while session FP rate worsens.

---

## 4. Models and inputs

| Priority | ID / name | Purpose |
|---|---|---|
| P0 | **48, 49** `tcn_gru` | production baseline; measures the effect of data/labels with the architecture fixed |
| P1 | **53** `conformer_lite` | local + global temporal context (watch overfitting on small positives) |
| P1 | **57** `ms_tcn` | multi-scale temporal patterns (reach vs sustained concealment); cap epochs, do not run 70 by default |
| P2 | 13 `res_tcn`, 05 `tcn`, 11 `dilated_tcn` | light baselines / ensemble diversity, only after Stage 2 |
| P2 | `tcn_gru_attn` (new; same as 49 with temporal attention pooling over non-pad frames) | removes the dependence on the GRU last state, directly addresses B2; **raise to P1 if the B2 audit shows last-state pooling** |
| Dropped from the first sweep | 12, 37, 41, 50 (`seq_len=64` forced to 96) | receptive fields/strides sized for 64; redundant with 11, 40, 48/49 |
| Not in first sweep | `gat_tcn`, `st_gcn` | only 8 upper-body keypoints; the confusion is mostly semantic/temporal (3/4 vs 6/14 depends on object ownership); reserve for Stage 4 if kinematic features are insufficient |

**Input features (Stage 4, controlled ablation, via the shared preprocess module so deployment parity is automatic):**
wrist–hip distance (ipsilateral/contralateral) normalized by torso length, minimum of the two and its temporal derivative; wrist-to-torso distance; wrist speed; approach/retreat velocity; hand-to-hand distance; persistent proximity of a hand to the body; wrist height relative to shoulders; `frame_valid` and `is_pad` if not already added. About +5–10 channels. Do not use HAR `hand_to_body` as a veto; its geometric ideas become trainable inputs.

**Auxiliary head (Stage 4):** action family (0/3/4/6/14), loss weight ≈ 0.3; inference keeps the binary head. Optional HAR soft target (≤ 0.1, weight ≤ 0.2) only on windows where HAR says `hand_to_body == False`, mask coverage ≥ 75%, **and** the session is human-confirmed benign.

---

## 5. Ensembles and thresholds

- Start with fixed pairs: **48+49** (production), **49+53** (campaign reference), **48+57** (temporal diversity) built from the **new** checkpoints.
- After Stage 2, take the **top-4 single models by replay** and evaluate at most 3–5 ensembles. Also compare fusion rules: MEAN vs min/AND vs geometric mean (ses2 `chunk_001`: 49 high, 48 low; `chunk_003_004`: 48 high, 49 ≈ 0).
- Threshold: 0.50 is a reference, not a truth. Choose one global threshold on replay using held-out recall sets and benign-session FP rate. **Never tune the threshold on ses2** and then declare ses2 solved.
- Per-window-type thresholds (chunk vs overlap) only when the recall set has ≥ 20 positive overlaps.

---

## 6. Evaluation sets and gates

### Sets (all split by session / recording unit)

| Set | Role |
|---|---|
| FP-train | confirmed-benign sessions that currently fire (hard-negative source) |
| **FP-heldout (locked)** | other confirmed-FP sessions; **ses2 lives here** |
| Benign-heldout | ≥ 200 random benign sessions for sessions-with-any-alarm per 1000 |
| Staged-heldout | staged 6/14 held out by person/day: core inside chunk, core across seam, partial-core sweep |
| Real positives | `09e7c4…` + **15–20 new staged robberies recorded through the real production pipeline** (real camera, real chunking/poses), about half crossing a chunk boundary/overlap. Session-level labels only. This is the only way to get a credible recall estimate: today there is a single real split-robbery session. |

### Gates

| Test | Pass criterion |
|---|---|
| Data integrity | splits before expansion; no variants of one clip across splits; grouped by person/day/session |
| ses2 (all 11 windows, held out) | no window with MEAN `p_robo ≥ 0.50` (stretch `< 0.30`) |
| ses2 `chunk_002` | `< 0.10` |
| ses2 4 s probe | both models and MEAN `< 0.10` |
| Session A overlap | `< 0.20` with and without mask |
| Benign-heldout | any-alarm sessions per 1000 reduced ≥ 5× vs 48+49 baseline (historical ~20–30 FP / 700–1000 sessions) |
| Category 3/4 held-out | FP rate reduced |
| Chunk vs overlap | FP and recall reported **separately**; never mixed into one F1 |
| `09e7c4…` overlap `001_002` | MEAN `≥ 0.50` |
| Real/in-store positives | event recall ≥ agreed floor (baseline − tolerance) |
| Staged-heldout, partial-core | recall not collapsed |
| Invariance | benign windows: `|Δp| < 0.10` across mask on/off, padding length, time-warp ×0.75–1.33, ±8-frame offset |
| Parity | PyTorch vs Triton/TensorRT `|Δp| < 1e-3` on ≥ 20 real tensors |
| Operational | alarms per 1000 sessions, max score per session, windows firing per visit |

A checkpoint that fixes ses2 by silencing all overlaps is a **failed** run. ses2 passing and split-overlap recall are joint conditions.

---

## 7. Answers to §9 of the trial plan

| # | Question | Decision |
|---|---|---|
| 1 | Four cells or one? | One: full/keep_grid. `filtered` and baseline-aug cells only as Stage 3 ablations. |
| 2 | Port `fp_hardened` before the sweep? | Not literally. First a well-defined weighted BCE and honest naming; port adapted focal/mining as an ablation. The pilot may run before this, but the sweep may not. |
| 3 | Stride? | 48 primary. Stride 96 only as a low-priority ablation. Stride 48 is a sliding window, **not** a substitute for real production overlaps (add them from sessions). |
| 4 | Architectures? | 48, 49, 53, 57 first; drop 12/37/41/50; add at most one attention-pooling variant; no GAT/ST-GCN in the first sweep. |
| 5 | Ensembles? | Fixed pairs, then ≤ 5 candidates among the top-4 models by replay; threshold from replay. |
| 6 | Is clip-only window training worth a full sweep? | Worth the **pilot** only. No sweep until session negatives + production overlaps are in (Stage 2). |
| 7 | Epochs? | Pilot 25, sweep cap 50, early stopping on replay metrics; extend only winners. |
| 8 | Preprocess v2 / mask before the trial? | Yes for the parts in B5 (validity gating, visible-joint normalization, parity contract). `is_pad`/`frame_valid` channels may wait for Stage 3. |
| 9 | Augment policy A1–A7? | Agree with A1–A6 (augment all train windows, including overlaps and 6/14). A7 modified: short/padded windows follow B2 (balanced padding, weight 0.5, motion before padding). Start with identity + 1 deterministic variant rather than 5. |
| Hole | `win_bin_full_hardened` naming | Rename until the effective loss/mining matches. |

---

## 8. Suggested config deltas (adapt to the existing `window_config.json` schema)

```json
{
  "window_len": 96,
  "stride": 48,
  "min_tail_for_final_window": 24,
  "mask_policy": "keep_grid_v2",
  "preprocess_version": 2,
  "positive_labeling": {
    "mode": "core_aware",
    "core_index_path": "core_index.json",
    "full_ov_frames_cap": 36, "full_ov_frac": 0.6,
    "partial_min_frac": 0.2, "partial_weight": 0.3,
    "stand_only": "ignore"
  },
  "padding": {
    "mode": "repeat_last",
    "record_pad_frac": true,
    "padded_negative_prob": 0.15,
    "padded_len_range": [48, 95],
    "pad_weight_threshold": 0.25, "pad_weight": 0.5
  },
  "sampler": {
    "type": "hierarchical",
    "levels": ["source", "clip_or_session", "window", "variant"],
    "positive_share": 0.35,
    "max_windows_per_clip_per_epoch": 4
  },
  "synthetic": { "extra_manifest_views_per_clip": 1, "on_the_fly": "moderate" },
  "loss": {
    "type": "weighted_bce",
    "weights": { "neg": 1.0, "cat34_neg": 1.5, "hard_neg": [2, 3], "pos_core": 2.0, "pos_split_overlap": 3.0 },
    "label_smoothing": 0.02
  },
  "train_defaults": { "epochs": 25, "early_stop_patience": 8 },
  "selection": { "primary": "replay", "recall_constraints": true },
  "cell_id": "win_bin_full_baseline"
}
```

---

## 9. Risks and stop conditions

| Risk | Mitigation / stop condition |
|---|---|
| Core detection errors (bad `core_index`) | confidence score, fallback trim, one spot-check pass; stop and fix if > 10% of clips use the fallback |
| Hard negatives suppress the arm trajectories shared with real robberies | capped weights, keep real/synthetic split positives at high weight, monitor `09e7c4…` and staged-heldout every epoch |
| Core-only positives ignore short/partial gestures | partial-core windows with soft weight, partial-core recall reported |
| Staged style ≠ real style (context fill/transplant) | in-store recorded positives evaluated separately |
| Seam becomes a shortcut | symmetric seam augmentation on both classes |
| Wrong "benign" labels poison hard negatives | session-level human confirmation for mined windows |
| ses2 memorization / evaluation leakage | ses2 held out until Stage 5 |
| Pose-only ceiling for 3/4 vs 6/14 | Stage 4 features; accept that stage 2 (SAM3) resolves residual ambiguity |
| Offline/production divergence | parity gate, `preprocess_version` in `p1.json`, replay under both paths |
| GPU wasted on a mislabeled baseline | do not launch the sweep before G0, G1, G2 |

**Stop and report** if: (a) padding-invariance fails after B2; (b) G1 shows recall loss on split overlaps; (c) G2 not met after Stage 2 — then re-run the Phase-0 sensitivity sweeps before changing architecture.

---

## 10. Corrections to `MOTION_FP_RETRAIN_PLAN.md`

1. The core-detection criterion in the earlier plan included "wrist–hip distance below its 30th percentile". That would also flag standing with hands at the sides. **Use wrist speed only (B1).**
2. The earlier plan proposed a new 34-channel input (`frame_valid`, `is_pad`) from the start. In this campaign they become a **Stage 3 ablation**; the pilot keeps the current channel count and fixes the class confound through data (B2).
3. The earlier plan's data-source table assumed session data from the start; here sessions enter at **Stage 2** (the pilot validates the clip-window pipeline first).

---

## 11. Checklist for the agent (in this order)

- [ ] B6 windowing tests + `min_tail_for_final_window`
- [ ] B1 `core_index.json` (+ plots) and core-aware window labels with `ignore`
- [ ] B2 `pad_frac` logging, padded negatives, pad weight, pooling audit of `tcn_gru`
- [ ] B3 hierarchical sampler, per-clip caps, variants identity + 1, positive share
- [ ] B4 `run_config.json` (+ in checkpoint), rename cell, weighted BCE, label smoothing option
- [ ] B5 keep_grid_v2 (visible-joint normalization, validity-gated velocity), parity contract, golden tensors, SHA check of 48/49
- [ ] B7 split by recording unit; locked sets defined
- [ ] B9 replay harness (chunk/overlap separate, both mask paths) reproducing Appendix A
- [ ] **Gate G0** → report to the owner
- [ ] Stage 1 pilot (R2 on 48/49; optional R1, R2f) → **Gate G1**
- [ ] B8 session index + mined hard negatives + synthetic split positives → Stage 2 (48, 49, 53, 57) → **Gate G2**
- [ ] Record 15–20 in-store staged robberies (owner task, in parallel from Stage 0)
- [ ] Stage 3 reduced sweep and ablations; ensembles from replay
- [ ] Stage 4 only if gates are missed
- [ ] Stage 5 release retrain including ses2, final parity gate

**Do not:** launch the full sweep before G2; select by clip/window F1 alone; tune thresholds on ses2; label stand-only windows as positive or as negative; ship a keep_grid-trained model into the mask-dropping production path without replay; change preprocessing, labels, loss and architecture in the same run.
