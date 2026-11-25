# Motion Stage-1 False Positives — Consolidated Retraining Plan

**Audience:** the implementation agent (has repo access: `GuadIA-IAModel`, `Kanvis-Orchestrator`).
**Source material:** `MOTION_FP_INVESTIGATION_BRIEF.md` (read it first; Section 5 and Appendix A are the ground-truth failure cases) plus a cross-review of three independent external proposals. This document is the merged decision, not a menu.
**Goal:** retrain the motion model (`tcn_gru`, binary 6|14 vs rest) so that ses2-like benign windows score low **by themselves**, while split-chunk robberies (e.g. `09e7c4…` overlap `001_002`) still fire. HAR `hand_to_body` is a diagnostic / optional training-time teacher only, never the fix.

---

## 0. TL;DR — what to do, in order

1. **Phase 0 (no training, 1–2 days):** verify checkpoint identity and PyTorch↔Triton parity, audit the training set for duration/padding shortcuts, and run *sensitivity sweeps* (mask on/off, padding length, time-warp) with the existing 48/49 checkpoints on benign windows. This tells us how much of the FP is **preprocessing artifact** vs **learned content**.
2. **Phase 1:** define one canonical **preprocessing v2** (fixed 12 Hz grid, never drop frames, explicit `valid`/`is_pad` channels, no last-frame-repeat padding). Single shared module for training and the orchestrator.
3. **Phase 2:** build a **production-shaped dataset** (chunks, overlaps, random 96-frame crops, short padded windows) from real benign sessions + cat 3/4 slices; **core-localized positives** from staged 6/14 with *context transplant*; **synthetic split-overlap positives**.
4. **Phase 3:** train the *unchanged* TCN+GRU architecture (new input channels only) with weighted BCE + capped hard-negative weights + MIL supplement + logit-saturation control. One change at a time in ablations.
5. **Phase 4:** select checkpoints **only** by session replay on **held-out** sessions, under a hard recall constraint. Do not use ses2 as both training data and acceptance test.
6. **Phase 5 (conditional):** geometry channels / auxiliary head / HAR-teacher soft targets only if pose-only v2 fails the gates.
7. **Parallel, not primary:** interim product mitigations (fusion rule, seam/track-switch skip) while the retrain is in progress.

---

## 1. Success criteria (what "fixed" means)

| Dimension | Target |
|---|---|
| ses2, all 11 windows | **0/11** windows with MEAN `p_robo ≥ 0.50`; stretch: max < 0.30 |
| ses2 `poses_probe_chunk002_seg0_4s.npy` | both models and MEAN `< 0.10` |
| Session A overlap (track switch seam) | `< 0.20` with **and** without `valid_mask` |
| Held-out benign sessions (never trained on) | large reduction of sessions-with-any-alarm vs 48+49 baseline (target ≥ 5×; the historical baseline is ~20–30 FP / 700–1000 sessions) |
| Real split robbery `09e7c4…` overlap `001_002` | MEAN `≥ 0.50` (recall preserved) |
| Held-out staged 6/14 (core inside chunk, core across seam, partial core) | recall within **3 points** of the 48+49 baseline at the same operating threshold |
| Invariance | on benign windows, `|Δp| < 0.10` between mask-on/mask-off and between padded/unpadded variants (this is a unit test, see §5.4) |
| Parity | PyTorch vs Triton/TensorRT: `|Δp| < 1e-3` on ≥ 20 real production tensors |

Clip-level val F1 is reported but is never a gate.

---

## 2. Probable causes (ranked) and how to confirm each one cheaply

These are hypotheses derived from the brief. **Phase 0 exists to confirm or kill them before spending GPU time.**

### H1 — Padding / duration shortcut (strong suspect)
- `temporal_resize(96)` **pads short sequences by repeating the last frame** and **uniformly subsamples long ones**. Training clips are 6–14 s: a 6 s clip at 12 Hz (72 frames) gets ~24 static padded frames; a 14 s clip (168 frames) gets time-compressed ×1.75 (gestures look 1.75× faster). Production chunks are exactly 8 s, so no compression; and masked chunks are *shorter* than 96 and padded.
- If clip duration correlates with the label (staged 6/14 were recorded with their own typical duration), the network can learn **"static tail / padding fraction ⇒ robbery"** and **"speed scale ⇒ class"** instead of the gesture.
- Evidence in the brief that fits: the 4 s probe (48 frames + 48 static padded) scores 0.948 while the first 16 frames alone score ~0; ses2 `chunk_001` with mask (T=73, i.e. padded) fires; `chunk_002` fires with mask (padded) but scores ~0 on full 96 rows without mask. Counter-evidence: overlap `001_002` fires even without the mask → H1 is **part** of the story, not all of it.
- **Confirm:** (a) per-class histogram of clip length, pad fraction after resize, compression factor, mean speed; (b) fit a trivial logistic regression on `[T, pad_frac, mean_speed, static_fraction]` → label. High AUC = shortcut. (c) Sweep on benign windows: truncate to `T ∈ {24,32,48,64,80,96}` and pad as production does; plot `p_robo` vs `T`.

### H2 — `valid_mask` frame dropping creates time-warp and velocity spikes
- Dropping invalid frames *before* `add_velocity` makes non-adjacent frames look adjacent ("teleport"), producing extreme velocities (documented in Session A; ses2 `chunk_002` flips 0% → 99% with the mask). Overlap windows also use `used_valid_mask_filter:false` in prod, so train/inference policies may differ per window type.
- **Confirm:** for each benign window run: mask off / mask on / mask on + gap-aware velocity / frames replaced by linear interpolation. Report `p_robo` for each.
- **Verify in code:** was `valid_mask` applied **at training time** at all? If training used full rows and production drops frames (or vice-versa), this alone is a bug.

### H3 — Train/production distribution mismatch (primary structural cause)
- One `.npy` per action, action-centered, whole-clip label vs. arbitrary 8 s windows and synthetic overlaps in production (~2n−1 windows per session). Staged 6/14 are `stand → core → stand`; the stand prefix/suffix is labeled positive, so "standing-ish low motion" leaks into the positive class.

### H4 — Pose-only kinematic collision (a ceiling, not a bug)
- Cats 3 (basket/cart), 4 (pick+carry), 14 (own bag), 6 (hide) are close in 8-keypoint space; the label difference is object ownership. Some residual FP/recall trade-off may be irreducible with pose alone. Stage 1 should be high-recall and *not saturated* on shopping; stage 2 (SAM3) resolves the rest.

### H5 — Saturated logits and fragile ensemble
- Model 48 on the probe has a logit gap ≈ +22.7 (p = 1.0). Overconfidence means a single OOD input yields 100% and MEAN of two models is easily dominated by one. Ensemble disagreement is large on several windows (`chunk_001`: 49 high / 48 low; `chunk_003_004`: 48 high / 49 ≈ 0).

### H6 — Deployment parity
- ses2 `chunk_006`: prod ≈ 0%, offline ≈ 100%; `chunk_004`, `chunk_003_004` also disagree. Until parity is resolved, offline numbers cannot be trusted as the training target. Also re-verify SHA/campaign id of `modelo_48` / `modelo_49` (Appendix B of the brief).

---

## 3. Decisions log (why this plan looks the way it does)

**Adopted from the external proposals (consensus):**
- Production-shaped training items (chunks + overlaps + random crops), not whole action files.
- Real benign sessions mined as negatives; hard-negative up-weighting.
- Core-localized supervision for staged 6/14 (MIL / soft labels), prefix/suffix ignored or low weight.
- Replay-based checkpoint selection with a hard recall constraint; parity gate before trusting offline numbers.
- Keep `tcn_gru` first; data + objective alignment should deliver most of the gain.

**Adopted with modification:**
- *Mask handling:* one proposal said "train both mask policies as augmentation"; another said "fix parity first". **Decision:** choose ONE canonical policy (preprocessing v2, §4 Phase 1). Mask perturbation is augmentation on top of that policy, never a substitute for parity.
- *Seam augmentation on negatives* (synthetic jumps at frame 48): good idea for the Session A failure, but it must be applied **symmetrically to positives**, otherwise "seam ⇒ benign" becomes a shortcut.
- *Synthetic split positives* (core cut at the peak, joined with benign walking): adopted, with seam blending and the same seam statistics as negatives.
- *HAR as teacher:* only as (a) a candidate filter for hard negatives and (b) a small-weight soft target on windows that are **human-confirmed benign**. Not on unreviewed sessions (circular: HAR errors would be baked in).
- *Auxiliary multiclass head (0/3/4/6/14), geometry channels:* moved to Phase 5 (conditional), to keep ablations interpretable.

**Not adopted (and why):**
- *Per-window-type thresholds (e.g. `T_chunk=0.50`, `T_overlap=0.65`) chosen a priori.* There is currently **one** real split-robbery session; a margin or threshold tuned on n=1 is meaningless. Revisit only when the recall set has ≥ 20 positive overlaps.
- *Bundling 7 changes into one run* (new channels + aux head + focal + distillation + MIL + …). Cannot attribute gains or regressions.
- *Using ses2 both as training hard-negative and as the acceptance test.* Passing a window you trained on proves memorization, not generalization. See §5.1.
- *Focal loss as first move.* Only as an ablation after weighted BCE baseline; strong focal + large hard-neg weights can destabilize and silence recall.
- *Raising the threshold as a fix.* Out of scope per brief; logits are saturated so it would not help anyway.

**Additions not present in the external proposals:** the padding/duration shortcut audit (H1), the train/test leakage rule for ses2, building a real recall set by recording staged robberies through the production pipeline, logit-saturation control, and Phase 0 sensitivity sweeps.

---

## 4. Implementation plan

### Phase 0 — Diagnostics with existing checkpoints (no training)

Create `experiments/training/campaign/phase0_diagnostics.py` (reuse `score_pose_ensemble.py` internals).

1. **Identity & parity**
   - Print SHA256 + campaign id for `modelo_48.pt`, `modelo_49.pt`.
   - Export ≥ 20 real production tensors (post-preprocess, the exact input to the engine) from MinIO; run PyTorch `.pt` and Triton/TensorRT; report `max |Δp|` and per-window table. Resolve `chunk_006`, `chunk_004`, `chunk_003_004` before anything else.
2. **Training-set audit** (`audit_training_set.py`)
   - Per class (esp. 0/3/4/6/14): clip length (frames and seconds), source fps, pad fraction and compression factor after `temporal_resize(96)`, mean/95th-percentile joint speed, static-frame fraction.
   - Whether `valid_mask` was applied in training; fraction of frames dropped per class.
   - Logistic-regression shortcut probe (see H1). Report AUC.
3. **Sensitivity sweeps** (`sweep_preprocess.py`) over benign windows (ses2 + every other benign session available), with 48, 49 and MEAN:
   - mask on / off / interpolate gaps ≤ 3 frames;
   - truncate-and-pad to `T ∈ {24,32,48,64,80,96}` (production padding);
   - time-warp ×{0.75, 0.9, 1.0, 1.1, 1.33};
   - window start offset sweep (stride 8 frames) over each chunk.
   - Output CSV + plots. **Decision rule:** if mask-off/full-96 already brings most ses2 windows < 0.5 → H1/H2 dominate and preprocessing v2 is the main lever; if windows stay hot across all variants → content is genuinely mis-learned and Phase 2/3 data work is the main lever.
4. Re-evaluate the campaign pair **49+53** vs field **48+49** on the same replay (cheap, informative).

**Gate:** Phase 0 report committed to the repo before Phase 1 starts.

### Phase 1 — Canonical preprocessing v2 (single source of truth)

Put it in one importable module (e.g. `kanvis_motion/preprocess_v2.py`) used by **training, offline scripts and `Kanvis-Orchestrator`**. Record `preprocess_version: 2` in checkpoint metadata and in `p1.json`.

```python
def preprocess_v2(poses, valid, hz=12, T_out=96, max_gap=3):
    """
    poses: [T, 8, 2] on the fixed 12 Hz grid (NaN = missing joint)
    valid: [T] bool (>=5 visible joints)
    returns: float32 [T_out, C], C = 16 pose + 16 vel + 1 frame_valid + 1 is_pad = 34
    """
    # 1. NEVER drop frames. Keep the fixed grid.
    # 2. Fill short invalid gaps (<= max_gap frames, 0.25 s) by linear interpolation;
    #    longer gaps stay invalid (pose=0, frame_valid=0).
    # 3. Per-frame normalization as in normalize_sequence, using only visible joints;
    #    invalid frames -> zeros.
    # 4. Velocity v[t] = x[t] - x[t-1] ONLY if frames t and t-1 are both valid, else 0.
    #    Clip |v| to a fixed cap (e.g. train-set p99). No cross-gap velocities.
    # 5. If T < T_out: right-pad with ZEROS, is_pad = 1.  (No last-frame repetition.)
    #    If T > T_out: take a slice (training: random crop; prod: windows are already 96).
    #    NEVER uniformly subsample long clips (it time-warps the gesture).
    # 6. Return [T_out, 34].
```

Model changes (minimal):
- Input dim 32 → 34 (`frame_valid`, `is_pad`).
- Make pooling **pad-aware**: take the GRU state at the last non-pad frame, or masked attention/mean pooling. Padded frames must not influence the output.
- Keep TCN+GRU, seq_len 96, 2-class (or 1-logit) head.

Resample **all training clips once, offline, to the canonical 12 Hz grid** (check source fps first — Phase 0 audit). Delete any dependence on `temporal_resize` for long clips.

Deployment: the TensorRT/Triton engine input shape becomes `[96, 34]`; plan the engine rebuild and the orchestrator preprocessing change together (feature-flag it; keep v1 path until v2 passes the gates).

### Phase 2 — Production-shaped dataset

Create `build_windows_dataset.py` producing the schema in §6.

**Window types**
- `chunk`: exact C2 grid, 96 frames.
- `overlap`: `concat(a[48:96], b[0:48])`, with the same preprocessing as chunks (v2 has no per-type mask policy).
- `crop`: random 96-frame crops at random offsets from continuous sessions and from long action clips (never centered on the action).
- `short`: random truncations `T ∈ [32,96)` with `is_pad` (p ≈ 0.15) so the model learns that padding carries no information.

**Negatives**
- *Real benign sessions.* Production robberies are rare, so sessions with no confirmed event are ≥ 99% benign: use them at scale (hundreds, not 5–10). Exclude any session that has a confirmed/suspected robbery or an unreviewed stage-2/3 alarm. For **hard negatives** (windows that fire), a quick *session-level* yes/no human confirmation is required — not micro-gesture labels.
- *Cat 3/4 clips* sliced into windows (walk-only, reach-to-shelf, carry segments); oversample the segments that look like the production FPs (ses2 `chunk_002` head).
- *All other categories* at normal weight.
- *Benign overlaps:* every consecutive chunk pair of a verified benign session.

**Positives (staged 6/14) — core localization + context transplant**
1. Detect the core automatically: smoothed wrist speed envelope and wrist–hip distance; core = largest connected segment where speed > 0.25·max or wrist–hip distance is below its 30th percentile, dilated ±6 frames. Fallback: trim ~12 frames (≈1 s) from each end. Plot and eyeball ~20 clips once; no further labeling.
2. Build windows by placing the core at random offsets inside a 96-frame window; fill the rest with **benign context frames** from real sessions (blend 3 frames at each seam). Also include windows with the original stand prefix/suffix, but at low weight.
3. Labeling rule per window (`core_frames = |core ∩ window|`):
   - `core_frames ≥ 24` (≥ 2 s) or ≥ 60% of the core → `y=1`, weight 1.0
   - `8 ≤ core_frames < 24` → `y=1`, weight 0.3 (soft/partial)
   - `< 8` (stand-only) → **ignored** (weight 0). Do not label as negative.
4. **Synthetic split positives:** place the core so it straddles the frame-48 seam of an overlap window (varying fractions 20–80% on each side) with benign context on both sides.
5. **Symmetric seam augmentation:** apply the same seam jitter/jumps (including track-switch-like jumps and visibility drops) to a matched fraction of **negative and positive** overlaps.
6. **Real positives:** `09e7c4…` `001_002` (high weight) **plus new in-store recordings** (see §5.2).

**Starting composition per epoch (to be ablated, not gospel):**

| Source | Share |
|---|---|
| real benign chunk/crop negatives | 30% |
| real benign overlap negatives | 10% |
| cat 3/4 slices | 15% |
| other categories | 10% |
| mined hard negatives (session-capped) | 10% |
| staged cores in transplanted context | 15% |
| synthetic split-overlap positives | 5% |
| real split / in-store positives | 5% |

Positives ≈ 25%. Cap per-session contribution per epoch so no single session (ses2 included) dominates.

**Splits:** by **session / person / recording day**, never by window or file. Hard-negative mining only touches the train split.

### Phase 3 — Training objective

```
L = Σ w_i · BCE(y_i, p_i) / Σ w_i  +  λ_mil · L_mil  +  β · mean(logit²)     (+ label smoothing ε)
```
- **Per-example weights `w_i`** (starting ranges, not established optima): ordinary negative 1.0; cat 3/4 negative 1.5; mined hard negative `1 + α·min(p_prev^γ, c)` with α=2, γ=1, c=1 (so 1–3); positive core 2–3; real/synthetic split-overlap positive 3; partial-core 0.3; ignored 0.
- **MIL supplement (staged whole clips):** sub-window logits `s_j` over sliding sub-windows; `P_clip = (1/τ)·logsumexp(τ·s_j)` with τ≈3–5; `λ_mil ≈ 0.3`. Core-window BCE is the **primary** positive signal; MIL is a constraint, not the main loss.
- **Saturation control (H5):** label smoothing ε ≈ 0.02–0.05 and/or logit penalty `β ≈ 1e-3`. Goal: no more +22 logit gaps on OOD benign windows.
- **Augmentation:** L/R mirror (swap joints), small coordinate noise, speed ×[0.85,1.15] via resampling, random keypoint dropout and `frame_valid` dropout 0–20% (gap-aware, grid preserved), random truncation with `is_pad`, seam injection (both classes).
- **Sampler:** class- and source-balanced batches per the composition table; oversample `is_hard_neg` and `real_split_pos`.
- **Hard-negative mining loop:** after each campaign, replay all train-split benign sessions; collect windows with `p > 0.2` (not just > 0.5); de-duplicate neighbors (stride ≥ 12 frames); diversify by session/pattern; add with capped weights; retrain. Keep the original FP set as a permanent regression suite.
- **Ensemble:** train ≥ 4 diverse models (seeds / width); choose the pair **and the fusion rule (MEAN vs min/AND vs geometric mean)** on replay.
- **Ablation order (one change per run):** (1) v2 preprocessing alone (same data) → (2) + production-shaped windows → (3) + core positives/context transplant → (4) + hard-negative weighting → (5) + MIL → (6) + saturation control → (7) focal vs weighted BCE → (8) fusion rule/pair.

### Phase 4 — Evaluation harness

Create `replay_eval.py` (supersedes val-F1 selection) — details in §5.

### Phase 5 — Conditional upgrades (only if Phase 3 misses the gates)

Order of escalation:
1. **Geometry channels** computed in `preprocess_v2` (so prod parity is automatic): wrist–hip distance normalized by torso length (ipsilateral/contralateral, min, temporal derivative), wrist height relative to shoulders, wrist speed, approach/retreat sign, bilateral activity. +7–10 channels.
2. **Auxiliary head** over action family (0/3/4/6/14), loss weight ≈ 0.3; inference still uses the binary head (or `p6 + p14`).
3. **HAR teacher (training-time only):** soft target ≤ 0.1, weight ≤ 0.2, only on windows where HAR says `hand_to_body == False` (`LEJOS_TODO_EL_GESTO` / `INCOMPATIBLE`), mask coverage ≥ 75%, **and** the session is human-confirmed benign.
4. Multi-scale heads (2/4/8 s over a shared encoder) or a light streaming state across chunks — last resort.

Stop condition for pose-only: if v2 + data work fixes held-out benign sessions without recall loss, **do not add features**.

### Phase 6 — Interim product mitigations (parallel, not the answer)

- Evaluate fusion `min`/AND for 48+49 on replay (ses2 `chunk_001`: 49 high, 48 low).
- Skip or discount overlap windows when the two halves have different track ids (Session A).
- HAR veto layer stays as an independent product layer.
These can ship before the retrain but must **not** be used to declare the training problem solved.

---

## 5. Evaluation protocol

### 5.1 Sets (all split by session)

| Set | Purpose | Used for training? |
|---|---|---|
| **FP-train** | benign sessions that currently fire (from `p1.json` history), human-confirmed benign | yes (hard negatives) |
| **FP-heldout (locked)** | other confirmed-FP sessions; **ses2 lives here** for the generalization test | **never** |
| **Benign-heldout (large)** | ≥ 200 random benign sessions for sessions-with-any-alarm per 1000 | never |
| **Staged-heldout** | staged 6/14 clips held out by person/day: core-in-chunk, core-across-seam, partial-core sweep | never |
| **Real positives** | `09e7c4…` + newly recorded in-store positives | held-out subset kept locked |

**ses2 rule:** the generalization verdict is taken with ses2 held out. Only after the gates pass, a *final release* model may be retrained including ses2 (and re-checked on the remaining locked sets). If too few FP sessions exist, use k-fold by session.

### 5.2 Build a real recall set (cheap, high value)
There is currently **one** real split-robbery session. Record **15–20 staged robberies through the actual production pipeline** (real camera, real chunking, real poses): ~half with the core inside one chunk, ~half straddling a chunk boundary / the overlap. This needs session-level labels only and gives production-shaped positives and a credible recall estimate.

### 5.3 Metrics per checkpoint / ensemble
- Session-level: any-alarm rate; alarms per session; max score per session.
- Window-level FP rate at 0.5 and 0.3.
- Recall by positive type (core-in-chunk, across-seam, partial-core fraction).
- Per-model and ensemble scores, ensemble disagreement, window type.
- Parity table vs Triton.
- Bootstrap confidence intervals by session (windows within a session are correlated).

### 5.4 Invariance tests (regression suite from Phase 0)
For benign windows the score must be flat under: mask on/off, padding length, time-warp ×0.75–1.33, ±8-frame offset. Fail if `|Δp| > 0.10`. This directly attacks H1/H2 and prevents re-introducing them.

### 5.5 Model selection
Minimize benign-session false-alarm rate **subject to** recall constraints (event recall on real positives, split-overlap recall, partial-core recall) set from the 48+49 baseline and business needs — **not** from the new run. Pick the single global threshold on replay; introduce per-window-type thresholds only if the recall set supports it (≥ 20 positive overlaps).

---

## 6. Dataset schema

```json
{
  "window_id": "str",
  "session_id": "str | null",
  "person_id": "str | null",
  "split": "train | val | locked_test",
  "type": "chunk | overlap | crop | short | synthetic_split | action_slice",
  "poses": "float32 [96, 8, 2] (raw, pre-preprocess; preprocess_v2 applied in the loader)",
  "frame_valid": "bool [96]",
  "is_pad": "bool [96]",
  "label": "0 | 1",
  "weight": "float",
  "core_frames": "int | null",
  "source": "real_benign | cat3_4_slice | other_cat | prod_fp | staged_core | synth_split_pos | real_split_pos | instore_pos",
  "is_hard_neg": "bool",
  "seam_injected": "bool",
  "prev_p_robo": "float | null",
  "har_teacher_p": "float | null (Phase 5 only, human-confirmed benign only)"
}
```

---

## 7. Risks to recall and mitigations

| Risk | Mitigation |
|---|---|
| Hard negatives suppress the arm trajectories shared with real robberies (overcorrection) | capped weights; keep real split positives and synthetic split positives at high weight; monitor `09e7c4…` overlap and the Staged-heldout recall **every epoch** |
| Core-only positives make the model ignore short/partial gestures | partial-core windows with soft weight; fraction sweep in evaluation |
| Staged style ≠ real robbery style (context transplant overfits to "staged look") | in-store recorded positives (§5.2); evaluate on them separately |
| Seam handling becomes a shortcut | symmetric seam augmentation on both classes |
| Mislabeled "benign" windows poison hard negatives | session-level human confirmation for any mined hard negative; exclude sessions with unreviewed alarms |
| ses2 memorization | ses2 held out until the gates pass |
| Pose-only ceiling (H4) | Phase 5 escalation; accept that stage 2 resolves residual ambiguity |
| Threshold drift after retrain | re-select threshold on replay; one global threshold until the recall set is large enough |
| Offline numbers diverge from production | parity gate first; `preprocess_version` logged in `p1.json` |

---

## 8. Things not to do

- Do not "fix" this by raising the threshold or by relying on the HAR veto.
- Do not train and accept on the same session.
- Do not change architecture, preprocessing, mask policy and loss in the same run.
- Do not hand-label micro-gestures; labels stay at session level (benign/robbery) and clip level (existing categories).
- Do not use last-frame-repeat padding or uniform subsampling of long clips anywhere in v2.

---

## 9. First questions to answer from the code (before Phase 1)

1. Is `valid_mask` filtering applied at training time? Same policy for chunks and overlaps?
2. Source fps of training clips vs the 12 Hz production grid; length distribution per class.
3. Exact `temporal_resize` behavior for `T > 96` in the training loader.
4. How the GRU output is pooled (last step? mean?) — matters for pad-aware pooling.
5. How many verified-benign sessions and confirmed-FP sessions (with `p1.json` + `poses_full`) are available; are `09e7c4…` tensors available?
6. Triton/TensorRT engine constraints for the new `[96, 34]` input and for rebuilding both `.pt` and engine from the same preprocessing module.
7. Counts per category in `bin_full`, and whether staged 6/14 were recorded by one or several people/days (split strategy).

**Honest caveat:** the causes in §2 are inferred from the brief, not yet measured. Phase 0 outcomes may reorder priorities; if sweeps show the model is hot regardless of preprocessing, weight the effort toward Phases 2–3 and the in-store recall set.
