# Motion Stage-1 False Positives — Investigation Brief

**Purpose:** Self-contained brief for **external agents** (no prior chat context).

**Primary goal:** **Train / adapt the motion model** so that **obvious non-robbery windows** (normal shopping, ses2-like sessions) score **low** — not only rely on downstream filters. We want concrete **data, loss, evaluation, and architecture** recommendations.

**Secondary context:** HAR v2 `hand_to_body` (and orchestration vetoes) are mentioned as **evidence** that motion is wrong on these windows; product may still use HAR before/after motion, but **that is out of scope as the main fix** for this review.

**Audience:** ML, pipeline, and product engineers.  
**Status:** Working notes (Nov 2025).  
**Repository:** `GuadIA-IAModel` (IAModel motion training + campaign docs). Kanvis orchestration/HAR lives in separate `Kanvis-Orchestrator` repo (paths cited below).

**This document is the primary handoff.** Other repo files (`KANVIS_CONTEXT_AND_RESEARCH_BRIEF.md`, etc.) add depth but are **not required** to answer the brief.

### 0. Agent handoff — what we need from you

**Problem statement:** In production, a **normal shopping session** (category **3** metadata, single person in frame) gets **`robbery_alarm: true`** on many **8 s chunks** and **overlap** windows. Motion scores are often **90–100%** on windows where humans (and HAR v2 geometry) say there is **no theft gesture**. The **TCN+GRU model should not classify these tensors as robbery** after proper training — today it does.

**What we already measured offline:** Replay of exported sessions with **modelo_48 + modelo_49** (MEAN @ 0.50), production-shaped chunks/overlaps, and comparison to HAR v2 on the same `poses_full` (see Sections 5–6, Appendix A). Use this as **failure cases the next training run must satisfy**.

**Deliverables requested from reviewing agents (training-first):**

1. **Concrete next training campaign plan:** datasets, sampling, labels, loss (e.g. MIL), augmentation, and whether positives/negatives should be **production windows** vs action clips.
2. **Hard-negative strategy** using **ses2** (and similar exports) without hand-labeling every micro-gesture — e.g. mine whole sessions as negatives, overlap negatives, weighting, contrastive pairs.
3. **Positive strategy** for staged **6/14** given **stand → core → stand** structure vs **8 s arbitrary** chunks (core-only supervision, down-weight prefix/suffix).
4. **Evaluation protocol** you would trust: session replay, chunk+overlap grid, parity preprocess/mask; **success criterion** = ses2-like sessions near **0% motion score** on most windows while **split-chunk robbery** sessions (e.g. `09e7c4…`) still fire on overlap where needed.
5. **Model / objective changes:** architecture, ensemble, threshold selection **on replay**, calibration, auxiliary heads (3/4 vs 6/14), or explicit conclusion that **pose-only 6|14 needs extra input channels** — with a training path if extra inputs are required.
6. Optional: training loop pseudo-code or dataset schema.

**Mention but do not center:** HAR `hand_to_body` veto / SAM3 gating — acceptable as **parallel product layer**, not as the answer to “how do we train motion?”

**Constraints (owner):**

- Do **not** propose labeling thousands of micro-actions (hair, phone, scratch, etc.); negatives must generalize.
- Training data today: **one `.npy` per action**, variable **6–14 s**; shopping actions may include **walk + gesture + walk** in one file; staged **6/14** are **stand → gesture → stand** without unrelated actions in the middle.
- Business cares about **few false alarms per day per store**; optimize for **session/window FP on real geometry**, not clip-level val accuracy alone.

**Non-goals:**

- Detailed orchestration redesign; frontend; SAM3 architecture review.
- “Raise threshold” without a retrain + replay justification.

**Suggested message when pasting this file to another agent:**

> Read `MOTION_FP_INVESTIGATION_BRIEF.md`. **Focus on training the motion model** so windows like ses2 (Section 5, Appendix A) score **non-robbery**, while keeping overlap recall for true split robos. HAR `hand_to_body` is **diagnostic** (motion is overconfident where HAR says no pocket gesture) — not the main solution we want from you. Deliver Section 0 training-first deliverables with ranked steps and risks to recall.

---

**Related artifacts (same repo, optional):**  
`IAModel/experiments/training/campaign/score_pose_ensemble.py`, `diagnose_chunk_overlap.py`, `motion_har_veto_demo.py`, `C2_CHUNK_OVERLAP_HANDOFF.md`, `KANVIS_CONTEXT_AND_RESEARCH_BRIEF.md`.

---

## Document map

| Section | Content |
|---------|---------|
| **0** | Agent handoff, deliverables, paste message |
| **1–2** | Summary, pipeline, preprocessing |
| **3** | Training data (variable-length actions, staged 6/14) |
| **4** | Offline scripts |
| **5** | Experiments (Session A, ses2, HAR contrast, probe) |
| **6–8** | Hypotheses, non-causes, **training recommendations** (Section 8 primary) |
| **9** | Open questions |
| **10–11** | Reproducibility, glossary |
| **A–D** | Appendices: prod alarms, debug lessons, campaign context, **post-retrain acceptance tests** |

---

## 1. Executive summary

We deploy a **binary motion classifier** (ensemble of two `tcn_gru` checkpoints, typically **modelo_48 + modelo_49**, rule **MEAN**, threshold **0.50**) on **fixed 8-second chunks** of upper-body pose (`poses_full.npy`, 8 keypoints, `[T, 8, 2]`), optionally filtered with `valid_mask.npy`, plus **half-and-half overlap windows** between consecutive chunks (`chunk_{k-1}_{k}`).

**Observed issue (what the model must learn to fix):** On ses2, **8 of 11** production-shaped windows get motion **≥ ~62%**, often **~95–100%**, for **normal shopping**. These should be **clear negatives** for a robbery detector after retraining.

**Diagnostic (not the target solution):** On the same tensors, **HAR v2** usually has **`hand_to_body: false`** — i.e. geometric rules already separate “pocket approach” from these poses; **motion does not**.

**Core training tension:** Training uses **one `.npy` per labeled action** (variable length, action-centered). Production scores **arbitrary 8 s slices** and **synthetic overlaps**. Staged **6/14** clips are **stand → gesture → stand**; real **3/4** clips may mix **walk + shop + walk** in one file. Categories **3, 4, 6, 14** overlap in pose space; binary **6|14 vs rest** on pose alone is hard.

**What we want from this review:** A **retraining roadmap** so motion **scores like HAR’s judgment on obvious cases** (low robbery probability on ses2 windows), without requiring the operator to depend on a veto to hide broken motion scores.

---

## 2. Production pipeline (reference)

| Stage | Input | Output |
|-------|--------|--------|
| Pose | Video @ ~12 Hz dense grid per chunk | `poses_full.npy`, `valid_mask.npy`, timestamps |
| Motion (stage 1) | Per chunk + overlap `chunk_{a}_{b}` | `p1` / `p2` scores, `robbery_score` (MEAN or AND), `robbery_alarm` |
| HAR v2 pocket | Same windows (after motion candidate) | `har_pocket`: `hand_to_body`, `orchestrator_branch`, `motion_suppress`, etc. |
| SAM3 / review | Subset gated by motion + HAR | Visual object / container checks |

Overlap construction (C2 v1): `concat(poses_a[48:96], poses_b[0:48])` → 96 frames; production often sets `used_valid_mask_filter: false` on overlaps.

Ensemble in field: **modelo_48 + modelo_49** from `modelos_v2/`; campaign reference for higher val F1: **modelo_49 + modelo_53** (`bin_full`, campaign `campaign_20261103_230735`).

Preprocessing chain (must match training): `poses_full` → (optional `valid_mask` drop) → `normalize_sequence` → `add_velocity` → `temporal_resize(seq_len=96)` → TCN+GRU.

**Pose tensor:**

- Shape `[T, 8, 2]` float32, normalized image coordinates ~0–1.
- Keypoints are upper body (YOLO26-pose indices mapped to 8 joints: shoulders, elbows, wrists, hips). Wrists are local indices **4** (left) and **5** (right); hips **6–7**.
- `valid_mask.npy`: `True` when ≥ **5** of 8 keypoints are visible (frames with fewer are often dropped at train/inference when mask filter is on).
- `NaN` missing joints → `0` after load in IAModel scoring scripts.

**`normalize_sequence` (per frame):** subtract joint mean per frame, divide by mean L2 norm across joints (removes absolute position/scale). **`add_velocity`:** frame-to-frame joint deltas concatenated (doubles feature dim). **`temporal_resize(96)`:** if `T < 96`, **pad by repeating the last frame**; if `T > 96`, uniform subsample indices — critical for short clips and 4 s probes.

**Scores:**

- Each checkpoint outputs 2-class logits; `p_robo = softmax(logit)[1]`.
- Ensemble **MEAN:** `robbery_score = (p_model_a + p_model_b) / 2`; alarm if `≥ threshold` (0.50 in field).
- Legacy JSON fields `p41`/`p48` in offline script map to model_a / model_b regardless of checkpoint id.

**Why overlaps exist:** Real robberies can span two 8 s chunks; overlap `chunk_{k-1}_{k}` = `poses_k-1[48:96]` ∥ `poses_k[0:48]` (96 frames). Side effect: **extra scoring opportunities** per session (~`2n−1` windows for `n` chunks).

---

## 3. Training data — important clarifications (owner-provided)

### 3.1 General actions (negatives and non-robbery positives)

- **One `.npy` file per performed action**, one person, one category label.
- **Duration is variable:** e.g. **6 s, 8 s, 14 s**, not fixed.
- A **shopping-related action** (e.g. category **3**) may **naturally contain multiple phases** in the same clip: e.g. **walk a bit → take product / put in basket → walk a bit**. The **whole clip** receives the action label.

### 3.2 Staged robbery / bag categories (positives **6** and **14**)

- Recorded **by the team** (staged), not typical customer flow in the aisle.
- Typical structure: **person standing ~1 s → robbery/concealment core → standing ~1 s → end**.
- Durations vary slightly, but **there is no unrelated action** in the middle (only brief standing before/after the core gesture).
- This is **not** the same temporal structure as a real **8 s production chunk** that may contain walking, browsing, or chunk boundaries.

### 3.3 Binary training target

- Campaign cell **`bin_full`:** positive = categories **6** and **14**; negative = all other categories (including **3**, **4**, **5**, etc.).
- The model never sees “object type” or “store vs personal bag” in the pose tensor — only normalized joint coordinates and derived velocity.

### 3.4 Category semantics (abbreviated)

| Cat | Role in `bin_full` | Notes for FP analysis |
|-----|--------------------|------------------------|
| **0** | Negative | Pass / no event |
| **3** | Negative | Put product in basket/cart/bag — **common real shopping** |
| **4** | Negative | Pick product and carry |
| **6** | **Positive** | Take product and **hide** (staged team recordings) |
| **14** | **Positive** | Put object in **own bag** (staged) |
| Others | Negative | 1,2,5,7,9,… — various aisle actions |

Campaign audit literature (in-repo) states that **3, 4, 6, 14** look similar in 8-keypoint kinematics; disambiguation needs **object/context**, not arm pose alone.

### 3.5 Known positive case where overlap helped (contrast)

Session **`09e7c4e01e8845a48775aea09e6a132a`** (documented in `C2_CHUNK_OVERLAP_HANDOFF.md`): isolated chunks ~0–1% motion; **`chunk_001_002` overlap ~100%** on a **true** split robbery. Overlap is **necessary for recall** but **increases FP surface area** on benign sessions (ses2).

---

## 4. Offline tooling built in this investigation

| Script | Role |
|--------|------|
| `score_pose_ensemble.py` | Replicate C2 motion scores on local `poses_full.npy` |
| `diagnose_chunk_overlap.py` | Stitch jump, mask teleport, scores for overlap halves |
| `motion_har_veto_demo.py` | Motion + **HAR v2 pocket** (`pocket_infer`, no legacy HAR); demo veto policy |

**HAR v2 vendor path (Kanvis):**  
`Kanvis-Orchestrator/kanvis-orchestrator/app/vendor/har`

---

## 5. Experiments and findings

### 5.1 Session A — two chunks, likely two tracks (local export, two separate chunk downloads)

**Context:** User exported **`poses_full1.npy`** (chunk_001, **T=94**) and **`poses_full2.npy`** (chunk_002, **T=96**) with **`valid_mask1/2.npy`**. Video looked like two people walking; production often has **one track per chunk** but **track id can switch** across chunks (`unique_tracks_seen > 1`).

**Files:** `poses_full1.npy`, `poses_full2.npy`, masks (user directory; not tied to ses2 session id).

| Window | With `valid_mask` | Without mask filter |
|--------|-------------------|---------------------|
| Chunk 1 alone | ~0% alarm | ~0% |
| Chunk 2 alone | ~0% alarm | ~0% |
| Overlap (2nd half + 1st half) | **~100% alarm** | ~50% (borderline) |

**Interpretation:**

- Per-chunk scores were low; alarm appeared on **synthetic overlap** when mask filtering was on.
- **Stitch discontinuity:** large pose jump at seam, visibility drop (8→5 keypoints), **mask removed intermediate frames** → “teleport” in the filtered series → extreme velocities after `add_velocity` + resize.
- User-visible video: looked like two people walking; tensor is **one track per chunk** concatenated.

**Exported overlap files:**  
`poses_overlap_1_2_second_half_first_half.npy`, `valid_mask_overlap_1_2_second_half_first_half.npy`

---

### 5.2 Session B — full session, single shopper (`Descargas/merged/ses2/`)

**Session id:** `64a58a936e9347bcb2cfb3b06ae27a4e`  
**Metadata:** `max_persons_in_frame: 1`, `unique_tracks_seen: 2`, 6 chunks, ~55 s.  
**Production `p1.json`:** multiple `robbery_alarm: true` on chunks and overlaps.

**Offline replay (modelo_48 + modelo_49, MEAN @ 0.50):**

| Window | Motion score (approx.) | Notes |
|--------|-------------------------|--------|
| chunk_001 | 62% | Model 49 high, 48 low; mask T=73 |
| chunk_002 | 99% | With mask; **without mask on full 96 rows ~0%** |
| chunk_001_002 overlap | 97–100% | **No large stitch jump**; overlap alarm even **without** mask |
| chunk_003 | 81% | |
| chunk_003_004 overlap | 50% offline vs ~91% prod | **Ensemble disagreement** on overlap (48 high, 49 ~0) |
| chunk_005, 006, 005_006 | 96–100% offline | Prod chunk_006 motion ~0% — **parity warning** (TensorRT vs `.pt` or inputs) |

**Decomposition of overlap `chunk_001_002`:**

| Segment | Score |
|---------|--------|
| chunk_001 `[48:96]` | ~0.3% |
| chunk_002 `[0:48]` | **~95%** |
| Full overlap | ~97–100% |

So this overlap is **not** primarily a boundary artifact; it is driven by the **first ~4 s of chunk_002** embedded in the overlap’s second half.

**Probe file:** `poses_probe_chunk002_seg0_4s.npy` (48 frames) → **~94.8%** MEAN, model 48 **p≈1.0**, large positive logit gap.

**Kinematics on probe:** min wrist–hip distance in normalized pose space **~0.06** on some frames (model feature space); **HAR v2 on same file:** `hand_to_body: false`, `resultado: INCOMPATIBLE`, `motivo: LEJOS_TODO_EL_GESTO`, `motion_suppress: true`, `orchestrator_branch: DISCARD_SUPPRESS`.

**Sliding windows on chunk_002 (with mask):** high scores on windows starting early (e.g. `[0:48)`, `[32:80)`).

**Short-window padding artifact:** For T=48, `temporal_resize` **pads to 96 by repeating the last frame** → half the network input can be **static**; this changes logits vs shorter un-padded slices (e.g. frames `[0:16)` alone scored ~0%).

---

### 5.3 Motion vs HAR v2 on full ses2 (calibration target for retraining)

**Why this section exists:** If HAR v2 (pocket geometry) says **no hand-to-body** on a window, the **motion model should ideally output low `p_robo` on the same `poses_full`** after better training. Today the opposite happens on ses2 — use the table below as **regression tests for the next checkpoint**.

**Tooling:** `motion_har_veto_demo.py` = `score_pose_ensemble.py` + HAR v2 `pocket_infer` (no `har_legacy`). The script also computes a **demo veto** (for illustration only; **not** the product fix we are asking agents to design).

**Full ses2 replay summary (embedded — was saved as `salida.json` in investigation):**

| Window | Motion score | Raw alarm | HAR branch | hand_to_body | suppress | Veto? | Final alarm |
|--------|-------------|-----------|------------|--------------|----------|-------|-------------|
| chunk_001 | 0.617 | yes | KEEP_INDETERMINADO | false | no | yes | **no** |
| chunk_001_002 | 0.975 | yes | DISCARD_SUPPRESS | false | yes | yes | **no** |
| chunk_002 | 0.990 | yes | KEEP_INDETERMINADO | false | no | yes | **no** |
| chunk_002_003 | 0.000 | no | DISCARD_SUPPRESS | false | yes | — | no |
| chunk_003 | 0.807 | yes | KEEP_INDETERMINADO | false | no | yes | **no** |
| chunk_003_004 | 0.498 | no | KEEP_INDETERMINADO | false | no | no | no |
| chunk_004 | 0.539 | yes | KEEP_INDETERMINADO | false | no | yes | **no** |
| chunk_004_005 | 0.201 | no | DISCARD_SUPPRESS | false | yes | — | no |
| chunk_005 | 0.957 | yes | KEEP_INDETERMINADO | false | no | yes | **no** |
| chunk_005_006 | 0.999 | yes | KEEP_INDETERMINADO | false | no | yes | **no** |
| chunk_006 | 1.000 | yes | KEEP_INDETERMINADO | false | no | yes | **no** |

| Aggregate | Value |
|-----------|-------|
| Windows | 11 |
| Motion raw alarms | 8 |
| HAR `hand_to_body: true` | **0** |
| After demo veto (illustrative only) | **0** |

**Training implication:** For every row with motion alarm **yes** and `hand_to_body` **false**, add or up-weight these windows as **hard negatives** (or as **distillation targets** toward score ≈ 0).

**Production `p1.json` motion alarms (same session):** true on `chunk_001`, `001_002`, `002`, `003`, `003_004`, `005`, `005_006`; **false** on `002_003`, `004`, `004_005`, `006`. Seven production motion alarms on a benign ~55 s visit — **all are model errors we want eliminated by training**, not masked downstream.

**HAR v2 pattern:** **No window** had `hand_to_body: true`. Motivos include `ZONA_GRIS`, `SIN_DATOS_VALIDOS`, `LEJOS_TODO_EL_GESTO`.

**Parity caveats (offline PyTorch `.pt` vs production Triton/TensorRT):**

| Window | Production `robbery_score` (approx.) | Offline replay | Comment |
|--------|-----------------------------------|----------------|---------|
| Most of 001–005 | Matches ~well | Same masks/models 48\|49 | ses2 validates tooling |
| chunk_006 | ~**0%**, alarm false | ~**100%**, alarm true | **Major mismatch** — investigate preprocess/engine |
| chunk_004 | ~26%, false | ~54%, true | Moderate mismatch |
| chunk_003_004 | ~91%, true | ~50%, false | **48 high / 49 ~0** on overlap — MEAN near threshold |

### 5.4 Reference scores — `poses_probe_chunk002_seg0_4s.npy`

Isolated first **48 frames** of ses2 `chunk_002` (main driver of overlap `001_002` alarm):

| Field | Value |
|-------|--------|
| `robbery_score` (MEAN) | **0.948** |
| model 48 `p_robo` | **1.0** (logit gap ~+22.7) |
| model 49 `p_robo` | **0.895** |
| HAR v2 | `hand_to_body: false`, `INCOMPATIBLE`, `LEJOS_TODO_EL_GESTO`, `motion_suppress: true` |

**Human perception:** User reports person **not clearly bringing hand to body** in video. **Desired model behavior:** `p_robo` **≪ 0.5** on this file (and on full `chunk_002`), same as a confident negative clip in training. **Actual:** saturated robbery logits. Min wrist–hip in normalized pose **~0.06** shows motion uses features that **do not align** with HAR pocket geometry — either the model must learn a **different discriminative boundary** or accept **extra input features**; agents should recommend which.

**Note:** On **full** 96-frame `chunk_002`, HAR may show `ZONA_GRIS` / `KEEP_INDETERMINADO`; on **4 s probe**, HAR can be stricter (`DISCARD_SUPPRESS`). Both deny `hand_to_body`.

---

## 6. Why we believe false positives happen (hypotheses)

### 6.1 Train / inference distribution mismatch (primary)

| Training | Production |
|----------|------------|
| One `.npy` = one action label, variable length, often **centered on the action** | Fixed **8 s** grid per chunk, regardless of behavior |
| Staged **6/14**: stand → core gesture → stand | Real shopping: walk, browse, mixed phases, **chunk cuts mid-gesture** |
| Shopping clip may include walk + act + walk in **one labeled file** | Each 8 s window is scored **independently** as binary robbery |
| Evaluation: **one score per training clip** | **~2n−1 scores per session** (chunks + overlaps) |

The model learns “this **entire file** is robbery” including peripheral standing frames, while production fires on **sub-windows** that only contain innocent-looking motion similar to the peripheral part of positive clips.

### 6.2 Semantic / kinematic collision (pose-only)

Categories **3** (basket/cart/bag shopping), **4** (pick and carry), **14** (put object in bag), and **6** (hide product) share **similar upper-body trajectories** in 8-keypoint space. The label difference is **object ownership and context**, not arm geometry. A binary **6|14 vs rest** classifier on pose alone is **ill-posed** for fine-grained shopping gestures.

### 6.3 Preprocessing and mask effects

- **`valid_mask` filtering** changes effective length and time-warping; can amplify scores on chunks 001–003 in ses2 (full chunk low without mask, high with mask).
- **Overlap + mask drops** can create large jumps between valid frames (session A).
- **`temporal_resize`:** padding short sequences (e.g. 48→96) injects **zero-velocity tail** — affects scores when scoring 4 s probes.

### 6.4 Overlap windows as extra hypothesis tests

Overlap is designed to recover **robbery split across chunk boundaries**. It also **doubles exposure** to FP: any high-scoring 4 s segment inside an overlap triggers **~100%** even when adjacent chunk halves alone are benign (ses2: chunk_001 tail benign + chunk_002 head hot).

### 6.5 Model / ensemble behavior

- **False positives are consistent** across many architectures in campaign audits (not random noise).
- **Ensemble disagreement** (48 vs 49) can lower or raise MEAN per window; field pair **48|49** may differ from campaign-best **49|53**.
- **Logits on clear FPs** are often **saturated** (large positive class-1 margin) — switching from probability to logits does **not** fix confident FPs.

### 6.6 HAR as a contrast signal (for dataset design, not as the fix)

HAR v2 encodes **pocket / approach geometry** explicitly. On ses2 it **denies** theft on the same windows where motion is **~100%**. That is evidence the **motion decision boundary is mis-learned or mis-evaluated**, not that HAR should permanently carry stage-1. Reasonable training uses this gap to **build negatives** (sessions like ses2) or **multi-task targets**, so the TCN learns to **agree on obvious non-robbery** without requiring a runtime veto.

---

## 7. What is probably *not* the main cause

- Wrong file paths for 48|49 in early debugging (was an issue once; ses2 scores align with `p1.json` where parity holds).
- User error in overlap construction for ses2 `001_002` (stitch is smooth; problem is **content of chunk_002 start**).
- “Model never trained on shopping” — negatives **include** cat 3/4, but **not in production window distribution** and not with multi-window-per-session evaluation.

---

## 8. Directions to reduce false positives (training-first)

### 8.1 Training data and objective (**primary — what we want agents to expand**)

1. **Production-shaped examples (same tensor pipeline as C2):**
   - Build training items as **8 s `poses_full` chunks** and **overlap windows**, not only whole action `.npy` files.
   - **Negatives:** slide windows over real continuous sessions with **no robbery event**; include **category 3/4** action clips sliced the same way; add **ses2** (and exports from `p1.json` FPs) as **must-pass** hard negatives.
   - **Overlaps:** train explicit **negative** concat windows when **both halves are benign** but today score high (ses2 `001_002`); train **positive** overlaps only from true split-robbery sessions (`09e7c4…`).
2. **Label handling with action-level supervision only:**
   - **Positives 6/14:** MIL or soft labels — **core gesture** frames high weight, **stand still** prefix/suffix low weight or “ignore”, so the model is not rewarded for labeling whole-file positives on non-gesture segments (reduces “everything looks like robbery periphery” effect).
   - **Negatives 3/4:** oversample windows that contain **only walk or reach-to-shelf** segments, matching prod FPs (chunk_002 head, etc.).
3. **Match inference:**
   - Same **`valid_mask`** policy as production per window type; include mask on/off in augmentation so the model is not brittle (ses2 chunk_002: ~0% without mask, ~99% with mask on full chunk).
   - Train with **`temporal_resize(96)`** including padded tails so the model sees the same artifacts as inference.
4. **Loss / sampling:**
   - Heavily up-weight **hard-negative production windows**; consider focal loss or class-balanced batching tuned on **replay FP rate**, not clip accuracy.
   - Optional: **distillation** — soft target `p_robo ≈ 0` on HAR-`INCOMPATIBLE` / `LEJOS` windows (if you accept using HAR as teacher during training only).
5. **Inputs beyond bare pose (if you conclude pose-only is insufficient):**
   - Concatenate cheap **kinematic scalars** (wrist–hip, approach proxies) or HAR features **as input channels** so the model can learn geometry the TCN currently confuses with robbery.

### 8.2 Evaluation and model selection (**required for any retrain**)

1. **Replay benchmark:** ses2 + other FP sessions + `09e7c4…` (split robbery); metric = **fraction of windows with `p_robo ≥ 0.5`**, and **sessions with any alarm**, at fixed recall on positives.
2. Pick checkpoint by **replay**, not val clip F1 alone; re-check **49+53** vs **48+49** on replay, not only campaign val grid.
3. **Parity gate:** golden tensors MinIO → PyTorch vs Triton before trusting offline scores (chunk_006 mismatch shows risk).

### 8.3 Product / orchestration (**out of scope as primary answer; mention only**)

Overlap skip rules, session aggregation, or HAR veto may reduce operator pain **before** a new model ships; they **do not replace** teaching the model that ses2 windows are negatives. Fix preprocess parity issues that inflate scores without retraining.

---

## 9. Open questions for reviewing agents (training-focused)

1. **MIL / weak labels** for 6/14 given **stand → gesture → stand** — best practice when only whole-clip labels exist?
2. **How many hard-negative production windows** (ses2-like) needed before motion scores drop **below 0.5** on those windows without killing **`09e7c4…` overlap recall**?
3. **Explicit negative training on overlap concat** when both halves are benign — recommended ratio vs standard action clips?
4. **Mask policy in training:** single policy or mixture to match prod and avoid mask-amplified FPs?
5. **Can a retrained pose-only model separate 3/4 from 6/14**, or must we add **input channels** / auxiliary head? If extra inputs are required, specify **trainable** features only from pose + mask + timestamps (no new human labels).
6. **Loss and checkpoint selection** on **session replay** — concrete metric formula and acceptance thresholds for ses2.
7. **Architecture:** stay on `tcn_gru` + velocity or change (e.g. multi-scale windows, streaming GRU across chunks) given **~2n−1 windows per session**?
8. **Threshold / ensemble:** after retrain, is MEAN @ 0.50 still appropriate, or should thresholds be **per window type** (chunk vs overlap)?

---

## 10. Reproducibility (if you have the repo + exports)

| Resource | Description |
|----------|-------------|
| **Investigation scripts** | `IAModel/experiments/training/campaign/{score_pose_ensemble,diagnose_chunk_overlap,motion_har_veto_demo}.py` |
| **Benchmark session ses2** | Export folder with `chunk_00k/poses_full.npy`, `valid_mask.npy`, `p1.json`, optional prebuilt `poses_overlap_*_dense.npy` |
| **Session id ses2** | `64a58a936e9347bcb2cfb3b06ae27a4e` |
| **4 s probe** | `poses_probe_chunk002_seg0_4s.npy` + `valid_mask_probe_chunk002_seg0_4s.npy` (generated in investigation) |
| **Session A overlap** | `poses_overlap_1_2_second_half_first_half.npy` + mask twin |
| **Checkpoints (field)** | `modelo_48.pt`, `modelo_49.pt` (tcn_gru, `seq_len=96`); campaign alt **49+53** under `artifacts/runs/campaign_20261103_230735/models/bin_full/` |
| **HAR v2 code** | `Kanvis-Orchestrator/kanvis-orchestrator/app/vendor/har/pocket_infer.py` |

Agents **without** local files should rely on **tables in Section 5 and Appendix A** as experimental ground truth.

**Example commands:**

```bash
cd IAModel/experiments/training/campaign

python3 score_pose_ensemble.py \
  --poses-full /path/to/poses_probe_chunk002_seg0_4s.npy \
  --valid-mask /path/to/valid_mask_probe_chunk002_seg0_4s.npy \
  --model-a /path/to/modelo_48.pt \
  --model-b /path/to/modelo_49.pt \
  --rule mean --threshold 0.50 --json

python3 diagnose_chunk_overlap.py \
  --poses-a /path/to/chunk_001/poses_full.npy \
  --mask-a /path/to/chunk_001/valid_mask.npy \
  --poses-b /path/to/chunk_002/poses_full.npy \
  --mask-b /path/to/chunk_002/valid_mask.npy \
  --model-a /path/to/modelo_48.pt --model-b /path/to/modelo_49.pt

python3 motion_har_veto_demo.py \
  --session-dir /path/to/ses2 \
  --p1-json /path/to/ses2/p1.json \
  --model-a /path/to/modelo_48.pt \
  --model-b /path/to/modelo_49.pt \
  --har-root /path/to/Kanvis-Orchestrator/kanvis-orchestrator/app/vendor/har \
  --json
```

**Dependencies:** Python 3, PyTorch, NumPy; HAR vendor on `PYTHONPATH` via `--har-root`; optional matplotlib for overlap plots.

---

## 11. Glossary

| Term | Meaning |
|------|---------|
| **6 / 14** | Positive robbery-related action categories in `bin_full` |
| **3 / 4** | Shopping-like negatives (basket, carry) — kinematically confusable |
| **poses_full** | Dense 12 Hz pose grid per chunk, shape `[96, 8, 2]` typical |
| **valid_mask** | Boolean per frame, ≥5 visible keypoints |
| **chunk_{a}_{b}** | Overlap window: second half of chunk a + first half of chunk b |
| **HAR v2 pocket** | `c3_har_pocket_v2`, hand-to-body / pocket geometry filter |
| **p1.json** | Per-session/chunk/overlap metadata: motion scores, `har_pocket`, alarms |
| **MEAN / AND** | Ensemble rules over two checkpoints |

---

## Appendix A — ses2 production motion alarms (from `p1.json`)

For cross-check with offline tools (threshold 0.50, MEAN):

| Window | prod `robbery_score` | prod `robbery_alarm` |
|--------|---------------------|----------------------|
| chunk_001 | 0.617 | **true** |
| chunk_001_002 | 1.000 | **true** |
| chunk_002 | 0.988 | **true** |
| chunk_002_003 | 0.0007 | false |
| chunk_003 | 0.766 | **true** |
| chunk_003_004 | 0.909 | **true** |
| chunk_004 | 0.260 | false |
| chunk_004_005 | 0.201 | false |
| chunk_005 | 0.801 | **true** |
| chunk_005_006 | 0.999 | **true** |
| chunk_006 | ~0 | false |

Session-level: **7 motion alarms** on a **~55 s normal shopping** export.

## Appendix B — Earlier debugging lessons (do not ignore)

1. **Wrong or mixed checkpoint files** in `modelos_v2/` once caused absurd scores; always verify SHA / campaign id for **48** and **49**.
2. **`valid_mask` on vs off** can flip scores from **~0% to ~99%** on the same chunk (ses2 `chunk_002` full 96 rows).
3. **Logits** on confident FPs are **extreme**; tuning softmax threshold alone is insufficient.
4. **Filters-only** (HAR veto) can hide ses2 alarms in demos but **do not satisfy** the training goal: motion itself should score **low** on those tensors.

## Appendix D — Acceptance tests after retraining (suggested)

| Test | Pass criterion (example — agents may refine) |
|------|-----------------------------------------------|
| ses2 all windows | **No** window with MEAN `p_robo ≥ 0.5` (or ≥ 0.3 if business wants margin), except parity-failed windows until fixed |
| ses2 `poses_probe_chunk002_seg0_4s.npy` | `p_robo < 0.1` for both 48 and 49 |
| Split robbery `09e7c4…` overlap `001_002` | MEAN `p_robo ≥ 0.5` (recall preserved) |
| Val clip F1 | Reported but **not** sole gate — must not regress replay |

## Appendix C — Campaign context (high level)

- **Cell:** `bin_full`; positive labels **6, 14** only.
- **Architecture:** `tcn_gru`, `seq_len=96`, input = normalized pose + velocity per joint.
- **Val metric risk:** low FP count on val clips (~0.5–1% clip FP) **understates** session-level FP when scoring **many windows per visit**; owner reports **~20–30 FPs per ~700–1000 sessions** at stage-1 historically (see in-repo research brief).
- **Recommended alternate pair from IAModel campaign:** **49+53** (~88% F1 val) vs field **48+49**.

---

*Document prepared for cross-agent review. Revision: expanded for self-contained handoff (Nov 2025).*