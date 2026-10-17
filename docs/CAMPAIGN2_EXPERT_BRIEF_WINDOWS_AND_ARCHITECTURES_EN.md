# Expert assignment — Kanvis Stage-1 training: window length & architecture strategy

**Date:** 2026-10-17  
**Audience:** External agent acting as a **senior expert in computer vision, human activity recognition (HAR), and deep learning**.  
**Language of your response:** English.  
**This document is 100% self-contained.** You do not need access to the repository, other files, or prior chats. Everything required to answer is below.

---

## YOUR TASK (read this first)

The owner is building **Kanvis**, a real-time shoplifting detector that uses **human skeleton sequences** (not full customer video) in supermarkets.

They have:

- A **legacy training pipeline** that produced models with **decent clip-level validation** but **much worse false-positive rates in production**.
- A **new pipeline (Campaign2 / train_v2)** that fixes many legacy bugs and trains on **8-second sliding windows at 12 Hz**, but a **full training run on ~19k clips is not finished yet**.

The owner wants **your expert judgment**, not clarification questions about “what to do with this document.”

### What you MUST deliver

Produce **one new Markdown file** (your complete answer). Suggested filename: `EXPERT_RESPONSE_WINDOWS_ARCH_YYYYMMDD.md`.

Your file MUST contain **all** of the following sections (use these headings or equivalent):

1. **Executive summary** (≤15 bullets): direct recommendations the owner can act on this week.
2. **Answers to every numbered question** in §8 (Q1–Q18). Do not skip any. If evidence is insufficient, state what you assume and give a conditional answer.
3. **Concrete proposal** (§9): a single recommended path — window length(s), stride, train vs eval protocol, architecture shortlist (4–10 items max), seeds, ensemble rule, and what **not** to run (e.g. 60-model grid).
4. **Legacy diagnosis**: why legacy training likely yielded “mediocre” production behavior despite okay val (use §6).
5. **Ranked roadmap**: next 4–8 weeks, ordered by **expected reduction in false positives per session/day** at acceptable recall, with engineering cost (S/M/L).
6. **Acceptance criteria**: how the owner should know the next campaign succeeded (metrics in **session/event units**, not clip F1 alone).
7. **Risks & open assumptions**: what could invalidate your advice.

**Take as much time as you need.** Reason carefully. Quantify where possible (even rough orders of magnitude). Do not reply with “what would you like me to do?” — the task is defined above.

---

## 1. System context (what Kanvis is)

### 1.1 Product goal

- **Input:** time series of 2D keypoints for one person (upper body).
- **Output:** a **theft alarm score** (Stage 1). Later stages (SAM3, HAR, fusion) may confirm alarms; **this assignment is only Stage 1 training/eval design**.
- **Business priority:** **false positives first**. Owner preference: roughly **50% event recall with 0 FP/day** is better than **75% recall with 5 FP/day**. Working development targets often cite **~10 false sessions per 1000 sessions** while iterating; aspirational end goal **~≤1 FP/day** at ~1000 sessions/day.
- **Store context today:** one store, **9 cameras**, ~1000 customer **sessions/day** (one session ≈ one tracked person through the store, with possible ID switches).

### 1.2 Skeleton format

- **8 joints** (legacy order): left/right shoulder, elbow, wrist, hip — abbreviated LS, RS, LE, RE, LW, RW, LH, RH.
- Stored as **`poses_full.npy`** shape **`[T, 8, 2]`** (x, y), often normalized to image coordinates (x/W, y/H). A filtered variant **`poses.npy`** may drop low-confidence frames.
- Optional **`valid_mask.npy`**, **`meta.json`** (fps, video source, users, clip duration, labels).

### 1.3 Labeled training data (October 2026 campaign)

| Quantity | Value |
|----------|--------|
| Total labeled clips | **~19,043** |
| Class **6** (robbery/concealment label) | **~2,592 clips** |
| Negatives | Other action categories (walking, paying, etc.) |
| Label granularity | **One category per clip** (human cut mp4 → npy + category). Subclips can have their own label. |
| Label noise (known) | Category 6 includes **concealment-like motion**; some clips are **business false thefts** (e.g. customer putting **own phone** in pocket). Skeleton alone cannot see object ownership. |

### 1.4 Legacy validation vs production (the core pain)

| Setting | Approximate performance |
|---------|-------------------------|
| **Legacy val (clip-level)** | Best pair **model #41 + model #48**, rule **AND @ threshold 0.80** on sigmoid scores → **F1 ~76%**, **recall ~64%**, **11 FP / 1619 negative clips (~0.68%)**. |
| **Production (session-level, reference)** | Order of **~20–30 false alarms per ~650–1000 sessions** (~2–3% of sessions flagged, not directly comparable to clip val). |

The owner believes **training/evaluation protocol mismatch** (full clips vs production chunks, splits, thresholds) explains much of the gap — not only “weak architecture.”

### 1.5 Production inference (how the model is actually fed)

Production service **C1** (separate repo `kanvis-monitoring`):

- RTSP video → pose on **Triton/TensorRT** → tracker → **one session per person/track**.
- Pose is written as **fixed-length chunks**, not full labeled clips:
  - **`window_seconds = 8`** (configurable in principle).
  - **`stride_seconds = 4`** between chunk starts (50% overlap).
  - Per chunk: e.g. `poses_full.npy`, `valid_mask.npy` uploaded (e.g. MinIO).
- Stage-1 model scores **each chunk**; aggregation (max over windows, ensemble AND, etc.) happens in the serving stack.
- **Important:** chunks can be **shorter than 8 s** at session start/end (padding/mask policy matters).

**Known second mismatch:** offline training data were often extracted with **Ultralytics YOLO-pose** pipelines; production uses **Triton**. Pose parity is a parallel engineering track; assume **some domain gap** remains until golden tests pass.

### 1.6 Example production-like fragment (owner sample)

File: `mergedfps.npy` (example path on owner machine)

| Property | Value |
|----------|--------|
| Shape | **(60, 8, 2)** float32 |
| Frames | 60 @ **12 Hz** → **5.0 seconds** of motion |
| Coords | Normalized ~0.10–0.49 |

This is **shorter than the 8 s / 96-frame contract** used in the new trainer.

---

## 2. Legacy training pipeline (what we did before Campaign2)

Path conceptually: `train_model_operations.py` + `campaign/train_campaign.py` + `campaign/evaluate_campaign.py`.

### 2.1 Data flow (legacy)

1. Discover clips under `data_result/<category>/<clip>/user_*/poses_full.npy`.
2. Optional filters (often **disabled**); **`single_user_only=True`** in many runs → drops multi-user clips (~**33% of robbery clips** lost in one audit sample).
3. **Stratified split by clip UID** (not robust session grouping) → train/val/test.
4. **PoseDataset:** load npy → NaN→0 → normalize → velocity → **`temporal_resize(seq_len)`** to fixed length (64 or 96 depending on model).
5. **Train** with Adam, often **no scheduler / no early stopping / no grad clip**; “best” checkpoint selected by **val accuracy**.
6. **Campaign eval:** threshold sweeps, ensemble grids (AND/MEAN), often tuned on **val**; **20 model IDs** from a pool of **60** configured experiments.

### 2.2 Why `temporal_resize` is harmful (legacy H4)

- Variable clip length (e.g. 4 s vs 30 s) is **warped** to the same `seq_len` by subsampling or **padding with last frame repeat**.
- Velocity is computed **before** resize on original timeline → **position and velocity channels become physically inconsistent** after warp.
- ~**44%** of clips needed **>13% padding** at seq_len=96 in one measured sample.
- **Clip duration correlates with category** → model can learn **length shortcuts** that do not exist in production (fixed 8 s windows).
- Some legacy **sliding-window eval** used **~3 s windows** while training used **whole clips** → another time-scale mismatch.

### 2.3 Other high-severity legacy bugs (summary)

| ID | Issue | Effect |
|----|--------|--------|
| **H1** | Best checkpoint saved as **live reference** to weights; file often = **last epoch**, not best val | Wrong weights shipped |
| **H2** | **Per-clip split** + same-day/session leakage | Optimistic val, poor generalization |
| **H3** | Threshold/ensemble **optimized on val**; prevalence in val (many robbery clips) ≠ production | Wrong operating point for FP/day |
| **H5** | Augmentations **shift/scale** null after normalization; little real on-the-fly aug | Weak generalization |

### 2.4 Legacy architectures (what was tried)

~**15 families**, **20 IDs** in the October campaign, including:

- Small TCN (tiny receptive field ~9 frames on some variants).
- Dilated TCN, ResTCN.
- **#41:** TCN-LSTM, hidden 256, seq_len **64**, ~1–2M params.
- **#48:** TCN-GRU, hidden 128, seq_len **96**.
- Transformers/Conformers **without positional encoding** on time.
- ST-GCN with **incorrect graph** (chain on joint index, not anatomy).
- BiLSTM/BiGRU using **`output[:, -1]`** (backward direction only “sees” last frame).

Production today still uses **#41+#48 AND @ 0.80** in many deployments.

**Audit conclusion:** architecture rankings under legacy protocol are **not trustworthy** until eval protocol is fixed.

### 2.5 Stagnant FP–recall frontier (legacy campaigns)

Reported pattern on validation (clip-level): pushing FP budget gives limited recall gains; adding data/architectures **did not move the frontier much** — hypothesis: **data, labels, and protocol** dominate, not the 20th architecture variant.

---

## 3. Prior external expert advice (2026-10-15, summarized)

An earlier expert response (same project) stated:

- **Do not** run another broad architecture sweep first.
- Fix: checkpointing, **grouped chronological eval**, frozen threshold, **physical-time windows**, masks, normalization, label semantics.
- **Recommended first model:** **8 s, 12 Hz, 96 steps**, **masked dilated TCN width 64**, five dilated blocks, **masked mean+max readout**, ~**0.14M params**, **62 input channels**, BCE + auxiliary **concealment** head, **3 seeds** (17, 42, 103), AdamW 100 epochs, select by **recall at FP budget** (not val accuracy).
- **Ensemble v2:** **mean of seed margins**, not AND #41+#48 for new stack.
- After baseline: compare **4 s and 12 s** windows — not dozens of lengths without protocol.
- Next compact arch comparators: **correctly pooled GRU**, **anatomical ST-GCN**, **small Transformer with positional encoding**.

You may **agree or disagree** with this prior advice; the owner wants **your** integrated proposal.

---

## 4. New pipeline: Campaign2 + train_v2 (current implementation)

### 4.1 Intent

- Train and evaluate with the **same temporal contract** target as production: **fixed-duration windows in real seconds**, not whole-clip time warp.
- Parallel GPU training on **RTX 5090** (up to **~8 concurrent jobs**).
- Keep legacy code read-only; new code under `campaign2/` and `train_v2/`.

### 4.2 Temporal & data processing (v2)

| Setting | Value |
|---------|--------|
| Resampling | **12 Hz** from source fps in meta |
| Window | **8.0 s** → **96 frames** |
| Stride (train/eval offline) | **1.0 s** |
| Storage | Build mmap cache of **full resampled clip**; sample **sliding windows** in the dataloader |
| Masks | Invalid joints/frames flagged; pad short clips at end with mask |
| Normalization | Torso-relative, isotropic coordinates, wrist–torso distances |
| Input dim | ~**62 channels** |
| Window labels | **`tail_positive`:** only last **~3 s** of class-6 clips labeled positive; MIL bag loss (LSE) |
| Loss | BCE theft + auxiliary concealment + MIL bag |
| Sampler | ~25% pos / 25% hard neg classes / 50% ordinary neg (window-level) |
| Best checkpoint | **deepcopy** of weights (H1 fixed) |
| Selection metric | **recall@FP** on val (~11 FP clip budget in config) |
| Threshold | Fit on **`threshold_calib` split T**; eval default split **`val_dev`** |
| Ensemble | Mean over **3 seeds** after training |

### 4.3 Training matrix (configured today)

- Architectures: **`masked_tcn64`**, **`gru64_pooled`** only (others not coded in v2 yet).
- **3 seeds** × **2 pose sources** (full vs filtered npy) = **12 training jobs** per matrix campaign.
- **100 epochs** (debug mode: 50 clips, 4 epochs — for infra test only).

### 4.4 Operational sequence

1. GPU VRAM probe → recommend parallel job count.  
2. Preflight: inventory all clips, build **two caches** (full + filtered poses).  
3. Parallel train all jobs.  
4. Calibrate threshold per checkpoint on split T.  
5. Eval on val_dev (clip-max aggregation over windows).  
6. Summaries + seed ensemble + leaderboard.

**Status:** debug run **passed end-to-end** (~0% recall on tiny val — expected). **Full ~19k clip run pending.**

### 4.5 Gaps vs production (still open)

- Offline stride **1 s** vs production stride **4 s**.
- Eval mostly **clip-level max**, not full **session replay** with cooldown/dedup.
- Training data = **labeled action segments**, not random **background windows** from long sessions.
- No systematic comparison of window lengths **2–8 s** yet.
- Session/event FP per day **not** primary offline metric yet.
- Pose offline vs Triton **not** fully aligned.

---

## 5. Owner’s new hypothesis (motivation for this assignment)

> “We used to train on **complete npy files** of **variable duration**. In production we score **fixed ~8 s chunks** (adjustable). Could that explain poor production quality? Should we **evaluate** the model with **2, 3, 4, 6, 8 s** inputs to pick the best production window? Should we even **train** with multiple chunk lengths?”

Campaign2 already uses **8 s sliding windows** on resampled clips (not legacy full-clip resize), but **does not** answer whether **2–8 s multi-scale eval/train** is worth it, nor whether **5 s fragments** (like the sample npy) need a different policy.

---

## 6. Legacy pitfalls vs Campaign2 fixes (for your diagnosis section)

Use this table in your answer — extend or correct it.

| Issue | Legacy | Campaign2 status |
|-------|--------|------------------|
| Time warp whole clip | Yes (`temporal_resize`) | **No** for v2 train (fixed 8 s windows) |
| Train/eval window length mismatch | Yes (clip vs 3 s eval) | **Partial** (8 s train; stride/aggr differ from prod) |
| Checkpoint bug H1 | Yes | **Fixed** |
| Split leakage H2 | Per-clip | **Partial** group split |
| Val threshold hunting H3 | Yes | **Partial** (T split + frozen intent) |
| Feature/normalization M2/M3 | Weak | **Improved** in v2 |
| RNN pooling bug M1 | Yes on #41/#48 | GRU pooled in v2; legacy still P0 baseline |
| Label semantics (phone pocket) | Unaddressed | **Still open** |
| Background windows in sessions | Not trained | **Still open** |
| Pose train/prod parity | Gap | **In progress** |
| Metrics clip vs session | Clip F1 | **Still mostly clip-level** |

---

## 7. Constraints for your recommendations

- **Compute:** RTX 5090, ~32 GB VRAM; ~8 parallel small-model jobs feasible.
- **Owner prefers lighter models** if within ~**2 percentage points recall** at the same FP budget (Pareto).
- **Do not recommend** a **60-model architecture grid** on October data; prefer **4–10 well-chosen candidates** × **3 seeds**.
- **Independent theft events/sessions** are **far fewer** than 2592 positive clips — treat **effective sample size** as sessions/events, not clips.
- Stage 1 must remain **deployable** (latency/memory matter, but 5090 is generous).

---

## 8. Numbered questions — you MUST answer ALL (Q1–Q18)

### Block A — Window length & temporal contract

**Q1.** After Campaign2’s **8 s @ 12 Hz** training, is the owner’s **full-npy vs 8 s chunk** concern still the **#1 train/prod mismatch**, or are **stride (1 vs 4 s)**, **session aggregation**, and **short chunks (<8 s)** now dominant?

**Q2.** Should the team run **multi-length evaluation** (same trained weights, vary window 2/3/4/6/8 s with proper mask/pad) **before** any multi-length training? **Yes/no** and why.

**Q3.** If yes to multi-length eval, which lengths to test **first** (owner list 2–8 s vs prior expert 4/8/12 s)? Give an **ordered list** and stopping rule.

**Q4.** For multi-length eval, should `seq_len` scale as **`12 × window_seconds`** (96 for 8 s) or stay fixed at 96 with temporal subsampling? Recommend one approach.

**Q5.** Should production window length be chosen by **clip-level val**, **session replay**, or **both**? Define the **primary metric** (e.g. event recall @ ≤10 FP/1000 sessions).

**Q6.** Is **multi-scale training** (random 2–8 s windows during train) likely to help FP/recall on **weak whole-clip labels**, or hurt without **interval annotations**? Under what conditions would you recommend it?

**Q7.** Is a **two-stream** design (e.g. 4 s “evidence” + 8 s “context”) worth the complexity vs single 8 s for **brief concealment gestures**?

**Q8.** Must offline eval mirror production **`stride_seconds = 4`** and **max-over-windows** aggregation to choose window length? **Yes/no** and protocol sketch.

**Q9.** For clips labeled as **whole actions**, should training windows be restricted to **inside the labeled segment** only, or sampled on the **full npy** including approach/walk-up? Impact on FP?

### Block B — Architectures

**Q10.** Given **~19k clips / ~2592 pos / few independent sessions**, what is a prudent **parameter budget** (order of magnitude)?

**Q11.** Provide a **final shortlist of 4–10 architecture families** (not 60) to train after the 8 s baseline, with **one-line rationale each**.

**Q12.** Should the team implement **anatomical ST-GCN lite** next, or is **TCN + pooled GRU + seed ensemble** sufficient for P2?

**Q13.** Are **Transformers / Conformers** justified on this data size if **positional encoding + mask** are done correctly, or expected **low ROI**?

**Q14.** Should **legacy #48** be retrained with the **v2 loader** (96-step windows) as a bridge, or skip?

**Q15.** **Deployment ensemble:** recommend one: **(a)** mean of 3 TCN seeds, **(b)** TCN+GRU fusion, **(c)** keep #41+#48 AND interim, **(d)** other — with justification.

**Q16.** Are skeleton **video models** (PoseC3D, MS-G3D, etc.) worth the cost here, or is bottleneck **data/protocol**?

**Q17.** For **5 s production fragments** (60 frames @ 12 Hz), does architecture choice shift (GRU vs TCN vs other)?

### Block C — Beyond windows & arch

**Q18.** Rank the **top 5 non-architecture levers** (labels, background mining, replay benchmark, pose parity, split design, hard negs, post-processing, etc.) for reducing **production FP** while preserving recall.

---

## 9. Required “Concrete proposal” section (template for your response)

Your **`EXPERT_RESPONSE_...md`** must include a section like this, filled in with your decisions (not placeholders):

```markdown
## Concrete proposal (binding recommendation)

### Production temporal contract
- window_seconds: ...
- stride_seconds: ...
- target_hz: ...
- seq_len: ...
- Short-fragment policy (<8 s): ...

### Training protocol
- Window lengths used in training: ...
- Multi-scale: yes/no, schedule: ...
- Stride in training: ...
- Label policy on class-6 clips: ...

### Evaluation protocol (mandatory before deploy)
- Splits: ...
- Metrics: ...
- Multi-length eval grid: ...
- Production replay: yes/no, size: ...

### Architecture plan (max N candidates)
1. ...
2. ...
(seeds, epochs, ensemble)

### Explicitly DO NOT do (save compute)
- ...

### 4-week execution order
Week 1: ...
Week 2: ...
...
```

---

## 10. Optional follow-up data (NOT required to answer)

If you want more precision later, the owner *could* provide: clip length distribution @ 12 Hz, full matrix-run metrics, or pose parity reports. **Your assignment is complete without them** using the facts in §1–§7.

---

## 11. Document metadata

- **Prepared for:** external expert agent (CV / HAR / deep learning).  
- **Prepared by:** Kanvis training team (Campaign2 implementation).  
- **Version:** 2026-10-17, self-contained assignment brief.

**End of brief. Begin your response Markdown file now.**
