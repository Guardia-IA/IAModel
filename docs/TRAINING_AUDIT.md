# Kanvis model training audit (IAModel) — input brief for an expert agent

**Date:** 2026-10-15
**Language:** English (this document and the response you must produce).
**Scope:** everything from the `.npy` pose files to a trained, evaluated checkpoint and a decision rule (threshold / ensemble).
**Out of scope:** pose extraction (YOLO/Triton/tracker). **The `.npy` files are assumed to be correct.**
**Paths:** relative to `IAModel/experiments/training/` unless stated otherwise.

---

## READ FIRST — Mandate for the receiving expert agent

### Who you are
You are a **senior expert** in **software engineering** (ML systems, training pipelines, reproducibility, testing, code review, MLOps), **artificial intelligence and machine learning** (model training and optimization, deep learning architectures, loss design, regularization, hyper-parameter search), skeleton/time-series action recognition, low-false-positive classification, calibration and decision theory, noisy-label and domain-shift learning, evaluation design with statistical rigor, and retail loss-prevention systems. **Think and work as that expert would**: read the code paths as a code reviewer looking for bugs and silent train/eval inconsistencies, reason as a machine-learning scientist about what limits the model, and design as a software engineer who must make the result reproducible and deployable. Take the stance of a **critical reviewer**: verify every claim in this document against the repository and artifacts where you can, and say explicitly when something is a hypothesis rather than a fact.

### Depth of reasoning (mandatory)
**Perform the deepest, most careful, genuinely excellent reasoning you are capable of. There is no time limit, no urgency and no penalty for taking long: take all the time you need — the owner explicitly asks for very good, in-depth reasoning, not a quick answer.** Think step by step, challenge your first conclusion at least once, and cross-check numbers before you write them. Do not stop at the first plausible explanation, do not shorten or summarize your work to save time or space, and do not give generic answers. Explore alternative hypotheses, re-check your conclusions before writing them down, and prefer quantified evidence to opinion. **The owner expects a reasoning-and-detection job, not an execution job:** you will most likely receive **only this document** (the owner may later supply extra files if your response says which ones would help), you are **not required to run code, train models or access the data**, and you must not leave any answer empty for lack of execution. Work analytically from the facts and figures in this brief (see **C.6 Owner-confirmed facts**, which are authoritative); if you *can* run something (repository, artifacts, simulated logits), do it and say so; if not, **write the experiment as an exact protocol** (data, split, metric, expected outcome, decision rule) so the owner can run it. Use any sources you consider appropriate (repository, literature on skeleton-based action recognition, low-FPR learning, calibration, conformal prediction, noisy labels, domain shift; open-source projects; web search if available) and **cite them**.

### NO INTERACTION — you must not ask questions
**You cannot ask the owner or anyone anything, and you must not write questions addressed to the owner or the user anywhere** (not in chat, not in the response file). The person who launches you will only read the file you produce. Therefore:
- Every question in §16 is a **design question for you to answer**, not a question for the owner. For facts that only the owner knows, **state the assumption you adopt**, give a **conditional recommendation** ("if X is true do A, otherwise B") and name the **cheapest verification** that would decide between the branches.
- Where this document seems contradictory or wrong, say so and state which reading you adopted.
- Never block on missing information or inaccessible files: say what you could not verify and continue.

### What you must produce
**A new Markdown (`.md`) file, written in English, that contains your complete analysis and the answers to everything asked in this document (nothing may be left only in chat).** Suggested path: `IAModel/docs/TRAINING_EXPERT_RESPONSE_<YYYYMMDD>.md` (if you cannot write there, use the closest writable location and state the path at the end). **Do not edit this document** (it is the input brief). If you produce scripts or additional deliverables, put them in separate files and link them from the main response.

Required structure of the response file:
1. **Executive summary** (plain language, ≤ 1 page): **a direct verdict on the question the owner is asking — "how can I improve my model's training, and am I doing something wrong?"** — what is wrong, what to do first, the best model recipe you recommend (architecture, `seq_len`, features, epochs…), the realistic achievable operating point (false positives per day / per 1 000 sessions, event-level recall) and your **ideas for post-processing/filters that cut false positives** (§16.9).
2. **Critical review of this audit**: what you confirm, what you reject, errors found, what remains unverified (mark each item *verified / computed / hypothesis*).
3. **Ranked diagnosis** of why production false positives are high and why recall is limited, with evidence, estimated share, confidence and the quickest confirmation for each cause.
4. **One coherent, prioritized end-to-end solution** (not a menu): data/labels, filters, windows and time contract, features/normalization, architecture, loss/sampling, calibration and **decision rule**, evaluation protocol, post-processing, ordered roadmap with acceptance criteria in production units.
5. **Answers to every question in §16** (16.1–16.9), one by one and numbered as here (Q1…Q61), each with: your answer, rationale, expected effect on FP and recall, how to test it, priority. A question left unanswered must be listed with the reason and the data that would resolve it.
6. **Assessment of the findings (H1–H5, M1–M10) and of the experiment plan (E0–E12, §14)**: accept / modify / reject, cost, expected effect.
7. **Your additional proposals** (new ideas, alternative formulations, data strategies, filters, evaluation designs).
8. **Assumptions and information gaps**: numbered list; for each: the assumption, the conditional recommendation and the cheapest verification. **No questions to the owner.**
9. **Appendix**: commands/scripts used, files inspected, numbers recomputed.

Keep claims traceable (file paths, line numbers, exact numbers) and clearly separate **measured** results, **estimates** and **opinions**.

---

## Project context

### C.1 What Kanvis is and what is intended
**Kanvis** is a system to **detect shoplifting in stores in real time** from camera video, **without analyzing the full image of the customer**: the **skeleton (pose tracking)** of each person is extracted with YOLO-pose + a tracker and the **motion** is classified with sequence models (TCN/LSTM/GRU/Transformer…). The premise is that 8 upper-body keypoints are enough to detect the "pick a product and hide it" gesture (category 6) cheaply and at scale, and that later stages (SAM3, HAR, C3 fusion) confirm or discard the alarm.

**Business objectives (confirmed by the owner):**
- **False positives (FP) come first**: budget ≈ **1 false alarm per day** over ≈ **1 000 sessions/day** (≈ 0.1 % of sessions, end to end). "20 FP/day is unacceptable". Ideally the residual FP is a *plausible near-miss* (e.g. putting a phone in a pocket), not an arbitrary movement.
- **Recall**: detect ≈ **2 of every 3 robberies (66 %)**, ideally 3 of 4 (75 %), measured **per event** (one alarm during the event is enough). Losing a robbery is acceptable; FP are not.
- Phase 1 (the motion model — covered by this document): **≤ ≈ 10 FP per 1 000 clips in production** (today 20–30) at recall ≥ ≈ 66 %. Phase 2 (SAM3/HAR/other filters): remove ≥ 90 % of the remaining FP at ≤ 10 % robbery loss.
- Generalist requirement: **no action-by-action tuning** (there are no labels, and none wanted, for "hand on back", "scratching", etc.); the solution must reject the space of everyday movements in a general way.

### C.2 Production chain (summary)
```
camera (15 fps) → YOLO26s-pose (TensorRT) + BoT-SORT/ReID tracker → session = one person from entry to exit
   → chunks of up to 8 s (.npy per chunk: poses_full.npy + valid_mask.npy)
   → STAGE 1 "motion" (this document): ensemble of 2 models, #41 (tcn_lstm, seq_len 64) and #48 (tcn_gru, seq_len 96)
        AND rule: score = min(p1, p2), alarm if both ≥ 0.80; applied to each chunk and to the overlapping window N+N+1
   → only if candidate: SAM3 (object state: in hand / in cart / hidden) + HAR (gesture rules) → C3 fusion → verdict
```
One session = one person: in an aisle with 4 people there are 4 independent sessions. Crossing an aisle ≈ 1 chunk; browsing ≈ 4 chunks; an employee restocking ≈ 40+ chunks (long sessions dominate per-session FP).

### C.3 Data and labeling
- A **training clip = one person doing one action**, cut by hand with an in-house tool (start/end) and labeled with a **category 0–14** (one label per clip, **no temporal annotation inside the clip**). One `.npy` per person is extracted per clip.
- Categories (confirmed): 0 walks through without stopping · 1 walks, stops, looks, does not buy · 2 picks and **puts back** · 3 picks and **puts it in basket/cart/bag (buys)** · 4 picks and **carries it in hand** · 5 walks **carrying objects** · **6 picks and HIDES the product (ROBBERY, positive class)** · 7 employee restocking · 9 looks without buying while carrying objects · 13 employee cleaning · **14 puts an object in a bag (own belongings)**; 8/10/11/12 have no clips. **Most confusable with robbery: 14, 3, 4, 2 (and 5/9)**: kinematically almost identical in 8 keypoints; the difference is *which object* and *the context*, not the arm pose.
- Provenance of robberies (per the brief): ≈ 63 % **staged recordings** by the team, ≈ 34 % **cuts from real sessions** (Dec 2025), ≈ 3 % continuous production. Negatives are mostly from real sessions.
- Volume in the October campaign (all splits): ≈ 19 000 clips, of which **2 592 are robbery**; the October extension added ≈ 3 650 negatives but **only +18 robberies**, which is why FN did not improve.
- Pose: 8 COCO keypoints 5–12 (shoulders, elbows, wrists, hips); no head, legs, hands or objects. `xyn` normalized, conf ≤ 0.5 → NaN, `valid_mask` (≥ 5 visible keypoints).

### C.4 Current state (references from other repo documents)
- **Best configuration (val, Oct campaign, clip level):** ensemble #41 + #48, AND @ 0.80 → F1 ≈ 76 %, recall ≈ 64 %, precision ≈ 95 %, **11 FP / 1 619 negatives (0.68 %)**, 117 FN / 323 robberies. (The July campaign, used for some figures below, had a val set of 1 241 clips: F1 ≈ 86 % with argmax and ≈ 75 % for the best low-FP ensemble; **the two val sets are different and not comparable**. The owner's own August-vs-October comparison is in **C.6.4**.)
- **The FP–recall frontier has not moved** with more models or data: FP ≤ 3 → recall ≈ 42–48 %; ≤ 8 → ≈ 58 %; ≤ 11 → ≈ 64 %; ≤ 20 → ≈ 72–74 %; ≤ 45 → ≈ 83–85 %. A sign that the limit lies in **data / labels / information**, not architecture.
- **Production:** ≈ 20 FP in 654 sessions in one day with the model alone (3.1 % of sessions; ≈ 20–30 per 1 000) vs 0.68 % in validation: a 3–4× gap partly explained by arithmetic ("several looks per session", k ≈ 4.6 → 1−(1−0.0068)^k ≈ 3 %) and partly by train↔production mismatches.
- Issues already identified outside this document (brief and prior expert response): action-centered training clips vs arbitrary 8 s production chunks; tracker ID switches; different `.engine` files between training and production; possible absence of `valid_mask` in the Triton inference path; 15 fps cameras vs 12 fps; time scaling by `seq_len`. **This audit covers training only** (assuming correct `.npy`) and complements those documents.

### C.5 Note about the local measurements
The "measured" figures in this document (per day, per source, metadata AUCs, etc.) come from an **old local sample** (10 790 clips, old extractor without `valid_mask`), **not** from the October dataset. They reveal *types* of problems and **must be re-measured** on the real dataset (Annex A). Where C.6 gives real figures from the October campaign artifacts, **C.6 prevails**.

### C.6 Owner-confirmed facts and campaign artifacts (AUTHORITATIVE — they override any assumption elsewhere in this document)

**C.6.1 What the owner wants from you.** (a) A response document that says **how to improve the training of the model, and whether anything is being done wrong**; (b) **the best achievable model version** (any change is allowed: more epochs, other architectures, other `seq_len`, other features/normalization, other loss/sampling, other decision rule); (c) **additional ideas for filters and post-processing** that reduce false positives (see §16.9). The owner will apply heuristic filters + SAM3 *afterwards*; the model must first be as good as possible on its own.

**C.6.2 Priority between FP and recall (owner's own words, translated).** "It is better to detect 2 of every 4 robberies (50 %) with **0** false positives per day than 3 of 4 (75 %) with **5** false positives per day." Concern: detect robberies, but **above all remove false positives**. Reading: FP are far more expensive than FN; the success metric is **event-level recall at a fixed, very low FP/day**. (You must turn this preference into an explicit utility/cost ratio — Q60.)

**C.6.3 Deployment and data origin.**
- **One store, 9 cameras.** All training data and production traffic come from this single store. Generalizing to other stores is *not* a stated requirement today (treat it as a future wish and say how it would change your recommendations).
- **Labeling workflow:** a person watches the clip (`.mp4`) in an in-house labeling tool and **selects the category**; the tool can **cut sub-clips from one clip and assign each its own category**. The `.npy` pose files are generated *from the labeled clip* and the model is trained with the `.npy` + the **human category**. **Every clip is bounded to one single action of one category** (no mixed actions inside a clip; no per-frame labels).
- **Robbery (cat. 6) coverage — the owner considers all variants covered:** hiding the product in front / behind, in one pocket / the other, under the sweatshirt, upper and lower pockets, robberies standing / sideways / **kneeling**. **Putting a mobile phone away was also labeled as robbery (cat. 6)** because it is an action that *could* be a robbery. Consequence for you to analyze (Q56): the positive class is "**concealing any object on the body**", which is semantically broader than "stealing"; a customer legitimately pocketing their own phone/wallet looks identical in pose.
- **Dataset used in the October campaign** (from `preflight.log`, two roots `data_yolo26s_010826` + `data_yolo26s_061026`; **19 043 clips, 11 categories** after the class map `drop_below_50`; categories 8/10/11/12 have < 50 clips and are excluded):

| cat | meaning | train | val | test | total |
|---|---|---|---|---|---|
| 0 | walks through without stopping | 2 951 | 491 | 493 | 3 935 |
| 1 | walks, stops, looks, does not buy | 433 | 72 | 73 | 578 |
| 2 | picks and puts back | 1 299 | 216 | 218 | 1 733 |
| 3 | picks and puts in basket/cart/bag (buys) | 1 324 | 220 | 222 | 1 766 |
| 4 | picks and carries in hand | 1 589 | 264 | 266 | 2 119 |
| 5 | walks carrying objects | 3 618 | 603 | 603 | 4 824 |
| **6** | **robbery (positive)** | **1 944** | **324** | **324** | **2 592 (13.6 %)** |
| 7 | employee restocking | 343 | 57 | 58 | 458 |
| 9 | looks without buying, carrying objects | 570 | 95 | 95 | 760 |
| 13 | employee cleaning | 44 | 7 | 8 | 59 |
| 14 | puts an object in own bag | 164 | 27 | 28 | 219 |
| | **total** | **14 279** | **2 376** | **2 388** | **19 043** |

  Split: stratified by folder category, 75/12.5/12.5, seed 42, **by clip UID (not by person/session/camera/day)**. Train rows after augmentation: 16 075 (14 279 real + 1 796 synthetic; the synthetic rows come from cats 1, 7, 13, 14 only).
  **Binary evaluation pool (cells `bin_*`, `single_user_only=true`):** pool of 15 432 clips; **val evaluated = 1 942 clips (323 robberies / 1 619 negatives = 16.6 % positives)**, although the split has 2 376 val clips → **434 val clips (18 %) are silently excluded from the binary evaluation** by `single_user_only`. The old-sample measurement said this filter drops ≈ 46 % of clips; here it is ≈ 19 %. *Unverified:* whether the binary **training** sets also exclude multi-user clips (the preflight reports 16 075 train rows for all cells) — please treat as finding **M10**.
- **A second table the owner supplied (preflight of a new extraction batch, YOLO26s-pose on an RTX 5090):** **4 088 clips**, 455 724 frames (≈ 111 frames/clip), per category: 0: 1 111 · 1: 106 · 2: 572 · 3: 643 · 4: 251 · 5: 1 193 · **6: 19** · 7: 41 · 9: 122 · 12: 1 · 13: 15 · 14: 14. **Only 19 robberies (0.46 %)**: this is **one particular batch** (confirmed by the owner; it matches the "October extension" of the brief: ≈ 3 650 negatives, +18 robberies), **not the whole dataset**. The owner states that the **total number of robberies is about 2 000** (the campaign preflight counts 2 592 clips of cat. 6; treat the true figure as ≈ 2 000–2 600 robbery clips, and note that clips ≠ independent events/sessions). Extraction speed measured: ≈ 0.028 s/frame (`.pt`) or ≈ 0.016 s/frame (`.engine`) of inference, FFmpeg decoding ≈ 204 min for the batch (decoding, not inference, is the bottleneck); ≈ 5–7 h per 4 000 clips, so **a re-extraction of the full dataset (e.g. with more keypoints) is feasible** (≈ 1–1.5 days).

**C.6.4 Current results (owner's comparison table, val, clip level).**

| | August campaign (best ops) | October campaign (best ops) |
|---|---|---|
| Ensemble | **11 \| 50 AND @ 0.68** (`bin_filtered_hardened`) | **41 \| 48 AND @ 0.80** (`bin_full`) |
| F1 | 76.1 % | 76.3 % |
| Recall | 63.6 % | 63.8 % |
| FP | 11 (0.69 %) | 11 (0.68 %) |
| FN | 117 | 117 |
| Precision | 94.9 % | 94.9 % |
| Best single @ 0.8 | #50: F1 76.7 %, FP 1.62 % | #49: F1 81.5 %, FP 1.98 % |

  Identical FN = 117 and FP = 11 with different models, data and campaigns (derived: the August val has ≈ 321 robberies / ≈ 1 594 negatives vs 323 / 1 619 in October) is **strong evidence that the FP–recall frontier did not move** (§C.4) — please interpret it. More artifact facts (run `campaign_20261006_231053`, val, `reports/_master`): 320 model rows, 41 040 ensemble-grid rows; **no single model reaches recall ≥ 60 % with FP ≤ 0.01 %** (0 configs); best single model by F1 under *argmax*: `mc_filtered_merge #56` F1 84.9 %, FP 4.24 %; binary single models at argmax: #49 recall 79.6 % / FP 2.78 %; #48 78.3 % / 3.03 %; #41 76.5 % / 3.71 %; at softmax ≥ 0.80: #49 75.5 % / 1.98 %; #48 73.4 % / 1.85 %; #41 70.6 % / 2.59 %; the best ensemble of each of the four binary cells ends at **10–11 FP and 60–64 % recall**. One binary cell took ≈ 24 900 s (≈ 6.9 h) of wall time for 20 models (the preflight had estimated ≈ 57 min: the estimate is ≈ 7× optimistic); the whole 8-cell campaign took ≈ 2 days.
  **Production reference:** ≈ 20 FP in 654 sessions in one day (3.1 %) with the model alone.

**C.6.5 Tracker behavior and what it implies for training.** In production, a person who is lost by the tracker and not correctly re-identified gets a **new ID → a new session**, and the new session is monitored exactly like any other; the robbery must be detected whether it happens in the first or the second fragment. The owner believes the **track ID itself is not important** as long as the skeleton is processed whenever possible (1 or 4 sessions per person because of occlusions/ID loss are all fine). The owner reports few FP caused by the tracker and defers that topic. **Implication:** the model must classify **arbitrary partial fragments** of a person's behavior (sequences that start mid-action, have gaps, are short, have no history) — not just clean, action-centered clips like the training clips (see H4, Q15, Q57).

**C.6.6 Freedom and constraints.**
- **No latency limit** of any kind. Models already run in Triton on an RTX 5090 "very fast". **Lighter is better**: if the winner is a very heavy Transformer and the runner-up is a very light architecture with similar numbers or a *reasonable* loss, the owner prefers the light one (so give a Pareto view: quality vs size/latency, and a rule for "similar").
- **You may propose anything:** more epochs, other architectures, other `seq_len`, other features, other normalization, other loss/sampling, other training scheme, additional data to collect.
- **Training time is not a constraint** (the last full campaign took ≈ 2 days; the owner will try to optimize it, but whatever is needed is acceptable).
- **Out of scope for now:** tracker/ID-switch FP, SAM3 and heuristic filters (to be applied later) — but you are asked for *ideas* for them (§16.9).

**C.6.7 Which questions of §16 these facts already answer (use them, do not re-ask):**

| Q | Owner-provided answer |
|---|---|
| Q1 | ≈ 1 FP/day (≈ 1 000 sessions/day) is the target; "0 FP/day" preferred if recall must be traded (C.6.2) |
| Q2 | **Event-level** (one alarm during the event is enough) |
| Q3 | FP ≫ FN in cost (C.6.2); reviewer model not specified → assume a human or SAM3 verifier downstream |
| Q4 | Yes: Stage-2 = heuristic filters + SAM3 (+ HAR, C3) are planned; but Stage-1 should still be as good as possible alone |
| Q5 | No latency limit |
| Q8 | Every clip is bounded to **one action** of one category; no frame-level labels inside the clip |
| Q9 | All robbery variants considered covered, including kneeling; pocketing a phone counts as robbery |
| Q12 | Production alarms exist and can be reviewed (they were used to measure ≈ 20 FP/654 sessions); the amount of reviewed/unlabeled production data is **unknown** |
| Q15 | ID-switch fragments are monitored as independent sessions; ID is not important (C.6.5) |
| Q6/Q7 | Single store, 9 cameras; staged ≈ 63 %, real cuts ≈ 34 %, production ≈ 3 % (brief); number of distinct people/sessions **unknown** |

---

## 0. Executive summary

The pipeline is coherent and well instrumented (UID-level splits, augmentation only in train, split manifests, threshold sweeps). But there are **5 high-severity problems** that probably explain much of the gap between validation metrics and production behavior (false positives), plus several medium problems in preprocessing, architecture and decision rules.

| ID | Sev. | Finding (one line) | Evidence |
|----|------|--------------------|----------|
| **H1** | **High** | The "best checkpoint by `val_acc`" **is not the best**: `best_state = model.state_dict()` keeps a live reference; at the end the **last-epoch** weights are loaded. | Code (`train_model_operations.py:3926-3931`) + **reproduced** with a minimal test |
| **H2** | **High** | **Session/source ⇄ label confounding** and random per-clip split → leakage between train/val/test. In the local sample, 76 % of the clips of the "21Diciembre" session are robberies while other days have ≈0 robberies; 69 % of clips overlap in time with another clip of the same video. | **Measured** from `meta.json` (§6) |
| **H3** | **High** | Model/threshold/ensemble selection on **val** with a huge grid and a target `FP rate ≤ 0.01 %` that is **not measurable** with ~1 000–1 600 negatives; `--split test` **re-optimizes** thresholds on test; val prior (~17–22 % robberies) ≠ production (~0.1–0.5 %). | Code `evaluate_campaign.py` + figures from `SLIDING_WINDOW_EVAL_INTERPRETACION_*.md` |
| **H4** | **High** | **Temporal handling**: `temporal_resize` warps time depending on clip duration (subsampling up to ×8; last-frame padding up to 24–44 % of frames) with no mask or length channel; training on ~10 s clips vs 3 s windows in sliding-window evaluation. | **Measured** (§7.3) |
| **H5** | **High/Med** | Two of the six augmentations (**shift** and **scale**) are **mathematically null** after `normalize_sequence`; the campaign **does not enable** random on-the-fly augmentation (only deterministic variants with p=0.5 + per-category expansion). | Code (§8) |
| M1 | Med | Architectures: BiLSTM/BiGRU use `out[:, -1]` (the backward half sees only the last frame); Transformer/Conformer/JointAttn have no positional encoding; TCN has no normalization and a tiny receptive field; ST-GCN graph is non-anatomical. | Code (§9) |
| M2 | Med | Features: no standardization (\|pos\| ≈ 11× \|vel\|), keypoint confidence unused, only raw (x, y, dx, dy); no angles or wrist–hip/wrist–torso distances. | Measured + code |
| M3 | Med | Spatial normalization: **per-frame** centroid of 8 joints (depends on the arms), **a single scale** per clip (median intra-clip CV 0.20), anisotropic coordinates (x/W, y/H). | Measured |
| M4 | Med | Quality filters **disabled by default**; `passes_filters` in `meta.json` ignored; `single_user_only=True` drops ~46 % of clips (33 % of robberies). **Full proposed filter specification in §4.2.** | Code + measured |
| M5 | Med | Minimal optimization: Adam at constant LR, no scheduler, no early stopping, no gradient clipping, no weight decay, one global seed. | Code |
| M6 | Med | Stacked loss/sampling techniques (asymmetric loss + hard-negative mining + UID boost + expansion) → mis-calibrated probabilities; `target_neg_pos_ratio` is **inert** in the campaign. | Code |
| M7 | Med | Class maps `drop_below_50` remove rare categories (e.g. 7, 13 = employees) from the negatives in binary cells. | Config |
| M8 | Med | **Quality/camera shortcuts**: robbery clips are larger, sharper and less noisy than negatives (AUC: bbox_aspect 0.78, jitter 0.73, motion 0.67, confidence 0.66, scale 0.63). | Measured (§4.2) |
| M9 | Med | **Decision rule/calibration**: the logit-margin sweep stops at margin 2.0 (≡ p = 0.88) while the softmax sweep goes to 0.98; temperature scaling is fitted and the threshold chosen by F1 on the **same** val set; thresholds for a 0.1 % FP budget live in the saturated tail of the softmax. | Code (§11.4) |
| M10 | Med (to verify) | **Evaluation pool ≠ split**: val has 2 376 clips but the binary evaluation uses 1 942 (`single_user_only=true`, 434 clips = 18 % dropped silently); the preflight reports the same 16 075 train rows for multiclass and binary cells, so it is **unclear whether binary training also drops multi-user clips** and whether train/val/test see the same population as production (where multi-person scenes exist). | Artifacts (C.6.3): `val_eval_summary.json`, `preflight.log` |

**Recommended order of action:** H1 (one line, no risk) → H2/H3 (re-evaluate with session-grouped split and frozen thresholds) → H4 (fixed temporal window) → H5/M3/M2 (preprocessing) → M9 (decision rule) → M1 (architecture) → the rest.

> **Important:** any architecture or hyper-parameter improvement evaluated with the current protocol (H2+H3) **is not reliable**. Fix the evaluation protocol first.

---

## 1. Scope, sources and limitations

### 1.1 What was reviewed
- Code: `train_model_operations.py` (4 534 lines, almost fully read), `model_config.py`, `class_map_utils.py`, `campaign/train_campaign.py`, `campaign/evaluate_campaign.py`, `campaign/calibration_utils.py`, `campaign/campaign_config.json`, `config_category_augmentation.json`, `class_maps/*.json`.
- Docs: `campaign/KANVIS_CONTEXT_AND_RESEARCH_BRIEF.md`, `KANVIS_EXPERT_RESPONSE_20261014.md`, `SLIDING_WINDOW_EVAL_INTERPRETACION_campaign_20260714_164642.md`, `RECOMMENDED_ENSEMBLE_BIN_FULL_HARDENED.md`, `INFERENCE_ENSEMBLE_HANDOFF.md`.
- **Real data on disk** (used only for measurements):
  - `/home/debian/Proyectos/GuardIA/ResultadosExperimentos/data_result` — **10 790 clips, 5 851 single-user** (old format: `poses.npy` + `poses_full.npy` + `meta.json`, **no `valid_mask.npy`**).
  - `/home/debian/Descargas/extracción_piloto/data_result` — 27 pilot clips (new format: with `valid_mask.npy`, `yolo_model`, `user_cat`).

### 1.2 Limitations (what could NOT be verified)
- The real training dataset configured in `config.py` (`/media/8TB/DatosEntrenamiento/data_clips/data_yolo26s_061026`) **is not mounted** on this machine. Distribution figures are from the **old local sample** and must be **re-measured on the real dataset** (commands/criteria in Annex A).
- No training was run and no campaign metric was reproduced; F1/FP figures quoted come from repo documents.
- Production inference code (C2) was not audited, only the parity documented in `INFERENCE_ENSEMBLE_HANDOFF.md`.

---

## 2. Training pipeline map

```
data_result/<cat>/<clip>/meta.json
                     /user_<track_id>/poses_full.npy   [T,8,2]  (xyn, NaN where conf is low)
                                     /poses.npy        [T',8,2] ("filtered" subset)
                                     /valid_mask.npy   [T]  bool (new format only)
        │
        ▼  collect_examples()  (L1455, L1275)   quality filters (disabled), single_user_only, label
PoseExample(pose_path, label, valid_mask_path, ...)
        │
        ▼  apply_class_map_spec (class_maps/*.json)  → exclude/remap
        ▼  make_binary_examples (robbery=1, rest=0)  (binary cells)
        ▼  split_examples_stratified_by_uid  (L2606) → train/val/test, seed 42, by folder category
        ▼  (train only) category / mass / manifest expansion → rows = deterministic variants of the same clip
        ▼  PoseDataset.__getitem__ (L1639)
              np.load → [valid_mask] → nan→0 → (aug) → normalize_sequence → add_velocity → temporal_resize(seq_len) → reshape [T, 8*4=32]
        ▼  DataLoader (shuffle | WeightedRandomSampler with hard-negative UID boost)
        ▼  train_one_epoch_advanced (L2854): Adam, CE/focal/asymmetric, hard-neg mining, MixStyle, SSL
        ▼  evaluate_with_metrics (L2986) every epoch; "best" by val_acc  (BUG H1)
        ▼  test at the end + torch.save(modelo_XX.pt)
        ▼  campaign/evaluate_campaign.py: threshold sweep + ensemble grid (AND/MEAN/CASCADE) on val
```

---

## 3. Files consumed (`.npy` / JSON)

| File | Used? | Where / how |
|---|---|---|
| `poses_full.npy` `[T,8,2]` | **Yes** if `pose_source="full"` (`*_full*` cells, including the winning `bin_full_hardened`) | `PoseDataset.__getitem__` |
| `poses.npy` `[T',8,2]` | **Yes** if `pose_source="filtered"` (`*_filtered*` cells; default value of the 60 `EXPERIMENTS`, but the cell overrides it) | same |
| `valid_mask.npy` `[T]` | **Yes, only with `pose_source="full"` and if it exists** (assigned at L1513-1525, applied at L1711-1713: `poses = poses[valid_mask]` → **drops frames**, leaving temporal gaps). Not applied with `filtered` (correct: `poses.npy` is shorter than the mask). | `PoseDataset` |
| `keypoint_mask.npy` | **No** | — |
| `meta.json` | **Partially**: `cat` (folder label), `users[]` (user count, `valid_frames`, `valid_pct`, `occlusion_ratio`, `total_frames`), `clip_duration`, `user_cat` (per-user label, takes priority over `cat`). **`passes_filters` is explicitly ignored** (`_user_quality_ok`, L1275). **`keypoint_confidence_avg`, `bbox_aspect_ratio`, `subject_velocity`, `clothes`, `fps`, `video_source`, `t_start/t_end` are not used.** | `collect_examples` |
| `clip.mp4` | **No** (only for error export) | `export_error_paths` |

**Observation (old local format):** `poses_full.npy` has 0 % NaN and there is no `valid_mask.npy`; the "full" model therefore receives all frames without a mask (including low-confidence ones, no longer distinguishable). **Observation (new format, pilot):** `poses_full.npy` has 0–1.1 % NaN and `valid_mask` drops 0–1.3 % of frames; after applying the mask **0 NaN remain** in the 27 pilot clips → `nan_to_num(…, 0)` is in practice inert *if* `valid_mask` is present (see §7.1 for the risk when it is not).

---

## 4. Filters and example selection

Implementation: `_user_quality_ok` (L1275) and `_collect_examples_from_root` (L1455). Defaults in `model_config.py`:

| Filter | Default | Real effect |
|---|---|---|
| `MIN_CLIP_SECONDS` | 0.0 | disabled |
| `MIN_VALID_FRAMES` | 0 | disabled |
| `MIN_VALID_PCT` | 0.0 | disabled |
| `MAX_OCCLUSION_RATIO` | 90.0 | almost disabled |
| `MIN_SEQ_LEN` | 4 (frames after mask) | only drops degenerate clips |
| `passes_filters` (meta) | ignored | — |
| `single_user_only` | **True** in `campaign_config.json` | only clips with exactly 1 user |

Consequences measured in the local sample (single-user clips: 5 851 of 10 790):
- `single_user_only=True` drops **46 %** of clips and **33 %** of robberies (296 of 444 kept). Clips with ≥ 2 users are more "realistic" (production has multi-person scenes); training is on an easier distribution than deployment. With `single_user_only=False` the risk is the opposite: **all users of a clip inherit the clip's `cat` label** (label noise: in a robbery only one is the thief) unless `user_cat` exists.
- `passes_filters=True` for only ~12 % of users (2 326 / 18 992) → average quality is low and **there is no filter**: the model trains on users with `valid_pct` as low as 11.9 % (real example in an inspected `meta.json`).
- **Duration outliers**: `clip_duration` up to 36 012 s; frames per user up to 3 048 (cat. 7). Without a cap, a 3 000-frame clip is subsampled ×30 down to `seq_len` (see H4).
- 437 clips have 0 users (ignored).

### 4.2 Full proposed filter specification (minimum frames, required keypoints, geometry, etc.)

Principles:
1. **Quarantine, do not delete**: every discarded clip is logged with its reason (`reject_reason`) in a CSV (`filter_report.csv`) and counts are reported **per filter and per class** (especially robberies). A discarded robbery is a potential FN that must be reviewed by hand.
2. **The same quality gate must exist at inference** (*quality gate*): a track that does not meet the minimums **raises no alarm** (or returns "no decision"). It is the cheapest lever to reduce FP on broken tracks and avoids training on a quality that is later decided with a different one.
3. Thresholds below are based on the distributions measured in the local sample (single-user clips, N = 5 851, 296 robberies). Those marked **(calibrate)** cannot be measured with the old format (no `valid_mask`/per-keypoint NaN) and must be calibrated on the new format on the real dataset.
4. **Always evaluate twice**: (a) test *with* the quality gate (model quality) and (b) test on **all tracks without the gate** (end-to-end system, including FP from bad tracks).

**Track/clip-level filters (A) — applied in train, val, test and inference:**

| ID | Proposed rule | What it protects | Measured retention (neg / robbery) |
|---|---|---|---|
| A1 | **Minimum valid frames** `T_valid ≥ 36` (3 s @ 12 fps; absolute minimum 24) after applying `valid_mask` | Sequences too short to see the action | `T_full ≥ 24`: 100/100 % · `≥ 36`: 95.9/99.7 % · `≥ 48`: 91.1/99.7 % · `≥ 64`: 81.4/97.3 % |
| A2 | **Minimum frames in `poses.npy` ("filtered")** `T_filt ≥ 24` (if that source is used) | Filtered source with gaps | `≥ 24`: 90.7/98.6 % · `≥ 36`: 84.6/92.9 % (this one **does** lose robberies) |
| A3 | **Fraction of valid frames** `valid_frames/total_frames ≥ 0.50` (new format: `valid_mask.sum()/T`) | Intermittent detection | `≥ 40 %`: 87.0/91.9 · `≥ 50 %`: 82.2/84.5 · `≥ 70 %`: 65.4/64.5 (aggressive) |
| A4 | **Length cap** `T ≤ 600` frames (~50 s): **crop into windows** instead of discarding; clips of thousands of frames (max 3 048; `clip_duration` up to 36 012 s) | Outliers forcing ×30 subsampling | `T ≤ 600`: 98.5/100 % |
| A5 | **Mean keypoint confidence** `keypoint_confidence_avg ≥ 0.70` | Noisy tracks | `≥ 0.70`: 95.9/99.3 % · `≥ 0.80`: 88.0/96.3 % |
| A6 | **Minimum body size** `scale ≥ 0.03` in image coords (≈ mean joint–centroid distance); express in **pixels** (calibrate the px threshold per camera resolution) | Person too far, huge relative noise | `≥ 0.03`: 88.3/98.3 % · `≥ 0.04`: 72.4/85.8 % (removes many negatives and robberies) |
| A7 | **Occlusion** `occlusion_ratio ≤ 30 %` | Heavily occluded tracks | `≤ 30`: 93.7/98.0 % |
| A8 | **Person cut by the border**: fraction of (frame, joint) with coordinate ≤ 0.001 or ≥ 0.999 `≤ 0.10` | Person entering/leaving the camera | `≤ 0.10`: 97.1/99.7 % |
| A9 | **Minimum motion** (mean \|Δ normalized position\| `≥ 0.02`) and 0 % consecutive identical frames | Frozen tracks / static detections (false skeletons on mannequins, posters) | `≥ 0.02`: 98.3/100 % |
| A10 | **Number of users**: exactly 1 track per clip (training); at inference each track is evaluated separately | Ambiguous label in multi-user clips | drops 46 % of clips |
| A11 | **Geometric coherence** (calibrate): in ≥ 90 % of frames `y_hip > y_shoulder` (upright person), shoulder width > 0 and torso length > 0 | Inverted/degenerate skeletons | — |
| A12 | **Identity jumps** (calibrate): centroid displacement between consecutive frames > 1.5 × body scale → split the clip into segments or discard | *ID switches* / two people fused in one track | — |
| A13 | **Duplicates**: remove/group clips of the same track with > 50 % temporal overlap within the same video (or assign them to the same split group) | Leakage and scene over-weighting | — |

**Frame- and keypoint-level filters (B) — define `valid_mask` and what is masked (calibrate on the new format):**

| ID | Proposed rule | Reason |
|---|---|---|
| B1 | **Visible keypoint** = `conf > 0.5` (the extractor already sets NaN if `conf ≤ 0.5`; keep a single threshold between train and inference) | Consistency |
| B2 | **Usable frame** = ≥ 6 of 8 keypoints visible **and** both shoulders **and** at least one hip visible (torso anchor for normalization) | Stable torso-based normalization |
| B3 | **Wrists** (indices 4 and 5, the ones that perform the action): at least one wrist visible in ≥ 60 % of frames and both in ≥ 30 % | Robbery is a hand action; without wrists the label is not recoverable from pose |
| B4 | **Gaps**: linearly interpolate gaps ≤ 3 frames (0.25 s); gaps of 4–6 frames → mask (do not interpolate); gaps > 6 frames (0.5 s) → **cut** the clip into independent segments | Avoids velocity jumps and false trajectories |
| B5 | **No residual NaN**: if any NaN remains after B4, the clip is discarded or explicitly masked (no `nan→0`); log in `filter_report.csv` | Avoid spurious (0,0) |
| B6 | **Effective length after cuts** ≥ A1 in each segment | Consistency with A1 |

**Label filters (C):**

| ID | Proposed rule | Reason |
|---|---|---|
| C1 | **Robbery "not recoverable from pose"**: category-6 clips that violate B3 or A5 go to a separate bucket `positive_unrecoverable` (not used for training nor for measuring model recall, but **counted** in end-to-end recall) | Avoid forcing the model to learn from noise and keep recall honest |
| C2 | **Manual review** of: (a) robberies with `T < 36` or `scale < 0.03`, (b) negatives of confusable categories (14, 3, 2) with low `conf`, (c) clips where `meta.cat` ≠ `user_cat` | Label noise |
| C3 | If `single_user_only=False`: use `user_cat`; if missing, **do not** propagate the clip's `cat` to all users | Label noise |

**Measured aggregate effect** of A1 (`T ≥ 36`) + A3 (`valid_pct ≥ 50`) + A5 (`conf ≥ 0.70`) + A6 (`scale ≥ 0.03`) + A8 (`border ≤ 0.10`) + A4 (`T ≤ 600`): **70.7 % of negatives and 83.1 % of robberies** are kept (246 of 296). 50 robberies are lost: review them (C1/C2) before deciding. The filter changes the class prior (from 5.1 % to 5.9 % robberies in the sample); look at the effect both in training **and** in the inference gate.

**Shortcuts found while measuring filters (quality/camera variables that separate robbery from the rest; robbery-vs-rest AUC, 0.5 = nothing):** `bbox_aspect_ratio` 0.775, `body scale` 0.626 (robberies larger: median 0.063 vs 0.039), `confidence` 0.664, **2nd-derivative jitter ≈ 0.73 inverted** (robberies are smoother: AUC 0.269), **mean motion** 0.67 inverted (AUC 0.331), wrists 0.63 inverted (0.37), `border` 0.69 inverted (0.308). I.e. robbery clips are **larger, sharper and less noisy** than negatives → the model can learn "quality/camera of the robbery session". Filters A5–A9 help, but these variables must also be **equalized/randomized** across classes (class-wise noise/scale augmentation, stratified sampling on these variables) and evaluated with a session split (H2).

**How to implement:** a single function `evaluate_quality(track) -> {ok: bool, reasons: [...], metrics: {...}}` in the shared module (`pose_core`), used by (i) `collect_examples`, (ii) the training-window generator and (iii) C2 in production; with a `quality_version` stored in the checkpoint. It replaces the current `_user_quality_ok` (which today filters almost nothing). Always report `filter_report.csv`.

---

## 5. Labels and class maps

- **Label** = folder category (`cat`) or `user_cat` if present. 15 categories (0–14); 6 = robbery (positive class). Categories without clips in the campaign dataset: 8, 10, 11, 12 (brief figures; the local sample has 15/3/0/6 clips in 8/10/11/12).
- **Binary**: `make_binary_examples` (L2749) → 1 if `cat==6` else 0. **Multiclass**: `build_label_mapping` (L2744) over the categories present after the class map.
- **Class maps** (`campaign/class_maps/`): `full` (unchanged), `drop_below_50` (excludes categories with < 50 clips), `merge_rare`, `drop_rare`, `verifier_confusable`.
  - **Risk M7:** in **binary** cells `drop_below_50` *removes* those categories from the negatives. Among them are **7 (employee restocking), 13 (employee cleaning)** and 14 (put object in bag, 219 clips in the brief): exactly the negatives production sees and confuses with robbery. `bin_full_hardened` uses `class_map_id: full` (good), but `bin_filtered_hardened` uses `drop_below_50`.
  - Class 14 ("put object in bag") is visually almost identical to robbery (6): a structural semantic confusion; see §11 and the questions in §16.

---

## 6. Splits (train / val / test) — **H2**

Implementation: `split_examples_stratified_by_uid` (L2606): stratified **by folder category**, seed 42, UID = relative path of the `.npy` inside `data_result`. Ratios from `suggest_split_ratios(N)` (`model_config.py`). Val size of the July reference campaign: 1 241 clips (≈ 281 robberies / ≈ 960 negatives); Oct campaign: 1 619 negatives / 323 robberies.

What is good: `assert_no_uid_leak_between_splits` exists, augmentation/expansion is applied only to the train split (after splitting), and manifests are saved (`split_manifest_exp_XX.json`, `training_plan.json`).

**Problem:** the UID is **the clip**, not the person, session or source video. Two consecutive clips of the same person/camera/scene can end up in train and test. Measured on the local sample:

| Measure | Value |
|---|---|
| Clips that **overlap in time** with another clip of the same source video | **68.9 %** (7 432 / 10 790) |
| Clips with another clip within ≤ 10 s | 83.8 % |
| Multi-user clips (if `single_user_only=False`, users of one clip may fall in different splits) | 41.7 % of clips with ≥ 2 users |

**Session ⇄ label confounding (local sample, single-user clips only):**

| Origin (prefix of `clip_name`) | Negatives | Robberies | % robbery |
|---|---:|---:|---:|
| 12/13/15/17/18/19/20 Dec 2025 | 5 492 | 1 | **≈ 0 %** |
| **21Diciembre2025** | 63 | **199** | **76 %** |
| `clips_output_*` | 0 | **96** | **100 %** |

(Per day, negatives are 1 829 (Dec 20) / 1 086 (17) / 1 055 (15) / 562 (19) / 530 (12) / 334 (18) / 96 (13) and robberies 0/0/1/0/0/0/0. 99.7 % of the robberies in the local sample come from **two origins** — one session on the 21st and `clips_output` — probably staged recordings.) At source-video level, robberies concentrate in 4 videos (`cervezas`, `confituras`, `legumbres`, `conservas`; 31–42 % robbery) while 28 videos from other takes (~1 700 clips) have 0 robberies.

Additional shortcut indicator: using only "nuisance" metadata of a single-user clip, the robbery-vs-rest AUC is **0.775 for `bbox_aspect_ratio`** and 0.664 for `keypoint_confidence_avg` (0.5 = no information). The model does not see the video, but it does see the **pose geometry** (which depends on camera/perspective/person) and the keypoint noise.

**Implications:**
1. Val/test metrics are **inflated** by session adjacency and by separating "robbery session" from "the rest".
2. This is consistent with the production symptom (many FP): the model has partly learned "this looks like session X".
3. **Verify whether the real dataset (`data_yolo26s_061026`) has the same structure** (Annex A).

**Recommendations:**
- **Grouped** split (GroupKFold / `StratifiedGroupKFold`) with group = (date, source video, temporal block ≥ 60 s) or, better, session/person identity if it can be inferred (`clothes` in `meta.json`, `video_source`, `t_start`).
- Report **leave-one-session-out** (or by day/camera) and the **dispersion** between folds.
- Test **only with never-seen sessions/days** and **fixed** (do not regenerate when ratios change).
- Add real production negatives (unlabeled normal operation) to the test set to measure FP/day for real.
- For multi-user: group by clip (all users of a clip in the same split).

---

## 7. Sequence preprocessing

### 7.1 NaN, mask and operation order (`PoseDataset.__getitem__`, L1707-1780)
Order: `np.load` → `poses[valid_mask]` (if present) → `nan_to_num(nan→0)` → augmentation (image coords) → `normalize_sequence` → `add_velocity` → `temporal_resize(seq_len)` → reshape `[T, J*4]`.

- With `valid_mask` present no NaN remain (measured on pilot) → harmless. Also `_collect_examples_from_root` (L1512-1529) **discards the clip** if `pose_source="full"`, there is no `valid_mask.npy` and `poses_full` contains NaN (correct safeguard). `len(valid_mask)` must equal `T` or the clip is silently skipped.
- **Latent risk (low today):** if a NaN joint ever reached `normalize_sequence` converted to (0,0), it would shift the centroid ≈ 0.07 and multiply the scale (the body is ≈ 0.054 in image coords) → huge outlier frames. Recommendation: an explicit **NaN-aware** policy (interpolate short gaps, mask long ones), count and report clips dropped by this safeguard (silent data loss), and make `PoseDataset` fail loudly if NaN remain after the mask.
- Dropping frames with `valid_mask` leaves **temporal jumps** (the "diff" velocity crosses the gap). With `poses.npy` ("filtered") the effect is larger: only ~60–90 % of frames are kept (ratio of medians T_filtered/T_full ≈ 0.58–0.87 by class in the local sample), so `add_velocity` mixes heterogeneous `dt`.

### 7.2 `normalize_sequence` (L1600) — M3
```python
center = poses.mean(axis=1, keepdims=True)      # PER-FRAME centroid of the 8 joints
poses  = poses - center
scale  = np.linalg.norm(poses, axis=-1).mean()   # ONE scalar for the whole clip
poses  = poses / scale
```
Measured (1 500 single-user clips, `poses_full`):
- Scale (mean distance to centroid in image coords): median **0.054** (p5 0.026, p95 0.093). The person occupies ~5 % of the image width → keypoint noise is large relative to the scale.
- **Intra-clip scale variation** (CV of the per-frame scale): median **0.196**, p90 **0.47** → a single scalar does not correct perspective/approach changes or posture (extended arms).
- Conceptual problems:
  1. Per-frame centroid of 8 joints (shoulders, elbows, wrists, hips) **moves with the arms** → the "origin" changes when the person extends a hand, contaminating the very signal to be detected (hand→pocket/bag). Better to anchor at the **hip or shoulder midpoint** (torso) and scale by **torso length** (shoulders↔hips).
  2. All global position/trajectory is removed (valid for invariance), but **location context** is also lost (e.g. hand near bag/shelf) — a design decision to validate with an ablation (with/without global position).
  3. **Anisotropic** coordinates: `xyn` normalizes x by width and y by height (e.g. 1920×1080) → horizontal distances compressed ×0.5625 relative to vertical ones and **different per camera** if resolutions differ. Rotations/mirroring in that space introduce shear. Recommendation: move to **pixels** (or an isotropic unit) before normalizing, or normalize both axes by image height.
  4. No visibility/confidence channel.

### 7.3 Velocity and temporal resizing — **H4**
```python
vel = np.diff(poses, axis=0, prepend=poses[0:1])      # on the already-normalized sequence, original T
...
temporal_resize(seq, target_len):  T>target → linspace indices (subsampling without averaging);
                                   T<target → padding by repeating the last frame
```
Measured (local sample, `poses_full`):

| seq_len | % clips with padding | padding frames (mean) | T/seq_len: p10 / median / p90 | % clips with T/seq_len > 2 |
|---:|---:|---:|---|---:|
| 64 | 19 % | 5.0 % | 0.78 / 1.62 / 4.03 | 35 % |
| 96 | 44 % | 13.7 % | 0.52 / 1.08 / 2.69 | 17 % |
| 128 | 65 % | 24.2 % | — | — |

Problems:
1. **Duration-dependent time warping**: a 4 s clip and a 30 s clip both end up with 64/96 steps; the "apparent speed" of position varies up to ×8, but the `vel` channel keeps the per-original-frame speed → the two channels are no longer coherent (Δposition between steps ≠ velocity).
2. **Padding with no mask or length channel**: the last frame and its velocity are repeated; global pooling layers (`mean` in TCN, Conformer…) average the padding; LSTM/GRU end by processing a constant tail. With `seq_len=96` almost half of the clips have padding.
3. **Length shortcut**: clip duration differs by class (median T: cat6=117, cat0=95, cat1=165, cat7=227; p90 of cat6=177 vs 1 005 for cat7). The padding/subsampling pattern makes the **original length inferable** by the model. If length depends on the annotation process (clips cut by hand around the action) and not on the action itself, it is a shortcut that **will not exist in production** (fixed windows).
4. **Mismatch with evaluation/production**: training uses whole clips with median ~10 s, while `sliding_window_eval.py` evaluates **3.0 s windows (36 frames @ 12 fps), stride 1 s** (`SLIDING_WINDOW_EVAL_INTERPRETACION_*.md`). Those 36 frames are **stretched** to 64/96 steps (the action looks slower than in training) or padded, depending on the model.
5. The first frame has velocity 0 (`prepend=poses[0:1]`) and rounded-index subsampling can duplicate/skip frames.

**Recommendations:**
- Train with **fixed-duration windows in seconds** (e.g. 3 s and 5 s, ≈ 36/60 frames @ 12 fps), extracted with overlap from each clip (temporal multi-crop), with the clip label or, better, the action-zone label; apply the **same crop** at inference. This removes the dependency on `temporal_resize`.
- If resizing is kept: resample **before** `add_velocity` (linear interpolation, not `round`), compute velocity **after** with consistent `dt`, and add a real **mask/length channel** (or `pack_padded_sequence` / attention mask).
- Report performance **by duration bucket** (T<64, 64–128, >128).

---

## 8. Augmentation — **H5**

There are **three layers** of augmentation in train (val/test are never augmented — correct):

1. **Category / mass / manifest expansion** (`expand_examples_with_category_augmentation` L625, `mass_augment`): creates extra rows of the **same clip** with deterministic `forced_ops` (static across epochs). Current `config_category_augmentation.json`: `enabled:true`, `default:0`, `categories:{}` → the effective expansion is defined by the per-cell preflight (target negative:robbery ratio 4.0 baseline, 6.0 `fp_hardened`, `config_mass_augmentation.json` in the "mass" cells).
2. **Deterministic grid variants** (`build_deterministic_variant_specs` L115; up to 64 variants of the `industrial` profile of `operations_npy/validate_npy.json`): with probability `TRAIN_DETERMINISTIC_PROB=0.5` a random variant is applied at each access.
3. **Random on-the-fly augmentation** (`_augment_poses_on_the_fly`, `AUGMENT_PROB=0.65`, max 2 ops): **disabled in the campaign** (`train_campaign.py` does not pass `augment_on_the_fly`, whose default is `False`).

Operations (L1334-1405) and real effect after preprocessing:

| Operation | Range | Effect after `normalize_sequence`? |
|---|---|---|
| `mirror` (x→1−x + L/R swap: pairs (0,1),(2,3),(4,5),(6,7)) | — | **Yes, useful** and correct (L/R swap properly implemented). Flips handedness: reasonable. |
| `rotate` (around (0.5,0.5), **anisotropic** coords) | ±15° | **Yes**, but in x/W,y/H space ⇒ pixel-space shear; ±15° is large for standing people (fixed camera plane). |
| `scale` (around (0.5,0.5)) | 95–111 % | **NULL** (isotropic scale is removed by scalar normalization) except for `clip(0,1)`. |
| `shift` | ±0.06 | **NULL** (per-frame centering removes translation) except for `clip(0,1)`. |
| `noise` (σ in image coords) | 0–0.006 | **Yes**: σ_max 0.006 vs body scale 0.054 ⇒ up to **~11 %** of the scale; applied **before** normalizing (independent per frame/joint, no temporal correlation). |
| `speed` | 0.85–1.15 | **Yes**, linear interpolation; narrow range (clips already vary ×8 in duration because of H4). |

Other problems:
- `np.clip(…, 0, 1)` after each operation clips joints near the border and deforms the pose (especially with `shift`).
- Category-expanded rows use fixed deterministic seeds per `(uid, forced_ops, idx)` → **they do not change between epochs** (effective dataset = fixed variants).
- Expansion **artificially increases the weight of the same clip** and, combined with the per-clip split (H2), overfits specific sessions/people.
- Augmentations that would help after normalization are missing: **keypoint drop** (mask wrist/elbow), per-frame **temporal jitter**, mild **anisotropic scaling** (simulate a different camera/angle), **random temporal crop** (multi-crop), **correlated noise**.

**Recommendations:** remove or replace `scale`/`shift` (or apply them *after* normalizing if a different normalization is used), enable real on-the-fly augmentation with a set of non-null ops, and measure each op with ablations (E5).

---

## 9. Architectures

Common input: `[B, T, 32]` (8 joints × (x, y, dx, dy), order `LS,RS,LE,RE,LW,RW,LH,RH`). `build_model` (L3599). **60 experiments** in `EXPERIMENTS` (`model_config.py`), 15 distinct architectures; the current campaign uses 20 ids:

`4,5,6,8` (TCN) · `10,11,12` (Dilated TCN) · `13` (ResTCN) · `25,26` (Transformer) · `37` (JointAttn) · `40,41` (TCN-LSTM) · `48,49,50` (TCN-GRU) · `53,54` (ConformerLite) · `56,57` (MS-TCN).

Stage-1 production ensemble models: **#41** (`tcn_lstm`, 256/256, `seq_len` 64, lr 3e-4, 40 ep, batch 32) and **#48** (`tcn_gru`, 128/128, `seq_len` 96, lr 5e-4, 45 ep, batch 64), AND@0.80 on `poses_full.npy`.

| Architecture (class, line) | Summary | Problems found |
|---|---|---|
| `tcn` (L1785) | 2× Conv1d k=5 + ReLU + Dropout → `AdaptiveAvgPool1d` → FC | Receptive field of only **9 frames** (~0.75 s), no normalization, global average pooling includes padding |
| `res_tcn` (L1807) | `in_proj` + residual blocks (2× Conv1d k=3) → avg pooling | No normalization; pooling includes padding |
| `dilated_tcn` (L1852) | Dilated TCN | Best of the TCNs (large RF); no normalization |
| `lstm` (L1898) | BiLSTM | `out[:, -1, :]` → the "backward" half has seen only the **last frame** (with padding, the repeated last real frame) |
| `transformer` (L1919) | Linear → CLS token → `TransformerEncoder` → FC on CLS | **No positional encoding** ⇒ the encoder is permutation-equivariant over T (it cannot tell temporal order except through feature values, including velocity); CLS also attends to padding |
| `stgcn` (L1947) | ST-GCN | Adjacency = **chain over joint index** (`eye` + neighbors i±1): links right-shoulder↔left-elbow, right-elbow↔left-wrist… **non-anatomical**; it should be shoulder-elbow-wrist per side + shoulders + hips + shoulder-hip |
| `pose_cnn2d` (L2012) | 2D CNN over (T×J) | BatchNorm with small batches; mixes the joint axis without semantics |
| `joint_attn` (L2049) | Per-joint + temporal attention | **No joint-identity embedding**; `x.mean(dim=1)` (L2113) **averages joints** before the temporal block → loses identity |
| `tcn_lstm` (L2125) | `in_proj` k=1 → N residual Conv1d k=3 blocks (**no normalization**) → BiLSTM → `out[:, -1]` → FC | Same BiLSTM issue: backward sees only the last frame; TCN RF = 2N+1 frames (**5 frames** in #41 and #48 with N=2) |
| `gru`, `gru_attn`, `tcn_gru` (L2178, 2208, 2249) | Bi-GRU variants | `gru`/`tcn_gru` use `out[:, -1]` (same problem); `gru_attn` aggregates with attention (better) |
| `conformer_lite` (L2299) | Conv + attention | `mean` over T includes padding; **no positional encoding**, relies on the convolutions |
| `ms_tcn`, `tcn_attn`, `gat_tcn` (L2368, 2425, 2470) | Multi-scale / attention / graph | `gat_tcn` with the same non-anatomical graph; no normalization |

Cross-cutting findings:
- **No architecture masks padding**; **none uses structural LayerNorm/BatchNorm** on the main path (only `pose_cnn2d` and `conformer_lite`).
- Model size vs data: TCN-LSTM 256/256 (#41) has on the order of 1–2 M parameters against ≈ 3–6 thousand single-user clips — over-parameterized relative to the effective number of **sessions** (hundreds, not thousands).
- The campaign rightly discarded weak CNN2D/GRU/MS-TCN by results, but **that comparison was made under H1, H2, H3** (checkpoint = last epoch, per-clip split, val-based selection) → the architecture ranking is **not conclusive**.
- Differences between the winning architectures are of the order of a few FP/FN over ~1 241 clips (e.g. 47 vs 36 FP) → within single-seed noise (there are no multiple seeds).

**Recommendations (ordered by cost/benefit):**
1. Fix BiRNN pooling: use `torch.cat([h_fwd_last, h_bwd_first])` (final state of each direction) or **masked** attention/mean/max pooling over T.
2. Add a length mask and exclude padding from pooling/attention.
3. Add **positional encoding** (sinusoidal or learned) to Transformer/Conformer; `LayerNorm`/`BatchNorm1d` in the TCN.
4. Rewrite the ST-GCN/GAT adjacency with the real 8-joint skeleton (shoulder–elbow–wrist ×2, shoulder–shoulder, hip–hip, shoulder–hip ×2).
5. `JointAttn`: joint embedding + aggregation that does not collapse identity (concat or attention-pool).
6. Before adding capacity, try a **simple baseline** (e.g. per-clip aggregate features + GBDT/LogReg) to learn how much real signal exists above the session shortcut.

---

## 10. Training (loop, loss, sampling, selection) — H1, M5, M6

### 10.1 H1 — the best checkpoint is not saved (high, one-line fix)
```python
# train_model_operations.py:3926-3931
if val_acc > best_val_acc:
    best_val_acc = val_acc
    best_state = model.state_dict()        # <-- reference, NOT a copy
...
if best_state is not None:
    model.load_state_dict(best_state)      # <-- loads the current weights (last epoch)
```
`state_dict()` returns tensors that **share memory** with the parameters; `Adam.step()` modifies them in place, so `best_state` ends up equal to the last-epoch weights. **Reproduced** in a minimal test (`best_state` changes and its `data_ptr` equals the parameter's). Consequences:
- The log prints "Best val_acc: X" but the saved and test-evaluated model is the **last-epoch** one.
- All checkpoints of previous campaigns (`modelo_XX.pt`) are **last-epoch**, with fixed epochs (30–70) and no early stopping → long runs probably overfit.
- Same bug in `train_model.py:1020-1023` (old version).

**Fix:** `best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}` (or `copy.deepcopy`). Better still: save the best by an **operational metric** (see 10.3) and retrain the 2 ensemble models.

### 10.2 Optimization (L3860-3862)
`Adam(lr=cfg.lr, weight_decay=cfg.get("weight_decay", 0.0))`, **constant LR**, fixed `epochs` per experiment (20–70), unweighted `CrossEntropyLoss` for validation, no scheduler, no early stopping, no gradient clipping, no EMA/SWA, no AMP. Fixed global seed (`SEED=42`, L1139-1141) set once at import; all experiments in one run share the RNG state and there are no per-seed repetitions ⇒ **training variance is not measured**.

Recommendations: `AdamW` (wd 1e-4–1e-2), cosine/OneCycle with warm-up, early stopping with patience on the operational metric, `clip_grad_norm_(1.0)`, ≥ 3 seeds per final candidate, and seed ensembling (cheap and stable).

### 10.3 Selection metric
Selection is by `val_acc` (accuracy) with ~77 % negatives → **accuracy is dominated by negatives** and insensitive to the real goal (recall at few FP). Use **PR-AUC / recall@fixed-FP / F-β** on session-grouped val.

### 10.4 Loss and sampling
- `loss_type`: `ce` (baseline), `asymmetric` (`fp_hardened` cells: γ−=4, γ+=1, binary only; multiclass falls back to focal), `focal` (γ=2).
- `hard_negative_mining`: ×1.5 weight on the top-20 % negatives per batch with highest `p_pos` (binary).
- **Hard-negative UID** boost via `WeightedRandomSampler` (weight ×3 by default) from the preflight manifest.
- `balanced` / `maintain_class_ratio` + `target_neg_pos_ratio`: **inert in the campaign**: `train_campaign.py` enables them only if `train_opts["maintain_class_ratio"]` is True and the `fp_hardened` profile only defines `target_neg_pos_ratio: 6.0` (without `maintain_class_ratio`) → the effective ratio is set only by the per-category expansion of the preflight.
- **MixStyle** (default 0): per-sample statistics on `[B,T,F]`; with per-clip normalized features it would destroy useful content; keep at 0.
- **SSL consistency** (weight 0 by default): noise + feature dropout; not validated.

Problems:
1. **Stacking** of asymmetric loss + hard-negative mining + boosted sampler + expansion: each technique shifts the prior and the probabilities. The thresholds `softmax ≥ 0.8` / `logit_margin ≥ 0` **have no probabilistic meaning** (calibration is broken); the campaign patches this with val threshold sweeps and `apply_temperature_calibration`.
2. Category expansion creates **copies of the same clip** (fixed augmentation) and the sampler draws them with replacement: the model sees the same pattern/session many times (H2).
3. No real **early stopping or per-epoch selection** (H1).

Recommendations: reduce to **one** rebalancing technique at a time (ablation), calibrate with **temperature/Platt on a separate calibration partition** (not the val set used to choose the threshold), and report ECE.

---

## 11. Evaluation and decision rules — H3, M9

### 11.1 What is measured
`evaluate_with_metrics` (L2986): confusion matrix, per-class P/R/F1, macro/weighted F1, top-3 and, in binary, three rules: `softmax_argmax`, `softmax_threshold` (0.8), `logit_margin` (0.0). `campaign/evaluate_campaign.py`: `softmax` sweep 0.50–0.98 (step 0.02), `logit_margin` sweep −0.5–2.0 (step 0.25) and an **ensemble grid** (`and`/`mean`/`cascade`) over pairs/triples of the 20 models, on **val** (`--split val`, default) or **test** (`--split test`).

### 11.2 Problems
1. **Selection with many comparisons on a small val**: ~20 models × ~25 thresholds + thousands of ensemble combinations over ≈ 1 000–1 600 negatives / ≈ 280–320 positives. The best candidate is the maximum of many noisy draws (selection bias / winner's curse). `pick_best_ensemble_row` chooses the **lowest `fp_rate` among those with `recall ≥ 60 %`**; with ~1 000 negatives one FP = 0.10 %, so the declared target `target_fp_rate_pct = 0.01 %` is **not even representable**.
2. **`--split test` re-optimizes on test**: `evaluate_cell` repeats `pick_best_sweep_row`/`pick_best_ensemble_row` with the predictions of the split it is given (L598, L624). **There is no path that applies the frozen val thresholds/ensemble to test.** If test is used this way, test metrics are optimistic too. (`load_best_ensemble_spec` exists, but it is consumed by `sliding_window_eval.py`, `export_ensemble_fp.py`, etc., not by `evaluate_cell` to freeze.)
3. **Prior shift**: val has ≈ 17–22 % robberies; production ≈ 0.1–0.5 % (config uses `interactions_per_day=1000`). **F1 and precision are functions of the prior**; the relevant metric is **FP/day** and precision at the production prior. Real campaign example: `11|12 mean` F1 87.8 % (FP=36 over ≈ 960 neg = 3.8 %) ⇒ at 1 000 interactions/day with ~1 robbery that would be ~37 false alarms/day.
4. **Whole-clip eval vs windowed inference**: the sliding-window document itself shows **the metric changes with the methodology** (11|40 AND@0.74: FP=6 whole-clip vs FP=42 in SW) and that **no predictor reaches FP ≤ 1**.
5. **No confidence intervals** and no number of seeds: with 281 positives the 95 % CI of recall is ±~4–5 pp; with ~960 negatives the CI of a 1–5 % FP rate is ±~0.7–1.4 pp.
6. **Multiclass**: P(robbery) is taken as `prob[:, 0]` in `collect_binary_predictions` for `num_classes != 2` (L127-129) — it assumes index 0 is the robbery class; correct only if `label_to_idx` orders it first. Verify for each class map.
7. **Confusable categories** (14 "put in bag", 3 "pick and put in basket", 2) should be evaluated **separately** — `folder_category` is in the records but there is no standard report of FP per category vs real prior.

### 11.3 Recommendations
- Freeze a **protocol**: (a) choose hyper-parameters and threshold on **grouped val**; (b) a **single** pass on test with frozen thresholds; (c) **session-level** bootstrap for CIs.
- Always report: **PR-AUC**, **recall @ target FP/day** (at production prior, e.g. 0.1 %), **FP per category**, and the curve at several priors.
- Adjust `target_fp_rate_pct` to something measurable and/or use **conformal / Clopper–Pearson upper bound** of the FP rate.
- Add real operation negatives to the test set to estimate FP/day for real.

### 11.4 Scores and decision rules as implemented (facts for question block 16.8)
- **Binary equivalences (exact):** for 2 classes, `argmax` ⇔ `softmax_p1 ≥ 0.5` ⇔ `margin = logit1 − logit0 ≥ 0`; and `softmax_p1 ≥ t` ⇔ `margin ≥ ln(t/(1−t))` (t = 0.80 ⇔ margin 1.386; t = 0.88 ⇔ 2.0; t = 0.98 ⇔ 3.89). So **for a single binary model, softmax-threshold and logit-margin are the same ranking and differ only in the grid of thresholds tried**.
- **The sweeps do not cover the same tail:** the margin sweep ends at 2.0 (p = 0.881) while the softmax sweep ends at 0.98 (margin 3.89). Operating points demanded by a ≈ 0.1 % FP budget are in the far tail: the margin sweep cannot reach them, and softmax resolution (step 0.02) is coarse exactly where probabilities saturate.
- **Saturation:** with overconfident networks, `p1` is ≈ 1.0 for many positives and negatives (float32 rounding yields ties at 1.0); the margin in logit space keeps the ranking, the softmax compresses it. Selection of thresholds in the tail should be done in logit space (or with a per-model quantile of negatives).
- **Production ensemble rule:** `score = min(p1, p2)`, alarm if both `p_robo ≥ 0.80` (C2 `C2_MOTION_ENSEMBLE_RULE=and`). `min` of probabilities ≡ AND of per-model thresholds; **mean of probabilities ≠ mean of logits** (different geometry; the logit mean is the log-linear/geometric pooling). Cascade uses fixed 0.40/0.55 (`cascade_low/high`).
- **Multiclass:** `p_robo` = softmax over all classes (mass shared with 14/3/4/2), or `prob[:, 0]` in evaluation (see 11.2.6); the alternative one-vs-rest score `logit_6 − logsumexp(other logits)` is not used.
- **Calibration:** `campaign/calibration_utils.py::fit_temperature` grid-searches T ∈ [0.5, 3.0] (51 points) minimizing NLL **on val**, then picks the threshold in 0.50–0.98 that **maximizes F1 on the same val** (not a FP budget, not a held-out set). A single global temperature does not change the ranking of one model, only the meaning of thresholds and the ensemble `min/mean`.
- **Training prior vs deployment prior:** the sampler, expansion and loss weights change the effective class prior; the logit offset for prior shift, `Δ = ln(π_prod/(1−π_prod)) − ln(π_train/(1−π_train))`, is never applied (with π_prod ≈ 0.1–0.5 % and π_train ≈ 15–30 % it is ≈ −4.5 to −6 nats).
- **Temporal aggregation:** per-window scores are consumed by N / N+N+1 windows with the same threshold; there is no accumulation of evidence across chunks (sum of log-likelihood ratios, k-of-n, CUSUM).

---

## 12. Training ↔ inference parity

The checkpoint stores `model_state_dict`, `label_to_idx`, `config`, `input_dim`, `seq_len`, `task`, `positive_class`, `metrics`, augmentation parameters… **but not the preprocessing configuration** (`normalize_sequence`, joint order, `valid_mask`, NaN policy, `fps`, pose version). Parity depends on C2 re-implementing the steps by hand (`INFERENCE_ENSEMBLE_HANDOFF.md`, "preprocessing": valid_mask → nan→0 → normalize → velocity → resize with **each model's own `seq_len`**).

Risks: (1) any change to `normalize_sequence` in this repo is not propagated; (2) different `seq_len` between #41 (64) and #48 (96) ⇒ each model sees a different version of time; (3) production windows of a different duration than the training clip (H4); (4) thresholds (0.80, etc.) depend on broken calibration (M6).

**Recommendations:** (1) move `normalize_sequence`/`add_velocity`/windowing to a versioned module (`pose_core` of the shared project) and store `preprocess: {version, ...}` in the checkpoint; (2) a parity test with a golden clip (golden E1/E2/E3 design in `docs/KANVIS_POSE_SHARED_PROJECT_SPEC.md` §10.2).

---

## 13. What is good (keep)

- UID per clip and `assert_no_uid_leak_between_splits`; augmentation/expansion **only** in train; reproducible split manifests.
- Correct L/R swap in `mirror` (`KEEP_KPS` = 5–12, pairs (0,1),(2,3),(4,5),(6,7)).
- Per-clip normalization (centering + scale) removes global-position and absolute person-size invariances.
- Evaluation separates val/test, computes a confusion matrix and several decision rules.
- Preflight with per-category inventory, `class_map` and an auditable training plan.
- Use of `valid_mask` with `poses_full` (NaN never reach the model).

---

## 14. Proposed experiment plan (ordered)

Common criterion: **session-grouped split**, ≥ 3 seeds, metrics with session-bootstrap CIs, main metric = **recall @ fixed FP/day at production prior** and PR-AUC.

| # | Experiment | Change | What is learned |
|---|---|---|---|
| E0 | Fix H1 and retrain #41/#48 with per-epoch selection | `best_state` with clone + selection by PR-AUC on val | Value of the real "best epoch"; whether the last epochs overfit |
| E1 | New protocol (H2+H3) | `StratifiedGroupKFold` by session; frozen test; frozen thresholds | **Honest baseline**; delta vs the reported F1 |
| E2 | Baselines without temporal pose | Aggregated per-clip features + GBDT; and a model that only sees nuisance metadata (`bbox_aspect`, `kp_conf`, `T`) | How much is session/duration shortcut |
| E3 | Quality filters | `valid_pct ≥ 50`, cap on T, remove degenerate clips vs no filter | Effect of data quality |
| E4 | Fixed windows (H4) | 3 s and 5 s multi-crop windows; same crop at inference | Clip/window mismatch; FP in SW |
| E5 | Augmentation | Remove `scale/shift`; add keypoint drop, temporal jitter, mild anisotropic scale; on-the-fly enabled | Which ops really help |
| E6 | Normalization | (a) hip center + torso scale; (b) pixels/isotropic; (c) with/without global position | Impact of M3 |
| E7 | Features | + confidence/visibility, + angles (elbow, shoulder), + wrist–hip/wrist–torso distances; per-channel standardization | Impact of M2 |
| E8 | Architecture (with fixed protocol) | BiRNN with correct pooling, positional encoding in Transformer, ST-GCN with real graph, TCN with norm. | Whether M1 changes anything; then pick the 2 best |
| E9 | Loss/sampling | CE vs focal vs asymmetric **one at a time**; with/without hard-neg; calibration on its own partition | Isolated effect of M6 |
| E10 | Ensemble | Seed averaging + architecture ensemble with frozen threshold | Stability vs grid selection |
| E11 | Decision rule and score space (see 16.8) | argmax vs softmax-threshold vs logit-margin vs calibrated probability; min/mean-prob/mean-logit/stacking; prior-shift correction; window aggregation (k-of-n, LLR sum, CUSUM) | Which rule gives the best recall at the FP/day budget with the smallest variance |

| E12 | Post-processing and cheap FP filters (see 16.9) | Persistence / k-of-n, per-session alarm cap, evidence accumulation across ID fragments, kinematic and zone gates, employee suppression, quality gate — each added **one at a time** on top of the frozen model | Recall lost and FP/day removed per filter, measured on a replay of unlabeled production + the labeled val sessions |

Deliverables per experiment: table with mean ± CI, PR curve, FP/day at real prior, FP per category (especially 14/3/2/7/13), and the split/manifest file.

---

## 15. Implementation checklist

- [ ] `best_state` with copy (H1) in `train_model_operations.py` **and** `train_model.py`.
- [ ] Session-grouped split + frozen test; document the group definition.
- [ ] Re-measure on the **real dataset** (Annex A): distribution by day/source, length, `valid_mask`, NaN.
- [ ] `evaluate_cell` with an "apply frozen val spec to test" mode.
- [ ] Fixed-duration windows in seconds in `PoseDataset` and at inference; mask/length channel.
- [ ] Replace `scale`/`shift` or move them after normalization; enable on-the-fly with non-null ops.
- [ ] Implement `evaluate_quality()` with the A/B/C filters of §4.2, `filter_report.csv` and the same gate at inference (C2); calibrate the thresholds marked "(calibrate)" on the new format.
- [ ] Torso-based normalization, isotropic space, explicit NaN policy (fail if NaN after mask).
- [ ] Masked pooling and/or fix `out[:, -1]`; positional encoding in Transformer/Conformer; anatomical graph.
- [ ] `AdamW` + scheduler + early stopping + `clip_grad_norm_` + 3 seeds.
- [ ] Decision rule per the §16.8 answers: score space, calibration on a held-out partition, prior-shift correction, threshold by FP budget with upper-bound CI, window aggregation.
- [ ] Report PR-AUC, recall@FP/day at real prior, FP per category, ECE; CIs by session bootstrap.
- [ ] Store a versioned `preprocess` + `pose_bundle_id` hash in the checkpoint when it exists.

---

## 16. Questions the expert must answer (design questions — do NOT ask them back)

**Purpose:** minimize false positives and maximize the percentage of robberies detected. Each item states **what decision your answer changes**. Answer each one with a concrete choice, number or rule, plus rationale, expected effect, how to test it and priority. For facts only the owner knows, **state the assumption you adopt** and give a conditional recommendation. "I don't know" is an acceptable answer only when accompanied by the measurement that would resolve it. **Questions whose factual premise the owner has already answered are listed in C.6.7: use those answers as given (do not treat them as unknowns) and answer the *design* part.** Questions 51–61 (16.9) were added at the owner's request after the first draft: they ask for the overall verdict, the best model recipe, and ideas for post-processing/filters.

### 16.1 Operational objective (defines the metric and the threshold)
1. **What maximum false-alarm budget should the design target?** Express it per store and day (and per camera and hour if relevant), given ≈ 1 FP/day on ≈ 1 000 sessions/day as the owner's stated goal. *Decides:* the operating point and the training objective (recall @ FP/day). Today "FP rate ≤ 0.01 %" is pursued on clips, which is not measurable.
2. **What should count as a "hit"?** Event-level (one positive window/track during the action is enough) or clip/window-level? *Decides:* whether windows are aggregated with k-of-n rules (which drastically reduce FP and raise event recall) or the per-clip decision stays.
3. **What FN:FP cost ratio and what reviewer model** (human in real time, deferred review, none) should be assumed? *Decides:* threshold, number of stages (Stage-2 verifier) and FP tolerance.
4. **Given a Stage-2 (video/object verifier) after the alarm, what recall and FP/day should Stage-1 aim for?** *Decides:* whether Stage-1 can be **high recall with moderate FP** (easier) instead of ≈ 0 FP using pose alone.
5. **What latency / context length is acceptable** (3, 5, 10 s)? Is a delayed alarm of a few seconds after the action acceptable? *Decides:* window length, causal vs bidirectional windows.

### 16.2 Data and labels (noise, bias, coverage)
6. **How many distinct sessions/people/days/cameras/stores do the robberies (cat. 6) cover?** (not clips). State the assumption if unknown. *Decides:* how well the model can generalize and which validation is possible (leave-one-session-out). Measured in the local sample: robberies concentrated in **2 origins**.
7. **Are the training robberies staged (actors) or real?** Same store/cameras/lighting as production? Were the negatives of the same sessions recorded at the same time with the same people? *Decides:* correction of the session⇄label bias (H2): paired negative sampling, or discarding the session as a signal.
8. **How was each robbery clip delimited?** Only the gesture (hiding the product) or also the walk before/after? How many seconds of the clip are truly "robbery"? *Decides:* **temporal** labels (localize the segment) vs clip labels; multiple-instance learning; window length.
9. **Are there robberies where the hiding happens out of camera / occluded?** What % are *not recoverable from pose* (wrists not visible)? *Decides:* rule C1/B3 and the realistic recall ceiling of a pose-only model.
10. **Label noise**: what % of cat-6 clips are not robbery and of negatives are? What criterion separates 6 (robbery), 14 (put in bag), 3 (put in basket), 2 (put back)? *Decides:* cleaning, label smoothing, robust learning, class merging.
11. **Is the "put in own bag" gesture (14) a robbery or not?** It is the most likely structural confusion. *Decides:* treat 14 as an explicit negative with high weight, or as an "ambiguous" class with a different alarm policy.
12. **Given unlabeled normal-operation data (hours of real cameras) and already generated production alarms with their review (TP/FP), how should they be used?** *Decides:* **hard-negative mining on production** and an active-learning loop: the most effective way to cut FP; also enables self-supervised pose pre-training.
13. **Given confirmed real robbery incidents for a blind test, how many are needed and how should they be used?** *Decides:* realistic final test and calibration of the prior.

### 16.3 Inference conditions (parity)
14. **Which window and stride does the model see in production** (3 s/1 s in evaluation, 8 s chunks + N/N+1 in production)? **Which fps** (cameras 15, pose 12–15 or up to ~18 in replay)? Should fps be aligned with training? *Decides:* temporal normalization (resample to a fixed fps) and the training window (H4).
15. **Should all tracks be evaluated or only those that pass a quality gate?** What should be done with tracks fragmented by *ID switches*? *Decides:* gate design (A/B/C), track segmentation, training on fragments.
16. **How should crowded/occluded scenes be handled?** What share of production tracks have ≥ 2 people nearby? *Decides:* whether multi-user clips must be included in training (today 46 % are dropped).
17. **Can scene context be used** (shelf zone/ROI per camera, hand–shelf distance, detected objects)? *Decides:* features lost when position is normalized away (hand near pocket/bag/shelf); raises recall and lowers FP.

### 16.4 Modeling decisions (the expert must choose)
18. **Anchor normalization on the torso (hips/shoulders) and scale by torso length?** Or keep global position/ROI? In pixels/isotropic? *Decides:* E6.
19. **Fixed windows in seconds with temporal multi-crop, or whole clip with resize?** Recommended: fixed windows. Which duration (2, 3, 5 s)? *Decides:* E4.
20. **Window-level labeling (localize the gesture) or clip-level?** Is *multiple-instance learning* acceptable (clip positive if any window is)? *Decides:* how window labels are built and whether the final fusion is *max/top-k* over windows.
21. **Binary robbery-vs-rest, or multiclass with the robbery class + explicit hard negatives (14/3/2/7/13)**; an "ambiguous" head? Asymmetric penalty per negative category? *Decides:* loss scheme and per-negative-type thresholds.
22. **Anomaly/one-class detection or few-positive learning** (e.g. self-supervised pre-training on millions of normal production tracks + fine-tuning on robberies)? *Decides:* whether setting up SSL pre-training is worthwhile (depends on Q12).
23. **Keypoint set**: add nose/ears (head orientation/gaze toward shelf or camera), knees/ankles (posture, crouching)? Or keep 8? *Decides:* new `KEEP_KPS` (implies re-extraction).
24. **Input features**: add per-keypoint confidence, angles (elbow/shoulder/torso), wrist–hip/wrist–torso distances, velocity/acceleration in torso-units/second? *Decides:* E7.
25. **Architecture**: given the real number of sessions (hundreds) and robberies, prefer a small, strongly regularized model (dilated TCN + masked pooling) or an ST-GCN/Transformer with pre-training? A seed ensemble of one model instead of AND of two architectures? *Decides:* E8/E10.
26. **Calibration and decision (summary question; details in 16.8):** calibrated probabilities (temperature/Platt on its own partition) + FP/day threshold, or an AND/mean/cascade rule chosen by grid search? Conformal / FP upper bound? *Decides:* selection protocol (H3).
27. **Temporal decision**: k-of-n rule over consecutive windows, minimum persistence (e.g. ≥ 2 s with P > threshold), hysteresis (high threshold to trigger, low to hold)? *Decides:* post-processing design (the usual lever against single-frame FP).
28. **FP prioritization**: is a "review" output (grey zone) acceptable besides "alarm"? *Decides:* two-level thresholds and cascade with Stage-2.

### 16.5 Evaluation and experiment governance
29. **Which validation protocol is acceptable**: leave-one-session-out, by day, by camera? What **minimum test size** (sessions and robberies) and which CI (session bootstrap)? *Decides:* how much new data must be collected for the metrics to be reliable.
30. **Concrete success criterion** (template to fill in): "event recall ≥ __ % with ≤ __ FP/day per store, lower 95 % CI ≥ __ , on unseen sessions". *Decides:* when to stop.
31. **Is retraining on reviewed production alarms every N weeks (active learning) acceptable**, and how should models/thresholds/quality gate be versioned?

### 16.6 Quick decision table (if the answer is X → do Y)

| Key answer | Action in training |
|---|---|
| There is a Stage-2 with video (Q4) | **High-recall** Stage-1 (low threshold) + k-of-n post-processing; evaluate event recall and FP/day after Stage-2 |
| Only ≈ 2–3 robbery sessions (Q6) | Absolute priority: **collect more robbery sessions** and paired negatives; without it no architecture generalizes; use leave-one-session-out and small models |
| Robberies staged, negatives from other sessions (Q7) | Train with negatives from **the same sessions/scenes**; add augmentation that equalizes scale/jitter; evaluate on never-seen sessions |
| Many hours of unlabeled production (Q12) | **Hard-negative mining** + SSL pre-training; retrain with reviewed FP each cycle |
| Temporal labels possible (Q8/Q20) | Re-label the gesture segment; train per window; decide by max/top-k |
| 14 is considered non-robbery (Q11) | Explicit negative with high weight + FP-per-category metrics; consider ROI context (Q17) |
| Latency tolerable ≥ 5 s (Q5) | 5 s windows or 2 consecutive 3 s windows with persistence |
| Scene context available (Q17) | Add ROI/shelf-distance features; ablation with/without global position |
| Production fps differs (Q14) | Resample to a fixed fps in the shared module and retrain |
| The session-grouped learning curve flattens (Q33) | Do not collect more clips of the same kind: prioritize **diversity** (more real sessions/people/cameras, more hard negatives 14/3/4/2, production data) |
| The curve keeps rising (Q33) | Collect more **real** robberies and more sessions before changing the architecture |
| Frontier identical in August and October (C.6.4) and the 2-of-4-at-0-FP preference (Q60) | Choose the operating point on the low-FP part of the frontier; spend effort on **information/data** (Q61) and post-processing (Q55), not on architecture sweeps |
| Phone/legitimate concealment labeled as robbery (Q56) | Split or soft-label the positive class; treat phone clips as a separate head or policy |
| Production sessions are fragments (C.6.5, Q57) | Train on random partial crops with gaps/dropout; evidence accumulation across fragments |
| Margin/softmax tail analysis shows saturation (Q36–Q38) | Choose thresholds in logit space (or by per-model negative quantiles) and report with upper-bound CIs |

### 16.7 Data size: how many clips are needed for a robust system?
32. **How many clips (and above all how many distinct sessions/people) do you estimate are needed for the system to be robust** with the target ≈ 66–75 % event recall and ≈ 1 FP/day on ≈ 1 000 sessions/day? Give **concrete numbers**, broken down into: (a) number of **real** and staged **robbery events**; (b) number of distinct **people/sessions/days/cameras/stores** in the robberies; (c) clips per **confusable** category (14, 3, 4, 2, 5, 9, 7, 13); (d) number of **negatives** and whether they should be labeled clips or **unlabeled production windows** (hours); (e) minimum **val** and **test** size (in sessions and robberies). *Decides:* what to collect/label and in which order, and whether the bottleneck is clips, diversity or labels.
33. **Which learning-curve experiment would you accept as proof?** Proposal: train on 25/50/75/100 % of **sessions** (not clips), ≥ 3 seeds, and measure **recall at fixed FP/day** on unseen sessions; what slope (recall points per doubling of data) justifies continuing to collect data? *Decides:* when to stop collecting and whether more of the same is enough.
34. **Does adding staged or real robberies help more?** What real:staged ratio do you recommend and how many execution variations per actor (angle, speed, object, bag, clothing) are needed? *Decides:* recording budget vs cutting real sessions.
35. **How many hours of "normal" unlabeled production** are needed for (a) hard-negative mining and (b) estimating FP/day with an acceptable confidence interval? *Decides:* size of the replay benchmark (see §17: ≈ 15 000 negative sessions to measure 0.1 % ± 0.05 %).

### 16.8 Score space and decision rule: softmax, argmax, logits… (requires an in-depth analysis)

> **This block requires the deepest analysis in the whole response.** Take all the time you need. Do not answer from memory: (i) derive the relations analytically; (ii) verify them empirically on the repository's real logits where accessible (checkpoints in `campaign/artifacts/` or `models-operation*`, `calibration_utils.collect_val_logits`, `evaluate_campaign.collect_binary_predictions`) or on simulated logits that mimic over-confident networks, stating clearly which; (iii) quantify, for each alternative, recall at the FP budget, its variance across seeds/resamples and its stability under distribution shift (session split, fps change, quality gate on/off); (iv) discuss failure modes (saturation, ties, calibration drift, prior shift); (v) commit to a recommended rule and give the expected gain. Facts about the current implementation are in §11.4.

36. **Decision score — which quantity should the system threshold?** Compare, for the binary and multiclass cases: (a) `argmax` (implicit threshold 0.5), (b) softmax probability of the robbery class with a threshold, (c) raw **logit margin** `logit1 − logit0` (or `logit_6 − logsumexp(other)` one-vs-rest in multiclass), (d) **calibrated probability** (temperature/Platt/isotonic/beta calibration), (e) energy/max-logit/OOD-style scores, (f) entropy/uncertainty-aware abstention. State which is best for a **≈ 0.1 % FP budget**, with the proof or experiment that supports it. *Decides:* the production score and the code path in C2.
37. **Are softmax-threshold and logit-margin really equivalent in this system?** For one binary model they are monotone transforms (§11.4); show where the equivalence breaks (ensembles, temperature, multiclass, float32 saturation at p ≈ 1.0, the coarse 0.02 grid vs the truncated margin grid ending at 2.0). Quantify how many candidate thresholds are unreachable or tied in the tail with the current sweeps and what grid/quantile-based search should replace them.
38. **Saturation and tails:** at the operating points required by a ≈ 0.1 % FP budget, how much ranking information does softmax lose compared with logits (ties, resolution)? Should thresholds be defined in logit space, as a **per-model quantile of validation negatives**, or as a target FP rate with a conformal/Clopper–Pearson bound? Propose the exact procedure and the sample size it needs (§17.1).
39. **Calibration:** is a single global temperature adequate, or should one use per-class/vector scaling, Platt, isotonic or beta calibration — and **on which partition** (calibration set disjoint from threshold-selection and from test; session-grouped)? How should the asymmetric loss, hard-negative mining, sampler weights and class-expansion change what "calibrated" means? Provide a calibration + reliability (ECE, reliability diagram) protocol and the expected effect on the stability of thresholds across retrainings.
40. **Prior shift (training prior ≠ production prior):** should the logits be corrected with the offset `ln(π_prod/(1−π_prod)) − ln(π_train/(1−π_train))`? How should π_train be defined given the sampler, class expansion and loss reweighting, and π_prod given 1 000 sessions/day and several windows per session? Quantify the effect on the threshold and on per-session FP.
41. **Ensemble combination in score space:** compare `min(p)` (AND), mean of probabilities, **mean of logits** (geometric pooling), max, rank averaging, **stacking** (logistic regression on logits fitted on a held-out set), and learned fusion — for 2 models vs a **seed ensemble** of one model. Which rule gives the best recall at the FP budget, with the lowest variance, and which is the most robust to a calibration mismatch between the two members (#41: seq_len 64, #48: seq_len 96)? Should the thresholds be per-model or global?
42. **Temporal aggregation of scores:** over consecutive windows (chunks N and N+N+1, 3 s sliding windows), compare max, mean, top-k, noisy-OR, **sum of log-likelihood ratios (logit accumulation) / CUSUM / sequential probability ratio test**, hysteresis, and k-of-n persistence. Which is best at event level and how much recall does persistence cost given robbery events last ~3–11 s? How should the per-session threshold depend on session duration (1 vs 4 vs 43 chunks)?
43. **Multiclass vs binary scoring:** is it better to train multiclass and read `p_robbery` (shared probability mass with 14/3/4/2) or the one-vs-rest margin, or to train binary with explicit hard negatives? Verify the `prob[:, 0]` assumption in `collect_binary_predictions` (§11.2.6). What are the consequences for calibration and for per-negative-category FP?
44. **Uncertainty and abstention:** would deep-ensemble disagreement, MC-dropout, entropy or distance-to-training-data (OOD) reduce FP at a given recall (selective classification / risk–coverage curves)? Quantify the expected gain and the extra cost, and whether it composes with the quality gate (§4.2).
45. **Threshold selection protocol:** replace "maximize F1 on val" by a procedure that targets the FP budget at production prior with a statistical guarantee (e.g. upper confidence bound on FP per 1 000 sessions, conformal risk control). Specify the data needed (calibration and test sets, sessions not clips), how to avoid selection bias when many candidates are compared, and how to **freeze and version** the thresholds.
46. **Stability across retrainings:** the owner wants "same gesture → same decision". Compare the decision rules above by the **variance of decisions across seeds/retrainings** and across small perturbations of the same clip (jitter, ±1 frame shifts, fps resampling). Which rule is the most stable?
47. **Metrics to report:** propose the final metric set (PR-AUC, recall @ FP/day at production prior, partial AUC in the low-FPR region, ECE, FP per category, event-level recall, per-session FP) and say which single number should be used to choose between models.
48. **Where in the pipeline should the decision logic live** (model head, C2 ensemble rule, C3 fusion): should C2 emit **logits/margins** instead of probabilities so that downstream stages (SAM3/HAR/C3) can fuse in log-odds space? Define the interface/contract that keeps training and production score semantics identical (cf. §12).
49. **Hyper-parameter coupling:** `seq_len`, window length and fps change the score distribution; should thresholds be re-fitted per `seq_len`/fps? Propose a rule to avoid silent threshold drift when preprocessing changes.
50. **Final recommendation:** give a single **recommended score + calibration + ensemble + aggregation + threshold-selection procedure**, the expected operating point (FP/day, event recall) with CIs, and the minimal experiment (from E11) that would falsify it.

### 16.9 Overall verdict, best model recipe, and ideas to cut false positives (requested by the owner)

> Facts you must use: C.6 (single store/9 cameras, labeling workflow, robbery taxonomy incl. phone pocketing, the August=October frontier, the 2-of-4-at-0-FP preference, no latency limit, lighter-is-better-if-similar, no training-time limit, fragmented sessions, Stage-2 = heuristics + SAM3 later). You cannot execute anything: reason, quantify with the figures given, and write protocols.

51. **Verdict — is anything being done wrong in the training?** Give a **ranked list** (most damaging first) of the mistakes, omissions and risky choices you find in the whole training pipeline (not only those listed in §0), each with: evidence, estimated effect on FP and on recall (in production units), confidence, and the cheapest check. Say explicitly which findings of this audit you consider **irrelevant or wrong**. Then answer: *"Given everything, is the observed ceiling (≈ 11 FP / ≈ 64 % recall on 1 619 negatives, identical in August and October) a modeling problem, a data/label problem, or an information problem (8 keypoints, no hands/objects)?"* and justify the split in percentages.
52. **The best model recipe.** Specify one concrete configuration to train next: input representation (keypoints, normalization, features, velocity/acceleration, confidence/visibility channels), window definition (seconds, fps, stride, `seq_len`), architecture (family, depth, width, receptive field, pooling, normalization), regularization (dropout, weight decay, label smoothing, augmentation ops and strengths), optimizer and schedule (LR, warmup, cosine, EMA/SWA), **number of epochs and the stopping criterion** (in terms of recall@FP, not `val_acc`), loss and sampling, number of seeds, and how the final ensemble is built. Provide the expected gain and the ablation order that would validate each choice. *Decides:* the next campaign.
53. **Light-versus-heavy rule.** The owner prefers a much lighter model if its numbers are similar. Propose (i) a **shortlist of 4–6 architectures** (e.g. dilated TCN, small GRU/LSTM with masked pooling, ST-GCN with an anatomical graph, a small Transformer/Conformer with positional encoding, a gradient-boosted/linear baseline on handcrafted pose features) with approximate parameter counts and latency on a 5090 and on a small CPU; (ii) a **quantitative definition of "similar"** (e.g. within one session-bootstrap standard error of recall at the FP budget, or ≤ 2 pp); (iii) whether a **distilled** light student of a heavy teacher/ensemble is worthwhile; (iv) whether the current choice of two architectures (TCN-LSTM and TCN-GRU, sizes 128/256) is justified or a seed-ensemble of one light model does as well.
54. **More epochs, longer context, bigger `seq_len`?** Given the H1 bug (the "best" epoch is not restored), the current epochs (40–60) and the observation that TCN-LSTM/GRU have a 5-frame receptive field, say whether the models are under- or over-trained, whether longer training with a proper schedule would change the frontier, and what `seq_len`/fps (including multi-scale or multi-resolution inputs, e.g. 2 s + 6 s branches) you would test. Estimate the effect on FP and recall.
55. **Post-processing and filter ideas to reduce FP — brainstorm and rank.** Beyond the model, propose **at least 15 concrete ideas** that run *before* or *alongside* SAM3 and that use only data that already exists (poses, tracks, cameras, time, session history, model scores, the 9 cameras' geometry). For each idea give: what it does, why it should remove FP, the **expected FP reduction and robbery loss**, the implementation cost, how to **validate it without labels** (replay of unlabeled production + the reviewed alarms), and whether it respects the generalist requirement (no action-by-action tuning). Mandatory topics to cover (add your own): (a) temporal persistence / k-of-n / hysteresis on consecutive windows; (b) a **per-session alarm cap and cool-down**, and evidence accumulation (log-odds sum) across chunks N/N+1; (c) **merging or chaining ID-fragment sessions** (spatio-temporal continuity near where the previous track vanished) so that evidence accumulates and duplicate alarms are not issued; (d) **employee suppression** (long sessions, repeated trajectories, dwell zones, restocking motion; categories 7/13) without losing real robberies by a person who looks like an employee; (e) **zone/ROI priors per camera** (aisle vs cashier vs entrance vs fitting/storage area) and per-camera thresholds or calibration; (f) **kinematic sanity gates** (wrist reaching the hip/pocket/torso region, hand trajectory, pose visibility, minimum movement amplitude) as *soft* features, not hard rules; (g) the **quality gate** of §4.2 (visible keypoints, track length, jitter) applied identically in training and inference; (h) **scene context** (other people nearby/occlusion, crowding, proximity to shelf, queue); (i) **time-of-day / day-of-week priors** and store-state (opening, closing, restocking hours); (j) **model disagreement / uncertainty** (ensemble variance, margin between the two models, OOD distance) as an abstain signal; (k) **different thresholds by session length**; (l) feeding the **model's per-window scores and pose summaries** (not just a binary alarm) to SAM3/HAR/C3 so that Stage-2 can fuse in log-odds; (m) active learning from reviewed alarms (Q12/Q58). Finish with a **recommended stacking order** (cheapest/most-general first) and the combined expected operating point.
56. **Label policy and class semantics.** The positive class mixes "stealing" with "concealing any object on the body", **including a mobile phone being pocketed** (a legitimate action by any customer), while cat. 14 ("puts an object in own bag") is a negative. Analyze the consequences for the model and for FP, and decide: keep as is; split cat. 6 into **sub-classes** (conceal-in-pocket / under-clothing / bag; phone vs other object; standing vs kneeling) and train multi-task or hierarchical; move the phone clips to a separate "ambiguous/legitimate-concealment" class with a different alarm policy; or relabel with a **soft/ambiguous** target. Quantify the effect on FP, and say what the best, cheapest re-labeling effort is (e.g. re-review only the clips the models disagree on or that sit near the decision boundary).
57. **Training for fragmented, partial sessions.** In production a person can be split into several sessions, a session can start mid-gesture and last only a few seconds, and tracks have gaps; training clips are clean, single-action and action-centered. Design the **training-data transformation** that matches production: random temporal crops that cut the gesture at both ends, simulated tracker gaps and dropped frames, simulated keypoint dropout from occlusion, concatenation of non-robbery context before/after the action, variable length with masking, plus a "no-history" robustness test. State how the *labels* of such crops are assigned (a crop that contains only the approach, only the first half of the gesture, only the aftermath) and which loss (weak/MIL, soft labels) to use. Quantify the expected effect on FP (production chunks of arbitrary 8 s) and on recall (fragments).
58. **Using production data for training (closing the loop).** The owner has production alarms already reviewed (≈ 20 FP in 654 sessions on one day) and hours of unlabeled footage in the same store. Specify the **hard-negative / active-learning loop**: how to mine (FP, near-threshold, disagreement), how to label cheaply, how to weight and mix with the labeled set (and protect robbery recall), how many reviewed FP are needed before the frontier moves, how to avoid overfitting to the 9 specific cameras, and whether **semi-/self-supervised pose pre-training** on unlabeled tracks is worth it. Provide the expected FP reduction per cycle.
59. **Single store, 9 cameras — splits, domain shift and honest evaluation.** The data come from one store and nine cameras but the split is random per clip. Design the validation scheme (**leave-one-camera-out**, by day, by person/session) that best predicts production, the minimum size of the frozen test set, and the measures against camera/viewpoint shortcuts (per-camera normalization, viewpoint/perspective augmentation, camera-adversarial training, per-camera calibration). Say how much of the current 76 % F1 you expect to survive an honest split, and answer: *should the model be trained for the 9 cameras jointly, per camera group, or with a camera embedding?* Also explain how you would **translate clip-level numbers (0.68 % FP over 1 619 clips) into FP per day** in production (sessions per day, chunks per session, correlation between chunks) so the owner can compare with the 1 FP/day target.
60. **Make the owner's preference an explicit utility.** The owner prefers 50 % recall with 0 FP/day to 75 % with 5 FP/day. Formalize it (cost ratio FN:FP, or a constrained objective: maximize event recall subject to FP/day ≤ k), derive from the observed frontier (FP ≤ 3 → ≈ 42–48 %; ≤ 8 → ≈ 58 %; ≤ 11 → ≈ 64 %; ≤ 20 → ≈ 72–74 %; ≤ 45 → ≈ 83–85 % recall, on 1 619 negatives) **which operating point is optimal** and what FP/day each corresponds to in production, and say what is *realistically reachable* now, after your recommended fixes, and after additional data/features. Provide a table recall vs FP/day with confidence intervals.
61. **What would move the frontier?** Rank the interventions by expected gain per unit cost: more robbery *sessions* vs more hard negatives (14/3/4/2/5/9); head/nose/ear and knee keypoints (re-extraction of the full dataset is feasible in ≈ 1–1.5 days on the RTX 5090); **hand/finger keypoints or hand crops**; per-wrist image embeddings / optical-flow patches (object-in-hand information) fused with the pose model; scene context features; temporal labels; a larger or smaller model. For each, say whether it is expected to move the **FP–recall frontier** (information gain) or only the position along it, and give the cheapest experiment that proves it (e.g. an upper-bound study with oracle features).

---

## 17. Initial data-volume estimate (author's working hypothesis, not measured)

> Provided as a starting point for the expert to confirm or correct (Q32). The statistical computations are exact; the training figures are **engineering judgment**.

### 17.1 How much is needed to **measure** (validation/test) — statistical computation
Unit = **independent session/event** (not clip; clips of the same session are correlated):

| What is to be shown | Formula | Required size |
|---|---|---|
| Session FP ≤ 0.1 % with 0 observed FP (rule of three, 95 % CI) | N ≥ 3 / 0.001 | **≈ 3 000 negative sessions** with no FP |
| Measure a 0.1 % FP rate to ± 0.05 % (95 % CI) | n ≈ 1.96²·p(1−p)/h² | **≈ 15 000 negative sessions** (≈ 15 production days at 1 000 sessions/day) |
| Measure a 1 % FP rate (Phase-1 goal: ≈ 10 per 1 000) to ± 0.3 % | same | **≈ 4 200 negative sessions** |
| Event recall = 66 % to ± 5 pp (95 % CI) | n ≈ 1.96²·p(1−p)/h² | **≈ 345 independent robberies** (≈ 176 for ± 7 pp, ≈ 87 for ± 10 pp) |
| Detect a 5 pp recall improvement between two models (paired) | depends on discordance | on the order of **≥ 300–400 independent robberies** |

Implications: the current val (≈ 1 600 negatives / 323 robberies, *clips* from few sessions) gives CIs of 0.38–1.21 % for FP and 58–69 % for recall, i.e. it **cannot distinguish** close configurations nor verify the 1 FP/day goal. A **production replay benchmark** is needed (≥ 15 000 real negative sessions + ≥ 300 independent robberies from distinct sessions) and a frozen test never used to choose anything.

### 17.2 How much is needed to **train** — engineering judgment
The number of clips matters less than their **diversity** (people, clothing, cameras, angles, lighting, objects, stores, how the gesture is executed). With a pose-only 8-point model and a 32-dimensional input, the information limit (§C.4) and the session shortcuts (H2, M8) weigh more than the number of clips:

| Item | Today (per brief) | Reasonable minimum (hypothesis) | Robust (hypothesis) |
|---|---|---|---|
| Total robberies (events) | 2 592 clips (≈ 63 % staged, ≈ 945 real) | ≥ 1 000 with ≥ 500 real | **3 000–5 000** with ≥ 1 500 real |
| Distinct people/sessions in robberies | unknown (in the local sample 2 origins dominate) | ≥ 150 | **≥ 300–500**, ≥ 3 stores/angles, ≥ 10 days |
| Variations per staged actor | unknown | ≥ 5 (object, bag, speed, angle, clothing) | ≥ 10, with different actors |
| Confusable categories (14, 3, 4, 2) | 14: 219 · 3: 1 766 · 4: 2 119 · 2: 1 733 | ≥ 500 **distinct** sessions per category (14 is far below) | ≥ 1 000 per category; 14 is the priority |
| Negatives | ≈ 16 000 clips | + **≥ 50 000–100 000 normal production windows** (unlabeled) for hard-negative mining | ≥ 500 000 windows + FP review loop |
| Val / test | ≈ 1 600 neg / 323 robberies (clips) | ≥ 3 000 neg and ≥ 175 independent robberies, by sessions | ≥ 15 000 neg and ≥ 345 independent robberies |

Observations supporting these figures (all from the repo itself): the October extension added ≈ 3 650 negatives but only +18 robberies and **did not move FN**; the FP–recall frontier is identical across campaigns; run-to-run noise of the same model (sd ≈ 3–4 F1 points and ≈ 1 pp FP) exceeds the differences between architectures; 99.7 % of the robberies in the local sample come from 2 origins.

### 17.3 How to check it with current data (cheap, before collecting more)
1. **Session-based learning curve**: train on 25/50/75/100 % of the sessions (stratified, grouped), ≥ 3 seeds, and plot *recall @ fixed FP/day* on unseen sessions (scripts exist: `campaign/run_learning_curve.sh`, `learning_curve_utils.py`, but with a per-clip split; adapt to groups). Decision rule: if each **doubling of data** raises recall < ≈ 3 pp at constant FP, more clips of the same kind do not help → collect **diversity** (real sessions, cameras, category 14, production).
2. **Curve by number of people** (not clips): subsample keeping the number of clips but reducing the number of sessions/people, to separate "more data" from "more diversity".
3. **Estimate the information ceiling**: aggregate-feature baseline + temporal model; if both saturate at the same FP–recall point, the limit is information (context is needed: ROI/object/hands), not data.

---

## Annex A — How to reproduce/verify the measurements on the real dataset

Replace `ROOT` with the real `data_result` folder (`OUTPUT_BASE/data_result`).

1. **Session/source ⇄ label confounding**: walk `*/*/meta.json`, keep single-user clips, group by date prefix of `clip_name` and/or `video_source`, compute the % of `cat==6` per group. If some groups are ≈ 0 % and others > 30 %, there is confounding.
2. **Temporal overlap**: per `(date, video_source)` compare `[t_start, t_end]` between clips; % of clips that overlap or are within ≤ 10 s of another.
3. **Metadata shortcuts**: rank AUC of `total_frames`, `valid_pct`, `keypoint_confidence_avg`, `occlusion_ratio`, `bbox_aspect_ratio` for robbery vs rest (≈ 0.5 = no shortcut; ≥ 0.7 = suspicious).
4. **Scale and padding**: on a sample of ~1 500 clips, compute the `normalize_sequence` scale, the per-frame intra-clip CV and the padding/subsampling % per `seq_len` (64/96/128).
5. **`valid_mask` and NaN**: % of dropped frames, NaN remaining after the mask, distribution of `T`.
6. **Bug H1**: run the minimal test:
   ```python
   import torch, torch.nn as nn
   m = nn.Linear(2,1); opt = torch.optim.Adam(m.parameters(), lr=0.1)
   best = m.state_dict(); w0 = best['weight'].clone()
   for _ in range(5):
       opt.zero_grad(); m(torch.ones(1,2)).sum().backward(); opt.step()
   print(not torch.equal(best['weight'], w0))   # True => alias
   ```
7. **Decision-rule analysis (16.8)**: dump per-clip logits of every candidate model on val/test (session-grouped), then compute margin/softmax rankings, tie counts at p ≈ 1.0, reliability diagrams, and recall at the FP budget for each rule.

8. **M10 / evaluation pool:** compare `splits/<cell>/split_manifest_exp_XX.json` UIDs against the examples actually used by `build_pose_dataset_for_eval` and by training (`single_user_only=True`): how many train/val/test clips, per category, are dropped; do the dropped clips (multi-user) concentrate in some categories or cameras; evaluate the same checkpoints with `single_user_only=False`.
9. **Camera/day structure (Q59):** from `meta.json` (`video_source`, clip name date prefix) derive camera and day per clip; report positives/negatives per camera and per day, and re-run the evaluation as leave-one-camera-out for the two production models.

## Annex B — Code references (`train_model_operations.py`)

| What | Line |
|---|---|
| `build_deterministic_variant_specs` | 115 |
| `expand_examples_with_category_augmentation` | 625 |
| `build_pose_dataset_for_eval` | 981 |
| `PoseExample` | 1145 |
| `_example_uid` | 1231 |
| `_user_quality_ok` | 1275 |
| augmentation ops (`_apply_*`) | 1334–1405 |
| `_collect_examples_from_root` | 1455 |
| `normalize_sequence` / `add_velocity` / `temporal_resize` | 1600 / 1614 / 1622 |
| `PoseDataset` (`__getitem__` 1707–1780) | 1639 |
| Models (TCN … GAT-TCN) | 1785–2520 |
| `split_examples_stratified_by_uid` | 2606 |
| `make_binary_examples` | 2749 |
| `_classification_loss_per_sample` / `_mixstyle_features` | 2794 / 2825 |
| `train_one_epoch_advanced` | 2854 |
| `evaluate_with_metrics` (binary rules at ~3095–3105) | 2986 |
| `build_datasets_and_loaders` | 3117 |
| `build_model` | 3599 |
| `run_experiment` (bug H1 at 3926–3931) | 3741 |
| `main` | 4335 |

Others: `campaign/train_campaign.py` (does not pass `augment_on_the_fly`), `campaign/evaluate_campaign.py` (`pick_best_*` L282–315, `ensemble_grid_binary` L388, `evaluate_cell` L448), `campaign/calibration_utils.py` (`fit_temperature`, `calibrate_model_on_val`), `campaign/campaign_config.json` (`experiment_ids`, `cells`, `aug_profiles`, `threshold_sweep`, `ensemble`).

## Annex C — Measured figures (old local sample; re-measure on the real dataset)

> **Real figures from the October campaign artifacts are in C.6** (category counts, splits, results, wall times). The list below is from the **old local sample** and is kept only to illustrate shortcut types.

- 10 790 clips, 5 851 single-user. Clips per category (all): 0:2 961 · 1:910 · 2:677 · 3:1 403 · 4:1 202 · 5:2 342 · 6:444 · 7:436 · 8:81 · 9:295 · 10:16 · 12:12 · 13:11. Single-user: 0:1 560 · 1:479 · 2:397 · 3:740 · 4:700 · 5:1 281 · 6:296 · 7:209 · 8:15 · 9:155.
- FPS ≈ 12 (11.8–12.6). Clip duration: median 10 s, p95 45 s.
- Frames per user (full): p1 27 · p25 74 · median 103 · p75 156 · p95 362 · p99 714. % with T<64: 17.8 %; T<96: 43 %; T<128: 64 %.
- Body scale (image coords): median 0.054 (p5 0.026, p95 0.093); intra-clip CV median 0.196 (p90 0.47); \|pos\|/\|vel\| after normalization ≈ 11 (0.625 vs 0.058).
- `passes_filters=True`: 12.2 % of users (2 326 / 18 992).
- Robbery-vs-rest AUC: `bbox_aspect_ratio` 0.775 · `keypoint_confidence_avg` 0.664 · `total_frames` 0.567 · `valid_pct` 0.487 · `clip_duration` 0.432 · `occlusion_ratio` 0.434 (and from filter analysis: scale 0.626, jitter 0.269, motion 0.331, wrist motion 0.37, border 0.308).
- Origin of robberies: `21Diciembre2025` 199 pos / 63 neg; `clips_output_*` 96 pos / 0 neg; rest of days ≈ 1 pos / 5 492 neg.
