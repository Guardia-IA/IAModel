# Kanvis — Shoplifting Detection from Pose Sequences
## Context, Data Audit, Results (Aug/Oct 2026) and Research Brief for an Expert Agent

> **Status: v5 (owner answers of 2026-10-13/14, rounds 1–5, incorporated).** Category semantics, data provenance (incl. curated clips = real sessions), production pose model, production ensemble rule, store/camera setup, test-split status, business targets (≈1 000 sessions/day, ≈1 FP/day budget end-to-end), fixed 12 fps option and the production FP level of the model (≈20–30 FP per 1 000 clips) are **confirmed by the owner**. §12 lists what remains open, including the fps-consistency question for the expert agent.
> **Language of this document: English** (requested). Everything below was derived from the repository, the two training campaigns and a dedicated audit of their artifacts. Numbers marked *(computed)* were calculated in this audit from the campaign artifacts; they are not in the original campaign reports.

> ## ⚠ ONE-SHOT, NO-QUESTIONS TASK
> **Take as much time as you need: analyse and reason as deeply as you possibly can; there is no time limit.** **You will receive this brief (plus the sample file `SAMPLE_production_chunk_mergedfps.npy`, and possibly access to the repository) and must produce a complete response file on your own. The person who launches you has no technical knowledge and cannot answer questions or make decisions during your work. DO NOT ASK QUESTIONS — not to the user, not in the chat, not inside the response file as requests for a reply. Whenever something is missing or ambiguous: choose the most plausible assumption, state it, give a conditional recommendation and continue.** See "NO INTERACTION" below.

---

## READ FIRST — Role, mandate and required output format (for the receiving agent)

### Who you are
You are acting as a **senior expert agent** combining deep expertise in:
- **Machine learning / deep learning for time series and skeleton-based action recognition** (TCN/GRU/LSTM/Transformer/GCN families, PoseC3D, self-supervised and pretrained motion encoders), loss design for imbalanced/low-FPR regimes, calibration, uncertainty, conformal/risk-controlling prediction, noisy-label learning, PU learning, domain shift and robustness;
- **Human behavior analysis and action understanding** (gesture semantics, intent, concealment vs. benign actions such as pocketing a phone, bagging own items, carrying objects; how people move in retail aisles; what is and is not observable from 2D upper-body pose);
- **Computer vision pipelines** (YOLO-pose + multi-object tracking/ReID, ID-switch handling, TensorRT deployment, SAM-based segmentation/ROI reasoning, object-in-hand/concealment cues, temporal sampling/fps issues);
- **Retail loss prevention and applied ML evaluation** (benchmark design, group-aware splits, production replay, session-level metrics, cost-sensitive operating points, statistical rigor with confidence intervals).

Take the stance of a critical reviewer: challenge the assumptions and conclusions in this document, verify claims against the repository and artifacts where you can, and say explicitly when something here is hypothesis rather than fact (items are marked as *verified*, *computed*, *hypothesis* or `[TO CONFIRM]`).

### Depth of reasoning (mandatory)
**Perform the deepest analysis and reasoning you are capable of. There is no time limit, no urgency and no penalty for taking long: take all the time you need.** Do not rush, do not stop at the first plausible explanation, and do not shorten or summarize your work to save time or space. Prefer a slow, exhaustive, self-critical analysis (explore alternative hypotheses, verify each claim, run small analyses where you can, re-check your conclusions before writing them down) over a fast answer. A long, careful and complete response file is expected.

**Reason in depth, with no time limit.** Take as long as needed; do not give a quick or generic answer. Use **any sources you consider appropriate**: the repository and artifacts, your own derivations and small experiments/analyses on the existing data, scientific literature (skeleton-based action recognition, anomaly/shoplifting detection, low-FPR learning, calibration, noisy labels, domain shift), open-source repositories, benchmarks and documentation, and web search where available. Cite the sources you rely on. Prefer evidence and quantification over opinion; when you cannot verify, say so and state how it could be verified. Explore alternative hypotheses before settling on a conclusion, and revisit the brief's own conclusions critically.

### What you must do
1. Read the whole brief, then inspect the referenced repository paths and artifacts (§13) as needed.
1a. **Address every item of §0b** (owner-highlighted structural facts): training-by-action vs production-by-fixed-8 s-chunk mismatch and your proposed solution (§0b.1); `full` vs `filtered` (§0b.2); why winners change between campaigns (§0b.3); fps contract and temporal-compression issue (§0b.4); production-vs-validation FP (20 FP / 654 sessions vs 11 FP / ~1 500 clips) (§0b.5); the generalist, no-per-action-tuning requirement (§0b.6); **the chunking design of production (§0b.7)**, **training/production `.engine` file difference (§0b.8)**, **the Triton production preprocessing path and its possible mismatch with training (§0b.10)** and the **analysis of the production sample `.npy` (§0b.9)**.
1b. **Work through the full design-variable questionnaire in §12.3** (chunk/window length, number and choice of body keypoints, fps, features, normalization, architecture, loss, data, labels, pipeline…). Each variable is a question: answer it with reasoning and, whenever possible, with a measurement or a concrete experiment design, and state the expected direction and size of its effect on FP and recall.
2. Answer the **open questions** (§12.2 and the pending items in §12.1) and **evaluate every proposal** in §10 and every experiment in §11: confirm, reject or refine it, and say what evidence supports your position. Propose additional ideas that are not in this document — the owner is looking for concrete **technological proposals that boost robbery detection and, above all, reduce false positives in a generalist way** (not clip-by-clip patching), using the data that already exists.
3. Prioritize by *expected FP reduction at ≈ 66 % event-level recall* per unit of engineering effort, and be explicit about assumptions, risks, expected gain (with rough numbers where possible) and how to validate each proposal on the production-like benchmark (§10 P0, E-replay).
4. Where useful, produce runnable artifacts (scripts, configs) **in separate files**, and reference them from your answer document. Do not run long retrainings or change production code unless explicitly authorized; plan them instead.

### The three things the owner needs above all (make them unmistakable in your response)
Your response file must make these three deliverables **explicit, prominent and impossible to miss** (e.g., three clearly titled top-level sections right after the executive summary), because the owner will read them first:
1. **DIAGNOSIS — where exactly is the problem behind the "high" false positives?** State your *ranked* root-cause analysis: which factors explain the gap between validation FP (≈ 7 per 1 000) and production FP (≈ 20–30 per 1 000), and which explain the FP floor even in validation (~0.7 % at ~64 % recall). For each cause give: evidence (measured vs. hypothesis), estimated share of the FP, and how to confirm it quickly. Do not list possibilities without ranking them; commit to a most-likely explanation and say what would change your mind.
2. **SOLUTION — what should be done, concretely, to get there?** One coherent, prioritized, end-to-end solution (not a menu): Phase 1 (stage-1 model: data, windows/fps contract, keypoints/features, objective, training and evaluation protocol, thresholds) and Phase 2 (HAR/SAM3/alternative filters/fusion), with the expected operating point in production-comparable units (FP per 1 000 sessions, event-level recall, alarms/day vs. the ≈ 1 FP/day budget), the main risks, and the order of execution.
3. **ANSWERS — a direct answer to every question in this brief:** §12.1 open items, §12.2 research questions (1–16) and the full design-variable questionnaire §12.3 (V1–V64), including the specific questions on **chunk/window length**, **number and choice of body keypoints (8 → fewer/more?)**, **fps (12 vs 15) and mismatch sensitivity**, and any filter beyond HAR/SAM3. An unanswered question must be listed explicitly as unanswered with the reason and what data would resolve it.

### What you have, and what to do if you cannot access something
- **Guaranteed inputs:** (1) this brief; (2) the sample production chunk `IAModel/experiments/training/campaign/SAMPLE_production_chunk_mergedfps.npy` (float32 `[60, 8, 2]`; see §0b.9 — open it and verify the claims made about it). All numbers in the brief were computed from the campaign artifacts and are quoted in the text, so you can reason from the brief alone.
- **Possible inputs:** the repository under `IAModel/experiments/` and the campaign artifacts listed in §13 (paths refer to the owner's machine). If a referenced path/file is not accessible, **say so, rely on the numbers in the brief and mark the corresponding claim as unverified**; never block on it. If you can run code (Python/numpy/torch), do small analyses to verify claims (e.g. the sample `.npy`, the preprocessing functions); if you cannot, say so.
- **Language and audience:** write in **English**. The reader of your response is a non-specialist who will forward it to the owner, so start with a **plain-language executive summary** (what is wrong, what to do first, expected result) and keep technical detail in the sections below. Be concrete (numbers, file names, ordered steps), not generic.
- **Where to write:** `IAModel/experiments/training/campaign/KANVIS_EXPERT_RESPONSE_<YYYYMMDD>.md` (same folder as this brief). If you cannot write there, write the file in the closest writable location and state the path at the end of your reply.

### Final checklist (verify before you finish — your response is incomplete if any item is missing)
- [ ] Executive summary in plain language; then the three prominent sections **DIAGNOSIS (ranked)**, **SOLUTION (single end-to-end plan)** and **ANSWERS index**.
- [ ] Every item of §0b (0b.1–0b.10) answered explicitly, including the chunking design (§0b.7), the `.engine` question (§0b.8), the sample `.npy` analysis (§0b.9) and the Triton preprocessing-parity risk (§0b.10).
- [ ] Every question of §12.1 (open items), §12.2 (1–16) and §12.3 (V1–V64) answered one by one (or explicitly marked unanswered with the reason and what data would resolve it).
- [ ] Every proposal P0–P9 and experiment E0–E10, E-fps, E-replay, E-bg assessed (accept / modify / reject).
- [ ] Fps/temporal-contract decision; expected operating point in production units (FP per 1 000 sessions, event-level recall, alarms/day vs the ≈ 1 FP/day budget).
- [ ] An *Assumptions and information gaps* section (assumption + conditional recommendation + cheapest verification for each). **No questions addressed to the owner or the user anywhere.**
- [ ] The brief itself was not modified.

### NO INTERACTION with the owner — everything must come from this document and the files it references
**You cannot ask the owner (or anyone) anything.** This brief will be handed to you as a one-shot input: the person who passes it to you cannot answer questions during your work; they will only read the response file you produce. Therefore:
- **Never stop to ask for clarification and never leave a deliverable blocked on a question.** Extract everything you need from this document, the repository, the referenced artifacts (§13) and public sources.
- Where information is genuinely missing, **make an explicit, justified assumption** (state it in an *Assumptions* section), and give a **conditional recommendation** ("if X is true do A, otherwise B"), plus the cheapest check the owner's team could run afterwards to determine which branch applies. Items marked "open" in §12.1 are exactly such cases: they are *research questions for you to resolve from the evidence or answer conditionally*, **not** questions for the owner.
- The final "questions" section of your response (see the structure below) must therefore be a list of **information gaps + the assumption you made + the checks to run**, not requests for a reply.
- If something in the brief seems contradictory or wrong, say so in your response and state which reading you adopted.

### Mandatory output rule — DO NOT edit this document
- **This file is the *input brief* and must remain unchanged.**
- **Write ALL of your answers, analyses, proposals, decisions, risk assessments, experiment plans and any information gaps (with the assumptions you adopted) in a NEW, separate Markdown file**, written **in English**, e.g. `IAModel/experiments/training/campaign/KANVIS_EXPERT_RESPONSE_<YYYYMMDD>.md` (same folder as this brief). If you produce several deliverables (e.g., plans, scripts' documentation), each must be a separate `.md` linked from that main response file.
- Suggested structure of the response file:
  1. **Executive summary** (≤ 1 page): your diagnosis of why production FP (≈ 20–30 per 1 000) ≫ validation FP (≈ 7 per 1 000), the top 5 actions, and the realistic achievable operating point.
  1b. **DIAGNOSIS section** (ranked root causes, evidence, confidence), **SOLUTION section** (single prioritized end-to-end plan, expected operating point) and **ANSWERS section index** (pointing to every answered question) — these three must immediately follow the executive summary (see "The three things the owner needs above all").
  2. **Critical review of the brief** (what you agree/disagree with, what is unverified, errors found).
  3. **Answers to each open question** (§12.1 Q-fps1/Q-fps2, Q-G…, §12.2 items 1–16), one by one, with evidence.
  3b. **Answers to the design-variable questionnaire (§12.3, V1…V64)**, one by one: your recommended value/choice, rationale, expected effect on FP/recall/stability, how to test it, and priority.
  4. **Assessment of each proposal P0–P9 and experiment E0–E10/E-fps/E-replay/E-bg**: accept / modify / reject, priority, cost, expected FP/recall effect, validation protocol.
  5. **Your additional proposals** (new ideas, alternative architectures, filters, data strategies, evaluation designs).
  6. **Recommended roadmap**: Phase 1 (stage-1 model) and Phase 2 (HAR/SAM3/alternatives) with milestones and acceptance criteria in production-comparable units (FP per 1 000 sessions, event-level recall, alarms/day).
  7. **Fps/temporal-contract decision** (12 Hz fixed vs 15 Hz; effect of a 12↔15 mismatch; migration plan).
  8. **Information gaps, assumptions adopted and checks to run** (concise, numbered; the owner will *not* answer you — for each gap state the assumption, the conditional recommendation and the cheapest verification). **Do not address questions to the owner.**
  9. **Appendix**: commands/scripts used, files inspected, numbers recomputed.
- Keep claims traceable: cite file paths, line numbers, artifact names and exact numbers; clearly separate *measured* results from *estimates* and *opinions*.

---

## 0. TL;DR (read this first)

1. **Goal.** Real-time retail shoplifting detection from YOLO-pose skeleton sequences. Business priority: **minimize false positives (FP) first**, keep recall acceptable (target recall ≥ 60 %, FP ≤ 0.01 % — never reached; best is ~0.68 % FP at ~64 % recall).
2. **Current best (validation, clip level):** ensemble of two sequence models (`tcn_lstm` #41 + `tcn_gru` #48), rule **AND @ 0.80** → F1 76.3 %, recall 63.8 %, precision 94.9 %, **11 FP / 1 619 negatives (0.68 %)**, **117 FN / 323 robberies**. The August campaign produced the same operating point with a different pair (11|50 AND @ 0.68: 11 FP, 117 FN).
3. **The FP–recall frontier has not moved with more data or more models** (identical in Aug and Oct; July pipeline with verifier + kinematic heuristics sat on the same frontier). Roughly: FP ≤ 3 → recall ≈ 42–48 %; FP ≤ 5 → ≈ 49 %; FP ≤ 8 → ≈ 58 %; FP ≤ 11 → ≈ 64 %; FP ≤ 20 → ≈ 72–74 %; FP ≤ 45 → ≈ 83–85 %. This strongly suggests we are **information-limited and data/label-limited, not architecture-limited**.
4. **The 3 000 "extra clips" added in October contained only +18 robbery clips** (2 574 → 2 592) and ~+3 650 negatives. This explains why FN did not improve (117 → 117).
5. **Critical audit findings (details in §7–§8):**
   - **(A) Evaluation bug/caveat:** the Oct validation only evaluated the *old* data root (1 942 of 2 376 val clips); the **434 val clips from the new root were silently excluded** (`evaluate_validation.build_split_examples` reads `plan["data_root"]`, a single root, not `plan["data_roots"]`). Training did use both roots.
   - **(B) Robbery data provenance (corrected after owner answers):** robbery clips = 1 647 `action-recordings` (staged by the team, 63.5 %) + 876 curated cuts from **real sessions** (Dec 2025, 33.8 %) + 69 clips from continuous production sessions (DatosOnline `clip_buffer`, 2.7 %); i.e. ≈ 945 (36.5 %) real-session robberies. Recall is similar for staged (34 % FN) and curated-real (35 % FN) clips, so there is *no evidence that staged robberies are easier*, but the 8 val robberies coming from the continuous-session pipeline were all missed (5 by all 20 models; n = 8, inconclusive). Negatives are mostly real sessions; FP rate is 0.55 % in real sessions vs 1.6 % in `action-recordings`.
   - **(C) Systematic errors, not random:** all 11 FP are also FP in ≥ 5 of 20 independent models (one in 20/20). 72 robbery clips are missed by ≥ 10/20 models; 8 by all 20. Ensembling cannot fix this → suspect **label noise / ambiguous clips / missing information** (hands/objects).
   - **(D) Train/inference mismatch candidates:** training clips are *action-centered segments* (start/stop pressed by the labeler, ≈3–10 s) whereas production scores *arbitrary* 8 s slices of a person's presence (transitions, partial actions, mixed actions never seen as training negatives); padding-by-repetition of short clips and variable-length uniform resampling (time scale depends on clip duration); fps handling (cameras 15 fps, ≥ 12 required); single-person, clean-skeleton training clips vs crowded aisles with **tracker ID switches / skeleton jumps** in production; pose engine: training `.npy` came from a `yolo26s-pose.engine`, production uses a *different* `.engine` file built on the same machine (§0b.8 — risk to be quantified); **production inference on Triton may not apply `valid_mask` (frames with < 5 visible keypoints kept, NaN→0 distorting centering/scale — §0b.10)**; production chunks are *up to* 8 s and variable-length (§0b.9); per-frame centering that removes global motion/position.
   - **(D2) Label semantics vs. what pose can see:** category 6 "takes a product and **hides** it" is nearly identical, in 8-keypoint upper-body kinematics, to 14 "**puts an object in a bag**", 3 "puts product in basket/cart/bag" and 4 "picks a product and carries it". The difference is *which object* (store product vs own belongings) and *context/zone*, not arm pose. This explains why cat 14 (3/25 = 12 % FP) is the persistent FP source in both campaigns.
   - **(E) Metric ≠ production:** 16.6 % robbery prevalence in val vs ~0.5 % in real sessions; one FP per clip becomes many chances per session (8 s chunks + overlapping windows). At 0.5 % prevalence the same detector would have **precision ≈ 32 %** (≈ 9 % at 0.1 %).
   - **(F) Selection bias:** best configs are picked among ~20 000 ensemble/threshold combinations on ~1 900 val clips with only 11 FP (Wilson 95 % CI for the FP rate: **0.38–1.21 %**; for recall: **58.4–68.8 %**). Differences of 1–2 FP between "best" configs are noise. The `test` split has never been used (confirmed).
   - **(G) Production FP level ≫ validation FP level.** The owner observes **≈ 20–30 model FP per 1 000 sessions (20 in 654 sessions)** in production (20 FP in 654 sessions in one day ≈ 3.1 % of sessions, model alone), vs 0.68 % (≈ 7 per 1 000 clips) in validation: a **3–4× gap** that is the primary thing to close. The current val numbers already match the business "≈10 per 1 000" stage-1 goal at ~64–66 % recall; production does not reach them (see §8.D, §8.E).
6. **Top proposals (ranked, §10):** (1) fix evaluation and build a **group-aware, source-stratified, real-session-first benchmark** with a session-level metric; (2) **label audit + hard-negative loop** using production (HAR/SAM3 verdicts, p1.json) — most FP are consistent across models; (3) **learned fusion (C3) trained on production sessions** instead of rule-based gating; (4) **shorter windows (≈3 s) with multi-instance/event-localization training + temporal aggregation** (answer to the chunk-size question: *probably helpful, but only if training changes too*); (5) **richer inputs**: global/context channels (walking vs standing, position, scale), head/legs, hand ROI/object cues; (6) **calibration + abstention (uncertainty) + conformal thresholds**; (7) skeleton-native architectures not yet evaluated (ST-GCN/CTR-GCN family, PoseC3D, pretrained motion encoders) — *expected modest gains unless 1–5 are done*.

---

## 0b. Structural facts highlighted by the owner (must be addressed explicitly in your response)

### 0b.1 Training = one `.npy` per labeled *action* (variable length); production = fixed 8 s *chunks*
- **Training:** the labeling tool cuts **one action of one person** (start/stop) → one `.npy` of **variable duration (e.g. 3 s, 4 s, 11 s)**, labeled with a single category. The action boundaries are chosen by the labeler, so the clip is *centered on the action*.
- **Production:** each session is cut into **fixed 8 s chunks** (plus N/N+1 overlapping windows) regardless of what the person does. A chunk may contain: only a fragment of an action, the end of one action and the start of another, standing/turning, or several actions.
- **Owner's example of why this matters:** the owner watches a person walking normally, who then stops, and the model raises "robbery" in one specific 8 s chunk. Likewise: a 6 s robbery movement split into two 3 s pieces — **the first 3 s may be only "looking at a product"**, which, seen in isolation, the model may well score as robbery (because during training *the whole clip was labeled robbery*, including its innocent first seconds). The label is weak at the sub-clip level: **"robbery clip" ≠ "every second of the clip is robbery-like"**, while production evaluates sub-clip pieces.
- **Question to the expert:** is this training/production mismatch a primary cause of the high production FP? (Our hypothesis: **yes, likely the main one**, together with ID switches and multiple looks per session.) **Which solution do you propose?** Candidate solutions to evaluate (not exhaustive):
  1. **Train on the production geometry:** build *fixed-length windows* (8 s, and/or multi-scale 3–4 s) from the **continuous per-person session streams** using the labeled intervals as supervision (frame-level labels inherited from the action interval; window label from overlap rules), with the *surrounding real context* instead of padding; random window offsets; same stride/overlap as production.
  2. **Weak-label handling inside clips:** multiple-instance learning (a clip is positive if *some* sub-window is robbery-like), temporal localization of the "conceal/take" sub-event (±1 s), or two labels per positive clip (*core* vs. *peripheral* frames); down-weight peripheral windows or treat them as "ambiguous/ignore" rather than as hard positives or negatives.
  3. **Teach the model what non-action windows look like:** windows between labeled actions, partial actions at chunk edges, transitions, and *pre-robbery* behavior (looking at a product) as **negatives or ambiguous**, so that "looks at product" alone does not trigger.
  4. **Temporal-ordering/state logic** on top of window scores (reach → conceal across consecutive windows) instead of single-window decisions.
  5. **Shorter window + aggregation** (§10 P2) combined with 1–3 so that the model scores sub-events and the aggregator requires the right sequence.
  6. Alternative: **keep clip-style training but change inference** — cut production sessions into *action-like segments* with a segmentation/change-point detector (motion onset/offset, stop-and-reach detection) and feed variable-length action segments to the model (closer to training). Compare against 1–5.
- **Owner answer (round 5, Q-L):** the training data are **labeled videos in which one person performs one action; from that person the owner extracts a *single* `.npy` (whatever its duration) and the whole `.npy` gets the category of the action.** There are therefore **no continuous per-person streams with labeled start/end frames** in the training set: the supervision unit is "one person-video = one `.npy` = one label", with no temporal annotation inside it. (Curated real-session cuts follow the same convention: one cut = one `.npy` = one label.) Whether the continuous streams can be regenerated from stored videos was **not stated** — assume it *may* be possible for production videos (the production chunk folders exist) but that no frame-level labels exist.
- Consequence: any solution that needs frame-level intervals must **derive them weakly** (MIL / pseudo-localization / unlabeled-window negatives from production sessions) or **synthesize production-style windows from the action clips** (embedding a labeled clip, or its core part, inside windows built with other sessions' normal motion). The expert must propose a plan that works with **action-level labels only** and say what extra annotation (if any) would pay off most.

### 0b.2 Two pose-file variants are used: `full` vs `filtered` — they are (almost) the *same data*
Two variants of the pose `.npy` exist and are trained as separate campaign cells (`*_full*` vs `*_filtered*`):
- **`full`** = `poses_full.npy` `[T, 8, 2]` with **all** frames of the track (NaN where keypoint confidence ≤ 0.5), used together with `valid_mask.npy` (frames with ≥ 5 visible keypoints); in `PoseDataset` the frames with `valid_mask == False` are **dropped** (`poses[valid_mask]`) and remaining NaN → 0.
- **`filtered`** = `poses.npy` `[T', 8, 2]`, which the extractor writes **only for frames that pass `_frame_usable`** (not all-NaN and ≥ 5 visible keypoints, i.e. exactly `valid_mask == True`); remaining NaN → 0 in the dataset.
- **Verified from code** (`pose_extractor_clean.py` L131–147; `train_model_operations.py` L1500–1530, L1710–1715): `valid_mask = visible_per_frame ≥ BODY_VISIBLE_MIN_KPS(5)` and `poses.npy` = frames satisfying that same criterion. Therefore **the model input tensors for `full` and `filtered` should be identical, frame by frame** (same example counts in the plans: 19 043 in Oct, 15 370 in Aug). The only differences would be edge cases (missing `valid_mask.npy`, rounding), to be verified by comparing tensors.
- **Production uses `poses_full.npy`** ("pilot bin_full"; C2 runs the motion slots on `poses_full.npy`) — note that **in production the chunk's `poses_full.npy` may include invalid frames** unless the same `valid_mask` dropping is applied at inference: **[verify train/inference parity, V51]**.
- **Implication:** `full` vs `filtered` cells are essentially *replicate trainings of the same data/split* with different random seeds/sampling order. *(computed)* Same architecture, same data, same split, F1 at `softmax_argmax`: **Oct** difference `full − filtered` per model: mean −0.4, **sd 3.9, max |Δ| 8.8 F1 points**; FP-rate difference sd 1.1 pt, max 2.8 pt. **Aug**: mean −1.7, **sd 5.3, max |Δ| 19.2**; FP-rate difference sd 1.5 pt, max 3.4 pt. If the two runs are exchangeable, the *per-run* sd is ≈ diff-sd/√2 ≈ **2.8 (Oct) – 3.7 (Aug) F1 points and ≈ 0.8–1.1 pt of absolute FP rate** — i.e. the training noise is **larger than the FP rate we are trying to reduce (0.68 %)** and larger than most differences between architectures (single-model F1 spans ≈ 70–82).
- **Question to the expert:** should we keep both cells, or only one (`full`, as production)? Is the apparent preference (Aug: `filtered`; Oct: `full`) just noise (our reading)? Are there real differences (e.g. missing `valid_mask`, NaN handling, augmentation seeds)?

### 0b.3 Why the winning architectures/cells change between campaigns (owner question)
With *the same 20–29 architectures* and +3 673 clips (≈ +2 000 in the owner's words), the winners changed completely (Aug: 11|50 on `bin_filtered_hardened`; Oct: 41|48 on `bin_full`). Our explanation (hypothesis, to be tested by the expert): (i) **run-to-run noise** as quantified above (sd ≈ 3–4 F1 points and ≈ 1 pt FP-rate for the *same* model/data); (ii) a **new random split** each time (val 1 917 vs 1 942 clips; only 244 of the Oct val clips were in Aug val, 1 451 were in Aug train); (iii) **winner's curse**: the "best" is the arg-max over 20 000–44 000 configurations (cell × rule × pair × threshold) measured on ~11 FP events; (iv) the added data were negatives (+18 robberies), changing calibration/thresholds. **Consequence:** the ranking of architectures is *not reliable*; the expert must propose a protocol to separate signal from noise (multiple seeds/splits, bootstrap CIs, picking by robust criteria), and say how many seeds are needed (V31, V59).

### 0b.4 Frame rate: training clips are 15 fps (confirmed)
- **Training clips are 15 fps** (owner). Cameras deliver 15 fps and the owner can force production `.npy` to 12 fps. Earlier assumption "12 fps training" was wrong: **the models are currently trained at 15 fps**, so a 12 fps production `.npy` would be a *mismatch today*. **Owner answer (round 4, Q-J):** the production *video* runs at **15 fps**, but **how many frames a chunk `.npy` contains depends on how long the person is on screen** (see the sample analysed in §0b.9: 60 frames = 4 s at 15 fps, not 120). The sample `session_meta.json` says `pose_fps: 12.0` and `frontend_playback.json` says `15.0`; the exact fps of the production `.npy` timeline must be derived by you from `pose_frame_timestamps.npy` / `session_meta.json` in the sample production folders (see §3.2 and §13) — treat it as 15 fps unless the timestamps show otherwise.
- **Temporal compression effect (hypothesis, verify in C2 code):** at 15 fps, an 8 s chunk = **120 frames**. If production applies the same `temporal_resize` as training, then for `seq_len = 64` models (e.g. **#41**) the chunk is subsampled **×1.9** and for `seq_len = 96` models (**#48**) **×1.25**, whereas a typical 4 s training clip (60 frames) is *not* subsampled (and is padded with repeated frames). **A 3 s theft inside an 8 s chunk therefore appears up to 1.9× faster to model #41 than in training** (the training speed augmentation covers only 0.85–1.15). This is an additional production-specific distortion that the validation set (clip-level) never exercises.
- **Question to the expert:** which fps contract (15 native vs 12) and which window/`seq_len` contract removes this distortion with minimal retraining? See §10 P0.6 (updated).

### 0b.5 Production FP vs validation FP — what was actually observed
- **Validation:** 11 FP in ≈ 1 500–1 600 val negatives (clip level; 0.68 %, Wilson 95 % CI 0.38–1.21 %).
- **Production (owner, 2026-10-13):** **20 FP in 654 sessions in one day (3.06 % of sessions)**, *model alone* (HAR/SAM3 only run on candidates, so these are stage-1 alarms before filtering).
- **Reading (hypothesis):** these are **different units**. Validation is one prediction per clip; production evaluates **every chunk and every N/N+1 window of a session**. If a session gets on average *k* independent looks, P(session FP) = 1 − (1 − 0.0068)^k; **3.06 % corresponds to k ≈ 4.6 looks per session** — a plausible number of 8 s chunks + overlap windows per session. So **a large part of the "gap" may be arithmetic (multiple looks), not a bug**; the remaining part comes from the distribution mismatches in 0b.1, §8.D and ID switches. The expert must test this (E-replay) and report the *session-level* FP rate expected from the *chunk-level* FP rate, and what chunk-level FP rate is therefore required to reach ≈ 1 FP/day on ≈ 1 000 sessions (≈ 0.1 % session-level ⇒ chunk-level FP ≈ 0.02 % for k ≈ 4.6).
- **Owner answer (round 4, Q-K) supports this reading:** a session = from entering to leaving the camera; crossing an aisle = 1 chunk (8 s); browsing = ~4 chunks (32 s); an employee restocking = minutes = **~43 chunks**. So the mean of ≈ 4–5 looks per session is plausible, but the distribution is **heavy-tailed**: a handful of long sessions (employees, long browsing) can contribute many looks each, so the FP rate *per session* is strongly duration-dependent. The expert must check whether the 20 FP are concentrated in long sessions / employees (§0b.7).
- The owner considers the validation count (11) as casuistic/variable (consistent with the CI); the production number is the real measurement.

### 0b.6 Generalist requirement: no per-action fine tuning
The owner does **not** have labeled examples of benign everyday actions such as *phone-to-pocket* or *hand-on-back*, and does **not** want to retrain action by action (scratching the head, scratching a foot, laughing, adjusting hair, etc. — "it would be crazy"). The solution must be **general**: it must make the model reject the *whole space of everyday movements* without enumerating them. Directions the expert should evaluate: treating all unlabeled production sessions (robbery ≈ 0.5 %) as negatives (PU learning); open-set/one-vs-everything formulations, outlier exposure and OOD/energy scores; self-supervised pretraining on large unlabeled streams to model "normal" motion; prototype/metric learning with a *robbery* prototype and large diversity of background; pseudo-labeling by HAR/SAM3 verdicts and automatic hard-negative harvesting from stage-1 alarms (no manual per-action labeling); uncertainty-based abstention; and temporal-ordering constraints (reach → conceal) that everyday gestures rarely satisfy.

### 0b.7 Production chunking: is "independent fixed 8 s chunks per session" the right design? (owner question, round 4)
**Facts (owner).** A *session* = one tracked person from entering to leaving the camera view. A session is cut into **consecutive fixed chunks of 8 s**, and the stage-1 model scores **each chunk independently** (in addition to the N/N+1 window logic described in §2). Typical sessions: crossing an aisle = **1 chunk (8 s)**; looking at products = **~4 chunks (32 s)**; an employee restocking = several minutes = **~43 chunks**.
**Why the owner chose chunks.** The alternative would be to write *one* `.npy` for the whole session at the end (equivalent to concatenating the chunks). With chunks, if a session lasts 43 s and the theft happens at second 17, the system does not need to wait for the end of the session: the model runs per chunk and *in theory* fires when the third chunk is processed (second 24). The owner explicitly asks the expert: **is this chunking correct, and what is the optimal way to do it?**
**Aspects the expert must analyse and decide (commit to a design):**
1. **Latency vs context:** what is the best trade-off between alarm latency (chunk length, stride, overlap) and accuracy? Is a non-overlapping 8 s chunk appropriate, or should the stream be scored with a sliding window (e.g., 3–4 s stride 1 s) or a **stateful/causal streaming model** (GRU state, causal TCN) that carries context across chunk boundaries?
2. **Boundary effects:** an action that straddles two chunks is split (first part "looking at product", second part "hiding"); N/N+1 overlap partly addresses it. What overlap/stride is optimal, and how should the training windows be generated so that they *mimic exactly this cutting* (ties to §0b.1)?
3. **Multiple looks / exposure:** the number of independent looks grows with session duration (1 vs 4 vs 43 chunks). With chunk-level FP rate *p*, session FP ≈ 1 − (1 − *p*)^k, so long sessions (employees, long browsing) dominate false alarms. What is the right **operating metric** (FP per camera-hour, per person-minute, per session) and what **per-session aggregation** is optimal (alarm only after ≥ 2 consecutive positive chunks / persistence + hysteresis, sequential testing/CUSUM, event de-duplication, max vs mean vs noisy-OR of chunk scores, threshold that depends on elapsed session time)? How much recall does persistence cost, given theft events are short (~3–11 s)?
4. **Short and variable-length chunks:** the last chunk of a session (or a one-chunk crossing) is shorter than 8 s (see §0b.9: 60 frames = 4 s). How does the model behave on partial chunks (padding by repeating the last frame vs subsampling), and should short chunks be handled with a different threshold, be merged with the previous one, or be skipped?
5. **Employees and long sessions:** employees restocking/cleaning (cats 7, 13) generate dozens of chunks per session. Is there a generalist way to treat them (session-level prior from dwell time/zone/uniform ReID, session-duration-aware threshold) without per-action tuning?
6. **Is concatenating the chunks into one session `.npy` better?** Evaluate: (a) end-of-session scoring on the full sequence (needs windowing anyway: models have `seq_len` 32–128 and `temporal_resize` would time-warp long sessions), (b) chunks (current), (c) overlapping sliding windows, (d) causal streaming model. Give a recommendation with expected FP/recall/latency effects and how to test it on the replay benchmark (E-replay).
**Needed to evaluate this:** the production replay of sessions (frame-aligned `chunk_*/poses_full.npy`, `p1.json`) and the distribution of session durations (§12.1). If those are not available to you, give the recommendation conditional on plausible distributions (1/4/43 chunks as quoted above).

### 0b.8 Training `.npy` and production `.npy` were produced by *different `.engine` files* (owner question, round 4)
**Facts (owner).** Training `.npy` files were generated with `yolo26s-pose.engine` (TensorRT), the same *model family* and machine as production, but **the production `.engine` is a different file from the one used for training extraction** (both built on the same machine). The owner asks: **can it matter that it is not exactly the same file?**
**Aspects the expert must analyse (hypotheses to quantify):**
- TensorRT engine builds are **not bit-reproducible**: tactic/kernel selection, precision (FP32/FP16/INT8 + calibration), workspace, batch size, static vs dynamic input shape and `imgsz`, TensorRT/CUDA/driver versions and GPU may differ between builds, which changes keypoint coordinates (sub-pixel jitter) and **confidences**. Since keypoints with conf ≤ 0.5 become NaN, confidence shifts change the **NaN/visibility pattern**, hence `valid_mask`, frame dropping and the time axis fed to the model.
- Do the same weights (`.pt`) and different engines yield statistically different pose streams (jitter, NaN rate, left/right swaps, ID-switch rate with `custom_tracker.yaml`)? Is the training extraction (`.engine` vs `.pt`: §2 states `.pt` was also used in some training extractions) consistent with production?
- Does a small coordinate/confidence shift move the 0.80-AND operating point? (Models are sharp; ensemble AND rule at 0.80 amplifies tail behavior.)
**Questions to answer:** (i) magnitude of the effect and how to measure it (re-extract the same videos with both engines and with `.pt`; compare keypoint displacement, NaN rate, track continuity, and **model-score deltas / alarm flip rate** at 0.80); (ii) a **contract** for pose extraction (pin the exact engine file, builder arguments, precision, `imgsz`, conf/IoU/NMS, tracker config, fps) and a regression test that fails when the production pose stream drifts; (iii) cheap robustness measures if the engines must differ (keypoint-jitter/dropout augmentation calibrated to the measured engine-to-engine difference, confidence-aware inputs, retraining on production-engine extractions).

### 0b.9 Analysis of the production sample `.npy` provided by the owner (`mergedfps.npy`, *computed*)
The owner supplied an example production file, stored next to this brief as **`IAModel/experiments/training/campaign/SAMPLE_production_chunk_mergedfps.npy`** (load it with `np.load`; inspect it yourself and verify the numbers below). Findings:
- **Shape `(60, 8, 2)`, float32, normalized image coordinates in [0, 1]** (x ∈ [0.26, 0.37], y ∈ [0.10, 0.49]); 8.1 % NaN. **60 frames = 4.0 s at 15 fps**, i.e. a production "chunk" can be **much shorter than 8 s** when the person is visible only briefly — so the "fixed 8 s chunk" is really **"up to 8 s"**. The model therefore also sees short, padded inputs in production (`temporal_resize` pads by repeating the last frame when T < `seq_len`).
- **Visibility profile:** frames 0–51 have all 8 keypoints; then 6, 5, 4 visible and **frames 55–59 have only 2 keypoints (shoulders)**. The person walks toward the camera and **leaves through the bottom edge**: mean y moves from 0.13 to 0.44 and torso height from 0.054 to 0.136 (**≈ 2.5× larger skeleton in ~3.5 s**), then the lower body disappears.
- **Frozen frames:** frames **55–59 are bit-identical** (zero displacement) — the last detection appears to be **held/repeated** rather than missing. The training extraction does not produce frozen frames (conf ≤ 0.5 → NaN). Frames with < 5 visible keypoints are removed by `valid_mask` for `full` models, but only if the production path applies that mask (open item Q-M; if not applied, a frozen pose tail looks like "standing still").
- **Perspective scale change:** `normalize_sequence` uses per-frame centering and a **global scale**, so a person approaching the camera (skeleton ×2.5 in size) yields apparent *limb-spread/velocity* changes that are really perspective, not gesture. Hypothesis: approach/exit trajectories (typical of aisle crossings, exactly the 1-chunk sessions) create spurious "arm" motion that can trigger FP; check per-frame scale normalization and perspective augmentation.
- The file carries **no timestamps**, so the fps of the underlying timeline cannot be derived from it alone (use `pose_frame_timestamps.npy` of the production chunk folders).
**Request to the expert:** use this sample as a concrete illustration of how production chunks differ from training clips (variable duration, edge exits, frozen tails, strong perspective scale change), and propose how to simulate these in training (windows from continuous streams, edge-exit/truncation augmentation, per-frame scale normalization, handling of frozen/held frames).

### 0b.10 How the model is actually fed in production: Triton inference server (owner, round 5) — and what it implies
**What the owner's inference-side agent described (verbatim summary).** At low level (replica of `test_model2` / Triton PyTorch backend, not the TensorRT-only path) the `.npy` is a `float32 [T, J, 2]` sequence (xy per keypoint; pilot J = 8), loaded with `np.load` (`load_poses_array`: mode `full` → `poses_full.npy`, mode `filtered` → `poses.npy`; **NaN → 0**). In Triton, `iamodel_inference._infer_logits_from_poses_tj2` can simulate an IAModel clip: it creates a temporary dir `user_0/`, `np.save(user_0/poses.npy, …)` (+ `poses_full.npy` if applicable), writes a `meta.json` with `track_id` and `total_frames` and optionally copies the array to a temporary `.npy` (`MOTION_USE_TEMP_NPY=true`). Then there are **two preprocessing paths** to the `.pt`:
- **(A)** `MOTION_SIMPLE_PREPROCESS=false` and IAModel importable → `PoseExample(pose_path=…)` + `build_pose_dataset_for_eval(…, seq_len of the checkpoint)` → tensor `x [1, seq_len, F]` with the same logic as training/eval.
- **(B)** fallback / simple preprocess → `tensor_simple_pipeline`: per-frame centering and global scaling (`normalize_sequence`), concatenate velocities (`add_velocity` → `[T, J, 4]`), resample to the checkpoint's `seq_len` (`temporal_resize`: uniform subsample or pad with the last frame), reshape to `[1, seq_len, J×4]` (e.g. F = 32).
Then `model.eval()`, `logits = model(x)[0]`, `p_robbery` = softmax over the robbery class per `checkpoint['label_to_idx']`/`positive_class` (binary: index of label 1). With TensorRT the `.pt` does not run a hot forward: the same `tensor_simple_pipeline` produces `[B, seq_len, F]` and the `.engine` returns logits; score semantics are identical.
**Verified in the repository (*verified*, `train_model_operations.py`):** `normalize_sequence` = per-frame mean over joints as the center, **one global scale = mean joint-to-center distance over *all frames and joints*** (lines ~1600–1611); `add_velocity` = `diff` with `prepend` (so velocity is computed *before* resize, hence it is in units of per-original-frame displacement); `temporal_resize` = `linspace` subsample if T > `seq_len`, repeat-last-frame padding if T < `seq_len` (~1622–1636); in training, `valid_mask.npy` (if present next to the file) **drops invalid frames**, then NaN → 0 (`__getitem__`, ~1710–1715); `INFERENCE_ENSEMBLE_HANDOFF.md` prescribes "if `valid_mask.npy` exists, filter invalid frames".
**Consequences and risks the expert must evaluate (hypotheses, with the evidence available):**
1. **Missing `valid_mask` in the simulated clip.** The description of the Triton path lists `poses.npy`, `poses_full.npy` and `meta.json` but **no `valid_mask.npy`**. If it is not written, the `full` models receive **all frames, including those with < 5 visible keypoints**, with absent keypoints set to 0 — while training dropped those frames. Zero-filled joints sit at the image origin (0,0), far from the person (≈ (0.3, 0.4) in the production sample), so they **distort the per-frame center and the global scale** and create huge spurious "motion"/velocity. This matters especially in exactly the cases of the production sample (§0b.9): person exiting the frame (2 visible keypoints), frozen tail frames, partial occlusions. *Computed on the sample (`SAMPLE_production_chunk_mergedfps.npy`, NaN→0, repository `normalize_sequence`/`add_velocity`):* keeping all 60 frames instead of dropping the 6 frames with < 5 visible keypoints changes the global scale from 0.0504 to 0.0660 (+31 %), changes the normalized coordinates of the 52 perfectly good frames by 0.13 body-scale units on average, and puts the 2-keypoint frames at normalized coordinates ≈ (4.0, 4.7) (a normal frame lies within ≈ ±1.5), with a maximum |velocity| of ≈ 6.7 on a clip that has none of that motion. This is the *size* of the distortion for one exiting person; whether it shifts the scores is for you to test with the real models. *Verify:* read `iamodel_inference` (if available) / replicate path (A) and (B) on `SAMPLE_production_chunk_mergedfps.npy` with and without `valid_mask`, and compare the resulting scores and tensors.
2. **Which path is live? (A vs B)** The two paths should be numerically identical but depend on an environment flag and on whether IAModel is importable inside the Triton container; the production behavior may silently differ from the validation harness (`evaluate_campaign`, which uses `build_pose_dataset_for_eval`). *Verify:* feed the same chunk through both paths and compare tensors/scores; propose a unit test (golden tensors) that runs in CI/at server start.
3. **Training zero-fill is also imperfect.** Even in training, frames with ≥ 5 visible keypoints but 1–3 NaN→0 joints distort center/scale. A cleaner contract (fill NaN joints by interpolation / use masked mean for the center and scale / add visibility channels) is a candidate improvement that costs little.
4. **Time axis.** `temporal_resize` uses *frame indices*, not timestamps: dropped/duplicated frames, a variable number of frames per chunk and fps changes all alter the effective time scale (§0b.4). A fixed-Hz resampling on timestamps before `temporal_resize` would make the contract explicit.
5. **Same `p_robbery` semantics** across Triton `.pt`, TensorRT and the campaign evaluation: therefore threshold 0.80 and the AND rule are comparable *only if* the preprocessing is identical. Ask: does production use `poses_full.npy` or `poses.npy` per model (the handoff says `full` for `bin_full` models)?
**Request to the expert:** include in the diagnosis a ranked assessment of "preprocessing mismatch in Triton" (missing mask, path A/B, zero-fill, resize on indices) and a minimal-effort **golden-tensor test** that proves training ≡ production preprocessing; propose the corrected contract.

---

## 1. Product and problem definition

**Kanvis** detects shoplifting in real time in retail stores from camera video.

- Each *labeled* training video contains **one person performing one action**. Labeling is done with an in-house tool: the labeler picks **one action of one person** (start/end), assigns a category, and the pose `.npy` of that person is then extracted for that segment. Category definitions (**confirmed by owner**; `-1` = uncategorized/pending, never chosen when labeling):

| id | Definition (translated) | Used in training? | Clips (Oct, all splits) |
|---:|---|---|---:|
| 0 | Customer walks through the aisle without stopping to look or buy | yes | 3 935 |
| 1 | Walks through the aisle, stops to look, does not buy | yes | 578 |
| 2 | Walks, stops, picks a product, looks at it and **puts it back** | yes | 1 733 |
| 3 | Walks, picks a product and **puts it in the basket/cart/bag** (buys) | yes | 1 766 |
| 4 | Walks, picks a product and **carries it on top** (hands/arms; buys by carrying) | yes | 2 119 |
| 5 | Walks through the aisle without stopping, **but carrying objects** | yes | 4 824 |
| **6** | **Customer picks a product and hides it (robbery)** | yes (**positive class**) | **2 592** |
| 7 | An employee is restocking | yes | 458 |
| 8 | The person who picks the product and the one who pays are not the same | no clips | 0 |
| 9 | Walks, stops to look, does not buy, **but carries objects** | yes | 760 |
| 10 | Employee-to-employee interaction (passing objects) | no clips | 0 |
| 11 | Customer consumes a product and abandons it at some point | no clips | 0 |
| 12 | Customer consumes a product and carries it the whole time | no clips | 0 |
| 13 | Employee cleaning | yes (Oct only) | 59 |
| 14 | **Putting an object in a bag** | yes | 219 |

  Semantic closeness to robbery (6): **14, 3, 4, 2** (hand-to-body/bag/cart motions) and **5/9** (customer already carries objects, so hands near the body are normal). Categories 8, 10, 11, 12 cannot be captured by single-person pose clips or have no data.
- Videos are converted with **YOLO-pose** into keypoint sequences saved as `.npy`. Typical clip length is **5–10 s** (some longer).
- Several sequence models with different architectures were trained and are combined.
- **Real time:** a *session* = one person entering/leaving the camera field. The session is cut into **8 s chunks**, one `.npy` per chunk. The model is applied **to each chunk and to a sliding window overlapping two chunks (N + N+1)**.
- **Main problem:** *too many unstable false positives*. Example: a person walks with a hand on their back → alarm; another person doing a similar gesture → no alarm. Validation numbers look good but production behavior is inconsistent. A previous advisory agent suggested the training procedure itself might be flawed. (This audit finds several concrete candidates: §8.)

### Objectives

| # | Objective | Metric (proposed definition) |
|---|-----------|------------------------------|
| O1 | **Drive false positives to practically zero** after the full pipeline (motion → SAM3/HAR → C3). **Owner budget: ≈ 1 false alarm per day on ≈ 1 000 sessions/day (≈ 0.1 % of sessions), and ideally that residual FP is a *plausible near-miss* (e.g. a person putting their phone in a pocket), not an arbitrary movement.** "~20 FP/day is unacceptable." | *Alarms per day* and *% of non-robbery sessions raising an alarm* (session-level, replayed on stored production sessions); FP severity tiers: *near-robbery gesture (tolerable)* vs *unrelated movement (not tolerable)* |
| O2 | **Detect ~2 of every 3 (66 %), ideally 3 of every 4 (75 %) robberies** (event level). *Losing an individual robbery is acceptable; FP are not.* | *% of robbery events with ≥ 1 alarm during the event* (several chunks/windows per event), plus clip-level recall as secondary |
| O1a | **Phase 1 – stage-1 model:** ≤ ≈ 10 FP per 1 000 clips in production (today 20–30) at ≥ ≈ 66 % recall | FP/1 000 on replayed production sessions; recall on labeled robberies |
| O1b | **Phase 2 – HAR/SAM3 (and alternative filters):** remove all or almost all of the ≈ 10 remaining FP (≥ 90 % rejection) at ≤ ≈ 10 % robbery loss | Rejection rate of FP candidates; recall retention on true candidates (needs HAR/SAM3 run on labeled val/test candidates) |
| O3 | Make behavior **stable** (same gesture → same decision; no calibration drift between retrainings) | Variance of decisions across seeds/retrainings; stability under small perturbations of the same clip; PR-curve/ECE consistency across data splits |
| O4 | Reduce dependence on val-set luck | Group-aware CV with CIs; report CIs for FP and recall |
| O5 | Faster iteration | Fewer, better-chosen architectures; evaluation harness that includes the *whole* pipeline |

---

## 2. Production pipeline (as described by the owner + repository evidence)

```
camera → YOLO-pose + tracker → per-session 8 s chunks (poses_full.npy etc.)
      → (stage 1 "motion") ensemble on chunk N and on overlap window N+N+1
            robbery_score = min(P_model1, P_model2)   [AND rule @ 0.80]
            robbery_alarm = candidate
      → only if candidate:
            C2 gateway: SAM3 review (ApplySam3Review)  — object-state check
            C3 engine : HAR (gesture rules over chunk poses)
      → C3 fuses motion + SAM3 + HAR → verdict written in the anchor chunk of p1.json
         (high motion alone never closes a robbery)
```

- **SAM3 (second pass in C2)** — ROI/SAM tracking infers the *object state at the end of the span* (usually N+N+1): `in hand`, `visible`, `in cart`, `hidden near body`, or `skipped/no_hands_in_roi`. Writes `roi_fp_review` into `p1.json`. **Has priority over HAR when ok.**
- **HAR (C3)** — reads chunk poses, detects gestures (wrist at waist, arm retraction), summarizes into `har[]`, `har_slide[]`, `har_robbery`. Used to confirm/nuance when SAM sees body/concealment, or **alone when SAM is not applicable** (person with back to camera, hands not visible).
- **C3 fusion** writes the verdict (e.g., *false positive*) in the anchor chunk of `p1.json`.
- **Stage-1 motion ensemble (confirmed by owner, "pilot bin_full"):** C2 runs two motion slots on `poses_full.npy`: **`p1 = model.pt`, `p2 = model2.pt`** (= experiments 41 and 48 in the Oct campaign; exact slot→experiment mapping to be kept in the integration notes). Each returns `p_robo = softmax(class robbery, logit1)`. With `C2_MOTION_ENSEMBLE_RULE=and` and `C2_MOTION_ENSEMBLE_THRESHOLD=0.80`: **`robbery_score = min(p1.p_robo, p2.p_robo)`**; `robbery_alarm`, `motion_theft_candidate` and `sam3_review_required` are true **only if both `p_robo ≥ 0.80`**. With the legacy rule `mean`, score = mean of the two and C2 does not set the alarm that way.
- **Sample `p1.json`** (owner's machine `/home/debian/Documentos/Kanvis/p1.json`; the relevant content is summarized here, schema `c2_continuous_v2_p1_1.0`): per window `w_0000` with `t_start/t_end` (8 s), `p1`/`p2` blocks each with `logit0, logit1, gap, p_robo`, and `robbery_score`. In that sample `robbery_score = 0.9924 = p1.p_robo` (≠ min = 0.9355), so it was produced **before** the AND rule was active (legacy output); current files should show `min`.
- **Production pose model (confirmed by owner):** **`yolo26s-pose`, TensorRT `.engine`** (an older sample `session_meta.json` showed `yolo26n-pose.engine`; superseded). **Owner answer (round 4): the training `.npy` were also generated with a `yolo26s-pose.engine`, but a *different engine file* from the production one (same machine)** (`config.py: YOLO_POSE_MODEL`; the `meta.json` field `yolo_backend` records `pt`/`engine` per clip — `.pt` may also appear in some clips, check the distribution). Same weights family ⇒ the shift is engine-build differences (precision, `imgsz`, TensorRT version, tactics) and NMS/tracking details → effect to be quantified (§0b.8, §10 P0.5).
- **Sessions (confirmed):** one session = one person. In an aisle with 4 people there are **4 independent sessions**, and each session produces the `.npy` of *its* person only. **Known production problem: track ID switches** → the skeleton "jumps" to another person or the track fragments; inside a chunk this looks like an abrupt arm/torso motion and can easily trigger the motion model. Training clips never contain such events (single visible person, single clean skeleton).
- **Camera fps:** cameras run at **15 fps**; the pipeline requires **≥ 12 fps**. See §10 P0.6 for the recommendation on fixing fps.
- **Pose fps:** `session_meta.json` says `pose_fps: 12.0`; `frontend_playback.json` says `15.0`; the sample chunk `diag.json` shows 74 frames over 4.03 s (≈ 18 fps) with `synthetic_frames_used` warnings (replay). Training clips use the video's native fps (default 12 when unreadable). **Velocity features are per-frame, so fps mismatches change feature scale.**
- **Tracker:** Ultralytics BoT-SORT with ReID (`custom_tracker.yaml`: `track_buffer: 300`, `new_track_thresh: 0.7`, `with_reid: True`, `gmc_method: sparseOptFlow`). IDs can still switch; the `user_X` folder id is the tracker id.
- **Previously built FP-reduction attempts** (all in the repo): sliding-window 3 s + temporal filters (`sliding_window_eval.py`), robbery-vs-confusable verifier (`bin_verifier_234`, classes 2/3/14), kinematic heuristics H1–H12 (`pose_robbery_heuristics.py`, `clean_theft.py`: reach, conceal, low_pick). Results in §6.4.

---

## 3. Pose data specification (`.npy`)

### 3.1 Keypoints
YOLO-pose gives 17 COCO keypoints; **only 8 upper-body keypoints are kept** (COCO indices 5–12):

| idx in file | COCO idx | name |
|---|---|---|
| 0 | 5 | left_shoulder |
| 1 | 6 | right_shoulder |
| 2 | 7 | left_elbow |
| 3 | 8 | right_elbow |
| 4 | 9 | left_wrist |
| 5 | 10 | right_wrist |
| 6 | 11 | left_hip |
| 7 | 12 | right_hip |

Skeleton edges (COCO ids): `[[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12]]`. **No head/face, no legs/knees/ankles, no hands/fingers, no objects.**

### 3.2 Files per tracked person
Layout produced by `pose_extractor_clean.py`: `data_result/{category}/{clip_name}/user_{track_id}/…`

| File | Shape / dtype | Meaning |
|------|---------------|---------|
| `poses_full.npy` | `[T, 8, 2]` float32 | **All** frames of the track. `(x, y)` normalized to `[0, 1]` by image width/height (`keypoints.xyn`). **`NaN`** where keypoint confidence ≤ 0.5 (`MIN_KP_CONF`), or when the track is absent in that frame. |
| `poses.npy` | `[T', 8, 2]` | "Filtered": only frames that pass `_frame_usable` (≥ `BODY_VISIBLE_MIN_KPS = 5` visible keypoints). `T' ≤ T`. |
| `keypoint_mask.npy` | `[T, 8]` bool | Which keypoints are visible per frame. |
| `valid_mask.npy` | `[T]` bool | Frame has ≥ 5 visible keypoints. |
| `meta.json` (clip level) | – | `clip_name, fps, frame_count, video_frame_count, min_valid_frames, cat, clip_duration, yolo_model, yolo_backend, users[]`; each user: `track_id, valid_pct, rel, valid_frames, total_frames, poses_full_count, poses_filtered_count, passes_filters, keypoint_confidence_avg, occlusion_ratio, bbox_aspect_ratio, subject_velocity, clothes{top,bottom}, user_cat, is_primary_robber`. |
| `clip.mp4` | video | Clip rescaled to 1080 px height (`CLIP_SCALE_HEIGHT`). |

Multi-person variants used for visualization/real-time exports: `[T, P, 8, 2]` (`P` persons, 2 in samples) and a confidence variant `[..., 3]` (x, y, conf). Real-time per-chunk folders (`chunk_001/`) contain `poses_full.npy`, `poses.npy`, `valid_mask.npy`, `keypoint_mask.npy`, `pose_frame_timestamps.npy` (float64, seconds, per frame) and `diag.json`; the session-level `merged.npy` concatenates chunks (`concat_chunk_npy.py`).

### 3.3 Extraction thresholds (training side)
`MIN_KP_CONF = 0.5`, `BODY_VISIBLE_MIN_KPS = 5`, `RELIABILITY_THR = 0.9`, `MIN_COVERAGE_RATIO = 0.8`, `DEFAULT_FPS = 12`, `SINGLE_USER_ONLY = True` in training (clips with several tracked persons are dropped). **Production** builds one session per person and extracts that person's `.npy`, so a chunk always contains one skeleton, but in crowded aisles that skeleton can be occluded or switched (ID switch). The owner considers multi-person training data desirable but not essential if single-person sessions work very well. Users with too little body visibility are not saved.

### 3.4 What the model actually sees (important)
Implemented in `train_model_operations.py` (`PoseDataset`, `normalize_sequence`, `add_velocity`, `temporal_resize`):

1. Load `poses_full.npy` (or `poses.npy`). For `pose_source="full"`, **frames with `valid_mask == False` are removed** (the time axis is *compressed*, not interpolated); remaining `NaN` → `0.0` (`nan_to_num`) — this is done **before** normalization, so a missing joint becomes (0,0) in image coordinates, i.e. an artifact far from the body. Clips with fewer than `MIN_SEQ_LEN` valid frames are skipped.
2. (Train only) random augmentation, prob 0.65, ≤ 2 ops among mirror / rotate / scale / shift / noise / speed (0.85–1.15).
3. **`normalize_sequence`**: per frame subtract the mean of the 8 joints; divide by the global mean joint-norm (clip-wide scalar). ⇒ **removes global position and scale** (no information about where in the image the person is, how far from the camera, or whether the body is moving through the scene).
4. **`add_velocity`**: append frame-to-frame deltas of the *normalized* coords → per-joint `(x, y, dx, dy)`.
5. **`temporal_resize`** to model `seq_len` ∈ {32, 64, 96, 128}: if `T > seq_len`, uniform subsampling; if `T < seq_len`, **padding by repeating the last frame** (so velocity = 0 in the padded tail). At 12 fps, `seq_len = 96` ≈ 8 s.
6. Flatten to `[seq_len, 8 × 4 = 32]` features per frame.

Consequences: the representation is **purely relative upper-body articulation**, blind to locomotion, scene geometry, objects, hands and head; and **short clips are represented by "motion + static tail"**.

---

## 4. Dataset

### 4.1 Sources and counts
Train/val/test split is **stratified by category, random per clip UID** (`stratified_by_folder_category_uid`, seed 42, 75/12.5/12.5). Only **single-user** clips (`single_user_only=true`) and categories with ≥ 50 clips are used (`drop_below_50` excludes 8, 10, 11, 12).

**Total valid clips per category (all splits)** *(computed from `training_plan.json`)*

| cat | Aug (data root 010826) | Oct (010826 + 061026) | Δ |
|---:|---:|---:|---:|
| 0 | 2 963 | 3 935 | +972 |
| 1 | 485 | 578 | +93 |
| 2 | 1 216 | 1 733 | +517 |
| 3 | 1 197 | 1 766 | +569 |
| 4 | 1 893 | 2 119 | +226 |
| 5 | 3 761 | 4 824 | +1 063 |
| **6 (robbery)** | **2 574** | **2 592** | **+18** |
| 7 | 424 | 458 | +34 |
| 9 | 652 | 760 | +108 |
| 13 | 0 (excluded) | 59 | +59 |
| 14 | 205 | 219 | +14 |
| **Total** | **15 370** | **19 043** | **+3 673** |

Robbery share: 16.7 % (Aug) → 13.6 % (Oct). Categories 8, 10, 11, 12 have no usable clips.

**Oct split sizes (bin_full):** train 14 279 / val 2 376 / test 2 388 (UIDs). Per-category (train/val/test): 0: 2951/491/493; 1: 433/72/73; 2: 1299/216/218; 3: 1324/220/222; 4: 1589/264/266; 5: 3618/603/603; **6: 1944/324/324**; 7: 343/57/58; 9: 570/95/95; 13: 44/7/8; 14: 164/27/28.
**Aug split (bin_full):** train 11 523 / val 1 917 / test 1 930.

**Evaluated validation set actually used:** Aug 1 917 (321 robbery + 1 596 non-robbery); **Oct 1 942 (323 + 1 619)** — i.e. only the old root (see §8.A).

**Recording sources** (parsed from clip names). **Confirmed by owner:** the data are **real sessions of one supermarket**; when a category needs more samples, the team **records actions on purpose** ("action-recordings": press start/end, register that video with a category; several categories). In both cases the clip *is the labeled action segment* of one person, not a continuous stream. The curated Dec-2025 `robos*` clips are also real sessions (confirmed):

| Source tag | Path pattern | What it likely is | Robbery clips | Non-robbery clips |
|---|---|---|---:|---:|
| `action-recordings` | `…kanvis_piloto_v1/kanvis-backend/public/action-recordings/<rec>_<camera>_full_<cat>` | Clips recorded through the product pilot's backend; whole-session "full" clips; per-category labeled (likely scripted/performed actions) | **1 647** | 2 835 |
| `curated_clips_dic2025` | `…videos_output_data_result2_6_robos*/clip…_000000_0000xx_6_clip_<a>_<b>_6` | **Real sessions** (Dec 2025), robbery segments manually cut and labeled (val mean ≈ 4 s; confirmed by owner) | **876** | 0 |
| `pablo_dic2025` | `…Videospablo_3_revisados_12Diciembre2025_…` | Manually reviewed Dec-2025 recordings (negatives only in this set) | 0 | 765 |
| `datos_1` (DatosOnline_N) | `…DatosOnline_N/sessions/cliente-demo/<store>/<camera>/<session>/clip_buffer_<t0>_<t1>_<cat>` | **Real customer sessions**, chunked by the production system | 23 | 6 353 |
| `datos_2` (`VideosRaw_DatosOnline01_08_2026`) | idem | Real sessions (Aug 2026) | 28 | 3 777 |
| `datos_3` (`DatosOnline4`, new root 061026) | idem | Real sessions (Oct 2026 batch) | 18 | 2 721 |

(The 2 835 `action-recordings` negatives include 887 from the new Oct root; the 18 `datos_3` robberies and 2 721 `datos_3` negatives are all new.)

⇒ **Robbery data**: 63.5 % `action-recordings` (staged), 33.8 % curated **real-session** cuts (876), 2.7 % continuous-pipeline real sessions (69) ⇒ ≈ 36.5 % from real sessions. **Negative data** (16 451 clips): 78 % real sessions, 4.7 % reviewed Dec-2025 recordings, 17 % `action-recordings`.
Prevalence of robbery in the *continuous* real-session stream of the dataset ≈ **69 / 12 920 ≈ 0.53 %** (consistent with real-life rarity); in the evaluation set it is **16.6 %** because staged and curated robberies are over-represented.

Stores/cameras (**confirmed**): **one store (`b1597c1f-…`, tenant `cliente-demo`) with 9 cameras** — which matches the 9 camera ids seen in the data; so every "camera" generalization test is *within the same store* and production = same cameras as training (a plus for camera-specific calibration/zone maps; a risk for any deployment to a second store). Camera ids (top): `416a662e` (3 316 clips), `0d84e739` (3 253), `1d133c9d` (2 806), `c0b2d0c2` (2 328), `6a2c7f58` (2 299), `d72d10c9` (1 266), `41dd1e65` (1 133), `ff935914` (607), `13a1322f` (394); robbery clips are almost exclusively on 6 cameras (`416a662e, c0b2d0c2, 41dd1e65, 0d84e739, 1d133c9d, 6a2c7f58`, 264–309 each) while `d72d10c9, ff935914, 13a1322f` have 2–7 robbery clips (they are almost pure negative cameras ⇒ **camera is a strong shortcut feature if any camera-specific pose statistics leak**).

### 4.2 Clip durations *(computed from chunk names; "full" clips have unknown duration)*

| group | chunked n | "full" n | mean | p10 / p50 / p90 (s) |
|---|---:|---:|---:|---|
| robbery (stats only on the 65 DatosOnline-chunked clips; 876 curated clips are cut as `…_<a>_<b>` sub-segments: val mean ≈ 4.2 s) | 65 | 1 651 | 9.4 | 3 / 8 / 16 |
| non-robbery | 10 399 | 5 287 | 11.1 | 4 / 9 / 18 |
| cat 0 | 3 185 | 750 | 8.3 | 4 / 8 / 12 |
| cat 5 | 3 291 | 1 022 | 8.4 | 4 / 8 / 12 |
| cat 7 | 413 | 45 | 29.6 | 4 / 16 / 84 |
| cat 14 | 40 | 179 | 12.8 | 3 / 11 / 25 |

Note the **length distribution differs by class and source** (curated robbery ≈ 4 s vs most negatives 8–10 s) — together with *padding by repetition* (§3.4) this is a possible **length/padding shortcut** (§8.D).

### 4.3 Session leakage check *(computed on Oct plan, DatosOnline chunked clips)*
13 424 sessions; 356 appear in > 1 split. 11.9 % of val clips (213/1 797) and 10.4 % of test clips come from a session that also has clips in train (overlapping chunks `clip_buffer_000003_000006` vs `…000000_000008`). No robbery val clip shares its session with a robbery train clip (0/111), so leakage is mostly on negatives. Splits are **not grouped by session, camera or day**. `action-recordings` clips have no session id (each recording is one clip, performed by team members) — leakage across recordings of the *same performer/day/scene* cannot be ruled out and is likely (same few people repeating the same staged action in the same spots).

---

## 5. Models

### 5.1 Task variants ("cells") evaluated by the campaign
8 cells = {multiclass, binary} × {`full` (poses_full) vs `filtered` (poses.npy)} × {class map / aug profile}:
`mc_full`, `mc_full_merge`, `mc_filtered`, `mc_filtered_merge`, `bin_full`, `bin_full_hardened`, `bin_filtered`, `bin_filtered_hardened`.

- **Multiclass (`mc_*`):** classes 0–14 (minus excluded); robbery = class 6. `merge` variants fold rare classes 8–13 into class 99. Decision for robbery: `argmax==6` or softmax(6) ≥ threshold. **Best MC models give F1 ≈ 83–85 % but FP ≈ 4–5 %** (MC accuracy ≈ 66 %) — unsuitable as the FP-controlled detector.
- **Binary (`bin_*`):** robbery (1) vs everything else (0). P(robbery) = softmax[1] (or logit margin). *"Hardened"* profile = asymmetric loss (γ_neg 4, γ_pos 1), hard-negative mining, target negative:robbery ratio 6.0 (vs 4.0). **Hardening gave no consistent gain** (Oct best is non-hardened `bin_full`).
- `full` vs `filtered`: no consistent winner (Aug: filtered/hardened; Oct: full).

### 5.2 Architectures (20 experiments kept for Oct; 60 exist in the catalog)
All take `[seq_len, 32]` and output logits. Hyper-parameters from `model_config.py`:

| id | arch | key params | seq_len |
|---:|---|---|---:|
| 4 | tcn | 128, drop .3, lr 5e-4, 40 ep | 64 |
| 5 | tcn | 128, drop .3, 50 ep | 96 |
| 6 | tcn | 256, drop .3, lr 3e-4, 60 ep | 96 |
| 8 | tcn | 256, drop .2, 50 ep | 128 |
| 10 | dilated_tcn | 128 ×3 layers, drop .1, lr 1e-3 | 64 |
| 11 | dilated_tcn | 128 ×4, drop .2, 50 ep | 96 |
| 12 | dilated_tcn | 256 ×4, drop .3, lr 3e-4 | 64 |
| 13 | res_tcn | 128, 3 blocks | 96 |
| 25 | transformer | d128, 4 heads, 3 layers | 64 |
| 26 | transformer | d256, 8 heads, 3 layers | 64 |
| 37 | joint_attn | joint-attn (d64, 2 layers) + temporal (d128, 3 layers) | 64 |
| 40 | tcn_lstm | tcn 128×3 → LSTM 128 | 96 |
| **41** | **tcn_lstm** | tcn 256×2 → LSTM 256, drop .2, lr 3e-4, 40 ep, bs 32 | **64** |
| **48** | **tcn_gru** | tcn 128×2 → GRU 128, drop .2, lr 5e-4, 45 ep, bs 64 | **96** |
| 49 | tcn_gru | tcn 128×3 → GRU 128, drop .25, 60 ep | 96 |
| 50 | tcn_gru | tcn 256×2 → GRU 256, drop .3, lr 3e-4 | 64 |
| 53 | conformer_lite | d128, 3 layers, conv-k 9 | 96 |
| 54 | conformer_lite | d256, 8 heads, 3 layers | 96 |
| 56 | ms_tcn | 128, 4 blocks | 96 |
| 57 | ms_tcn | 256, 4 blocks | 96 |

Models are small (1–5 MB `.pt`). Training: Adam (lr per table, optional weight decay), **cross-entropy** in the baseline (asymmetric in `hardened`), `WeightedRandomSampler` for binary balance, on-the-fly augmentation + category-based "augmentation recipe" bank (rare/confusable classes 1, 7, 9, 13, 14), **checkpoint = best `val_acc`** (accuracy is dominated by the negative class and the val set is later reused for thresholds and ensemble selection ⇒ optimistic). One cell takes up to ~430 min.

**Pruned from the Oct campaign** (from the Aug results): pose_cnn2d (33–35), gru (42–44), tcn #7, joint_attn #36, ms_tcn #55, and MC-only weak ones (1, 2, 24, 27, 51).
**Implemented but never evaluated in the Aug/Oct campaigns** (visible in the leaderboards): `lstm` (15–22), `stgcn` (30–32), `gru_attn` (45–47), `gat_tcn` (58–60), `res_tcn` 9/14, `transformer` 23/28/29, `conformer_lite` 52. (Earlier campaigns, e.g. July, are not known to have evaluated them either.)

### 5.3 Decision modes / ensembles
Single model: `softmax_argmax`, `softmax_threshold` (default 0.8), `logit_margin`. Ensembles over pairs/triplets: **AND** (all P ≥ t; `robbery_score = min`), **MEAN** (mean P ≥ t), **cascade** (low/high thresholds 0.4/0.55). Threshold sweep 0.5–0.98 step 0.02. **Production uses AND @ 0.80 of 41 + 48** (integration message already delivered).

---

## 6. Results

All numbers: **validation split, clip level**, positive = robbery.

### 6.1 August 2026 — `campaign_20260805_220606` (val 321 pos / 1 596 neg; 29 distinct models evaluated)

| Selection | cell | models / rule / thr | F1 | Recall | Prec | FP (FP %) | FN |
|---|---|---|---:|---:|---:|---:|---:|
| **Best ensemble (F1 with low FP)** | bin_filtered_hardened | 11 \| 50, AND @ 0.68 | 76.1 | 63.6 | 94.9 | 11 (0.69 %) | 117 |
| Lowest-FP ensemble | bin_full_hardened | 49 \| 50, AND @ 0.74 | 74.0 | 60.4 | – | **9 (0.56 %)** | 127 |
| | bin_full | 48 \| 50, AND @ 0.86 | 74.3 | – | – | 12 | – |
| | bin_filtered | 42 \| 49, AND @ 0.80 | 74.1 | – | – | 14 | – |
| High-F1 operational | bin_full_hardened | 11 \| 50, MEAN @ 0.74 | 77.4 | 66.0 | – | 15 | – |
| Best single model | bin_filtered | modelo_49 argmax | 82.0 | 83.5 | – | 4.07 % | – |
| Highest F1 overall (high-FP regime) | bin_filtered | 37 \| 49, MEAN @ 0.50 | 85.1 | 85.4 | – | 3.07 % | – |
| Best single binary (argmax, `bin_full`, full leaderboard) | | modelo_50 | 81.2 | 80.7 | – | 3.63 % | – |
| Best multiclass single | mc_full_merge | modelo_12, argmax(c6) | 83.0 | – | – | 4.95 % | – |

11 FP by category: {1: 3, 14: 3, 2: 2, 3: 1, 5: 1, 9: 1}.

### 6.2 October 2026 — `campaign_20261006_231053` (val 323 pos / 1 619 neg *evaluated*; 20 experiments)

| Selection | cell | models / rule / thr | F1 | Recall | Prec | FP (FP %) | FN |
|---|---|---|---:|---:|---:|---:|---:|
| **Best ensemble (production candidate)** | bin_full | **41 \| 48, AND @ 0.80** | **76.3** | **63.8** | **94.9** | **11 (0.68 %)** | **117** |
| Lowest-FP ensemble | bin_filtered | 40 \| 41, MEAN @ 0.74 | 74.1 | 60.7 | – | 10 (0.62 %) | 127 |
| | bin_full_hardened | 05 \| 50, MEAN @ 0.74 | 73.7 | – | – | 11 | – |
| | bin_filtered_hardened | 12 \| 48, MEAN @ 0.74 | 74.7 | – | – | 11 | – |
| High-F1 | bin_full_hardened | 25 \| 50, MEAN @ 0.62 | 83.2 | 76.5 | – | 24 | – |
| | bin_full | 48 \| 49, MEAN @ 0.62 | 82.4 | 74.6 | – | 21 | – |
| Absolute max F1 | bin_filtered_hardened | 06 \| 48, MEAN @ 0.50 | 85.4 | 84.8 | – | 45 (2.78 %) | – |
| Best single (low FP) | bin_filtered_hardened | modelo_48, thr 0.8 | 75.1 | 63.2 | – | 0.99 % | – |
| Best single (high F1) | bin_full | modelo_49, thr 0.8 | 81.5 | 75.5 | – | 1.98 % | – |
| Best single at per-model best-F1 threshold (`bin_full`) | | 49 @ 0.52 / 48 @ 0.58 / 11 @ 0.52 | 82.4 / 81.6 / 78.9 | – | – | 44 / 41 / 47 | 66 / 72 / 82 |
| Best multiclass single | mc_filtered_merge | modelo_56 (ms_tcn), argmax(c6) | 84.9 | 88.5 | – | 4.24 % | – |

The Oct 11 FP by category: {0: 4, 14: 3, 1: 2, 2: 1, 4: 1}; 117 FN are all robbery (cat 6). The Aug best pair (11|50) re-evaluated in Oct at the Aug config (AND@0.68): F1 69.2, recall 54.2, FP 8, FN 148 (calibration drifts between retrainings); re-tuned MEAN@0.74: F1 75.5, FP 12, FN 120.

### 6.3 Comparison and statistical reading

- Aug vs Oct best: **identical operating point** (11 FP, 117 FN, F1 76.1 vs 76.3). Different pairs/cells win in each campaign because each campaign **retrains from scratch with a new random split and new data**, and the winner is the arg-max over ~20 000–44 000 (cell × rule × pair × threshold) configurations; the top configurations differ by ≤ 1–2 FP. (In Oct only 4 configs with FP ≤ 12 are within 1.5 F1 points of the best; in Aug 3.)
- **Different validation sets:** of the 1 942 Oct-evaluated val clips, 1 451 were in the **Aug train** split, 244 in Aug val, 240 in Aug test. Cross-campaign numbers are **not directly comparable**.
- **Confidence intervals (Wilson 95 %)** *(computed)*: FP 11/1 619 → **0.38–1.21 %**; recall 206/323 → **58.4–68.8 %**.
- **FP–recall frontier** (best recall for a max FP count, over all binary cells/rules/pairs/thresholds) *(computed from `val_ensemble_grid.csv`)*:

| max FP | Oct recall / F1 | Aug recall / F1 |
|---:|---|---|
| 3 | 48.0 % / 64.4 | 42.4 % / 59.3 (FP 2) |
| 5 | 48.6 % / 64.7 | 48.9 % / 65.0 |
| 8 | 57.6 % / 72.0 | 58.9 % / 73.1 (FP 7) |
| 11 | 64.4 % / 76.8 | 63.9 % / 76.4 |
| 15 | 69.0 % / 79.8 (FP 13) | 66.0 % / 77.4 |
| 20 | 74.0 % / 82.1 | 71.7 % / 80.6 |
| 30 | 79.3 % / 84.1 | 75.7 % / 82.0 (FP 29) |
| 45 | 84.8 % / 85.4 | 82.6 % / 84.1 (FP 44) |

  → **+3 673 clips and a re-tuned campaign did not move the frontier.**

- **Targets** (recall ≥ 60 % *and* FP ≤ 0.01 %): never met; `campaign_best_configs_val.csv` is empty for both campaigns. Nearest: 0.56–0.68 % FP at ≥ 60 % recall (≈ 60× the target).

### 6.4 Earlier experiments in the repo (context)
- `mass_20260707_121819` (val 1 200 / 192 robbery): best 36|40 MEAN@0.68 F1 73.7, 12 FP, 73 FN; ~100 k synthetic mass-augmentation rows gave **no operational gain**.
- `fp_pipeline_v1` (July): stage-1 ensemble + robbery-vs-{2,3,14} verifier + kinematic heuristics (`s_kin`, reach→conceal, conceal-only) → **8 FP, recall 52.6 %, F1 67.1** — *no better than* the AND-ensemble frontier (FP ≤ 8 → ~58 %).
- Sliding-window (3 s, stride 1 s) + temporal filters on `campaign_20260714_164642` (val 1 241): FP −40–50 % but recall −20–25 pts (e.g. `modelo_12` binary: FP 47 → 28 default / 15 aggressive; recall 62 → 37 %). 49|57 + SW: FP 6 → 2, recall 64.8 → 43.8 %. **Important:** those models were *trained on whole clips*, not on 3 s windows. No config reached FP ≤ 1. Hard FP cases: class 1 ("cart / zone 1") with persistently elevated score over ~16 s.

### 6.5 Gap to the owner's target
Target: recall ≈ 66–75 % **and** FP ≈ 0 (≤ a few per day, not ~20). On the clip-level frontier (§6.3): recall 66 % needs ~15 FP/1 619 negatives (≈ 0.9 %) and 75 % needs ~25–30 FP (≈ 1.6–1.9 %); FP ≤ 3 gives only 42–48 %. In other words, **with the current pose-only information the two goals are in direct conflict at clip level**; reaching both needs (i) event-level instead of clip-level recall (several chances per robbery), (ii) a second stage (SAM3/HAR/C3, ideally learned) that removes the bulk of the remaining FP at small recall cost, and (iii) new information (§10). Stage 1 should therefore be tuned as a **high-recall, moderate-FP candidate generator whose output is cheap to verify**, *if* the verifier is shown (E7) to reject most FP; if not, stage 1 must stay at the low-FP operating point (AND @ 0.80) and recall is capped near 60 %. **Arithmetic of the owner's two-phase plan:** with ≈ 1 000 sessions/day and ≈ 10 stage-1 FP/1 000 sessions, ≈ 10 candidates/day reach HAR/SAM3; a ≈ 1 FP/day budget therefore requires a verifier that rejects **≥ 90 %** of them (false-reject of true robberies ≤ ~10 %, otherwise recall falls below 66 %). Conversely, at today's production level (20–30/1 000) the verifier would need ≥ 95–97 % rejection.
---

## 7. Error analysis *(computed, Oct, ensemble 41|48 AND@0.80, `bin_full`)*

### 7.1 False positives (11)
| cat | source | camera | clip dur | `prob_pos` (one ensemble member) |
|---:|---|---|---|---|
| 0 | datos_2 | d72d10c9 | 6 s | 0.98 |
| 0 | datos_1 | 41dd1e65 | 3 s | 0.25 |
| 0 | datos_1 | 41dd1e65 | 4 s | 0.87 |
| 0 | datos_1 | ff935914 | 6 s | 0.20 |
| 1 | datos_2 | d72d10c9 | full | 0.72 |
| 1 | datos_1 | 416a662e | 7 s | 0.99 |
| 14 | action-rec | 41dd1e65 | full | 0.15 |
| 14 | action-rec | 1d133c9d | full | 0.14 |
| 14 | action-rec | 41dd1e65 | full | 0.74 |
| 2 | action-rec | 1d133c9d | full | 0.83 |
| 4 | datos_1 | d72d10c9 | 4 s | 0.11 |

(The `prob_pos` column in `val_errors_ensemble_*.csv` is one of the two members' probabilities; both must exceed 0.80 for an AND alarm. Treat this column as indicative.)

- **Per-category FP rate in val:** cat 14: **3/25 = 12 %**; cat 1: 2/62 = 3.2 %; cat 0: 4/383 = 1.0 %; cat 2: 1/148 = 0.7 %; cat 4: 1/243 = 0.4 %; cats 3, 5, 7, 9: 0. Cat 14 has only 164 train clips (+14 added in Oct) and has been a top FP source in **both** campaigns (Aug: 3; Oct: 3).
- **Per-source FP rate:** real sessions (datos_1+2) 7/1 279 = **0.55 %**; `action-recordings` 4/244 = **1.64 %**; `pablo_dic2025` 0/96.
- **Per-camera (FP/neg val):** `41dd1e65` 4/94 (4.3 %), `d72d10c9` 3/134 (2.2 %), `ff935914` 1/46 (2.2 %), `1d133c9d` 2/215 (0.9 %), `416a662e` 1/301 (0.3 %). Small counts; hint that some cameras are harder.
- **Consensus:** each of the 11 FP is also an FP in **5–20 of the 20 single models** (per-model best-F1 thresholds). FP lists: 316 clips are FP in ≥ 1 model, 131 in ≥ 3, 75 in ≥ 5, 27 in ≥ 10, 12 in ≥ 15. → **errors are systematic** (data/label/information), not model variance.
- Cross-campaign: of the 11 Aug FP clips (cats 1, 1, 1, 14, 14, 14, 2, 2, 3, 5, 9), **10 fell in the Oct train split and 1 in the Oct test split** (none in Oct val) — so the Oct models were trained on them and the new FP set is a different sample of the same hard population. Likewise, of the 117 Aug FN, 85 were in Oct train, 17 in val, 15 in test.

### 7.2 False negatives (117 of 323 robbery val clips)
- By source (FN / robbery val): `action-recordings` 73/213 (**34.3 %**), curated 36/102 (**35.3 %**), **continuous-pipeline real sessions (`datos_1/2`, production-like format) 8/8 (100 %)** (n is tiny); note that curated *real* robberies have the same FN rate as staged ones.
- By clip type: "full" clips 74/214 (34.6 %); chunked/curated 43/109 (39.4 %). By duration bucket (robbery val, [n, FN]): ≤ 3 s [35, 10]; 4–6 s [65, 26]; 7–9 s [6, 4]; ≥ 10 s [3, 3]; full [214, 74].
- By camera (FN / robbery val): `41dd1e65` 17/31, `6a2c7f58` 15/36, `0d84e739` 13/33, `416a662e` 12/43, `1d133c9d` 11/32, `c0b2d0c2` 9/42 — the weakest cameras by recall are `41dd1e65` (55 % missed) and `6a2c7f58` (42 %), strongest `c0b2d0c2` (21 %) and `416a662e` (28 %). (Curated clips have no camera id.)
- **Consensus:** 260 robbery clips are missed by ≥ 1 of 20 models; 143 by ≥ 5; 72 by ≥ 10; 37 by ≥ 15; **8 by all 20** (3 datos_2, 2 action-rec, 2 datos_1, 1 curated) → 5 of 8 real-session val robberies are missed by every model.

---

## 8. Critical findings and hypotheses (why metrics look good but behavior is unstable)

Ordered by estimated impact × confidence.

**A. Evaluation excluded the new data (verified).** `evaluate_validation.build_split_examples` builds the pool from `collect_examples(data_root=plan["data_root"])` — **only the first root** — while `plan["data_roots"]` lists both. Result: `pool_total = 15 432` (= Aug data size) and **434 / 2 376 val UIDs (all `csv_buenos/…` UIDs; datos_3 + new action-recordings) were never evaluated** (same for test: 462 / 2 388). Training used 19 059 examples (log "Ejemplos totales (tras filtrado): 19059"). Fix: pass all roots (`GUADIA_DATA_RESULT_ROOT` with `os.pathsep`, or iterate `data_roots`). *The Oct numbers therefore do not tell us anything about the new batch.*

**B. Source/provenance and the real-session pipeline (counts verified; owner confirmed: `action-recordings` = actions deliberately recorded by the team; curated Dec-2025 `robos*` clips = real sessions).** Robbery clips: 63.5 % staged (`action-recordings`, same few performers, known spots), 33.8 % real-session curated cuts, 2.7 % from the continuous production pipeline (`clip_buffer`, the same format as production chunks). Negatives: 78 % real continuous sessions, 4.7 % reviewed real Dec-2025 recordings, 17 % `action-recordings`. Findings: (i) recall is the same for staged and curated-real robberies (34 % vs 35 % FN) ⇒ staged data are not obviously easier; (ii) the 8 val robberies from the continuous pipeline were all missed (n = 8, inconclusive but worth a dedicated test, since they are the only ones in *production format*); (iii) FP rate is higher inside `action-recordings` (1.64 % vs 0.55 % in real sessions), where positives and negatives share the source — consistent with *the model learning semantics poorly rather than source shortcuts*; (iv) cameras `d72d10c9, ff935914, 13a1322f` have almost no robbery clips (2–7 each), so camera can act as a prior. A source/camera-prediction probe (§10 P1.4) would settle whether shortcuts exist.

**C. Systematic errors ⇒ label noise / ambiguity / missing information.** All 11 FP and 72 FN are consistent across ≥ 5–10 independent models. Candidate causes: (i) wrong or ambiguous labels (e.g., "buy by taking the product" vs "steal" differ by *what happens to the object afterwards*, not by arm pose); (ii) clip-level labels with the true event covering a few seconds of a longer clip (weak labels); (iii) information not in 8 upper-body keypoints (object in hand, bag, payment, store zone); (iv) pose-tracker errors (ID switch merges two people → fake "gesture").

**D. Train ≠ inference representation (hypotheses to test, each cheap):**
1. **Padding shortcut.** Short clips are padded with repeated last frame (zero velocity). Curated robberies ≈ 4 s ⇒ ~50 % static tail at `seq_len=96@12fps`; most negatives 8–10 s. Production chunks are *up to* 8 s (a full chunk has no padding, but short sessions give shorter chunks, e.g. 4 s in the sample of §0b.9, which *are* padded by `temporal_resize`). *Test:* report metrics stratified by padding fraction; train with random temporal crops instead of padding; evaluate on 8 s-cropped windows.
2. **FPS/time scale.** Cameras deliver 15 fps (≥ 12 required); **training clips are 15 fps (owner-confirmed)**; production video is 15 fps but production chunk `.npy` have a variable number of frames (a short chunk can be 4 s or less; sample files say `pose_fps` 12 or 15 — derive it from the timestamps, see Q-J in §0b.4/§0b.9). The key issue is **consistency**: (i) per-frame velocity features scale with fps; (ii) `temporal_resize` maps *any* clip length T to `seq_len` by uniform subsampling (T > seq_len) or last-frame padding (T < seq_len), so the effective time scale **depends on clip duration** (at 15 fps a 4 s action = 60 frames is not subsampled; an 8 s production chunk = 120 frames is subsampled ×1.25 for `seq_len=96` and ×1.9 for `seq_len=64`; a 20 s clip ×3); (iii) frame dropping of invalid frames (`valid_mask`) compresses time. *Test:* score val clips after resampling to 10/12/15/18 Hz and after embedding them in 8 s production-style windows; time-based resampling to one fixed Hz everywhere (see §10 P0.6 and §0b.4).
3. **Pose engine mismatch.** Production uses a `yolo26s-pose` `.engine` (TensorRT); training extraction used a *different* `yolo26s-pose.engine` file (owner, round 4; some clips may be `.pt` — check `yolo_backend` in `meta.json`). Same weights family, so differences are build-dependent precision/NMS/jitter (§0b.8), but the 0.5 keypoint-confidence NaN gate and `valid_mask`-based frame dropping are sensitive to small confidence shifts (they change *which frames exist*). *Test:* re-extract a subset of val clips with the production engine and compare per-clip scores (see §10 P0.5).
4. **Missing-joint handling.** `NaN → 0` *before* centering places missing joints at the image origin (artifact); frames with < 5 visible joints are dropped (time compression). Production chunks may contain many such frames in crowded/occluded scenes, **and the Triton production path may not drop them at all (no `valid_mask.npy` is listed in the simulated clip — §0b.10)**; a pose that exits the frame can even appear as frozen identical frames (§0b.9).
5. **Single clean skeleton in training vs crowded aisles + ID switches in production.** Production sessions are per person, but with up to ~4 people in an aisle the tracker can swap IDs and the skeleton "jumps" to another person; occlusions by other customers/shelves create partial skeletons. None of this appears in training (single visible person, single skeleton), so a jump or fragment is out-of-distribution and its response is unpredictable — a prime suspect for *unstable, apparently random* FPs. *Test:* flag chunks with pose/bbox discontinuities and measure alarm rate vs. clean chunks; simulate ID switches (splice tracks of two people) as hard-negative augmentation.
6. **Training segments are action-centered; production slices are not.** Every training clip was cut by a labeler between "start" and "end" of one action. The model has never seen *background* windows (person standing still, turning, transitions between actions, half an action at the chunk edge, two consecutive actions in one 8 s window). In production these are the bulk of the windows. The negatives that exist (cat 0/1/5…) are *other labeled actions*, not "nothing in particular".
6. **Centering + global scale normalization** deletes locomotion/scene context: "hand on the back while walking" cannot be distinguished from "hand to the back while standing" other than via the arm articulation itself. The model has **no notion of walking vs stopping, position, distance, or direction**.

**E. Metric mismatch with production.**
- *Prevalence:* val has 16.6 % robberies; real sessions ≈ 0.5 % (or lower). With recall 63.8 % and FPR 0.68 %: precision ≈ **32 %** at 0.5 % prevalence, ≈ **8.6 %** at 0.1 %. In a store with 1 000 interactions/day (campaign config value), clip-level FPR 0.68 % ≈ **6.8 FP/day** before HAR/SAM3, vs target 0.01 % ≈ 0.1/day.
- *Multiple looks per session:* each session yields ~`2n − 1` evaluations (n chunks + n−1 overlaps). If clip-level FPs were independent with p = 0.68 %: a 1-min session (n = 8 → 15 evals) has ≈ **9.7 %** chance of ≥ 1 candidate FP; 2 min (29 evals) ≈ **17.9 %**. Independence is false (errors are correlated within a person/session), so this is an upper bound, but it shows why per-clip numbers are optimistic. The downstream HAR/SAM3/C3 stage is currently the only thing reducing this. **Confirmed by owner:** SAM3 and HAR run in production, are independent of the motion model, and **have never been run on the val/test clips** (only on random other clips) — so the *end-to-end* FP-rejection rate and recall cost are **unmeasured**.
- *Distribution of what is evaluated:* training clips are action-centered (event happens inside a 3–10 s clip); production chunks are arbitrary 8 s slices of a person's presence (event may be cut, or absent for most of the chunk).

**F. Optimism from model/threshold/ensemble selection.** Checkpoint by best `val_acc`; ensemble pair + rule + threshold chosen on the same val; thousands of candidates; only 11 FP → heavy winner's-curse. The `test` split (2 388 Oct UIDs, 1 930 Aug) has **never been evaluated** (confirmed). Retraining shows calibration drift: the Aug pair at its Aug thresholds drops recall 63.6 → 54.2 % on Oct data/splits.

**G. Train/val leakage.** ~10–12 % of val/test clips share a session with train clips (overlapping chunks of the same person); `action-recordings` have unknown grouping. Splits are not grouped by camera/day. Probably mild for FP but inflates both recall and FP-rate stability.

**H. Hard negative classes are underrepresented relative to their difficulty.** Cat 14 (3/25 FP) has 164 train clips; cat 1: 433; cat 13: 44. Mass-augmentation and "hardened" loss did not fix it (July/Aug experiments), consistent with "need *new real* hard negatives, not more synthetic variants".

---

## 9. What the previous attempts say about where the ceiling is

| Attempt | Effect on FP–recall frontier |
|---|---|
| More models / architectures (20 → 60 catalog; 29 evaluated) | Within noise: ≥ 8 different pairs reach ≈ 11 FP @ ≈ 62–64 % recall |
| More data (+3 673 clips, almost all negatives) | None observed (and new-root val not even evaluated) |
| Hardened loss + hard-negative mining + neg:pos 6:1 | Not consistently better |
| ~100 k synthetic mass-augment rows | No operational gain |
| Sliding window (3 s) + temporal/kinematic filters | FP −40–50 %, recall −20–25 pts (moves *along* the frontier) |
| Verifier (robbery vs 2/3/14) + heuristics H1–H12 (July) | FP 8 @ recall 52.6 % (≈ same frontier) |
| Ensembles (AND/MEAN/cascade, pairs/triplets) | Best "free" FP reduction vs single model (single: ~40+ FP @ 63 % recall → ensemble 11 FP @ 64 %) but plateaus |

**Interpretation:** all knobs that re-use the *same 8-keypoint relative-pose information with the same labels* slide along one curve. To move the curve we need **new information** (context, hands/object, longer session behavior, production hard negatives) or **cleaner supervision** (label audit, event-level labels).

---

## 10. Proposals (ranked) and answers to the specific questions

### 10.0 Owner's strategy: a *generalist* two-phase plan (guides everything below)
The owner does **not** want to fix false positives one clip at a time ("there will always be new ones"). The plan is: **Phase 1 — train the stage-1 model so it minimizes FP in general while keeping ≈ 66 % of robberies; Phase 2 — use HAR/SAM3 (and alternative filters) to eliminate the remaining FP.** Quantitative frame: ≈ 1 000 sessions/day; today stage 1 raises ≈ 20–30 FP per 1 000 clips in production; goal ≈ ≤ 10 per 1 000, and then the verifier removes ≥ 90 % so that ≈ 1 FP/day remains (ideally a near-robbery gesture such as putting a phone in a pocket). HAR/SAM3 only run on stage-1 candidates and have not been measured on labeled data.

**Phase 1 — generalist ways to minimize stage-1 FP (not case-by-case):**
1. **Use the production sessions as a massive, automatically refreshed negative pool.** Production sessions of "all kinds of people doing all kinds of actions" exist. Cut them exactly as production does (the fps contract decided in §10 P0.6, chunks of up to 8 s, N/N+1 windows) and treat every window not labeled as robbery as a negative (positive-unlabeled assumption; robbery prevalence ≈ 0.5 %, so contamination is small and can be cleaned by the label audit). This gives the model the true background distribution (transitions, standing, turning, partial actions, crowded scenes) that labeled action-segments lack, and it is the single most *general* FP-reduction lever.
2. **Optimize the metric that matters, not accuracy:** partial-AUC / recall-at-fixed-FPR objectives (e.g., one-way pAUC losses, top-k hardest-negative losses, or constrained learning "minimize FN subject to FPR ≤ α" with a Lagrange multiplier); **checkpoint and hyper-parameter selection by recall @ FPR ≈ 0.3–0.5 %** on group-aware validation (today: best `val_acc`, which is dominated by easy negatives).
3. **Automated, iterative hard-negative mining at scale:** replay stage 1 on all stored sessions each round, harvest *all* alarms (and near-threshold windows), add them as negatives (with caps per session/camera to avoid one scene dominating), retrain, repeat; human review only on a random sample to estimate label noise. This replaces manual clip-by-clip fixing by a loop that finds *new* FP types automatically.
4. **Background / "near-miss" classes (outlier exposure).** Add a large-diversity "other/background" class and explicit near-robbery negatives (phone/wallet/glasses/keys into pocket, hand on back, adjusting clothes, putting own item in own bag = cat 14), so the model learns a margin around the confusable region; the *residual* FP then tends to be exactly the tolerated tier ("phone into pocket"). Report FP by severity tier.
5. **Robustness augmentations that mimic production nuisances:** ID-switch/skeleton-jump splices, keypoint dropout and jitter (occlusion by other people), fps/speed jitter (§10 P0.6), random window offsets and lengths, temporal cut-mix of a robbery segment into a negative background window (for MIL localization). Avoid spurious shortcuts (length padding, source cues).
6. **Fixed temporal contract** (one fixed fps — 15 or 12, see P0.6 —, windows in seconds, no padding shortcut) — §10 P0.6 — so that thresholds transfer between retrainings and between data and production.
7. **Stability by construction:** ensembles trained on *different data bootstraps and views* (AND rule keeps FP low), seeds/bootstrap CIs, conformal/quantile threshold set on a **replay calibration set of production sessions**, and acceptance gate = FP per 1 000 sessions on held-out days.

**Phase 2 — HAR / SAM3 and alternatives (after Phase 1):**
1. **Measure first:** run HAR and SAM3 on all stage-1 candidates of the replay set and of val/test (with labels): rejection rate of FP and false-reject rate of true robberies.
2. **Learned fusion (C3)** over stage-1 scores + HAR + SAM3 states + pose quality + zone (§10 P3); calibrate to ≈ 1 FP/day.
3. **Alternative/complementary filters** (§10 P4, P5): ID-switch/pose-quality gates, locomotion/zone gates, hand-ROI/object-in-hand classifier (SAM3 distilled to run on every candidate cheaply), persistence/ordering rules, per-session cooldown.
4. **Define tolerated residual FP:** "person putting a phone away" is acceptable at ≤ 1/day; design thresholds so the residual FP are near-robbery gestures and not unrelated movements.


### P0 — Make measurement trustworthy (days, prerequisite for everything)
1. **Fix the multi-root evaluation** (§8.A) and re-run Oct eval; report old-root vs new-root metrics separately.
2. **Group-aware splits:** group by `session_id` / recording id, stratify by `camera_id` and day; create (a) *random-group* split and (b) **leave-one-camera-out / leave-one-day-out** CV. Report FP and recall with **Wilson/bootstrap CIs**.
3. **Real-session-first benchmark.** Build a dev/test set that mimics production: real sessions only, **8 s chunks + N/N+1 windows exactly as in production**, prevalence-realistic, plus all labeled real robberies (69 today — *collect more*; see P9). Metric: **alarms per camera-hour** and **% sessions with ≥ 1 alarm**, at pose-model-stage and at full-pipeline level (including SAM3/HAR/C3).
4. Hold out `test` and touch it **once**. Choose checkpoint by a **val FP-controlled criterion** (e.g. pAUC at FPR ≤ 1 %, recall @ FPR = 0.5 %) instead of `val_acc`.
5. Re-extract poses of the val/test clips with the **production** `yolo26s-pose.engine` pipeline at production fps (or re-run production inference on stored videos) to quantify train–prod shift (§8.D.3). Compare per-clip scores across the training engine, the production engine and `.pt`.
6. **Temporal contract — OPEN QUESTION + PROPOSAL (fps).** *Context (owner):* cameras run at 15 fps; the owner **can force the `.npy` to be emitted at exactly 12 fps** (dropping/resampling frames) in production and use 12 fps for training, i.e. "always 12, collection and training". The owner's open questions: (Q-fps1) *Is a fixed 12 fps the right choice?* (Q-fps2) *How critical is a mismatch if a model trained at 12 fps receives npy at 15 fps (or the reverse)?*  **Update (owner, round 3):** training clips are **15 fps**; production `.npy` can be forced to 12 fps. Options: (A) keep **15 fps end-to-end** (no retraining for fps; make sure production emits exactly 15 fps — cameras already do); (B) move **both** to 12 fps (resample the 15 fps training `.npy` offline, retrain). Option A is cheaper and removes today's potential 12-vs-15 mismatch immediately; B only pays off if compute/bandwidth or fps jitter at 15 is a problem. The expert should decide, quantify E-fps, and also address the **window-length/`seq_len` compression issue of §0b.4**, which matters more than 12 vs 15 per se.
   **Proposal (for the expert to interpret/validate):** (a) **Run pose + tracking at the native 15 fps** (better ID continuity, more frames for the tracker) and **emit the model input at a fixed 12 Hz by timestamp** (nearest-frame or linear interpolation of keypoints; propagate NaN if either neighbor is NaN) — not by naive frame dropping; (b) **train at the same fixed 12 Hz**: the existing training `.npy` come at their native fps (`meta.json: fps`, unknown distribution, 12 only as fallback), so write an **offline resampler** that converts every `poses_full.npy`/`keypoint_mask.npy`/`valid_mask.npy` to 12 Hz from `meta.json` fps (no YOLO re-run needed) and records `fps_effective = 12`; (c) define windows in **seconds** (8 s = 96 frames at 12 Hz) and give the model exactly `seq_len` frames — crop windows rather than uniformly rescaling whole clips; pad only with an explicit mask; (d) add **fps-jitter augmentation** (resample to 10–15 Hz then back to 12 Hz grid / speed 0.8–1.25) so small timing errors are not critical; (e) log `pose_frame_timestamps` gaps and interpolate short dropouts.
   **Analysis of the mismatch question (hypothesis, to be measured):** the pipeline computes `(x, y, dx, dy)` with `add_velocity` **before** `temporal_resize`. If a 15 fps chunk (120 frames per 8 s) is fed to a model trained at 12 fps with `seq_len = 96`, uniform subsampling restores the frame count (positions look ≈ 12 fps) but **velocity channels remain 15 fps-scaled (≈ 0.8× smaller)**; for models with `seq_len = 64` the subsampling factor changes accordingly. Temporal convolutions/GRU see motions ≈ 25 % "slower/faster". The training speed augmentation covers 0.85–1.15, so 0.8 is slightly outside. Expected effect: **moderate, not catastrophic, but it shifts calibration** (probabilities near the 0.80 AND threshold are sensitive: a few % shift in logits can change FP counts by tens of percent). The reverse (12 → 15 fps data into a 15 fps-trained model) is symmetric. A mismatch is therefore *critical for threshold stability even if not for accuracy*. **Cheap experiment (E-fps):** take the val clips, resample them to 10/12/15/18 Hz with interpolation, score with the current 41|48 AND@0.80, and plot FP/recall vs fps; this quantifies sensitivity without any retraining. Decide the final contract after seeing the curve; if sensitivity is large, retrain with fps-jitter augmentation.

### P1 — Data and labels (highest expected FP impact)
1. **Label audit of consensus errors:** review the ~75 clips that are FP in ≥ 5/20 models and the ~72 FN missed by ≥ 10/20 (lists are exportable: `val_fp_manifest.csv`, `fp_clips/`). Re-label/remove ambiguous ones; consider an explicit **"ambiguous/ignore"** class. Use **confident learning (cleanlab)**, loss-trajectory/forgetting statistics across the 20 models to flag suspected noisy labels automatically.
2. **Production hard-negative loop:** every p1.json window that SAM3/HAR/C3 labels as *false positive* (or a human rejects) becomes a labeled negative with the exact production chunk format; periodic retraining with those; cap per camera/session to avoid dominance. This directly targets the stable "hand-on-back while walking" pattern.
3. **Add real-session robberies** (currently 69) and **real-session hard negatives of categories 14/1/0/2**; weight by source so real sessions are not drowned (re-weighting or source-balanced sampler); possibly **domain-adversarial / source-dropout** training to discourage source cues.
4. **Source-shortcut diagnostics:** train a classifier to predict `source`/`camera` from the poses the model sees; if easy, add domain-invariance measures. Evaluate with *source-matched* negatives (action-recordings negatives vs action-recordings robberies).
5. **Background/"no-event" windows (new, from owner answers).** Clips are already action-segments (start/end pressed by the labeler), so segment-level labels exist. What is missing is everything *between* labeled actions: from the same real sessions, mine **8 s windows and 3 s windows that lie outside labeled segments** and in transitions (standing, turning, walking with hands in odd places, two actions in one window), label them as negative/"background" (weak labels are fine — a session without a robbery event is negative everywhere), and train with production-like window sampling (random offset, 8 s, with N/N+1 overlap). Also add **ID-switch/skeleton-jump negatives** (§8.D.5). This directly reproduces the production input distribution and is probably the cheapest large gain in FP. For robberies, optionally refine the segment to the take/conceal sub-event (±1 s) to train 3 s models; bootstrap with `clean_theft.py` reach/conceal peaks + human check.
6. **Re-think the label semantics of confusable categories.** 6 (hide product) vs 14 (put object in bag), 3 (put in basket/cart/bag), 4 (carry product), 2 (take, look, put back) are almost the same arm kinematics; the distinguishing facts are *object ownership/type* and *zone*. Options: (a) accept that pose-only cannot separate them and treat 14/3/4 as **explicit hard negatives that must be resolved by SAM3/HAR/context** — so the stage-1 target becomes "something is being put near the body/bag" rather than "robbery"; (b) feed **zone/location features** (checkout area vs aisle; shelf vs cart region) so bag-packing at the exit is not confused with hiding in the aisle; (c) collect real-session examples of 14 with locations.

### P2 — Temporal window question: **8 s → 3 s?**
- **Hypothesis:** an 8 s window dilutes a ~3 s act (buying and stealing last "~3 s" within an 8 s window according to the earlier plan note) with unrelated motion (walking, looking), and short action-centered training clips are mismatched with long production chunks. A shorter window increases temporal localization and lets "stop & look" or "walking with hand behind back" segments be classified for what they are.
- **Evidence in repo:** 3 s + temporal filters cut FP ≈ 40–50 % but cost ≈ 20+ recall points **with models not trained for 3 s**. Curated robbery clips are only ≈ 4 s median; many `seq_len=64@12 fps` ≈ 5.3 s models already see sub-8 s.
- **Recommended experiment (not just changing the chunk size):**
  - Train at **multi-scale windows** (e.g., 2, 3, 4, 6 s; fixed Hz), random crops of the labeled clips instead of padding; MIL / max-pooling over windows of a clip (weak labels) or event-level labels when available (P1.5).
  - **Inference:** score all 3 s windows with stride 0.5–1 s inside each 8 s chunk and across chunk boundaries (this also removes the N/N+1 boundary artifacts), aggregate with **k-of-n consecutive windows** + peak/mean features; feed the *window-score time series* into C3 (P3).
  - Compare **at equal operating points on the real-session benchmark (P0.3)**; expected outcome: better recall at the same FP for events inside the window; risk: more evaluations per session ⇒ more FP chances unless aggregation is strict.
  - Also test **longer context**: a second-level model over the sequence of chunk scores of the *same track* (pre-event approach to shelf, post-event leaving), because stealing is a sequence (reach → conceal → leave), not a pose.

### P3 — Learned fusion instead of rule-only gating (C3)
Replace/augment rule logic (SAM state priority, HAR flags) by a small **calibrated classifier** (logistic regression / gradient-boosted trees / tiny GRU over window scores) with features: `p1, p2, min, gap, logit0/1`, per-window time series stats, **HAR flags**, **SAM3 object state + confidence**, `no_hands_in_roi`, pose quality (% valid frames, keypoint confidence, occlusion ratio), dwell time, camera/zone, number of persons in frame, track age. Train on **production-labeled sessions** with the real FP budget; optimize **cost-sensitive** loss and calibrate for a target alarms/hour. This is the most direct way to turn the three signals into a stable decision; today we only know the cascade qualitatively.
Add **disagreement-based abstention**: if SAM3 and motion disagree strongly and HAR is ambiguous → "review/low priority" instead of hard alarm.

### P4 — Additional filters / gates beyond HAR and SAM3 (cheap, rule-or-learned)
1. **Pose-quality gate and ID-switch/jump detector** (H10 in `PLAN_REDUCCION_FP.md`): % frames with ≥ 5–8 keypoints, mean keypoint conf, bbox aspect ratio, occlusion_ratio, **frame-to-frame jump of hip-center/shoulder width/bbox (physically impossible speed), ReID score drop, track-age/fragmentation**. A chunk containing a jump should not raise a hard alarm; split the chunk at the discontinuity and score the clean sub-segments. Owner-confirmed production problem (ID switches), so this gate has a concrete, measurable target.
2. **Locomotion gate:** from hip-center trajectory and bbox motion in image/world coords: *walking-through* with hand-on-back gesture ⇒ down-weight; *stationary at shelf/cart zone* ⇒ up-weight. (The models currently cannot know this; compute as side features.)
3. **Zone/ROI + dwell:** shelf zone → cart zone → exit/blind zone transitions; "item picked in shelf zone and not in cart afterwards" (SAM3 already approximates). Camera-specific zone maps.
4. **Temporal persistence/hysteresis:** require sustained score over ≥ k windows *and* a reach→conceal ordering (H4) — the earlier test shows it moves along the frontier, so use it as a **second-stage** after better features.
5. **Interaction filter:** presence of staff/cashier or another customer within proximity; back-turned persons (hands not visible) ⇒ rely on HAR only or abstain.
6. **Cross-view consistency** (if several cameras overlap): require agreement of ≥ 2 cameras for high-severity alarms.
7. **Re-identification of repeat FP patterns:** per-person (track-level) alarm cooldown after a SAM/HAR-confirmed FP (suppress repeated alarms from the same session).

### P5 — Richer inputs (information, not capacity)
1. **All 17 keypoints** (head orientation / nose-eyes for gaze toward cameras/staff, knees/ankles for locomotion), or whole-body models (RTMPose-wholebody, MediaPipe Hands) for **hand state** (closed/open/holding) — the single largest missing cue: *object in hand*.
2. **Add global channels to the existing tensor:** hip-center position & velocity in the image, torso scale (distance proxy), bbox aspect ratio, per-frame keypoint confidence, `valid` flag — **as separate channels with dropout** to avoid camera overfitting; keep the relative stream as is. Replace `NaN→0` with an explicit mask channel + temporal interpolation for short gaps.
3. **Small RGB/hand-ROI branch** around wrists (e.g., 64–112 px crops, lightweight 2D CNN or video model such as X3D-S/MoViNet/VideoMAE-tiny) predicting *object-in-hand / object-concealed*; SAM3 does something similar at the end of the span — consider distilling SAM3 states into a lightweight real-time classifier to run on **every** window instead of only after a candidate.
4. **Optical-flow/motion energy near hands** as cheap features.
5. **Appearance/clothing** descriptor (already in meta) must *not* be fed to the classifier (leakage/shortcut) but can serve to merge track fragments.

### P6 — Training procedure (stability and calibration)
1. Replace checkpoint selection and thresholding by **FP-aware metrics**, use **EMA/SWA**, **label smoothing**, **mixup in feature space** carefully, **class-balanced batch sampler** with source/camera stratification.
2. **Temperature/Platt scaling per model on a calibration split** and **conformal prediction** for the alarm threshold (guaranteed FP rate on exchangeable data); report **ECE**; make thresholds independent of the retraining run (the Aug pair lost ≈ 9 recall pts on Oct data at fixed thresholds).
3. **Deep ensembles with different seeds/inputs** (not only different architectures): since errors are consensus errors, diversity must come from *different information* (e.g., pose-only model + hand-ROI model + kinematic features model), not from more TCN/GRU variants. Use **uncertainty (ensemble variance, MC-dropout)** as an abstention signal.
4. **Group-DRO / worst-camera optimization** to prevent a few cameras from dominating errors.
5. **Self-supervised pretraining on unlabeled production pose streams** (masked-pose modeling, contrastive pose-sequence learning, e.g., AimCLR/ActCLR-style, or fine-tune a pretrained motion encoder such as MotionBERT/other skeleton-transformers): large unlabeled volume exists (every production chunk), labeled negatives/positives are the bottleneck.
6. **Positive-unlabeled / noisy-label losses** (e.g., symmetric CE, co-teaching, small-loss selection) given the consensus-error evidence.

### P7 — Architectures worth adding (answering "any additional architecture from the commented list?")
Given §9 (architecture is not the bottleneck) these have **modest expected upside**; prioritize after P0–P3:
1. **`stgcn` (30–32) and `gat_tcn` (58–60)** — never evaluated in the last two campaigns; skeleton-graph models are the natural inductive bias, but with 8 joints the graph is tiny; expect parity with TCN-GRU, potentially better robustness to missing joints.
2. **`gru_attn` (45–47), `lstm` (15–22)** — only for ensemble diversity; low priority.
3. **Beyond the catalog:** CTR-GCN / InfoGCN / ST-GCN++ (skeleton), **PoseC3D** (heatmap volumes — robust to pose noise/missing joints; also usable with 17 or whole-body keypoints), skeleton transformers (SkateFormer/Hyperformer) *only with pretraining*; **multi-scale temporal pooling heads** and **attention pooling over time** (rather than last-state) to localize the action inside an 8 s chunk.
4. **Prune further** the families that are consistently below the others to cut campaign time (the 20 kept already do; ~95 % of candidates are within noise of each other).

### P8 — Operational policy (if FP ≤ 0.01 % remains unreachable)
- **Gray zone:** 0.55 ≤ score < threshold → "soft alert"/review queue, not a hard alarm; count *actionable alarms/day* as the product KPI.
- **Double confirmation:** hard alarm only if stage-1 AND verifier AND (SAM3 *or* HAR) agree; severity levels.
- **Per-camera thresholds** from per-camera calibration (cameras `41dd1e65`, `d72d10c9`, `ff935914` are FP-heavy in val).

### P9 — Data collection plan (we assume data exists; this is about targeting)
Priorities by expected FP/recall value: (1) real-session robberies with diverse cameras/clothing/occlusion; (2) real hard negatives of cats 14, 1, 0, 2, 4 (the FP categories) captured *from production alarms*; (3) multi-person scenes; (4) back-turned / partially visible scenes; (5) night/different lighting. Stop adding generic negatives of easy categories (0/5) — they are saturated.

---

## 11. Suggested experiment matrix (for the expert agent)

| ID | Experiment | Success criterion (real-session benchmark) | Cost |
|---|---|---|---|
| E0 | Fix multi-root eval + CIs + test-once protocol | Oct numbers recomputed incl. new root | hours |
| E1 | Stratified metrics by source / camera / padding fraction / duration | Identify shortcut magnitude | hours |
| E2 | Re-extract val clips with the production YOLO engine file (and with the training engine and `.pt`) at production fps | Δ(FP, recall, NaN rate, alarm-flip rate at 0.80) between engines | 1 day |
| E3 | Remove padding shortcut (random temporal crops, fixed-Hz resampling); retrain 41/48 | FP ↓ at same recall; stability ↑ | 1 day |
| E4 | Label audit of consensus errors + cleanlab | % relabeled; Δ metrics after cleanup | 2–3 days (humans) |
| E5 | 3 s multi-scale windows + MIL/event labels + k-of-n aggregation | Recall @ FP budget vs 8 s baseline | 3–5 days |
| E6 | Add global/context channels + mask channel (dropout) | FP in "walking, hand on back" slice ↓ | 1–2 days |
| E7 | Learned C3 fusion on production sessions | Alarms/camera-hour at recall target | 1 week |
| E8 | Hand-ROI / object-in-hand branch (distilled from SAM3) | Recall on real-session robberies; FP on cat 14/1 | 1–2 weeks |
| E9 | ST-GCN / gat_tcn / PoseC3D, pretrained motion encoders | Only if E3–E6 show headroom | 1 week |
| E10 | Calibration + conformal thresholds + uncertainty abstention | Threshold transfers across retrainings | 2 days |
| E-fps | Resample val clips to 10/12/15/18 Hz and score with 41\|48 AND@0.80; then fixed-12 Hz contract (offline resampler + fps-jitter augmentation) | FP/recall curve vs fps; thresholds stable across fps | 1 day |
| E-replay | **Replay benchmark on ≈ 1 000 stored production sessions** (production chunking + N/N+1 windows) with the current models | FP/1 000 sessions; reproduce the observed 20–30/1 000 and attribute it (ID switches, background windows, fps) | 2–3 days |
| E-parity | **Golden-tensor test**: feed `SAMPLE_production_chunk_mergedfps.npy` and several training clips through (i) the training dataset path, (ii) Triton path A, (iii) Triton path B, with and without `valid_mask`; compare tensors and scores (§0b.10) | Identical tensors/scores (or quantified deviation) | hours |
| E-chunk | Compare per-chunk, overlapping, sliding-window and streaming scoring + per-session aggregation on the replay set (§0b.7) | FP per camera-hour / per 1 000 sessions and latency at ≈ 66 % event recall | 2–3 days |
| E-engine | Same videos through training engine / production engine / `.pt`; compare keypoints, NaN pattern, IDs and scores (§0b.8) | Alarm-flip rate and score deltas | 1 day |
| E-bg | Train with **background/unlabeled-window negatives** mined from production sessions (PU-style) + ID-switch negatives | FP/1 000 sessions ↓ toward ≤ 10 at ≥ 66 % recall | 3–5 days |

---

## 12. Open questions

### 12.1 Answered by the owner (2026-10-13) and what is still open

**Answered (incorporated above):** category definitions (§1); `action-recordings` = team-recorded actions; curated Dec-2025 clips = real sessions (§4.1, §8.B); production pose = `yolo26s-pose` `.engine`, cameras 15 fps, one session per person with ID switches (§2); production ensemble `p1 = model.pt`, `p2 = model2.pt`, AND @ 0.80, `robbery_score = min` (§2); test split never evaluated; **HAR/SAM3 run only on stage-1 robbery candidates, independently of the model, and were never run on val/test clips** (§8.E/F); one store, 9 cameras; **≈ 1 000 sessions/day**; **FP budget ≈ 1/day end-to-end, preferably a near-robbery gesture such as putting a phone away**; **stored production sessions with normal actions, false positives and "everything" exist** (usable for replay benchmark and hard-negative mining — only for the model stage); owner can force **12 fps** for both collection and training; production model FP level ≈ **20 per ~700 clips (20–30 per 1 000)**; owner wants a **generalist solution** (not clip-by-clip patching): (1) optimal training for minimum FP at ~66 % recall, then (2) HAR/SAM3 and alternative filters.

**Answered in round 3 (2026-10-14):** (Q-G) **20 FP in 654 sessions in one day, model alone** (§0b.5); (Q-I, partly) **training clips are 15 fps** (§0b.4); (Q-H, partly) there are enough clips — a large package used for train/val/test, and the rest of the data comes from production, where the owner observes model + filters; sessions with normal actions, FP and everything exist; (Q-F) **no labeled phone-to-pocket or hand-on-back examples exist** and the owner wants a **generalist** (not action-by-action) solution (§0b.6); the owner highlighted the **training-by-action vs production-by-fixed-8 s-chunk** difference (§0b.1) and the **full vs filtered** question (§0b.2) and the changing winners between campaigns (§0b.3).

**Answered in round 4 (2026-10-14):**
- **Q-J:** production video is **15 fps**, but the number of frames per chunk `.npy` depends on how long the person is on screen (sample: 60 frames) — §0b.4, §0b.9.
- **Q-K:** a session = person enters → leaves the camera; crossing an aisle = 1 chunk (8 s), browsing = ~4 chunks (32 s), employee restocking = ~43 chunks — §0b.5, §0b.7. The owner asks whether per-chunk scoring is correct and optimal (§0b.7).
- **Q-I:** training `.npy` were generated with the `yolo26s-pose.engine`; the production `.engine` is a **different file** built on the same machine — §0b.8 (question for the expert).
- **Q-H (clarified, round 5):** the `.npy` viewer is only **a web platform where you upload a `.npy` and the skeleton is rendered — nothing more** (no video overlay, no labeling, no verdict storage). So **do not assume** an existing tool for human verification, labeling or replay; propose what minimal tooling is needed if your plan requires it. Whether production chunk folders with verdicts can be exported was not specified — design E-replay under both assumptions (available / not available).

**Open items — to be resolved BY THE EXPERT from the repository/artifacts or answered *conditionally* (do NOT ask the owner; see "NO INTERACTION" in the header):**
- **Q-fps1 / Q-fps2 (§0b.4, §10 P0.6):** which fps contract (15 end-to-end vs 12) and how critical is a 12↔15 mismatch for the 0.80-AND operating point?
- **Q-chunk (§0b.7):** is independent per-chunk scoring correct and what is the optimal chunk/stride/overlap/aggregation design (incl. streaming/causal alternatives and the single-session-`.npy` alternative)?
- **Q-engine (§0b.8):** effect of different `.engine` files for training vs production extraction, how to measure it and what contract to enforce.
- **Q-L (answered, round 5):** the labeled data are **one `.npy` per person-video of one action, whole-clip label, no frame-level intervals**; whether continuous streams can be regenerated from stored videos is **not stated**. Produce a plan that works with action-level labels only (weak localization/MIL, synthetic production-style windows built from action clips and unlabeled production negatives) and, as a secondary branch, what changes if streams can be re-extracted (§0b.1).
- **Q-M (answered, round 5, see §0b.10):** production runs through a **Triton inference server** with two preprocessing paths (A = IAModel `build_pose_dataset_for_eval`, B = `tensor_simple_pipeline`: `normalize_sequence` → `add_velocity` → `temporal_resize(seq_len)`), NaN → 0, `.pt` or `.engine`. **Still undetermined:** whether `valid_mask.npy` is produced/applied in the simulated clip (it is not listed), which path is live, and the `seq_len` of `model.pt`/`model2.pt` (64/96 if experiments 41/48). Evaluate the readings, quantify the risk and propose a golden-tensor test.
- **Q-session-FP:** are the 20 FP / 654 sessions concentrated in long sessions (many chunks), employees, specific cameras or ID-switch tracks? (Test on the replay data; if unavailable, give the expected distribution under the k-looks model and the decisions it implies.)

### 12.2 For the downstream expert agent (open research questions)
1. **Is the performance ceiling due to information (8 keypoints) or labels?** What fraction of consensus FP/FN are label errors vs genuinely indistinguishable by pose?
2. How much of the measured performance is **source/camera shortcut** (`action-recordings` vs real sessions)? What are metrics on a source-matched benchmark?
3. What is the **train→production gap** when poses come from the production engine/fps/chunking? Does re-extraction explain the "unstable" behavior?
4. **Window length:** what window size and aggregation maximize recall at a fixed alarms/hour? Does 3 s multi-scale + MIL beat 8 s? What is the optimal overlap policy compared with N/N+1?
5. **Which additional signal gives the largest FP reduction per unit of engineering cost:** locomotion/context channels, hand-state, object-in-hand classifier, zone/dwell, head orientation, multi-camera consensus?
6. Can **SAM3 outputs be distilled** into a real-time per-window model so that the object-state check applies to every window instead of only to stage-1 candidates?
7. How should **C3 fusion** be learned and calibrated to hit a target alarms/camera-hour with guarantees (conformal / risk-controlling prediction)?
8. How to **detect and handle the "hand on back while walking" pattern** (and cat-14/cat-1-like patterns) explicitly: a dedicated negative class, a locomotion gate, or a hard-negative curriculum?
9. Can **unlabeled production pose streams** (self-supervised pretraining) improve representation enough to move the frontier?
10. Is multi-class training helpful as an **auxiliary** task for a binary head (shared encoder, multi-task) instead of separate MC and binary models? (MC models are strong at recall but with 4–5 % FP.)
11. What is the minimum number of **real-session robberies** needed (learning-curve on real sessions) to see recall improve on real events? (A learning-curve harness exists: `learning_curve` in `campaign_config.json`.)
12. Does **ensemble diversity by input modality** beat architecture diversity given that FP errors are consensus errors?
13. **Chunking design (§0b.7):** per-chunk independent scoring vs overlap vs sliding window vs streaming/causal model vs one session `.npy`; optimal chunk length, stride, per-session aggregation (persistence, hysteresis, sequential tests) under a FP-per-hour/session-duration-aware budget; handling of short (< 8 s) chunks and very long (employee) sessions.
14. **Pose-engine consistency (§0b.8):** effect of different TensorRT `.engine` builds of the same `yolo26s-pose` weights (and of `.pt` vs `.engine`) on keypoints, NaN pattern, tracker IDs and model scores; extraction contract and drift test.
15. **Production sample (§0b.9):** effect of perspective scale change (approach/exit), edge exits, frozen/held frames and truncated chunks on the model, and how to simulate/normalize them in training.
16. **Latency target:** given theft events of ~3–11 s, what is the achievable detection latency at the target operating point and how does it trade against FP?

### 12.3 Design-variable questionnaire — every factor that can affect model quality
The owner asks the expert to treat **each variable below as an open question**. For each: (a) does it matter here (evidence/reasoning), (b) what is the best setting or range, (c) expected effect on FP and recall (direction and rough magnitude), (d) how to measure it cheaply (ablation/experiment), (e) interactions with other variables. Where the brief already contains partial evidence it is referenced. Group A–E are the highest-suspected levers; the rest must still be answered.

**A. Temporal definition of the input**
- **V1. Chunk/window length.** Is 8 s optimal? What are the effects of 2, 3, 4, 5, 6, 8, 10 s on recall, FP and stability? Does the optimum depend on the action (steal ≈ 3 s vs. long stationary behaviors)? Does the right length differ for training vs. inference? (§10 P2: hypothesis that ~3 s multi-scale helps, only if training changes too.)
- **V2. Single vs. multi-scale windows.** Should the model see several scales (e.g., 2/4/8 s) jointly or via an ensemble of scale-specific models? How to fuse them?
- **V3. Stride and overlap at inference.** Is the N/N+1 sliding window the right policy? What stride (0.25–2 s) and overlap maximize event-level recall at fixed alarms/day? How many evaluations per session does this create, and how does it multiply FP?
- **V4. Temporal context beyond the window.** Does adding pre-event/post-event context (approach to shelf, leaving) improve decisions? Should there be a second-level model over the sequence of window scores per track?
- **V5. Aggregation across windows.** max / mean / top-k / k-of-n consecutive / noisy-OR / learned (GRU over scores)? Which gives the best recall at ~1 FP/1 000 sessions?
- **V6. Frame rate (fps) and temporal resampling.** 12 Hz fixed vs. 15 Hz vs. others (6/8/10/12/15/20); resampling method (nearest/linear/by timestamp); is 12 enough for the gestures involved? Sensitivity to a 12↔15 mismatch (§10 P0.6).
- **V7. `seq_len` per model** (32/64/96/128) and interaction with fps and window length: what is the effective temporal resolution of each model and the receptive field of TCN/GRU? Is padding by repetition harmful? Is uniform subsampling harmful?
- **V8. Handling of missing frames / short sequences / gaps.** Interpolation vs. mask channel vs. dropping frames (current: `valid_mask` frames dropped, `NaN → 0`); minimum valid frames/coverage; what to do when the track is lost mid-chunk.
- **V9. Clip/segment labeling granularity.** Whole-clip label vs. event-level (start/end) vs. frame-level; weak-label/MIL strategies; how much does imprecise event timing hurt?

**B. Body representation (keypoints and features)**
- **V10. Number and choice of keypoints.** Today 8 (shoulders, elbows, wrists, hips). Should we use fewer (e.g., wrists+elbows+shoulders only; drop hips), more (all 17 COCO: nose/eyes/ears for head orientation, knees/ankles for locomotion), or whole-body (hands/fingers, face)? Quantify the effect of each addition/removal on FP and recall, and on robustness to occlusion and ID switches.
- **V11. Hand-level information.** Do hand keypoints (21 per hand), hand open/closed state, or wrist-orientation features add the missing "object-in-hand" cue? Cost/benefit vs. adding an RGB hand-ROI branch.
- **V12. Head/gaze/body orientation.** Does facing direction (toward/away from camera, toward shelf/staff) discriminate benign from suspicious behavior? Back-turned persons: what can be inferred?
- **V13. Pose model choice and confidence.** `yolo26s` vs. n/m/l/x, other pose models (RTMPose, ViTPose, MoveNet…); input resolution (1080 height clips vs. production stream resolution); confidence threshold `MIN_KP_CONF = 0.5`; `.pt` vs. `.engine` precision. Effect on keypoint noise, NaN rate, and downstream FP.
- **V14. 2D vs. 3D/pseudo-3D.** Would 2D→3D lifting or depth/scale cues help (distance to camera, arm foreshortening)? Camera viewpoint dependence across the 9 cameras.
- **V15. Coordinate normalization.** Per-frame centering and global-scale normalization (current) vs. torso-centered/rotation-normalized vs. keeping image coordinates; **which global information must be preserved** (position, distance, walking direction, speed)? Effect on "walking with hand on back" FP.
- **V16. Feature engineering.** Raw (x, y) + velocity (current) vs. adding acceleration, bone vectors/lengths, joint angles, wrist–hip/wrist–shoulder distances, symmetry features, motion energy, jerk, per-joint confidence/visibility. Which are redundant or harmful?
- **V17. Smoothing/denoising.** Temporal filtering (One-Euro/Kalman/Savitzky–Golay) of keypoints before the model; effect on jitter-induced FP.
- **V18. Left/right handedness and mirroring.** Is mirror augmentation correct for this task (handedness, scene asymmetry)? Do left-right swaps hurt?
- **V19. Multi-person context.** Should the model receive other people's poses/proximity (cashier, companions, crowd)? Which negatives require it?

**C. Tracking and session construction**
- **V20. Tracker choice and parameters** (BoT-SORT + ReID, `track_buffer`, `new_track_thresh`, `appearance_thresh`): how often do ID switches occur and how do they correlate with alarms? Alternatives (ByteTrack, OC-SORT, Deep-OC-SORT, StrongSORT, pose-based re-identification).
- **V21. Skeleton-jump detection and repair.** Thresholds, split-at-discontinuity policy, simulation for training.
- **V22. Session definition.** Entry/exit logic, minimum duration, per-person sessions in crowds, occlusion-induced fragmentation: effect on windows and FP.
- **V23. Camera geometry.** Differences among the 9 cameras (height, angle, distance, resolution, lighting): per-camera performance, per-camera normalization/thresholds, camera-conditioned models.

**D. Model design and training**
- **V24. Architecture family.** TCN/dilated/ms-TCN/res-TCN, LSTM/GRU, Transformer/Conformer, ST-GCN/CTR-GCN/gat-TCN, PoseC3D, joint-attention, pretrained motion encoders: where is the inductive bias most appropriate for 8–17 joints and 3–8 s windows? Is architecture even the bottleneck (brief: probably not)?
- **V25. Capacity/regularization.** Hidden size, depth, dropout, weight decay, label smoothing, EMA/SWA, early stopping; effect on calibration and FP stability across seeds.
- **V26. Task formulation.** Binary vs. multiclass vs. multi-task (binary head + category head), hierarchical (benign-vs-concealment → robbery-vs-confusables), one-vs-rest, ordinal/risk score; background class; ignore/ambiguous class.
- **V27. Loss function.** CE vs. focal vs. asymmetric (current "hardened") vs. pAUC/recall@FPR objectives vs. constrained/Lagrangian vs. logit-adjusted vs. contrastive/metric losses; neg:pos ratio; per-class costs.
- **V28. Sampling and class balance.** `WeightedRandomSampler`, neg:pos 4–6, source/camera/category stratified sampling, hard-negative mining, curriculum; effect on calibration.
- **V29. Data augmentation.** Which augmentations help or hurt (mirror, rotate, scale, shift, noise, speed, keypoint dropout, temporal crop/shift, ID-switch splice, mixup/cutmix, mass synthetic augment)? Ranges and probabilities; the brief reports no gain from 100k synthetic rows.
- **V30. Checkpoint and model selection criterion.** `val_acc` (current) vs. recall@FPR, pAUC, FP per 1 000 sessions; selection on group-aware/held-out-day sets; avoiding the winner's curse.
- **V31. Hyper-parameter search.** Which hyper-parameters actually matter (lr, batch size, epochs, seq_len)? Seeds variance: how much of the F1/FP difference between models is just seed noise?
- **V32. Pretraining / self-supervision / transfer.** Masked-pose modeling, contrastive pretraining on unlabeled production streams, pretrained skeleton models; expected gain with ~19k labeled clips.
- **V33. Ensembling.** Rule (AND/MEAN/cascade/stacking), number of members, diversity source (architecture vs. data bootstraps vs. input views), thresholds per member; why AND @ 0.80 and is it optimal; correlation of member errors (brief: consensus errors).
- **V34. Calibration and thresholds.** Temperature/Platt/isotonic per model, conformal thresholds, per-camera thresholds, threshold stability across retrainings (Aug pair lost ~9 recall pts on Oct data), calibration set selection.
- **V35. Uncertainty and abstention.** MC-dropout/deep-ensemble variance/energy/OOD scores to abstain or down-weight alarms; selective classification at a given coverage.

**E. Data, labels and evaluation**
- **V36. Data volume per category and learning curves.** Which categories (0/1/2/3/4/5/9/13/14, robbery) limit performance; marginal value of +N clips of each type; stop-adding rules (brief: more generic negatives had no effect).
- **V37. Real vs. staged data and domain gap.** Effect of staged `action-recordings` vs. real sessions (train and eval); re-weighting; domain adaptation; is the continuous-pipeline format (`clip_buffer`) different from curated cuts?
- **V38. Label quality and semantics.** Noise rate per category; ambiguity between 6 / 14 / 3 / 4 / 2 / 5 / 9; need for new classes (near-miss gestures: phone/wallet/keys/glasses, own bag); inter-annotator agreement; cleaning methods (cleanlab, small-loss, loss trajectories, consensus errors across the 20 models).
- **V39. Background/"no-event" data.** Production-style windows without labeled action (PU learning), how many are needed, how to sample, how to prevent contamination by unlabeled robberies.
- **V40. Split strategy.** Group by session/recording/day/camera/person; leave-one-camera/day-out; test-set discipline (currently never used); size needed for the target FP resolution (1 FP per 1 000 requires ≥ ~3 000 negative sessions for a CI of ± 0.1 %).
- **V41. Evaluation unit and metrics.** Clip-level vs. chunk-level vs. session-level vs. event-level; alarms/day, FP/1 000 sessions, recall per event, time-to-alarm, severity tiers (near-miss tolerated vs. unrelated movement); confidence intervals; stability metrics (seed/bootstrapping, perturbation consistency).
- **V42. Prevalence and base rate.** How to translate val metrics to production at ~0.5 % prevalence (or lower); expected precision; cost model for FP vs FN.
- **V43. Replay benchmark design.** How many stored sessions, which days/cameras, how to label efficiently (SAM3/HAR assisted + human sampling), how to avoid leakage between hard-negative mining and evaluation.
- **V44. Single-person training vs. crowded production.** Effect of the `single_user_only` filter and of occlusion by other people; how to simulate.

**F. Pipeline-level (stage 2 and fusion)**
- **V45. Stage-1 operating point.** Should stage 1 favor recall (more candidates) with a strong verifier, or stay at low FP? Optimal split of the FP budget between stages given measured HAR/SAM3 rejection and false-reject rates.
- **V46. HAR design.** Which gestures/rules reduce FP most; thresholds; behavior when hands are not visible; learn them from data instead of rules?
- **V47. SAM3 usage.** Object-state reliability by camera/lighting/occlusion; failure modes (`skipped/no_hands_in_roi`); distillation into a lightweight per-window model; use on every window vs. only on candidates; latency/cost.
- **V48. Fusion logic (C3).** Rule-based priority vs. learned calibrated classifier; features; cost-sensitive objective; how to include pose quality, zone, dwell time and camera.
- **V49. Alternative/complementary filters.** Locomotion gate, zone/ROI/dwell, ordering reach→conceal, per-session cooldown, multi-camera consensus, ownership/bag cues, "carries objects" state (cats 5/9): which are worth building?
- **V50. Policy layer.** Gray zone/soft alerts, severity levels, review queue, per-camera thresholds, cooldowns; effect on staff-perceived annoyance vs. detection.

**G. Engineering and reproducibility**
- **V51. Train/inference parity.** Same code path for preprocessing in training and production (normalization, velocity, resize, NaN handling)? Version pinning; golden test vectors that must give identical outputs offline vs. online.
- **V52. Numerical/precision issues.** FP16/INT8 engine vs. FP32 training effects on keypoints and on the classifier logits.
- **V53. Latency and compute budget.** Cost of 3 s multi-scale evaluation, ensembles, extra keypoints, hand ROI/SAM per window on the real-time system.
- **V54. Monitoring and drift.** Online metrics (alarm rate per camera/hour, score distribution drift, keypoint-quality drift), automatic hard-negative harvest, retraining cadence, rollback rules.
- **V55. Privacy/legal constraints** on storing sessions/video for retraining (affects the replay and mining loops).

**H. Meta-questions the expert must also answer**
- **V56. What is the theoretical ceiling** of 8-keypoint upper-body 2D pose for separating 6 from {14, 3, 4, 2, 5, 9, near-miss gestures}? Which information is provably missing?
- **V57. Which 3 variables above would you change first** to move the FP–recall frontier, and what does your evidence say about their expected effect size?
- **V58. Which variables are interacting** (e.g., fps × seq_len × window length × normalization × velocity), and what factorial/ablation design isolates them with a small compute budget (given ~430 min per training cell)?
- **V59. What are the noise floors** (seed variance, split variance, CI width) below which differences should be ignored, and how many seeds/splits are needed?
- **V60. What did we not think of?** List any additional variables, failure modes or data sources that could affect quality.

**I. Production-specific questions raised by the owner (round 4–5)**
- **V61. Production chunking design (§0b.7).** Independent fixed 8 s chunks vs overlap vs sliding window vs streaming/causal model vs one session `.npy`; chunk length/stride; per-session aggregation; latency at ≈ 66 % event recall; handling of 1-chunk sessions and 43-chunk employee sessions; FP exposure growing with session duration.
- **V62. Training-by-action vs production-by-chunk (§0b.1).** How to build training windows with action-level labels only (MIL, synthetic embedding into background windows, background negatives from production); which window sampling reproduces production.
- **V63. Pose-engine consistency (§0b.8).** Effect of different `.engine` builds (precision, `imgsz`, versions, tactics) on keypoints/NaN/IDs/scores; extraction contract and drift test.
- **V64. Triton preprocessing parity and short/truncated chunks (§0b.9, §0b.10).** Missing `valid_mask`, path A vs B, zero-fill in centering/scale, frozen tail frames, perspective scale change; the corrected contract and a golden-tensor test.

---

## 13. Reproducibility pointers

**Included with the brief:** `SAMPLE_production_chunk_mergedfps.npy` (same folder as this file).
**Repo:** `/home/debian/dev/Proyectos/GuadIA-IAModel/IAModel/experiments/`
- Extraction: `pose_extractor_clean.py`, `config.py` (`PATH_ROOTS`, `OUTPUT_BASE`, `YOLO_POSE_MODEL`, `CLIP_SCALE_HEIGHT`), `custom_tracker.yaml`, `NPY_POSE_VISUALIZER_FRONTEND.md` (npy spec).
- Training: `training/model_config.py` (60 experiments, `ROBBERY_CLASS = 6`, split heuristics), `training/train_model_operations.py` (`collect_examples`, `PoseDataset`, `normalize_sequence`, `add_velocity`, `temporal_resize`, losses), `training/evaluate_validation.py` (**multi-root bug** in `build_split_examples`), `training/test_model.py` (live video), `training/clean_theft.py` + `campaign/pose_robbery_heuristics.py` (kinematics), `models_extra/movement_states.py`.
- Campaign: `training/campaign/run_campaign.sh`, `campaign_config.json`, `preflight_campaign.py`, `train_campaign.py`, `evaluate_campaign.py`, `summarize_campaign.py`, `sliding_window_eval.py`, `export_ensemble_fp.py`.
- Docs: `campaign/PLAN_REDUCCION_FP.md`, `campaign/INFERENCE_ENSEMBLE_HANDOFF.md`, `campaign/SLIDING_WINDOW_EVAL_INTERPRETACION_campaign_20260714_164642.md`, `campaign/IMPROVE_RUNS.md`.

**Artifacts (analysis machine; may not be accessible to you — if so, rely on the numbers quoted in this brief and mark claims as unverified):**
- Aug: `/home/debian/Documentos/Kanvis/artifacts/runs/campaign_20260805_220606/`
- Oct: `/home/debian/Documentos/Kanvis/artifacts/runs/home/angel/IAModel/experiments/training/campaign/artifacts/runs/campaign_20261006_231053/` (`plans/<cell>/training_plan.json` has `split_uids`, `split_stats_by_category`; `reports/<cell>/val_errors_*.csv` columns `uid,true_label,folder_category,prob_pos,logit_margin,pose_path,clip_path,clip_video_path,clip_dir,clip_name,category_str,meta_json_path,user_dir,model_path,outcome,pred_label`; `reports/<cell>/val_ensemble_grid.csv` with `cell_id,decision_mode,models,thresholds,tp,fp,fn,tn,precision_pct,recall_pct,f1_pct,fp_rate_pct,support_pos,support_neg`; `reports/_master/*`).
- Saved FP/FN lists: `/home/debian/Documentos/Kanvis/artifacts/runs/campaign_20261006_231053_lists/` (`fp_41_48_and_0.80.txt`, `fn_41_48_and_0.80.txt`).
- Real-time samples: `/home/debian/Documentos/Kanvis/falsorobo/` (`p1.json`, `meta.json` = `session_meta_v2`, `chunk_001/`, `merged.npy`, `frontend_playback.json`), `/home/debian/Documentos/Kanvis/p1.json`.
- Earlier runs: `fp_pipeline_v1_analysis/`, `eval_pack_campaign_20260714_164642/`, `Despliegue modelos/v1_16Julio`.

**How the audit numbers were computed:** parsing `split_uids` of each `training_plan.json` (source, camera, session, chunk times from the clip name), joining with `val_errors_*.csv` (`outcome` = `FP`/`FN`), Wilson intervals, FP/FN overlap across the 20 per-model error files, frontier from `val_ensemble_grid.csv`, plan-vs-eval pool sizes from `training_plan.json` + `val_eval_summary.json`. Evaluation commands: `cd experiments/training/campaign && ./run_campaign.sh eval-fg --run-id <RUN> --cells <cell>`; ensemble export: `python export_ensemble_fp.py --run-id <RUN> --cell <cell> --split val --models modelo_41 modelo_48 --rule and --threshold 0.80`.

**Caveats of this document:** source-type interpretations (staged vs real), category semantics, production pose model, and `robbery_score` semantics are inferred; the 434-clip evaluation gap and consensus-error statistics are verified from artifacts; the production-math in §8.E assumes independence and clip-level FPR transfer and is an *order-of-magnitude* illustration, not a measurement.
