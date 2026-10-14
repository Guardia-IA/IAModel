# KANVIS — Expert response and implementation plan

**Date:** 14 October 2026  
**Scope:** one store, nine cameras; stage-1 pose detection and the HAR/SAM3/C3 cascade.  
**Inputs:** the complete v5 brief and the supplied production `.npy`. The repository, checkpoints, raw videos, timestamps and campaign CSVs referenced in the brief are **not available in this workspace**. They were not inspected. No training was run, no production code was changed, and neither input was modified.

**Evidence labels used throughout:** **Measured here** = recomputed from the supplied array or explicit counts; **Reported** = supplied by the brief, without access to its underlying artifacts; **Inference** = interpretation; **Planning target** = an experiment or acceptance goal, not demonstrated performance. References `[B §…]` identify sections of `KANVIS_CONTEXT_AND_RESEARCH_BRIEF.md`; source line numbers refer to that unchanged file. Public primary sources are listed in §13. No efficacy number from a public benchmark is assumed to transfer to this store.

## Executive summary

The system is being asked to make a distinction that eight body points cannot always show: putting a store product into a bag can have the same arm motion as putting away a personal item. However, this is **not a reason to accept the present production false-alarm level**. The first problem is that the comparison mixes one prediction per validation clip with many predictions per production session. The next is a plausible, large preprocessing fault: absent joints become zeros before centering, and production may retain frames that training removes. The supplied sample demonstrates severe artificial motion under this preprocessing. Action-centered training versus arbitrary production windows, tracking switches, and reuse of validation for thousands of selections compound the problem.

Two corrections change the plan materially. First, some listed false positives have a supposed ensemble-member probability below 0.80; those cannot be alarms of a correctly joined AND@0.80 ensemble. The error export and exact deployed rule must be reconciled. Second, stage-1 recall of 66% followed by a verifier retaining 90% of true candidates yields **59.4% final recall**. To finish at 66%, stage 1 must reach at least **73.33% event recall** if verifier retention is 90%.

**The recommended course is one sequence of work:**

1. Prove input-tensor, score, label-map and alarm-rule parity; repair multi-root evaluation; reproduce production at the session level.
2. Keep the existing 15-fps system for the immediate parity correction. For the next training version, use a **timestamp-based 15-Hz contract**, 4-second evidence and 8-second context, evaluated every second from a per-track ring buffer. Storage chunks can remain eight seconds.
3. Train the existing compact model family on production background and action-level positive bags, with missingness masks, actual-time velocities, preserved global motion, and group-separated evaluation. Mine difficult negatives automatically but verify suspicious labels; SAM/HAR rejection is not ground truth.
4. Measure HAR/SAM3 on the same candidates and real events. Add object continuity and missing-evidence features, then a small calibrated C3 fusion model. Keep uncertain cases visible in coverage and workload metrics.
5. Freeze the entire policy and validate on later days, with sufficiently many negative sessions and independently labeled theft events.

**The business acceptance point is ≤1 false staff alarm per 1,000 non-theft camera sessions and ≥66% event recall.** A useful intermediate target is stage 1 ≤10 false-candidate sessions per 1,000 at ≥74% event recall, followed by ≥90% rejection of those false candidates and ≥90% true-candidate retention. This would give about **1 FP/day and 66.6% recall** at 1,000 sessions/day. It is a coherent target, **not an achieved or guaranteed forecast**. An interim stage-1 target at 66% remains useful for development, but does not by itself support the final recall requirement.

No evidence currently establishes that the complete system can attain this point. The cheapest way to resolve that uncertainty is a replay with exact production inputs and independently verified event labels, not another broad architecture campaign.

## 1. DIAGNOSIS — ranked causes, with evidence and falsification checks

### 1.1 What most likely explains the validation-to-production gap

The reported 11/1,619 is **6.794 FP per 1,000 negative clips**. The reported 20/654 is **30.581 false alarms per 1,000 all sessions**, if 20 means 20 distinct falsely alarmed sessions. Their ratio is 4.50, but their units differ. If 20 counts repeated windows rather than distinct sessions, even the latter interpretation changes. All 654 may also include some theft sessions. These denominator checks precede causal attribution.

| Rank | Cause | Evidence and confidence | Estimated contribution, honestly bounded | Cheapest check; result that changes this ranking |
|---|---|---|---|---|
| 1 | Multiple opportunities per session and inconsistent denominators | Reported independent chunk and overlap scoring; session lengths include 1, 4 and 43 chunks. High confidence that units differ; no measured within-session dependence. | **Could account for the entire apparent 4.5× gap** without any increase in window FPR. With independence, 4.56 equivalent looks map 0.6794% to 3.0581%. This is an illustrative fitted exposure, not the measured number of looks or a causal share. | Rejoin raw window scores to sessions; count distinct false sessions, alarms and look counts. If one-look production negatives themselves have much higher FPR after matching duration/quality, this cause cannot explain the residual. |
| 2 | Triton preprocessing and missing-joint artifacts | Measured on sample: all-frame zero-fill scale 0.06604372 versus 0.05038930 after dropping <5-visible-joint frames, +31.07%; artificial normalized velocity spikes. Actual live mask/path remains unknown. High confidence in artifact, medium confidence in deployment mismatch. | **No defensible population percentage.** Could dominate short/occluded false alarms; could explain none of the deployed gap if live tensors already match. Budget the first engineering day here because the test is cheap and decisive. | Capture tensors immediately before forward in training, Triton A, Triton B and TensorRT. Match exact sample/checkpoint; paired replay with only parity repair. Near-identical live tensors and scores lowers this rank. |
| 3 | Action-centered supervision versus arbitrary windows, duration warping and weak labels | Reported training unit is one action, one label; inference slices arbitrary context. 0/8 reported production-format positive detections versus about 65% for other positives is a serious warning. High plausibility; size unmeasured. | Likely a major **residual** after exposure/parity correction. A 20–50% relative reduction in false-candidate sessions is a **trial success target**, not a measured attributable share. Positive event recall may also improve substantially. | Recreate production geometry from real streams, compare legacy versus production-window training on locked days. If geometry-matched negatives and real event recall do not improve, semantic/context limits rise in priority. |
| 4 | ID switches, fragmented tracks, occlusion and perspective/normalization errors | ID switches are owner-confirmed; sample shows visibility loss, frozen repetitions and changing scale. No identities/video provided. | Unknown; potentially concentrated in a small but alarm-heavy quality slice. A quality gate can remove alarms at a recall cost; count that cost instead of declaring it free. | Review a stratified set of all alarm tracks and matched non-alarm tracks; compare switch flags and paired split-at-jump replay. Failure to associate switches with alarms reduces rank. |
| 5 | Validation optimism, leakage and incomplete evaluation | Reported 434/2,376 validation UIDs omitted, clip grouping leakage, best checkpoint/pair/threshold selected on reused validation, thousands of configurations. FP probability table internally inconsistent. | Explains an unknown portion of the apparent validation advantage and ranking instability, rather than directly generating production alarms. **No validated numerical share.** | UID/logit rejoin; assert plan/eval set equality; test the frozen policy on a later-day group holdout. Similar holdout results reduce its role in the gap. |
| 6 | Pose-engine build, resolution, fps or class-map differences | Different engine files confirmed; input shape/build/weights/precision not supplied. Filename/family equality does not prove weight equality. NVIDIA documents numerical variation between builds [S1]. | Usually lower-priority than gross masking/timing faults if extraction settings and weights match; potentially large when resolution, confidence or weights differ. Unknown share. | Re-extract the same videos under matched engines, first with fixed detections/tracks, then end to end; measure threshold flips. Tiny paired alarm deltas demote this cause. |

**Commitment:** the leading explanation of the *reported gap* is exposure/metric mismatch, followed by preprocessing and training-window mismatch. The leading explanation of **arbitrary, unstable-looking movement alarms** is missingness/tracking/representation faults. The leading explanation of **plausible bagging/concealment false alarms** is missing object and transaction context. These statements answer different questions and should not be merged into one list of mutually exclusive causes.

There is no basis for assigning additive percentages of the 20 alarms to these causes. The same alarm may have an ID switch, missing wrists, repeated evaluations and a confusable label. Use paired interventions and manually coded overlapping cause indicators; report unique removed sessions and changed true detections after each intervention. A numerical pie chart here would fabricate evidence.

### 1.2 What explains the validation floor

1. **Missing semantic information and ambiguous supervision.** Categories 6, 14 and part of 3 can share pose and object trajectories while differing in product ownership, payment or legitimate bagging. That missing variable cannot be recovered reliably from upper-body coordinates. Cat-14 3/25 is reported, not independently verified, and its 95% Wilson interval is roughly 4–30%; the point estimate alone is unstable.
2. **Weak action-level labels, timing/visibility artifacts and training shortcuts.** A positive action clip does not label each prefix as positive. Short curated clips and long negative clips may teach padding/duration rather than concealment. Zero-fill also corrupts some valid training frames.
3. **Too few independent production-format theft examples and difficult background examples.** There are 945 reported real-origin positive clips, not merely 69; only 69 are in the continuous pipeline format. Counts of independent events may be much lower. Adding 18 positive clips does not supply a strong recall-learning experiment.
4. **Selection and sampling uncertainty.** At 11 FP, moving one or two errors is not credible evidence of superiority. Consensus across shared-data models supports a hard-example population, not a proof of mislabeled data, independence, impossibility or an architecture ceiling.

**The attainable pose-only ceiling is numerically unknown.** The campaign data support deprioritizing broad architecture searches; they do not prove that the frontier is fixed or that additional useful negatives cannot help. The evaluation pools changed and the new root was omitted.

## 2. SOLUTION — one end-to-end plan

### 2.1 Phase 0: trustworthy replay and parity before model changes

Freeze the currently deployed weights, extraction engine, preprocessor, label mapping, AND rule and stage-2 versions. Record hashes. Recompute both members from the same UID/input and enforce `candidate == (p1 >= 0.80 and p2 >= 0.80)` and `score == min(p1,p2)`. A below-threshold member in an alleged AND false positive fails this gate. Repair the multi-root collector and require exact equality of planned and evaluated UIDs; duplicate UIDs across roots require a deterministic provenance-aware key, not silent overwrite.

Run the supplied audit on the sample. Then capture **real** tensors from the four inference paths. A reimplementation from prose is a diagnostic illustration, not proof of production parity. Maintain two different contracts: **legacy-v1 parity**, reproducing the old checkpoint's training transform exactly, and **timestamp/mask-aware v2**, used only with newly trained checkpoints. Repair a demonstrated live-v1 mismatch and replay it; do not change normalization or velocity units under the old checkpoint while retaining its threshold.

Build a development replay of about 1,000 randomly selected sessions from all nine cameras and multiple days, enriched separately with independently labeled real theft events. Preserve time, track interruptions, candidate arrival time and stage-2 outputs. Run the same policy at every stage, with one common session/event identity. Exportable `p1.json` is useful but not required: if unavailable, replay raw videos or frame-aligned poses and have the harness write a minimal JSON/CSV ledger. The existing skeleton uploader is **not** a labeling or review tool. If video cannot be accessed, pose-only metrics can be computed but theft semantics and end-to-end stage-2 performance remain unverified.

### 2.2 Phase 1: production-matched candidate generator

**Temporal contract:** retain native 15-fps tracking and use a timestamp-based 15-Hz classifier grid. Preserve gaps and masks. New inference uses one trailing 8-second per-track ring buffer, updated every second, with an 8-second context head and a 4-second evidence head. At 15 Hz these are 120 and 60 grid bins. Storage remains chunks of up to eight seconds; file boundaries have no predictive significance. At entry/exit use explicit masks and the short evidence head; never stretch four seconds to eight or classify a several-minute session after uniform resizing.

**Representation:** start with eight shoulders/elbows/wrists/hips, explicit per-joint visibility, optional genuine confidence, torso-anchored local coordinates with smoothed observed torso scale, velocity per second computed **after** timestamp resampling, and a separate global branch for center trajectory, scale change, quality and zone. Missing local joints become neutral coordinates only after masked normalization, with a mask that distinguishes absence from a genuine center point. Keep the time axis when a frame is invalid. Short gaps up to 0.2 seconds may be interpolated only with same-track valid endpoints, while retaining original-observation flags. Longer gaps terminate evidence continuity. Treat the threshold/coverage settings as development parameters, not anatomy guarantees.

Save all 17 COCO keypoints when extraction is next run because YOLO already computes them. Do not require 17 visible points to keep a frame, do not drop hips from the eight-point baseline, and do not make a whole-body pose model the first investment. Additional lower-body/head features are optional masked channels. Object appearance is the more relevant new semantic modality.

**Supervision:** negative bags are complete independently reviewed non-theft sessions, sampled with random offsets, transitions and real occlusions. Positive bags use the existing whole-action labels: overlapping subwindows with a bag-level MIL loss, rather than declaring every positive crop positive. A short positive clip whose decisive moment is unlocalized remains a positive bag with uncertain instance labels. Add a modest human annotation effort for take/conceal intervals and object evidence on real events; do not assume pre-existing frame labels. Synthetic embedding into background is a limited robustness ablation with continuity checks, not the principal source of event labels.

**Generalist data loop:** start with a reviewed 1,000-session pilot, then expand to 5,000–20,000 development-only raw/weakly labeled production sessions, caps per camera/day/session and broad random background, plus verified difficult alarm/near-threshold examples. Complete manual review of the expanded pool is not assumed. Do not turn every unlabeled session or SAM/HAR-rejected alarm into a hard negative. The selected alarm pool is enriched for theft and occlusion. Use unknown/weak labels, confidence-weighted training and review of candidates; compare supervised learning against a prior-sensitive PU objective only after estimating the **window-level** positive prior. The real/staged positive selection mechanism violates the simplest PU representativeness assumption [S2].

**Architecture and objective:** retrain the compact TCN-GRU/TCN-LSTM baseline under the new contract before adding one masked graph baseline. A shared multi-scale temporal encoder with an auxiliary action-category head is sufficient for the first design. Retain bag-level binary supervision, an auxiliary category loss weight around 0.1–0.3 as an ablation, and a CE baseline against a moderate-tail pAUC/hard-negative objective. Use a negative memory bank or large negative evaluation pool; minibatch 64 cannot stably optimize a 0.02% false-positive tail. Choose checkpoints by session-level recall under a candidate budget on group-separated development days, not validation accuracy.

**Selection:** two seeded finalists initially, three seeds for a surviving recipe and five if differences are small. Select on disjoint groups and calibrate the frozen aggregator on different negative sessions. The numeric threshold may change after retraining; stable risk is the goal, not preserving 0.80. Keep the existing two-member AND as a comparator, then test input/data diversity. Optimize final candidate episodes, not independent window FPR.

### 2.3 Phase 2: object evidence and calibrated fusion

Run HAR/SAM3 on every stage-1 candidate episode, including true events, occlusions, skipped reviews and engine failures. Estimate **false-candidate session rejection** and **true-event retention** jointly, with camera/visibility strata. HAR and stage 1 both use pose and can share failure modes; separate software does not make them statistically independent. SAM3 segments/tracks objects [S3]; Kanvis's object-state logic is a separate application layer and does not establish product ownership or payment.

The next new feature is **causal object continuity**: an object observed at a shelf, associated with a hand, then moved toward a bag/body or a cart, with explicit uncertainty when visibility is lost. Evaluate across the event rather than judging only the final frame. Disappearance alone is not concealment. A small RGB wrist/hand/bag ROI branch, trained on reviewed object states, can complement or later distill this signal. Start on candidates to keep compute predictable. Run a student on every window only if the replay demonstrates that doing so rescues stage-1 misses at acceptable false alarms and capacity.

C3 first uses regularized logistic regression on member margins, multi-scale/time summaries, HAR evidence, SAM states/confidence/availability, pose quality, track discontinuity, scene zones and object continuity. With sufficient grouped data compare small boosted trees; defer a score-GRU until there are enough independent positive events. Use camera identity cautiously; geometry and availability features are more transferable than an almost-negative camera prior. Explicit missingness indicators are mandatory.

For deployment, require either corroborated evidence or a calibrated high-specificity pose/HAR route for cases without visual evidence. **No hands visible is unknown, not benign.** Unknown/abstained events count in total recall and review workload. A review queue receiving twenty alerts is still twenty staff interruptions if staff must inspect them; renaming hard alerts does not satisfy the budget.

### 2.4 Quantitative gates and realistic expectations

| Milestone | Stage-1 false candidate sessions /1,000 negatives | Stage-1 event recall | Stage-2 false-candidate rejection / true-event retention | Final outcome at 1,000 negative sessions/day |
|---|---:|---:|---|---|
| Reported current system | About 30.6 using 20/654; uncertainty and denominator caveats apply | **Unknown**; clip recall is reported 63.8% | Unknown | End-to-end FP and recall unknown |
| Early development gate | ≤10 | ≥66% | Measure, do not assume | Useful stage-1 progress; cannot establish final ≥66% |
| Preferred acceptance design | ≤10 | ≥74% | ≥90% / ≥90% | ≤1 expected FP/day; ≥66.6% recall |
| Conditional fallback if background gains are smaller | ≤20 | ≥74% | ≥95% / ≥90% | ≤1 expected FP/day; ≥66.6% recall, with a harder verifier burden |
| Current exposure level with strong verifier | 30.6 | ≥74% would still have to be demonstrated | ≥96.73% / ≥90% | ≤1 expected FP/day; ≥66.6% recall |

All targets apply to the **same complete pipeline and event definition**. False candidates can create several verifier jobs per session unless deduplicated. The table describes session rejection, not just rejection of individual windows. No measured evidence yet supports any of its future rows.

The initial parity/replay work should take days if exportable data and videos exist; one constrained retraining round and candidate-stage measurement roughly one to two engineering weeks; object-state labeling/student work can take several additional weeks. These are planning estimates, not commitments based on inspected infrastructure. Engineering effort should be stopped or redirected at failed gates, not expanded into a 60-model search.

## 3. ANSWERS index

| Requested coverage | Location in this response |
|---|---|
| All structural facts §0b.1–0b.10 | §4, one row per item; §5 sample; §9 temporal design |
| §12.1 open and partially resolved owner items | §6, explicit identifiers |
| Research questions §12.2, 1–16 | §7, R1–R16 |
| Design variables §12.3, V1–V64 | §8, one numbered answer per variable |
| Proposals P0–P9 | §10.1 |
| E0–E10, E-fps, E-replay, E-bg, E-parity, E-chunk, E-engine | §10.2 |
| Additional ideas and acceptance roadmap | §2, §10.3 and §11 |
| Assumptions, unresolved measurements and conditional checks | §12 |
| Primary sources, scripts, files inspected and recomputed numbers | §13–§14; `kanvis_tools/` |

## 4. Critical review and explicit answers to §0b

| Item | Direct answer and correction |
|---|---|
| 0b.1 — action versus fixed chunk | Yes, a likely major distribution/supervision mismatch. Use production-style negative sessions and positive MIL bags from action clips. Prefer real re-extracted context if available; no inherited frame labels exist. Cropping a positive action clip does **not** make every crop positive. |
| 0b.2 — full versus filtered | If `poses_full[valid_mask] == poses` and metadata/augmentation are equal, tensors should be identical. Verify every paired UID and mask, then retain one canonical raw-full source plus explicit timestamps/masks. Treat the two campaign cells as replicates, not input modalities. Do not attribute their winner change to more information in `full`. |
| 0b.3 — changing winners | Most plausibly seed/sampling/split variation plus selection over many correlated configurations, with calibration and data composition changes. Lock common groups and compare paired predictions across 3–5 seeds. Exchangeability of cells is a hypothesis until tensor equality is demonstrated. |
| 0b.4 — fps and compression | Keep 15 Hz first; new 4/8-second windows contain 60/120 bins. Current 64/96 models map varying real durations into fixed token counts; their effective rate is not simply 64/15 or 96/15 seconds. Position token timing and pre-resize velocity units can disagree. See §9. |
| 0b.5 — production versus validation FP | 20/654 sessions and 11/1,619 clips are not comparable rates. An effective 4.56 independent looks would reconcile them, but actual looks and dependence must be replayed. Business acceptance uses sessions and staff alarms, not a clip target. |
| 0b.6 — generalist requirement | Meet it with broad continuous background, production nuisance robustness, object/scene evidence, verified mining and explicit uncertainty. No need to invent a named class for every scratch/phone/back gesture. No pose-only method can reject every unseen benign movement while retaining all matching theft trajectories. |
| 0b.7 — chunk design | Eight-second storage is reasonable. Independent eight-second decisions are not the preferred detector design. Separate storage from scoring; use a rolling 4/8-second view every second with one candidate episode and causal aggregation. Longer employee tracks retain proportional exposure accounting and do not become automatically benign. |
| 0b.8 — different engines | Can matter, but different bytes alone do not establish meaningful drift or identical weights. Pin weight hash, engine hash/build manifest and extraction settings; paired re-extraction and alarm flips decide the magnitude. Distinguish pose-engine drift from classifier `.pt`→`.engine` drift. |
| 0b.9 — production sample | Verified shape, missingness, repetitive tail and zero-fill artifact. Duration/fps, true movement, lower-image exit and cause of repetition cannot be established without timestamps/video. Vertical shoulder/hip midpoint separation grows about 2.130×; Euclidean midpoint distance about 2.132×, not 2.5×. See §5. |
| 0b.10 — Triton preprocessing | Highest-priority code-level audit. No supplied mask in the simulated directory is a serious suspicion, not proof that the live path omits it. A/B should be tested on exact captured tensors; neutral filling and timestamp-preserving masks require v2 retraining. Current padding may repeat nonzero velocity if it repeats already-augmented feature rows. |

Additional corrections to the brief:

- **Impossible AND probability rows:** B lines 493–508 list values 0.11, 0.14, 0.15, 0.20, 0.25, 0.72 and 0.74 for alleged ensemble false positives. If they are either actual member's positive probabilities at that same run/input, AND@0.80 cannot fire. Treat FP identities, their joined member scores and related consensus attribution as provisional until reconciled. The aggregate confusion matrix may still be right; these rows do not independently prove it.
- **The independence formula is not an upper bound.** The valid union bound is `P(any FP) ≤ min(Σ p_i,1)`; dependencies can make the independence calculation too high or too low. More looks do not create independent evidence.
- **Eight misses are not safely dismissed.** Conditional on eight distinct independent events, 0/8 yields a two-sided exact 95% recall upper bound of 36.94%. At true recall 206/323, eight misses have probability about 0.000296. Correlated clips, mislabeled intervals, different input formats or selection could explain it; it warrants an urgent audit, not a reliable extrapolation to all real thefts.
- **Equal counts are not a learning-curve result.** Aug and Oct tested different clips, 434 Oct cases were missing, and most extra data were negatives. Identical 11 FP/117 FN does not prove the frontier cannot move.
- **Model agreement is not independence.** All models share representations, labels, provenance and much of their training procedure. Agreement can reflect real ambiguity, shortcuts, bad extraction or wrong labels.
- **Padding claim requires code inspection.** B lines 303–304 put velocity before resize, yet describe a zero-velocity padded tail. Repeating the final `(x,y,dx,dy)` feature row repeats its last velocity unless explicitly reset. The provided audit follows the literal described order and flags this difference; actual source is unavailable.
- **Three thousand negatives do not give a generic ±0.1% interval.** For 3 FP/3,000 the Wilson 95% interval is about 0.0340–0.2936%. Zero FP in 2,995 independent negative sessions is enough for a one-sided 95% upper bound below 0.1%; this is an upper-bound calculation, not a precise estimate or protection against shifted days.
- **Calibration is not a new feature.** Monotone temperature scaling does not improve a single-model ROC ranking. Conformal coverage is not automatically session FPR or a daily hard cap. Audit the complete loss/decision unit [S4].
- **Meaning of robbery labels:** this system detects suspected shoplifting/concealment; the visible action need not prove a completed theft or intent. Keep the operational event label separate from an assertion about a person's guilt.

## 5. Production sample — measurements and what they actually establish

The supplied file was opened with `np.load(..., allow_pickle=False)`. Its SHA-256 is `e34b166de168a6ca932a37b1b124ce436bf0ff64c72713ba91b57dbb5dd1b493`. See `kanvis_tools/sample_pose_audit.json` and the runnable audit for exact values.

| Quantity | Measured result | Interpretation |
|---|---:|---|
| Shape / dtype | `(60, 8, 2)` / float32 | Eight joints and 60 records; no embedded timestamps, confidence or video geometry |
| NaN coordinates | 78/960 = **8.125%** | Visibility must remain explicit; zero is not an observed joint |
| Full visibility | Frames 0–51: eight joints | Most of the early sequence is clean |
| Tail visibility | 52: six; 53: five; 54: four; 55–59: two | Six frames fail the legacy ≥5-joint rule; frame 53 still contains missing joints |
| Exact repetition | Frames 55–59 are bit-identical | Repetition is real; exporter hold, repeated video, tracker caching and genuine low-resolution stasis cannot be distinguished |
| Mean coordinate shift, good first 52 frames, keeping all versus legacy filtering | **0.13215953** mean absolute coordinate difference; **0.20466962** mean joint Euclidean difference | Tail visibility changes clip-wide scale and therefore even clean early features |
| Legacy reconstructed global scale | All frames: **0.06604372**; ≥5-visible frames: **0.05038930** | All-frame scale is **31.07% larger** |
| Maximum absolute reconstructed velocity | All frames: **6.71575**; drop invalid: **8.74528** | Dropping low-visibility frames alone is insufficient; zero-filled wrist at last retained frame creates a larger spike |
| Clean retained prefix maximum absolute velocity | About **0.27305** before final two retained frames | Tail artifacts are tens of times larger than preceding articulation changes |
| Vertical shoulder-to-hip midpoint separation | First fully visible: **0.06406249**; last fully visible: **0.13645834**; ratio **2.13008** | Euclidean distance is0.06406613→0.13657993, ratio2.13186. Neither agrees with an initial0.054/2.5× torso claim |
| Other apparent-size definitions | Mean radius about **2.42×**; final/initial shoulder width about **3.03×** | Scale-change magnitude is definition-dependent and can include articulation/orientation |
| Conditional time at 15 Hz | 60 bins cover nominal **4.0 s**; first-to-last sample span **3.933 s** | At 12 Hz the analogous values are 5.0/4.917 s. True rate cannot be recovered from frame count alone |

The finite coordinates occupy the upper half of normalized image height: maximum y is approximately 0.4948. Therefore, **leaving the bottom edge of the full image is not demonstrated by this file**. A crop/ROI transformation, occlusion or confidence loss could explain disappearing hips. Approaching the camera is plausible from the increasing apparent body size, but video is required to distinguish depth, crop changes and articulation.

![Measured sample visibility, geometry and preprocessing artifacts](kanvis_tools/sample_pose_diagnostics.png)

The diagnostic figure uses **frame index**, not invented time. The equations reconstruct the preprocessing described in the brief, not unavailable repository code. No checkpoint was supplied, so neither score shifts nor an actual alarm on this sample were measured.

With 54 retained frames, literal feature-row padding would add 10 rows for #41 and 42 for #48. If the last velocity of 8.74528 is repeated, the artifact persists throughout that artificial tail. If the real implementation zeroes padded velocity, this particular amplification does not occur. This is an explicit source-verification branch, not an asserted production bug.

The v2 remedy is **mask-aware, timestamp-preserving preprocessing**, real production windows, scale-change side channels, edge/occlusion examples and causal handling of missing tails. Per-frame scale normalization alone is not sufficient: a missing wrist can still distort the anchor, and an uncertain torso estimate can inject new noise.

## 6. Answers to §12.1 — owner items and remaining gaps

| Identifier | Decision or conditional answer; cheapest verification |
|---|---|
| Q-fps1 | **15 Hz end to end is the preferred immediate and next-version contract** because labeled extraction is reported at 15 Hz and cameras already supply it. Fixed 12 Hz is adequate for many coarse retail arm motions and reduces sequence bins 20%, but it requires joint offline conversion/retraining and does not add semantics. Choose 12 only if a measured sustained-throughput or timestamp-quality limit warrants it. |
| Q-fps2 | Mismatch matters to timing/velocity and tail calibration; a universal recall penalty cannot be inferred. For the same real motion, native per-frame deltas at 15 Hz are approximately 0.8× those at 12 Hz; reverse is 1.25×. If resampling positions then recomputing per-second velocity is correct, this unit mismatch disappears. Test 10/12/15/18-Hz and clock/skip perturbations with current weights, and matched 12/15 training finalists. |
| Q-G | Use reported 20 FP/654 **sessions**, stage 1 only. Audit whether these are distinct false sessions or 20 repeated candidate windows, and count genuinely negative sessions. No corrected aggregate can be computed without the ledger. |
| Q-F | No per-action phone/back examples are assumed. Broad reviewed normal-session bags and general object/motion features satisfy the stated generalist goal. Review automatic hard negatives by outcome, not by inventing an exhaustive action taxonomy. |
| Q-H | Existing viewer only renders skeletons. If exportable poses/verdicts/video exist, add a lightweight review screen with video, synchronized pose, candidate times, labels, reviewer/version and session links. If verdict exports do not exist, run inference offline to create the ledger. If only `.npy` is exportable, restrict claims to pose replay and explicitly leave visual-verifier/theft semantics unvalidated. |
| Q-I / Q-engine | A different build can change confidences and keypoint coordinates; model family does not establish identical weights. Freeze hashes/geometry/precision; paired extractor and classifier tests isolate each effect. Larger resolution/NMS/letterbox differences can dominate floating-point tactics. No engine-to-engine effect size is measurable from the bare pose array. |
| Q-J | Variable frame count is compatible with a short session; it does not prove variable fps. Adopt reported 15 Hz only as an assumption when timestamp files are unavailable. Verify timestamp median intervals, gaps, duplicates, monotonicity and clock domain before resampling. Metadata 12/15/18 discrepancies require trace reconciliation, not choosing whichever looks plausible. |
| Q-K / Q-chunk | Keep eight-second storage; replace file-boundary predictions in the next trained version with causal rolling 4/8-second views every second. Deduplicate candidate episodes, preserve interrupted tracks and evaluate long sessions separately. Whole-session offline files are useful for replay, not latency-critical inference. |
| Q-L | Whole-action positive labels are bags. MIL/localization with uncertainty works without frame annotation. Negative production sessions supply arbitrary-window supervision. If continuous videos survive, re-extract and annotate a small real-event subset; that becomes preferred context supervision. If not, synthetic composition is optional, carefully screened augmentation, never a substitute for a real replay test. |
| Q-M | Live A/B path, application of `valid_mask`, engine precision and actual checkpoint `seq_len` are unresolved. Assume #41=64/#48=96 only for the legacy reconstruction. At startup read checkpoint metadata and emit actual input/output names, label index, preprocessor version and flags. Compare captured tensors, logits and alarm predicates with the offline training transform. |
| Q-session-FP | Concentration in long tracks/employees/cameras/ID switches is unmeasured. Under equal independent look FPR, one/four/43 chunks with `2n−1` looks imply about 0.679%, 4.660%, 43.981% false-session probability. These are illustrations, not forecasts. Plot actual duration-binned rates and FP contribution; preserve long tracks in the KPI, avoid automatic employee immunity or rising thresholds that conceal recall loss. |

The category semantics, real origin of curated cuts, one store/nine cameras, current AND rule and independent per-person sessions are taken from the owner-confirmed passages. The brief's closing caveat calling those facts “inferred” is stale relative to its v5 header; this response follows the specific confirmed passages. This does not verify the actual deployed binary/configuration.

## 7. Answers to §12.2 — research questions 1–16

### R1. Information ceiling or labels; fraction of consensus errors

Both are plausible; **their fractions are unknown**. Audit the reported 75 consensus FP and 72 consensus FN with synchronized video and two reviewers, plus a random control sample. Code mutually usable labels: wrong action label; uncertain intent/ownership; action outside scored span; missing object evidence; pose/tracker fault; and genuinely hard visible gesture. Report overlap and agreement, not forced exclusive categories. Identical observed pose can map to different legitimate outcomes; this establishes non-identifiability for some cases, not a numerical store-wide Bayes error. Re-evaluate both unchanged and adjudicated labels; avoid “improvement” obtained by deleting difficult test examples.

### R2. Source/camera shortcuts and source-matched performance

Cannot quantify without predictions/metadata. Use grouped source-matched tests with both positives and negatives from staged recordings and continuous/curated real events. A source/camera probe establishes that information exists in the features, **not that the theft head relies on it**. Add within-camera/source matched strata, counterfactual normalization/geometry perturbations and representation/head ablations. Domain adversarial training or Group-DRO is conditional on a harmful dependency and adequate positive coverage, not a default response to high probe accuracy.

### R3. Train-to-production gap from extraction/fps/chunking

Use a paired sequence: same old clip under training extraction → production extraction with matched identities → live preprocessor → exact real windows → final session aggregation. This isolates extractor, tensor, geometry and exposure effects. Re-extraction may explain unstable alarms, but its role is unmeasured. First correct a proven mask/path issue; never attribute all change to engine bytes when resolution/tracking/window content also changed.

### R4. Window size and overlap

Recommend **4-second evidence plus 8-second context, stride 1 second**, with positive MIL bags and real negative windows. Compare a 3-second evidence ablation and 6-second single-scale baseline. There is no established optimal length from existing experiments: the July three-second test changed inference under whole-clip-trained weights and lost 20–25 recall points. Equalize final session FP, event recall, latency and cost; do not compare raw window counts. See V1–V9 and §9 for all requested lengths.

### R5. Most useful extra signal per effort

First masks/quality and global locomotion/scale channels: cheap, directly address arbitrary movement artifacts. Second verified production background and zone/object continuity: greater semantic value. Third candidate-only wrist/bag RGB/object branch: more cost but genuinely new information. Head/gaze and full legs are secondary; two-camera confirmation is opportunistic where coverage exists and should not be mandatory. There is no measured store-specific numeric ranking; prioritize paired replay benefit per engineering day.

### R6. SAM3 distillation

Technically feasible as a **noisy-teacher object-state task**, not by treating every SAM verdict as theft truth. Use reviewed teacher outputs with masks, uncertainty, camera/occlusion strata and temporal object association. Train student states such as visible-in-hand/cart/body-region/unknown, not inferred ownership. Compare against the teacher and human labels. Apply the student to all windows only after demonstrating rescue of pose-missed events and acceptable incremental false candidates; otherwise keep it candidate-only. Distillation cannot recover an object invisible in the source pixels.

### R7. Learned C3 and risk guarantees

Use grouped, out-of-fold stage-1 outputs to train a regularized fusion classifier; freeze it, then select/calibrate its **whole-session policy** on separate negatives and verify on untouched days. For a fixed threshold/policy, use one-sided exact binomial bounds on `any false staff alarm in a negative session`. If threshold search occurs on the certification sample, use a prespecified fixed-sequence/multiple-testing procedure, or a separate certification split [S4]. Guarantees require appropriate exchangeability/independence or a justified dependence-aware method; no guarantee says every future day contains at most one FP. Report day-clustered uncertainty and drift outside those assumptions.

### R8. Hand on back while walking, cat 14/1

Use global trajectory, stop/reach ordering, pose quality and broad production hard negatives; no dedicated phone/back class is necessary. “Walking” is a feature, not a veto: theft can occur while walking. Category 14 requires object provenance/legitimate bagging context where available. Never use a hard gate declaring all carrying or bagging harmless. Review diverse candidate negatives and train the general decision boundary; keep residual near-miss severity as a metric.

### R9. Self-supervised normal-stream pretraining

May improve robustness, motion structure and missingness handling, but no evidence establishes a gain at the extreme false-positive tail. First test the production-window supervised baseline. If unlabeled volume is large, pretrain on masked joints/time with invariance to measured extraction nuisance, then compare supervised fine-tuning using identical labeled groups and compute. Reconstruction/anomaly scores alone may flag unfamiliar benign behavior and miss familiar-looking theft; they are not the recommended alarm criterion. Transfer encoders need compatible joint semantics and normalization [S6].

### R10. Multiclass auxiliary training

Yes, a useful controlled ablation: shared encoder, binary bag head, auxiliary category head weighted 0.1–0.3, with unknown/background and ignored ambiguous labels. The reported high multiclass recall does not prove a better binary low-FPR ranking. Test paired recall at fixed false-session budget and check that mislabeled/confusable categories do not drag the binary head toward a source shortcut. Do not retain the multiclass argmax as an independent alarm.

### R11. Minimum real theft count

No universal training minimum. Count **independent events**, not clips, cameras or overlapping windows. Build nested learning curves at roughly 25/50/100/200 available real events with three seeds, matched held-out real events, and uncertainty. Reuse curated-real events where source grouping is recoverable; never count all 945 clips as independent. For evaluation, roughly 100 positives leave about ±9 percentage-point sampling uncertainty near 66%; around 350–400 are needed for about ±5 points under simple independence, more with clustering. Training can improve before those counts, but the improvement cannot be certified precisely on eight events.

### R12. Diversity by modality

More likely to improve the frontier than additional architecture variants because object/global features expose variables absent from current pose. This is an inference, not measured superiority. Compare one strong pose encoder, a second same-input seed, and a new-view/ROI branch with matched cost; examine **conditional errors on the pose candidates and misses**, not just correlation of raw scores. Tune fusion on groups. AND with a blind object branch can reject true events that are visually occluded; missing modality must be explicit.

### R13. Chunking, aggregation and long sessions

Use storage chunks plus a continuous ring buffer; episode-level causal fusion; one-second stride; masks for initial/final incomplete views. Score a 4-second head before eight seconds are available. Train these cases. Deduplicate the same gesture, reset only after sufficient low evidence, and allow a new separated event in a long session. Compare max-score, short persistence, top-k/time summaries and learned fusion at equal false-session budget. Reject noisy-OR independence and a requirement for two full positive eight-second chunks; many thefts are shorter than sixteen seconds.

### R14. Pose-engine consistency

Freeze source weights and build manifest, preserve input geometry/confidence/NMS/tracker rules, and compare both keypoint and downstream alarm changes on the same videos [S1,S8]. First match tracks by verified identity to isolate pose; then full tracking replay captures switch effects. Differences in INT8 calibration, image shape, letterboxing or threshold crossing may dominate arithmetic rounding. Never set an arbitrary “subpixel difference is safe” rule: the acceptance statistic is paired false sessions and event recall at the operating point.

### R15. Sample nuisances and simulation

The verified distortions are visibility-tail loss, repetition, scale/position changes and short-record padding under legacy equations (§5). Train with naturally occurring production examples first. Then add realistic confidence/dropout and short gap bursts, screen-edge/ROI occlusions and bounded scale drift calibrated from clean trajectories. Keep hidden/held flags and timestamps; do not remove all identical poses because stillness is legitimate. Preserve global scale change while normalizing local articulation. Artificially removing wrists from every positive could make the model ignore important concealment evidence; measure recall by occlusion stratum.

### R16. Latency

For nonoverlapping eight-second chunks, wait after the decisive visible cue is 0–8 seconds, about four seconds if its phase is uniform, **plus processing**; split events may need a later window. Current N/N+1 construction may add more, depending on anchoring. With one-second rolling updates, scheduling delay is 0–1 second; optional two consecutive positive updates adds about one second. Recognition must still wait for sufficient reach/object/concealment evidence, often seconds after onset. A **planning target** is p95 final alarm within three seconds of a visible decisive concealment cue and within one rolling update plus measured verifier processing where evidence is already sufficient. The onset-based delay for a 3–11-second action can be longer. Exact percentiles and ≥66% recall at that delay are unmeasured; report deadline recall curves instead of a universal latency promise.

## 8. Answers to §12.3 — V1 through V64

**How to read effect estimates:** none of the following experiments was trained or replayed here. “Potentially large” means a mechanism could account for a substantial affected-alarm slice; its actual gain is bounded by that slice's measured frequency. “Modest/uncertain” means it should not receive a budget premised on a large gain. Unless a numerical calculation or reported experiment is explicitly cited, **a local FP/recall effect size is unknown and zero gain or regression remains possible**. Every comparison must recalibrate on the same independent calibration groups and report final session/event metrics. An arbitrary percentage forecast for every variable would be misleading.

### A. Temporal input

**V1 — Window length.** Use 4-second evidence plus 8-second context initially, identical definitions in training and inference. Two seconds may capture hand motion but omit pick provenance; three is a useful evidence ablation; four/five often include a short interaction; six is the single-scale control; eight includes reach/conceal context but more background; ten adds context/latency and may obscure a brief action. None is established optimal. Matched training of 3, 4, 6, 8 seconds precedes the full 2/3/4/5/6/8/10 sweep. Expected: better temporal alignment can improve both recall and FP; short windows alone have reported 20–25-point recall losses with legacy weights. **High priority**; interacts with V2/V3/V7/V9/V62.

**V2 — Single versus multi-scale.** Prefer one compact shared encoder with 4- and 8-second masked pooling/evidence heads rather than several expensive separately selected ensembles. Fuse scale evidence with availability and temporal order, not “any scale positive.” Compare against 6- and 8-second single heads at equal session FPR and compute. Expected: better coverage of short acts and context, with uncertain magnitude and risk of more false looks. **High priority after contract**; V1/V5/V24/V53.

**V3 — Stride/overlap.** Start 1 second; compare 0.5 and 2 seconds, then 0.25 only if missed event timing justifies quadrupling scores. An 8-second window over a fully observed D-second session produces `floor((D−8)/stride)+1` full windows, plus a declared entry/exit policy. For D=32: 25/49/97 full windows at strides 1/0.5/0.25. They are strongly correlated; session FPR must be measured. Smaller stride lowers scheduling delay and can raise event coverage, while raw alarm exposure increases. **High priority**; V5/V41/V53/V61.

**V4 — Context outside a window.** Keep a small causal history of quality, scores, reach/object states and motion for roughly 10–30 seconds. Initially use summary features, not a new large recurrent model. Future post-event frames must not enter online predictions before arrival. Expected: could reject benign isolated postures and link split reach/conceal actions, at a possible latency/recall cost. Test 8 versus 16/30-second history and deadline recall. **Medium priority**; V5/V48/V49/R16.

**V5 — Aggregation.** Max-score is a useful sensitive baseline; whole-session mean dilutes short thefts; noisy-OR assumes unjustified probability structure. Prefer causal episode features plus calibrated fusion. Compare two adjacent 1-second positive updates, top-k within 3 seconds and no-persistence control; do not require two full 8-second positive chunks. Persistence may suppress spikes but miss brief/occluded theft; July results demonstrate a real trade-off. Calibrate each policy at ≤1/1,000 final false sessions. **High priority**; V3/V9/V45/V61.

**V6 — Fps/resampling.** Native 15 tracking and timestamp-based 15-Hz classifier input are recommended. Twelve is a valid joint-retrained fallback; 6/8/10 Hz may undersample brief hand trajectories and require anti-aliasing; 20 Hz from 15-fps video creates no new observation. Use nearest with bounded timestamp error or linear interpolation only over short same-track observed gaps; log synthesis. Compute velocity afterwards per second. The calculable 12↔15 legacy delta ratio is 0.8/1.25; alarm changes are unknown. **Immediate priority**; V7/V8/V16/V51.

**V7 — Sequence length.** In v2 set length from seconds × fixed Hz: 4×15=60, 8×15=120; if 12 Hz, 48/96. A fixed L=32/64/96/128 is neither fps nor duration when whole-clip resize is used. TCN physical receptive field is grid interval times `1+Σ(kernel−1)×dilation` for a simple sequential stack; actual blocks must be inspected. GRU can cover the observed sequence but may depend on padded last state. Use valid-time pooling/masks, no repeated feature-row motion. Expected potentially material removal of duration shortcuts. **High**; V1/V6/V24/V64.

**V8 — Gaps/short inputs.** Preserve timestamps and original visibility; interpolate only ≤0.2-second same-track gaps with valid endpoints and keep imputation flags. No extrapolated wrist beyond exit; long gaps split evidence/reset state. Short entry/exit views use explicit masks and a trained shorter head; coverage threshold is selected on recall-versus-quality curves, initially examine ≥60/80% usable bins rather than mandating either. Invalid input is unknown, not automatically benign. Expected fewer artifact alarms with possible recall loss on occluded events. **Immediate**; V21/V35/V64.

**V9 — Label timing.** Use whole-action labels as MIL bags [S5]; do not mark each sampled prefix positive. Obtain approximate take/conceal intervals on a small diverse real-event subset, initially 50–100 available unique events, with visibility/uncertainty flags. Frame-perfect labels are unnecessary for the first improvement. Test bag loss versus inherited crop labels, and evaluate timing-offset perturbations. Expected major supervision alignment, magnitude unmeasured. **High**; V26/V38/V62; weak bag attention is a localization hypothesis, not automatically accurate timestamps.

### B. Body representation

**V10 — Keypoint count/choice.** Retain eight initially; hips are useful anchors for hand-to-waist geometry and scale. Six arm-only joints is a removal ablation, not the default. Save the 17 outputs already computed by YOLO and compare 8, 8+head, 8+legs, 17 with per-joint masks. Do not require legs/head to be visible whenever arms are usable. More joints may improve locomotion/orientation or add occlusion noise; no numeric gain is supported. Whole-body/fingers follows hand-pixel auditing. **Medium**; V13/V15/V19/V44.

**V11 — Hands.** Twenty-one finger points describe grip, not product identity or ownership, and distant pixels may not support them. Prefer candidate-only temporal RGB hand/object/bag ROIs where resolution is adequate; compare with finger pose on human-reviewed object-state cases. Keep observed empty hand separate from “hand not detected.” Expected greatest semantic upside on pose-aliased examples, but unknown store-wide share and latency cost. **High in Phase 2**; V47/V48/V53.

**V12 — Head/gaze/orientation.** Coarse head/body facing can route visibility and shelf-facing evidence; COCO eye/nose points do not establish actual gaze or suspicious intent. Use masked features; never treat back-facing as guilt or innocence. Test 8+head increment on back-view and held-out-camera strata. Likely modest/uncertain discrimination improvement, possibly useful quality routing; camera/source shortcuts are a risk. **Later than global/object cues**; V10/V23/V37.

**V13 — Pose estimator/resolution/confidence.** Pin yolo26s extraction first, including actual image geometry and confidence gate. Pair 50–100 diverse videos across both engines; only switch to m/l or RTMPose/ViTPose after identifying extraction-limited failures and profiling. Save confidences; on new models compare 0.3/0.5/0.7 hard gates versus confidence-aware input. Larger pose AP does not imply fewer theft alarms. Resolution/letterboxing can matter more than tactics [S8]. Potentially large if a gross extraction mismatch exists; otherwise uncertain. **Immediate parity, conditional estimator change**; V52/V63.

**V14 — 3D/lifting.** Defer monocular lifting. A learned transform of the same 2D observations adds a prior, not measured object/ownership/depth information; it can regularize viewpoint but hallucinate under occlusion. First preserve image scale and calibrated floor geometry where valid. Compare 2D, 2D+global scale and pretrained 3D only on held-out viewpoints with confidence/latency. Expected uncertain, probably less value than real object evidence. **Low**; V10/V15/V23/V32.

**V15 — Normalization.** Use a stable torso anchor/scale estimated from observed joints, lightly causal-smoothed, plus a separate global center/velocity/scale-change stream. If hips are absent, shoulder-based anchoring requires a declared fallback/uncertainty mask. Aspect-correct x/W,y/H before Euclidean geometry. Do not force orientation invariance that removes shelf-facing information. Expected potentially substantial reduction of walking/back and scale-change artifact alarms; fraction unknown. Test masked legacy mean/global scale versus torso/local+global features in a 2×2 data/input ablation. **High**; V8/V16/V23/V60.

**V16 — Features.** Start local xy, per-second vxy, visibility, real confidence when available, global center trajectory and scale/quality. Add bone vectors, wrist-to-hip/shoulder distances and coarse motion energy as one ablation. Acceleration/jerk amplify pose noise and are low priority; many angles/distances duplicate the coordinates and should not inflate capacity blindly. Mask derivatives across gaps and switches. Expected improved learnability/stability, not a new ownership observable; magnitude unknown. **High for masks/global, medium derived geometry**; V6/V15/V17.

**V17 — Smoothing.** Compare none versus causal One-Euro/Kalman-style smoothing, tuned on actual noise, with ≤0.1–0.2-second added lag as a planning constraint. Noncausal Savitzky–Golay using future frames is unsuitable unless its lookahead is counted. Filter within a verified identity, never across a switch or disappearance. Expected lower jitter-driven FP but possible recall loss on fast concealment. Test paired scores, spike alarms and time-to-alarm. **Medium**; V8/V16/V21.

**V18 — Mirroring/handedness.** Mirror local pose and swap joint identities consistently; transform velocity and scene/RGB channels together where applicable. Use modest probability, initially 0.5 for local-only augmentation. A fixed scene's shelf/cart geometry is not mirror invariant unless transformed coherently. Accidental left/right swaps are nuisances to detect, not supervision. Expected better handedness coverage, usually modest; wrong scene mirroring can harm. Compare on real left/right events with camera-held-out slices. **Medium**; V15/V29/V49.

**V19 — Other people.** Retain one target skeleton but include crowd density, nearby-person distance, staff/cashier interaction and visibility quality as soft side features if available. A companion handing an item can alter interpretation; a nearby person does not make an action benign. Person-count/proximity is cheaper than a full multi-person graph. Test crowded versus clean strata and target identity labels. Potentially meaningful where swaps/occlusions dominate; magnitude unknown. **Medium**; V20/V44/V48.

### C. Tracking/session construction

**V20 — Tracker.** First instrument current BoT-SORT: identity discontinuities, fragmentation, ReID uncertainty and alarms. `track_buffer=300` is potentially ~20 seconds at 15 update Hz, not a guarantee of continuous observed pose; verify actual update cadence. Compare ByteTrack/OC-SORT or another tracker on labeled multi-person videos under identical detector outputs; prioritize downstream alarms plus IDF1/HOTA-like continuity, not detector score alone. High `new_track_thresh` can miss entrants, long buffers can preserve stale state. Expected major benefit only if switch-linked alarm share is substantial. **High diagnostic, conditional replacement**; V21/V22/V63.

**V21 — Jump repair.** Combine torso-normalized center displacement, bone-length/scale discontinuity, bbox/ReID evidence and visibility; screen initial jump thresholds around 1–3 torso lengths per frame but calibrate by camera and real motion. Split contaminated sequences and reset temporal state. Do not mark a whole theft-containing chunk negative merely because it has a jump. Synthetic splices are corruption/unknown augmentation, or verified-negative augmentation only for benign parents. Expected fewer artifacts with missed evidence during resets. **High**; V8/V20/V29/V61.

**V22 — Session definition.** Keep person entry/exit per camera, define gap continuation using validated tracking evidence, and separately identify fragmented physical encounters where possible. Count short sessions in workload and recall; minimum-duration exclusions must remain visible as coverage loss. A person seen in two cameras produces two camera sessions but one physical theft event when matched. Test rates under fragmentation and duration bins with a frozen event identity manifest. Metric changes can otherwise simulate a gain. **High**; V20/V41/V43.

**V23 — Camera geometry.** Use documented image size, crop/letterbox mapping, shelf/cart/exit zones and ground-plane geometry only where valid. Stratify nine cameras and future dates; camera holdout is within-store generalization, not proof of second-store transfer. Prefer pooled/shrunk calibration; sparse cameras do not support independent 0.1% tail thresholds. Context can improve FP while an almost-negative camera identity shortcut hurts recall. Compare geometry channels against raw camera-ID ablation. **Medium/high**; V15/V34/V37.

### D. Model/training

**V24 — Architecture.** Compact mask-aware TCN-GRU/TCN-LSTM with valid-time/MIL pooling first; one ST-GCN/GAT-TCN comparator afterwards. Causal TCN has explicit temporal receptive field; GRU can carry state but requires track-isolated resets. Transformers/Conformers and pretrained motion encoders are later tests. PoseC3D uses a **time × spatial heatmap volume**, not measured 3D body depth [S7]. Architecture-only upside is probably modest/uncertain; no ceiling proved. Compare equal input/data/budget with paired seeds, not one best row. **Low before input/data fixes**; V7/V25/V32.

**V25 — Capacity/regularization.** Start 128 hidden width and 2–3 temporal blocks, dropout 0.1–0.3; compare 256 only if justified. AdamW decay 1e−4–1e−3 is an initial range, not a new optimum. Test EMA/SWA individually and label smoothing 0/0.02/0.05; rare-positive margins may worsen with smoothing. Checkpoint by grouped constrained recall. Expected less overfitting/seed dispersion, magnitude unknown. Report all seeds and calibration tail, not only ECE. **Medium**; V27/V30/V38.

**V26 — Task.** Binary candidate-evidence/MIL head plus generic background and optional auxiliary interaction/category head. Ambiguous ownership or bad temporal labels receive unknown/ignore weights with reasons; final abstention remains in recall denominators. Hierarchical interaction→concealment→context can help but multiplies missed-event risks. Do not infer every positive crop's label or use ordinal suspicion as ground truth. Expected potentially material correction of weak-label FP; test bag versus inherited labels and auxiliary weight 0/0.1/0.3. **High**; V9/V38/V45.

**V27 — Loss.** CE is the controlled baseline; compare one tail-aware pAUC/hard-negative ranking loss [S9] after label audit, with a negative memory bank and moderate training tail α≈0.005–0.02. Current asymmetric/focal hardening has no consistent reported advantage; more negative penalty can reduce recall and calibration. Constrained session policy selection is mandatory even if loss is window-level. Extreme α≈0.0002 with 64-example batches is statistically unstable. No loss recovers absent semantics. Expected uncertain frontier improvement, high selection stability value. **Medium after data**; V28/V30/V39.

**V28 — Sampling/balance.** Source/camera/day/session-balanced mixture, broad random negatives plus capped verified hard negatives; initially 4:1 negatives:positives versus 6:1 ablation, not an assumed optimum. Cap overlapping windows from one session and preserve distinct real positive events. `WeightedRandomSampler` changes score priors; calibrate on actual production units. Expected more robust hard-background learning, possible recall regression if rare hard positives are overwhelmed. Test identical recipe with/without hard mining and source caps. **High**; V27/V34/V37/V39.

**V29 — Augmentation.** Prioritize real offset/crop variation, empirical dropout/jitter/gaps, limited local mirror and mild speed 0.9–1.1 (widen to 0.8–1.25 only if realistic), small rotations about ±5° and local scale ±10% as starting ranges. Calibrate nuisance ranges to extraction. Avoid scene-inconsistent shifts, impossible skeleton cutmix and massive duplicated-row augmentation. Switch splices label corruption, not automatically benign action. Expected robust nuisance handling, unknown gain; reported 100k synthetic rows did not help. **Medium/high**; V8/V18/V21/V62.

**V30 — Selection.** Replace `val_acc` with event recall at bounded false-candidate **sessions**, using low-FPR pAUC secondarily. Separate fitting/tuning, calibration and locked prospective test groups. Checkpoint selection and threshold search cannot share the certification sample without multiple-testing correction [S10]. This may lower reported numbers while improving trust; selection alone does not improve representation. Report paired day/session uncertainty and every finalist. **Immediate**; V31/V34/V40.

**V31 — Search/seeds.** Focus on contract/pooling/sampling first; limited lr/batch search around successful #41/#48 settings. Three paired seeds for surviving recipes, five for small differences; two or three grouped day splits/holdouts assess composition. Full/filter variance is only a seed proxy if tensor/split equality and exchangeable noise hold. No universal seed number certifies one-FP wins. Expected more reliable rankings and less deployment surprise, not guaranteed FP reduction. **High discipline**; V25/V30/V59.

**V32 — Pretraining.** Mask-aware local production SSL after the supervised background baseline; do not include held-out persons/days in transductive pretraining without declaring it. Compare scratch/frozen/fine-tuned under equal labeled groups. AimCLR-style contrastive invariances or MotionBERT transfer need joint-layout/domain adaptation; existing eight-point arrays are not a plug-in match [S6]. Pretraining may improve robustness, but normal-motion anomaly scoring is not theft evidence. Gain unknown. **Later/conditional**; V10/V24/V39.

**V33 — Ensemble.** Preserve #41/#48 AND@0.80 as the immutable legacy comparator. For v2, compare two seeds/data bootstraps and a genuinely new input branch; use member-specific calibrated thresholds or regularized fusion. Examine unique TP/FP overlap and an oracle complementarity bound before adding compute. Min-score is not a calibrated theft probability; sharing inputs limits diversity. Expected diminishing architecture-only gain; multimodal frontier gain plausible but unmeasured. **Medium**; V34/V35/V48.

**V34 — Calibration.** Calibrate the **complete session/episode policy**, then freeze. Temperature/Platt can align scores; isotonic is unstable with sparse tails. A monotone single-model transform has zero ranking improvement [S11]. Per-camera thresholds require thousands of representative negatives or partial pooling, not 46–134 examples. The threshold should change after retraining if necessary to preserve risk. Conformal coverage is conditional on its formal unit/assumptions, not a daily cap [S4]. **High before promotion**; V30/V40/V54.

**V35 — Uncertainty/abstention.** Use missing evidence, quality/switch flags and ensemble disagreement as features or a declared review route. MC dropout adds forward passes; consensus FP may be confidently wrong. Energy/OOD scores detect some unfamiliar inputs, not intent. Report coverage, hard-alarm recall, review-resolved recall and all staff jobs. FP can fall by withholding uncertain alerts, but recall usually can too; no free gain assumed. Compare risk–coverage curves at fixed workload. **Medium**; V8/V47/V50.

### E. Data/labels/evaluation

**V36 — Volume/learning curves.** Prioritize independent production-format real events and diverse confusable/background sessions. Keep representative easy negatives; cat 0 is not “saturated” when four reported FP remain. Start additions of 25–50 real events, 100–200 verified difficult negative sessions and 1k/5k/10k/30k background sessions where available. Freeze splits/model/weights and count groups, not rows. Stop after two rounds whose marginal gain is within paired noise, not at a universal category quota. Expected potentially large data gain, unknown magnitude. **High**; V37/V39/V59.

**V37 — Real versus staged.** Retain staged coverage but cap its weight relative to independent real events; group staged performer/recording day when recoverable. Evaluate curated-real separately from continuous-format real; their equal reported FN to staged does not prove distribution equality. Match duration/visibility/camera before complex adaptation. Source-balanced sampling/production extraction first; adversarial domain removal only after harmful shortcuts are demonstrated. Expected improved transfer if actual gap exists, uncertain size. **High protocol, later adaptation**; V23/V28/V38.

**V38 — Label semantics/noise.** Store observable take/conceal/object handling separately from actual adjudicated theft if known; unknown ownership remains unknown. Two blinded video reviewers and adjudication, with random controls plus consensus errors, estimate noise and agreement. Cleanlab/loss histories are prioritization aids and need group-out-of-fold scores; instance-dependent ambiguity violates simple noise assumptions [S12]. Small-loss/co-teaching can discard rare real positives. Avoid automatic relabeling. Expected potentially material improvement if labels are wrong, but fraction unmeasured. **High**; V9/V26/V39.

**V39 — Background/unlabeled streams.** Broad production negative bags are the generalist lever; use exact rolling geometry including idle/transition/exit/crowd windows. Start reviewed 1k-session pilot then 5k–20k development sessions. High-score U is quarantined/soft, because alarm selection enriches theft. Estimate window/event prior from random review before nnPU/PNU; selected-clip prevalence is not population or window prevalence [S2]. Expected potentially large FP transfer benefit; verify positive retention. **Highest retraining priority**; V28/V38/V62.

**V40 — Splits/sample size.** Group by complete session/physical event, staged recording/performer/day where possible; use later-day holds, source matching and within-store leave-camera-out secondary tests. No overlapping chunks/events across splits. Reserve enough negative sessions to certify risk: zero/2,995 yields one-sided 95% UCB≤0.001 under independence; 3/3,000 does not. Prefer ≥10k prospective negative sessions over many days. Selection and calibration need separate groups, and day dependence needs sensitivity analysis. **Immediate**; V30/V41/V43.

**V41 — Units/metrics.** Report window/chunk FPR diagnostically; candidate and final false **sessions** per 1,000 negatives; unique staff alarms/day; false alarms/person-hour/camera-hour; real-event recall and deadline curves; missed due to quality/outage/abstention; near-miss versus unrelated FP. An early high score outside the event is not a detected theft. Event matching must include permitted time interval and actual alarm delivery time. Day/session bootstrap and binomial intervals have distinct assumptions. Expected honest operation and no denominator-induced “gains.” **Immediate**; V22/V42/V50.

**V42 — Base rate.** Use `precision = πr/[πr+(1−π)f]` only with same-unit prevalence/recall/FPR. Clip-level reported rates yield about 32% precision at π=0.5%, 8.6% at 0.1%; these are scenarios, not production session precision. At final r=.66/f=.001: about 76.8%/39.8%, respectively. The selected 69/12,920 clips do not measure actual session prevalence. Budget FP first and quantify the cost of missed verified events. **High**; V41/V45.

**V43 — Replay.** Discovery: ~1,000 representative sessions across nine cameras/multiple days, plus a separate positive event pack. Development mining/fitting cannot overlap calibration/test. Certification: preferably ≥10,000 negatives and as many independent real events as available, prospective days. Annotate all candidates and a score-stratified/random noncandidate sample; a sample estimates missed events but is not exact all-event recall without complete event finding or sampling correction. Minimal viewer needs video/pose/timeline/verdict/reviewer persistence. If exports absent, replay to write a ledger; if only poses exist, visual-stage claims remain open. **Immediate**; V38/V40/V54.

**V44 — Crowds.** Keep a target person tensor, but train target-labeled tracks from crowded scenes; simply lifting `single_user_only` can assign theft to innocent companions. Include actual occlusion/switch context and reset corrupted identity state. Test crowded/clean FP/recall and abstention separately; random dropout is a complement, not replacement. Expected gain potentially large in a switch-heavy slice; unmeasured population share. **High**; V19/V20/V21.

### F. Stage 2 and fusion

**V45 — Stage-1 point.** Low-FP current point remains the baseline until verifier efficacy is measured. Afterwards optimize complete pipeline: ≥74% event recall/≤10 false sessions per1k, with ≥90% true retention and ≥90% FP rejection, is a coherent target. A 20-FP/1k stage needs ≥95% rejection. Expanding candidates before a reliable verifier risks more staff interruptions. Measure joint conditional retention/rejection, not independent percentages. **High**; V47/V48/V50.

**V46 — HAR.** Retain interpretable wrist-body/retraction/reach-order summaries with confidence and visibility, using per-second geometry and causal context. Learn or calibrate soft thresholds on real candidates; do not encode an endless gesture blacklist. Hands absent means evidence unavailable, not no concealment. Pose-based HAR shares upstream errors and may not improve semantic aliasing. Expected modest/conditional FP reduction, possible recall loss as July heuristics showed. Test HAR increment over motion at fixed final risk and occlusion strata. **Medium**; V15/V47/V48.

**V47 — SAM3.** Candidate-only event-span object review first; preserve availability, prompt/version, object association confidence and latency. End-of-span invisible object is not proof of hiding, and skipped/no-hands is not benign. Teacher segmentation/object-state quality needs human evaluation; distill reviewed states only. Full-window use is justified only if it rescues otherwise missed real events and meets compute. Possible large semantic benefit on observable objects; numerical rejection/retention unmeasured. **High Phase 2**; V11/V45/V53 [S3].

**V48 — C3 fusion.** Regularized logistic regression first, boosted trees when data supports them, score-GRU later. Features: member margins/min/disagreement, 4/8-second history, HAR/SAM evidence/availability, quality/switch, root motion, zone and object continuity. Group-out-of-fold stage-1 predictions prevent fitting unrealistically clean in-sample scores. Calibrate whole-session alarms on separate data. Expected improved combination versus unconditional signal priority, unknown gain. Missing SAM is not numerical zero without a missingness flag. **High after measurement**; V34/V45/V47.

**V49 — Alternatives.** Build quality splitting, soft locomotion/zone/dwell and causal object reach→hand→bag/cart history first. Carrying state is context, not immunity. Cooldown deduplicates one gesture but cannot reduce the chance of the first falsely alarmed session. Cross-camera corroboration is optional where overlap exists; mandatory two-view approval loses single-view events. Purchase/ownership context would add information if permitted/available but is not assumed. Test each incremental feature on paired candidate episodes. **Medium/high**; V19/V21/V50.

**V50 — Policy.** Separate hard alerts, unattended logs and staff review jobs; report all attention demand. Calibrated severity/near-miss tiers can improve usefulness but cannot relabel 20 interruptions as success. Suppress duplicate episodes with a declared reset/cooldown, allow genuinely new events. Do not raise thresholds with session age merely to hide employee exposure; any duration-aware policy must preserve long-session event recall. Expected fewer duplicates, no automatic session-FPR gain. **High**; V5/V35/V41.

### G. Engineering

**V51 — Preprocessing parity.** One versioned transform is shared by training/evaluation/Triton/export; checkpoint metadata specifies source, joint order, features, timing, masks and label indices. Prove legacy A/B inputs identical before forward; no silent fallback. Use golden complete/missing/frozen/gap/short examples and actual logits. Expected potentially large gap removal if live mismatch exists, no measured population effect. **Immediate first**; V52/V63/V64.

**V52 — Numeric precision.** Test pose-engine and classifier-engine precision separately. Fixed-tensor FP32→FP16/INT8 logits may shift thresholds; tiny coordinates crossing confidence0.5 can change masks discontinuously. INT8 calibration must represent rare candidates/occlusion, not only easy backgrounds. Compare member deltas, alarm flips and paired session/event metrics, with model-specific tolerances. Different engine build bytes are not automatically a defect [S1]. **Immediate diagnostic, defer compression**; V13/V51/V63.

**V53 — Compute/latency.** Profile actual GPU and nine-stream peak concurrency, not model file size. A shared 4/8-second encoder amortizes work; candidate episode dedup reduces SAM jobs. Saving all17 detector outputs need not increase pose forward cost; whole-body/hand estimation does. Measure p50/p95/p99 stage timings, queue/backpressure, RAM, events dropped and decision delivery timestamps. Maintain ≥20–30% measured capacity headroom as a planning target. No hardware-specific milliseconds can be promised. **High engineering**; V3/V11/V47.

**V54 — Drift/monitoring.** Track candidate/final alarm rates by camera/day/duration/quality, score/visibility/gap/scale distributions, stage2 skip/error rates, duplicates and delayed/dropped jobs. Review a random noncandidate sample to avoid feedback blindness. No automatic retraining/promotion based only on verifier labels; use a gated monthly or drift-triggered recipe with prospective holdout and rollback. Expected stable risk and faster discovery, not immediate ROC gain. **High for rollout**; V34/V39/V50.

**V55 — Privacy/legal.** Assume EU/Spain only as a deployment branch, not a verified store location. Preserve the controller's documented purpose, minimization, access/retention policy, transparency and processor boundaries; review whether model-training reuse and systematic monitoring require an impact assessment. Poses linked to sessions/video are not automatically anonymous. Restrict transient ReID to authorized tracking and avoid unnecessary identity/clothing features. Storage duration must follow applicable requirements, not an invented ML retention exemption. Use AEPD/EDPB guidance [S13,S14]. **Prerequisite for data export, not an unsolicited deployment pause**; V43/V54.

### H. Meta-questions

**V56 — Theoretical ceiling.** Unknown numerically. Two different videos with opposite labels can map to identical eight-joint normalized pose; then any classifier must give the same score. Ownership, payment, objects, hidden fingers, erased position and context are absent. The Bayes error depends on deployment class-conditional distributions; architecture count cannot determine it. Derived angles/3D lifting do not create independent observations, although they can help finite-data learning. Quantify empirical pose-versus-video adjudication gaps on random and hard cases. **High diagnostic**; V38/V47.

**V57 — First three changes.** First V51/V64 parity/missingness; second V39/V9/V62 production background with correct weak-positive windows; third V15/V16 robust local pose plus global motion/quality. V30/V40/V43 replay is the prerequisite measurement layer. Estimate gain as affected false-session fraction × preventable fraction, minus new false sessions; no additive invented shares. Object evidence becomes the next priority if remaining FP is semantic. **Immediate/high**; test each paired before attribution.

**V58 — Interactions/budget.** Hard interactions are timing×length×velocity; masks×center/scale×padding; weak labels×window offset×persistence; source/camera×global channels; tracking×state reset; tail loss×negative pool. After free parity/fps tests, use one #48-like backbone for a 2×2: production backgrounds on/off × global/mask representation on/off, all with a fixed corrected temporal baseline. Then single versus 4/8 scales, then CE versus one tail loss. Screen one seed, replicate finalists three seeds. The 430 minutes is reported **per campaign cell**, not necessarily one model. **High experiment discipline**.

**V59 — Noise floors/seeds.** One FP/1,619 changes FPR 0.06177 percentage points; one TP/323 changes recall0.3096 points. Reported 11-FP Wilson range ≈0.38–1.21%, recall206/323≈58.4–68.8%. The selected winner's nominal CI does not correct thousands of trials. Full/filter difference SD is an indirect proxy at argmax, not measured AND-tail seed variance. Three/five seeds and 2–3 grouped holdouts are practical starting counts; paired discordant-event/session analysis determines significance. **High**; V30/V31/V40.

**V60 — Additional failure modes.** Audit aspect-ratio distortion, coordinate/crop mapping, timestamp clock resets and queue delay, stale/synthetic detections, left/right identity swaps, reflection/occlusion, two-handed transfers, wrong target-person labeling, multi-camera duplicate events, verdict anchoring to the wrong chunk, engine/class-label versions and feedback-biased mining. Payment/ownership and product provenance may change the label independently of pose. Feature leakage from “primary robber” metadata is unacceptable. Test delayed callbacks, partial exports, outages and stage2 skip paths; all may change actual recall/FP. **High on correctness, conditional on prevalence**.

### I. Owner's production questions

**V61 — Production chunking.** Eight-second files stay as storage; rolling 4/8-second evidence at1-second stride plus episode fusion is the next trained detector. Compare independently scored chunks, actual N/N+1, rolling and causal stateful models with identical real streams and latency. Whole-session `.npy` is an offline container; never resize minutes to128 tokens for online classification. Short sessions use trained masked heads; long employees remain exposed and count in person-hour/session rates. Persistence/cooldown alone cannot guarantee ≤1/1k. **High**; V1/V5/V22.

**V62 — Action labels to production windows.** A positive action file is one positive bag; its rolling instances are unlabeled until weak localization or human intervals justify labels. Verified benign production sessions supply negative bags/windows. Use realistic random offsets, boundary/transition views and masked partials. Synthetic embedding requires same-camera/body-consistent continuity and a separate synthetic flag; mixing different identities can create artificial evidence. If original continuous video is available, real re-extraction supersedes composition. Expected potentially major transfer gain; unknown magnitude. **High**; V9/V29/V39.

**V63 — Engine contract.** Weight/engine hashes, TensorRT/CUDA/GPU versions, precision/calibration, shape profiles, image geometry/letterbox, confidence/NMS/IoU, tracker and timestamp rules are recorded. Paired same-video tests separate extraction geometry, numerical pose, identity and classifier export. Retrain on production-equivalent extraction if meaningful drift persists; augment only to measured jitter/missingness. Same machine is not a guarantee and distinct file is not proof of failure. **Immediate**; V13/V52 [S1,S8].

**V64 — Triton/short tails.** Capture the live training/A/B/TRT inputs; assert source/mask/joint/features/label map/resize match. Sample proves missing wrist/zero-fill spikes even after dropping invalid frames. Repeated tail cause remains unknown. Current legacy corrections reproduce training exactly; v2 preserves time, uses masked torso normalization, neutral fill after normalization, causal derivatives and padding masks, with new weights/thresholds. Actual `seq_len`/mask/path flags are unresolved metadata checks, not blocked recommendations. **Immediate**; V8/V15/V51.

## 9. Exact temporal and serving contract

### 9.1 Why fps, window seconds and token count must be specified together

For a legacy clip with T frames resized to L tokens, sampled indices are approximately `i_j = j(T−1)/(L−1)`. Token spacing is therefore `(T−1)/[(L−1)f]` seconds when all original intervals are equally spaced. For an eight-second, 120-frame input at15 Hz:

| Legacy input | Approximate spacing between output tokens | Effective output position sample rate |
|---|---:|---:|
| L64 | 119/(63×15) ≈0.12593 s | ≈7.94 Hz |
| L96 | 119/(95×15) ≈0.08351 s | ≈11.97 Hz |
| Short 60-frame input, L64 or L96 | 1/15s during observed prefix, then repeat padding | Observed prefix15 Hz; remaining rows synthetic |

A three-second movement uses fewer tokens in the long clip than in an unresized short clip, so its **token-duration** is compressed. However, velocity was computed at the **original frame spacing**, then sampled. It does not automatically become a finite difference between the selected positions. Describing the entire tensor as simply “1.9× faster” misses this positions/velocity inconsistency. Dropping invalid frames introduces another irregular interval without telling the model.

Changing fps also has two different meanings: downsampling the same real-time trajectory with timestamps, or relabeling the same records as if their time interval changed. E-fps must separate these tests. If both 12- and15-Hz records are resampled into a fixed physical window and velocities recomputed per second, equivalent positions/velocities should be similar apart from sampling/interpolation. If legacy pre-resize per-frame velocities are retained, their ratio is about0.8/1.25 for smooth constant-speed motion. No known deterministic mapping translates that ratio into a recall percentage or alarm logit.

### 9.2 Versioned contract and migration

| Field | legacy-v1 parity baseline | retrained-v2 recommendation |
|---|---|---|
| Pose source | Exact full/filtered source and mask rule used by checkpoint | Raw full tracks, observed masks/confidence, timestamps |
| Time | Reproduce legacy frame selection/resize faithfully | Monotone timestamps; fixed15 Hz grid, same clock domain |
| Window | Arbitrary action/chunk legacy behavior retained for diagnosis | Rolling4s evidence/8s context; stride1s |
| Input length | Read checkpoint, assumed64/96 only until verified | 60/120 grid bins; masked short views |
| Missing joints | Exact old fill behavior for parity | Masked anchor/scale; neutral local fill **after** normalization; observation/imputation masks |
| Invalid frames | Exact old `valid_mask` drop | Keep elapsed time; mark unavailable; split/reset long gaps |
| Scale | Old clip-global scalar | Stable causal torso/local scale plus preserved global scale derivative |
| Velocity | Old pre-resize per-frame delta | Per-second derivative after grid construction; invalid across gaps/switches |
| Padding | Match actual repository behavior, including velocity rule | No fabricated observed tail; pad with mask and zero derivative |
| Precision/labels | Exact checkpoint/backend and positive label index | Versioned export, label map and numerical equivalence test |
| Threshold | 0.80 AND only for reproduced baseline | Calibrated full-session policy; numeric value learned anew |

1. Capture the live v1 tensor and score for the sample and a diverse golden pack. Correct only demonstrated parity faults; replay existing weights.
2. Rebuild v2 training examples from raw/full arrays with real fps/timestamps. For old clips lacking timestamp records, use reported native15 Hz only when trusted metadata/video confirms it; mark approximate timing explicitly.
3. Train and calibrate v2 independently. Keep a complete manifest recording which contract produced every example and checkpoint.
4. Run both versions on identical prospective shadow streams, including final HAR/SAM/C3 and staff-job accounting. Promote only after locked-gate success. Retain v1 rollback artifacts.
5. If sustained15 Hz cannot be maintained, switch the **v2 recipe as a whole** to12 Hz (4s=48,8s=96). Re-extracting YOLO is unnecessary for simple clocked pose resampling, but low source sampling cannot be repaired by invented frames. Recalibrate and test. A per-stream silent fallback to another Hz is rejected.

### 9.3 Golden-tensor and prefix tests

The supplied `pose_contract_audit.py` reconstructs equations from the brief and can compare already captured tensors. **It cannot certify unseen Triton paths.** The team's integration test must capture the tensor immediately before forward in each actual path, then compare:

- Shape/dtype/joint/feature order, fps grid and real-time bounds, masks and finite values.
- Exact equality for the same FP32 preprocessing code; otherwise report maximum absolute/relative error and tolerance justified by the measured score/alarm impact. Use a preliminary `1e−6` FP32 tensor tolerance only as a diagnostic, not a universal exporter criterion.
- Member logits, softmax positive index, margins, `min` score and alarm boolean. Any contradictory AND alarm fails; backend numeric tolerances do not excuse wrong rules or label maps.
- Full/filtered equality at the same UID when masks exist; mask absence is an explicit error/branch, never silent fallback.
- Complete pose; each joint missing individually; no valid pose; short4-second view; long view; timestamp gaps; edge/ROI occlusion; repeated records; identity splice; both candidate members on opposite sides of threshold.
- **Causal prefix consistency:** append later frames to a stream and confirm earlier delivered outputs do not change. A clip-global scale or offline interpolation with future endpoints can revise old features; preserve actual online decision history and forbid future labels/frames from leaking into its evaluation.

Stateful/causal models require per-person state reset at identity changes, explicit track identity routing and no state leakage between batch slots. Triton supports sequence-aware batching [S15], but an ordinary stateless endpoint does not acquire these guarantees automatically. For the first v2 deployment, the ring buffer with stateless window inference is easier to audit than persistent learned state; test stateful streaming only against this baseline.

## 10. Assessment of every proposal and experiment

### 10.1 P0–P9

All effect entries refer to final session/event units. Costs are approximate engineering effort with data available; absent exports/video add integration/annotation effort. No row supplies an already demonstrated improvement.

| Proposal | Decision / order | Cost | Expected effect, risks and validation |
|---|---|---|---|
| **P0** measurement/parity | **Accept first**, refine fps choice to15 Hz; add score-join, label-map and tensor assertions | ~1–3 days plus export work | Makes results trustworthy; may reveal worse numbers. Fix roots, UID coverage, group splits, paired engines and exact replay. Separate calibration/test; test current pipeline as well as motion. |
| **P1** data/labels | **Accept with safeguards**, first retraining lever | ~3–7 days initial, ongoing review | Potentially largest generalist transfer gain. Broad background, real-event bags, reviewer audit and capped mining; unknown/SAM-rejected alarms are not hard-negative truth. Source probe alone cannot justify adversarial training. |
| **P2** shorter/multi-scale | **Modify** to4/8-second matched training, stride1s,3-second ablation | ~3–5 days per contained comparison | May improve timing/context; more looks and inherited positive crops can worsen FP. Compare to6/8-second controls at equal session risk, include boundaries/short views and latency. |
| **P3** learned C3 | **Accept conditional on labeled candidates**, logistic first | ~3–7 days after ledger | Better use of evidence/unknown states is plausible; few positives can overfit. Use group-out-of-fold scores, independently verified labels and final-policy calibration. |
| **P4** gates | **Modify**: quality splitting/reset and soft motion/zone; reject unconditional walking/back/hands-missing vetoes | ~1–3 days initial | Can remove corrupted alarms, with measurable recall/coverage loss. Ordering/persistence may only move along frontier. Cooldown affects repeats; optional cross-view loses coverage if mandated. |
| **P5** richer inputs | **Accept staged**: masks/global first, save17, RGB/object history next | ~1–3 days features; ~1–2 weeks ROI study | Global features address lost locomotion; RGB offers semantic evidence. Finger/head/3D do not prove ownership. Optical flow is optional if pixels exist; clothing only for authorized tracking, not theft judgment. |
| **P6** procedure/calibration | **Modify**: CE control, one tail objective, honest independent calibration, optional EMA/auxiliary/SSL | ~2–5 days per focused test | Stability likely improves; calibration does not manufacture ROC separation. No fixed0.80 guarantee; no automatic conformal protection against changed days. Group-DRO/noisy-label/PU methods need assumptions and adequate groups. |
| **P7** architectures | **Defer**, retain one graph/heatmap comparator after input/data changes | ~3–7 days limited test | Likely modest/uncertain architecture-only gain; could be larger if representation was limiting. No full LSTM/attention/Transformer catalog search. Prune duplicates only after fixed-protocol evidence. |
| **P8** policy | **Accept with accounting correction** | ~1–2 days policy/instrumentation | Severity/review can improve usability, but staff review jobs count. Sparse per-camera thresholds need pooling. Double confirmation can lower recall; validate whole policy and all abstentions. |
| **P9** collection | **Accept**, count independent events and retain random background coverage | Ongoing, initially ~2–5 days label/export | Real production events/occlusions/confusables outrank duplicate clips. Generic categories0/5 are not proven saturated. Use learning curves and added diversity, not universal quotas. |

Specific proposal refinements: P0.6's forced12 Hz is **not** the default; timestamp consistency matters more than12versus15. P1.5 cannot assume known negative intervals outside labeled events because that annotation does not exist. P1.6's candidate-concealment framing is useful, but benign bagging remains a negative for the **final** theft alarm and cannot be relabeled theft to inflate recall. P5's object state is richer than finger coordinates; P6's aggressive co-teaching/small-loss could delete genuinely hard real thefts. P8's gray-zone workload remains counted. Each accepted method requires a measured increment over the frozen pipeline, not its own incomparable best threshold.

### 10.2 Complete experiment matrix

| ID | Ruling / sequence | Refined experiment and success gate | Approximate cost and principal risk |
|---|---|---|---|
| **E0** | Accept, first | Multi-root coverage equality, raw member/ensemble rejoin, exact confusion counts and CI. Separate reported old/new roots. No silent missing UID. | Hours–1 day; corrected scores can invalidate earlier claims. |
| **E1** | Accept, first diagnostics | Source/camera/duration/padding/quality/day/track-switch strata; denominators and grouping supplied. Matched controls, no causal claims from tiny bins. | Hours; sparse slices can mislead. |
| **E2** | Accept, merge with E-engine | Same videos, fixed person matching then full tracking; production/training engine/.pt under pinned geometry/fps. Measure pose, masks, score/FP flips. | ~1–2 days; confounded track matching if not separated. |
| **E3** | Modify | New timestamp/mask-aware fixed4/8-second contract plus bag labels; legacy baseline retained. Train41/48-like finalists, paired seeds after screen. | ~1–3 days per actual run recipe; “random positive crop=positive” would be wrong. |
| **E4** | Accept with independent audit | Consensus errors + random controls; two video reviewers/adjudication. Out-of-fold cleanlab ranking only. Preserve unknown and old labels. | ~2–5 human days; cannot validate semantics from skeleton alone. |
| **E5** | Modify | Train3/4-second evidence plus8-second context and MIL; compare causal aggregation including no persistence or short persistence at matched session budget. | ~3–5 days; more looks/weak localization can lose recall. |
| **E6** | Accept high priority | Masked normalization + global motion/scale/quality channels in the 2×2 background/input ablation; aspect-correct coordinates. | ~1–3 days; raw camera identity can become a shortcut. |
| **E7** | Accept after candidate labeling | First measure existing HAR/SAM, then learned C3 over grouped out-of-fold candidates. Demonstrate joint retention/rejection at the same final operating point. | ~1 week; small positive count and missing-state bias. |
| **E8** | Accept conditional | Audit wrist pixels; candidate-only RGB/object continuity/student, teacher states reviewed. Test both pose candidates and pose-missed real events. | ~1–2 weeks; noisy distillation and occluded hands. |
| **E9** | Defer | One masked graph/heatmap/pretrained comparator on the winning data contract, not a full catalog. Require repeatable constrained recall gain. | ~1 week; architecture winner's curse. |
| **E10** | Modify | Calibrate complete policy on separate sessions; exact bound or prespecified risk-control threshold search. Report coverage, not merely ECE. | ~2 days plus sample accrual; small tail set cannot certify risk. |
| **E-fps** | Accept diagnostic, modify conclusion | Current-weight matched-time resampling10/12/15/18 Hz, clock mislabeling and gap tests; distinguish velocity units from information sampling. Compare matched15 and12 trained finalists only if throughput requires12. | Hours–1 day diagnostic; cannot infer intrinsic fps superiority from legacy mismatch. |
| **E-replay** | Accept, prerequisite | ~1k representative discovery sessions; exact chunks/actual overlaps and full cascade; count distinct sessions/events/arrival latency. Extend certification to≥10k negatives/later days. | ~2–3 days if exports exist;1k cannot certify1/1k risk. |
| **E-parity** | Accept, first | Real training/A/B/TRT captured tensors/logits, class mapping, masks and AND predicate; include supplied sample and pathological golden pack. | Hours–1 day; prose reconstruction is not actual serving proof. |
| **E-chunk** | Accept, after parity | Same streams, same labels: current chunk/N+N1 versus rolling4/8-second windows with stride 1 second versus causal state, with matched trained models. Full candidate episode/latency/long-session accounting. | ~2–4 days plus necessary retraining; inference-only geometry change is not fair. |
| **E-engine** | Accept, operationally E2 | Pose engine and classifier export tested separately; pinned weights, preprocessing and tracker. Judge paired session flips and recall. | Same work asE2, do not double-budget it. |
| **E-bg** | Accept, first retraining data change | Broad verified/weak production background with random+near-threshold mixture and capped identity corruption. Audit contaminated U and protect real positives. | ~3–5 days initial; global0.5% does not describe alarm-pool contamination. |

A success criterion common to trained experiments is **better event recall at the same false-session/staff-job budget on held-out days**, or fewer false sessions at the same event recall with no unacceptable latency/coverage loss. A convenient dev target is a ≥20% relative false-session reduction or ≥3–5 recall-point gain that survives paired uncertainty; these are minimum practical advancement targets, not predictions of each experiment's gain. Improvements smaller than statistical/seed noise are retained as inconclusive.

### 10.3 Additional proposals beyond the brief

1. **Audit score provenance before behavioral explanations.** Log model hashes, label indices, preprocessor/engine/rule versions and UID/window identity beside every member score. This specifically catches the impossible AND probability table, stale p1 schema and wrong anchor windows.
2. **Causal prefix invariance and artificial-observation flags.** Validate that later frames cannot revise past outputs. Store observed versus held/interpolated pose flags. Global scale computed from a future tail must not contaminate a previously delivered result.
3. **Separate object observability from theft labels.** Annotate what can be seen—object take, hand association, cart/bag entry, occlusion—and factual theft only when independently known. This makes distillation, fusion and coverage measurable without inventing intent.
4. **Alarm-selected contamination control.** Keep a random background reservoir and a separately sampled near-threshold/candidate reservoir, with selection probabilities and label-confidence provenance. Review both; weighting prevents the mined pool from silently redefining “normal.”
5. **Adversarial benign matching rather than named action fixes.** Pair observed take/conceal trajectories with verified legitimate object handling at similar pose/duration/quality. If pose scores remain indistinguishable while object/context resolves them, prioritize the missing modality rather than labeling another gesture class.
6. **Queue and version failure tests.** Test candidate scheduling bursts, out-of-order stage2 callbacks, late verdicts, failed ROI extraction and model hot swaps. An alarm delivered after a person leaves is a missed timely event even if the offline score was high.
7. **Prospective review of silent misses.** Sample noncandidate sessions, with score-stratified inclusion probabilities, to prevent candidate-only review from hiding events stage1 missed. Full event recall requires a reliable independent event census or a sampling-aware estimate.

## 11. Roadmap, statistical gates and stop rules

### 11.1 Ordered milestones

| Milestone | Concrete output | Acceptance criterion | If it fails |
|---|---|---|---|
| G0: integrity/parity | Pinned versions; complete UID ledger; captured golden tensors/logits; reproduced alarm predicate | All roots included; no contradictory score/label-map/rule; live legacy tensor parity or explicit measured correction | Fix correctness first; retain current model as frozen comparator, defer new training claims |
| G1: production diagnosis | ~1k representative replay sessions, labeled event pack, quality/exposure strata, end-to-end stage measurements | Reproduce stage-1 distinct-session count at comparable days/configuration; account for missing/failed sessions and all final verdicts | Reconcile clocks/versions/selection/export; do not explain unexplained gaps with architecture |
| G2: v2 candidate model | Mask/time-aware4/8-second recipe; broad background/MIL; 3-seed finalists | Early ≤10 false-candidate sessions/1k at≥66% event recall; preferred nominal ≥74% to preserve final recall after filtering | Compare the 2×2 interventions and inspect real-event misses; if semantic FP dominates, advance object evidence rather than more pose variants |
| G3: verifier/fusion | HAR/SAM retention/rejection report; causal object features; small calibrated fusion | Joint target≥90% true-event retention and≥90% false-candidate-session rejection when G2 is≤10/1k | Raise stage-1 sensitivity only when measured complete cascade supports it; keep unknown coverage explicit |
| G4: locked prospective validation | Frozen full policy and untouched future-day negatives/real events; confidence and latency reports | Final point target≤1 FP/1k negatives and≥66% event recall; stronger statistical release gate described below | Continue shadow analysis/data collection; no claim the target is achieved merely from dev tuning or fewer hard alerts |

The **stronger statistical release gate** is a one-sided95% FP upper bound≤0.001 and a predeclared lower recall bound≥0.66 on the complete frozen policy, plus acceptable unrelated-FP severity, coverage and latency. The nominal arithmetic74%×90%=66.6% has very little statistical margin: a measured66.6% on a few hundred events cannot establish a lower bound above66%. Aim for roughly72–75% measured final recall during development if a credible≥66% lower bound is required with a few hundred independent test events. This is a testing-margin requirement, not a promise that the model can attain it.

If the organization accepts a point-estimate pilot while positive-event sample size is small, label it explicitly as **provisional** and publish its interval. Do not silently substitute that pilot for the stronger gate. No user decision or reply is needed to carry out the integrity/replay/development work described here.

### 11.2 What sample sizes actually say

For a frozen policy on independent, correctly labeled negative sessions, with x false sessions among n, one-sided95% exact upper FPR is `BetaQuantile(.95; x+1,n−x)` for x<n. At zero errors it is `1−0.05^(1/n)` [S16].

| Test observation | One-sided95% upper false-session rate, approximate | Implication at 1,000 negative sessions/day |
|---|---:|---|
| 0/1,000 | 0.2991% | Up to about2.99 expected FP/day still compatible; does not substantiate1/day |
| 0/2,995 | ≤0.1000% | Mathematical minimum zero-FP demonstration under assumptions |
| 0/3,000 | 0.09981% | Barely supports the rate; poor protection against shifted/clustered days |
| 10/10,000 | 0.1696% | About1.70/day upper bound; the point estimate1/day is insufficient |
| ≤4/10,000 | ≤0.1000% | Useful exact gate with more days than a3k zero-FP test |
| ≤20/30,000 | ≤0.1000% | Larger gate; still requires representativeness and policy independence |

Sessions share cameras, employees, customer groups and days. Window-level binomial CIs are not valid substitutes for independent sessions. Even session binomial intervals may be optimistic under shared-day dependence. Collect many distinct days, report complete-day alarm counts, and use paired day/block resampling or a prespecified sequential method where justified. Ordinary bootstrap with zero observed FP can falsely suggest zero uncertainty; retain exact/model-based rare-event bounds and explicit dependence caveats.

For recall, use an independently found event census or a positive test pack whose selection is declared. Enriched theft events can estimate recall conditional on their selection; they do not measure prevalence. Measure detection during the independently labeled event and a predeclared grace interval separately. Events with no pose, tracker loss, abstention, queue drop or failed verifier remain misses unless a valid delivered alert resolves them.

**Average is not a hard daily cap.** Under a simple Poisson model with mean one false alarm/day, the probability of two or more is `1−2/e≈26.4%`. The actual process may be more clustered. Report average, p95/day and worst day. Hard-suppressing alerts after the first alarm would lose subsequent thefts and cannot be used to hide a weak detector.

The historical 0.01% clip-FPR aspiration and the current1/1,000-session budget (0.1%) differ tenfold and are different units. This response optimizes the latter; retain the former as a secondary aspiration only with a clear denominator.

### 11.3 Cost and expected-gain prioritization

**First:** parity/export/rule checks and replay—hours/days, highest uncertainty removed per effort. **Second:** production background/weak-positive training and mask/global features—days, largest plausible generalist gain. **Third:** measure existing stages and learn small fusion—days/week, cheapest test of the final budget. **Fourth:** object-ROI/history—weeks, most relevant new semantic evidence. **Fifth:** one graph/SSL comparison—only after the preceding measurements show residual representational headroom.

Before planning an FP improvement, measure affected-alarm fraction f and conditional fix success r. With no newly created alarms, maximum explained reduction is f×r. For example, if 30% of false sessions contain a verified preprocessing artifact and 80% of those disappear under a paired repair, the gross reduction is24%; if only5% are affected, it is4%. This example is arithmetic, **not a measurement of Kanvis**. Report newly created false sessions and lost true events alongside the gross reduction. It avoids promising stacked30%+20%+50% improvements for overlapping mechanisms.

## 12. Assumptions, unresolved measurements and conditional actions

These are **information gaps with adopted assumptions**, not requests for a reply. The plan proceeds without resolving them in this session.

| Gap / status | Adopted assumption | Conditional recommendation | Cheapest team verification |
|---|---|---|---|
| Repository/checkpoints/artifacts absent — unresolved | Brief accurately reports campaign counts except flagged contradictions | Use this diagnosis as a research/implementation plan; rerun raw audits before accepting deployed metrics | Materialize cited repo/plans/CSV files in the engineering environment and run UID/logit joins |
| Meaning of20/654 — unresolved exact ledger | Twenty distinct stage-1 falsely alarmed sessions,654 all sessions | If repeated windows instead, report alarm count and false-session count separately; use negative denominator | Count distinct session IDs behind the 20 candidates and independently verify labels |
| Actual fps/time grid — unresolved | Native extraction reported15 Hz; sample rate not encoded | Use15 Hz timestamp contract; if true grid differs, convert jointly and retrain, never silent metadata relabeling | Inspect `pose_frame_timestamps.npy`, clock source, frame skips/duplicate flags |
| Live Triton path/mask/seq_len — unresolved | A/B and64/96 are plausible from brief | Demonstrated mismatch gets legacy parity fix; richer v2 transform requires new checkpoints | Capture actual input tensor and checkpoint metadata, not only temp filenames |
| AND error identities/probabilities — contradictory | Aggregate11FP/117 FN is reported, not independently verified | Preserve as provisional; do not trust member score/FP joins until consistent | Recompute both model logits keyed by same UID/window/hash/rule |
| Full versus filtered equality — unverified | Same retained frame set when masks exist | Collapse duplicate modality cells if equal; investigate missing-mask/data differences otherwise | Pairwise raw and post-transform tensor equality for all common UIDs |
| Engine weight/precision/geometry equality — unresolved | Same reported family, not proven same weights | Pin exact production extractor; re-extract/retrain if downstream differences material | Hash source weights/build configs; same-video paired tests |
| Sample motion/repetition cause — unresolved | Size change/lower-joint loss/repetition measured; physical cause unknown | Train realistic nuisance coverage; do not assert bottom-edge exit or exporter fault | Synchronized raw video, ROI mapping and observed/held-frame flags |
| Positive event independence — unresolved | Clip counts exceed number of unique events | Group by actual source event/performer/day; eight production clips may be correlated | Recover parent videos/session/times and duplicate overlaps |
| Background/event timing labels — confirmed absent inside action clips | Whole-action labels only, broad production streams may exist | MIL bags plus reviewed/weak backgrounds; re-extract true context if videos available | Check source video retention/export and annotate a small real-event subset |
| Viewer/export/review tooling — clarified minimal viewer only | No persistent review or replay tool assumed | Add video/pose/timeline review ledger; export existing verdicts if possible, otherwise recompute | Inspect export API/data folders; use provided schema as the minimal target |
| HAR/SAM rejection/retention — unresolved | No measured labeled validation/test performance | Candidate-only measurement first; relax stage1 only after proven cascade benefit | Run full frozen pipeline on independently labeled candidates and real event pack |
| Object observability/ownership — unresolved | Pixel visibility and factual theft context vary | Use unknown/object-state labels; no hidden-object=theft shortcut | Camera-specific wrist/object pixel and blinded label audit |
| Hardware/queue budget — unresolved | Current small classifiers feasible; added streams require profiling | Shared scales and candidate-only object stage first;12 Hz only if joint throughput study favors it | Measure GPU queue/p95 delivery/drop rates under nine-stream peak load |
| Real prevalence — unresolved |0.5% and0.1% are planning scenarios, not measured session prevalence | Preserve real negative exposure and separately enriched event tests; recompute precision with actual prevalence | Independent random-session/event sampling with inclusion probabilities |
| Multi-camera overlap/transactions — unresolved | Not required or available in the baseline | Optional corroboration/provenance can help; do not mandate unsupported two-camera/payment input | Inspect overlap maps and permitted object/transaction linking capabilities |
| Store jurisdiction/retention policy — unresolved | EU/Spain conditional branch because deployment country not specified | Follow applicable controller policy/authority guidance before video/identity reuse; use minimized pose manifests meanwhile | Confirm documented deployment jurisdiction, purpose/access/retention and training-use basis |
| Long-session FP concentration — unresolved | Longer sessions increase possible looks, not necessarily independent alarms | Duration-aware measurement and episode dedup; no employee immunity or artificial denominator splitting | Replay duration/employee/quality strata and examine20 false sessions |

**Explicitly unanswered empirical quantities:** actual deployed A/B/mask/length; source-matched ROC; exact engine/fps alarm delta; fraction of label versus information errors; duration/camera/switch contribution to20FP; optimal window/stride; true SAM/HAR retention/rejection; actual end-to-end point/latency; event-count learning curve; actual privacy/hardware/export constraints. The reason in every case is absent repository/video/scores/timestamps or ground truth. Each has a conditional recommendation and minimum verification above or in its corresponding V/R answer. These gaps do not justify asking the nontechnical recipient to make a technical choice.

## 13. Primary sources and how they support the recommendations

Accessed 14 October 2026. Sources substantiate mechanisms/methods; none measures Kanvis's operating point. Descriptions are paraphrases; no long passages are quoted. Public runtime documentation must be pinned to the team's installed versions before implementation.

| ID | Primary source | Specific use / limitation |
|---|---|---|
| **S1** | [NVIDIA TensorRT — How TensorRT Works, determinism](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html) | Build tactic/numerical variation and fixed-engine execution; not proof that Kanvis engines differ materially |
| **S2** | [Kiryo et al., Non-Negative PU Risk, NeurIPS2017](https://papers.nips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html) | P/U risk correction; representative positives/class prior assumptions must be checked |
| **S3** | [Meta official SAM3 repository](https://github.com/facebookresearch/sam3) | Prompted object segmentation/tracking. Application object state/ownership is separate; pin actual checkpoint/version rather than assuming latest |
| **S4** | [Angelopoulos et al., Learn then Test](https://arxiv.org/abs/2110.01052) | Calibration/risk control and threshold-search testing; guarantee unit/assumptions must match the final policy |
| **S5** | [Ilse et al., Attention-based Deep MIL, ICML2018](https://proceedings.mlr.press/v80/ilse18a.html) | Bag-level labels/aggregation; attention is not automatically correct concealment timing |
| **S6** | [MotionBERT, author repository, ICCV2023](https://github.com/Walter0807/MotionBERT) | Pretrained motion representation transfer; joint/normalization/domain mapping is required |
| **S7** | [Duan et al., Revisiting Skeleton-based Action Recognition / PoseC3D](https://arxiv.org/abs/2104.13586) | Heatmap-volume action representation; not monocular measured3D skeleton recovery |
| **S8** | [Ultralytics official prediction documentation](https://docs.ultralytics.com/modes/predict/) | Input preprocessing/shape/letterbox settings, to be pinned to deployed version |
| **S9** | [Zhu et al., When AUC Meets DRO, ICML2022](https://proceedings.mlr.press/v162/zhu22g.html) | Tail-aware pAUC optimization; no guarantee of local extreme-tail sample adequacy |
| **S10** | [Cawley & Talbot, Selection Bias, JMLR2010](https://www.jmlr.org/papers/v11/cawley10a.html) | Finite-sample model-selection overfitting, separate selection and evaluation |
| **S11** | [Guo et al., Modern Neural Network Calibration, ICML2017](https://proceedings.mlr.press/v70/guo17a.html) | Temperature calibration; no new semantic information |
| **S12** | [Northcutt et al., Confident Learning](https://arxiv.org/abs/1911.00068) | Label-quality diagnostics under assumptions; here only review prioritization |
| **S13** | [EDPB final Guidelines3/2019, video devices](https://www.edpb.europa.eu/documents/guideline/guidelines-32019-on-processing-of-personal-data-through-video-devices_en) | Purpose, minimization, transparency and video-data responsibilities in applicable EU deployment |
| **S14** | [AEPD official videovigilancia guidance](https://www.aepd.es/guias/guia-videovigilancia.pdf) | Spain-specific video/data guidance, conditional on actual deployment jurisdiction |
| **S15** | [NVIDIA Triton — Batchers / Sequence Batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) | Stateful request routing and sequence-aware serving; needs explicit model integration |
| **S16** | [NIST handbook — proportion confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) | Wilson/exact proportion interval principles; mathematical independent-observation calculations |

## 14. Reproducibility appendix and runnable deliverables

### 14.1 Inspected and unavailable inputs

**Inspected:** all 834 lines of `upload/KANVIS_CONTEXT_AND_RESEARCH_BRIEF.md`; the full supplied binary array; independently computed audit JSON; rendered diagnostic PNG; primary pages in §13; the delivered utility code and example evaluation outputs.

**Unavailable and not inspected:** `/home/debian/dev/Proyectos/GuadIA-IAModel/IAModel/experiments/`; all `/home/debian/Documentos/Kanvis/…` campaign/extraction/production paths; checkpoints; videos; masks/timestamps/confidence arrays adjacent to the owner's production sample; Triton container code/configuration. References to their behavior in this response are attributed to the brief. No command here fixes the unavailable `evaluate_validation.py`; its repair is a proposed engineering change.

Source file hashes before/after this analysis:

```text
KANVIS_CONTEXT_AND_RESEARCH_BRIEF.md
7d8aee4b44531d45db1363a837e96a7822ddc4753d9795803f22ce77ccb7ed7b
SAMPLE_production_chunk_mergedfps.npy
e34b166de168a6ca932a37b1b124ce436bf0ff64c72713ba91b57dbb5dd1b493
```

### 14.2 Utilities

Start with [the utility README](kanvis_tools/README.md). All files are included alongside this response in the ZIP package.

- [pose_contract_audit.py](kanvis_tools/pose_contract_audit.py): loads raw[T,J,2] without modifying it; missingness/repetition/geometry/legacy reconstruction, optional real timestamps or explicitly assumed fps. Does not run a classifier or prove live parity.
- [sample_pose_audit.json](kanvis_tools/sample_pose_audit.json): exact measured sample results and assumptions.
- [sample_pose_diagnostics.png](kanvis_tools/sample_pose_diagnostics.png) and [plot_sample_pose_audit.py](kanvis_tools/plot_sample_pose_audit.py): inspected static scientific diagnostic and reproducer.
- [replay_eval.py](kanvis_tools/replay_eval.py): strict manifest/score integrity, AND-score semantics, session FPR/event recall, confidence bounds, arrival-time latency and declared cooldown. No model/annotation/calibration/production mutation.
- [replay_eval_README.md](kanvis_tools/replay_eval_README.md): schemas, examples, timestamp/event conventions and limitations. Example JSON files use **synthetic data**, not Kanvis outcomes.
- [golden_tensor_compare.py](kanvis_tools/golden_tensor_compare.py): compares actual captured tensors and optional logits/metadata; capture itself must be implemented in the team's serving paths.
- [kanvis_contract_v2.json](kanvis_tools/kanvis_contract_v2.json): reviewable proposed training/serving contract; not a drop-in production configuration.

From the extracted deliverable folder, with Python 3.10+, NumPy and SciPy; plotting additionally needs Matplotlib:

```bash
python kanvis_tools/pose_contract_audit.py /absolute/path/SAMPLE_production_chunk_mergedfps.npy --assume-fps 15 --output measured_audit.json
python kanvis_tools/replay_eval.py --self-test
python kanvis_tools/replay_eval.py --manifest kanvis_tools/replay_manifest_example.json --scores kanvis_tools/replay_scores_example.json --threshold 0.80 --event-grace-s 0 --cooldown-s 0 --output replay_metrics.json
python kanvis_tools/golden_tensor_compare.py training_tensor.npy triton_tensor.npy --atol 0.000001 --rtol 0
```

The score harness's default zero cooldown counts raw positive window verdicts as episodes. A real staff-alarm budget needs the actual production emission policy and a complete-day ledger; a simulated greedy cooldown is not equivalent to unimplemented hysteresis/fusion. It does not merge multi-camera physical events: predeclare anchor events/per-view metrics or add a separate physical-event join. Missing final candidate verdicts make final metrics unavailable. Binomial bounds are descriptive under clustering; day-bootstrap/ascertainment corrections are external tasks.

### 14.3 Main recomputations

Counts in the brief imply: 323 positives − 117 FN = 206 TP; 1,619 negatives − 11 FP = 1,608 TN; recall 63.777%; precision 94.931%; F1 ≈76.3%; FPR 0.679432%; 20/654 =3.058104%; equivalent independent looks ≈4.55568; 434 missing validation cases imply 433 negatives and 1 positive when the planned 324-positive/2,052-negative split is correct. The sample audit independently confirms the metrics in §5. The exposure/precision/cascade/CI calculations are mathematics under stated assumptions, not new production measurements.

**Verification completed:** utilities compiled; sample audit reproduced; evaluator self-tests and edge checks ran; invalid fps/single-frame timestamp/other-joint-layout handling reviewed; diagnostic PNG visually inspected; required R/V/P/E coverage checked; source hashes preserved. No retraining or actual end-to-end replay was possible without the referenced datasets/models. Accordingly, the response supplies a complete diagnosis and executable measurement plan, not a claim of completed deployment improvement.
