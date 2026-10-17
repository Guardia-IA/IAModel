# Expert response — Kanvis Stage-1: window length & architecture strategy

**Date:** 2026-10-17
**Responding as:** senior expert in CV / skeleton-based HAR / low-FPR learning and evaluation design
**Inputs used:** the self-contained assignment brief (§1–§11) and the repo-context brief (same date). No other files, no prior chats.
**Language:** English

**Conventions.** Confidence tags: **[H]** high (follows from first principles or from the numbers in the brief), **[M]** medium (strong prior, should be verified cheaply), **[L]** low (judgment call, depends on data I have not seen). Where I derive a number the brief does not state, I say so. Effect sizes are rough orders of magnitude, not promises.

---

## 1. Executive summary

1. **Keep the production temporal contract at 8 s / 12 Hz / 96 frames / stride 4 s** for now. Do not change it until a session-replay benchmark says otherwise. [M]
2. **Verify the time base of production chunks this week.** The sample `mergedfps.npy` has 60 frames. That is exactly 8 s × 7.5 fps, not necessarily 5 s × 12 Hz. If production chunks carry native-fps frames without resampling to 12 Hz, you have a silent velocity/time-scale mismatch that could dominate everything else. [M, must verify]
3. **Full-npy vs 8 s chunk is no longer the #1 mismatch.** In order: (i) eval unit and multiple-looks (session = max over many windows), (ii) negative distribution (curated action segments vs continuous sessions), (iii) pose and time-base parity, (iv) label semantics. Stride and short fragments are real but second-order. [H]
4. **Back-of-envelope:** a clip-level FPR of 0.68% with only ~4 effectively independent windows per session gives about 2.7% session FPR (1−(1−0.0068)^4). That is already the observed 2–3%, before any distribution shift. The gap may be largely an aggregation-unit artefact. [H]
5. **Build the production-replay benchmark first**, as two separate sets: a *negative replay* (≥3,000 random unflagged sessions) and a *positive replay* (all stored theft sessions). Validate it by checking that legacy #41+#48 reproduces ~20–30 FP/1000 on it. [H]
6. **Multi-length eval: yes, run it (hours of compute).** Read it as a robustness/short-fragment diagnostic, not as "which length is best", because a model trained at 8 s will favour 8 s. Order: 8 (ref), 5, 4, 6, 3; 12 optional; 2 s diagnostic only. [H]
7. **Multi-scale training: not now.** Train at 8 s with overlap-based window labels (with an ignore zone) and a *truncation augmentation* (25% of samples, 3–8 s, padded and masked) to cover start/end-of-session fragments. Revisit once ≥300 positives have `[t0,t1]` intervals. [M]
8. **Short-fragment policy:** re-anchor the final window to end at session end (window = [T−8, T]) instead of emitting a short chunk; for sessions <8 s, right-pad with mask if ≥4 s, abstain below 4 s. [M]
9. **Architectures:** ≤0.3M params per model, six candidates (§9), 3 seeds each, no 60-model grid, no PoseC3D/MS-G3D. [H]
10. **Deployment ensemble:** mean of logits over 3 seeds × {TCN, GRU} (6 members) if replay shows it is not worse than TCN-only; otherwise 3-seed TCN. **Never AND of two large models.** Keep #41+#48 AND as the incumbent until the v2 stack passes the replay gates. [M]
11. **Highest-ROI non-architecture lever:** production-representative negatives. Mine background windows, review the top-scoring windows of *unflagged* production sessions weekly, and use unflagged sessions as free negatives for tail calibration. [H]
12. **Labels:** triage all class-6 clips for "own item / not theft" (~4–7 reviewer-hours), annotate `[t0,t1]` on 300–500 positives, and use out-of-fold scores to direct label review. [M]
13. **Post-processing is cheap and probably worth 1.5–3× FP reduction:** persistence (k-of-n consecutive windows), per-track cooldown/dedup, abstention on poor mask quality or short/ID-switch tracks, per-camera thresholds. [M]
14. **Acceptance in session/event units:** event recall at ≤10 FP/1000 sessions (dev gate), certified on ≥3,000 negative sessions; certifying 1 FP/day needs ≥10k negative sessions or a multi-week shadow run. ≤1 FP/day at useful recall is unlikely from skeleton-only Stage 1 and probably requires Stage 2. [M]
15. **Eval hygiene:** chronological split with embargo; select checkpoints on a smoothed multi-point metric (partial AUC / mean recall at FP ∈ {5,10,20}/1000), not on one noisy 11-FP point; never use F1 for decisions. [H]

---

## 2. Answers to Q1–Q18

### Block A — Window length & temporal contract

**Q1. Is full-npy vs 8 s chunk still the #1 mismatch?**
No. That mismatch (H4) is largely closed by v2 (fixed 8 s @ 12 Hz windows, no time warp). The dominant remaining gaps, in my estimated order of impact on session FP:

1. **Eval unit / multiple looks [H].** A session is scored by the max over many windows, while val is one score per curated clip. Example: session of 60 s at stride 4 s and window 8 s gives 14 windows; even with strong temporal correlation, 3–5 effectively independent looks is plausible. At per-look FPR p = 0.68%, session FPR ≈ 2–3.4%. This alone reproduces the production number.
2. **Negative distribution [H].** Val negatives are human-cut action segments (paying, walking, ...). Production negatives are continuous shopping behaviour: reaching, adjusting clothes, phone use, bagging, transitions, occlusions, ID switches. The model has never seen most of that manifold.
3. **Pose/time-base parity [M].** Includes the possible fps mismatch (Exec. summary #2).
4. **Label semantics [H].** Phone-pocketing and similar are indistinguishable from concealment by skeleton alone.
5. **Stride 1 vs 4, short chunks [M].** Stride 1 inflates both recall (phase alignment) and FP (more looks) relative to stride 4, so it is the wrong yardstick for deployment decisions but not the cause of production FP. Short chunks matter only at session boundaries, but those are also where tracker artefacts concentrate.

Derived fact: from F1 76%, recall 64%, FP 11, the legacy val set has ≈ 160 TP, so ≈ 250 positives and ≈ 1,619 negatives (≈ 13% prevalence). At production prevalence of, say, 0.5% (assumption), the same operating point (TPR 64%, FPR 0.68%) gives precision ≈ 32%; at 0.1% it is ≈ 9%. Clip F1 is therefore not informative about production.

**Q2. Multi-length evaluation before multi-length training?**
**Yes.** It is cheap (inference only on small models) and it directly informs the short-fragment policy. But be explicit about what it measures: a model trained at 8 s will almost always look best at 8 s when evaluated, because eval matches training. So E1 answers "how much do shorter fragments hurt this checkpoint?", not "what is the optimal window?". Choosing the optimal length requires training dedicated models per length (E2a). At 0.14M params this is a few GPU-hours per length, so do not be dogmatic: after E1, train dedicated 5 s and/or 4 s models if E1 shows a drop larger than the noise.

**Q3. Which lengths first, and stopping rule?**
Ordered: **8 s (reference) → 5 s (the actual observed fragment length, or whatever production emits) → 4 s → 6 s → 3 s → 12 s (optional, only as a "does more context help" probe).** 2 s is diagnostic only: the concealment gesture itself is ~1–3 s, 8 joints in 2D at 12 Hz give too little context, and I expect it to be dominated.
Stopping rule: use paired bootstrap on the same grouped sessions. Stop exploring shorter lengths when two consecutive decrements both show Δrecall@FP < +2 pp (CI including 0) versus 8 s. Adopt a different production length only if it is better by ≥ 3 pp recall at the same session-FP budget (CI excluding 0), or equal within ±2 pp *and* operationally preferable (latency, compute). Otherwise stay at 8 s.

**Q4. `seq_len = 12 × seconds` or fixed 96 with subsampling?**
**Proportional (`12 × seconds`), keeping 12 Hz.** Subsampling changes Hz, which changes the velocity statistics and silently mixes a speed change into a length change. With proportional length you keep physical time per frame fixed and vary only information content. Implementation notes:
- Evaluate each length in two ways: native length (e.g. 48 frames) and padded to 96 with mask. If masking is correct (masked norm layers, mask-aware readout, no leakage through zero-padded convs), these must agree to numerical noise. If they do not, you have found a masking bug; this is a free unit test.
- Receptive field (~125 frames) exceeds short windows; that is fine as long as padding is masked.
- Treat speed robustness (resampling by ×0.75–1.33) as a *separate* experiment, not a window-length experiment.

**Q5. How to choose production window: clip-level val, session replay, or both?**
**Both, with different roles.** Clip-level val (grouped, chronological) is for fast iteration and checkpoint selection. **Session replay decides the production window and the operating point.**
Primary metric: **event recall at ≤10 FP per 1000 negative sessions** (dev gate), where:
- an *event* is one labelled theft interval, detected if an alarm fires on that track within `[t0−2 s, t1+4 s]`;
- a *false positive* is an alarm (after cooldown/dedup) on a negative session, or an alarm outside the tolerance of a theft event;
- report also recall at 2, 5, 20 FP/1000 and the partial AUC for FPR ≤ 1%.
Report the number of negative sessions and events behind every number, with day-block bootstrap CIs.

**Q6. Multi-scale training (2–8 s) on weak whole-clip labels?**
**Likely neutral-to-harmful without interval labels.** Cropping 2–4 s windows from a clip labelled positive creates positive windows that often do not contain the gesture (label noise increases as windows shrink). Conditions under which I would recommend it:
1. ≥300–500 positive clips carry `[t0,t1]` intervals (or reliable pseudo-intervals from a calibrated MIL model);
2. E1 shows a significant penalty for fragments of the length production actually emits;
3. restricted to {4, 6, 8 s}, with window label from overlap (below), not from clip label.
The cheaper alternative that captures most of the benefit is the **truncation augmentation**: 25% of training samples are truncated to U(3, 8) s with pad+mask, labelled by overlap with the positive interval. This mimics session-start/end fragments without teaching the model that short crops of a positive clip are positive.

**Q7. Two-stream 4 s evidence + 8 s context?**
**Not worth it now.** A TCN with RF ≈ 125 frames and mean+max readout already sees both scales and can localize evidence via max pooling. Two streams double the hyperparameters, complicate ensembling and calibration, and the real limit is label granularity, not temporal scale. Revisit only if (a) interval labels exist and (b) error analysis shows brief gestures (<2 s) being diluted. A lighter step first: replace or complement mean pooling with LSE/top-k pooling.

**Q8. Must offline eval mirror stride 4 and max-over-windows?**
**Yes, for any decision that goes to production (window length, threshold, ensemble, post-processing). No, for day-to-day training iteration.**
Protocol sketch:
- For each session, produce windows at stride 4 s for **four phase offsets** (0, 1, 2, 3 s); compute metrics per phase and report mean ± spread. Phase sensitivity is a real source of variance in production (a gesture can land at the start, middle, or end of a window).
- Apply the same fragment policy as production (re-anchored last window).
- Aggregate: max over windows, then the post-processing rules (persistence, cooldown/dedup).
- Stride 1 s is kept only as a development speed/recall probe; do not use it to pick thresholds.

**Q9. Windows inside the labelled segment only, or full npy?**
**Full npy, labelled by overlap.** Inside-only training teaches "action-onset cues", never shows the approach/walk-up or post-event behaviour, and produces FPs on exactly those phases in production. Recommended window labelling for class-6 clips:
- positive if overlap with the positive interval ≥ 1.5 s (≥ half of a 3 s tail);
- negative if overlap is 0 with a ≥1 s margin (approach windows of positive clips are *good hard negatives*: same person, same scene);
- **ignore** (zero loss weight) the in-between windows.
This also removes a subtle problem with a pure "tail positive" scheme: in production a 3 s gesture can sit anywhere in the 8 s window (start, middle, end). Training should randomize the offset (continuous random window start, not a fixed 1 s grid) so the gesture appears at all positions. Because of the 4 s stride, any 3 s gesture is fully contained in at least one 8 s window (start positions that contain it span 5 s ≥ 4 s), so recall is not structurally capped, but its position within the window varies.
Also use clip-balanced sampling (weight windows by 1/windows-per-clip) so long clips and long sessions do not dominate.

### Block B — Architectures

**Q10. Parameter budget?**
**0.05–0.3M parameters per model; do not exceed ~0.5M** until independent positive events exceed several thousand. Reasoning: the effective sample size is independent events/sessions (probably hundreds to ~1–2 thousand), not 2,592 clips; a useful heuristic is ≲ 100–300 parameters per independent positive event. Legacy #41 (1–2M params) was likely well over-parameterized for the effective sample size. Use dropout 0.1–0.3, weight decay, EMA of weights, augmentation, and seed ensembling (cheap variance reduction) instead of capacity.

**Q11. Final shortlist (6 candidates).** Details in §3. In priority order:
1. **Masked dilated TCN, w64** (primary): matches the data regime and the contract; in Campaign2.
2. **Masked BiGRU w64 with masked mean+max pooling**: second inductive bias for diversity; in Campaign2.
3. **LightGBM on hand-crafted window features**: information-ceiling probe, shortcut detector, and possible ensemble member; hours of work, minutes of compute.
4. **TCN with anatomical GCN-lite front-end** (2 spatial layers over the 8-joint anatomical graph + learned adjacency, then the same TCN trunk): tests whether spatial structure helps beyond engineered distances, without a new family.
5. **Small Transformer d64** (2–3 layers, conv stem 96→48, positional encoding, key-padding mask): one run, no tuning; expected low ROI but cheap on a 5090.
6. **Legacy #48 retrained with the v2 loader** (bridge): attributes gains to protocol vs architecture.

**Q12. ST-GCN lite next, or TCN + pooled GRU + seed ensemble sufficient for P2?**
**TCN + pooled GRU + seed ensemble is sufficient for P2.** With only 8 joints in 2D, the spatial graph is tiny and most of its value is already in the wrist–torso/hip distance channels. Do the GCN-lite *front-end* (candidate 4) only after the replay benchmark exists and only as a one-run comparison. A full ST-GCN/CTR-GCN reimplementation is not justified.

**Q13. Transformers/Conformers justified?**
**Low expected ROI, but affordable to test once.** With correct positional encoding and masking they will probably tie TCN/GRU at this data size and be less stable and less calibrated. Give them one slot (candidate 5), with a conv stem to cut sequence length, and no hyperparameter sweep. If they do not beat TCN within noise, drop them.

**Q14. Retrain legacy #48 with the v2 loader?**
**Yes, one config × 3 seeds.** It is the production incumbent, and without it you cannot separate "the protocol fixed it" from "the architecture fixed it". Together with the P0 replay of the legacy checkpoints this gives a 2×2 (legacy/v2 protocol × legacy/v2 architecture). Skip #41.

**Q15. Deployment ensemble?**
**(d): mean of logits over 3 seeds × {TCN, GRU}, calibrated on the session-level order statistic**, with a fallback to **(a)** the 3-seed TCN mean if the replay shows no benefit from adding GRU (paired bootstrap). Reasons: seed averaging is pure variance reduction; family diversity (convolutional vs recurrent) gives some decorrelation; and averaging logits behaves smoothly where AND of saturated sigmoids does not. **Do not use AND at 0.80**: with saturated sigmoid tails the 0.80 threshold sits on a nearly flat part of the score distribution, so FP counts jump with tiny changes, and AND of two highly correlated scorers reduces FP less than expected while cutting recall. **(c)** (keep #41+#48) only as the interim until v2 passes the replay gates; never swap on clip-level metrics alone.

**Q16. PoseC3D, MS-G3D, etc.?**
**No.** PoseC3D relies on keypoint heatmap volumes and many more joints/resolution than 8 noisy upper-body points; MS-G3D is designed for 25-joint 3D skeletons with large datasets. The bottleneck is label semantics, negative distribution, pose parity and evaluation unit, not model capacity. The skeleton-only information ceiling is the real limit (own phone vs stolen item); the next investment after Stage 1 stabilizes is Stage 1.5 hand-ROI features, not a heavier skeleton model. The GBDT probe will estimate that ceiling cheaply.

**Q17. Does architecture choice shift for 5 s fragments?**
**Only marginally.** GRU handles variable length natively; the TCN is fully convolutional with RF (125) larger than the fragment, which is fine under masking. The difference between TCN and GRU on short inputs will be smaller than the effect of (a) whether training saw truncated inputs, (b) mask-aware normalization and readout, and (c) the policy for fragments (re-anchor, abstain). Prefer the ensemble (both families), not a per-length architecture choice.

### Block C — Beyond windows & architecture

**Q18. Top 5 non-architecture levers for production FP (ranked)**

| Rank | Lever | Why | Expected FP effect (rough) | Cost |
|---|---|---|---|---|
| 1 | **Production-replay benchmark + session-level threshold (chronological, embargoed)** | Fixes the unit of measurement; the clip-level operating point is too lenient for session max-aggregation; enables every other decision | 1.5–3× from operating point alone [M] | M |
| 2 | **Production-representative negatives**: background-window mining, weekly review of top-scoring windows in unflagged sessions, hard-neg flywheel | Negatives are free in production (1,000 sessions/day) and cover the manifold the model has never seen | 2–5× at fixed recall [M] | M (ongoing) |
| 3 | **Pose / time-base parity** (score-parity test; train on Triton-extracted poses if gap) | Unknown magnitude, could be dominant; wrist jitter feeds velocity channels directly | 1–3× if a gap exists [L–M] | S to diagnose, L to fix |
| 4 | **Label semantics + interval labels** (own-item triage, `[t0,t1]` on 300–500 positives, OOF label audit) | Removes irreducible noise from positives and gives the model the right window labels | 1.3–2× [M] | M |
| 5 | **Post-processing** (persistence, cooldown/dedup, abstention, per-camera calibration) | Cheap, deterministic, testable on replay | 1.5–3× at 5–15 pp recall cost [M] | S |

Next in line: chronological split with embargo (belongs with #1), more independent theft sessions (not more curated negatives), Platt/temperature calibration on a separate partition, light hand-ROI cues (Stage 1.5).

---

## 3. Concrete proposal (binding recommendation)

### Production temporal contract
- **window_seconds:** 8
- **stride_seconds:** 4 (evaluate 2 in replay as an option; change only if replay shows a clear gain at equal FP)
- **target_hz:** 12, with explicit per-frame timestamps and resampling in the service; **verify first** that chunks are not native-fps frames (60 frames in 8 s = 7.5 fps)
- **seq_len:** 96 (= 12 × 8)
- **Short-fragment policy (<8 s):**
  - session ≥ 8 s: re-anchor the final window to [T−8, T] (full-length input, with overlap) instead of emitting a truncated chunk;
  - session 4–8 s: right-pad with mask, score, but require a higher persistence/threshold margin (calibrated on replay);
  - session < 4 s: abstain (log counts);
  - split windows at track discontinuities (ID-switch jumps) and treat the jump as a mask gap.

### Training protocol
- **Window lengths used in training:** 8 s (primary), plus truncation augmentation (25% of samples, length U(3, 8) s, padded and masked).
- **Multi-scale:** **no** for the next campaign. Conditional E2b (uniform {4, 6, 8 s}) only if interval labels exist on ≥300 positives *and* E1 shows a significant short-fragment penalty. E2a (a dedicated model at the E1-winning shorter length) only if E1 shows the 5 s penalty is real.
- **Stride in training:** continuous random window starts (frame-level), clip-balanced sampling; the 1 s grid is for eval/dev only.
- **Label policy on class-6 clips:** window positive if overlap with the positive interval ≥ 1.5 s; negative if no overlap with ≥1 s margin; ignore in between. Use the 3 s tail as the default interval until annotated intervals exist; refine from the annotated subset. Clips flagged "own item / not theft" become either ignored or targets of a separate auxiliary head (concealment-non-theft), never plain positives.
- **Other:** logits + BCE + MIL (LSE) + concealment auxiliary; cosine LR, AdamW, EMA of weights; sampler prior shift corrected by calibrating on natural prevalence; augmentation: mirror (swap L/R joints), time-stretch ±15%, jitter noise matched to the measured Triton-vs-Ultralytics jitter, joint dropout and gaps, small rotation (±10°). Use `poses_full` + valid mask only (production emits that); keep `poses` (filtered) only as a one-off ablation inside the current 12-job matrix, and adopt it only if production can reproduce the identical filter at runtime.

### Evaluation protocol (mandatory before deploy)
- **Splits:** chronological by day, grouped by source video/session/track, with ≥24–48 h embargo between partitions: train (earliest ~70%), T (threshold/calibration, ~10%), val_dev (~10%), final test (latest ~10%, touched once per phase) and a sealed shadow window. Add a leave-one-camera-out diagnostic (not for selection).
- **Metrics:** event recall @ FP/1000 negative sessions ∈ {2, 5, 10, 20} with day-block bootstrap CIs; partial AUC (FPR ≤ 1%); per-camera FP; seed-to-seed spread; paired bootstrap for every comparison. Checkpoint selection on mean recall at FP ∈ {5, 10, 20}/1000 (or pAUC), smoothed over epochs/EMA; not on a single 11-FP point. Do not report F1 for decisions.
- **Threshold:** order statistic on T at *session level after post-processing*; logit temperature/Platt on T; frozen before val_dev/test.
- **Multi-length eval grid:** 8, 5, 4, 6, 3 (12 optional; 2 diagnostic), each native-length and padded-to-96 control; stride 4 with 4 phase offsets.
- **Production replay:** **yes.** Negative replay ≥3,000 random unflagged sessions (target ≥10,000 over time), positive replay = all stored theft sessions (≥100 events if possible), using stored production tracks and exact chunking/overlap, not re-tracked data. First sanity check: legacy #41+#48 must reproduce ≈ 20–30 FP/1000 on the negative replay.

### Architecture plan (max 6 candidates)
1. Masked dilated TCN w64 — 3 seeds (17, 42, 103).
2. Masked BiGRU w64, masked mean+max pooling — 3 seeds.
3. LightGBM on window features (distances, speeds, hand-to-hip proximity/dwell, jerk, mask quality; plus camera/length features *only* as a shortcut probe) — 1 config, grouped CV.
4. TCN + anatomical GCN-lite front-end — 3 seeds, one run of hparams.
5. Small Transformer d64 with PE + mask + conv stem — 1 config, 3 seeds, no tuning.
6. Legacy #48 under the v2 loader — 3 seeds.
Settings: 100 epochs with cosine LR + EMA, one pose source (`poses_full`), TCN-only hparam budget ≤6 configs (width {32, 64, 96}, dropout, tail/overlap threshold). **Ensemble:** mean of logits within family over seeds, then across {TCN, GRU}; add candidates 3/4 only if replay shows a paired-bootstrap gain. Total ≈ 15 neural runs; this fits one afternoon-to-overnight on the 5090.

### Explicitly DO NOT do (save compute)
- A 60-model (or any large) architecture × length grid; vary one axis at a time.
- 2 s windows as a candidate production length.
- Multi-scale training without interval labels.
- Two-stream models for now.
- PoseC3D / MS-G3D / full ST-GCN reimplementation.
- AND-of-two-large-models ensembles at 0.80.
- Threshold or ensemble tuning on val_dev/test; per-clip splits; stride-1 results as deployment evidence.
- Re-adding focal + asymmetric + mining + UID-boost loss stacks.
- `single_user_only` filtering that discards multi-person robberies.
- More *curated* negatives from the same categories; invest in production negatives.
- Selecting checkpoints on a single FP point.

### 4-week execution order
- **Week 1:** (a) time-base check on production chunks (timestamps, fps, frame drops) and a score-parity test (same 50–100 videos through Ultralytics vs Triton path; compare jitter spectra and model scores); (b) finish the full 12-job matrix; (c) lock the chronological split; (d) build the session-replay evaluator (stride 4, 4 phases, max aggregation, cooldown), run legacy #41+#48 through it and confirm ≈ 20–30 FP/1000; (e) start class-6 triage (own-item flag).
- **Week 2:** E1 multi-length eval on TCN/GRU seeds; GBDT probe; bridge #48; set up the unflagged-session harvest and the weekly top-score review queue; begin `[t0,t1]` annotation (300–500 positives); out-of-fold scores for label audit.
- **Week 3:** retrain with overlap labels + ignore zone + truncation aug + hard negatives + background windows; E2a only if E1 demands; GCN-lite and Transformer single runs; session-level calibration; post-processing grid (persistence, cooldown, abstention, per-camera thresholds).
- **Week 4:** choose the ensemble on replay (paired bootstrap); shadow-mode deployment in parallel with #41+#48 (no alarm impact) for ≥7–14 days; if a pose-parity gap was found, start Triton re-extraction of training clips (weeks 3–6).

---

## 4. Legacy diagnosis: why "okay val, poor production"

Estimated contribution to the production gap (my ranking; magnitudes are guesses to be confirmed on replay):

1. **Evaluation unit mismatch / multiple looks [H].** Clip-level FPR 0.68% vs session = max over ~14 correlated windows. With k ≈ 3–5 effective looks, 2–3.4% session FPR is expected from the legacy numbers alone. Much of the "gap" is aggregation, not model failure.
2. **Winner's curse and operating-point selection [H].** 11 FP out of 1,619 has a 95% Poisson interval of roughly 5.5–20 FP (0.34–1.2%). That point was chosen as the best of ~20 models × many pairs × thresholds (H3), on val. The expected FPR on fresh data is meaningfully higher (1.5–3× is plausible).
3. **Negatives are curated action segments [H].** The production negative manifold (reaching, adjusting, phone use, occlusion, ID switches, transitions) was never sampled. Val prevalence (~13% positives, derived) also inflates F1 and precision relative to production.
4. **Leakage (H2) [H].** Per-clip split with same-day/camera/person leakage inflates recall and deflates FP on val.
5. **Time-warp shortcuts (H4) [H].** `temporal_resize` warped 4 s and 30 s clips to the same length; clip duration correlates with class, so the model can use apparent speed/duration, a shortcut that does not exist on fixed windows. Velocity computed before resize became physically inconsistent.
6. **Calibration artefacts [M].** Focal + asymmetric + mining + UID boost produces saturated sigmoid tails; "AND @ 0.80" then sits on a flat region, FP counts are unstable, and the threshold means little across data shifts.
7. **Checkpointing and capacity [M].** H1 (last epoch shipped) plus 1–2M-param models trained without scheduler/early stopping/grad clip on a small effective sample size: high variance, overfit to session specifics.
8. **Pose domain gap [M, unknown magnitude].** Raw (vx, vy) velocity channels amplify keypoint jitter; TensorRT/FP16 changes jitter statistics; the model may read jitter as motion energy. Plus a possible fps/time-base mismatch (Exec. summary #2).
9. **Multi-person / ID switches [M].** `single_user_only` removed ~33% of robberies (positives biased to isolated people), while production tracks include crowding, occlusion and ID-switch splices (teleporting joints → huge velocities → FP).
10. **Label semantics [H].** Own-phone-in-pocket and similar are in class 6; skeleton cannot separate ownership. This is an irreducible FP source at Stage 1.
11. **Architecture bugs [M].** `out[:, -1]` BiRNN pooling, no-PE Transformers, wrong ST-GCN graph: these make the *ranking* of architectures untrustworthy, and they biased the choice of the incumbents.

**Status after Campaign2 (corrections to the brief's §6 table):**

| Issue | Status | My comment |
|---|---|---|
| Time warp | Fixed | Keep truncation augmentation to cover short fragments |
| H1 checkpoint | Fixed | Also stop selecting on one 11-FP point (selection noise) |
| H2 leakage | Partial | Move to chronological + embargo, group by session/track |
| H3 threshold | Partial | Move threshold to session level after post-processing |
| Features/normalization | Improved | Check jitter sensitivity of velocity channels; consider smoothing or noise aug |
| RNN pooling | Fixed in v2 | Legacy remains only a baseline |
| Label semantics | **Open** | Own-item triage + intervals |
| Background windows | **Open** | Highest-ROI data lever |
| Pose parity | **Open** | Run score-parity test now |
| Metrics unit | **Open** | Replay benchmark |
| Window label scheme | **New concern** | "tail_positive" fixes gesture position inside the window; use overlap labels + ignore zone |
| Sampler prior | **New concern** | 25/25/50 window sampling shifts the prior; calibrate on natural prevalence only |

---

## 5. Ranked roadmap (4–8 weeks)

Ordered by expected reduction in session FP/day at acceptable recall, with engineering cost (S/M/L).

| # | Item | Expected effect | Cost | When |
|---|---|---|---|---|
| 1 | Production-replay benchmark (neg + pos replay), chronological split + embargo, session-level thresholding | Correct operating point and measurement; 1.5–3× | M | Wk 1–2 |
| 2 | Time-base and pose-parity diagnostics (timestamps/fps, score-parity test) | Possibly large; unknown until measured | S | Wk 1 |
| 3 | Negative flywheel: background-window mining, weekly review of top-scoring unflagged windows, unflagged sessions as free tail-calibration negatives | 2–5× at fixed recall | M, ongoing | Wk 2→ |
| 4 | Post-processing: persistence (k-of-n), cooldown/dedup, abstention (mask quality, short/ID-switch tracks), per-camera thresholds | 1.5–3×, −5–15 pp recall | S | Wk 3 |
| 5 | Label work: own-item triage, `[t0,t1]` on 300–500 positives, OOF-guided audit | 1.3–2× | M | Wk 1–3 |
| 6 | Retrain with overlap labels + ignore zone + truncation aug + hard negs (+ shortlist models) | Compounds 3–5; model-level gains | S–M | Wk 3 |
| 7 | Short-fragment policy (re-anchor last window; abstain <4 s) | Reduces boundary FP | S | Wk 3 |
| 8 | Multi-length eval (E1), then E2a only if justified | Selects the window with evidence | S | Wk 2–3 |
| 9 | Shadow deployment ≥7–14 days vs incumbent | Validates replay representativeness | S–M | Wk 4 |
| 10 | Re-extract training clips on the Triton path (if a parity gap is found) | 1–3× if gap exists | L | Wk 3–6 |
| 11 | Multi-scale training / GCN-lite / Transformer / two-stream | Marginal | M each | Wk 6–8, only if justified |
| 12 | Stage 1.5 hand-ROI cues; collect more *independent theft sessions* | Addresses the skeleton-only ceiling | L | Wk 6–8+ |

---

## 6. Acceptance criteria

All in session/event units, on chronological held-out data, with CIs.

- **Gate 0 — benchmark validity:** legacy #41+#48 on the negative replay reproduces production FP (≈ 20–30 per 1000 sessions) within its CI. If not, the replay is not representative; fix it before trusting any comparison.
- **Gate 1 — dev success:** v2 ensemble reaches **≥ 50% event recall at ≤ 10 FP/1000 negative sessions** (upper 95% CI of FP ≤ ~15/1000) *and* beats legacy #41+#48 at the same session-FP budget by ≥ 10 pp recall (paired bootstrap, CI excluding 0). Certified on ≥3,000 negative sessions and as many events as exist (≥100 preferred). Rule of three: zero FP in 3,000 sessions bounds FPR at ≤ 0.1%; 10 FP/1000 needs ~3,000 sessions for a ±~35% relative CI.
- **Gate 2 — shadow validation:** ≥7–14 days in shadow mode; observed FP/1000 within 1.5× of the replay estimate and recall within the replay CI.
- **Gate 3 — aspirational:** ≥ 50% recall at ≤ 1 FP/day (= ~1/1000 sessions). Certifying this needs ≥10,000 negative sessions or several weeks of shadow. My expectation [M]: skeleton-only Stage 1 will not reach this alone because of label semantics; plan Stage 1 as a high-recall filter feeding Stage 2.
- **Robustness:** seed-to-seed std of recall@FP ≤ 3 pp; no camera with > 3× the mean FP rate; phase-offset spread (4 phases) ≤ 5 pp recall; padded-vs-native consistency tests pass.
- **Reporting:** every claim states n_negative_sessions, n_events, CI method. No clip F1 in decision tables.

---

## 7. Risks & open assumptions

1. **Effective sample size is unknown.** I assumed independent events/sessions are far fewer than 2,592; if they are much fewer (≈ hundreds), even the 0.14M TCN may over-fit and the GBDT probe may win.
2. **Time base:** the 7.5 fps hypothesis is inferred from 60 frames / 8 s; the file could be a legitimate 5 s fragment. The check is cheap; if it fails, priorities change.
3. **Session length and independence:** the "k ≈ 3–5 effective looks" figure is an assumption; the replay benchmark will measure it.
4. **Tail-positive semantics:** I assumed the positive gesture sits in the last ~3 s of class-6 clips. If positives are actually earlier, overlap-labelling with the 3 s default will mislabel windows; the interval annotations resolve this.
5. **Unflagged production sessions as negatives** carry a small unlabeled-theft contamination; this makes calibration slightly conservative but is acceptable if theft prevalence is ≪ 1%.
6. **Label ceiling:** own-item vs theft is not observable in skeleton data; part of the FP floor is irreducible at Stage 1.
7. **Replay fidelity:** stored production tracks may differ from live behaviour (tracker version, load-dependent frame drops); a replay not built from stored production chunks may be optimistic.
8. **One store / 9 cameras:** thresholds and gains may not transfer to a new store; keep a leave-one-camera-out diagnostic and re-calibrate per deployment.
9. **Drift:** lighting, clothing, season and layout changes shift score distributions; schedule monthly threshold re-calibration on fresh unflagged sessions.
10. **My effect sizes are rough priors.** Treat them as ordering guidance, not forecasts; the replay benchmark is what turns them into numbers.
