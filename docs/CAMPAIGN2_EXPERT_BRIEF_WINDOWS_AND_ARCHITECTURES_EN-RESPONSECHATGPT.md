# Expert response — Kanvis Stage-1 window length & architecture strategy

**Date:** 2026-10-17  
**Basis:** The self-contained Campaign2 expert brief supplied with this assignment. Recommendations below distinguish stated facts from proposed operating choices. No repository inspection or additional data was available.

## 1. Executive summary

- **Do not restart a broad architecture search.** Campaign2 removes the worst legacy time-warp and checkpointing defects; the highest-value remaining work is production-like evaluation, labels, background negatives, and pose parity.
- Keep **8.0 s at 12 Hz (96 steps)** as the reference training and initial production contract. Do not change production window length based on clip-level metrics alone.
- First run **same-weight, multi-length evaluation** at 4, 6, and 8 seconds. Add 2 and 3 seconds as diagnostic probes, not automatic candidates for deployment; test 12 seconds only if replay shows that 8 seconds lacks context and the service can tolerate it.
- For variable-duration evaluation, preserve **physical time**: 12 steps per second. Do not temporally subsample a short interval into 96 steps, because that changes gesture speed and the meaning of velocity.
- Make **session replay the deployment gate**. The primary operating metric should be event recall at a frozen threshold subject to a false-positive budget, initially **≤10 false-positive sessions per 1,000 sessions**; report FP/day and event recall as well. The aspirational target of ≤1 FP/day is a later, harder gate.
- Offline replay must reproduce production’s **8-second windows, 4-second stride, short-fragment policy, aggregation, cooldown/deduplication, and score threshold**. A 1-second offline stride is useful for diagnostics and training coverage, but is not a faithful final deployment evaluation.
- Keep the v2 **masked dilated TCN-64** as the first baseline; use 3 independent seeds (17, 42, 103). Select checkpoints and thresholds on designated development/calibration data, then freeze them before final test replay.
- Compare a **correctly pooled GRU** and one small, correctly masked Transformer with positional encoding only after the baseline/replay path is trustworthy. Implement a lightweight anatomical ST-GCN if those comparisons justify another inductive bias; do not make it a prerequisite.
- Prefer the **mean of three TCN seed margins** as the first v2 ensemble. Compare it with the best single seed at the same FP budget; deploy the ensemble only if it improves the held-out operating frontier enough to justify added inference cost.
- Do not carry forward the legacy **#41 + #48 AND @ 0.80** as the preferred new-stack ensemble. Keep it only as a frozen reference baseline until a production replay comparison replaces it.
- Do not train random 2–8-second multi-scale windows with weak clip labels as the first intervention. Multi-scale training can teach duration invariance, but can also label background/approach motion as positive and blur the target without reliable event intervals.
- Add representative **background windows from full sessions**, hard negatives (especially phone-to-pocket and ordinary reaching), and session/event-level labels. The effective sample size is independent sessions/events, not the count of overlapping windows.
- Run a golden pose-parity test between offline pose extraction and Triton/TensorRT, including missing joints, coordinate normalization, frame timing, and short tracks. Model comparisons are not interpretable while this domain gap is unknown.
- Treat 5-second fragments as **60 physical-time frames with a mask**, not as 96 frames made by time-warping. The current TCN and pooled GRU can both handle them if masking/pooling is correct; fragment policy and calibration matter more than choosing a different architecture.
- A campaign succeeds only if the frozen system reduces false alarms **per session/day at a stated event recall**, on chronological/session-disjoint replay data—not merely if clip F1 improves.

## 2. Answers to every numbered question (Q1–Q18)

### Block A — Window length & temporal contract

#### Q1. Is full-npy vs 8 s chunk still the #1 mismatch?

**No—not for Campaign2 training.** The old full-clip `temporal_resize` mismatch was severe: it warped durations, could repeat terminal frames, made position and pre-resize velocity inconsistent, and exposed the model to clip-duration shortcuts. Campaign2’s fixed-duration 8-second windows substantially address that issue.

For the current stack, the leading unresolved mismatches are likely:

1. **Session-level evaluation and aggregation:** clip-max metrics do not estimate false alarms after repeated windows, overlap, cooldown, and deduplication.
2. **Training/evaluation sampling versus production stride:** offline stride is 1 s, while production starts windows every 4 s. This changes coverage and the number/correlation of opportunities to alarm.
3. **Background distribution:** labeled action clips are not representative random windows from long, mostly negative customer sessions.
4. **Short windows and padding/masking:** production sessions can start or end with fewer than 8 seconds.
5. **Offline/Triton pose parity** and remaining split leakage/label ambiguity.

The exact ordering of these causes is not proven without ablations and replay data. I would prioritize them in that order because the product metric is false alarms per session/day. The 8-second contract is a necessary improvement, not proof that train and production are aligned.

#### Q2. Run multi-length evaluation before multi-length training?

**Yes.** Evaluate the same trained 8-second weights at different physical durations first. This isolates whether the model’s learned evidence transfers to shorter or longer observations without confounding the result with changed training labels, optimization, or data sampling.

Use a consistent resampling rate and feature pipeline; evaluate every candidate through the same session replay, and calibrate each candidate’s threshold on calibration data only. Record both clip diagnostics and session/event metrics. This experiment is diagnostic: if the model was trained only on 8-second windows, poor short-window results may reflect distribution shift rather than prove that short windows are intrinsically inferior.

#### Q3. Which lengths first, and what stopping rule?

Recommended order:

1. **8 s** — reference and known production contract.
2. **4 s** — meaningful shorter context and an efficient first comparison.
3. **6 s** — tests a moderate reduction in context without going very short.
4. **2 s** — diagnostic for brief evidence; likely to be noisy and context-poor.
5. **3 s** — diagnostic between 2 and 4 seconds.
6. **12 s** — only if 8 seconds appears to miss context, and only if a longer production window is operationally acceptable.

This is deliberately not a large uniform grid. The owner’s 2/3/4/6/8 list is a reasonable diagnostic set; I would prioritize 4 and 6 before 2 and 3 for decision-making. The earlier 4/8/12 suggestion is also useful, but 12 seconds should be an extension rather than a first-line candidate because production currently uses 8 seconds and longer windows may include more irrelevant motion.

**Stopping rule:** stop evaluating a length if it cannot approach the reference’s event recall at the same false-positive-session budget on the development replay, or if it materially increases FP/session without a meaningful event-recall gain. Predefine a practical margin—for example, require a clear improvement beyond run-to-run/session-sampling uncertainty, not a nominal fraction of a percentage point. Keep no more than two lengths for a final, untouched chronological test. Do not select on that final test.

#### Q4. Should `seq_len` scale as 12 × seconds or stay at 96 with temporal subsampling?

Use **`seq_len = 12 × window_seconds`**, preserving physical-time sampling at 12 Hz: 24 steps for 2 s, 36 for 3 s, 48 for 4 s, 72 for 6 s, 96 for 8 s, and 144 for 12 s.

Do not squeeze 2–6 seconds into 96 steps by temporal interpolation/subsampling. That changes apparent gesture speed and velocity semantics, and it does not test the intended shorter physical observation. The model’s convolutional receptive field and pooling must be valid for variable sequence lengths; mask padding and invalid frames explicitly. If the current implementation only accepts 96 steps, adapt its masked temporal operations or build a separate inference wrapper without changing the time axis. Any architectural constraint should be stated and tested rather than hidden by time-warping.

#### Q5. Clip-level validation, session replay, or both?

**Both, with session replay primary.** Clip/window-level evaluation is a useful diagnostic for discrimination and failure analysis, but it cannot capture repeated false alarms across a customer session.

Primary metric: **event recall at a frozen threshold subject to ≤10 false-positive sessions per 1,000 negative/total sessions** on the defined replay population. State exactly how a “false-positive session” is counted and separately report false-positive events, because one session can contain multiple detections. Report:

- event recall, with event-level matching/tolerance rules specified;
- false-positive sessions per 1,000 sessions and FP/day at the observed daily volume;
- false-positive events per day;
- number of sessions/events, precision where prevalence is representative, and uncertainty intervals;
- recall at fixed FP budgets and the full session-level operating curve;
- latency/delay from event onset to first alarm, and duplicate alarms per event;
- slice metrics for short tracks, camera, pose source, and hard-negative type.

Use clip/window metrics for diagnosis only. Thresholds must be calibrated on a separate calibration split and frozen before evaluation on `val_dev`/test as appropriate. The final test should be chronological and session-disjoint.

#### Q6. Is random 2–8 s multi-scale training likely to help with weak whole-clip labels?

**Not as the next experiment.** It may improve robustness to varying available duration, but with weak whole-clip labels it risks training on the wrong evidence. A positive action clip can contain approach, walking, recovery, or unrelated motion; random windows may not contain concealment at all. If all sampled windows inherit the positive label, the model is explicitly rewarded for treating background as theft-like, which can raise production FP.

Recommend multi-scale training only after one or more of these conditions hold:

- reliable event intervals or substantially improved temporal labels exist;
- the MIL/tail-positive scheme has been validated on manually inspected windows;
- random negative/background windows from full sessions are included;
- every window has an explicit validity mask and a defensible label/weighting rule;
- multi-scale training is tested against the 8-second baseline on the same session replay and held-out events.

A conservative later test would keep 8 s as the dominant training duration and sample a limited proportion of 4–6 s windows with label-aware sampling, rather than uniform 2–8 s sampling. Do not change the production contract and training distribution at the same time.

#### Q7. Is a 4 s evidence + 8 s context two-stream design worth it?

**Not yet.** Brief concealment gestures make a short evidence stream plausible, while the longer stream may distinguish ordinary reaching from suspicious behavior. But two streams add calibration, fusion, latency, missing-data, and deployment complexity. They can also double opportunities for false alarms if the fusion rule is not carefully constrained.

First compare 4-second and 8-second single-stream models under replay. Consider two-stream only if the shorter model demonstrably detects events missed by the 8-second model and the 8-second stream demonstrably suppresses its false positives. Require a held-out session-level gain at the same FP budget and measure added latency. If tested, start with a simple calibrated score fusion, not a complex learned fusion head; tune only on calibration data.

#### Q8. Must offline evaluation mirror production 4 s stride and max-over-windows?

**Yes, for the final window-length/deployment decision.** It is fine to use 1-second stride for training window coverage, error analysis, or a diagnostic sensitivity study. But a candidate cannot be chosen for production based on an offline protocol with different window start times and aggregation.

Protocol sketch:

1. Replay full held-out sessions using the exact production resampling, window start/end rules, 8-second maximum window duration, **4-second start stride**, and short-fragment padding/mask policy.
2. Run the same per-window model and ensemble scoring used in serving.
3. Apply the same aggregation, threshold, cooldown, and deduplication rules as production. If current production uses max-over-windows, reproduce it exactly; do not substitute clip-max as a proxy.
4. Count false-positive sessions/events and matched true events, including the defined onset tolerance and duplicate handling.
5. Repeat with each candidate window length while keeping the comparison fair. If stride is held at 4 seconds, document that short windows then leave temporal gaps; alternatively test a separately specified stride policy as a distinct factor. Never silently give one window length denser scoring than another.
6. Use calibration sessions to choose thresholds and freeze them before the evaluation split.

A 1-second-stride offline score can supplement this replay, but cannot replace it.

#### Q9. Train only inside labeled segment or include approach/walk-up from full NPY?

Neither extreme should be applied blindly. **Do not label every window from a whole positive clip as positive.** That is likely to create false-positive shortcuts on approach/walk-up and background behavior. Equally, excluding all context can make the model brittle and may remove useful pre/post-action cues.

For class-6 clips, retain the current **tail-positive / MIL approach only as a hypothesis to validate**, not as established truth. Audit sampled windows against the actual clip construction and label semantics. Positive windows should overlap the plausible concealment interval when known; ambiguous windows should be masked, down-weighted, or handled as MIL bags rather than assigned a confident positive label. Use full-clip context to identify and sample hard negatives, not to indiscriminately inherit the clip label.

The ideal policy is event-interval supervision. Until intervals are available, define a reproducible window-label policy, manually review representative positive/negative windows, and quantify how often each policy labels obvious background as positive. Include ordinary approach/walk-up windows as negatives only when the annotation supports that interpretation; otherwise mark them uncertain.

### Block B — Architectures

#### Q10. Prudent parameter budget?

Start around **0.1–1 million parameters**, with the first baseline near the prior proposal’s **~0.14M**. A compact model in this range is a sensible bias for approximately 19k clips but only a few independent theft sessions/events, especially when overlapping windows do not increase independent sample size.

A 1–2M model can remain a comparator if it shows a repeatable session-level gain. Do not make parameter count the objective: compare quality at the same FP budget, calibration stability across seeds, latency, and failure slices. The brief does not establish a universal parameter ceiling; this is a conservative experiment budget, not a mathematical limit.

#### Q11. Final architecture shortlist (4–10 families)

Use the following **six** candidates, sequentially rather than all as an unfiltered grid:

1. **Masked dilated TCN-64 (baseline):** efficient local-temporal pattern learner, supports variable-length sequences with appropriate masks, and has a small deployment footprint.
2. **Masked dilated/residual TCN-96 (capacity ablation):** tests whether modestly more channel capacity helps; retain only if the gain is stable at the same FP budget.
3. **Correctly pooled GRU-64:** recurrent temporal dynamics comparator; pool over valid outputs or use a properly designed sequence representation, never rely on the legacy backward-direction `output[:, -1]` bug.
4. **TCN + pooled GRU hybrid:** tests local convolutional features plus longer temporal aggregation; only after each component is reliable and only if the added complexity yields a measured replay gain.
5. **Anatomical ST-GCN-lite:** uses the actual shoulder/elbow/wrist/hip connectivity rather than a chain in joint-index order; useful if spatial joint relations add value beyond engineered features.
6. **Small Transformer encoder with positional encoding and masks:** a controlled attention comparator; include only after the baseline, GRU, and replay protocol are stable.

The first campaign should be narrower: TCN-64, pooled GRU-64, and possibly TCN-96, each with 3 seeds. Add ST-GCN or Transformer based on failure analysis and available engineering time. This keeps the shortlist bounded and avoids spending compute on many nearly identical models. Compare `poses_full` versus filtered `poses` as a data/pose ablation, not as evidence that architecture itself improved.

#### Q12. Implement anatomical ST-GCN-lite next, or is TCN + pooled GRU + seed ensemble sufficient for P2?

**TCN + correctly pooled GRU + seed ensemble is sufficient for the immediate P2 decision.** Do not block production-quality evaluation on ST-GCN. The legacy ST-GCN used an incorrect graph, so its prior result does not fairly test the family; a small anatomical implementation is worth a later controlled experiment if the current models plateau or errors appear joint-relationship-specific.

A sensible sequence is: baseline TCN and seeds → session replay and hard-negative/label analysis → pooled GRU → ensemble comparison → ST-GCN-lite if there is evidence of remaining spatial-structure errors. Keep the graph anatomically defensible and validate joint ordering/masks with unit tests.

#### Q13. Are Transformers/Conformers justified, or low ROI?

They are **technically justified as one small comparator**, provided time positional encoding, attention masks, padding semantics, and regularization are correct. But expected near-term ROI is lower than fixing the data/protocol gaps. The dataset has ~19k clips but far fewer independent positive sessions/events; many overlapping windows are correlated. A Transformer can overfit these correlations and add calibration/latency burden without moving the production FP/recall frontier.

Do not run a Transformer/Conformer sweep now. After the baseline is stable, train one small positional Transformer with 3 seeds and compare on frozen session replay. Add a Conformer only if the Transformer shows a credible gain and there is a specific reason to expect convolution-attention hybridization to help. The brief provides no evidence that either family will outperform a compact TCN.

#### Q14. Retrain legacy #48 with the v2 loader as a bridge?

**Yes, as a low-priority bridge/reference if the implementation is cheap; otherwise skip it.** Retraining #48 under the v2 loader can help separate the effect of the old data/evaluation protocol from the effect of the architecture. It must use the same 96-step 8-second windows, masks, normalization, grouped splits, checkpoint selection, calibration, and session replay as the v2 baseline.

Do not compare its legacy reported metrics directly against new metrics or assume it is a strong candidate because it was in the old #41/#48 pair. If porting #48 requires substantial work, prioritize the clean pooled-GRU comparator instead; the scientific value is in the controlled architecture comparison, not preserving the model ID.

#### Q15. Deployment ensemble recommendation?

Choose **(a) mean of three TCN seed margins** as the first v2 ensemble. Convert each model’s output to a consistently defined score/margin (with calibration learned only on the calibration split), average the margins, then calibrate the ensemble operating threshold separately. This is a straightforward way to reduce seed variance without combining different inductive biases prematurely.

Compare it against the best single seed at identical session-level FP budgets. Deploy the ensemble only if its held-out event recall or stability improves enough to justify 3-model inference. If latency/cost requires one model and the ensemble is within about **2 percentage points of event recall** at the same FP budget, prefer the lighter model, consistent with owner preference.

Do not choose the legacy **#41 + #48 AND @ 0.80** as the new default. AND can suppress false positives but may lose recall, its thresholds were tuned under a flawed protocol, and legacy architecture rankings are not trustworthy. Keep it as a frozen reference during transition. Do not average raw scores from heterogeneous models without calibration. A TCN+GRU fusion is a later challenger only if it wins the same replay comparison.

#### Q16. Are PoseC3D, MS-G3D, or other skeleton video models worth the cost?

**Not now.** The likely bottleneck is data/protocol, label ambiguity, background coverage, and pose parity—not a demonstrated lack of model capacity. PoseC3D/MS-G3D-style approaches also require a careful representation/input pipeline and may introduce compute and deployment complexity. The supplied facts do not establish that they are a good fit for eight 2D joints and this low-FP objective.

Revisit only after session replay, hard-negative mining, interval-label improvements, and pose parity are in place, and if a documented error analysis suggests the compact temporal models cannot represent the needed pattern. Any trial should be one bounded candidate with a clear hypothesis, not a new broad grid.

#### Q17. For 5-second fragments (60 frames @ 12 Hz), does architecture choice shift?

**No major architecture shift is warranted.** A 5-second fragment is 60 physical-time frames. Feed 60 valid steps, or pad to the model’s supported batch shape with a mask that excludes padding from temporal pooling and losses. Do not repeat the last frame and pretend it is real motion; do not resample 5 seconds to 96 steps.

Both a masked TCN and a pooled GRU can handle the fragment. TCN receptive field and pooling should be checked so the valid 60 frames are not dominated by padding; GRU pooling must ignore invalid timesteps. The more important questions are whether a partial fragment has enough evidence, how scores are calibrated for partial observations, and whether session-start/end fragments are represented during training and replay.

Recommended short-fragment policy: preserve actual duration and a valid-frame mask; score when there is a minimum amount of valid evidence chosen from replay analysis; calibrate partial-fragment scores separately only if the data support it; otherwise suppress or defer scoring below that minimum rather than treating padding as evidence. Log how many sessions/events this policy excludes or delays. Do not assume that a 5-second example implies all sub-8-second fragments should be scored identically.

### Block C — Beyond windows and architecture

#### Q18. Rank the top five non-architecture levers for reducing production FP while preserving recall

Ranked by expected impact, with uncertainty because there is no session replay or ablation result yet:

1. **Production-like session replay benchmark and frozen operating point.** This makes the actual objective measurable: false-positive sessions/events per day at a chosen event recall. It also exposes repeated-window, cooldown, deduplication, and short-fragment effects. It may not itself improve the model, but it prevents optimizing the wrong thing and is the essential gate for all subsequent changes.
2. **Background-window collection and hard-negative mining.** Sample real negative windows from long customer sessions and mine false alarms, including ordinary reaching, phone-to-pocket, paying, adjusting clothing, and other visually/kinematically similar actions. Keep the mining loop session-disjoint from final evaluation. This directly targets the deployment negative distribution.
3. **Label semantics and temporal annotation.** Audit class-6 examples for concealment-like motions versus true theft intent, particularly the known own-phone-in-pocket ambiguity. Add event intervals or uncertainty labels where feasible; skeleton alone cannot infer object ownership. Improve the target definition and label noise rather than asking architecture to resolve unobservable intent.
4. **Offline-to-production pose parity.** Golden-test YOLO-pose versus Triton/TensorRT for coordinate scale, joint order, missing/low-confidence joints, frame rate/timestamps, tracking, and normalization. A pose-domain shift can produce systematic false alarms that model scaling will not fix.
5. **Grouped chronological split design and leakage control.** Group by session/person/track and preferably day/camera where feasible; keep all correlated windows and derived clips in one partition. Fit thresholds only on calibration data and reserve an untouched chronological test. This gives credible estimates and avoids overfitting thresholds to a convenient validation prevalence.

Close behind these are production post-processing (cooldown, event merging, hysteresis, per-session cap) and systematic hard-negative/error-slice analysis. Post-processing can reduce duplicate alarms quickly, but must be evaluated jointly with event recall and alarm delay; it should not hide a high per-window FP rate.

## 3. Concrete proposal (binding recommendation)

### Production temporal contract

- `window_seconds`: **8.0** as the initial reference.
- `stride_seconds`: **4.0** between production window starts, matching the stated C1 configuration.
- `target_hz`: **12**.
- `seq_len`: **96 valid-time positions for a full window**; for shorter evaluation windows, use `12 × actual_window_seconds` valid positions rather than time-warping to 96.
- **Short-fragment policy (<8 s):** preserve actual duration, use explicit valid-frame/joint masks, and exclude padding from normalization, pooling, and losses. Do not repeat the last frame as if it were observed motion. Establish a minimum valid duration from replay; until established, log partial-fragment results and avoid treating them as equivalent to full windows. If the current serving interface requires fixed tensor shape, right-pad only with an explicit false mask.
- Keep the production stride and post-processing contract fixed while comparing models. If changing stride or cooldown, treat that as a separate controlled experiment.

### Training protocol

- **Window lengths used in training:** initially **8 s only** for the primary baseline.
- **Multi-scale:** **No for the first campaign.** After label/window audits and background sampling, consider a limited, label-aware 4–6 s augmentation as a separate experiment; keep 8 s dominant. Do not use uniform random 2–8 s windows with inherited weak positive labels.
- **Stride in training:** retain the v2 **1-second sliding-window sampling** for coverage if storage/compute permit, but prevent highly overlapping windows from being treated as independent examples in split design or metric confidence. Sampling balance should be checked at the session level, not just window level. Record sampled-window counts and class composition.
- **Label policy on class-6 clips:** preserve the current tail-positive/MIL approach only as a provisional policy. Audit it against manually reviewed windows. Mark plausible event windows positive; use MIL for bags where only clip-level evidence is known; mask/down-weight ambiguous context rather than assigning all windows a positive label. Add a concealment auxiliary head only if the auxiliary target is consistently annotated and demonstrably helps the primary objective.
- **Negatives:** add representative background windows from full sessions and mined hard negatives as soon as a clean session-disjoint dataset is available. Do not use final test sessions for mining.
- **Data/splits:** split by session/person/track grouping and chronology before window generation. Keep all windows from a source clip/session in one partition. If same-day/camera conditions are strongly correlated, include a held-out day/camera slice where data volume permits.
- **Optimization:** AdamW, up to 100 epochs, early stopping/checkpoint selection based on the designated validation criterion, and fixed reproducible seeds **17, 42, 103**. Save a true copied best checkpoint and full config. Gradient clipping is reasonable for recurrent models; monitor train/validation curves and calibration rather than assuming 100 epochs is always needed.
- **Selection:** choose checkpoints by event/session recall at the defined FP budget where replay is available. Until replay exists, use the best available proxy only for development and explicitly do not call it deployment-ready.

### Evaluation protocol (mandatory before deploy)

- **Splits:** session/group-disjoint training, calibration (`T`), development (`val_dev`), and untouched chronological final test. No source session or near-duplicate clip may cross partitions. Document grouping keys and prevalence.
- **Metrics:** primary = event recall subject to **≤10 FP sessions per 1,000 sessions** on replay; report FP sessions/1,000, FP/day at observed traffic, false-positive events/day, event recall, alarm delay, duplicate alarms, precision at representative prevalence, and uncertainty intervals. Also report recall at a stricter aspirational operating point when sample size permits.
- **Threshold:** fit on `threshold_calib` only; freeze threshold, ensemble weights, aggregation, cooldown, and event-matching rules before `val_dev`/test. If a threshold is separately calibrated for a candidate length, that is allowed only on the same designated calibration split and must be clearly recorded.
- **Multi-length eval grid:** first **8, 4, 6 s**, then **2, 3 s** as diagnostic probes; test **12 s** only if the 8-second replay reveals a context limitation. Preserve 12 Hz physical-time sampling and use `seq_len = 12 × seconds`. Treat window length and stride as separate factors.
- **Production replay:** **Yes, mandatory.** Start with the largest clean chronological session set available; target at least several hundred to 1,000 sessions for an initial useful read, then accumulate enough independent days/events to estimate the desired low FP/day rate. Report exact denominators. A small replay is a smoke test, not evidence of ≤1 FP/day. Replay must use the real 4-second stride and serving aggregation/post-processing.
- **Comparisons:** use paired comparisons on the same sessions. Report per-camera/day and hard-negative slices; bootstrap confidence intervals by session/day, not by overlapping window. Do not repeatedly tune against the final test.

### Architecture plan (max 6 candidates)

1. **Masked dilated TCN-64** — primary baseline; target roughly 0.1–0.2M parameters if the implementation matches the proposed design.
2. **Masked dilated/residual TCN-96** — modest capacity ablation; run only if the baseline underfits.
3. **Correctly pooled GRU-64** — temporal-dynamics comparator with mask-aware pooling.
4. **TCN + pooled GRU hybrid** — only if the single-family models show complementary errors.
5. **Anatomical ST-GCN-lite** — later challenger with correct shoulder/elbow/wrist/hip graph.
6. **Small Transformer encoder with positional encoding and explicit masks** — one bounded comparator, not a sweep.

**Initial execution is only candidates 1 and 3**, plus candidate 2 if time permits. Use seeds **17, 42, 103** for each serious candidate; train the full three-seed baseline before drawing architecture conclusions. Start with 100-epoch ceilings and checkpoint by the predeclared criterion. Record parameter count, latency, seed variance, and failure slices.

**Ensemble:** mean of the three TCN seed margins, followed by threshold calibration on `T`. Compare to the best single TCN seed at the same FP budget. If the ensemble is not measurably better or is too costly, deploy the single model. Retain #41+#48 AND as a frozen transition baseline only. Do not fuse heterogeneous scores until their calibration is checked.

### Explicitly DO NOT do (save compute)

- Do not run a 60-model architecture grid or a large combinatorial search over architecture × window length × pose source × thresholds.
- Do not pick a production window using clip F1/max alone.
- Do not time-warp 2–6-second windows to 96 steps or treat repeated padding frames as real observations.
- Do not train all windows from a positive clip as positive without validating the label semantics.
- Do not tune the final threshold, ensemble, cooldown, or window length on the final test set.
- Do not interpret overlapping windows as independent samples or report window-level confidence intervals as if they were session-level.
- Do not assume the legacy #41/#48 ranking is valid under the new protocol.
- Do not prioritize PoseC3D/MS-G3D, a large Transformer/Conformer sweep, or two-stream fusion before session replay, background negatives, label audit, and pose parity.
- Do not deploy a short-window policy that scores padded/masked fragments as if they were full 8-second observations.

### 4-week execution order

**Week 1 — Make the benchmark trustworthy (highest priority).**
- Freeze and document grouped chronological splits before generating windows; audit source/session leakage.
- Implement production replay with 8 s windows, 4 s stride, real short-fragment handling, exact score aggregation, cooldown/deduplication, and event matching.
- Add session/event FP and recall metrics plus session/day-level confidence intervals.
- Golden-test offline versus Triton pose inputs and feature tensors on representative full and partial tracks.
- Manually inspect a stratified sample of class-6 windows and false-positive-prone negatives.

**Week 2 — Establish the clean baseline and error taxonomy.**
- Run masked TCN-64 with seeds 17, 42, 103 on the v2 8-second contract.
- Calibrate on split T, freeze thresholds, and replay on `val_dev`.
- Compare `poses_full` and filtered `poses` as a controlled data-source ablation; do not mix the conclusion with architecture changes.
- Build a categorized list of false alarms (phone-to-pocket, reaching, paying, tracking/pose failure, partial tracks, other) and missed events.

**Week 3 — Test the highest-value alternatives and data levers.**
- Train pooled GRU-64 with the same splits, features, seeds, and selection rule; add TCN-96 only if baseline diagnostics indicate underfitting.
- Run same-weight multi-length diagnostics at 4 and 6 seconds, then 2 and 3 seconds; use production replay for any candidate considered for deployment.
- Add a first batch of background windows and hard negatives from training-only sessions; retrain a controlled baseline if the data volume and labeling review support it.
- Compare seed ensemble versus best single seed at the same session FP budget.

**Week 4 — Freeze a candidate and validate honestly.**
- Select one production candidate using development replay, then freeze architecture, weights, threshold, stride, aggregation, and post-processing.
- Run once on the untouched chronological test; report event recall, FP sessions/1,000, FP/day, false-positive events/day, alarm delay, uncertainty, and slices.
- If the candidate fails, identify whether the dominant issue is labels, background distribution, pose parity, or model capacity before starting another experiment.
- Prepare the next data collection/annotation batch and a monitored shadow deployment; do not claim the aspirational ≤1 FP/day target from a small test.

## 4. Legacy diagnosis

Legacy performance was not a reliable predictor of production behavior. The most consequential issues were interacting protocol and implementation defects, not simply insufficient architecture capacity.

| Issue | Legacy behavior | Campaign2 status / remaining implication |
|---|---|---|
| Whole-clip time warp | Variable durations resized to fixed length; some clips padded by repeating last frame | Fixed 8-second windows remove the main mismatch; preserve physical time for other lengths |
| Position/velocity consistency | Velocity computed before temporal resize | Improved by a fixed-time window pipeline, but verify feature computation after resampling |
| Training/evaluation temporal mismatch | Whole clips in training, some ~3-second windows in eval | Partially fixed; offline 1-second stride and clip-max still differ from production 4-second stride/session replay |
| Checkpoint selection (H1) | Best checkpoint could be a live reference and file could contain last-epoch weights | Fixed by deepcopy; verify artifacts/configs and test checkpoint reload |
| Split leakage (H2) | Per-clip split, not robust session grouping; same-day/session leakage possible | Only partially fixed; enforce session/person/track grouping and chronological holdout |
| Threshold/ensemble tuning (H3) | Thresholds and AND/MEAN rules often optimized on validation; validation prevalence differed from production | Calibration split T is an improvement; freeze before evaluation and use representative session prevalence |
| Augmentation (H5) | Shift/scale augmentations could be null after normalization | v2 normalization is improved; audit augmentations in normalized feature space and add only realistic ones |
| RNN pooling (M1) | Legacy #41/#48 used `output[:, -1]`, problematic for bidirectional sequence representations | Pooled GRU in v2 avoids the known pattern if masking is implemented correctly |
| Graph structure | ST-GCN used a chain by joint index, not anatomical edges | Any future ST-GCN must use correct anatomy and tested joint ordering |
| Features/normalization (M2/M3) | Weak feature construction/normalization | Torso-relative, isotropic, wrist–torso features are an improvement; still check stability for low-confidence/missing joints |
| Label semantics | Class 6 includes concealment-like actions and own-phone pocketing | Still open; skeleton cannot establish object ownership or theft intent |
| Background distribution | Training mostly labeled action clips, not random windows from long sessions | Still open; major risk for production FP |
| Pose parity | Offline YOLO-pose versus production Triton/TensorRT | Still in progress; must be tested on golden examples |
| Metric unit | Clip F1/accuracy used as selection criteria | Mostly unresolved; session/event replay is mandatory |

The old #41+#48 AND result (roughly 76% F1, 64% recall, 11 false-positive negative clips out of 1,619) cannot be compared directly with 2–3% of production sessions flagged. The denominators, temporal sampling, aggregation, and prevalence differ. In production, a session can generate multiple overlapping windows and an alarm depends on the serving threshold and aggregation. A low clip false-positive rate can therefore still create an unacceptable session-level alarm rate across roughly 1,000 sessions/day.

The old time-warp issue plausibly contributed substantially, but it does not alone explain the remaining gap. The more defensible diagnosis is a combination of evaluation mismatch, leakage/threshold optimism, checkpoint risk, background-distribution shift, ambiguous labels, and pose-source shift. Campaign2 fixes several implementation defects, but until replay is aligned to production, architecture rankings remain provisional.

## 5. Ranked roadmap (next 4–8 weeks)

Rank is by expected reduction in false positives per session/day while preserving recall, not by scientific novelty. Impact is a reasoned priority, not a measured guarantee.

| Rank | Workstream | Expected FP/recall value | Engineering cost | Timing |
|---:|---|---|:---:|---|
| 1 | Production session replay, frozen threshold, exact aggregation/cooldown/dedup | Very high measurement value; reveals and enables targeted reduction | M–L | Weeks 1–2 |
| 2 | Background-window collection + training-only hard-negative mining | Very high direct potential for reducing common production false alarms | M–L | Start weeks 1–2; iterate weeks 3–8 |
| 3 | Class-6 label audit, temporal intervals/uncertain labels, phone-pocket hard negatives | High; reduces contradictory supervision and false theft-like alarms | M–L | Weeks 1–4, ongoing |
| 4 | Offline/Triton pose parity and feature golden tests | High if the domain gap is material; prevents systematic deployment errors | M | Weeks 1–2 |
| 5 | Session/person/track-grouped chronological splits and threshold calibration discipline | High for trustworthy estimates and generalization; may expose prior optimism | M | Week 1 |
| 6 | Production post-processing: event merging, cooldown, hysteresis, per-session dedup | Medium–high operational benefit; must monitor recall/delay | M | Weeks 2–4 |
| 7 | Controlled 4/6-second evaluation and pooled-GRU comparison | Medium; helps optimize temporal evidence and architecture only after replay is valid | S–M | Weeks 3–5 |
| 8 | Anatomical ST-GCN-lite or small Transformer, selected by error analysis | Low–medium expected near-term value; useful if a specific failure hypothesis exists | M–L | Weeks 5–8 |

A practical sequence is to complete ranks 1, 4, and 5 before interpreting any new architecture leaderboard; work on labels and background windows should begin in parallel because it may have the largest eventual direct impact. Do not treat post-processing as a substitute for fixing systematic false-positive scores.

## 6. Acceptance criteria

The next campaign should be accepted only when all of the following are met:

1. **Protocol reproducibility:** exact data version, grouped splits, pose source, feature configuration, seeds, checkpoint hashes, threshold, aggregation, stride, cooldown, and event-matching rule are recorded and reproducible.
2. **No leakage:** all windows derived from the same source session/person/track remain in one split; threshold calibration and final test are separate.
3. **Production replay:** candidate is evaluated on full sessions with the production 4-second stride and matching aggregation/post-processing, including short fragments.
4. **Primary operating point:** report event recall at **≤10 false-positive sessions per 1,000 sessions** on the designated replay set. This is the stated iterative target, not a guarantee of the final business goal.
5. **Daily interpretation:** convert measured FP/session to FP/day using the actual number of sessions/day and state the denominator. At 1,000 sessions/day, 10 FP/1,000 sessions corresponds to about **10 false-positive sessions/day**, not the aspirational ≤1/day. A ≤1/day target at 1,000 sessions/day corresponds roughly to **≤1 FP session per 1,000 sessions**, assuming the same counting definition and traffic mix.
6. **Recall preservation:** compare candidates at the same FP budget. Prefer a lighter candidate if it is within approximately **2 percentage points of event recall** and has materially lower latency/complexity, per owner preference.
7. **Uncertainty:** provide event/session counts and confidence intervals resampled by session/day. Do not claim success from a handful of positive events or from overlapping windows treated as independent.
8. **Hard-negative slices:** report FP rates for phone-to-pocket, reaching, paying, partial tracks, camera/day, and pose-quality categories where labeled examples exist.
9. **Event quality:** report duplicate alarms per event, alarm delay, missed event types, and any minimum-duration policy’s effect on recall.
10. **Final holdout:** do not tune on the final chronological test. A single final test result should confirm a decision already made on development replay.
11. **Shadow monitoring:** before broad rollout, monitor FP sessions/day, alarm delay, score distribution, missing-pose rate, short-fragment frequency, and camera-specific drift in shadow/limited deployment.
12. **Aspirational target treated honestly:** ≤1 FP/day at ~1,000 sessions/day requires far more evidence than a small clip validation set. If the available test contains too few sessions to estimate this rate, label it **not yet established**, even if zero false positives are observed.

## 7. Risks & open assumptions

- **Independent event count is unknown.** The ~2,592 positive clips are not 2,592 independent theft events; confidence and expected generalization depend on the true number of distinct sessions/events.
- **Label validity is limited.** Class 6 mixes concealment-like movement with business false thefts such as putting one’s own phone in a pocket. Skeleton alone cannot resolve ownership or intent.
- **Tail-positive policy may be misaligned.** The brief says only the last ~3 seconds are labeled positive; it does not establish that the target action always occurs there. Audit before treating this as ground truth.
- **Definition of FP is operationally important.** A false-positive session, a false-positive event, and a flagged window are different units. The replay benchmark must define event merging and session counting.
- **Production aggregation details may be incomplete.** The brief mentions max-over-windows and ensemble AND, but cooldown/deduplication and precise threshold semantics may vary. Mirror actual serving code and config.
- **Short-fragment behavior is not yet specified.** The minimum valid duration and whether partial tracks should be deferred, scored, or suppressed must be chosen from replay evidence.
- **Pose parity may dominate model choice.** Coordinate scaling, joint confidence/missingness, frame timestamps, tracker ID switches, and runtime implementation differences can shift the input distribution.
- **Split grouping may need more keys.** Session IDs alone may not prevent leakage if clips are derived from the same source video/day/person or if adjacent clips overlap. Use source provenance where available.
- **Prevalence matters.** A clip-balanced validation set does not provide production precision or FP/day without representative session sampling and a clearly defined denominator.
- **Threshold calibration is sample-hungry at low FP rates.** A calibration split with few negative sessions cannot reliably identify a ≤1/1,000 operating point. Use conservative uncertainty bounds and accumulate more replay data.
- **Architecture gains may be smaller than data effects.** The prior legacy frontier was stagnant, and the new pipeline has not completed the full run; do not assume the new baseline has already succeeded.
- **Window-length comparisons can be confounded.** Changing duration also changes stride coverage, amount of context, padding frequency, and number of scoring opportunities. Keep these factors explicit.
- **Compute concurrency is not the bottleneck by itself.** Eight parallel jobs may fit the RTX 5090, but the recommended plan intentionally limits experiment count to preserve time for review, replay, and data quality.
- **The proposed numerical targets are operating targets, not statistical guarantees.** The brief supplies aspirational ≤1 FP/day and iterative ~10 FP/1,000 sessions. Whether the system can meet them at acceptable recall remains empirical.
- **Campaign2 status is incomplete.** The full ~19k-clip run is pending; this recommendation is a protocol and experiment plan, not a claim about unobserved model results.

## Final decision

Proceed with the 8-second masked TCN baseline and three seeds, but make **production-like session replay, pose parity, and hard-negative/label work** the campaign’s center of gravity. Use multi-length evaluation before multi-length training; preserve physical time and masks; choose the production contract by event recall at a frozen session-level FP budget. Expand architecture only when a controlled replay experiment or error analysis gives a specific reason to do so.
