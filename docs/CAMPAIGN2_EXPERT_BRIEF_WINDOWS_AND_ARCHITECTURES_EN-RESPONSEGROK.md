# Expert Response — Kanvis Stage-1: Window Length & Architecture Strategy

**Date:** 2026-10-17  
**Author:** Senior expert (computer vision / HAR / deep learning)  
**Scope:** Stage-1 skeleton-sequence theft detector only. Business priority = false positives first.

---

## 1. Executive summary

- **Primary remaining train/prod mismatches are no longer full-clip vs 8 s warp.** Campaign2 already fixed the worst time-warping. Dominant residual gaps are: (a) offline stride 1 s vs production stride 4 s, (b) clip-max aggregation vs session-level replay with cooldown/dedup, (c) short fragments (<8 s) at session edges, (d) absence of true background windows from long sessions, and (e) pose source gap (YOLO vs Triton).
- **Do multi-length evaluation first, multi-length training later (or never).** Run the same 8 s-trained weights on 4 s / 6 s / 8 s (and optionally 3 s / 12 s) with correct masking; decide production window from session-replay metrics, not clip F1.
- **Keep production temporal contract at 8 s / 4 s stride / 12 Hz / 96 frames** unless multi-length eval + session replay clearly shows a better operating point. Short fragments (<8 s) must be masked, never zero-padded or last-frame-repeated into a full 96-step tensor without an explicit validity mask.
- **Architecture shortlist is deliberately tiny:** masked dilated TCN-64 (baseline), correctly pooled GRU-64, anatomical ST-GCN-lite, small Transformer-with-PE. Train 3 seeds each. No 60-model grid.
- **Ensemble:** mean of the three best TCN seeds (or TCN+GRU mean if the GRU is within ~2 pp recall at same FP budget). Drop the legacy #41+#48 AND rule as soon as a v2 model clears the session FP target.
- **Do not retrain legacy #48 as a bridge** unless you need a quick sanity check; the v2 loader + new selection metric already supersedes it.
- **Top non-architecture levers (ordered by expected FP reduction):** (1) session/event replay benchmark with frozen threshold, (2) background-window mining from real sessions, (3) label cleanup / “business false theft” (phone-in-pocket) handling, (4) pose train/prod parity, (5) hard-negative mining + post-processing cooldown.
- **Selection metric must change:** stop optimizing clip F1 or val accuracy. Optimize **event recall at a fixed session-FP budget** (target while iterating: ≤10 false sessions per 1000 sessions; aspirational ≤1 FP/day).
- **Parameter budget:** prefer models ≤0.5 M parameters; anything >2 M is almost certainly over-parameterized given the effective number of independent positive events.
- **Week-1 priority:** finish the pending ~19 k full Campaign2 matrix (masked_tcn64 + gru64_pooled × 3 seeds × 2 pose sources), calibrate thresholds on the held-out T split, then run multi-length eval + a first session-replay pilot.
- **Acceptance:** a campaign succeeds when, on a held-out set of real sessions (≥500–1000 sessions, ≥30–50 independent theft events), the chosen model + threshold + ensemble yields ≥50 % event recall at ≤10 FP / 1000 sessions (and preferably shows a path to ≤1 FP/day).
- **Risks that can invalidate the advice:** residual label noise (phone-in-pocket), large pose domain gap, severe class imbalance at the true event level, or session ID switches that fragment positive events.
- **Do not** spend compute on Transformers/Conformers or PoseC3D/MS-G3D until the protocol and data levers above are exhausted; expected ROI is low.
- **Two-stream (4 s + 8 s) is not justified** until single-scale 8 s is solid and multi-length eval shows a clear complementary signal.
- **Training windows should stay inside labeled segments** for positive class-6 clips (tail_positive policy is already a step in the right direction); sampling approach/walk-up as positive will inflate FP.

---

## 2. Answers to every numbered question (Q1–Q18)

### Block A — Window length & temporal contract

**Q1.** After Campaign2’s 8 s @ 12 Hz training, is the owner’s full-npy vs 8 s chunk concern still the #1 train/prod mismatch, or are stride (1 vs 4 s), session aggregation, and short chunks (<8 s) now dominant?

**Answer:** The original full-npy / temporal_resize mismatch is **no longer #1**. Campaign2 already trains and (mostly) evaluates on fixed 8 s physical-time windows. The remaining dominant mismatches, in approximate order of impact on production FP, are:

1. Offline stride 1 s + clip-max aggregation versus production stride 4 s + session-level aggregation / cooldown / deduplication.  
2. Short fragments (<8 s) at session start/end and the masking/padding policy applied to them.  
3. Training data drawn only from labeled action segments versus the true production distribution of long background sessions.  
4. Residual pose domain gap (offline YOLO-pose pipelines vs Triton).  

These four now dominate the train/prod gap that the owner observed.

**Q2.** Should the team run multi-length evaluation (same trained weights, vary window 2/3/4/6/8 s with proper mask/pad) before any multi-length training? Yes/no and why.

**Answer: Yes.** Multi-length evaluation is cheap (inference only), isolates the effect of window length from training dynamics, and gives an immediate ranking of which production window length is most compatible with the already-trained 8 s models. Multi-length *training* is expensive, requires careful label policy (especially for weak whole-clip labels), and is only justified if evaluation shows a clear benefit that cannot be obtained by simply changing the production window or by post-processing. Running evaluation first is the rational order.

**Q3.** If yes to multi-length eval, which lengths to test first (owner list 2–8 s vs prior expert 4/8/12 s)? Give an ordered list and stopping rule.

**Ordered list (first to last):**

1. 8 s (baseline, matches current training contract)  
2. 4 s  
3. 6 s  
4. 3 s (optional, only if 4 s is competitive)  
5. 12 s (optional, only if 8 s is clearly under-contextualized)  
6. 2 s (lowest priority; very short gestures risk losing context and increasing FP on ordinary motion)

**Stopping rule:** Stop after 8/4/6 s if one of them clearly dominates on the primary session-level metric (event recall @ ≤10 FP/1000 sessions) and the others are ≥3–5 pp worse on the same budget. Only expand to 3 s or 12 s if the top two lengths are within ~2 pp of each other and you need a tie-breaker or evidence of multi-scale complementarity.

**Q4.** For multi-length eval, should `seq_len` scale as `12 × window_seconds` (96 for 8 s) or stay fixed at 96 with temporal subsampling? Recommend one approach.

**Answer:** Prefer **scaling `seq_len = 12 × window_seconds`** (48 for 4 s, 72 for 6 s, 96 for 8 s, etc.) together with the same physical-time resampling and the same validity mask. Fixed-96 + temporal subsampling re-introduces a mild form of the old time-warping pathology and changes the effective receptive field in an uncontrolled way. Scaling keeps the temporal density (12 Hz) and the physical meaning of dilations / recurrent steps identical; only the number of steps changes. Architectures that are length-agnostic (TCN with global pool, GRU with masked mean/max, Transformer with PE + mask) handle variable length without code changes.

**Q5.** Should production window length be chosen by clip-level val, session replay, or both? Define the primary metric.

**Answer:** **Both, but session replay is primary.** Clip-level metrics are useful for rapid ranking and for debugging, but they do not capture cooldown, deduplication, ID switches, or the true session-length distribution.  

**Primary metric (binding):**  
Event recall (or session recall of true theft events) at a fixed false-session budget of **≤10 false sessions per 1000 sessions** (iterative target). Secondary: the same metric at the aspirational budget of ≤1 FP/day (≈1 false session per 1000 sessions at current volume). Report also the raw number of independent events detected and the FP rate per camera-hour if available.

**Q6.** Is multi-scale training (random 2–8 s windows during train) likely to help FP/recall on weak whole-clip labels, or hurt without interval annotations? Under what conditions would you recommend it?

**Answer:** On the current **weak whole-clip labels** it is more likely to **hurt** than help. Random short windows drawn from a long positive clip will often land outside the actual concealment gesture; the model then receives a positive label for ordinary motion, which is a direct FP generator. The existing `tail_positive` + MIL policy already mitigates this for the 8 s regime.  

**Recommend multi-scale training only if** (a) you obtain interval-level annotations (or strong MIL with known positive sub-intervals), **or** (b) multi-length evaluation shows that 4 s and 8 s are complementary and a single model must serve both, **and** you keep a strict “window must overlap the labeled positive interval by ≥X %” rule. Until then, stay single-scale (8 s).

**Q7.** Is a two-stream design (e.g. 4 s “evidence” + 8 s “context”) worth the complexity vs single 8 s for brief concealment gestures?

**Answer: No, not yet.** Brief gestures are already partially addressed by the `tail_positive` / MIL loss and by the possibility of evaluating at 4 s. A two-stream model doubles inference cost, complicates the production serving path, and requires careful fusion and threshold calibration. Only revisit after a clean single-scale 8 s (or 4 s) model has been deployed and the residual error analysis shows that short evidence + longer context is the dominant remaining failure mode.

**Q8.** Must offline eval mirror production `stride_seconds = 4` and max-over-windows aggregation to choose window length? Yes/no and protocol sketch.

**Answer: Yes** for the final selection of production window length and operating point.  

**Protocol sketch:**  
- Resample every clip (or, better, every real session) to 12 Hz.  
- Extract windows of the candidate length with stride = 4 s (or the production stride under test).  
- Apply the identical mask / short-fragment policy that production will use.  
- Score every window; aggregate by max (or the exact production aggregator).  
- Apply the same cooldown / deduplication that the serving stack uses.  
- Compute event recall and false-session rate on a held-out set of full sessions.  
Clip-max with stride 1 s can still be used for rapid intermediate ranking, but the go/no-go decision must use the production-mirroring protocol.

**Q9.** For clips labeled as whole actions, should training windows be restricted to inside the labeled segment only, or sampled on the full npy including approach/walk-up? Impact on FP?

**Answer:** Restrict positive windows to **inside the labeled segment** (or, better, to the tail of the segment as already implemented). Sampling approach/walk-up as positive teaches the model that ordinary locomotion preceding a concealment is itself positive; that is a classic source of production false positives on customers who simply walk past shelves or put a hand in a pocket for non-theft reasons. Negative windows may of course be drawn from the full timeline (and from background sessions).

### Block B — Architectures

**Q10.** Given ~19 k clips / ~2592 pos / few independent sessions, what is a prudent parameter budget (order of magnitude)?

**Answer:** **≤0.5 M parameters** is the comfortable zone; **≤2 M** is the hard upper bound you should rarely cross. The effective number of independent positive *events* is far smaller than 2592 clips (many clips share the same person, day, or even the same gesture). Models larger than ~2 M will over-fit the limited event diversity and the residual label noise. Prefer the lighter end of the Pareto frontier (within ~2 pp recall of the heavier model at the same FP budget).

**Q11.** Provide a final shortlist of 4–10 architecture families (not 60) to train after the 8 s baseline, with one-line rationale each.

**Shortlist (6 families):**

1. **Masked dilated TCN-64** (5 blocks, masked mean+max readout) — already the Campaign2 baseline; strongest inductive bias for fixed-rate temporal patterns and cheapest to train.  
2. **Correctly pooled GRU-64** (masked mean / max / last-valid) — complementary sequential inductive bias; catches longer-range dependencies that pure TCN may miss.  
3. **Anatomical ST-GCN-lite** (correct skeleton graph, 2–3 spatial-temporal blocks, global pool) — explicitly models joint topology; useful if residual errors are spatial rather than purely temporal.  
4. **Small Transformer encoder** (2–4 layers, d_model 64–128, sinusoidal or learned PE, full attention mask) — tests whether global attention helps once positional encoding and masking are correct.  
5. **TCN-GRU hybrid** (TCN front-end → GRU) — lightweight fusion of the two strongest inductive biases already in the shortlist.  
6. **Dilated TCN-128** (same depth, wider) — single controlled capacity check; only keep if it beats TCN-64 by >2 pp at the same FP budget.

(Stop at these six. Do not add PoseC3D, MS-G3D, full Conformers, or large Transformers.)

**Q12.** Should the team implement anatomical ST-GCN lite next, or is TCN + pooled GRU + seed ensemble sufficient for P2?

**Answer:** **TCN + pooled GRU + 3-seed ensemble is sufficient for the immediate next campaign (P2).** Implement anatomical ST-GCN-lite only after the TCN/GRU ensemble has been evaluated on session replay and residual error analysis indicates that spatial joint topology is a limiting factor. ST-GCN is a useful third comparator but not a blocker.

**Q13.** Are Transformers / Conformers justified on this data size if positional encoding + mask are done correctly, or expected low ROI?

**Answer:** **Expected low ROI.** Even with correct PE and masking, Transformers have higher sample complexity and are more sensitive to the limited number of independent events and to label noise. A small Transformer (item 4 above) is worth a single controlled experiment (3 seeds) for completeness; a full Conformer or large Transformer is not justified until the data and protocol levers are exhausted.

**Q14.** Should legacy #48 be retrained with the v2 loader (96-step windows) as a bridge, or skip?

**Answer: Skip.** The architectural differences (pooling bug, lack of proper mask, different selection metric) make a direct retrain of #48 a low-value bridge. The new masked TCN-64 and pooled GRU-64 already supersede it. If you need a quick sanity check that the v2 loader itself is not broken, a single seed of the old architecture under the new loader is enough; do not invest in a full 3-seed campaign for #48.

**Q15.** Deployment ensemble: recommend one: (a) mean of 3 TCN seeds, (b) TCN+GRU fusion, (c) keep #41+#48 AND interim, (d) other — with justification.

**Answer: (a) mean of the three best TCN seeds**, with a possible upgrade to (b) if the best GRU seed is within ~2 pp recall of the TCN at the same FP budget.  

Justification: mean of margins (or of logits before sigmoid) is more stable and better calibrated than the legacy AND rule; it does not require two completely different architectures to agree, and it is easy to implement in the serving stack. Keep the legacy AND only as a temporary production fallback until the v2 model has passed the session-replay acceptance criteria.

**Q16.** Are skeleton video models (PoseC3D, MS-G3D, etc.) worth the cost here, or is bottleneck data/protocol?

**Answer: Bottleneck is data/protocol.** PoseC3D / MS-G3D bring substantially higher compute, more parameters, and a 3-D convolution inductive bias that is rarely the limiting factor on short upper-body 2-D keypoint sequences. The stagnant FP–recall frontier under the legacy protocol already indicated that architecture was not the main lever; the same is true under the improved protocol. Skip them.

**Q17.** For 5 s production fragments (60 frames @ 12 Hz), does architecture choice shift (GRU vs TCN vs other)?

**Answer: Mild shift only.** Both TCN and GRU handle 60-frame inputs without difficulty provided the validity mask is applied correctly (no last-frame repeat into a full 96-step tensor). A pure TCN may be slightly more robust to the missing context because its receptive field is local; a GRU will simply see a shorter sequence. The bigger issue is the **masking / padding policy**, not the choice between TCN and GRU. Keep the same architectures; enforce that short fragments are scored with an explicit mask and that the production aggregator never treats a heavily masked window as a high-confidence positive unless the unmasked evidence is strong.

### Block C — Beyond windows & arch

**Q18.** Rank the top 5 non-architecture levers for reducing production FP while preserving recall.

1. **Session / event replay benchmark with frozen threshold** — closes the largest remaining train/prod metric gap; forces every decision (window length, threshold, ensemble) to be measured in the units that matter (FP/day, event recall).  
2. **Background-window mining from real long sessions** — supplies the missing negative distribution that pure labeled action clips cannot provide; directly attacks the ordinary-motion false positives.  
3. **Label cleanup / “business false theft” handling** (phone-in-pocket, own-item concealment) — either by re-labeling, by an auxiliary “ownership-unknown” head, or by down-weighting such clips; reduces the irreducible FP floor that skeleton-only models cannot solve.  
4. **Pose train / production parity** (golden tests, identical keypoint ordering, confidence thresholds, coordinate normalization) — removes a systematic domain shift that currently inflates both FP and FN.  
5. **Hard-negative mining + post-processing cooldown / deduplication** — mines the residual high-scoring negatives after each campaign and adds a simple temporal filter in the serving stack; cheap and high leverage.

(Honorable mentions: better chronological / session-grouped splits, explicit short-fragment policy tests, and a small amount of temporal jitter augmentation that respects the physical timeline.)

---

## 3. Concrete proposal (binding recommendation)

### Production temporal contract
- **window_seconds:** 8.0 (subject to multi-length eval; change only if session replay clearly favors 4 s or 6 s)  
- **stride_seconds:** 4.0 (match current production; do not train or evaluate final models with stride 1 s)  
- **target_hz:** 12  
- **seq_len:** 96 (= 12 × 8)  
- **Short-fragment policy (<8 s):** always emit an explicit validity mask; never last-frame-repeat or zero-pad without mask; score only if the fraction of valid frames exceeds a calibrated threshold (start at 0.6–0.7); otherwise treat the window as non-informative (score = 0 or ignore in aggregator).

### Training protocol
- **Window lengths used in training:** single-scale 8.0 s only for the next campaign.  
- **Multi-scale:** no (unless interval annotations become available or multi-length eval forces it).  
- **Stride in training:** 1.0 s is acceptable for data efficiency while the model is being trained; final selection and threshold calibration must be performed with production stride 4 s.  
- **Label policy on class-6 clips:** keep `tail_positive` (last ~3 s positive) + MIL bag loss; never label approach/walk-up frames as positive.

### Evaluation protocol (mandatory before deploy)
- **Splits:** chronological / session-grouped train / threshold_calib (T) / val_dev / held-out test; no leakage of the same person-day across splits.  
- **Metrics (primary):** event recall @ ≤10 false sessions per 1000 sessions; secondary: same at ≤1 FP/day aspirational budget; also report clip-level numbers for debugging only.  
- **Multi-length eval grid:** 8 s / 4 s / 6 s (then 3 s / 12 s only if needed) with `seq_len = 12 × window_seconds`, production stride, and identical mask policy.  
- **Production replay:** yes — minimum 500–1000 real sessions containing ≥30–50 independent labeled theft events; apply exact production aggregator + cooldown.

### Architecture plan (max 6 candidates)
1. Masked dilated TCN-64 (baseline, ~0.14 M)  
2. Correctly pooled GRU-64  
3. Anatomical ST-GCN-lite (after TCN/GRU are measured)  
4. Small Transformer (2–4 layers, d=64–128, PE + mask)  
5. TCN-GRU hybrid (optional, only if both parents are strong)  
6. Dilated TCN-128 (capacity check only)

- **Seeds:** 17, 42, 103 (or any three fixed seeds).  
- **Epochs:** 100 with early stopping on recall@FP budget on val_dev; AdamW, modest weight decay, gradient clip.  
- **Ensemble:** mean of the three TCN seeds (upgrade to mean of best TCN + best GRU if the latter is within ~2 pp).  
- **Pose sources:** continue to train on both full and filtered npy for one more matrix; prefer the source that wins on session replay.

### Explicitly DO NOT do (save compute)
- 60-model architecture grid or any broad sweep of legacy families.  
- Retraining legacy #41 / #48 as a full campaign.  
- Multi-scale training before multi-length evaluation and interval annotations.  
- PoseC3D, MS-G3D, large Transformers, or Conformers.  
- Two-stream 4 s+8 s models.  
- Optimizing thresholds or ensembles on the same split used for model selection.  
- Using clip F1 or val accuracy as the primary selection metric.

### 4-week execution order
- **Week 1:** Finish the pending ~19 k Campaign2 matrix (masked_tcn64 + gru64_pooled × 3 seeds × 2 pose sources). Calibrate thresholds on T. Run multi-length eval (8/4/6 s) with production stride. Stand up a first session-replay pilot (≥500 sessions).  
- **Week 2:** Select the best single model + window from session-replay metrics. Implement and validate the short-fragment mask policy end-to-end. Begin background-window mining pipeline.  
- **Week 3:** Train the remaining shortlist items that are still missing (ST-GCN-lite, small Transformer) if the TCN/GRU ensemble has not yet met the acceptance criteria. Mine hard negatives from the replay set.  
- **Week 4:** Final ensemble, frozen threshold, full session-replay acceptance test. If acceptance is met, prepare production swap (keep legacy AND as fallback). If not, diagnose residual errors (label noise vs pose gap vs missing background) and feed into the next campaign.

---

## 4. Legacy diagnosis

Why legacy training produced “okay” clip-level validation but mediocre production behavior:

- **Temporal contract mismatch (H4).** Variable-length clips were warped by `temporal_resize` to a fixed seq_len. Velocity was computed on the original timeline, then positions were warped, destroying physical consistency. Production scores fixed 8 s chunks; the model therefore learned length and padding artifacts that do not exist at inference.  
- **Checkpoint bug (H1).** “Best” weights were often the last epoch, not the true best validation checkpoint.  
- **Split leakage (H2).** Per-clip stratified splits allowed same-day / same-session leakage; validation was optimistically biased.  
- **Threshold / ensemble hunting on val (H3).** Operating points were chosen on a validation set whose class prevalence (many robbery clips) does not match production (~1000 sessions/day with rare true events). The resulting threshold is far too aggressive for the true FP budget.  
- **Wrong selection metric.** Accuracy / clip F1 on a balanced-ish val set does not optimize the business objective (event recall at low FP/day).  
- **RNN pooling bug.** BiLSTM/BiGRU models that took only the last time step after a bidirectional pass effectively ignored most of the backward context.  
- **ST-GCN graph error.** Joint connectivity followed array index instead of anatomy, destroying the spatial inductive bias.  
- **Missing background distribution.** Models never saw the long stretches of ordinary customer behavior that dominate production; they therefore fire on any motion that statistically resembles the (noisy) positive class.  
- **Label noise.** Category-6 contains phone-in-pocket and other business false thefts that are indistinguishable from true concealment given only upper-body skeleton.  

Campaign2 has already closed the worst of the temporal, checkpoint, and normalization issues. The residual production gap is now dominated by the items listed in the executive summary (stride/aggregation, short fragments, background mining, pose parity, and label semantics).

---

## 5. Ranked roadmap (next 4–8 weeks)

Ordered by expected reduction in false positives per session/day at acceptable recall, with engineering cost (S/M/L).

| Rank | Action | Expected FP impact | Cost | Notes |
|------|--------|--------------------|------|-------|
| 1 | Finish Campaign2 matrix + multi-length eval + session-replay pilot | High (closes metric gap) | S–M | Week 1; already mostly implemented |
| 2 | Background-window mining from real sessions + hard-negative mining | High | M | Supplies the missing negative distribution |
| 3 | Freeze threshold on T, evaluate exclusively on session/event units | High (removes optimistic bias) | S | Protocol change, not code-heavy |
| 4 | Short-fragment mask policy + production-stride offline eval | Medium–High | S | Prevents edge-case FP |
| 5 | Label audit / phone-in-pocket handling (re-label or auxiliary head) | Medium–High | M | Reduces irreducible FP floor |
| 6 | Pose train/prod parity golden tests | Medium | M | Parallel engineering track |
| 7 | TCN + GRU 3-seed ensemble + optional ST-GCN-lite | Medium (architecture is secondary) | S–M | Only after protocol is solid |
| 8 | Post-processing cooldown / deduplication in serving stack | Medium | S | Cheap production win |

Do not reorder architecture exploration above items 1–4; the data and protocol levers have higher expected return.

---

## 6. Acceptance criteria

A new campaign is considered successful when **all** of the following hold on a held-out set of real production-like sessions (minimum 500–1000 sessions containing ≥30–50 independent labeled theft events):

1. **Event recall ≥ 50 %** at a false-session rate **≤ 10 per 1000 sessions**.  
2. The same model + threshold, when evaluated at the aspirational operating point, shows a credible path toward **≤ 1 FP/day** (i.e., the precision-recall curve does not collapse).  
3. Short fragments (<8 s) do not produce a disproportionate share of false positives (mask policy validated).  
4. Threshold and ensemble rule were calibrated on a split that is disjoint from the final test sessions.  
5. Clip-level metrics are reported only as secondary diagnostics; the go/no-go decision is made on the session/event numbers above.

If the campaign fails these criteria, the residual error analysis (label noise vs pose gap vs missing background vs architecture) must be written before any further architecture work is started.

---

## 7. Risks & open assumptions

- **Label noise floor.** If a non-negligible fraction of category-6 clips are business false thefts (phone-in-pocket, own-item), skeleton-only Stage-1 has an irreducible FP rate that no architecture or window length can remove. Mitigation: label audit or auxiliary head.  
- **Pose domain gap.** If the offline YOLO-pose distribution differs systematically from Triton (keypoint jitter, missing joints, coordinate bias), all offline gains will partially evaporate in production. Assumption: the parallel pose-parity track will close this before the next major model swap.  
- **Effective sample size.** The ~2592 positive clips may correspond to far fewer independent events (same person, same day, same gesture style). Over-fitting to that limited diversity remains the dominant statistical risk; hence the strict parameter budget.  
- **Session fragmentation.** Tracker ID switches can split a single theft event across multiple sessions or merge unrelated people; this affects both training labels and evaluation counts.  
- **Stride / aggregation mismatch.** If the offline evaluation continues to use stride 1 s + clip-max while production uses stride 4 s + session logic, any “improvement” may be illusory.  
- **Short-fragment policy.** An incorrect mask or padding rule for the first/last windows of a session can generate a steady stream of false positives that never appear in offline clip evaluation.  
- **Assumption of stationarity.** The advice assumes that the single-store, 9-camera distribution remains representative; a sudden change in customer behavior, camera placement, or lighting can invalidate calibrated thresholds.  
- **Compute assumption.** The recommendation assumes that ~8 parallel small-model jobs remain feasible on the 5090; if VRAM pressure forces sequential training, the 3-seed × shortlist matrix still fits inside a few days.

---

**End of expert response.**  
All Q1–Q18 answered, concrete proposal filled, legacy diagnosis, ranked roadmap, acceptance criteria, and risks provided. Owner can execute from Week 1 immediately.