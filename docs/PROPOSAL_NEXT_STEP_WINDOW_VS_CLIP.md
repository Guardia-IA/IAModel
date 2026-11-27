# Robbery detection: next step after the window-vs-clip audit

**Date:** 2026-11-27
**Audience:** the implementation agent (no prior chat). This file is the brief.
**Status:** proposal. Phase 0 is read-only (new evaluation scripts, no training code touched). Phase 1 starts only after Phase 0 reports and the owner accepts.
**Inputs reviewed:** `WINDOW_VS_CLIP_CAMPAIGN_AUDIT.md` and three independent proposals, referred to here as **Proposal A** (file `2b3c6d61`), **Proposal B** (`4bd6f40d`) and **Proposal C** (`1746e643`). Section 9 says what was kept and what was dropped from each.

Tags used below: **[measured]** = number taken from the audit. **[derived]** = arithmetic on audit numbers (re-check it). **[hypothesis]** = not proven; the measurement that decides it is named next to it.

---

## 1. Executive summary

**Decision**

1. **Stop** training 8 s windows on the same curated action clips. **Do not adopt** any of the three external proposals as written (section 9).
2. The audit's central claim, "window training is worse than clip training", is **not yet established**. It compares clip-resampled scores of one model family with native-window scores of another, on different splits, with different thresholds and different FP definitions. The deployed full-clip checkpoints have **never been measured on what production actually feeds them**: raw 96-frame buffers at real speed. Phase 0 (about 2–3 days, no training, mostly data already on disk) closes that gap.
3. After Phase 0, run **one** training experiment (Phase 1): native-rate 96-frame buffers produced by the production chunker, real store negatives plus mined false alarms, symmetric quality gating, labels that never turn fragments of a robbery into negatives, checkpoint selection by **event-level recall at a fixed false-alarm budget**, and a diverse model pair.
4. The target (3–5 false alarms per 1 000 interactions) is a **per-interaction** bar. It translates into roughly **0.03–0.17% false positives per 8 s decision** (section 3). That is 10–100× below every curated number in the audit. Pose-only may not reach it. Phase 2 defines the extra signal and the kill criterion so that conclusion is reached with data, not argument.

**Three facts the agent must keep in mind**

- **Aggregation.** An interaction contains several decisions. The alarm rate per interaction is about `1 − (1 − p)^k`. With the deployed pair's curated rate p = 2.19% and k_eff = 5 this gives ≈ 10.5%, which already matches the owner's ~10% store observation without any domain shift [derived]. Windows are correlated, so the true k_eff is lower, but the point stands: **curated per-clip FP and store per-interaction FP are different quantities**.
- **Train/serve mismatch.** Full-clip training time-resamples the whole labeled clip to 96 frames. A 4 s theft clip is stretched about 2×; a 20 s negative is compressed about 2.5×. Clip duration leaks into apparent motion speed. Production feeds raw 8 s buffers at real speed. [hypothesis H1, decided by P0-2]
- **Prior shift.** Models were trained at a positive prior of about 20–35% (full-clip test is 19.4% positive; window sampler used `positive_share = 0.35`). The store prior is far below 1%. A 0.50 cut is therefore a very loose operating point in deployment, and thresholds picked on curated test data do not transfer.

---

## 2. Answers to the six questions of audit §10

**Q1. Is §6 fair?** I agree with the arithmetic and disagree with the interpretation.
- Agreed: no reported window checkpoint reaches ≥66% recall at low FP, and the 0.80 threshold collapses recall. Also agreed: the audit's "decision unit" excuse (§8.2) is weak. The window test has about 1.08 negative windows per negative clip [derived: 2 060 negative windows / ≈1 911 negative clips, assuming a positive-clip rate near 19%; verify], so multiplicity does not explain a 4.7× gap.
- Disagreed: the unfair rows are all of §6.1–6.2, because they put **clip-resampled** scores next to **native-window** scores. They differ in split, unit, threshold protocol (the full-clip "best" rows were swept **on test**, the window rows were fixed) and in what counts as a false positive (window negatives include windows **inside robbery files** next to the gesture; clip negatives do not).
- What would change my mind: if the full-clip checkpoints, scored on native 96-frame windows from clips held out by **both** runs, show ≤3% FP at ≥80% recall, then §6 is fair and the window recipe is defective. If they show ≥8% FP, §6 is an artifact of the comparison and the real finding is that **all** current models are weak at the production unit. P0-2 produces exactly this table.

**Q2. Why did matching the 8 s buffer not reduce FP? Two separate questions.**

*(a) Why the window test looks worse.* Candidates, ranked by how cheaply they can be tested:
1. Full-clip models were never measured on native windows (H1). Test: P0-2.
2. The FP rate includes negative windows cut from robbery files next to the gesture. An alarm there is arguably a correct alarm at interaction level. Test: P0-1 (FP split by negative source and by distance to the core).
3. Label ambiguity at the edges (`pos_partial`, and in-file negatives close to the core) produces low-confidence scores. This fits the recall collapse from 0.50 to 0.80. Test: plot score against core-overlap fraction.
4. Prior shift: `positive_share = 0.35` plus an argmax cut. Test: P0-3 (threshold sweep on store-like negatives).
5. Quality gate asymmetry (25 320 windows dropped, 2 586 positives protected). **This cannot raise FP on this test**: the test set went through the same gate, so glitchy negatives were removed from it too. If anything it makes the test optimistic. It matters for production transfer, and for recall if production `sanity_pose` drops positives that training kept (P0-6). Fix it regardless, but do not present it as the cause of the FP gap (Proposal C does).
6. The 4 s minimum clip rule. If both runs share the same pool, at most 23 737 − (18 864 + 2 354 + 2 371) = **148 clips (0.6%)** were removed [derived]. Count how many are positive (P0-7); it is probably minor.

*(b) Why the store still sees ~100/1 000 with the deployed full-clip pair.* Layers, not one cause (section 3): aggregation across windows, train/serve framing, missing store behaviour in the training negatives, and a loose operating point. The documented case in `MOTION_FP_INVESTIGATION_BRIEF.md` (a normal shopping session raising 7 alarms in about 55 s) has the shape of a **persistent benign pattern firing repeatedly**, not independent noise. That favours "specific behaviours absent from training" over "bad luck per window", and it makes hard-negative mining the right lever.

**Q3. Next experiment.** One path: Phase 0 (measure at the production unit, no training) then one Phase 1 training recipe (section 6). Not a menu.

**Q4. Can thresholds or ensembles on existing checkpoints reach §2?**
- Window checkpoints: no (audit §5.3–5.4).
- Full-clip checkpoints: not on their own. Curated points such as 49+53 AND @ 0.92 (recall 67.2%, FP 0.47%) or 11+49 AND @ 0.86 (66.4%, 0.31%) were chosen on test, on resampled clips. Whether they keep ≥66% recall on raw 8 s buffers is exactly H1. They are worth measuring as an **interim relief** (section 8), but 0.31–0.47% per decision still maps to roughly 10–40 per 1 000 interactions at k_eff 3–9 [derived, table in section 3], not 3–5.

**Q5. Walking, standing, hand-to-body.** Policy:
- **No hard global rules.** "Walking ⇒ not a theft" is wrong (many thefts start walking). "Must show hand-to-body" is wrong (open-cart thefts). Both are rejected.
- Walking and idling are the bulk of store time. The model must learn them from **real store negatives**, not from a handwritten veto.
- Hand-to-body (including the existing `pocket_filter` / HAR_v2 work, if it is still in use) is an **evidence feature**, never a gate.
- A rule-based veto is allowed only if it passes the ablation gate in section 7.
- If the residual false alarms are dominated by a shopper's **own object going to a pocket or bag**, pose cannot separate them from concealment. That needs an extra signal (section 7). Say so with data after the cause audit, not before.

**Q6. What is needed from the owner?** Section 10. Important: **P0-1, P0-2, P0-3 (curated part), P0-4, P0-6 and P0-7 need only data already on disk.** Only the store-log parts (P0-5, Phase 1 negatives) need owner data.

---

## 3. Why the store gap exists: layers

| # | Layer | Mechanism | Evidence in the audit | What decides it |
|---|---|---|---|---|
| L1 | **Aggregation** | Alarm per interaction ≈ `1 − (1 − p)^k`. | 2.19% × k_eff 5 ≈ 10.5% [derived]. | Store logs: decisions per interaction, alarmed interactions, alarms per alarmed interaction (P0-5). |
| L2 | **Train/serve framing** | Full-clip models trained on whole-clip time-resampled vectors; production gives real-speed buffers where a 3 s theft fills 37.5% of the window and can sit anywhere in it. | Production feeds 8 s buffers (audit §4.1). C-checkpoints were never scored on native windows. | P0-2 paired matrix (H1). |
| L3 | **Domain** | Store behaviour (browsing, shelf reach, own phone/wallet, bag adjust, occlusion, second person, chunk seams, track-ID switches) is absent from curated negatives. | Audit §8.4–8.5; deployed pair is ~2% FP on curated but ~10% in store. | FP-cause audit of real alarms (P0-5). |
| L4 | **Operating point** | Mean of two correlated `tcn_gru` models at 0.50, under a 20–35% training prior. | Near the 66–68% recall floor the curated grid has diverse pairs at 0.31–0.47% FP (11+49 AND @ 0.86, 49+53 AND @ 0.92) versus 2.19% for the deployed 48+49 at 82% recall: 5–7× lower FP for about 15 points less recall. Same-family 48+49 gains little (AND @ 0.50: 1.36%). | Threshold sweep on store validation negatives. |
| L5 | **Window-run recipe issues** | Edge-label ambiguity, gate asymmetry, sampler prior, objective (`macro_f1`). | Audit §5.1, §5.3. | P0-1, P0-6, score vs overlap plot. |

**Required per-decision FP for the store target** [derived, exact]:

| k_eff (decisions per interaction) | 1 / 1 000 | 3 / 1 000 | 5 / 1 000 |
|---|---:|---:|---:|
| 1 | 0.10% | 0.30% | 0.50% |
| 3 | 0.033% | 0.10% | 0.167% |
| 5 | 0.020% | 0.06% | 0.10% |
| 9 | 0.011% | 0.033% | 0.056% |

**Store rate implied by a given per-decision FP** [derived]: p = 2.19% gives 22 / 64 / 105 / 181 per 1 000 for k = 1 / 3 / 5 / 9. p = 0.47% gives 4.7 / 14 / 23 / 42. p = 0.31% gives 3.1 / 9.3 / 15 / 28.

The first number the agent must produce is the real **k_eff** from store logs. Everything below depends on it.

---

## 4. Evaluation protocol (applies to every phase)

**Units**
- **Positive unit: robbery event.** Detected if **any single** production decision window that contains at least 80% of the annotated core (`core_index.json` for curated clips) raises an alarm. This is an OR across windows, **not** an AND. A 3 s theft is therefore detected from one window.
- **Negative unit: interaction.** A false alarm is an interaction with at least one alarm. Report also alarms per 1 000 decisions and per 1 000 interaction-hours as diagnostics.
- Window-level recall (as in audit §5) stays only as a diagnostic. It counts `pos_partial` windows and several windows per event, so it understates product recall.

**Splits.** Frozen manifests, written to **new paths** (never overwrite the existing window manifest or `core_index.json`). Store data split by store / camera / day. Thresholds on validation only. Test touched once.

**Sample sizes** (exact binomial 95% CIs, computed):

| Quantity | Support | 95% CI |
|---|---|---|
| Event recall observed 68% | N = 30 | 47–83% |
| | N = 50 | 53–81% |
| | N = 100 | 58–77% |
| | N = 460 (curated held-out) | 64–72% |
| FP per 1 000 interactions | 2 000 interactions, 8 FP | 1.7–7.9 |
| | 2 000 interactions, 0 FP | 0–1.8 |
| | 5 000 interactions, 15 FP | 1.7–4.9 |
| | 5 000 interactions, 20 FP | 2.4–6.2 |

To show a **one-sided 95% upper bound ≤ 5 / 1 000**: about 2 000 interactions if the true rate is 2/1 000, about 3 500 if 3/1 000, about 14 000 if 4/1 000. So 2 000 interactions is a **development** holdout. Final acceptance is **shadow mode in stores** (section 8).

**Reporting rules.** Always give N for positives and negatives. Always give recall **by folder (6 vs 14)** and by theft type if annotated (hand-to-body, open-cart, walking start). Always say whether "per 1 000" means decisions, interactions or clips.

**Labeling cost.** Negatives dominate the data. Do **not** annotate whole sessions. Review **alarms and near-alarms** only (score ≥ 0.2 from the deployed pair). Unreviewed non-alarm windows count as negative. Exclude any interaction listed in the incident log as a real robbery.

---

## 5. Phase 0: measurements, zero training

All scripts are **new files** in a new directory (for example `.../window_campaign/phase0/`). Do not edit training code. Do not score window checkpoints with `score_pose_ensemble.py`. Full-clip checkpoints are scored with a **new** read-only scorer that reuses the campaign's own preprocessing and skips only the whole-clip resample.

**P0-1. Decompose the window-test FP** (existing window checkpoints, existing test loader).
Split negative windows into (a) windows from negative-folder clips and (b) negative windows inside robbery files, the latter bucketed by seconds from the core. Report FP at argmax and at 0.80 for each bucket and each model. Expected use: if (a) alone is far below 11.9% for model 49, part of the "window failure" is a labeling-definition artifact.

**P0-2. Paired 2×2 matrix (decides H1).**
- Held-out set = clips in the validation or test part of **both** manifests (campaign and window run), matched by clip uid. About 4% of clips, roughly 950 clips and ≈180 positives if splits are independent. State the real numbers.
- Inputs: **U1** = whole clip resampled to 96 frames (campaign style). **U2** = native 96-frame windows with the same window index as the window run (stride 48), no resample.
- Required cells: C@U1, C@U2, W@U2. Optional: W@U1. C = full-clip checkpoints 48, 49, 53, 11. W = window checkpoints of the same ids.
- Report per model and per pair (mean and AND): FP and event recall at 0.50, plus a threshold sweep, plus pAUC for FPR ≤ 1%.
- Pre-registered reading:
  - C@U2 FP ≥ 2× C@U1 FP at the same recall ⇒ **H1 confirmed**. The deployed models operate off their training distribution in production; native-rate retraining is justified; any interim threshold must be set on U2/replay, not U1.
  - C@U2 ≈ C@U1 and both ≪ W@U2 ⇒ H1 rejected, the window recipe really is defective; Phase 1 must fix labels/gate/sampler **and** be compared against a "full-clip recipe plus store negatives" arm.
  - C@U2 ≈ W@U2 ⇒ the audit's finding is a unit artifact; Phase 1 proceeds from the native-rate design.

**P0-3. Re-aggregate at product level.** Recompute everything from P0-2 at event level (OR over windows) and, for the store logs when available, at interaction level. Include the existing window checkpoints at 0.50, 0.80 and a sweep. Report whether any point reaches event recall ≥ 66% with lower FP than before. This corrects the audit's window-level recall, which is stricter than the product definition.

**P0-4. Production-replay harness.** Build pseudo-streams: each held-out curated positive clip embedded in continuous negative context (curated negatives now, store negatives later), passed through the **actual production chunker code** (import it, do not reimplement) at ≥ 8 random phase offsets, including cross-chunk `tail(N) + head(N+1)` seams. Score with the deployed 48+49. This gives event recall at the production unit with N ≈ 460 and **no new recordings**, and it becomes the regression test for every later change.

**P0-5. Store-log analysis** (needs owner data, section 10). From logged sessions run through the deployed pair: decisions per interaction (distribution, k_eff), interactions with ≥ 1 alarm, alarms per alarmed interaction, score histogram. Then a **cause audit** of ≥ 150 alarms with this taxonomy: ordinary gait / idle; shelf or cart reach; own phone / wallet to pocket; bag or clothing adjust; occlusion or second person; track-ID switch or chunk seam; pose glitch; true theft; ambiguous. Use "ambiguous" freely; do not force labels. Seed case: the session in `MOTION_FP_INVESTIGATION_BRIEF.md`.

**P0-6. Gate audit.**
(i) Fraction of curated positive events that survive **production** `sanity_pose` plus track continuity. If it is well below 100%, that is a hard recall ceiling that no model change can fix.
(ii) Fraction of positive vs negative windows failing each training-gate rule **before** the positive protection (centroid jump, teleport, teleport-after-mask, visibility). If the training-gate pass rates differ strongly by class, the shortcut risk is real; score the models on gate-failing negatives.

**P0-7. Short clips.** Count positive and negative clips under 4 s in the full pool and how many were removed from the window manifest.

**Deliverable.** `phase0_report.md` with tables for P0-1 to P0-7, the decision-tree outcome, and the measured k_eff. If store data is not available yet, deliver the curated parts first and say the rest is pending.

**Decision tree after Phase 0**
- Event recall of deployed 48+49 on replay (P0-4) **< 66%**: the recall floor is already violated at the production unit. Priority becomes recall, and raising thresholds is not an option.
- H1 confirmed: Phase 1 as written. H1 rejected: Phase 1 plus a "clip-recipe + store negatives" control arm.
- Top 3 causes ≥ 60% of audited alarms: targeted mining will work. Causes spread evenly: broad domain data is needed.
- "Own object to pocket/bag" ≥ 40% of audited alarms: plan Phase 2 in parallel from the start.

---

## 6. Phase 1: the training recipe

Start only after the Phase 0 report and owner acceptance. Use new run ids and new checkpoint names. **Never overwrite** the deployed `modelo_48` / `modelo_49`.

### 6.1 Input contract
- 96 frames at 12 Hz, native speed, produced by the **production chunker code** (same `sanity_pose`, same `single_user_only`, same track handling). No whole-clip time resampling in this arm.
- Positives: windows whose core is fully inside, with the core placed at **random offsets** inside the window (start, middle, end), not only at fixed stride 48. In production a 3 s theft appears at any position.
- Negatives: random windows from continuous store streams, including cross-chunk seams and track switches.
- Keep clips under 4 s. Place them inside real benign context (section 6.2, S5) instead of dropping them.

**Containment check.** A theft of duration D is fully inside at least one window if `D ≤ W − S` (W = 8 s window, S = production stride). With S = 4 s this holds for D ≤ 4 s. **Confirm the real production stride.** If S is larger, short thefts can be split across windows and the "one window is enough" assumption fails.

### 6.2 Data sources and weights (starting values, tune on validation)

| Source | Content | Share of negatives / positives |
|---|---|---|
| S1 | Real store benign interactions through the production chunker (train pool) | ~45% of negatives |
| S2 | **Mined hard negatives**: windows the deployed pair scored ≥ 0.2, reviewed and confirmed non-robbery | ~25% of negatives, never above 30% in a batch |
| S3 | Existing curated negative folders | ~30% of negatives |
| S4 | Curated positives (folders 6, 14), plus real/staged production positives **only if there are enough left for evaluation** | positives |
| S5 | Splice augmentation (optional): paste the core segment of a curated positive into real walking/standing background, aligning root position and heading, cross-fading about 0.5 s with velocity continuity | positives |

**Seam rule for S5.** Apply the same pasting procedure to **negative** segments so seams are class-neutral. Train a tiny probe to detect seams; if its AUC against class exceeds 0.55, drop S5. A model that learns "seam ⇒ robbery" would repeat the gate-asymmetry shortcut.

**Mining loop** (train pool only): round 0 scores the pool with the deployed pair; train; score the pool again; add new top false alarms (score ≥ 0.3); retrain. Two or three rounds. Validation and test alarms may be reviewed to label the evaluation, but are **never** fed back into training.

### 6.3 Label rule
- `pos`: window contains ≥ 80% of the core.
- `ignore`: 0 < core overlap < 80%, and **every other window from a robbery file** (standing before the grab, tucking after it). Never label fragments of a robbery as `neg`.
- `neg`: all windows from non-robbery sources (S1, S2, S3).
- **Symmetric gating:** apply the production `sanity_pose` function to every window of every class. No positive protection. Report the number of positives lost (P0-6 gives the baseline).
- Report recall per folder (6 vs 14) and per theft type from day one.

### 6.4 Models
- Retrain **49** (`tcn_gru`, 3 TCN layers), **53** (`conformer_lite`), **11** (`dilated_tcn`). Retrain 48 only if a same-family comparison is needed. 48 and 49 stay distinct graphs.
- Three seeds each. Operating points near 0.1% FPR are noisy; report seed variance and select by validation, not by test.
- The pair that matters is **diverse** (49+53 or 11+49). The full-clip grid and the window grid both show same-family pairs (48+49) barely reduce errors.

### 6.5 Objective and sampling
- Draw negatives from the full pool, not a 3 200-sample cap per epoch. Use as many samples per epoch as time allows; benchmark one epoch first.
- Positive share per batch 15–25%. Do not trust 0.50 afterwards (section 6.7).
- Hard-negative emphasis (focal loss with γ between 1 and 2, or loss reweighting on S2). Watch calibration if focal is used.
- Augmentation: keep the existing set (mirror, noise, scaling, rotation); add **speed perturbation about ±15%** and joint dropout for occlusion. This replaces the accidental speed variation that whole-clip resampling used to give.
- No `macro_f1` or accuracy for selection.

### 6.6 Checkpoint selection
- Metric: **event-level recall at a fixed validation false-alarm budget** (for example per-decision FP = the p from section 3 at the measured k_eff), or pAUC for FPR ≤ 1% if the validation negatives are too few to resolve 0.1%.
- Validation needs at least about 20 000 negative decisions and at least 1 000 interactions. Below that, a 0.1% FP point is only a handful of events and selection is noise.
- A checkpoint with zero recall on either positive folder cannot win.

### 6.7 Calibration, threshold, decision rule
- Temperature-scale on validation. Choose thresholds on **validation store negatives**, never on test.
- Target per-decision FP = section 3 value for the measured k_eff; target event recall **≥ 68%** (2-point margin over the 66% floor).
- Rule candidates, chosen on validation: mean of the diverse pair, AND of the diverse pair, 2-of-3 vote. Each is a **single-window** decision. A short alarm-merging cooldown is fine; requiring two consecutive hits is not.
- A 3 s theft fires from one window (section 6.1 containment check).

### 6.8 Pre-registered outcomes
- Success: on the held-out store test, event recall ≥ 66% (N ≥ 460 replay positives, CI reported) **and** ≤ 5 false alarms per 1 000 interactions.
- Gate to Phase 2: interaction FP between 5 and 15 per 1 000, or > 40% of residual alarms are own-object-to-pocket/bag.
- Failure: event recall < 66% at the FP budget, or interaction FP > 15 per 1 000 (then audit data and pipeline before blaming pose).
- I do **not** put a predicted number here. No evidence in the audit supports a forecast (Proposal C's 1.5–4 per 1 000 windows is unsupported).

### 6.9 Cost (estimates, confirm with a one-epoch timing run)
- Pipeline work (production-chunker loader, symmetric gate, mining loop, event/interaction evaluator): 1–2 engineer-days.
- Training: roughly 8–20 GPU-hours in total (3 architectures × 3 seeds × 2–3 mining rounds). The earlier 6-model baseline took 1.19 h with only 3 200 samples per epoch, so a full-negative epoch is much heavier.

---

## 7. Phase 2: only if the Phase 1 gate says pose is not enough

Choose by the cause audit, not by preference.

**Residual alarms are gait / idle / seams / glitches.** More mined negatives, tighter production `sanity_pose`, seam handling. No new signal.

**Residual alarms are the shopper's own object to pocket or bag, or hand-to-body look-alikes.** Pose cannot separate these from concealment. Options, cheapest first:
1. **Signal A, shelf / product-zone context.** Did the tracked person touch a shelf or merchandise polygon in the last 10–15 s? Use as a **feature** in a small learned second stage (inputs: pair scores, peak position, hand-to-body evidence, shelf-visit flag). It does not need two windows, so a 3 s theft still fires from one. Validate on staged thefts that begin outside the shelf zone (counter, hand-off) before using it as a veto.
2. **Signal B, triggered RGB verifier.** Run a small classifier on the pose-flagged windows only (about 1–2% of traffic). Compute is negligible. It needs owner approval for privacy and legal reasons and buffered frames of alarmed windows.
3. **Alarm tiers** (product lever, not a model lever): high-confidence alarms immediate, medium-confidence logged for review. Use only if pose-only plateaus at 5–15 per 1 000 and the owner accepts it.

**Ablation gate for any rule-based veto.** It is allowed only if on held-out stores it (i) removes ≥ 30% of the residual false alarms, (ii) costs ≤ 2 points of event recall overall **and** ≤ 3 points in each of: open-cart, walking-start, hand-to-body thefts, and (iii) is reported next to the same data without the veto. Handwritten gait-symmetry or "peak sub-window" rules (Proposal C) may be tried as **features** for the learned stage, not as first-class gates.

The 1 / 1 000 wish is a stretch goal. I would not expect it without Signal B.

---

## 8. Acceptance and rollout

1. **Interim relief (optional, days, no training).** If P0-2/P0-4 show that a full-clip diverse pair (49+53 AND or 11+49 AND) keeps event recall ≥ 66% with CI lower bound ≥ 60% on replay (N ≈ 460) **and** lowers the alarm count on store logs, run it in shadow mode first. Both models must run in production for AND. Do not trust the curated 0.31–0.47%.
2. **Offline gate.** Phase 1 candidate passes section 6.8 on the frozen store test.
3. **Shadow mode in stores**, 2–4 weeks, new pair running next to the deployed one without raising alarms. Count alarms per 1 000 interactions with the same interaction definition as the owner's 100/1 000. Review only the alarms. Accept when the observed rate and its 95% upper bound meet the bar agreed in advance (section 4 table). Check recall on any real or staged events in the same period.
4. **Rollout** behind a flag with rollback. No architecture or threshold change in production before this.

---

## 9. Review of the three external proposals

| | Keep | Drop or correct |
|---|---|---|
| **A** (`2b3c6d61`): mine real-store false alarms, add a kinematic veto, no retraining first | Hard negatives come from the deployed model's own store alarms. "Do not require two windows" and a short-peak escape hatch. Stop treating curated tables as the optimisation target. | Veto-first is the main lever, but its features are undefined. Claims that mining plus veto reaches 3–5/1 000 have no basis. N ≥ 30 positives gives a recall CI of 47–83%. No aggregation analysis. "Stop using the curated corpus" is too strong: it holds almost all the positives. |
| **B** (`4bd6f40d`): production replay, session-level frozen splits, evidence filter, conditional retraining | Best evaluation discipline: frozen split by store/camera/date, tolerance-based event matching, `ignore` for ambiguous cases, CIs and N, cause categories, "curated FP is not store FP", honest "new signal may be needed". | No training recipe, threshold or prior treatment, no costs. Annotating complete sessions is far more work than reviewing alarms. It misses that replaying **curated** positives through the production chunker already gives N ≈ 460 events for free. |
| **C** (`1746e643`): Gate 0 baseline, retrain 49/53/11 on Stage-2 data, AND pair, peak veto, RGB/ROI cascade | The multiple-testing arithmetic across windows. Gate 0 on existing checkpoints. Symmetric gating. Never label robbery fragments as negative. Real store negatives with seams. Diverse pair. Recall-at-FPR checkpoint selection. Calibration. The cascade with an explicit kill criterion. The `D ≤ W − S` containment argument. The idea that whole-clip resampling creates a speed cue (this became H1). | The "four defects" are asserted as the dominant cause without evidence. The gate-asymmetry argument runs the wrong way for the test FP (section 2, Q2). At most 148 clips are affected by the 4 s rule. "3.3 passes over negatives" is not severe under-training by itself. Predicted outcomes (1.5–4 per 1 000 windows, 8–15 per 1 000 store) are not supported. It assumes full-clip thresholds fail on raw buffers instead of testing it. The handwritten pendulum-gait veto is unvalidated. It retrains three architectures before knowing whether H1 holds. |

---

## 10. What is needed from the owner

**Not needed to start (data on disk):** P0-1, P0-2, P0-3 (curated), P0-4 (with curated negative context), P0-6, P0-7. Paths: window checkpoints `artifacts/models/win_bin_full_baseline/modelo_{48,49,53,57,13,11}.pt`; full-clip checkpoints under `.../campaign_20261103_230735/models/bin_full/modelo_*.pt`; both manifests; `core_index.json`.

**Needed for the store part, in priority order:**
1. **Production code or exact spec** for the chunker (chunk length, stride, concat across chunks, track-ID switch handling), `sanity_pose`, alarm aggregation and any cooldown.
2. **Store pose streams** (the same NPY chunks the pipeline already exports, if archived) for benign interactions, plus the alarm log with scores and timestamps from the deployed pair. Target: ≥ 5 000 interactions for the training/mining pool and ≥ 2 000 disjoint interactions (different days, ideally an unseen store/camera) for the frozen holdout.
3. **Definition of "interaction"** and the denominator behind the owner's ~100 / 1 000: per `track_id` visit, per session, or per time block, and how many interactions per day per store.
4. **Incident list** of real robberies in the logs, to exclude them from negatives and to use as positives.
5. **Reviewer time** for about 300 alarms and near-alarms (labeling takes seconds each; budget a few hours).
6. **Acceptance decision:** event recall ≥ 66% and 3–5 per 1 000 interactions, with 1 / 1 000 as stretch; shadow mode as the final test.
7. **Only if Phase 2 triggers:** shelf-zone polygons per camera, buffered RGB for alarmed windows, and approval for using them.

If store pose streams are not archived, the first deliverable is a **capture plan**: log every window with score ≥ 0.2 plus full-session NPY for 1–2 weeks of normal operation.

---

## 11. Decision template (audit §11)

```text
Title:
  Measure at the production unit, then retrain on native-rate windows with real store negatives and mined false alarms.

One-sentence bet:
  The store false-alarm rate is a product of aggregation across windows, a train/serve framing mismatch and missing store behaviour in the negatives, so measuring the deployed models at the production unit and retraining on production-faithful windows with mined store negatives will beat any further slicing of the curated clips.

Evaluation that decides success:
  - Positive unit: robbery event, detected if any single 8 s decision window containing >= 80% of the core alarms.
  - Negative unit: store interaction (>= 1 alarm = false alarm); per-1 000-decision rate as diagnostic.
  - Recall floor: >= 66% on N >= 460 production-chunker replay events (curated held-out), plus all real/staged store events available (report N, CI).
  - FP bar: <= 5 per 1 000 interactions on a frozen holdout of >= 2 000 store interactions (development); final: shadow mode, upper 95% bound <= 5 / 1 000.
  - Separate store bar: same as above; the holdout IS store data, curated tables are regression only.
  - Failure looks like: event recall < 66% at the FP budget; interaction FP > 15 / 1 000; or 5-15 / 1 000 with > 40% own-object-to-pocket/bag residuals (then Phase 2).

Data you will add or stop using:
  Add: store pose streams (train pool >= 5 000 interactions, holdout >= 2 000), mined hard negatives, production-chunker replay of curated positives.
  Stop: window-on-curated-clips as the success benchmark; the asymmetric training gate; negative labels on fragments of robbery clips.

Label rule:
  pos = window with >= 80% of the core; ignore = partial overlap and every other window of a robbery file; neg = all non-robbery sources; same production sanity_pose on every window.

Models:
  Phase 0 uses existing checkpoints (full-clip and window), read-only.
  Phase 1 retrains 49, 53, 11 (3 seeds) under new names. 48 and 49 stay distinct. Deployed checkpoints are never overwritten.

Decision rule:
  Single-window mean or AND of a diverse pair (or 2-of-3), thresholds from validation store negatives. A 3 s theft fires from one window (needs production stride <= 4 s; to confirm). Two consecutive hits never required.

What you will not change:
  Frozen window manifest and core_index.json; 96-frame / 12 Hz input contract; recall floor; the ban on two-window confirmation as the main lever; no scoring of window checkpoints with score_pose_ensemble.py; no thresholds tuned on test.

Cost:
  Phase 0: 2-3 engineer-days, < 2 GPU-h (inference). Phase 1: 1-2 engineer-days plus roughly 8-20 GPU-h. No new recordings unless the store logs are not archived.

First result you will show the owner, and by when:
  phase0_report.md: curated-only tables (P0-1, P0-2, P0-3, P0-4, P0-6, P0-7) within 2 working days of acceptance; store tables (P0-5, measured k_eff, cause audit) within 2 working days of receiving the logs.

Risks you are explicitly accepting:
  - H1 may be rejected, in which case the window recipe really was defective and Phase 1 needs the control arm.
  - Pose-only may plateau at 5-15 / 1 000; Phase 2 then needs a new signal and owner approval.
  - Store annotation and log export are owner-side work and gate the store results.
  - Few real or staged store robberies: recall at the store unit is only as good as N allows.
```

---

## 12. Guardrails and execution order

**Do not**
- Train or tune on the test sets. Choose thresholds only on validation.
- Overwrite or regenerate the window manifest, `core_index.json`, or the deployed `modelo_48` / `modelo_49`.
- Score window checkpoints with `score_pose_ensemble.py`.
- Require two consecutive windows to alarm.
- Add a global walking, standing or mandatory hand-to-body rule.
- Mix leakage: store data split by store/camera/day; clips matched by uid across manifests.
- Treat curated-clip FP as store FP, or per-window FP as per-interaction FP. Always state the unit.
- Re-diagnose the old window bugs (all-negative output, sampler, accuracy-based selection) unless a new measurement shows they are back.

**Order**
1. P0-7 and P0-6 (count short clips; gate audit).
2. P0-1 (FP split by negative source).
3. P0-2 and P0-3 (paired matrix, event-level re-aggregation). **Gate: read the decision tree.**
4. P0-4 (production replay of curated positives).
5. P0-5 once store logs arrive (k_eff, cause audit).
6. Write `phase0_report.md`; the owner decides on Phase 1.
7. Phase 1 pipeline, one-epoch timing, mining rounds, seeds, validation selection, one test pass.
8. Shadow mode, then rollout behind a flag.
