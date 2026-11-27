# Full-clip vs 8-second window training — audit and request for a proposal

**Language:** English  
**Date:** 2026-11-27  
**Audience:** An agent with **no** prior chat. This file is the brief. Read it, then answer the questions in §10 with a concrete proposal in the format of §11.  
**Do not** retrain, retune thresholds, or edit training code until the owner accepts a proposal.

Related docs (background, not a substitute for the numbers below):

- `IAModel/experiments/training/window_campaign/WINDOW_CAMPAIGN_AGENT_HANDOFF.md` — how the window trainer was built and which training bugs were already fixed.
- `IAModel/experiments/training/window_campaign/WINDOW_CAMPAIGN_FINAL_PLAN.md` — original plan. Stage 2 (real session chunks and production overlaps) was **not** run.
- `IAModel/experiments/training/campaign/C2_DEPLOY_ENSEMBLE_48_49_BIN_FULL.md` — what is deployed today.

---

## 1. What you are being asked

We trained the same robbery-detection architectures twice:

1. **Full clip.** One labeled action clip is time-resampled to 96 frames and scored once.
2. **8-second windows, as production scores motion.** Each clip is cut into 96-frame slices at 12 Hz (8.0 s) with stride 48 (50% overlap). Each slice is a training and test example.

The window run was supposed to match production and to cut false robbery alarms without giving up robbery detection. **On the held-out tests we have, it did not.** At the same robbery recall, false-positive rates are several times higher than the full-clip campaign. Ensembles that already existed on full clips also beat the new window ensembles.

The owner’s reading, which these numbers support: slicing the same curated clips into 8 seconds did **not** make the detector finer.

**Your job:** propose the next move that can hit the product target in §2. A proposal that only repeats “train 8-second windows on the same action clips” is not useful unless you show why this run was not a fair test of that idea and what you would change.

---

## 2. Product target (hard)

Robbery detection must stay usable **and** store alarms must drop sharply.

| Constraint | Value | Source |
|---|---|---|
| Robbery **recall floor** | **≥ 66%** | Owner, 2026-11-27. A point at 40% recall is not a candidate, even if its false-positive rate looks good. |
| Robbery **F1** | Keep it in the same band if the precision/recall trade allows it. The owner previously accepted F1/recall in the **65–75%** band **only** in exchange for a large false-positive cut. Do not trade below ~66% recall. | Owner, prior turn + this turn. |
| False alarms in a store | Today about **100 false robbery alarms per 1 000 interactions** (~10%). Desired: **3–5 per 1 000** (0.3–0.5%). A stricter wish of **1 per 1 000** (0.001) was also stated. | Owner field observation, not a logged benchmark file. |
| What must not be the main lever | Requiring **two consecutive** 8 s windows to both fire. Most thefts last about **3 seconds** and land in a **single** window. A two-window rule would drop those thefts and miss the recall floor. | Owner. |

“Interaction” here is the owner’s store unit (a customer interaction that can raise an alarm), **not** the curated-clip test unit in §5–§6. The two rates must not be treated as the same measurement. See §8.

Deployed production rule today (`C2_DEPLOY_ENSEMBLE_48_49_BIN_FULL.md`):

```text
robbery_score = (P_robbery(modelo_48) + P_robbery(modelo_49)) / 2
alarm         = robbery_score >= 0.50
```

Those two checkpoints come from the **full-clip** campaign below. Production still **feeds them 8-second buffers**.

---

## 3. Shared task

| Item | Value |
|---|---|
| Task | Binary. Positive = robbery. |
| Positive folders | **6** and **14**. Everything else used as negative is class 0. |
| Pose | `poses_full.npy`, `single_user_only` (`user_0` only). |
| Rate the models were built for | 96 frames. At 12 Hz that is 8.0 s. |
| Architectures compared below | **48** `tcn_gru` (2 TCN layers), **49** `tcn_gru` (3 TCN layers), **53** `conformer_lite`, **57** `ms_tcn`, **13** `res_tcn`, **11** `dilated_tcn`. 48 and 49 are the production pair. They share the architecture name and have different depth/dropout. |
| Data root on the training server | `/media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result` |

---

## 4. Approach A — full clip (`campaign_20261103_230735`)

### 4.1 What one example is

One curated action clip. The whole `poses_full.npy` is **time-resampled** to `seq_len=96` and classified once. A 4 s theft and a 20 s clip become the same 96-frame tensor. The label is the folder: the entire clip is positive if it lives under 6 or 14.

This is **not** how production scores a live session. Production never resamples a whole visit into one vector. It scores successive 8 s buffers.

### 4.2 Where the numbers are

Local copy of the campaign reports:

```text
/home/debian/Documentos/Kanvis/home/artifacts/runs/campaign_20261103_230735/reports/bin_full/
```

| File | What it is |
|---|---|
| `test_eval_summary.json` | Test pool and per-model metric blocks. |
| `test_per_model_best.json` | One chosen threshold per model on **test** (the “best” operating point saved by that eval). |
| `test_ensemble_grid.csv` | Pair grid. Columns include `decision_mode`, `models`, `thresholds`, `tp`, `fp`, `fn`, `tn`, `precision_pct`, `recall_pct`, `f1_pct`, `fp_rate_pct`, `support_pos`, `support_neg`. |
| `test_best_ensemble.json` | The single ensemble the eval script marked best on test. It optimised the low-FP end, **not** the recall floor in §2. |

Cell id: `bin_full`. Split: train/val/test from that campaign’s plan. Test size from `test_eval_summary.json`:

| | |
|---|---|
| Pool | 23 737 clips |
| Test clips | **2 379** (one score each) |
| Test positives | **461** |
| Test negatives | **1 918** |
| Positive rate on this test | 461 / 2 379 ≈ **19.4%** |

Reported `fp_rate_pct` is `fp / 1918`, not `fp / 2379`.

### 4.3 Single models on that test

From `test_per_model_best.json` (`decision_mode=softmax_thr`). Counts: `fn_count` and `fp_count` are in the file; `tp = 461 - fn`, `tn = 1918 - fp`.

| Model | Threshold | Recall | FP rate | F1 | FP count | FN count |
|---|---:|---:|---:|---:|---:|---:|
| 48 `tcn_gru` | 0.50 | 80.5% | **2.92%** | 83.6 | 56 | 90 |
| 49 `tcn_gru` | 0.52 | 80.7% | **2.55%** | 84.4 | 49 | 89 |
| 53 `conformer_lite` | 0.92 | 80.3% | **2.09%** | 85.0 | 40 | 91 |
| 57 `ms_tcn` | 0.56 | 82.4% | **5.11%** | 80.9 | 98 | 81 |
| 11 `dilated_tcn` | 0.50 | 82.2% | **2.97%** | 84.5 | 57 | 82 |
| 13 `res_tcn` | 0.50 | 78.1% | **4.85%** | 78.8 | 93 | 101 |

At ~80% recall the strong full-clip models sit around **2–3%** false positives on these curated negatives (about 40–56 false clips out of 1 918).

### 4.4 Ensembles on that same test

Extracted from `test_ensemble_grid.csv` for the pairs that matter. Support is the same 461 / 1 918.

**Mean of probabilities** (the production style of rule):

| Pair | Thresholds | Recall | FP rate | F1 |
|---|---|---:|---:|---:|
| 48+49 | 0.50 / 0.50 | **81.8%** | **2.19%** | 85.7 |
| 48+49 | 0.80 / 0.80 | 72.2% | 1.15% | 81.6 |
| 48+49 | 0.92 / 0.92 | 67.0% | 0.78% | 78.7 |
| 49+53 | 0.50 / 0.50 | **84.6%** | **2.09%** | **87.5** |
| 49+53 | 0.80 / 0.80 | 74.0% | 0.99% | 83.1 |
| 49+53 | 0.92 / 0.92 | 68.8% | 0.73% | 80.1 |
| 11+49 | 0.50 / 0.50 | 82.9% | 1.93% | 86.8 |
| 11+49 | 0.92 / 0.92 | 66.6% | **0.36%** | 79.2 |
| 11+49 | 0.98 / 0.98 | 60.7% | 0.16% | 75.3 |

**AND** (alarm only if both models pass their threshold):

| Pair | Thresholds | Recall | FP rate | F1 |
|---|---|---:|---:|---:|
| 48+49 | 0.50 / 0.50 | 74.2% | 1.36% | 82.5 |
| 48+49 | 0.86 / 0.86 | 65.7% | 0.83% | 77.7 |
| 49+53 | 0.50 / 0.50 | 75.3% | **1.15%** | 83.6 |
| 49+53 | 0.92 / 0.92 | **67.2%** | **0.47%** | 79.5 |
| 11+49 | 0.50 / 0.50 | 74.4% | 0.89% | 83.6 |
| 11+49 | 0.86 / 0.86 | 66.4% | **0.31%** | 79.2 |

`test_best_ensemble.json` records mean of **11+49 at 0.98/0.98**: recall **60.7%**, FP rate **0.156%** (fp = 3, fn = 181). That point is **below** the 66% recall floor. It shows the full-clip scores can be pushed to a handful of false clips, and it also shows the cost. Do not cite it as a point that already meets §2.

Inside the allowed band (recall ≥ 66%), full-clip ensembles already reach **well under 1%** FP on this curated test. The clearest examples: **49+53 AND @ 0.92** (recall 67.2%, FP 0.47%) and **11+49 AND @ 0.86** (recall 66.4%, FP 0.31%).

Validation reference quoted for the **deployed** pair (not the test table above): mean 48+49 @ 0.50 on `bin_full` val was about **F1 86.3%**, **recall 82.4%**.

---

## 5. Approach B — 8-second windows (`win_bin_full_baseline`)

### 5.1 What one example is

Same curated clips, sliced in memory. No new on-disk 8 s files are required.

| Parameter | Value | Where |
|---|---|---|
| `window_len` | 96 frames = **8.0 s** at 12 Hz | `window_config.json` |
| `stride` | 48 frames = **50% overlap** | same |
| Overlap definition that training actually implements | Inside **one** `.npy`: window `[48:144)` is the overlap of `[0:96)` and `[96:192)`. | `window_index.py` |
| Overlap that production also has, and training does **not** build | Concatenation of the tail of chunk N and the head of chunk N+1 when those chunks are separate files, including across a track-id switch. | Planned as Stage 2. Not in this run. |
| Clip filter | `min_clip_seconds = 4.0`. Clips whose meta duration is under 4 s are dropped before windowing. | `window_config.json` |
| Labels | **Core-aware.** A window overlapping the robbery gesture is `pos_core` or `pos_partial`. A window that is only standing is `ignore`. A window below the partial overlap rule can be labeled negative even inside a robbery file. | `positive_labeling` in `window_config.json` |
| Quality gate | Drops bad motion (teleport, centroid jump, visibility). `pos_core` and `pos_partial` are protected unless the window is empty. | This run kept 23 030 windows, dropped 25 320, protected 2 586 positives. Dominant drop reasons: centroid jump 12 791, teleport 8 423, teleport-after-mask 2 388, low visibility 1 250. |
| Balance | Hierarchical sampler, `samples_per_epoch=3200`, `positive_share=0.35`, at most 4 windows from the same clip per epoch. Global class-weight from imbalance is **off** (`auto_pos_class_weight: false`). | Config + the run that produced the numbers below. |
| Epochs | 25. Checkpoint metric `macro_f1`. A checkpoint with robbery recall 0 cannot win. | `train_defaults` |
| Benchmark ids | 48, 49, 53, 57, 13, 11 | `experiment_benchmark` |
| Split | 80/10/10 by **clip uid**, frozen manifest. This run verified **18 864 / 2 354 / 2 371** clips. | Not the same manifest as campaign `20261103_230735`. |

Code lives in `IAModel/experiments/training/window_campaign/`. The overnight batch finished in about **1.19 h** with exit code 0 after earlier bugs were fixed (those bugs are in the other handoff). The models **did learn**. The comparison below is not “the window run predicted all-negative”.

Checkpoints from that run (on the training server):

```text
artifacts/models/win_bin_full_baseline/modelo_{48,49,53,57,13,11}.pt
```

Window checkpoints do **not** use the full-clip checkpoint schema. Do not score them with `score_pose_ensemble.py` (it expects campaign keys and resamples the whole npy). The window eval script is `eval_window_ensemble.py`.

### 5.2 Test unit for the numbers below

The loader’s test label counts for this run:

| `label_state` | Windows |
|---|---:|
| `pos_core` | 667 |
| `pos_partial` | 60 |
| **scored positives (`n_pos`)** | **727** |
| `neg` | 2 060 |
| `ignore` | 25 |

Positive rate among scored pos+neg windows ≈ 727 / (727+2060) ≈ **26%**. That is higher than the 19% clip positive rate, because robbery files contribute several positive windows and long negatives contribute several negative windows.

`fp_rate` below is the share of **negative windows** flagged, analogous to `fp / 1918` in the clip test, on a **different unit**. §8 says how far that analogy goes.

There is **no** local copy of the window `test_per_model_best.json` in the campaign folder above. The figures in §5.3 and §5.4 are the test lines from the completed remote run (single-model test log, then `eval_window_ensemble.py` on the same test loader). Thresholds were **not** swept to a best F1. The reported points are argmax (equivalent here to a 0.50 softmax cut on a 2-class softmax) and a fixed softmax threshold of **0.80**.

### 5.3 Single models on the window test

| Model | Argmax recall | Argmax FP | Argmax macro-F1 | Softmax @ 0.80 recall | Softmax @ 0.80 FP |
|---|---:|---:|---:|---:|---:|
| 48 `tcn_gru` | 76.6% | **11.0%** | 0.820 | 42.8% | 2.5% |
| 49 `tcn_gru` | 80.7% | **11.9%** | 0.828 | 55.2% | 3.6% |
| 53 `conformer_lite` | 72.5% | **8.7%** | 0.822 | 53.2% | 2.2% |
| 57 `ms_tcn` | 70.3% | **7.9%** | 0.819 | 35.6% | 1.0% |
| 13 `res_tcn` | 82.7% | **17.1%** | 0.796 | 55.0% | 6.3% |
| 11 `dilated_tcn` | 75.1% | **11.5%** | 0.810 | 34.7% | 2.5% |

At the high-recall operating point, window FP is about **8–12%** for the models that matter (49 is 11.9% at 80.7% recall). Raising the threshold to 0.80 cuts FP into the 1–4% range and **drops recall through the 66% floor** on every single model in this table. Model 49 at 0.80 is 55% recall. Model 53 at 0.80 is 53% recall.

### 5.4 Ensembles on the same window test

Same checkpoints, scores combined on the **same** window (not across time).

| Pair | Rule | Threshold | Recall | FP rate |
|---|---|---:|---:|---:|
| 48+49 | mean | 0.50 | 79.1% | **10.9%** |
| 48+49 | AND | 0.50 | 71.9% | **7.8%** |
| 48+49 | mean | 0.80 | 47.9% | 2.4% |
| 48+49 | AND | 0.80 | 39.2% | 1.8% |
| 49+53 | mean | 0.50 | 79.2% | **8.5%** |
| 49+53 | AND | 0.50 | **66.7%** | **5.3%** |
| 49+53 | mean | 0.80 | 48.0% | 1.6% |
| 49+53 | AND | 0.80 | 40.9% | 0.8% |
| 49+57 | mean | 0.50 | 77.4% | 8.4% |
| 49+57 | AND | 0.50 | 65.6% | 4.5% |
| 49+57 | mean | 0.80 | 41.4% | 1.0% |
| 49+57 | AND | 0.80 | 27.9% | 0.8% |

The only window ensemble point that still clears ~66% recall with the lowest FP in this run is **49+53 AND @ 0.50: recall 66.7%, FP 5.3%**. That is about **53 false windows per 1 000 negative windows**. The 0.80 AND points fall to ~28–41% recall.

48+49, the production family, barely helps: mean @ 0.50 is 79.1% recall and 10.9% FP, essentially model 49 alone.

---

## 6. Side-by-side: the window run did not improve on the clip run

Match **recall**, then compare FP. Rounded to the same precision as the logs.

### 6.1 Single model, high-recall region

| Model | Full clip (chosen threshold) | 8 s window @ 0.50 |
|---|---|---|
| 49 | recall 80.7%, FP **2.6%**, F1 84.4 | recall 80.7%, FP **11.9%** |
| 48 | recall 80.5%, FP **2.9%**, F1 83.6 | recall 76.6%, FP **11.0%** |
| 53 | recall 80.3%, FP **2.1%** (threshold 0.92), F1 85.0 | recall 72.5%, FP **8.7%** |
| 57 | recall 82.4%, FP **5.1%**, F1 80.9 | recall 70.3%, FP **7.9%** |
| 11 | recall 82.2%, FP **3.0%**, F1 84.5 | recall 75.1%, FP **11.5%** |

Model 49 keeps the same 80.7% recall and the false-positive rate goes from 2.6% to 11.9% (about **4.7×**). Model 53 on full clips is better on **both** recall and FP than model 53 at either reported window threshold.

### 6.2 Ensembles near the recall floor and at high recall

| Rule | Full clip | 8 s windows |
|---|---|---|
| 49+53 mean, high recall | recall 84.6%, FP **2.1%**, F1 87.5 @ 0.50 | recall 79.2%, FP **8.5%** @ 0.50 |
| 49+53 AND, recall just above 66% | recall 67.2%, FP **0.47%** @ 0.92 | recall 66.7%, FP **5.3%** @ 0.50 |
| 48+49 mean, high recall | recall 81.8%, FP **2.2%**, F1 85.7 @ 0.50 | recall 79.1%, FP **10.9%** @ 0.50 |
| 48+49 AND, mid-70s / low-70s recall | recall 74.2%, FP **1.36%** @ 0.50 | recall 71.9%, FP **7.8%** @ 0.50 |
| Best low-FP point that still has recall ≥ 66% | 11+49 AND @ 0.86: recall 66.4%, FP **0.31%** | 49+57 AND @ 0.50: recall 65.6%, FP **4.5%** (just under the floor). Next legal point is 49+53 AND @ 0.50 at **5.3%**. |

On this evidence the 8 s training is worse in the region the product cares about. The full-clip campaign already had ensembles under **1%** FP with recall still above **66%**. The window campaign, at that recall, is around **5%** FP. Pushing the window threshold until FP looks like the old campaign drops recall to about **40%**, which misses §2.

### 6.3 Store arithmetic, so the target is not confused with the tables

| Operating point | If the rate applied to 1 000 negative decisions |
|---|---|
| Owner’s store today | ~**100** false alarms |
| Owner’s desired store | **3–5** (wish: 1) |
| Full-clip 49+53 AND @ 0.92 (curated clips, recall 67%) | ~**5** |
| Full-clip 11+49 AND @ 0.86 (curated clips, recall 66%) | ~**3** |
| Window 49+53 AND @ 0.50 (curated windows, recall 67%) | ~**53** |
| Window 49 argmax (curated windows, recall 81%) | ~**119** |
| Deployed 48+49 mean @ 0.50 on the **curated clip** test | ~**22** (recall 82%) |

The last row is the important tension. The **deployed** models are the full-clip 48+49, and their curated-clip test FP is ~2%, but the owner still sees ~10% of store interactions alarm. **Passing the curated test at 2% has not produced 3–5 false alarms per 1 000 store interactions.** A proposal that only maximises the curated-clip table will not, by itself, answer the store complaint. See §8.

---

## 7. Why “just match production’s 8 seconds” did not come out finer

These are observations from the two runs, not a claim that every 8 s design is useless.

1. **The gesture is already short.** A theft of about 3 s fits inside one 8 s window, and it also fits inside a full clip that is then resampled to 96 frames. Resampling the whole labeled action did not remove the gesture the folder was recorded for. Cutting that same clip into 8 s slices did not add a gesture the full-clip model was missing.

2. **The window label is stricter, and the FP rate still got worse.** On a full clip, standing before the grab is part of a positive example. On a window, standing is `ignore`, and a slice that only barely overlaps the grab can be negative. Firing on that slice counts as a false positive. That makes the window metric harsher. The observed gap (2.6% → 11.9% at the same 80.7% recall for model 49) is larger than a relabeling footnote. It is still a real degradation on the metric the window run itself uses.

3. **Same-family ensembles do not remove the errors.** 48 and 49 are both `tcn_gru`. Their window AND only moves FP from ~12% to ~8% while recall falls from ~81% to ~72%. Diversity (49 with 53 or 57) helps more and still leaves ~5% FP at the recall floor. The full-clip grid already showed the same pairs doing much better, so the missing piece is not “we forgot to average two models”.

4. **Thresholding cannot hit both ends of §2 on the window scores we have.** The high-recall region is ~8–12% FP. The ~1% FP region is ~30–45% recall. There is no reported window point with recall ≥ 66% and FP near 0.3–0.5%.

5. **This run never trained the thing the plan said would attack store false positives.** `WINDOW_CAMPAIGN_FINAL_PLAN.md` Stage 2 is real session chunks, production overlaps across chunk boundaries, and hard negatives mined from benign sessions. This benchmark sliced the **same action-clip corpus**. It did not add the long walking / idle / occlusion context that makes a store interaction different from a folder named “negative gesture”.

---

## 8. What this comparison does **not** prove

Use these limits in the proposal. Do not “correct” them away.

1. **The tests are not paired.** Full-clip test = 2 379 clips from campaign `20261103_230735`. Window test = windows from a later frozen split (18 864 / 2 354 / 2 371) plus a 4 s minimum duration and a quality gate. Same kind of data, not the same rows. Absolute gaps of a few tenths of a percent would be noise. Gaps of 4× in FP at matched recall are not.

2. **The decision unit differs.** One 16 s negative clip is one negative in approach A and about three windows in approach B. One bad window fails approach B even if a single resampled vector of that file would have been correct. Part of the FP increase is the unit. §7.2 is why that does not explain the whole gap.

3. **Threshold protocol differs.** Approach A “best” rows are a sweep (`test_per_model_best.json` and the ensemble grid). Approach B rows are fixed 0.50 and 0.80. A sweep on window scores might find a slightly better knee. It cannot invent a 66% recall / 0.4% FP point if the scores do not rank that way; the two published thresholds already show the knee falling through the recall floor. If you propose a sweep, say what knee you expect and how you will know it failed.

4. **Curated-clip FP is not store FP.** The deployed full-clip ensemble is already ~2% FP on curated negatives and ~10% of store interactions according to the owner. Benign store motion (walking, standing, browsing, chunk edges, second person) is under-represented in both tests. Beating approach A on another action-clip split does not guarantee 3–5 alarms per 1 000 interactions.

5. **Production overlap across files was not trained or measured.** A live 8 s buffer can be `concat(tail of chunk N, head of chunk N+1)`. The window index only overlaps inside one file.

6. **`sanity_pose` is already on in production.** A proposal whose only filter is “drop clips that fail pose sanity” repeats a gate the owner says is already deployed. The quality gate inside the window trainer is a separate, training-time filter and is not that production check.

---

## 9. Constraints on any acceptable proposal

- Robbery recall on the evaluation you choose stays **≥ 66%**. Report the positive support, not only a percentage.
- State the false-positive rate on **negatives**, and translate it into alarms per 1 000. Say whether those 1 000 are curated clips, 8 s windows, or store interactions. The owner’s pain is the third one.
- A 3 s theft must still be detectable from **one** 8 s window. Do not make “two consecutive hits” the primary detector.
- 48 and 49 must stay distinct checkpoints if you keep them. Same architecture name, different graphs.
- Do not delete or regenerate the frozen window split manifest or `core_index.json` as a side effect of an experiment. A new split needs an explicit new manifest path.
- Do not score window checkpoints with the full-clip ensemble script.
- The previous window training bugs (all-negative predictions, sampler filled with negatives, checkpoint selected by accuracy) were fixed **before** the numbers in §5. Do not spend the proposal re-diagnosing them unless you have a new measurement that says they are back.
- The owner wants a **proposal**, then a decision. Do not launch a multi-hour training job as the answer to this document.

---

## 10. Questions to answer directly

Answer each one. “It depends” is acceptable only with the measurement that would decide it.

1. **Is the conclusion in §6 fair?** If you disagree, say which row is the unfair one and what number would change your mind. If you agree, say so in one sentence and move on. Do not reopen the comparison without a new table.

2. **Why did matching the 8 s production buffer on these clips fail to reduce false positives?** Pick the main cause (label definition, decision unit, missing session negatives, quality-gate bias, threshold calibration, architecture, something else) and say what evidence in this document supports it. Separate “why the window test looks worse” from “why the store still sees ~100/1 000 with the deployed full-clip models”.

3. **What is the next experiment?** One primary experiment, not a menu. Specify data, label rule, model(s), decision rule, what you will refuse to change, expected recall and FP, and the result that means the experiment failed. The experiment must be capable of speaking to **store** false alarms, not only to another slice of the same action-clip folders.

4. **Can threshold or ensemble changes on the models we already have reach §2?** The tables say no for the window checkpoints, and they say the full-clip checkpoints already have ≥66% recall at <1% FP **on curated clips** while the store is still at ~10%. If your answer is “retune 48+49”, explain why that would move store interactions when the curated test is already far better than the store. If your answer is “no, we need different negatives or a downstream veto”, describe the veto and how a 3 s theft survives it.

5. **What will you do with walking, standing, and hand-to-body / pocket / bag motion?** The owner suggested discarding alarms whose peak is ordinary gait or standing, and requiring a hand-to-body pattern on the short peak. They also noted that an open-cart theft may not have hand-to-body, and that many thefts start while the person is walking, so a global “person is walking ⇒ not a theft” rule is wrong. Either design those checks properly or reject them and replace them. Pose-only models may be unable to go from ~8–10% window FP to 0.3% at high recall; if that is your view, say what extra signal you need (products, shelf, tracker, session context) and what you can still do with pose alone.

6. **What do you need from the owner before anyone trains?** Name the missing artifact (a set of real store sessions with false alarms, staged robberies through the production chunker, a frozen session split, a decision about whether 3–5/1 000 is the acceptance test). If nothing is missing, say that training can start from data already on disk and give the path.

---

## 11. Required shape of the proposal

Return this and nothing else as the decision document. Keep it short enough that the owner can accept or reject it.

```text
Title:
One-sentence bet:

Evaluation that decides success:
  - Positive unit (clip / 8 s window / session):
  - Negative unit:
  - Recall floor: >= 66% on N positives (state N)
  - FP bar: X per 1 000 negatives of that unit
  - Separate store bar, if the test above is not the store: 3–5 false alarms / 1 000 interactions
  - Failure looks like:

Data you will add or stop using:
Label rule:
Models (existing checkpoints vs retrain):
Decision rule (single threshold / mean / AND / veto). Confirm a 3 s theft still fits in one positive decision:
What you will not change:
Cost (hours, GPU, new recordings):
First result you will show the owner, and by when:
Risks you are explicitly accepting:
```

If you believe the right move is to **stop the window campaign** and spend the effort on store false-alarm mining against the deployed 48+49, that is a valid proposal. Write it in the template. If you believe Stage 2 of the window plan is still the right bet, say which of §5’s failures it is supposed to fix and which it will not fix.

---

## 12. File index

| What | Path |
|---|---|
| This brief | `IAModel/experiments/training/window_campaign/WINDOW_VS_CLIP_CAMPAIGN_AUDIT.md` |
| Window trainer config | `IAModel/experiments/training/window_campaign/window_config.json` |
| Window ensemble eval | `IAModel/experiments/training/window_campaign/eval_window_ensemble.py` |
| Window training handoff (bugs already fixed) | `IAModel/experiments/training/window_campaign/WINDOW_CAMPAIGN_AGENT_HANDOFF.md` |
| Full-clip reports | `/home/debian/Documentos/Kanvis/home/artifacts/runs/campaign_20261103_230735/reports/bin_full/` |
| Deployed pair | `IAModel/experiments/training/campaign/C2_DEPLOY_ENSEMBLE_48_49_BIN_FULL.md` |
| Window checkpoints (training server, not necessarily this workstation) | `artifacts/models/win_bin_full_baseline/modelo_{48,49,53,57,13,11}.pt` under the window campaign run directory |
| Full-clip checkpoints (as recorded inside the JSON) | `.../campaign_20261103_230735/models/bin_full/modelo_*.pt` |
