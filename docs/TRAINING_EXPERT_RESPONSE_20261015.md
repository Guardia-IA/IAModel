# Kanvis training: expert response and next-campaign specification

Date: 2026-10-15. Scope: supplied `TRAINING_AUDIT.md`, from correct pose arrays to deployment decisions. Language: English, as requested by the brief. This is a new response; the input has not been edited.

Evidence labels: **R** = reported by the supplied audit/owner, not independently reproduced; **V** = verified mathematical or documented library behavior; **C** = computed here from supplied numbers; **H** = hypothesis or engineering proposal. Numerical improvement ranges below are **planning hypotheses, not measured forecasts or confidence intervals**. No repository, checkpoints, real logits, videos, or October data were supplied. Referenced source-code line numbers are the audit's pointers, not lines inspected here.

## 1. Executive summary

**Yes: there are consequential training and evaluation mistakes to fix. However, the evidence does not establish that architecture is the principal limitation, or that the current information ceiling has been reached.** Fix the checkpoint snapshot, replace clip-random evaluation with grouped chronological replay, freeze the decision rule before testing, and reconcile the training/evaluation population first. These changes establish whether subsequent gains are real. Then fix physical time, masks, normalization and partial-action training. Do not run another broad architecture sweep first.

The most fundamental ambiguity is semantic: category 6 includes pocketing one's own phone, whereas the business calls that a false theft alarm. Correct classification of that training label can therefore be a business error. A skeleton can identify concealment-like movement, but cannot generally identify ownership or intent. Keep an auxiliary concealment target, separately audit legitimate/unknown concealment, and reserve a theft probability claim for evidence that supports that meaning.

**Recommended next model:** an eight-joint, 8-second, 12-Hz masked residual dilated TCN, 96 time steps, width 64, five blocks with dilations 1/2/4/8/16, two kernel-3 convolutions per block, masked mean plus max pooling. Use torso-relative isotropic coordinates, physical-time velocity, visibility and a few wrist–torso distances; no acceleration initially. Train three independent seeds with AdamW, 100-epoch ceiling, five warm-up epochs, cosine decay and operational-metric checkpoint selection. Use one transparent session-balanced sampling policy and ordinary BCE, with an auxiliary concealment head. Details and exceptions are specified below. This is a defensible first configuration, **not an empirically demonstrated best model**.

Use a mean of seed margins, optionally monotone Platt-calibrated on a separate population-matched partition; aggregate with a maximum over fixed windows as the initial baseline. Select the threshold on **negative-session aggregate scores**, using an order-statistic risk bound, then freeze it. Persistence is an ablation, not a default presumed improvement. Return scores and quality flags to Stage 2.

The observed production reference is **30.6 false-positive sessions per 1,000 fragments**, with an illustrative independent-session exact 95% interval **18.8–46.8**. At 1,000 fragments/day that is the same numerical FP/day range. The present evidence supports neither 66% event recall at 1 FP/day nor any improved operating point. Even the lowest supplied frontier point projects to approximately 8.5 FP/day under the illustrative effective-look model. A reasonable next experiment aims for **≤10 FP/day with ≥60% event recall**, progressing to **≥73.3% Stage-1 event recall** if Stage 2 loses 10% and the end-to-end target remains 66%. These are acceptance targets, not promises.

Cheap filters to investigate first are duplicate-alarm suppression, mask-aware quality handling, score consistency under small temporal shifts, soft ROI/quality features and reviewed-production hard negatives. Use conservative fragment chaining only for deduplication; never suppress an apparent employee simply because of dwell time. No combination of unvalidated filters may be advertised as providing 90% FP removal. If 10 FP/day and 73.3% recall are actually achieved, a separately verified 90%-FP-removal/90%-recall-retention verifier would yield **1 FP/day and 66% recall**.

## 2. Critical review of the supplied audit

### 2.1 Confirmations, corrections and evidence boundaries

| Item | Status | Assessment |
|---|---|---|
| H1 snapshot aliasing | V mechanism; R occurrence | PyTorch explicitly documents that a saved `state_dict()` reference is not an immutable snapshot [S1]. Clone or serialize at improvement. The actual repository cannot be certified from this attachment. |
| October counts | C/R | Category totals sum to 19,043, with 2,592 cat-6 clips. Evaluation excludes 434/2,376 = 18.27% of validation clips. Among positives it excludes only 1/324 = 0.31%; among negatives 433/2,052 = 21.10%. This is strongly class-selective population selection, not merely 18% random missingness. |
| Current confusion matrix | C | TP=206, FN=117, FP=11, TN=1,608. Recall 63.777%, precision 94.931%, F1 76.296%. The arithmetic is coherent. |
| Production arithmetic | C | 20/654 = 3.058%, or 30.58/1,000; “20–30/1,000” slightly understates this particular observation. Need to establish whether 20 is alarms or distinct false-alarm sessions. |
| Repeated frontier | H, overstated in audit | Two selected points with identical error counts on different pools suggest stagnation. They do not establish identical ROC frontiers, identical missed events, statistical equivalence, or an information ceiling. Evaluate both checkpoints on the same blind grouped replay and compare paired errors. |
| Additional data and unchanged FN | H, causal overclaim | Adding few positives can limit benefit, but extra informative negatives can improve recall at fixed FPR. Unchanged FN is not proof that +18 positives caused stagnation. |
| Single-store scope | R | Nine cameras, one store; cross-store collection is optional future work, not a current acceptance prerequisite. |
| H2 leakage | R/H | Clip grouping is inadequate when clips overlap or share recording identity. Old-sample 68.9% overlap and two origins must not be copied onto October as measured facts. Source origins are not necessarily two independent sessions. |
| H3 selection | R/V | Test-time retuning invalidates a final test. A 0.01% FPR has fewer than one expected error in 1,619 negatives. A 0.1% session goal is a different estimand. |
| H4 time | V mechanism/R prevalence | Deleting frames destroys elapsed-time information; resize-after-velocity can make channels inconsistent. Last-frame repetition is padding, not slow-motion interpolation. The audit sometimes conflates the two. |
| H5 null augmentations | V conditional | Translation and positive isotropic scale cancel under the stated normalization, absent clipping, numerical floors or altered masks. Clipping makes these operations non-null by distortion, not by a useful camera simulation. |
| M1 RNN | V/R | Last-output pooling mishandles the reverse final state [S2]. But the forward RNN still spans the sequence: a five-frame convolutional receptive field does **not** mean the TCN-RNN sees only five frames. Stacked bidirectional layers can also mix context into the final token. |
| M1 attention | V qualified | A pure attention encoder without position is order-insensitive at pooled output, conditional on token contents. Velocity already encodes local temporal information; a Conformer convolution also supplies order. “Cannot use time at all” is too strong. |
| M1 parameter argument | H | Millions of parameters are not proof of overfitting; the audit's 3–6k training-clips remark is an old-pool estimate, not the October 14,279 train inventory. Effective independent groups still matter. |
| M2 feature scales | H | An 11:1 numerical scale does not prove velocity is ignored: learned first-layer weights can compensate. Standardization is a cheap conditioning experiment. |
| M3 anisotropy | V qualified | In x/W,y/H, Euclidean angles are distorted for non-square frames. Ordinary horizontal reflection with joint swap is not itself shear. Rotation in anisotropic space is problematic. Mean joint radius 0.054 is not a measured body bounding-box width of 5%. |
| M4 quality thresholds | R/H | Disabled filters are not automatically a training error. Hard gates proposed from an old sample remove 16.9% of positives; that is material recall loss. Upright geometry can reject kneeling/bending. Masking is preferable to broad exclusion. |
| M7 rare categories | C correction | October retains 7 (458), 13 (59), 14 (219). The claim that drop-below-50 removes them is false for the supplied campaign table. Check the actual map; retain all trustworthy binary negatives regardless of rare multiclass support. |
| M8 metadata AUC | H | Metadata AUC is a diagnostic of separability, not proof that the production classifier relies on a shortcut. AUC below .5 can be inverted; .269 implies .731 after reversal. |
| M9 scores | V correction | Binary sigmoid, logit margin and positive-temperature calibration have identical rank in exact arithmetic. A rare-FPR threshold need not be near probability 1. Saturation must be measured, not presumed. The stated prior-offset range is also inaccurate; see Q40. |
| M10 multiple users | R/H | One labeled actor per action does not imply one detected skeleton per video. Disabling `single_user_only` without track-level labels can introduce severe false positives in training. Recover target identity first. |
| CI claims | C correction | 11/1,619 exact two-sided CP interval is 0.340–1.212%, not 0.38–1.21% (the latter resembles another interval convention). Specify the method. These intervals assume independent clips and are optimistic if clips cluster. |
| Data-volume claims | C/H | Rule-of-three is approximate, normal-width sample calculations are approximate, and 300–400 positives only power a 5-point paired gain under favorable discordance. “Statistical computations are exact” is too broad. |
| Generalism | V scope | Use existing negative categories for audit/sampling, without introducing hand-on-back/scratching rule labels. Camera context and quality gates need no action-by-action tuning. |

### 2.2 Findings that must not drive the decision

Reject blanket “more epochs will solve it,” universal employee suppression, hard upright-person rules, blanket exclusion of ambiguous positives from reported recall, claims that independent-window arithmetic proves domain shift, and a numeric attribution of production errors to a bug without counterfactual results. Do not equate zero observed false positives with zero future risk. Do not treat a phone-concealment label as proof of theft. Do not tune a new threshold after looking at the frozen test.

## 3. Ranked diagnosis

Percent attribution is not identifiable from aggregate counts. To satisfy the requested prioritization, the following **subjective planning allocation sums to 100%**, but it is neither an estimated fraction of observed errors nor a posterior probability. Causes interact; experiment effects must not be added.

| Rank | Cause | Planning allocation | Confidence in relevance | FP / recall implication and cheapest check |
|---|---|---:|---|---|
| 1 | Label meaning versus theft + missing object/ownership context | 30% | High semantic, unknown frequency | Some genuine cat-6 predictions become business FP. Audit every reviewed FP and 200 stratified positives for legitimate/unknown concealment; no action taxonomy expansion required. |
| 2 | Clean action-centered clips versus arbitrary fragmented windows | 25% | High mismatch, unknown magnitude | Could explain a substantial part of 30.6 FP/day; matched replay/windows may lower FP but lose cropped positives. Replay the same model with exact current preprocessing and then shared preprocessing. |
| 3 | Source/actor confounding, inadequate negative diversity | 20% | Medium on October | Hidden evaluation optimism and recurrent production mistakes. Hash/overlap/group inventory plus chronological split and nuisance-only baseline. |
| 4 | Model selection, pool exclusions, many looks per session | 15% | High procedural | Changes measured performance and expected alarm counts; does not itself improve the learned boundary. Re-evaluate a frozen pair on all labeled target tracks. |
| 5 | Snapshot, pooling, normalization, optimization defects | 10% | High mechanism, medium occurrence | Correctable but sign/magnitude unknown. Paired fixed-data ablations, three seeds. |

For the requested modeling/data/information split: **modeling 20%, data/labels/protocol 50%, missing information 30%** is a working resource-allocation opinion (overlapping causes above have been regrouped). It must not be presented to investors or engineers as measured causal attribution. A proper factorial study and matched semantic cases can change it substantially. Tracker failures are not promoted to a primary diagnosis because the owner reports few tracker-induced FP and defers that topic.

## 4. One prioritized end-to-end solution

### 4.1 Target definition and records

Create a label record with `event_id`, target track, source session/recording, actor identity when known, camera, day, action interval if known, provenance, original category, `concealment_motion`, `theft_status ∈ {confirmed, legitimate, unknown}`, and review confidence. Preserve original labels and label-policy version. Unknown ownership is not forced to negative. Verified phone pocketing is legitimate for theft and positive for concealment. Cat 14 stays a theft-negative when legitimately owned; its kinematic auxiliary label is reviewed rather than inferred from category alone.

Use all trustworthy target tracks, including multi-person scenes. Quarantine ambiguous track assignment. Never propagate one actor's category to every skeleton. Every rejected sample and gate abstention gets a reason. All real theft events remain in the end-to-end denominator, including occluded and unrecoverable cases.

### 4.2 Splits before transformations

Partition source groups before windowing, augmentation, mining or calibration. Use chronological blocks across all nine cameras as the main future-deployment test; keep overlapping recordings, staged actors/takes and known visits together. A 60-second block alone is insufficient if the same source session straddles boundaries. Connected components of overlaps and common source sessions are a safer grouping primitive. Use an embargo at least as long as retained temporal context at each boundary. Do not infer identity from clothing with unwarranted certainty.

Development has three grouped folds for model/hyperparameter selection. Calibration C fits score mappings. Threshold set T selects the operating threshold using negative-session scores only after architecture/aggregation is frozen. Test U is unopened until the full policy is frozen. When data are scarce, use grouped cross-fitting within development; never borrow U. Leave-one-camera-out is a secondary robustness stress test, not the primary estimate for these same nine deployment cameras. Report actor-held-out staged results separately from real incidents.

### 4.3 Physical time and partial observations

Use full pose arrays and timestamps/frame indices. Map normalized coordinates into isotropic units `(x*W/H, y)` before distances. Resample positions/masks to 12 Hz over actual elapsed time, with anti-aliasing when reducing rate; no stretching a 30-second clip into eight seconds. One model window is `[t-8s,t)` with 96 samples; stride 1 second. Windowing crosses file-chunk boundaries with a ring buffer, not a new time scale for N+N+1. At track end process the remaining fragment once with a mask. Match this behavior in training.

Interpolate only gaps ≤0.25 seconds with valid endpoints, and keep an interpolation flag. Longer gaps remain missing; neither duplicate positions nor bridge derivatives through them. Zero-fill missing **normalized** features and expose masks. Compute velocity after resampling in torso lengths/second; mark derivative invalid across missing endpoints. For noncausal centered interpolation allow only the explicitly bounded short look-ahead, and use it identically in replay. The owner has no latency ceiling, so this is acceptable.

Anchor at mid-hips when reliable, otherwise mid-shoulders with a separately fitted anatomical offset; expose anchor availability. Use a robust local torso-scale estimate from valid shoulder/hip distances, e.g. trailing one-second median with a training-derived floor. Do not force a tiny torso length to become an enormous scale factor. Center/scale drift becomes a soft quality feature. No automatic rejection for kneeling, reaching behind, low wrist visibility or absence of a long history.

Generate 8-second windows at random offsets and short 1–8-second fragments. A positive action clip is a weakly labeled **bag** of candidate windows. Preserve the complete bag somewhere in each training sampling cycle. Do not label every arbitrary crop positive: approach-only and aftermath-only fragments get unknown target unless temporally reviewed. Verified negatives may provide negative crops after excluding label-ambiguous intervals. Use a bag-level smooth-max loss for unlocalized positives and ordinary window loss for known intervals. Cap each event's total loss contribution, independent of number of windows. Temporal annotation of a small representative subset is preferable to guessing crop labels.

### 4.4 Concrete next training recipe

| Component | Exact first configuration (H; fixed before run) |
|---|---|
| Input | Eight joints; per joint x/y, vx/vy, coordinate-valid and derivative-valid flags: 48 channels. Add six distances (each wrist to each hip and shoulder midpoint) plus six validity flags, and anchor/scale availability: about 62 channels, finalized in a versioned schema. Raw confidence added only if genuinely available, never fabricated from a frame mask. |
| Scaling | Training-only median/IQR standardization of continuous channels, floor IQR at 1e-3; masks not standardized; clip standardized continuous values to ±10 and monitor clipping frequency. |
| Network | 1×1 projection to width 64; five residual blocks, each two k=3 convs at dilation 1/2/4/8/16; per-token LayerNorm over channels, GELU, dropout .15. Re-mask after each convolution/residual. Symmetric convolution acceptable. Receptive field 1+4(1+2+4+8+16)=125 samples, covering a 96-step window. |
| Readout | Masked mean and masked max concatenated (128 dimensions), linear 128→64, GELU, dropout .15, theft logit and auxiliary concealment logit. All-missing input returns explicit abstain, not a score. |
| Size | Approximately 0.14 million parameters for this definition; verify from instantiated model. Norm over time is avoided because padding/length otherwise alter statistics. |
| Loss | BCEWithLogits on known theft targets, weight 1; auxiliary concealment BCE weight .2 on known motion targets; for weakly localized bags use normalized log-mean-exp pooling at temperature .5 over window logits before BCE. Unknown theft targets are masked. Positive bag labels require reviewed theft status. |
| Batch/sampler | Batch 64 windows/bags subject to memory; first draw independent source/event groups, then one crop/bag per group. Target 25% verified positives, 25% reviewed hard negatives, 50% ordinary labeled negatives where available. Avoid repeat copies within batch. No additional class weights, focal loss or UID multiplier initially. Log actual composition. |
| Optimizer | AdamW LR 3e-4, betas (.9,.999), weight decay 1e-3 excluding bias/norm; clip gradient norm 1.0. Five-epoch linear warm-up then cosine to 3e-6 by epoch 100. |
| Training | Ceiling 100 epochs; evaluate every epoch; no early stop before epoch 30; patience 15 evaluations without operational improvement. An epoch is one pass over real training groups, not expanded variant rows. Snapshot best weights with clone/serialization. |
| Selection | Development event recall at threshold satisfying the development negative-session risk budget; use partial low-FPR AUC only as a prespecified tie-break. No accuracy/F1 checkpoint selection. Report a 10-FP/day exploratory budget separately if 1/day cannot be estimated. |
| Seeds | 17, 42, 103; reset all RNGs per run; store split, loader, software and determinism settings. |
| Augmentation | Left/right reflection p=.5 with joint swaps in body-relative coordinates; temporal speed 0.9–1.1 p=.3 using coherent resampling; correlated coordinate noise SD .01 torso lengths p=.3; joint dropout p=.2 (one joint, .1–.4 s); gap p=.2 (.1–.5 s); random boundary crops. Never impute a crop's theft label from the parent without retained evidence. |
| Disabled initially | Acceleration, MixStyle, label smoothing, SSL, EMA/SWA, synthetic global translation/scale, broad 15° rotations, camera embeddings and learned stacking. Each adds an unproven interaction. |
| Ensemble | Equal mean of the three seed theft margins. Test a single seed against the ensemble with paired non-inferiority; deploy the lighter single model if within the defined tolerance and meeting the same risk budget. |

If too few confirmed theft labels exist after semantic review, train the same architecture's **concealment head first** using original cat-6 semantics; report concealment recall and theft-alarm FP separately. Do not silently present that head as a calibrated theft detector. This conditional branch prevents relabeling from eliminating most positive training supervision.

### 4.5 Decision contract and statistical control

For binary two-logit checkpoints use `m=z_positive−z_negative`; for a one-logit head use its logit. A mean of seed margins is a deterministic ranking score, not automatically log odds of theft. Fit an increasing Platt mapping on C if enough appropriate labels exist; otherwise publish uncalibrated margins. The alarm threshold is selected on the **aggregate score actually emitted by the deployed policy**: initially session maximum over the 8-second windows. All overlapping windows and startup fragments must be included.

Let T contain n independent negative-session scores sorted ascending, S_(1)…S_(n). Choose the smallest rank k satisfying `BinomCDF(k−1; n, 1−α) ≥ 1−δ`; threshold at S_(k) and alarm only on strictly greater scores. Here α=.001 for ≤1 false-positive session per 1,000, δ=.05. This is a distribution-free order-statistic tolerance bound for a **fixed scorer and exchangeable sessions**, not immunity to shift [S4]. At n=2,995, only approximately the maximum can meet the zero-exceedance requirement. If no k exists, collect more data or state no certification; do not relax the bound silently. Ties use strict exceedance conservatively; a tie-breaking rule must be fixed before evaluation.

For repeated alarms/session, Bernoulli FPR is insufficient: evaluate counts per exposure-day and their dispersion separately. The one-per-session cap makes false-positive-session count equal alarm count only within that fragment definition. Deduplication across fragments changes the unit, so freeze visit-linking and count store-level unique notifications too. Count true events per incident across cameras, not per fragment.

On U, compute exact one-sided binomial bounds for independent-session summaries and day/block-bootstrap sensitivity intervals for correlated store traffic. For daily counts use days as blocks and report the observed mean and high-percentile daily count; do not claim a hard daily maximum from a bound on mean rate. If days are too few for reliable block inference, call the result provisional. Camera/day drift can invalidate exchangeability; monitor it.

### 4.6 Roadmap and acceptance gates

1. **P0, correctness:** clone snapshots in both training paths; validate positive-class mapping, masks and pool counts; golden-array training/inference parity. Pass: restored state equals chosen epoch, score agreement within declared numeric tolerance, no unlogged sample drops.
2. **P0, honest baseline:** replay current #41/#48 unchanged, frozen threshold, on chronological held-out tracks. Pass: count all alarms, all incidents and gated cases with consistent units. A worse number is useful evidence, not a failed audit.
3. **P1, matched model:** fixed time, masks, partial-action labels, normalization; train repaired incumbent and the small TCN with three seeds. First exploratory target ≤10 FP/day and ≥60% event recall. Promote only after paired replay shows benefit with uncertainty reported.
4. **P1, label/data loop:** review 200 positives and an initial 200–500 distinct hard negatives plus a random traffic sample; collect matched staged negatives and real incidents. Preserve held-out days. Keep only gains that survive unseen days.
5. **P2, decision/filter refinement:** test a small registered set of aggregators and filters; freeze choice before T/U. If no policy meets 1/day with useful recall, state that failure; retain candidate-only operation or the separately validated verifier.
6. **P2, operational acceptance:** end-to-end point recall ≥66%, one-sided recall lower bound ≥60%, one-sided FP upper bound ≤1/1,000 at 1,000 sessions/day on a frozen test, plus honest per-day dispersion. A lower-recall policy can be accepted explicitly under the owner's FP-first preference; do not report it as meeting the 66% target.
7. **P3, frontier expansion:** additional joints and object-context oracle study only after the above isolates the limitation. Cross-store testing becomes mandatory only when deployment expands.

## 5. Answers to all 61 design questions

Priorities: P0 correctness/measurement; P1 next training cycle; P2 after baseline; P3 optional research. Effects are directional unless explicitly computed; no unobserved model-performance estimate is a confidence interval.

### 16.1 Operational objective

#### Q1 — False-alarm budget
**Answer/rationale:** Maximize event recall subject to an upper confidence bound of ≤1 false-positive session per 1,000 negative sessions, and separately ≤1 mean false notification/store-day at the actual traffic volume. At nine cameras and H open hours this is an average budget of 1/(9H) per camera-hour, not an equal per-camera quota. With H=12, that is .00926; H is illustrative. A 0.01% clip budget must not substitute for the business metric.
**Effect/test/priority:** Tightening the budget reduces FP and generally recall; run frozen-policy replay and measure daily unique notifications and session exceedances. **P0**.

#### Q2 — Hit definition
**Answer/rationale:** A hit is at least one candidate/notification attributable to a verified incident, with first-event overlap or a fixed ≤8-second post-event grace period. Score a multi-camera incident once. Record latency and the camera/fragment producing the first hit. A false alarm near an unrelated incident is not automatically a TP.
**Effect/test/priority:** Event aggregation can recover missed windows without raising recall artificially through duplicate counting; persistence may lower, not raise, recall. Review event-to-alarm assignments and repeat with zero grace as sensitivity. **P0**.

#### Q3 — Cost and reviewer
**Answer/rationale:** Adopt a constrained objective rather than an invented fixed cost ratio. Assume candidates go to SAM3 or human review, and only confirmed notifications are business alarms. If no reviewer exists, Stage 1 must itself satisfy the final budget. With λ robberies/day the stated preference implies `c_FP/c_FN > .05λ`; derivation is in Q60. λ is not given.
**Effect/test/priority:** Separating candidates from alarms prevents hidden review burden. Measure reviewer throughput, conditional FN and notification count. **P0**; no unique optimal cost ratio can be identified from the stated preference alone.

#### Q4 — Stage-1 target
**Answer/rationale:** For Stage-2 FP retention .1 and positive-event retention .9, Stage 1 needs ≤10 FP/day and ≥73.33% event recall for a 66% final result; ≥83.33% for 75%. The brief's ≥66% followed by 10% loss yields only ≥59.4%. Stage-2 retention must be measured on candidates produced by the chosen Stage-1 policy.
**Effect/test/priority:** This sets a more demanding recall target than the present 63.8% clip result. Replay the full conditional cascade; do not multiply metrics from unrelated pools. **P0**.

#### Q5 — Latency and context
**Answer/rationale:** No latency limit is imposed. Start with 8-second history, 1-second stride, and explicit partial-track processing; compare 4 and 12 seconds after the baseline. Bidirectional within-window processing is allowed. Waiting eight seconds after track entry is unnecessary: process masked startup fragments and final fragments too.
**Effect/test/priority:** More context may improve ambiguity resolution but can dilute a brief gesture. Compare event recall, FP/day and latency by 1/2/4/8+ second fragment duration. **P1**.

### 16.2 Data and labels

#### Q6 — Independent coverage
**Answer/rationale:** Known: one store, nine cameras, 2,592 positive clips. Unknown: distinct positive events, actors and days. Adopt the conservative assumption that many clips are correlated, without inventing a count of hundreds. Build a provenance graph and report independent group sizes; if only a handful of positive groups remain, collect independent events before deep architecture comparisons.
**Effect/test/priority:** Honest grouping may reduce measured recall and increase measured FP; improved diversity can improve both. Inventory source IDs, overlap and actor/take metadata. **P0**.

#### Q7 — Staged versus real
**Answer/rationale:** Use the reported approximate 63/34/3% staged/real-cut/production provenance, but do not apply old-sample origin counts as current truth. Collect legitimate actions by the same actors during the same takes/cameras as staged concealments. Preserve real incidents for separate evaluation.
**Effect/test/priority:** Paired negatives remove a source shortcut, potentially lowering production FP with little genuine recall loss; no percentage is identifiable. Compare source-only AUC and held-out-real recall before/after paired sampling. **P1**.

#### Q8 — Delimitation
**Answer/rationale:** Owner confirms one bounded action per clip; this does not mean every frame or arbitrary crop contains the discriminative concealment. The number of informative seconds is unknown. Annotate onset/end of the decisive interval in 200 diverse positives initially; use weak bags on the remainder.
**Effect/test/priority:** Prevents approach-only positives from teaching broad false alarms while preserving short-event evidence. Validate against temporally annotated fragments and full events, not just parent-clip labels. **P1**.

#### Q9 — Occluded positives
**Answer/rationale:** Variant coverage, including kneeling, does not imply pose observability. Audit wrist visibility and label recoverability independently. Keep every real incident in business recall, reporting visible-subset recall only as a secondary metric. If fraction u is truly unobservable and never detectable, useful targeted recall cannot exceed roughly 1−u without other evidence; random alarms are not recovery.
**Effect/test/priority:** Hard removal can reduce FP but lower end-to-end recall sharply. Stratify real events by visibility and inspect all rejected positives. **P1**.

#### Q10 — Noise and class criteria
**Answer/rationale:** Separate annotation error from intentional semantic broadening. A phone labeled category 6 is consistent with the owner's motion label, but inconsistent with confirmed-theft semantics. Review all known FP, 200 stratified cat-6 clips and 50–100 each of 14/3/4/2; preserve ambiguous cases as unknown and track reviewer agreement.
**Effect/test/priority:** Likely reduces contradictory supervision; not guaranteed to make identical poses separable. Measure errors at a fixed FP budget after re-review, with untouched blind labels. **P1**.

#### Q11 — Category 14
**Answer/rationale:** Legitimate own-bag placement remains theft-negative; add it to hard-negative sampling without declaring bag placement universally innocent. Its auxiliary concealment label depends on observed motion. A pose-only model should express uncertainty, not infer ownership from destination alone.
**Effect/test/priority:** More cat-14 coverage may cut bag-related FP but also reduce recall of pose-identical thefts. Compare paired bag actions with different object provenance and a separate real-incident set. **P1**.

#### Q12 — Production loop
**Answer/rationale:** Mine current alarms, high-score nonalarms, disagreement and a random background sample. Review target actor and incident meaning; group duplicates before sampling. Put reviewed FP into training only if their source days are outside C/T/U. Keep a frozen benchmark and a separate rolling monitor.
**Effect/test/priority:** First 200–500 distinct reviewed hard negatives are a reasonable pilot, not a guaranteed breakpoint. Prespecify a worthwhile gain of ≥20% relative FP reduction at ≤2-point recall loss; observe rather than assume the result. **P1**; detailed loop in Q58.

#### Q13 — Blind incidents
**Answer/rationale:** Aim for 345 independent verified incidents for approximately ±5 percentage-point precision at 66% recall; about 176 for ±7 points and 87 for ±10, using normal planning approximations. With fewer incidents, run the test and publish the wider exact/grouped interval. Do not manufacture independence by cutting more clips.
**Effect/test/priority:** Measurement precision only, no automatic FP/recall improvement. Deduplicate incidents across cameras and report staged and real recall separately. **P0**.

### 16.3 Inference conditions

#### Q14 — Time alignment
**Answer/rationale:** Current production is 8-second chunks plus N/N+1; evaluation sometimes uses three seconds. Adopt the shared eight-second/12-Hz/one-second-stride contract in §4. Preserve elapsed time; source 15 Hz and variable-rate replay are resampled by timestamps. If timestamps are unavailable, use trustworthy source fps and original frame indices; quarantine unknown timing.
**Effect/test/priority:** Reduces timing drift but creates a new score distribution requiring retraining and new thresholds. Golden-array checks at 12/15/18 Hz must agree after resampling within numerical/aliasing tolerance. **P0/P1**.

#### Q15 — Quality and fragments
**Answer/rationale:** Evaluate every target track; truly empty/invalid geometry returns abstain. Use missingness as model input and avoid the audit's blanket 36-frame/50%-visibility gate. Track fragments must be classifiable without preceding history. Conservative chaining may deduplicate notifications, but classification cannot require successful chaining.
**Effect/test/priority:** Retains short-event recall; FP quality effects need measurement. Report all-event recall including abstentions and no-history slices; compare all-track coverage. **P1**.

#### Q16 — Crowding
**Answer/rationale:** Crowded-scene frequency is unknown. Include target-labeled tracks from multi-person scenes; preserve crowding/occlusion flags. Do not interpret a video with two detections as inherently untrainable. Missing target labels go to review, not inherited clip labels.
**Effect/test/priority:** Closes the 21.1% negative-pool exclusion risk and may improve crowded-scene FP. Audit train/eval manifests by detected-user count and test crowd-stratified performance. **P0/P1**.

#### Q17 — Scene context
**Answer/rationale:** Camera, time, track position and geometry may be used as soft context. Shelf/checkout ROIs require a versioned map; object detections are not assumed to exist at Stage 1. Keep the pose encoder separate from context so its value is ablatable.
**Effect/test/priority:** Context can reduce routine-action FP but may learn schedules/cameras as class proxies. Test matched within-camera/time positive-negative pairs and chronological holdouts; retain context only on a paired gain. **P2**.

### 16.4 Modeling decisions

#### Q18 — Spatial normalization
**Answer/rationale:** Use isotropic coordinates and torso anchor/robust scale as §4; retain global position only in a separate optional soft-context branch. Avoid normalization on arm-moving centroids. Do not apply aggressive per-camera transformations without calibration geometry.
**Effect/test/priority:** Expected better geometric consistency, unknown gain. Factorial test old versus torso anchor and anisotropic versus isotropic, same split/seeds; report kneeling and sideways events. **P1**.

#### Q19 — Windows
**Answer/rationale:** Fixed eight seconds at 12 Hz (96 samples) first, because it matches production chunk scale while replacing accidental chunk boundaries. Four- and twelve-second variants are the next two controlled comparisons; do not search dozens of lengths. Short fragments are masked within the same contract.
**Effect/test/priority:** Can lower FP from duration shortcuts and improve boundary-event recall; longer context can also dilute action. Replay equal exposure and threshold each policy separately on T. **P1**.

#### Q20 — Supervision
**Answer/rationale:** Use explicit interval labels on the small reviewed subset; MIL-style bag supervision for unlocalized positive actions. Negative crops require trustworthy negative clips. Unknown partial crops contribute only consistency/auxiliary losses, not fabricated positive theft targets. A bag must retain at least one evidence-bearing window to use a positive label.
**Effect/test/priority:** Removes a likely source of overbroad positives while preventing crop-induced false negative labels. Compare window and event recall on manually localized events. **P1**.

#### Q21 — Head and negative taxonomy
**Answer/rationale:** Binary theft head plus auxiliary concealment head; existing categories remain audit/sampling strata. No per-action alarm thresholds. Start with one BCE loss policy, not category penalty stacking. Multiclass is a controlled comparator once class indexing is fixed.
**Effect/test/priority:** Separates semantic ambiguity from motion recognition, but cannot recover invisible ownership. Track all confusable-category FP and theft recall at equal session budgets. **P1**.

#### Q22 — Anomaly/SSL
**Answer/rationale:** Do not use one-class abnormality as the theft alarm: ordinary rare behavior is common relative to theft. If ≥100k diverse unlabeled windows exist, test masked-pose reconstruction or contrastive temporal pretraining with source-separated fine-tuning; use only development days.
**Effect/test/priority:** SSL may improve data efficiency, with zero or negative improvement possible. Accept only a paired low-FPR gain after the same fine-tuning budget; otherwise omit. **P3**.

#### Q23 — Additional joints
**Answer/rationale:** Keep eight joints for the repaired baseline, then re-extract the available full 17-point skeleton for one matched ablation. Head landmarks and knees/ankles can resolve orientation and kneeling but not ownership. Head keypoints are not precise gaze estimates.
**Effect/test/priority:** Likely posture/visibility robustness; business FP reduction unknown. Same source IDs, split and preprocessing, matched 8 versus 17 joints; allow ~1–1.5 days extraction as reported, not a benchmark reproduced here. **P2**.

#### Q24 — Features
**Answer/rationale:** Coordinates, elapsed-time velocity, validity and wrist-to-torso distances first. Add elbow angles and torso orientation only in one feature ablation. Acceleration is second-order noise-sensitive and stays off initially. Confidence is optional only if exported.
**Effect/test/priority:** Redundant features can make learning easier but add no new information if deterministic from pose. Measure at fixed FP/day and inspect stability under dropout/jitter. **P1**.

#### Q25 — Architecture
**Answer/rationale:** Start with the width-64 masked dilated TCN specified in §4, three seeds. Keep the repaired #41/#48 pair as the incumbent comparator. Anatomical ST-GCN and correctly pooled GRU are the next compact alternatives, not a new 320-row campaign.
**Effect/test/priority:** Lower compute and potentially better regularization; no promised frontier gain. Compare paired error sets, seed variance, latency and deployment memory. **P1**.

#### Q26 — Calibration/decision summary
**Answer/rationale:** Mean seed margins, optional increasing Platt calibration on C, threshold on session maxima using T and a tolerance bound; final U once. Do not choose by F1 or by a giant AND grid. Keep current AND as a comparator, not as a probabilistic product.
**Effect/test/priority:** Better risk control; calibration alone does not improve a single-model ranking. Use E11 protocol below with fixed policies. **P0/P1**.

#### Q27 — Persistence
**Answer/rationale:** Initial baseline is max-over-windows plus deduplication. Test 2-of-3 windows with one-second stride, or top-two mean, as alternatives. Fix thresholds anew for each aggregator. Hysteresis controls alarm episodes; it need not delay the first trigger.
**Effect/test/priority:** At unchanged threshold, adding a persistence requirement cannot increase recall relative to max. At matched FP budget it may permit a lower threshold and recover recall, but overlapping windows are correlated. Evaluate short/fragmented incidents separately. **P2**.

#### Q28 — Review zone
**Answer/rationale:** Use `no_candidate`, `review_candidate`, `notification` as operational states. Candidate threshold and notification threshold are distinct; candidates still consume resources and must be counted. Low-quality or semantically ambiguous cases route to review, never a confidently innocent label.
**Effect/test/priority:** Can reduce user-facing FP only through a measured verifier; withholding candidates otherwise lowers recall. Report both stage-specific and end-to-end rates. **P1/P2**.

### 16.5 Evaluation and governance

#### Q29 — Protocol/size
**Answer/rationale:** Chronological grouped test across the deployed nine cameras; actor/source grouping for staged data; camera holdout as stress test. Minimum threshold evidence: 2,995 independent negatives at zero exceedances for one-sided 95% control at .1%. Preferred independent test: 15k negative sessions across ≥15 days and ~345 verified incidents where feasible. T and U are separate.
**Effect/test/priority:** More credible intervals, not automatic improvement. Bootstrap days/source groups, with paired model comparison; do not use bootstrap alone for zero-FP certification because it degenerates at zero. **P0**.

#### Q30 — Success
**Answer/rationale:** Final success: point event recall ≥66%, one-sided lower 95% bound ≥60%, one-sided upper FP bound ≤1/1,000 negative sessions, frozen unseen sources, reporting daily exposure. Stronger 75% recall is a stretch. A 50%-recall policy meeting the FP budget is preferable to 75% at 5/day under the owner's stated preference, but does not meet the original recall target.
**Effect/test/priority:** Explicit pass/fail avoids a flattering F1 substitute. Compute bounds on U and fail honestly when underpowered. **P0**.

#### Q31 — Retraining cadence
**Answer/rationale:** Review weekly; retrain every 2–4 weeks only after meaningful new distinct cases, initially ~200 reviewed hard negatives or a new failure cluster. Version source manifest, label policy, preprocessing, model, calibration, aggregation, gate and threshold together. Shadow deploy then compare with incumbent and retain rollback.
**Effect/test/priority:** Can reduce recurring FP but risks forgetting rare theft variants. Require unchanged-budget paired replay and no unexplained camera/day degradation. **P1**.

### 16.7 Data size

#### Q32 — Collection numbers
**Answer/rationale:** Counts are collection hypotheses, not sufficiency guarantees. Start by deduplicating current clips, not declaring them inadequate from their raw count.

| Data purpose | Concrete proposed target |
|---|---|
| Positive training pilot | 500–1,000 independent events if available, ≥200 confirmed real incidents; staged remainder with matched legitimate actions |
| Stronger training coverage | 1,000–2,000 independent events, aim ≥500 real; >150 distinct actors/visits and ≥30 recording/operating days; all nine cameras; one store sufficient for present scope |
| Staged design | ≥30–50 distinct actors, 5–10 paired variations each; expand actor count before hundreds of near-identical repetitions |
| Confusable training negatives | Cat14: 500 distinct target-action episodes; cats3/4/2: 500 each; cats5/9: 300 each; cats7/13: 200 each. Inventory existing independent coverage before collecting the shortfall. |
| Mining | 50k–100k unlabeled windows initially, group/deduplicate; review 200–500 hard candidates plus 200 random episodes per initial cycle |
| Development | Prefer ≥5k independent negative sessions and ≥175 positive events across grouped folds |
| Calibration C | Target ≥5k traffic-representative sessions plus enough reviewed events to fit a simple calibration; log selection weights if enriched |
| Threshold T | Prefer 15k negative sessions; absolute zero-error .1% one-sided bound needs 2,995 for a fixed scorer |
| Frozen U | Another ≥15k negative sessions and ~345 independent true incidents; staged results are supplementary, not replacement for real precision |

Do not require all counts before beginning: run the baseline with available groups and wider intervals. Unknown incident frequency may make hundreds of real thefts impractical in one store; extend time and state uncertainty instead of substituting staged clips silently.
**Effect/test/priority:** Expected diversity/negative coverage benefit, no guaranteed 66–75% at 1/day. Use learning curves to determine whether additional collection helps. **P1**.

#### Q33 — Learning curve
**Answer/rationale:** Nested 25/50/75/100% source-group subsets, three seeds, fixed development folds and policy family. Measure event recall at the same FP risk budget and error overlap. Continue similar-data collection if estimated gain is ≥3 percentage points per doubling and the interval supports useful improvement; if near zero with a narrow interval, switch information/source diversity. An uncertain slope is not proof of saturation.
**Effect/test/priority:** Identifies whether more data changes the frontier rather than just the threshold. Also hold clips constant while varying independent actors/groups. **P1/P2**.

#### Q34 — Real/staged ratio
**Answer/rationale:** Real deployment incidents have higher evaluation value; staged paired recordings offer controlled coverage. Aim for ≥50% real events in the curated positive training pool when obtainable, without discarding useful staged diversity to force a ratio. If real events are scarce, report a separate real validation stratum and cap each staged actor's weight. Use 5–10 variations/actor.
**Effect/test/priority:** May reduce source mismatch; effect depends on quality and coverage. Compare all-current, source-balanced and real-enriched samples at equal number of groups. **P1**.

#### Q35 — Hours
**Answer/rationale:** At 1,000 fragments/day, 15k sessions are about 15 operating days. With illustrative 12 open hours/day this is 180 store-hours or 1,620 camera-hours across nine cameras; not 180 camera-hours. At one window/second, 50k–100k windows are only 13.9–27.8 summed track-hours, but overlapping windows are not independent evidence. Spread mining over ≥7–14 days, all cameras and traffic conditions.
**Effect/test/priority:** Diverse exposure improves FP discovery; estimation requires independent session/day accounting. Record actual hours, sessions, window counts and occupancy. **P0/P1**.

### 16.8 Score space and decision rule

#### Q36 — Score choice and derivation
**Answer:** Threshold the ensemble of raw **binary margins**, with a separate monotone probability map when a probability is useful. Let binary logits be z0,z1 and m=z1−z0. Then `p1=exp(z1)/(exp(z0)+exp(z1))=1/(1+exp(−m))`, so `p1>t ⇔ m>log(t/(1−t))`. Binary argmax corresponds to m>0, subject to explicit tie handling; PyTorch argmax at equal logits returns the first index, so a careless ≥ comparison can disagree at ties.

For multiclass, let r be the actual positive index, `m_r=z_r−logsumexp(z_other)`; then p_r=sigmoid(m_r). Multiclass argmax **does not** mean p_r≥.5: e.g. probabilities .40/.35/.25 select class 1 at .40. Reading raw positive logit alone is unsafe because adding the same constant to all logits changes it without changing class probabilities.

Positive-temperature or increasing Platt transformations preserve binary ranking. Isotonic can create ties; unconstrained nonmonotone calibration can change ranking unpredictably. Energy/max-logit measure something different, are sensitive to logit offsets, and are not guaranteed theft or OOD tests. Entropy can be low for confidently wrong negatives. At .1% FPR, preserving score resolution and validating the negative tail matters more than naming it probability.
**Effect/test/priority:** Single-model margin versus exact sigmoid produces **zero theoretical frontier gain**; gain is possible only from numerical resolution, better threshold search or changed multi-model aggregation. Compare exact decision equivalence on stored logits and count tail ties. **P0**.

#### Q37 — Equivalence and grids
**Answer/rationale:** The probability grid .50:.02:.98 has 25 thresholds. The margin grid −.5:.25:2 has 11. Probability thresholds .90/.92/.94/.96/.98 (five of 25) all exceed the margin grid's maximum sigmoid(2)=.880797. Probability .80 corresponds to 1.386294, .88 to 1.992430, and .98 to 3.891820. Most interior grid points also do not coincide exactly; one grid is not a subset of the other.

Strictly monotone per-model transformations preserve an AND if each threshold is transformed consistently. Even `min(sigmoid(m1),sigmoid(m2))=sigmoid(min(m1,m2))` exactly when the same sigmoid is used: **AND itself does not break equivalence**. Different temperatures require member-specific threshold conversion. Probability averaging generally differs from averaging margins; explicit example: m=(10,−2) gives mean probability ≈.5596 but sigmoid(mean m)=.9820. Multiclass temperature scaling can change the ranking of p_r across examples because the competing logit distribution differs; binary positive-temperature scaling cannot.

Search distinct negative **session aggregate** order statistics, retaining every attainable tail decision and treating ties conservatively. No real tail-logit histogram is supplied, so the number of actual missed operating points/ties is unknown.
**Effect/test/priority:** Removes grid artifacts without inventing new information. Store the full attainable FP/recall curve and tie counts, rather than a fixed 0.02 sweep. **P0**.

#### Q38 — Saturation and tail bounds
**Answer/rationale:** A small numerical check here used float32 sigmoid on integer margins 10…30. Margins 17…30 all became exactly 1.0 in this environment: 14 distinct margins collapsed into one probability value. This is a synthetic precision demonstration, **not a claim about Kanvis logits**; library kernels and dtype can change the onset. Float64 reduces but does not eliminate finite precision.

Use raw margins in at least float32, preferably float64 for offline order-statistic calculations; keep raw scores before probability rounding. Negative quantile selection must be on complete-session statistics and include uncertainty. For a fixed policy and zero observed FP, `p_U=1−.05^(1/n)`; n≥2,995 supports .1%, n≥29,956 supports .01%. After choosing a threshold adaptively, use the order-statistic tolerance procedure in §4, not a naive confidence interval at the chosen minimum-error row.
**Effect/test/priority:** Can restore lost numerical ranking, but cannot guarantee improved recall. Audit the deployed tensor dtype and saved-score precision, not just Python evaluation. **P0**.

#### Q39 — Calibration
**Answer/rationale:** Increasing Platt `q=σ(a s+b), a>0` on separate C is the initial choice because it can correct both slope and offset. Temperature `σ(s/T)` cannot adjust a prior-dependent intercept. Isotonic has too much freedom in a sparse extreme tail; beta calibration is optional, vector scaling only for a justified multiclass task. Fit regularized mappings and report likelihood/Brier score, adaptive-bin reliability and specifically high-score bins with counts. Overall ECE can look good when rare positives are ignored.

Sampler/weighting affects what probability the model learns. C must represent the intended **window** label population if it calibrates window scores; a session risk calibration needs session-aggregate labels. If C is case-control enriched, use known sampling weights or call it case-control calibrated. Do not calibrate window probabilities with session prevalence. Monotone calibration may stabilize reporting, not guarantee a universal threshold across retrainings.
**Effect/test/priority:** No single-scorer ROC gain; possible fusion/interpretation benefit. Cross-validate calibration within C without touching T/U; compare tail reliability and threshold variability across seeds. **P1**.

#### Q40 — Prior shift
**Answer/rationale:** Under pure label shift with invariant class-conditionals and a calibrated reference posterior, Bayes gives `logit p_prod = logit p_ref + logit π_prod−logit π_ref`. The relevant reference prior is the distribution on which the posterior was calibrated, not necessarily the raw training fraction. With ordinary class-only resampling and proper CE it may be the effective sampled prior; hard-negative selection/focal/asymmetric losses generally prevent a single prior correction from repairing the model.

Computed examples: (.15→.001): −5.1722; (.15→.005): −3.5587; (.30→.001): −6.0595; (.30→.005): −4.4460. Thus the brief's “−4.5 to −6” does not cover the stated range. At a production posterior threshold .5, reference margins must exceed the corresponding positive offset, e.g. 5.1722 for .15→.001. A .001 session prevalence cannot be substituted for the window prior, which also depends on event length and the window generator.

Adding a constant changes no ranking. If the margin threshold is refitted to the same FP budget, decisions and FP/session are identical. Keeping .8 fixed after an offset suppresses both FP and recall by an amount requiring real logits; it is not free improvement. Source/quality/time mismatch violates pure label shift [S5].
**Effect/test/priority:** Default **no speculative prior offset** in the rank-based alarm rule. Apply only to justified probability reporting and test class-conditional stability. **P1**.

#### Q41 — Ensemble geometry
**Answer/rationale:** Commit to equal mean seed margins for the first challenger. Min/AND is a veto policy that can protect against a single overconfident member but loses events one member misses. Probability mean bounds each member's influence; mean margins preserves high-tail resolution but can be dominated by a mis-scaled member. Max favors recall and often FP. Rank averaging removes monotone scale mismatch but depends on the reference population and coarse tail ranks. Stacking can exploit complementary errors but needs independent, sufficiently rich labeled data.

For two models with unequal margin scales, fit increasing per-model Platt maps on C before testing their mean log odds, or use separately calibrated thresholds for AND. Do not multiply their probabilities as independent evidence. A seed ensemble does not automatically beat architecture diversity. Compare the current pair and three TCN seeds on the same replay, with paired bootstrap and per-seed residual overlap.
**Effect/test/priority:** Expected stability benefit, unknown recall gain; provisional worthwhile gain ≥2 points at the same FP bound or meaningful variance reduction. **P1/P2**.

#### Q42 — Temporal aggregation and duration
**Answer/rationale:** Baseline max preserves one-window event evidence. Mean can wash out a brief gesture in a long session; top-k needs k fixed; noisy-OR and summed logits double-count overlapping windows. A posterior logit equals an LLR plus a prior term only when calibrated, and summing posteriors repeatedly also repeats that prior. CUSUM/SPRT guarantees require correctly specified conditional likelihoods or valid sequential evidence; ordinary neural scores do not qualify.

Test max against top-two mean and 2-of-3, separately tuned to equal negative-session risk. Two detections one second apart can miss brief fragments and may merely repeat the same overlapping pose evidence. No numerical persistence loss is identifiable without event-score traces.

For independent windows of negative exceedance p, session FPR is `1−(1−p)^K`; for heterogeneous K use `E[1−(1−p)^K]`, not the expression at mean K. Independence is generally false here. Estimate from replay. For duration-aware thresholds, calibrate session maxima in a few prespecified duration bins with shrinkage and adequate support; when making online decisions, use elapsed-duration policies calibrated end-to-end, not unknown final duration. Bonferroni alpha spending is a conservative alternative if each per-look tail p-value is valid, but it can be very insensitive.
**Effect/test/priority:** Session calibration addresses repeated testing; k-of-n is conditional, not guaranteed. Validate 1/4/43-chunk sessions and startup fragments explicitly. **P1/P2**.

#### Q43 — Multiclass indexing
**Answer/rationale:** Read `label_to_idx[positive_class]`, never hard-code column zero. Assert the positive class exists after mapping and test with synthetic one-hot logits for each class. In multiclass p_r and `z_r−logsumexp(z_other)` have the same rank at a fixed temperature; their names do not create different classifiers. Binary with explicit hard-negative coverage is the next baseline, with auxiliary concealment as specified.
**Effect/test/priority:** Wrong indexing could invalidate the multiclass campaign, but is not established without its map. Correct it before any binary-versus-multiclass conclusion; report FP separately for 14/3/4/2/5/9/7/13. **P0**.

#### Q44 — Uncertainty
**Answer/rationale:** Use seed disagreement and quality as review signals, not certified OOD detection. Deep ensembles share blind spots; entropy is low on confidently wrong examples. MC-dropout adds repeated inference and offers no guarantee beyond an already available three-seed ensemble. Distance in an encoder embedding may capture camera novelty instead of theft ambiguity.
**Effect/test/priority:** A useful trial target is ≥10% relative FP reduction at ≤2-point event-recall loss, not a forecast. Measure risk–coverage on all events, including abstentions, and interactions with quality gates. **P2**.

#### Q45 — Frozen risk protocol
**Answer/rationale:** Select model/aggregation on development only; fit mapping on C; choose threshold on T using §4's order statistic; test once on U. If M policies are chosen using the same risk set, either reserve a new certification set or allocate δ/M and correct the selection procedure. Do not apply a pointwise CP interval to the winner of 41,040 rows as if it were prespecified.

Version the exact score tensor dtype, strictness of comparison, mask policy, windowing, quality gates, deduplication and threshold. Confidence is conditional on independent/exchangeable sampling and fixed deployment population; day-level correlation and drift need sensitivity analysis and future monitoring.
**Effect/test/priority:** Controls false claims; may select a more conservative point and lower recall. Test with deliberate configuration mismatch and reject stale thresholds. **P0**.

#### Q46 — Retraining stability
**Answer/rationale:** No representation guarantees identical decisions near a boundary. Compare probability of decision flips across seeds, bootstrap threshold fits, ±1-frame shifts, 12/15-Hz resampling and modest pose noise. Report flips separately for near-threshold and far-from-threshold sessions; average scores over seeds before thresholding. A review band and hysteresis can stabilize user notifications.
**Effect/test/priority:** Seek ≥25% relative reduction in perturbation flips with ≤2-point recall loss at equal FP budget; this is an experiment criterion. Raw-margin thresholding avoids probability saturation but does not remove learned-boundary variance. **P2**.

#### Q47 — Metrics and selection scalar
**Answer/rationale:** Choose the model with highest development **event recall subject to the session/day FP upper-bound constraint**. Tie within non-inferiority tolerance goes to lighter model. Report unique FP/day, false-positive sessions/1,000, all-event recall, gate coverage, delay, conditional Stage-2 retention, low-FPR partial AUC, precision/PR-AUC with explicit prevalence, Brier/tail calibration, seed dispersion, per-category/camera/day/duration slices and duplicate alarms.
**Effect/test/priority:** Avoids optimizing a high-prevalence F1 irrelevant to store operations. PR-AUC and F-beta are diagnostics, not substitutes for the constrained primary score. **P0**.

#### Q48 — Interface
**Answer/rationale:** C2 owns the versioned pose score and window/session aggregation; C3 owns the final verifier fusion and user notification. Emit `model_id`, `label_policy_id`, `positive_class`, `preprocess_hash`, camera/track/fragment IDs, timestamps, observed duration, raw member margins, aggregate score, calibration version and probability semantics, visibility/quality masks, uncertainty, candidate threshold version and decision. Return explicit abstain and reason.

C3 may use margins and pose summaries as features in a held-out trained fusion. It must not add model/SAM3 log odds as independent evidence without conditional calibration: the cues are dependent. Preserve logits privately for audit, not as human-facing certainty.
**Effect/test/priority:** Prevents silent semantic/threshold drift and improves traceability. Golden request through training preprocessing, C2 and C3 verifies identical class and score semantics. **P0/P1**.

#### Q49 — Configuration coupling
**Answer/rationale:** Any change to fps, window size, stride, joint set, normalization, masks, gates, precision, head, ensemble or aggregation invalidates the old calibration/threshold unless parity is explicitly demonstrated. `seq_len=96` alone says nothing about physical time. Bundle all fields with a hash and refuse mismatched deployments.
**Effect/test/priority:** Prevents unexpected FP and recall changes; no automatic model gain. Test rejection of a checkpoint paired with an obsolete preprocessing or threshold bundle. **P0**.

#### Q50 — Committed rule and falsification
**Answer/rationale:** Three-seed mean binary margins → optional increasing Platt mapping on C → maximum over exact fixed windows → one notification/fragment with conservative duplicate suppression → order-statistic threshold on T → single U replay. This is the initial policy; persistence and stacking remain challengers. “Best” remains empirical.

**Operating point:** currently observed 30.6/1,000 negative-fragment approximation with exact iid interval 18.8–46.8; current event recall is unknown. Next target ≤10/day and ≥60%, then ≥73.3% with verifier compatibility, finally ≤1/day end-to-end. No valid predictive CI exists for a model not trained. A hypothetical U result of 10 FP/15,000 and 228 hits/345 incidents would have point FP .667/1,000 and recall 66.1%. Its one-sided95% FP upper bound is1.131/1,000, so it **fails** the proposed risk gate despite its favorable point estimate; its one-sided recall lower bound is61.66%.

**Minimal E11 falsification:** take frozen incumbent and TCN seeds, score the same grouped replay; register mean-margin/max, incumbent AND/max and mean-margin/2-of-3 as just three policies. Choose on development, calibrate/threshold independently, and compare on untouched U with paired day/event resampling. Reject the proposed rule if it misses the risk target, is >2 points inferior to the lightest valid comparator, or its gain disappears on real incidents. **P1/P2**.

### 16.9 Overall verdict and improvements

#### Q51 — Ranked mistakes and ceiling
**Answer/rationale:** Ranked list and confidence are in §§2–3. First fix metric/pool/selection honesty and snapshot correctness; address semantic labels and matched windows before architecture size. The working 20/50/30 modeling/data-information allocation is explicitly a planning opinion, not a fitted decomposition. Unchanged August/October selected counts cannot quantify a ceiling.
**Effect/test/priority:** Protocol fixes may reduce advertised F1 while improving truthfulness; training fixes may move the frontier. Common-pool paired scoring, source-held-out evaluation and an object-context oracle experiment distinguish the competing causes. **P0/P1**.

#### Q52 — Best next recipe
**Answer/rationale:** Use the complete §4.4 configuration, not an unspecified family. Correct the incumbent first to avoid confusing a small architecture with a bug fix. Ablation order: snapshot/pool protocol → time/masks/partial labels → torso/isotropic normalization → features → small TCN/pooling → single transparent sampler → three seeds → optional context. Save every chosen epoch immutably.
**Effect/test/priority:** Worthwhile whole-package target: ≥30% relative FP reduction at ≤2-point recall loss, or ≥5-point recall increase at the same FP budget. These thresholds define useful success, not predicted gains. Repeat three seeds and preserve untouched evaluation. **P1**.

#### Q53 — Light-versus-heavy rule
**Answer/rationale:** Shortlist below. Latencies are deliberately broad **unmeasured engineering envelopes** for batch-one ~96-step input, excluding pose extraction, IPC and Triton queueing; neither CPU type nor runtime was supplied. They cannot support a hardware purchase or SLO.

| Candidate | Approximate parameters | RTX 5090 forward estimate | Small CPU forward estimate |
|---|---:|---:|---:|
| Aggregate features + logistic/300-tree depth-3 booster | ~100 coefficients / ~4.5k tree nodes (not neural parameters) | GPU not needed | .05–3 ms plus feature generation |
| Specified masked dilated TCN, width64 | ~.14M | .2–3 ms | 1–15 ms |
| One-layer bidirectional GRU h64 + masked pooling | ~.05–.08M | .3–5 ms | 1–20 ms |
| Anatomical ST-GCN, four blocks width64 | ~.1–.4M | .3–5 ms | 2–30 ms |
| Two-layer Transformer d64, FF256, four heads, positions/masks | ~.11–.2M | .2–4 ms | 2–30 ms |
| Repaired TCN-LSTM/GRU ensemble, current widths | ~1–3M combined, verify | 1–10 ms | 10–150 ms |

Benchmark instantiated parameter count, warm p50/p95 over ≥1,000 runs, CPU model/threads and runtime, both batch1 and actual serving batch. “Similar” means paired one-sided 95% upper bound on heavy-minus-light recall ≤2 points while both meet the same FP risk budget; if underpowered, report inconclusive rather than declaring equivalence from one standard error. Prefer a ≤2-point measured tradeoff only with explicit experimental uncertainty.

Distillation is worthwhile if a heavy teacher shows a stable ≥3-point advantage or sufficiently lower FP and serving cost matters. Train with supervised labels plus teacher soft targets on development-only data; validate the rare tail independently because distillation can erase it. The current two-family choice is not justified merely by architecture names: compare residual-error complementarity with a three-seed light model.
**Effect/test/priority:** Primary benefit is cost/stability; accuracy improvement unknown. **P2**.

#### Q54 — Epochs and context
**Answer/rationale:** Cannot infer under/overtraining from 40–60 epochs. H1 means chosen versus saved epochs differ; it does not prove late epochs are worse. Plot train loss and grouped operational curves with true saved snapshots. The TCN-RNN's recurrent component already covers full input despite five-frame conv RF. Train to 100 with warm-up/cosine/early stopping as specified; extend to 150 only if the curve is still improving under the fixed development rule.

Test 4/8/12 seconds at 12 Hz =48/96/144 steps, then one 2s+8s dual-resolution candidate if error review shows brief gestures diluted by longer windows. Increasing seq_len without specifying seconds is rejected.
**Effect/test/priority:** Could help or hurt; a planning range of 0–5 recall points at matched FP is a hypothesis, not supported forecast. Compare learning curves and matched temporal slices. **P1/P2**.

#### Q55 — At least 15 filters
**Answer:** The ranked 20-item table in §7 is the complete answer, with FP/recall hypotheses, implementation cost, label limits and validation for each. Stack cheap deduplication/quality representation first, then soft context/uncertainty, then validated temporal aggregation; employee/time rules never hard-suppress by default.
**Effect/test/priority:** Filters overlap, so no multiplication of their individual FP reductions. The only defensible combined point is measured on the full frozen replay. Conditional 10/day×.1 and .7333×.9 =1/day and .66 illustrates the requirement, not a forecast. **P2**.

#### Q56 — Phone and ambiguous labels
**Answer/rationale:** Preserve original cat6 but add verified legitimate/confirmed theft/unknown status plus concealment motion. Legitimate phone examples are theft-negatives and concealment-positives; unknown objects are unknown for theft. Avoid splitting into many front/back/pocket/kneeling subclasses: that adds annotation cost without supplying ownership. Separate a motion head from the business target.

If a fraction f of production FP are legitimate concealment and fraction r of those can actually be distinguished with the available evidence, maximum direct FP reduction from resolving them is f×r, not f. For identical pose distributions r may be near zero at unchanged theft recall. Review 200 positive clips stratified by source/camera and every reviewed alarm first; boundary-only review misses confidently mislabeled examples.
**Effect/test/priority:** Fundamental semantic repair, unquantifiable gain before auditing f and separability. Test matched phone/product actions and evaluate the theft and motion heads separately. **P0/P1**.

#### Q57 — Fragments and crop labels
**Answer/rationale:** Randomize action placement in 8s windows, mask 1–8s startup/end fragments, simulate .1–.5s gaps and joint occlusion, and use coherent time resampling. Concatenate context only from the same source/actor timeline where available; arbitrary skeleton stitching creates physically implausible jumps. Keep timestamps and labels for the retained interval.

Approach-only/aftermath-only positive-parent crops are **unknown**, not theft-positive or automatically theft-negative. First-half concealment crops need interval review or positive-bag membership without an individual positive target. Use MIL as §4; a truncated bag with no verified evidence cannot inherit a positive label. Negative sources stay negative only if reviewed label meaning supports that crop.
**Effect/test/priority:** Experimental target ≥20% relative FP reduction on arbitrary production windows at ≤2-point all-event recall loss; improvements in short fragments assessed separately. No-history replay at every possible event cut is the decisive test. **P1**.

#### Q58 — Active learning cycle
**Answer/rationale:** Every week deduplicate mined tracks and sample ~40% reviewed alarms, 25% near-threshold nonalarms, 15% seed disagreement, 20% uniform background, stratified across cameras/days. These percentages are review allocation, not unbiased traffic sampling. Record inclusion probability for population estimates. Unknown cases get second review; seed a retained positive regression set with hard real incidents.

Initial cycle 200–500 distinct reviewed hard negatives plus 200 random background episodes; supplement scarce class14 and real incidents. Training uses the single 25/25/50 positive/hard/ordinary sampler, with a hard-negative cap per source and no extra ×3 UID boost. Fix benchmark and hold out future days; camera balancing is useful, but current same-store deployment does not require removing all camera information.

First-cycle 10–30% relative FP reduction is a **planning hypothesis**, not an expected measured result; later cycles can plateau or reverse. The actual number needed could be far above 500 if failures are diverse. Continue if ≥20% relative FP reduction or ≥3-point recall gain at equal budget survives unseen days. SSL is deferred until this simpler loop fails and sufficient unlabeled diversity exists.
**Effect/test/priority:** Test old/new models on the same untouched subsequent days and preserve all-event recall. **P1**.

#### Q59 — Single-store evaluation and model sharing
**Answer/rationale:** Joint model across nine cameras, no camera embedding initially; chronological grouped holdout as primary and leave-one-camera-out as a stress test. Maintain positive groups across camera views of the same incident. Use source/time/actor-connected groups and an embargo. Camera-specific calibration needs tail support and shrinkage; separate per-camera neural models fragment scarce positives.

No honest numeric prediction can be made for how much of current 76% F1 survives. F1 changes with prior: if recall=.6378 and per-session FPR=.006794 at prevalence .001, precision is about 8.59% and F1 about 15.1%; at illustrative session FPR=.03087 precision is about 2.03% and F1 about 3.94%. These are **prevalence illustrations**, not measured model deterioration or predicted grouped-test F1.

With clip FPR=.006794 and illustrative K_eff=4.6, session risk is .030874, or 30.87/day at 1,000 sessions. Matching 20/654 gives K_eff≈4.556; this is a fitted arithmetic explanation, not a measured independent window count or proof that domain shift contributes zero. Use actual per-session score traces and daily alarm counts.
**Effect/test/priority:** Honest measurement first; joint model maximizes data sharing. Camera-adversarial training is optional only after matched camera-label coverage exists and nuisance tests show a problem. **P0/P1**.

#### Q60 — Utility and frontier table
**Answer/rationale:** Let λ be true theft events/day and costs c_FN,c_FP. Expected cost is `c_FN λ(1−R)+c_FP F`. Preferring (R=.50,F=0) to (.75,5) gives `.50λ c_FN < .25λ c_FN+5c_FP`, so `c_FP/c_FN > .05λ`, or `c_FN/c_FP <20/λ`. This is a lower bound, not “FP costs 20 times FN.” λ and relative preferences among other points are unknown; use constrained risk minimization.

The table below uses supplied approximate recall bands and **C** exact two-sided CP intervals for FP counts out of 1,619 nominally independent negatives. The daily conversion assumes 1,000 negative sessions and K_eff=4.6; transformed intervals condition on that fixed K and **are not production confidence intervals**. Correlation, model-selection bias, uncertain K and changed units make real uncertainty larger. Recall intervals are rounded iid illustrations using ~323 positives; the exact hits for four frontier bands were not supplied.

| Supplied FP | Supplied clip recall | Illustrative recall 95% range | Clip FP%, exact 95% CI | Daily FP if one look | Daily FP at K_eff4.6 (transformed CI) |
|---:|---:|---:|---:|---:|---:|
| 3 | 42–48% | ~36.6–53.6% | .185 (.038–.541) | 1.85 | 8.50 (1.76–24.62) |
| 8 | ~58% | ~52.5–63.5% | .494 (.214–.971) | 4.94 | 22.53 (9.79–43.91) |
| 11 | 63.78% | 58.27–69.03% exact | .679 (.340–1.212) | 6.79 | 30.87 (15.53–54.57) |
| 20 | 72–74% | ~66.8–78.8% | 1.235 (.756–1.901) | 12.35 | 55.58 (34.31–84.52) |
| 45 | 83–85% | ~78.5–89.0% | 2.779 (2.034–3.702) | 27.79 | 121.61 (90.22–159.29) |

**No supplied point meets 1 FP/day**, even under the favorable one-look approximation. At K_eff4.6 the required per-look p is `1−.999^(1/4.6)≈.0002175`, or .02175%; fewer than .36 expected FP in 1,619 clips. The optimal feasible point cannot be recovered from this coarse frontier: the zero/one/two-FP thresholds and their recalls are absent. The always-no-alarm rule is feasible with zero recall; any useful lower-FP point requires new replay evidence. At an exploratory 10/day budget the 3-FP point is the only listed point below budget in this conversion, with 42–48% **clip**, not event, recall.

Now: no demonstrated final feasible policy. After correctness/data fixes: pursue ≤10/day at ≥60% event recall and then ≥73.3%, without claiming achievability. After added context and a measured verifier: 1/day with 66% is a conditional engineering goal. Neither a confidence interval nor an optimum can be fabricated for those untrained variants.
**Effect/test/priority:** Select the constrained point on actual held-out session scores, including zero exceedances and their statistical bound. **P0**.

#### Q61 — What moves the frontier
**Answer/rationale:** Ranked gain/cost table below; gains remain hypotheses.

| Rank | Intervention | Frontier or threshold | Cheapest decisive experiment |
|---:|---|---|---|
| 1 | Label semantics + target identity cleanup | Can move frontier by removing contradictory supervision; may expose irreducible overlap | Review paired phone/product examples, track-label audit, retrain one fixed model |
| 2 | Reviewed hard negatives, especially14/3/4/2/5/9 | Frontier through coverage, not merely lower threshold | Add 200–500 unique cases, fixed positive regression set, unseen-day replay |
| 3 | Matched temporal windows/partial supervision | Frontier under deployment distribution | Repaired incumbent with identical time contract, boundary-event evaluation |
| 4 | More independent real incidents plus paired staged controls | Frontier through diversity | Grouped learning curves and real-only blind slice |
| 5 | Existing camera/ROI/quality context | New contextual information, possible shortcut | Pose-versus-pose+context paired ablation on future days |
| 6 | Temporal labels | Better supervision using same information | Localize 200 events and compare weak versus interval supervision |
| 7 | Full17 joints (head/knees/ankles) | Adds posture/orientation information; not ownership | Matched 8-versus17 re-extraction, same event IDs |
| 8 | Per-wrist image/object-state embeddings | Potentially important new object information | Human oracle: annotate object possession before/after on 200 TP/FP cases, simulate fusion ceiling, then train a small crop encoder only if useful |
| 9 | Hand/finger keypoints | New articulation, often low quality at distance | Inspect hand pixel size/visibility first; oracle finger visibility versus failures |
| 10 | Optical flow near wrists | Some new motion/appearance information, overlaps pose | Paired crop-flow ablation on visibility-stratified cases; no assumed object ownership |
| 11 | More/smaller architecture capacity | May improve approximation/regularization, not input information | Repaired TCN/GRU/ST-GCN shortlist, same groups/seeds/budget |
| 12 | Calibration and threshold shifts alone | Position along a single-model frontier | Verify identical score ordering and compare exactly matched thresholds |

The oracle study is an upper-bound diagnostic, not deployable performance. Object visibility may still leave ownership unknown; a phone-like object can be merchandise. For all new visual features, evaluate on independent incidents and keep pose-only and fused results separate.
**Effect/test/priority:** Spend first on labels, production negatives and time parity; defer large models until those are controlled. **P1–P3**.

## 6. Assessment of H1–H5, M1–M10 and E0–E12

Costs below are engineering effort estimates: small = hours to ~1 day; medium = ~2–5 days; large = >1 week or substantial data collection. They exclude unavailable infrastructure and are not quotations. Expected effects are hypotheses, except arithmetic/mechanical corrections.

### 6.1 Findings

| Finding | Decision | Cost | Expected effect and acceptance check |
|---|---|---|---|
| H1 snapshot | Accept mechanism; confirm occurrence | Small | Save the selected epoch instead of mutable last weights; benefit may be zero if last epoch is best. Test known weight snapshots and optimizer continuation. |
| H2 grouped leakage | Accept risk, remeasure October | Medium | More honest evaluation, possibly lower reported recall/higher FP. Require no source/event/overlap-component crossing. |
| H3 selection | Accept, raise to P0 | Medium | Eliminate test retuning and winner's-curse claims. Frozen apply-only evaluator must not call best-row search. |
| H4 time | Accept | Medium | Coherent timing and boundary robustness; plausible material FP reduction, unknown magnitude. Golden arrays and arbitrary-fragment replay. |
| H5 augmentation | Modify severity | Small | Null shift/scale removal mostly saves effort; new valid augmentation may help. Not established as a principal ceiling cause. |
| M1 architecture | Accept specific defects; reject blanket claims | Medium | Fix reverse pooling, masks, graph and positions; entire RNN context is not five frames. Equal-protocol comparison. |
| M2 feature scales | Modify | Small | Better conditioning possible, no proof current model ignores velocity. Continuous-channel versus no-scaling ablation. |
| M3 normalization | Accept as experiment | Medium | More consistent body geometry; torso anchors can fail too. Report sideways/kneeling/missing-hip slices. |
| M4 quality | Modify strongly | Medium | Prefer masks/soft quality over wholesale gate; accept hard rejection only for structurally unusable input. All-event denominators mandatory. |
| M5 optimizer | Accept omissions, not diagnosis of failure | Small | Schedule/WD/clipping improve controlled optimization; constant Adam is not intrinsically wrong. Learning curves decide epochs. |
| M6 loss/sampler | Accept simplification | Small | Easier attribution/calibration; asymmetric loss is not inherently invalid. Compare proper held-out operating curves. |
| M7 rare maps | Reject October employee-removal assertion; retain general risk | Small | October table retains7/13/14. Inventory actual class maps and never remove useful binary negatives for multiclass count reasons. |
| M8 nuisance | Accept as hypothesis | Medium | Source quality matching may improve generalization, but nuisance correlation alone is not causal proof. Matched-source tests. |
| M9 score/calibration | Accept procedural defects, correct mathematics | Small/medium | Matching thresholds removes grid artifacts; no inherent ROC improvement from monotone scaling. Calibrate and certify independently. |
| M10 pool | Accept as high-priority unresolved issue | Small inventory; medium label recovery | 433 negatives excluded versus one positive; measure coverage and target identity before broadening data. |

### 6.2 Experiment plan

Every comparison uses identical frozen group definitions and source-provenance records. Three seeds for finalist training, paired event/day resampling, tail-count reporting, and a fixed primary metric. If testing many hypotheses, use development folds and one final certification, rather than repeatedly opening the test.

| Experiment | Decision | Revised exact intervention / expected information | Cost |
|---|---|---|---|
| E0 | Modify | Fix clone and record all epochs, but select by constrained event recall rather than PR-AUC alone. Compare chosen-epoch/last-epoch predictions on development. | Small + two incumbent retrains |
| E1 | Accept first | Grouped chronological replay, all target tracks, no test tuning; risk-set/final-test separation. Establish honest incumbent. | Medium |
| E2 | Accept | Nuisance-only metadata model and pose-summary logistic/GBDT baseline. Same groups. Strong baseline is diagnostic, not proof that temporal information is useless. | Small |
| E3 | Modify strongly | Empty-input reject versus soft quality versus proposed hard gate. Include excluded events as misses and count conditional coverage. Reject gate if apparent gains come from denominator removal. | Medium |
| E4 | Modify | 8s/12Hz first, 4s and12s follow; random offsets and short fragments; do not propagate positive labels into evidence-free crops. | Medium |
| E5 | Accept modified ops | Remove null transforms; correlated body-unit noise, joint/gap dropout, coherent speed; one factor at a time. Hypothesis: improve robustness without class-correlated artifacts. | Medium |
| E6 | Accept | Isotropic coordinates, torso anchor and scale; separate optional global context. Factorial four-cell study sufficient initially. | Medium |
| E7 | Accept | Validity/physical velocity/distances first; angles second, acceleration last. Observe tail FP, not only average loss. | Small/medium |
| E8 | Modify scope | Repaired incumbent, small TCN, small GRU; ST-GCN/Transformer only after those. Architecture rankings from the prior protocol are provisional. | Medium/large |
| E9 | Accept | Ordinary BCE with one sampler versus focal versus asymmetric, no stacked mechanisms; calibrate each independently. Add hard-negative loop as a separate factor. | Medium |
| E10 | Accept modified rule | Three-seed mean margins versus incumbent AND and best single light model. Use paired non-inferiority and error complementarity, not exhaustive triples. | Medium |
| E11 | Accept with restricted primary policies | Check binary mathematical equivalence first; register three primary aggregate policies as Q50. Expand only if a concrete failure is diagnosed. | Small scoring; medium replay |
| E12 | Accept with honest label limits | Frozen model, one filter at a time, then cumulative replay; audit retained and suppressed alarms, sampled nonalarms and verified incidents. No claim of recall from unlabeled video alone. | Medium |

Recommended execution order: E1 and E0 correctness → E2/E4/E6/E7 → E5/E9 → E8/E10 → E11 certification → E12 cumulative filters. This does not mean waiting to inspect score semantics until late: mapping/threshold correctness is P0, while advanced aggregation comparisons occur after the input/model is stable.

## 7. Additional proposals and 20 concrete filters

### 7.1 How to interpret the estimates

The table's reductions are **hypothetical pilot ranges relative to the incoming FP stream**, at a fixed policy before threshold refitting. They are not measured forecasts and need not hold together. “Loss” is the fraction of previously detected true incidents suppressed, not percentage points of population recall. Where a signal is not already available (e.g. ROI map), creating it is an explicit prerequisite. All ideas respect the generalist requirement unless a caution is stated; none requires labeling specific ordinary gestures.

Unlabeled replay can measure trigger suppression, duplicate counts, coverage and computational cost. It **cannot measure true FP reduction or robbery loss** without labels. For every experiment: review a stratified sample of retained and suppressed alarms, retain sampling probabilities, audit random nonalarms, and replay all independently confirmed incidents. The latter estimates incident recall only for the incident sampling frame, which must be disclosed. Reviewed existing alarms alone cannot reveal thefts the incumbent never detected.

| Rank / idea | Mechanism and generalism | Hypothetical FP reduction / true-detection loss | Cost | Cheapest validation |
|---|---|---|---|---|
| 1. Exact duplicate window deduplication | Same camera/track/time evidence yields one candidate; universally applicable | Notification FP0–20%; unique-session FP0%; loss0% for exact duplicates | Small | Compare duplicate keys and unique event IDs; ensure only exact repeated evidence is removed |
| 2. Episode cap and cool-down | One notification per actor episode; hysteresis ends an episode after quiet period; 30s cool-down initial trial | Notification FP0–40%; first-session FP0%; loss0–5% if a second true incident occurs | Small | Count repeat episodes and test multiple distinct incidents/actor; do not hide first-alarm risk |
| 3. Structurally invalid input abstention | No usable joints/timestamps/anchor returns no decision; same train/inference policy | FP0–10%; loss0–3% | Small | Inventory rejection reasons, all-event denominator, inspect rejected positives |
| 4. Mask/visibility soft features | Learn missingness without treating zero-filled missing joints as geometry | FP0–20%; loss0–5% | Medium retrain | Replay by wrist/hip visibility; compare with same windows without quality features |
| 5. Two-of-three persistence | Need two window exceedances at one-second stride; no named-action rules | FP0–40%; loss0–25%, especially brief fragments | Small | Use real traces and short-event slice; correlation prevents p² reasoning |
| 6. Top-two score mean | Two highest scores in a local 3s episode, rather than full-session mean | FP0–30%; loss0–15% | Small | Compare equal-risk thresholds; include one-window-only events |
| 7. Hysteresis | High trigger, lower continuation threshold, e.g. development-calibrated quantiles rather than arbitrary .8/.5 | Repeated FP0–30%; first-alarm FP approximately0%; loss0–5% from overlong merges | Small | Compare episode splitting/merging and first trigger times |
| 8. Conservative fragment chaining | Same-camera spatial/velocity continuity, short gap, unambiguous match; use initially for duplicate suppression | Notification FP0–25%; loss0–5% from wrong associations | Medium | Manually audit 200 candidate links and nonlinks; ambiguous matches remain separate; do not require linking for classification |
| 9. Duration-aware risk calibration | A few session-length/elapsed-time bins or validated sequential policy | FP0–35% among long tracks; loss0–15% | Medium | Report 1/4/43-chunk bins, use enough negative tail data and compare overall budget |
| 10. Correlation-aware evidence accumulator | Bounded score accumulation using learned conditional/sequential features; no naive sum of overlapping posterior logits | FP0–30%; loss0–15% | Medium | Fit only on development, preserve gaps, test time-shuffled and duplicate-window stress cases |
| 11. Soft ROI context | Shelf/cashier/entry proximity as contextual feature; no absolute forbidden zones | FP0–30%; loss0–10% | Medium (map creation if absent) | Matched actions inside/outside ROIs, future days and ROI-version checks |
| 12. Per-camera calibration with shrinkage | Pool information across9 cameras, avoid unsupported independent .1% quantiles | FP0–20%; loss0–10% | Medium | Camera tail sample sizes, held-out-day per-camera results and pooled risk; no camera excluded |
| 13. Kinematic soft summaries | Wrist–torso approach, motion amplitude, path coherence and joint validity as features | FP0–20%; loss0–10% | Medium | Kneeling/behind-body/slow-event slices; prohibit hard pocket-region veto |
| 14. Crowding/occlusion context | Other existing tracks nearby and visibility change; route ambiguous evidence to review | FP0–20%; loss0–15% | Medium | Crowded-scene target labels and negative coverage; no “crowded=ignore” rule |
| 15. Employee-like context, never hard suppression | Long dwell/repeated trajectories/category7/13 patterns adjust review priority, not immunity | FP0–30%; loss0–20% if misused; soft feature target loss≤2% | Medium | Include customer dwell and employee theft/unknown-role incidents; if no such examples, disallow hard suppression |
| 16. Time/store-state soft priors | Opening/closing/day-of-week features with shrinkage; no silent alarms-off schedule | FP0–15%; loss0–10% | Small/medium | Held-out weeks, atypical-hour events and schedule drift; source-label confounding check |
| 17. Seed disagreement abstention | High mean with high variance is reviewed instead of automatically notified | FP0–25%; loss0–15% | Small with ensemble | Risk–coverage and event denominator, compare confident FP that survive |
| 18. Shift/jitter consistency | Rescore ±1-frame boundary shift and one mild perturbation; aggregate instead of veto if practical | FP0–20%; loss0–10% | Medium,2–3× score calls | Paired true events and arbitrary boundaries; reject if short-fragment recall suffers |
| 19. Embedding distance / support | Low support relative to training groups becomes a review flag | FP0–20%; loss0–20% | Medium | Camera/source-stratified risk–coverage; novel true incidents must not disappear from denominator |
| 20. Rich Stage-2 handoff + reviewed mining | Pass margins, poses, quality and episode history; learn joint fusion when labels exist; feed reviewed residuals to next training cycle | Fusion: unknown until verified; mining pilot10–30% FP reduction, loss0–5% | Medium/large | Held-out conditional candidate cohort, then full end-to-end replay; measure correlated verifier errors |

**Recommended cumulative order:** exact deduplication → explicit invalid-input handling and soft masks → correct session maxima and risk threshold → conservative episode management → optional soft ROI/camera context → uncertainty/consistency → persistence only if it beats max at equal risk → Stage-2 fusion. Employee/time priors are low-priority soft signals. Conservative fragment chaining is optional and may be deferred with the owner's tracker scope.

**Combined point:** The current 30.6/day reference is not converted into a promised 1/day by multiplying these ranges. For planning, getting to10/day requires a 67.3% reduction from30.6, and getting to1/day requires96.7%. Those are large reductions. If a pre-verifier stack produces8 FP/day and .74 event recall, then a verifier retaining10% of its FP and90% of its TPs would produce .8 FP/day and .666 recall; both conditional retentions must be measured on that exact stack. A purely suppressive postprocessor cannot increase the detected-event set at a fixed upstream policy.

### 7.2 Additional experiments beyond the original plan

1. **Matched semantic ambiguity study:** paired actions by the same actor/camera/time with phone, purchased merchandise and concealed merchandise. Blind pose-only review versus video/object-state review. Estimate how often pose evidence is genuinely indistinguishable. This distinguishes label overlap from optimization failure without assuming either.
2. **Error-set overlap:** compute Jaccard overlap and paired discordance of August/October/current/challenger false alarms and missed events on one common pool. Identical counts with different examples can still justify a better ensemble.
3. **Chronological label audit:** sample beyond triggered alarms, including random periods and high-score nonalarms. This addresses verification bias and estimates missing incidents; report sampling weights and uncertainty.
4. **Window-boundary stress suite:** sweep every event start offset across the eight-second grid, test truncated starts/ends and gap locations. Report worst-decile event recall and average, rather than only centered windows.
5. **Exposure-aware monitoring:** daily camera uptime, sessions/fragments, windows, alarm episodes and track-length distribution. A rising FP/day caused by twice the exposure differs from a rising per-session risk.
6. **Class-index metamorphic checks:** add a constant to every logit (predictions unchanged), swap class labels plus logits consistently (semantic predictions unchanged), convert probability thresholds to margins (same decisions apart from declared ties).
7. **No-motion/time-order controls:** compare original, reversed and shuffled valid-time sequences on development. If shuffling barely changes results, investigate shortcuts or static pose dependence, but do not infer useless time from a single control.
8. **Oracle contextual upper bound:** supply human object-state labels to a shallow fusion on development and test independently. If even oracle state adds little, expensive image models are unlikely to solve the remaining errors; if it adds much, prioritize small crop features over a larger pose Transformer.
9. **Uncertainty-aware model promotion:** a single narrow test set should not force a winner. Use a predeclared non-inferiority tolerance and promote the light incumbent when challengers have no demonstrated meaningful gain.

## 8. Assumptions and information gaps

No user response is required. Each missing fact has an adopted assumption, conditional action and cheapest verification.

| # | Assumption / gap | Conditional recommendation | Cheapest verification |
|---:|---|---|---|
| 1 | Repository implements snippets as quoted | If changed, reevaluate H1/M1/H3 status; retain generic checks | Inspect cited functions/commit and snapshot unit test |
| 2 | 20 FP means distinct false-alarm fragments | If raw alarms, recompute session and notification rates separately | Review one-day alarm IDs and counts |
| 3 | 1,000 sessions/day uses the same fragment definition as654 | If visits instead, reconstruct both exposure denominators | Daily counts of raw IDs and merged visits |
| 4 | October labels include legitimate phone concealment | If already semantically split, preserve that policy and retrain appropriate heads | Stratified200-positive review and label schema |
| 5 | Positive independence is unknown | If few groups, prioritize collection and wider CIs; no claim from thousands of clips | Source-event deduplication manifest |
| 6 | Some confirmed theft labels can be recovered | If not, train concealment candidates and refrain from calibrated theft claims | Review original event evidence/metadata |
| 7 | Timestamps/frame indices recoverable | If absent, use verified source fps; unknown timing is quarantined | Inspect array length, metadata fps and extraction frame indices |
| 8 | Per-keypoint confidence may not exist | Use visibility derived from NaN/masks; add true confidence only after export | Schema inventory, no guessed confidence values |
| 9 | Multi-person target identities missing in some clips | Quarantine target-unknown cases; never duplicate positive labels across users | `user_cat`/target-track mapping audit |
| 10 | Old quality statistics not representative | Do not adopt .70-confidence/36-frame/50%-visibility gates from old retention figures | Recompute current per-class distributions |
| 11 | Stage2 performance is unverified | Candidate-only operation until conditional FP/TP retention is measured | Replay reviewed candidates through actual verifier |
| 12 | Traffic stable enough for threshold exchangeability | If camera/day drift appears, recertify and use cautious shadow mode; no automatic score normalization from unknown contaminated traffic | Day/camera tail plots and recent adjudicated sample |
| 13 | Actual robbery prevalence unknown | Use constrained risk objective; prior correction only after justified estimation | Random sampled production review plus incident log |
| 14 | Latency unrestricted; serving cost still matters | Prefer compact model unless heavy model's low-FPR gain is demonstrated | Measured p50/p95 and peak memory on actual devices |
| 15 | ROI maps not yet verified | Pose-only baseline first; build simple versioned polygons before context experiment | Inspect9 camera views/maps once |
| 16 | Existing validation used repeatedly | Treat it as development, not final evidence; create a future chronological U | Campaign history and threshold-selection logs |
| 17 | Confirmed real incidents may be too few for345 | Publish wide real-only CIs and supplementary staged stress tests; never present staged recall as real incident recall | Independent incident inventory |
| 18 | No repository/data available in this workspace task | Deliver exact protocols and clearly labeled calculations, not fabricated training outcomes | Supplied attachment is the only inspected project artifact |

## 9. Appendix: calculations, reproducibility and sources

### 9.1 What was actually inspected and executed

Inspected the complete 828-line supplied `upload/TRAINING_AUDIT.md` in sequential sections, including C.6, findings, filters, code excerpts, all61 questions, experiment plan and annexes. No referenced repository file or campaign artifact was available for direct inspection. The line pointers below are reproduced from its Annex B for the owner's implementation work, not claimed direct evidence:

- `train_model_operations.py:1600/1614/1622` normalization/velocity/resize; `1707–1780` item preprocessing.
- `train_model_operations.py:2606` split; `2749` binary remap; `2986` evaluation; `3741` experiment; `3926–3931` snapshot.
- `campaign/evaluate_campaign.py:127–129` positive-column assumption; `282–315` selectors; `388` ensemble grid; `598/624` split-specific selection according to the audit.
- `campaign/train_campaign.py` augmentation option propagation; `campaign/calibration_utils.py` temperature and threshold logic.

Executed local Python arithmetic with NumPy and SciPy: category total, confusion metrics, exact binomial intervals, effective-look conversion, one-sided zero-error sample sizes, stage-retention arithmetic, prior offsets and a float32 sigmoid saturation demonstration. PyTorch was unavailable in the local Python environment, so **the H1 minimal tensor experiment was not executed here**. Its general behavior was verified against official documentation. No Kanvis training, real-logit experiment, deployment benchmark or actual model comparison was run.

### 9.2 Reproducible arithmetic

The following complete script reproduces the central calculations; it is included in this file so no separate script is needed.

```python
import math
import numpy as np
from scipy.stats import beta, binom

def cp_two_sided(k, n, confidence=.95):
    a = (1-confidence)/2
    return (0. if k == 0 else beta.ppf(a, k, n-k+1),
            1. if k == n else beta.ppf(1-a, k+1, n-k))

def cp_upper(k, n, confidence=.95):
    return 1. if k == n else beta.ppf(confidence, k+1, n-k)

tp, fn, fp, tn = 206, 117, 11, 1608
recall = tp/(tp+fn)
precision = tp/(tp+fp)
print('R/P/F1', recall, precision, 2*tp/(2*tp+fp+fn))
print('R interval', cp_two_sided(tp, tp+fn))
for k in (0, 3, 8, 11, 20, 45):
    p=k/1619
    lo, hi=cp_two_sided(k,1619)
    project=lambda x: 1000*(1-(1-x)**4.6)
    print('frontier',k,p,(lo,hi),project(p),(project(lo),project(hi)))
print('production',20/654,cp_two_sided(20,654))
print('effective K',math.log1p(-20/654)/math.log1p(-11/1619))
for alpha in (.001,.0001):
    print('zero-FP minimum n',alpha,math.ceil(math.log(.05)/math.log1p(-alpha)))
print('stage1 recall needed',.66/.9,.75/.9)
print('look risk needed',1-.999**(1/4.6))
for reference in (.15,.30):
    for production in (.001,.005):
        logit=lambda p: math.log(p/(1-p))
        print('prior offset',reference,production,logit(production)-logit(reference))
m=np.arange(10,31,dtype=np.float32)
p=np.float32(1)/(np.float32(1)+np.exp(-m))
print('synthetic float32',list(zip(m.tolist(),p.tolist())))
print('saturated count',int(np.sum(p==1)))
# For a FIXED scorer, choose the rank on independent negative-session scores:
n,alpha,delta=15000,.001,.05
eligible=[k for k in range(1,n+1) if binom.cdf(k-1,n,1-alpha)>=1-delta]
print('risk-threshold rank',min(eligible) if eligible else 'insufficient data')
print('hypothetical test10/15000 upper per1000',1000*cp_upper(10,15000))
```

### 9.3 Additional statistical cautions

- **Conditional cascade recall:** R_final=R1×P(Stage2 hit | Stage1 hit, true incident). Using a verifier recall measured on all incidents or on a different Stage1 is invalid.
- **Gate cost:** old-sample retention .831 implies that even perfect conditional recall after the gate cannot exceed83.1% overall under that sample's population; .66 conditional recall would become .5485. This is an illustration, not a current estimate.
- **Zero FP is not zero risk:** at n=1,619 with zero errors, the one-sided95% upper rate is about .185% (about1.85/1,000 for one look). Even this cannot certify .1%. The two-sided upper interval is larger, about .228%.
- **Paired recall power:** for an improvement Δ=.05 and paired discordance d, rough required n≈(1.96+.84)²(d−Δ²)/Δ². At d=.10/.20/.30 this gives approximately306/620/934 independent events. Thus300–400 is conditional, not universal. Grouping reduces effective n.
- **Sample width:** .1%±.05% normal approximation requires about15,351 negatives, not exactly15,000; 1%±.3% about4,226. Exact/clustered designs can require more. Recall .66±.05 gives about345 under a normal iid planning calculation.
- **Calibration is not certification:** a model may be well calibrated on average and still violate a rare false-alarm budget; a risk-controlled threshold need not output calibrated probabilities.
- **Fixed total versus daily variability:** a 1/day average does not mean each day has at most one false alarm. With an illustrative Poisson mean1, P(two or more)=1−2/e≈26.4%; actual counts may be overdispersed. Report bursts and daily quantiles.
- **Negative traffic contamination:** unlabeled “normal” replay may contain undetected theft. Calling every trigger false gives a conservative alarm-count estimate but not a validated FP estimate; assuming every nontrigger negative inflates certainty about recall.

### 9.4 Primary references consulted

These sources support general mechanisms; none demonstrates a Kanvis-specific gain. Consulted 2026-10-15. Links are included for a portable Markdown artifact.

- **[S1] PyTorch, Saving and Loading Models.** Official warning on mutable best-state snapshots and inference mode. https://docs.pytorch.org/tutorials/beginner/saving_loading_models
- **[S2] PyTorch, LSTM.** Official distinction between last output and final forward/reverse hidden states. https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- **[S3] Guo et al. (2017), On Calibration of Modern Neural Networks, ICML.** Temperature calibration is an empirical method, not rare-tail certification. https://proceedings.mlr.press/v70/guo17a.html
- **[S4] Tong, Xia, Wang and Feng (2020), Neyman–Pearson classification: parametrics and sample size requirement, JMLR21(12).** False-positive-constrained classification and the importance of finite-sample requirements. https://www.jmlr.org/papers/v21/18-577.html
- **[S5] Lipton, Wang and Smola (2018), Detecting and Correcting for Label Shift with Black Box Predictors, ICML.** Label shift requires invariant class-conditional distributions; it is not generic domain-shift correction. https://proceedings.mlr.press/v80/lipton18a.html
- **[S6] Yan, Xiong and Lin (2018), Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition, AAAI.** Anatomical spatiotemporal graph modeling is a credible comparator, not evidence that it will beat a small TCN here. https://ojs.aaai.org/index.php/aaai/article/view/12328

### 9.5 Completion and artifact location

All questions Q1–Q61 are answered, including conditional branches for unavailable facts. H1–H5, M1–M10 and E0–E12 are assessed. No question is returned to the owner. No measured production improvement is claimed. Proposed sample sizes, training configuration, filters and pass/fail criteria are reviewable specifications for the next campaign.

Response file: `/workspace/scratch/4c7cba3540c0/IAModel/docs/TRAINING_EXPERT_RESPONSE_20261015.md`.


