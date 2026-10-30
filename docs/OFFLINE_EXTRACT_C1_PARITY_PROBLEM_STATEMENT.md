# Problem statement: offline batch pose extraction — C1-equivalent outputs at maximum throughput

**Audience:** external agent / architect (design a solution; no code required in this document).  
**Project:** Kanvis pilot — **IAModel** training data + **C1** in-store inference (same Triton pose stack).  
**Language:** English (handoff).  
**Last updated:** 2026-10-30.

---

## 1. Purpose of this document

We need a **design and implementation plan** for offline extraction of pose training data from thousands of **MP4 clips** such that:

1. **Outputs are semantically and numerically equivalent** to what **C1** produces when it runs the **IAModel-full pose export** on the **same video file** (not the live RTSP-only path).
2. **Wall-clock time** is minimized via **safe parallelism** (GPU and host resources on a **single RTX 5090** machine with Triton on the same host).
3. The solution must **not** trade parity for speed (no “close enough” decode or different model path unless explicitly validated and accepted).

This document states **context, current practice, constraints, measurements, and success criteria** so another agent can propose a concrete architecture.

---

## 2. Executive summary

| Item | Summary |
|------|---------|
| **What we do today** | Batch job `extract-csv`: CSV rows → FFmpeg segment copy → per-clip pose extract via **`triton_pose_extract`** (`engine=c1_like`) → Triton **`yolo26s-pose`** + **`pose_core`** (ByteTrack) → `user_*/poses_full.npy`, `valid_mask.npy`, `meta.json`. |
| **What C1 does (reference)** | In production, live cameras use **RTSP + GStreamer + NVDEC** for speed. For **train/inference parity**, C1 **Level 2 (Gold N2)** **re-infers the session MP4** (`clip_buffer.mp4`) with the **same Triton + pose_core rules** and publishes **`merged.npy`** (dense timeline) to MinIO. |
| **Stated equivalence** | **Ingress differs** (RTSP live vs offline MP4 file). **Once a BGR frame is available**, preprocess → Triton → decode end2end → tracker → IAModel export **must match C1’s file-based reinfer path**. |
| **Main pain** | Batch takes **days**; **GPU utilization stays very low (~1–17% SM)** because decode + tracker + I/O dominate; parallelism is limited without hurting CPU/disk. |
| **Secondary pain** | Legacy corpus was produced with **older local-YOLO extract**; re-extract planned with Triton pipeline for **single domain** before retraining. |
| **Ask** | Propose how to **replicate C1’s post-frame pipeline exactly** while **parallelizing** clip and/or frame processing so a **multi-day batch** becomes **acceptably short**, without breaking Gold N2 contracts. |

---

## 3. Background and history

### 3.1 Repositories and roles

| Repo / component | Role |
|------------------|------|
| **`GuadIA-IAModel`** / `experiments/triton_pose_extract` | Offline batch: CSV → clips → `.npy`; wrappers `run_extract.sh`, `extract_clips.py`. |
| **`kanvis-triton-pose`** | Vendor library **`pose_core`**: preprocess, Triton client, `decode_end2end`, ByteTrack, `ClipExtractor`, export IAModel full. |
| **`kanvis-monitoring` (C1)** | Live pose pipeline (RTSP, pools, session close, MinIO upload, **`clip_buffer_reinfer`**). |
| **Triton Docker** | Model **`yolo26s-pose`** (TensorRT plan), gRPC **47811**, HTTP ready **47810**. |

Both C1 and IAModel **call the same Triton model** and share **`pose.yaml`** / **`pose_bundle.json`** (`pose_bundle_id`, e.g. `yolo26s-pose_v1`).

### 3.2 Training vs live continuous export

- **Training (Campaign2 / TCN)** consumes **`poses_full.npy`** on a **dense grid**: one row per decoded video frame (`frame_policy=keep_all`), NaN + `valid_mask=False` when no usable pose.
- **C1 continuous RTSP** uses **~12 Hz ingest**, hold-last, chunking (~8 s windows), different timeline semantics — **not** the gold reference for offline batch parity.
- **Gold reference** for “does offline match C1?” is **`merged.npy` vs offline `user_*/poses_full`** on the **same `clip_buffer.mp4`**, validated with `compare_c1_golden.py` (see session **`77f5e970…`**, almost PASS with 2 mask rows at threshold border).

### 3.3 Legacy data

Corpus **`data_yolo26s_010826`** (and similar) was extracted with **`pose_extractor_clean.py`** (local YOLO). Compared to current Triton extract:

- Systematic **T − 1** vs ffprobe on some clips.
- Small coordinate drift (~1–3% mean |Δ|, higher max on edge frames).
- **Decision:** re-extract **entire corpus** with Triton + C1-equivalent file path for reliability (store closed 3 days — time available).

---

## 4. What “exactly like C1” means (contract)

### 4.1 Non-negotiable equivalence (post-frame)

After the **i-th BGR frame** of the MP4 is obtained, the following **must match C1 `clip_buffer_reinfer` / IAModel Gold N2**:

| Stage | Requirement |
|-------|-------------|
| **Geometry** | Resize / letterbox per **`pose.yaml`**: e.g. max **640×360**, letterbox **640**, BGR→RGB as configured. |
| **Triton** | Model **`yolo26s-pose`**, FP16 input NCHW [0,1], batching per config; **`decode_end2end`** with **`parse_conf`** (e.g. 0.18 → decode threshold ~0.09). |
| **Tracker** | **ByteTrack** via `pose_core`; **tracker reset at frame 0** of each file; same **`tracker_yaml`** from bundle. |
| **Keypoints** | COCO indices **5–12** → **8 KP**, normalized **[0, 1]**, confidence mask **`min_kp_conf` = 0.5**. |
| **`valid_mask`** | **`frame_valid`**: ≥ **5** visible keypoints (IAModel full export rules). |
| **Temporal grid** | **`keep_all`**: **`T = ffprobe `nb_read_frames`** (dense index **i ↔ frame i**); missing track → **NaN row**, not shortened arrays. |
| **Multi-person** | Batch: **`export_primary_user_on_multi_track`** → one **`user_*`** (dominant track), equivalent to dominant person in reinfer. |
| **Artifacts** | `poses_full.npy` `(T, 8, 2)`, `valid_mask.npy` `(T,)`, `meta.json` with `frame_count`, `pose_bundle_id`, etc. |

**Documentation of record:**  
`IAModel/docs/IAMODEL_GOLD_N2_DECODE_PARITY.md`,  
`IAModel/docs/MENSAJE_IAMODEL_PARIDAD_POSES_FULL_C1.md`,  
`IAModel/docs/MENSAJE_IAMODEL_GOLD_77F5.md`.

### 4.2 Where RTSP vs MP4 is allowed to differ

| Aspect | C1 live (RTSP) | Offline batch (MP4) | Must match for training parity? |
|--------|----------------|---------------------|----------------------------------|
| **Source** | Network stream | File on disk (CSV-cut or full file) | N/A (different source) |
| **Decode before reinfer** | GStreamer **NVDEC** (fast) | For **Gold N2 / C1 reinfer on file**: **OpenCV `VideoCapture`** | **Yes for gold** — agreed contract |
| **Frame rate** | Subsampled ~12 Hz in session | **All decoded frames** (`keep_all`) on clip MP4 | Same **per frame index** on **same MP4** |
| **Output layout** | MinIO `merged.npy` + chunks | `data_result/.../user_*/poses_full.npy` | **Semantic equivalence** on same MP4 |

**Team statement (product owner):**  
*C1 is RTSP and we are MP4, but the frames are the same; once the frame is obtained, everything else must be identical.*

**Interpretation for engineers:**  
Parity is defined on **identical MP4 bytes** (e.g. `clip_buffer.mp4` or training clip). Live RTSP is **not** the batch reference. Offline must align with **C1’s file reinfer**, not with live-only RTSP timing.

### 4.3 Explicit non-goals (unless validated)

- Using **FFmpeg NVDEC** (`ffmpeg_nvdec`) for batch **without** proving pixel/T parity vs OpenCV on a sample — currently documented as **speed-only**, not Gold N2.
- Training on **`--session-close`** sidecar exports mixed with standard CSV batch (different filters / `user_0` IoU path).
- Replacing Triton with local YOLO for production training corpus.

---

## 5. Current offline pipeline (as implemented)

### 5.1 Flow

```text
CSV (video_path, t_start, t_end, category)
  → build job queue (filters: duration, category limits, skip_existing if user_*/poses_full.npy exists)
  → FFmpeg stream copy (optional segment) → temp or in-place MP4
  → ThreadPoolExecutor: N workers (default parallel_clips ≈ 2 on 5090)
       each worker: PoseExtractStack (Triton gRPC client + preprocess + tracker)
       engine "c1_like" (IAModel):
         read_frames_bgr(decode_backend=opencv, decode_c1_pixel_parity=true)
         parallel preprocess (8 workers)
         Triton infer in batches (triton_batch_size=8)
         ByteTrack frame loop + export + primary user prune
  → write under {output_base}/data_result/{cat}/{clip_name}/
  → logs/csv_extract_report.json
```

**Entrypoint:**  
`IAModel/experiments/triton_pose_extract/run_extract.sh extract-csv`

**Config:**  
`IAModel/experiments/triton_pose_extract/triton_pose_extract_config.json`

### 5.2 Environment (pilot host wudebian)

- **`KANVIS_TRITON_POSE_ROOT`**: clone of **`kanvis-triton-pose`** on host (not inside Triton container).
- **Triton:** `127.0.0.1:47811` (or `kanvis-pilot.ai:47811`).
- **Storage:** large HDD (`/media/8TB/...`) — I/O may be a bottleneck.
- **GPU:** RTX 5090, Triton co-located.

### 5.3 Observability (recent)

- **`[PROGRESS]`** lines every **20 s**: % done, OK/SKIP/FAIL, ETA (throughput + median/workers model).
- **`bench-csv --sample 50`**: empirical clips/h and ETA projection.

---

## 6. C1 reference path (what we must match)

### 6.1 Gold N2 — `clip_buffer_reinfer`

On session close (when enabled):

- Input: **`clip_buffer.mp4`**
- Decode: **OpenCV** (not live NVDEC)
- Infer + track + export: Triton + **`pose_export_iamodel_full`**
- Output: **`merged.npy`**, **`merged_valid_mask.npy`**, dense chunks aligned to same timeline

IAModel offline extract on the **same file** should match **`merged.npy`** (index-aligned), ε = **1e-4** on coords and identical **`valid_mask`** per row (Gold N2 target).

### 6.2 Known residual gap (Oct 2026)

Session **`77f5e970…`**: T=226 aligned, **2 rows** `valid_mask` mismatch at confidence threshold border, max |Δcoords| ~0.016.  
C1 internal **`run_clip_buffer_reinfer_triton_pool`** vs IAModel **`c1_like_extract`** still slightly divergent on tracker/batch ordering — **Option A/B** discussed: C1 subprocess `run_extract.sh` at close vs shared module in vendor (**`TRITON_EXTRACT_PERF.md`**).

---

## 7. The problem to solve

### 7.1 Primary problem

**Design a batch architecture that:**

1. Produces **`poses_full.npy` / `valid_mask.npy`** **indistinguishable from C1 file reinfer** on the same MP4 (Gold N2 criteria).
2. Minimizes **end-to-end wall time** for **~10k–22k+ clips** (order of magnitude: **multi-day** today).
3. **Increases GPU utilization** meaningfully during batch (today **~1–17% SM** average).

### 7.2 Secondary constraints

- **Resume-safe:** skip clips that already have valid `user_*/poses_full.npy`.
- **Mixed clip lengths:** short `clip_buffer`-like segments vs long `_full_` / action-rec clips (different seconds/clip).
- **Yield:** many clips **SKIP** (multi-person, filters) — still consume wall time; ETA must account for them.
- **Worker warmup:** each parallel worker loads **ByteTrack + `yolo26s-pose.pt`** locally once (~30–90 s); Triton holds TRT inference.

### 7.3 Why “just parallelize more” failed intuition

Measured (**`TRITON_EXTRACT_PERF.md`**, Oct 2026):

- Median **~23.6 s/clip**; GPU not saturated.
- Bottleneck: **CPU OpenCV decode all frames into RAM**, **CPU tracker**, **disk I/O**, not TRT.
- Increasing **`parallel_clips`** to 4 often **increases contention** without raising GPU usage.

Parallelism must target **decode / preprocess / infer pipelining** and/or **shared Triton feeding**, not only more threads on the same sequential per-clip loop.

---

## 8. Objectives (for the solving agent)

### 8.1 Functional objectives (P0)

| ID | Objective |
|----|-----------|
| **F1** | Bit-for-bit or Gold-validated equivalence: offline **`user_*/poses_full`** vs C1 **`merged.npy`** on same MP4 (`compare_c1_golden` PASS). |
| **F2** | **`T == ffprobe nb_read_frames`**, dense **`keep_all`**, same bundle and thresholds (0.5 / 5 KP). |
| **F3** | Single code path or provably identical graph: **post-frame** ops shared between C1 reinfer and offline batch (ideal: one module in **`kanvis-triton-pose`**). |
| **F4** | Batch mode remains **CSV → FFmpeg copy → extract**; no regression on filters / primary user / reporting. |

### 8.2 Performance objectives (P0)

| ID | Objective |
|----|-----------|
| **P1** | Reduce median **seconds/clip** on representative 10 s clip (baseline ~30 s wall → target band **documented in TRITON_EXTRACT_PERF**: Phase 0 ~15–20 s, Phase 2 ~5–10 s if ingest pipelined **without breaking OpenCV gold decode**). |
| **P2** | Raise ** sustained GPU SM** during batch (target band **25–40%+** minimum in Phase 0; higher if infer pipeline fed continuously). |
| **P3** | Scale **throughput (clips/h)** with **bounded** worker count; provide formula or config for **5090 + 2 Triton instances**. |
| **P4** | Live **ETA** during run (already: 20 s progress lines); optional: project to full inventory count. |

### 8.3 Operational objectives (P1)

| ID | Objective |
|----|-----------|
| **O1** | Clear **benchmark protocol** (one 10 s clip, `nvidia-smi dmon`, before/after). |
| **O2** | Document **config knobs** (parallel_clips, triton_batch_size, preprocess_workers, Triton instance_group) with **parity guardrails**. |
| **O3** | Path to **Option A** (C1 invokes same extract at close) or **Option B** (shared fast offline in vendor). |

---

## 9. Constraints and guardrails

- **Do not** change Triton model or IAModel KP layout without retraining agreement.
- **Do not** use live RTSP ingest as the batch reference pipeline.
- **Gold decode** for parity tests: **`decode_backend=opencv`**, **`decode_c1_pixel_parity=true`** (`IAMODEL_GOLD_N2_DECODE_PARITY.md`).
- **System SHM:** not available in current `pose_core` client (`use_system_shm: false` until vendor implements).
- **Parallel clip workers:** each has **own gRPC client + tracker state** — cross-clip batching into one Triton request requires **careful design** (tracker is per-clip).
- **Disk:** output on spinning 8TB — consider NVMe staging if agent proposes I/O optimizations.

---

## 10. Measurements and artifacts (baseline)

| Metric | Baseline (Oct 2026, 5090) |
|--------|---------------------------|
| GPU SM during `extract-csv` | ~1–17% |
| Median wall time / clip | ~23.6 s (mixed corpus) |
| `clip_buffer` ~10 s video | ~28–46 s wall |
| Default `parallel_clips` | 2 |
| `triton_batch_size` | 8 |
| `preprocess_workers` (c1_like) | 8 |

**References:** `IAModel/docs/TRITON_EXTRACT_PERF.md`, `experiments/triton_pose_extract/README.md`.

---

## 11. Solution space (hints, not prescriptions)

The solving agent should evaluate and compare:

1. **Pipeline parallelism per clip** (decode thread → bounded queue → preprocess pool → Triton batching → track) **keeping OpenCV decode** for gold.
2. **Reuse C1 `triton_pool` / reinfer loop** for offline files (same code as MinIO export, different frame source adapter).
3. **Triton-side capacity** (`gpu_instance_count`, `max_batch_size`, dynamic batching) vs **client batch size 16**.
4. **I/O**: NVMe output buffer, fewer small writes, optional async npy write.
5. **Worker model**: separate **decode-heavy** from **infer-heavy** pools vs homogeneous workers.
6. **Unified binary at C1 close (Option A)** to eliminate pool vs `c1_like` drift.
7. **What NOT to do first:** enable **ffmpeg_nvdec** for “C1 parity” — live NVDEC ≠ file reinfer contract.

---

## 12. Success criteria (acceptance)

### 12.1 Parity acceptance

- [ ] **`compare_c1_golden`** PASS on **≥ 3** diverse gold sessions (short pasillo, multi-anchor if applicable, different lengths).
- [ ] **`decode_backend_used`: `opencv`** in `meta.json` for gold runs.
- [ ] Spot-check **legacy compare** (`compare-legacy`) on re-extracted corpus: **T equal**, coords within agreed ε on overlapping frames.

### 12.2 Performance acceptance

- [ ] **`bench-csv --sample 50`** shows **≥ 2× throughput** vs documented Oct 2026 baseline **or** documented justification why parity bounds cap gain.
- [ ] **`nvidia-smi dmon`** during 5 min batch shows **mean SM ≥ 25%** (Phase 0 target from internal perf doc) or agent explains trade-off.

### 12.3 Deliverables expected from agent

1. **Recommended architecture** (diagram + phase 1 / phase 2).
2. **Config table** for 5090 pilot.
3. **Risk list** (parity regressions, OOM, Triton queueing, tracker nondeterminism).
4. **Test plan** (gold clips, bench protocol, rollback).
5. **Optional:** code change list by repo (`IAModel`, `kanvis-triton-pose`, `kanvis-monitoring`).

---

## 13. Key file paths (implementation index)

| Path | Description |
|------|-------------|
| `IAModel/experiments/triton_pose_extract/run_extract.sh` | CLI wrapper |
| `IAModel/experiments/triton_pose_extract/extract_clips.py` | `extract-csv`, `bench-csv`, `preflight-csv` |
| `IAModel/experiments/triton_pose_extract/triton_pose_extract/batch_csv.py` | Parallel CSV batch |
| `IAModel/experiments/triton_pose_extract/triton_pose_extract/c1_like_extract.py` | C1-like loop (decode + prep + infer + track) |
| `IAModel/experiments/triton_pose_extract/triton_pose_extract/video_decode.py` | OpenCV vs ffmpeg_nvdec |
| `IAModel/experiments/triton_pose_extract/triton_pose_extract/batch_progress.py` | Live ETA |
| `IAModel/experiments/triton_pose_extract/compare_c1_golden.py` | Gold validation |
| `IAModel/docs/TRITON_EXTRACT_PERF.md` | Perf analysis and phased plan |
| `IAModel/docs/IAMODEL_GOLD_N2_DECODE_PARITY.md` | Decode contract |
| `IAModel/docs/MENSAJE_IAMODEL_C1_SESSION_CLOSE_EXTRACT.md` | Session-close (separate from CSV batch) |

---

## 14. Open questions for the solving agent

1. What is the **best shared abstraction** (`FrameSource`?) so **C1 reinfer** and **IAModel extract-csv** call **one post-frame implementation**?
2. How to **pipeline OpenCV decode** (mandatory for gold) without loading **entire clip in RAM** before infer?
3. Can **multiple clips** share one **Triton request stream** without breaking **ByteTrack state per clip**?
4. Is **deterministic replay** (fixed batch boundaries, FP16) required to close the **~0.01–0.02 coord** gap vs C1 pool?
5. Recommended **`parallel_clips` × triton_batch_size × preprocess_workers** for **5090** with **parity tests** automated in CI?
6. Should full corpus re-extract use **staging on NVMe** + rsync as default ops pattern?

---

## 15. One-sentence problem statement

**We must re-extract thousands of MP4 training clips through the same IAModel+Triton pose stack C1 uses for file-based reinfer (frame-identical inputs, identical post-frame processing), while restructuring offline batch parallelism so a 5090-hosted Triton server is fed efficiently and wall-clock time drops from multi-day to an operationally acceptable duration without violating Gold N2 numeric parity.**
