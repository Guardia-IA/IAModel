# Solution: C1-equivalent offline pose extraction with pipelined parallelism

Companion to [`OFFLINE_EXTRACT_C1_PARITY_PROBLEM_STATEMENT.md`](./OFFLINE_EXTRACT_C1_PARITY_PROBLEM_STATEMENT.md).  
**Status:** implemented in `IAModel/experiments/triton_pose_extract` (2026-10-30); pending throughput measurement on the 5090 pilot host.

---

## 1. Key finding: "like C1" already *is* our code

C1 in production uses **Option A** at session close (`kanvis-monitoring/pose/clip_buffer_triton_pose_extract.py`): it spawns **`IAModel run_extract.sh extract --session-close`** on `clip_buffer.mp4` and publishes that `user_*/poses_full.npy` as `merged.npy` (`pose_export_source=triton_pose_extract_at_close`, `TRITON_EXTRACT_DECODE_BACKEND=opencv`).

Therefore the reference post-frame graph **is** `triton_pose_extract/c1_like_extract.py`:

```text
cv2.VideoCapture.read()  →  prepare_infer_image → letterbox → to_tensor
  →  Triton yolo26s-pose, batches of 8 consecutive frames of the same clip
  →  decode_end2end(parse_conf·0.5) → detections_to_mock → ByteTrack.update (in frame order, reset at frame 0)
  →  keep_all rows (NaN when absent), min_kp_conf 0.5, frame_valid ≥5 KP
  →  primary-user / filters → poses_full.npy, valid_mask.npy, meta.json
```

Parity with C1 is therefore not a matter of imitating a different pipeline; it is a matter of **not changing any of those operations** while changing *when* they run.

## 2. Where the time went (code review, before measuring)

| Finding | Effect | Fix |
|---------|--------|-----|
| `read_frames_bgr` with `decode_c1_pixel_parity=true` ran an **extra FFmpeg/NVDEC decode of every clip** after OpenCV already matched ffprobe (loop body executed before the break check); result was discarded. | 2× decode per clip, GPU/CPU waste, contention. | Loop replaced: only try OpenCV if it was not attempted; no extra decode when OpenCV T == ffprobe. |
| `ffprobe -count_frames` for T **fully decodes the file** (third decode, CPU). | ~0.3–6 s/clip depending on length. | `ffprobe -count_packets` (demux only) first; `-count_frames` only when the decoded count disagrees (gold criterion preserved). |
| Whole clip decoded into RAM (`list[np.ndarray]` full-res), then preprocessed, then inferred, then tracked: **no overlap**, one sync gRPC request at a time per worker. | GPU idle most of the time; RAM spikes (1080p × 120 s ≈ 10 GB) → `parallel_clips` capped at 2. | Streaming pipelined engine (below). |
| Per-clip timings were computed and discarded by `batch_csv.py`. | No visibility into the bottleneck. | Timings stored per job in `csv_extract_report.json` and averaged in `[PROGRESS]` lines. |

## 3. The pipelined engine (`c1_like_pipeline.py`)

Per clip, inside each batch worker thread:

```text
decoder thread ──► chunk of 8 frames ──► infer pool (N threads, 1 gRPC channel each)
 (OpenCV, streaming)                        │  prep_pool.map(prep)  →  client.infer(batch)
                                            ▼
                             ordered queue of futures (bounded: N+1 chunks in flight)
                                            │
                              consumer (worker thread): for chunk k in order
                                 for frame in chunk: track_frame(...)   ← ByteTrack sequential
```

**Invariants that guarantee identical output:**

1. Frames come from the same `cv2.VideoCapture.read()` sequence (`iter_frames_opencv` ≡ `_read_opencv`).
2. Preprocess is a pure function per frame.
3. Each Triton request contains **exactly the same frames as the sequential `_infer_batched`** (indices 0–7, 8–15, …, remainder). Clips are never mixed in a request; a chunk is never split.
4. Post-inference (`track_frame`) runs **strictly in frame order** on the worker thread, one tracker instance per worker, `reset()` at frame 0. Only the *arrival* of Triton results is concurrent; their *consumption* is sequential.
5. Export and metadata go through the same `finalize_users_export` as the sequential path.

**What is concurrent:** decode of chunk k+1…k+N, preprocess, and the gRPC round-trips, while the tracker consumes chunk k. Triton sees `parallel_clips × infer_inflight` requests of batch 8 instead of `parallel_clips`; with `max_queue_delay_microseconds: 3000` and 2 instances it can dynamically batch them (this already happened with 2 workers; it is not new behaviour).

**Fallbacks (same semantics as before):**

- If decoded frames ≠ `-count_packets`, run `-count_frames`; if still ≠, the clip is re-run through the **sequential path** (`read_frames_bgr` reconcile over backends) — exactly what happened before for every clip.
- `--session-close`, `--subject-bbox-json`, `--export-track-id` (C1 close path) **always use the sequential path**; C1 production is untouched by this change.
- Any exception in Triton/preprocess cancels the producer and propagates (no silent partial export).

**Memory:** bounded to `(infer_inflight + 1) × 8` full-resolution frames per worker instead of the whole clip → `parallel_clips` can be raised safely.

## 4. Configuration (`triton_pose_extract_config.json` → `extract`)

| Key | Default | Notes |
|-----|---------|-------|
| `pipeline_enabled` | `true` | Env override `TRITON_EXTRACT_PIPELINE=0/1` for A/B. |
| `infer_inflight` | `3` | Concurrent chunk requests per clip worker. 2–4 is the useful range on a single 5090. |
| `ffprobe_fast_count` | `true` | `-count_packets` first; `-count_frames` only on mismatch. |
| `decode_backend` | `opencv` | **Must stay `opencv` for parity**; pipelined mode is only used when it is. |
| `parallel_clips` (top level) | `2` | Try 3–4 now that memory is bounded; measure. |
| `triton_batch_size` | `8` | **Do not change** (batch boundaries are part of the parity contract; C1 pins 8 too). |

## 5. Validation

Bit-for-bit A/B (sequential vs pipelined) on real clips with the real Triton:

```bash
cd ~/IAModel/experiments/triton_pose_extract
source ../training/venv/bin/activate
export KANVIS_TRITON_POSE_ROOT=…  KANVIS_TRITON_URL=127.0.0.1:47811
./run_extract.sh parity-ab --video /ruta/clip1.mp4 --video /ruta/clip2.mp4
./run_extract.sh parity-ab --video-dir /ruta/carpeta_mp4 --limit 10
```

Expected: `PASS`, every `poses_full/valid_mask/keypoint_mask/poses` identical, same T, and `wall pipe < wall seq`.

Offline mock test (no GPU, fake deterministic Triton) already passes: identical arrays, same batch composition `[8,8,8,8,8,5]`, fallback paths exercised.

Gold N2 remains the external check: `compare_c1_golden.py` against a C1 session `merged.npy` (C1 itself produced it with this same code path).

## 6. Measuring throughput

```bash
# A/B on 50 real clips
TRITON_EXTRACT_PIPELINE=0 ./run_extract.sh bench-csv --input-root … --output-base /tmp/bench_seq  --sample 50 --parallel 2
TRITON_EXTRACT_PIPELINE=1 ./run_extract.sh bench-csv --input-root … --output-base /tmp/bench_pipe --sample 50 --parallel 2
TRITON_EXTRACT_PIPELINE=1 ./run_extract.sh bench-csv … --parallel 4

# Full run: [PROGRESS] every 20 s with ETA and per-stage averages
./run_extract.sh extract-csv --input-root … --output-base … --parallel 3
```

The `medias/clip:` line tells you where time goes now:

- `ffmpeg_cut` high → HDD seek on source videos; consider NVMe staging of sources.
- `decode_wall` ≈ `total` → CPU decode bound; more `parallel_clips` (CPU cores permitting).
- `infer_wait` ≫ `triton_rtt` → waiting on prep/decode, not Triton.
- `triton_rtt` high with GPU idle → gRPC/host copy; consider Triton `instance_group` 3–4 or SHM once vendor supports it.
- `track` high → ByteTrack/Ultralytics CPU; one tracker per worker is already the minimum; check `torch` is using GPU for the `.pt` predictor warmup.

## 7. Not done (deliberately)

- **No NVDEC for parity runs.** `ffmpeg_nvdec` remains available as a speed-only option and is not equal to the C1 file reinfer contract.
- **No cross-clip batching into one Triton request.** It would change batch boundaries; Triton dynamic batching already merges concurrent requests server-side without changing per-sample inputs.
- **No change to `kanvis-triton-pose` or `kanvis-monitoring`.** The post-frame graph is shared by import; C1 at close still runs the sequential path.

## 8. Files

| Path | Change |
|------|--------|
| `triton_pose_extract/c1_like_pipeline.py` | New pipelined engine |
| `triton_pose_extract/c1_like_extract.py` | Dispatch to pipeline; `track_frame` / `finalize_users_export` shared |
| `triton_pose_extract/video_decode.py` | `ffprobe_nb_read_packets`, `iter_frames_opencv`, redundant-decode fix |
| `triton_pose_extract/extract_stack.py` | `infer_pool`/`prep_pool`, `clone_triton_client` |
| `triton_pose_extract/pipeline_config.py` | `pipeline_enabled`, `infer_inflight`, `ffprobe_fast_count`, env `TRITON_EXTRACT_PIPELINE` |
| `triton_pose_extract/batch_csv.py`, `batch_progress.py` | Per-job timings in report + progress averages |
| `scripts/parity_ab_pipeline.py`, `run_extract.sh parity-ab` | Bit-for-bit A/B tool |
| `triton_pose_extract_config.json` | Pipeline on by default for batch |
