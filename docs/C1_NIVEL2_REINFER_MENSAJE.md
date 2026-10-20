# Mensaje C1 — Nivel 2: re-infer `clip_buffer` → MinIO

**Para:** equipo C1 (`kanvis-monitoring`)  
**De:** IAModel  
**Sesión gold de referencia:** `67537c38e24f49ef993ad865a0f02c5c`  
**Contrato:** `kanvis_pose_v1` — ver también [`MENSAJE_IAMODEL_PARIDAD_POSES_FULL_C1.md`](./MENSAJE_IAMODEL_PARIDAD_POSES_FULL_C1.md) §10–§11.

---

## Resumen ejecutivo

| Pregunta | Respuesta |
|----------|-----------|
| ¿Falló nuestro extract offline? | **No.** `triton_pose_extract` sobre `clip_buffer.mp4` es la **referencia** (train / Campaign2). |
| ¿Por qué `compare_c1_golden` → `pass: false`? | El **`chunk_001` en MinIO** sale del **ingest RTSP en vivo**; el gold offline sale del **MP4 decodificado frame a frame**. Mismo instante (~máscaras OK), **píxeles/pipeline distintos** → Δ coords ~**0,4** (ε = 1e-4). |
| ¿Es bug de sync / frontend? | **No.** Alineación §10.2 (`playback`) OK (`dt_alignment_violations: 0`, `valid_mask_mismatch: 0` en 67537). **No pedimos más tuning de `frontend_playback`.** |
| ¿Mismo Triton/bundle? | **Sí.** `pose_bundle_id`: **`yolo26s-pose_v1`** C1 = offline (67537). |
| ¿Qué hay que hacer? | **C1** publica MinIO (**merged + chunk_***) tras **re-inferir `clip_buffer.mp4`** con el **mismo tubo** que `triton_pose_extract`, **no** paridad numérica con la cola RTSP histórica. |

**Objetivo acordado:** MinIO = mismo pipeline que IAModel sobre `clip_buffer`, para intercambiar datos con entrenamiento e inferencia C2.

---

## Qué implica Nivel 1 vs Nivel 2

| | Nivel 1 (piloto actual) | Nivel 2 (implementación C1) |
|---|-------------------------|-----------------------------|
| **Formato `.npy`** | Reglas IAModel full (0,5 / 5 KP, `keep_all`, NaN, sin hold-last) | Igual |
| **Chunk RTSP vs offline MP4** | **No** exige coords idénticas | **Sí:** upload MinIO = re-infer MP4 denso |
| **Piloto operativo** | Chunk 67537 **GO** Nivel 1 (máscaras alineadas) | Gold numérico **PASS** tras re-infer |

---

## Trabajo C1 (esperado)

1. Hook **`clip_buffer_iamodel_export`** / `export_iamodel_poses_from_clip`: decode **100 %** frames del `clip_buffer` + infer + ByteTrack + export IAModel full.
2. Flag **`continuous.pose_merged_export_from_clip_buffer`**: al cerrar sesión, publicar **`merged*`** (T ≈ `frame_count` del MP4) y **chunks** = recorte sobre esa timeline densa, **sustituyendo** poses solo-RTSP en upload cuando el flag esté activo.
3. Validación: scripts gold en repo C1 + **`compare_c1_golden.py`** (IAModel) vs `user_0` offline en wudebian.

**Referencia IAModel (comando gold offline):**

```bash
cd IAModel/experiments/triton_pose_extract
source ../training/venv/bin/activate
./run_extract.sh extract \
  --video "$SESSION/clip_buffer.mp4" \
  --out "$OUT" --cat gold --parallel 1 --validate
```

Config efectiva gold N2: **`engine=c1_like`**, decode **`opencv`** (mismo tubo que C1 **`clip_buffer_reinfer`** / `VideoCapture`; **`ffmpeg_nvdec`** opcional batch entrenamiento 5090), Triton **`yolo26s-pose`**, **`frame_policy=keep_all`**, preprocess **640×360 → letterbox 640**, **BGR→RGB**, export **0,5 / 5 KP**, gap-fill **none**. Override: env **`TRITON_EXTRACT_DECODE_BACKEND`**. Detalle: `triton_pose_extract_config.json` + `pose.yaml` en `kanvis-triton-pose`.

**Sujeto (1 persona):** mismas reglas que ClipExtractor (track dominante / `export_primary_user`); no hace falta fijar `track_id` de sesión RTSP en el re-infer MP4.

---

## Cómo validamos el cierre (IAModel)

| Check | PASS |
|-------|------|
| **T** | `len(merged) == len(offline poses_full)` (67537 offline: **241**) |
| **Coords + máscara** | `compare_c1_golden.py`: **`pass_mask` + `pass_coords`**, ε = 1e-4 |
| **chunk_001** | Mismo script, `--chunk chunk_001 --align playback` (recorte denso, no RTSP) |
| **Bundle** | `pose_bundle_id` igual (ya OK en 67537) |

```bash
python3 compare_c1_golden.py \
  --c1-session /home/angel/TestKanvis/67537c38e24f49ef993ad865a0f02c5c \
  --offline-clip /home/angel/TestKanvis/gold_offline_67537c38/gold/clip_buffer \
  --chunk chunk_001 --align playback \
  --merged-align auto \
  --json .../gold_report_n2.json
```

**Nota script (oct-2026):** si `merged_T < offline_T` (p. ej. regen RTSP pisa merged denso), el informe **no aborta**: `merged_whole_clip` con `pass=false`, `align_failed` o `T_equal=false`, y el test **chunk** sigue. Con **N2 feliz** (`merged_T == offline_T`), `--merged-align auto` usa comparación **fila i ↔ i** (`dense_index`).

### Sesión `c1a198…` (oct-2026)

| Fuente | T | Notas |
|--------|---|--------|
| C1 merged | **172** | `clip_buffer_reinfer`, `regenerate_skipped`, **T = ffprobe** |
| IAModel extract (antes) | **162** | decode NVDEC ≠ `ffprobe -count_frames` → **`T_equal` bloqueado** |
| Chunk MinIO | NaN / sync | Sin `playback_start_at` en bundle → `has_sync_start=false`; usar `--align auto` o `merged_dense` |

**IAModel (fix):** (1) decode **`opencv`** (paridad C1 reinfer MP4) + **`ffprobe nb_read_frames`**; (2) infer **`keep_all`**. Tras re-extract: `frame_count == poses_full_T == ffprobe`, `decode_backend_used` = **`opencv`**. Entonces:

```bash
./run_extract.sh extract --video "$SESSION/clip_buffer.mp4" --out "$OUT" --cat gold --parallel 1 --validate
python3 compare_c1_golden.py ... --merged-align auto --align auto --json gold_report_n2.json
```

**C1 (pendiente menor):** incluir `playback_start_at` (o `pose_timeline_start_at` + offset) en `meta.json` sesión para §10.2 estricto; chunk post-slice denso no debe quedar solo NaN si el recorte tiene persona.

Adjuntar JSON + commit/PR C1. IAModel (o C1) ejecuta en **wudebian** (mismo Triton que piloto).

**Opcional C2:** mismos npy → `score_pose_ensemble.py` (41→96, 48→64) debe coincidir con C2 en probabilidades.

---

## Evidencia 67537 (reproducida en ambos lados)

- Chunk: **97×8×2**, `valid_mask` 100 % en ventana, `quality_pass`.
- Compare (`align=playback`): `valid_mask_mismatch: 0`, `rows_both_valid: 97`, **`max_abs_delta_coords ≈ 0,41`**, **`pass: false`** → gap **RTSP vs MP4**, no índices ni bundle.
- Debug filas i=0,48,96: `video_s ≈ 5,94 s` en frame 0 (overlay ~5,94 s), máscaras OK, Δ coords grande.

---

## IAModel (train)

Seguimos entrenando solo con **`user_*` offline**; no bloqueamos batch por el FAIL RTSP. Cuando C1 entregue re-infer 67537, repetimos gold; **`pass: true`** = Nivel 2 cerrado para esa sesión.

---

## Referencias

- [`MENSAJE_IAMODEL_PARIDAD_POSES_FULL_C1.md`](./MENSAJE_IAMODEL_PARIDAD_POSES_FULL_C1.md)
- [`C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md`](./C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md)
- Script: `experiments/triton_pose_extract/compare_c1_golden.py`
