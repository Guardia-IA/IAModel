# Respuesta IAModel — extract al cierre sesión C1 (clip multi-persona, 1 person_id)

Copiar/adaptar para Kanvis C1.

---

**Asunto:** Cierre sesión C1 — `--session-close` / `--export-track-id` (no batch single-user)

Hola,

Contexto entendido: en piloto, **N personas en pasillo → N sesiones**; el MP4 de cada sesión es el **plano entero** (~96 s, ej. `ff3c1f32200a4ed6b8f9f84808ccebe9`, 1156 frames). C1 llama hoy:

`extract_clips.py extract --video clip.mp4 --cat gold --validate`

sin track/person → log **`2 track(s) filtros (filtered=2 frames=1156)`**, sin `user_*`, solo `clip.mp4`, `meta.json`, `pose_bundle.json` → `triton_pose_extract_at_close_exhausted` y fallback reinfer.

---

### 1) ¿CLI/config para un solo track?

**Antes (batch entrenamiento):**

| Mecanismo | Comportamiento |
|-----------|----------------|
| `filters.single_user_only` + `export_primary_user_on_multi_track: true` (default JSON) | Exporta todos los tracks que pasen calidad, **poda** a `user_{track}` con más frames válidos (`primary_user.py`). |
| `--no-single-user-only` | **No recomendado en cierre:** además de permitir multi-track, **desactiva** `export_primary_user_on_multi_track` y deja varios `user_*`. |
| `--single-user-only` | Fuerza descarte si ByteTrack ve 2+ IDs (log distinto). |

**Nuevo — modo cierre sesión (inferencia, no batch):**

```bash
export TRITON_EXTRACT_DECODE_BACKEND=opencv   # paridad C1 reinfer
./run_extract.sh extract \
  --video clip.mp4 \
  --out /scratch/... \
  --cat gold \
  --validate \
  --session-close
```

- `--session-close`: `single_user_only=false` en el motor, **mantiene** `export_primary_user_on_multi_track` → un solo `user_*` (dominante) aunque haya 2 cuerpos en pasillo.
- `--export-track-id ID`: exporta **solo** `user_{ID}` (ID **ByteTrack interno** del extractor); relaja filtros `BODY_VISIBLE_*` para ese export (igual que session-close).

Opcional en JSON (`filters`): `"session_close_export": true`, `"export_track_id": 3` (misma semántica).

**No usar** `--no-single-user-only` en el camino de cierre.

---

### 2) ¿Track_id C1 y/o bboxes por frame?

- **`--export-track-id`**: sí, si C1 envía el **mismo entero** que asigna ByteTrack en esa pasada offline (mismo vídeo, mismo tracker). Solo se escribe `user_{ID}/poses_full.npy` (+ poses/masks).
- **Lista de bboxes por frame desde C1**: **no implementado** en esta versión. Roadmap razonable: fichero sidecar (JSON/npy) + selección de detección por IoU por frame antes de acumular `poses_full`.
- Mientras tanto: C1 puede pasar `--export-track-id` tras un smoke/diagnose, o confiar en `--session-close` + poda primary (track con más frames válidos en el clip).

---

### 3) ¿Matching por solape bbox si IDs no coinciden?

**No** en el extractor actual. Si el ID de sesión C1 ≠ ID ByteTrack offline, hay que:

- re-mapear en C1 antes de llamar a extract, o
- pedir matching IoU (futuro sidecar), o
- usar `--session-close` sin `--export-track-id` y alinear por **primary user** (heurística frames válidos).

---

### 4) ¿Qué significa `filtered=2`?

**No es `single_user_only`.** Ese caso loguea:

`single_user_only y N tracks → sin export`

El mensaje **`N track(s) filtros (filtered=N frames=T)`** sale cuando ByteTrack **sí** vio tracks (`users` no vacío) pero **ninguno** pasó `_passes_filters` en `c1_like_extract.py`:

- `BODY_VISIBLE_MIN_FRAMES`
- `BODY_VISIBLE_MIN_RATIO`
- cobertura mínima `acc.total >= 0.2 * total_frames`

(`min_kp_conf` afecta validez de keypoints en acumulación, no ese log directamente.)

Con **`--session-close`** (o `--export-track-id`) se **omite** ese gate de export para el camino de cierre; en `meta.json` veréis `iamodel_session_close_export`, `iamodel_export_track_id` y por usuario `export_relaxed: true` si se exportó pese a no pasar BODY_VISIBLE.

**meta.json de la sesión `ff3c1f32200a…`:** no está en el repo IAModel; si sigue en scratch C1, adjuntad `gold/clip_buffer/meta.json` (campos útiles: `users`, `track_count` si existe, `frame_count`, `iamodel_*`).

---

### Comando recomendado C1 al cierre

```bash
./run_extract.sh extract \
  --video "${SESSION}/clip.mp4" \
  --out "${SCRATCH}/iamodel_extract" \
  --cat gold \
  --parallel 1 \
  --validate \
  --session-close
# Si tenéis track ByteTrack estable para esa sesión:
#   --export-track-id "${SESSION_TRACK_ID}"
```

Dataset batch / entrenamiento: **sin cambios** (`single_user_only` + primary user). `--session-close` es solo invocación al cierre en tienda.

Saludos,  
IAModel / triton_pose_extract
