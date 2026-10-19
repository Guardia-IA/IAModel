# Handoff C1: export continuo (`clip_buffer`) vs contrato de entrenamiento IAModel

**Fecha:** 2026-10-19  
**Audiencia:** equipo **C1 / kanvis-monitoring** (pipeline continuo, export MinIO, C2 inferencia) + IAModel (entrenamiento / Campaign2)  
**Propósito:** documento **autocontenido** con lo observado en exports reales de producción, por qué **no son comparables** a los clips/npy usados para entrenar (aunque ambos usen **Triton + yolo26s-pose**), y **preguntas / propuestas** para alinear contratos o corregir bugs.

**Caso reproducible principal:** sesión `d9a71a34123a44d293575afec5a460cb`  
**Export local de referencia:**  
`/home/debian/Documentos/Kanvis/session_pose_export_20261019_104309/`  
(incluye `clip_buffer.mp4`, `merged*.npy`, `chunk_*/poses_full.npy`, `valid_mask.npy`, `p1.json`, `frontend_playback.json`, `playback_poses.json`)

**Contexto FP producción:** análisis previo sobre ~11 sesiones del mismo export (falsos positivos stage-1 ensemble 41|48 AND@0.80). Este documento profundiza en **paridad temporal/export**, que explica tanto FP como desconfianza vídeo↔npy.

---

## 1. Resumen ejecutivo

1. **Mismo motor de pose ≠ mismo dato.** Entrenamiento IAModel usa clips offline (p. ej. `clip_full`) con ventanas densas y contrato `poses_full` + `valid_mask`. Producción C1 publica sesiones **`clip_buffer`** con chunks **de duración variable**, resampleo parcial, gaps temporales y, en casos límite, **coordenadas congeladas** al salir de plano.

2. **El nombre `chunk_00N` no implica “ventana fija de 8 s @ 12 fps (96 filas)”.** En `d9a71`, los cuatro chunks suman **~12 s de timeline de pose**, no 32 s de vídeo. `chunk_003` dura **~0,57 s / 13 filas** (micro-segmento, no “8 s rotos”).

3. **Desalineación vídeo ↔ pose documentada:** en `frontend_playback.json`, **`pose_offset_seconds = 5.0`**: la pose overlay empieza en **t = 5 s** del MP4 (19,5 s, 234 frames @ 12 fps). Comparar npy desde t=0 del vídeo **sin offset** da la impresión de datos incorrectos.

4. **Bug / inconsistencia de export (evidencia en disco):** en `chunk_004`, la cola de `poses_full` tiene **10+ frames idénticos** (hold-last), `valid_mask` **False** en la cola, pero **`merged_valid_mask` marca True** en esas mismas filas. Eso rompe confianza en `merged` y en playback.

5. **Implicación modelo:** entrenar con npy “limpios” y desplegar sobre estos exports **infla FP** y hace imposible auditar “a ojo” sin reglas de alineación. **No es solo umbral 0.80**; es **contrato de secuencia**.

6. **Pedido a C1:** responder preguntas §7, validar propuestas §8, tratar `d9a71` / `chunk_004` como **regresión** hasta explicación o fix.

---

## 2. Dos pipelines, un mismo Triton

| Aspecto | Entrenamiento / eval IAModel | Producción C1 continuo |
|--------|------------------------------|-------------------------|
| Origen | CSV → MP4 recortado → extract batch (`triton_pose_extract`, `c1_like`, etc.) | RTSP → sesión → **`clip_buffer.mp4`** + chunks MinIO |
| Unidad lógica | Clip con duración conocida (a menudo **~8 s** o `clip_full`) | **Sesión** partida en **`chunk_001…N`** + buffer |
| Filas `poses_full` | Grilla temporal alineada al clip; objetivo **~96 @ 12 fps** cuando aplica | **`mono_span_ms` real** por chunk (2,9 s, 4,3 s, **0,57 s**, 4 s…) |
| Inferencia robo (C2) | (Eval offline) pipeline campaña | `source_window: "clip"` en `p1.json`; ventanas **más cortas** que 8 s |
| Máscara | `valid_mask` coherente con ≥5 KP, conf | Misma regla declarada (IAModel full), pero **incoherencias merged vs chunk** |
| Vídeo vs pose | Clip y npy suelen compartir origen | MP4 **más largo** que timeline pose; **offset** explícito en playback |

**Conclusión:** usar Triton en ambos lados **no garantiza paridad**. Hay que fijar un **contrato export v2** (duración, fps, NaN/hold-last, merged, cierre de chunk) y tests de regresión.

---

## 3. Material analizado

### 3.1 Export `session_pose_export_20261019_104309`

- **11 sesiones** FP / revisión (UUID sin guiones).
- Por sesión: `meta.json`, `p1.json`, `merged.npy`, `merged_valid_mask.npy`, `merged_pose_timestamps.npy`, `clip_buffer.mp4`.
- Por chunk: `poses_full.npy`, `valid_mask.npy`, `poses.npy`, timestamps, a veces `meta.json`, `diag.json`, frames JPG.

### 3.2 Sesión detallada `d9a71a34123a44d293575afec5a460cb`

- **1 persona**, **1 track** (`max_persons_in_frame: 1`, `unique_tracks_seen: 1`).
- Cámara: `c0b2d0c2-e414-4635-94d2-b36447c868f8` (no pasillo multi-persona).
- **`clip_buffer.mp4`:** 19,5 s, **234 frames**, **12 fps** (`ffprobe`).
- **4 chunks** publicados; **alarma stage-1 solo en `chunk_001`** (p41≈0,95, p48≈0,99); C3 piloto ya marca `robbery_false_positive: true` con HAR `stop`.

---

## 4. Hallazgos (con números)

### 4.1 Vídeo vs timeline de pose (`d9a71`)

Campos en `frontend_playback.json` → `sync`:

| Campo | Valor |
|--------|--------|
| `playback_start_at` | 2026-10-19T07:53:36.551461+00:00 |
| `pose_timeline_start_at` | 2026-10-19T07:53:41.551461+00:00 |
| `pose_timeline_end_at` | 2026-10-19T07:53:50.641223+00:00 |
| `pose_offset_seconds` | **5.0** |
| `overlay_start_video_time_s` | **5.0** |
| `overlay_end_video_time_s` | **~17.034** |
| `pose_fps` | 12.0 |

**Regla de alineación (playback):**

```text
video_time_s ≈ overlay_start_video_time_s + pose_timeline_time_s
             ≈ 5.0 + pose_timeline_time_s
```

- Vídeo total: **19,5 s**.
- Pose útil en overlay: **~12,03 s** (5,0 → 17,03 s).
- **~2,5 s** de cola de vídeo **sin pose** en export (~17 → 19,5 s).

`playback_poses.json`: **177 frames**, `video_time_s` de **5,000** a **17,034**; `pose_timeline_time_s` de **0** a **12,034**. Coherente con merged (**177 filas**, span **12,034 s**).

### 4.2 Los chunks **no** son ventanas fijas de 8 s

Medidas desde `pose_frame_timestamps.npy` y `meta.json` / `diag.json`:

| Chunk | Filas `poses_full` | Span (s) | `source_frames_count` | `pose_resampled` | `mono_span_ms` (diag) |
|-------|-------------------:|---------:|------------------------:|:-----------------:|----------------------:|
| chunk_001 | 30 | 2,897 | 24 | true | 2897 |
| chunk_002 | 52 | 4,342 | 42 | true | 4342 |
| chunk_003 | **13** | **0,568** | 11 | true | **568** |
| chunk_004 | 82 | 4,008 | 82 | false | 4008 |
| **merged** | **177** | **12,034** | — | — | — |

**Comprobación densidad (chunk_002):** 4,342 s × 12 fps ≈ **52 frames** → las **52 filas cuadran** con la duración **real del segmento**, no faltan “40 frames” **dentro** de esos 4,3 s.

**Expectativa “8 s → 96 filas”:** solo aplicaría si el chunk cerrara con **`mono_span_ms ≥ 8000`** y grilla 12 fps completa. Aquí **ningún chunk** llega a 8 s.

**Gaps entre chunks (timestamps UTC):** ~0,04–0,11 s entre fin de un chunk e inicio del siguiente (no continuidad perfecta).

**Por qué existe `chunk_004` en un vídeo de 19,5 s:** el ordinal **no** es “segundos 24–32 del vídeo”. Es el **cuarto trozo publicado** del buffer (~12 s de pose repartidos en 4 publicaciones). La intuición “19/8 ≈ 3 chunks” **no aplica** al criterio de cierre actual.

### 4.3 `chunk_003` (13 frames): micro-segmento, no ventana rota de 8 s

- Duración **0,57 s**, 13 filas, `sequence_gaps_total: 2`.
- Encaja entre el final de `chunk_002` (~7,35 s en timeline pose) y el inicio de `chunk_004` (~8,03 s).
- **Pregunta abierta para C1:** ¿qué evento (flush, notify, cierre sesión parcial, stride) genera un chunk **< 1 s**?

### 4.4 `chunk_004`: cola congelada y baja calidad

| Señal | Valor |
|--------|--------|
| `valid_mask_true_ratio` | **~0,305** |
| `quality_pass` | **false** |
| `diag.issues` | `low_valid_pose_ratio`, `sequence_gaps_detected` |
| `sequence_gaps_total` | **21** |
| Últimos **10** frames `poses_full` | **Idénticos** (coords repetidas) |
| Cola `valid_mask` | **False** (p. ej. últimos 15 frames inválidos) |
| Muñecas | `zero_visible_ratio` ≈ **0,67** en muestra SAM3 |

**Interpretación:** persona **sale de plano**; el pipeline mantiene **filas temporales** con ** última pose repetida** (hold-last / relleno de rejilla). Visualizar `poses_full` sin `valid_mask` muestra **hombros fantasma congelados** — comportamiento observado por revisión humana, **confirmado numéricamente**.

**Nota contrato IAModel (doc C1 handoff):** con `pose_gap_policy: nan` / export IAModel `gap_fill: none`, se esperaría **NaN** en filas no medidas, no coords repetidas con `valid_mask=false`.

### 4.5 Inconsistencia **`merged_valid_mask` vs `valid_mask` del chunk** (regresión)

Índices en `merged` (timestamps alineados):

| Chunk | Índices en merged |
|-------|-------------------|
| chunk_001 | 0–29 |
| chunk_002 | 30–81 |
| chunk_003 | 82–94 |
| chunk_004 | 95–176 |

Para **chunk_004** (82 filas):

- `chunk_004/valid_mask`: cola **False** en frames congelados.
- `merged_valid_mask[95:177]`: **True en el 100 %** (incluida la cola).
- `merged` slice vs `chunk_004/poses_full`: **mismas coords** (incluida cola congelada).

**Conclusión:** el merge **propaga o recalcula** validez de forma **incorrecta** respecto al chunk IAModel. Cualquier consumidor de **`merged` + `merged_valid_mask`** (playback, replay entrenamiento, C2) puede **tratar como válido** lo que el chunk marca inválido.

### 4.6 Stage-1 (C2) y capa piloto (contexto FP)

En múltiples sesiones del mismo export:

- Regla: **AND @ 0.80**, `robbery_score = min(p41, p48)`.
- **`source_window: "clip"`** sobre segmentos **cortos** (a menudo **2–4 s**, 21–52 filas).
- HAR frecuente: `stop`, `walk`, `hand_to_body` → `heuristic_no_theft`.
- **`pilot_robbery_review.robbery_false_positive: true`** en muchas alarmas, pero producto puede contar **candidatos stage-1**.

`d9a71` es FP **“limpio”** (1 track, chunk_001 100 % valid): el clasificador se dispara en **persona parada**, no por multi-track.

---

## 5. Por qué “no cuadra con el vídeo” (checklist revisor)

1. Comparar vídeo desde **t=0** sin sumar **`overlay_start_video_time_s` (5 s)**.
2. Asumir **`chunk_002` = 8 s** del vídeo → en realidad **~4,34 s** de pose (**8,0–12,4 s** de vídeo).
3. Mirar **`poses_full`** sin **`valid_mask`** en salida de plano (`chunk_004`).
4. Esperar **234 filas** npy (= frames MP4) → solo **177** filas de pose en **12 s** de overlay.
5. Confundir **`merged_valid_mask`** con calidad real del chunk.

---

## 6. Implicaciones para IAModel / Campaign2

1. **No mezclar** exports `clip_buffer` crudos con clips offline en el mismo split **sin** reglas de filtrado (duración mínima, `quality_pass`, coherencia máscaras).
2. **Hard negatives** desde producción: usar **`poses_full` + `valid_mask` del chunk**, recorte **`overlay_start/end`**, descartar chunks con `valid_ratio` bajo o cola congelada (coords idénticas + mask false).
3. **Replay / eval paridad:** exigir **`mono_span_ms ≥ 8000`** o etiquetar explícitamente como **`partial_chunk`** para no comparar con val 8 s.
4. **Métricas producto:** distinguir **`motion_theft_candidate`** vs **`robbery_false_positive`** / decisión C3.
5. **Extracción Triton offline** (`triton_pose_extract`) puede seguir siendo referencia **si** el input MP4 es el mismo **`clip_buffer`** y se documenta offset; eso **no sustituye** arreglar export C1 para inferencia en vivo.

---

## 7. Preguntas para el equipo C1 (kanvis-monitoring)

### 7.1 Semántica de chunks

1. ¿Qué condición exacta cierra un chunk y asigna `chunk_id` (`track_sessions`, export continuo)? ¿Relación con `window_seconds: 8` y `stride_seconds: 4` del YAML?
2. ¿Por qué **`chunk_003`** puede publicarse con **`mono_span_ms ≈ 568`** (13 filas)? ¿Es bug, flush, o comportamiento documentado?
3. ¿Por qué hay **`chunk_004`** en una sesión con **~12 s** de pose total y **19,5 s** de vídeo? ¿Criterio de “último chunk” (`is_last_chunk` en meta sigue **false** en `chunk_004`)?

### 7.2 Alineación vídeo ↔ pose

4. Confirmar **`pose_offset_seconds = 5.0`**: ¿pre-roll fijo del buffer, latencia trigger→pose, o configurable?
5. ¿El MP4 **`clip_buffer`** debe cubrir **exactamente** `[overlay_start, overlay_end]` o intentionally incluye pre/post sin pose?
6. ¿Qué reloj manda en timestamps (`pose_frame_timestamps`: UTC epoch vs mono)? ¿Garantías de monotonicidad entre chunks?

### 7.3 Resampleo y filas objetivo

7. Con **`pose_target_frames_per_chunk: 96`** y **`pose_target_fps: 12`**, ¿cuándo se aplica **`pose_resampled: true`** y por qué **`chunk_001`** queda en **30** filas (sin llegar a 96)?
8. ¿C2 / motion models consumen **`poses.npy`**, **`poses_full`+`valid_mask`**, **`merged`**, o mezcla según `source_window: clip`?

### 7.4 Hold-last, NaN y calidad

9. Con **`pose_gap_policy: nan`** y export IAModel **`gap_fill: none`**, ¿por qué **`poses_full`** en **`chunk_004`** repite coords **10+ frames** idénticos en lugar de NaN?
10. ¿`valid_mask=false` implica **exclusión** en inferencia C2? Si sí, ¿por qué se exportan coords no-NaN en esas filas?
11. ¿`quality_pass: false`** debería **bloquear** publicación a MinIO / alarma stage-1?

### 7.5 Bug merged vs chunk (prioridad alta)

12. **¿Por qué `merged_valid_mask` es True en filas donde `chunk_004/valid_mask` es False** (mismos timestamps / coords)? ¿Ruta de código que construye merge?
13. ¿`playback_poses.json` usa merged, chunks, o recomputa `valid`? (En `d9a71`, frames finales playback `valid=True` mientras chunk_004 tail es inválida/congelada.)

### 7.6 FP producción (contexto)

14. ¿La UI / backend cuenta alarmas por **`robbery_alarm`** / **`motion_theft_candidate`** o respeta **`robbery_false_positive`** / C3?
15. ¿Se puede exportar **`debug_tracks.jsonl`** / trazas de track ID en pasillo para correlacionar multi-track?

---

## 8. Propuestas concretas (para debate e implementación)

### 8.1 Contrato export “IAModel parity”

Documentar y testear:

| Regla | Propuesta |
|-------|-----------|
| Duración mínima chunk | No publicar inferencia motion si `mono_span_ms < 8000` **salvo** flag `partial_chunk: true` en `meta.json` |
| Filas | Si `pose_resampled`, exigir **96 filas @ 12 fps** **o** no resamplear y exportar duración real con flag |
| Salida de plano | Si no hay detección válida: **NaN en `poses_full`**, `valid_mask=false`, **no** repetir coords (prohibir hold-last en export IAModel full) |
| merged | **`merged_valid_mask` = AND lógico** de máscaras chunk **por timestamp**; test regresión `d9a71` |
| Último chunk | `is_last_chunk=true` + `last_chunk_reason` cuando cierra sesión; truncar cola congelada |

### 8.2 Guardrails C2 (corto plazo)

- No alarmar si `quality_pass == false` o `valid_mask_true_ratio < 0.5` (umbral a acordar).
- No alarmar en **`partial_chunk`** (< 8 s) **o** exigir score en ventana deslizante 8 s real.
- Usar **`robbery_false_positive`** / C3 para métricas operativas.

### 8.3 Tests de regresión sugeridos

1. **`test_export_d9a71_chunk004_tail`:** últimos 10 frames: `valid_mask` false → coords NaN **o** no exportar filas; merged mask igual que chunk.
2. **`test_no_microchunk`:** ningún chunk con `mono_span_ms < 1000` salvo cierre sesión explícito.
3. **`test_playback_sync`:** `playback_poses.frames[i].video_time_s` = 5.0 + timeline dentro ε.
4. **`test_clip_buffer_coverage`:** documentar relación MP4 duration vs overlay span.

### 8.4 Artefactos a adjuntar en respuesta C1

- Log sesión `d9a71` en kanvis-monitoring (cierre chunks).
- Versión imagen / git sha kanvis-monitoring.
- Fragmento código: merge `merged_valid_mask`, export hold-last, cierre `clip_buffer`.

---

## 9. Cómo reproducir (IAModel / local)

```bash
EXPORT="/home/debian/Documentos/Kanvis/session_pose_export_20261019_104309/d9a71a34123a44d293575afec5a460cb"

# Vídeo
ffprobe -show_entries format=duration -show_entries stream=nb_frames,avg_frame_rate \
  "$EXPORT/clip_buffer.mp4"

# Span por chunk
python3 <<PY
import numpy as np
from pathlib import Path
s = Path("${EXPORT}")
for c in sorted(s.glob("chunk_*")):
    ts = np.load(c/"pose_frame_timestamps.npy")
    vm = np.load(c/"valid_mask.npy")
    print(c.name, "T=", len(ts), "span=", round(ts[-1]-ts[0], 3), "valid=", round(float(vm.mean()), 3))
PY

# Cola congelada chunk_004
python3 <<PY
import numpy as np
from pathlib import Path
s = Path("${EXPORT}/chunk_004")
pf = np.load(s/"poses_full.npy")
vm = np.load(s/"valid_mask.npy")
print("last10 identical:", sum(np.allclose(pf[i], pf[-1]) for i in range(-10, 0)))
print("valid tail:", vm[-10:].tolist())
PY

# merged_valid_mask vs chunk_004 valid_mask
python3 <<PY
import numpy as np
from pathlib import Path
root = Path("${EXPORT}")
mts = np.load(root/"merged_pose_timestamps.npy")
mvm = np.load(root/"merged_valid_mask.npy")
ts4 = np.load(root/"chunk_004"/"pose_frame_timestamps.npy")
vm4 = np.load(root/"chunk_004"/"valid_mask.npy")
i0 = int(np.argmin(np.abs(mts - ts4[0])))
print("chunk4 valid tail:", vm4[-10:].tolist())
print("merged same range:", mvm[i0 + len(ts4) - 10 : i0 + len(ts4)].tolist())
PY
```

---

## 10. Referencias cruzadas IAModel

- `docs/C1_POSE_PIPELINE_HANDOFF.md` — ventanas 8/4, hold-last, `valid_mask`, gap_fill IAModel vs interpolate YAML.
- `docs/POSE_EXPORT_IAMODEL.md` — perfil MinIO IAModel (si aplica).
- `experiments/training/campaign/KANVIS_CONTEXT_AND_RESEARCH_BRIEF.md` — ensemble 41|48 AND@0.80, gap val vs prod FP.
- `docs/AGENT_TRITON_NPY_BRIEF.md` — extracción offline Triton (no sustituye contrato C1).

---

## 11. Solicitud de respuesta

Pedimos al equipo C1:

1. **Confirmación / refutación** punto por punto de §4 (especialmente §4.5 merged vs chunk).
2. Respuestas escritas a **§7** (puede ser en PR/issue interno).
3. Plan para **§8** (prioridad: **merged_valid_mask**, hold-last en export IAModel full, semántica micro-chunks).
4. Si el comportamiento es **by design**, actualizar documentación oficial (“`chunk_00N` ≠ 8 s”) y contrato que debe consumir IAModel/C2.

---

*Documento generado a partir de análisis de exports locales 2026-10-19 (sesiones FP producción + profundización `d9a71`). Contacto lado IAModel: equipo entrenamiento / Campaign2.*
