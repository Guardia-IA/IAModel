# Mensaje para IAModel — Gold N2 sesión `77f5e97045f54f0489e894b93c77ae4f`

Copiar y pegar (adjuntar JSON indicados).

---

**Asunto:** Gold Nivel 2 piloto — casi PASS merged (2 filas mask, Δ≈0,016) — sesión `77f5e970…`

Hola,

C1 piloto (oct-2026) con Nivel 2 activo:

- `continuous.pose_merged_export_from_clip_buffer: true`
- `pose_export_iamodel_full: true`
- Re-infer **OpenCV** sobre `clip_buffer.mp4` al cierre → `merged.npy` + `chunk_*` (`pose_export_source: clip_buffer_reinfer`)
- Decode gold: **OpenCV** (`TRITON_EXTRACT_DECODE_BACKEND=opencv`, `engine=c1_like`) — alineado con C1 reinfer
- **Mismo bundle:** `yolo26s-pose_v1` en C1 y offline

**Sesión de referencia (1 persona, pasillo ~19 s):**

| Campo | Valor |
|--------|--------|
| `session_id` | `77f5e97045f54f0489e894b93c77ae4f` |
| MinIO | `s3://kanvis-chunks/sessions/cliente-demo/b1597c1f-6045-4d61-a147-75e1711aef1f/6a2c7f58-b319-4ea0-b1eb-91879fcc89e0/77f5e97045f54f0489e894b93c77ae4f/` |
| Local C1 | `/home/angel/TestKanvis/77f5e97045f54f0489e894b93c77ae4f/` |
| Offline extract | `/home/angel/TestKanvis/gold_offline_77f5e97045f5/gold/clip_buffer/user_0/` |
| `merged_T` / offline `T` | **226** (= ffprobe) |
| C2 / p1 | **OK** (`p1.json`, chunk notify al cierre N2) |

**Resultado `compare_c1_golden` (`merged_whole_clip`, align `dense_index`, ε=1e-4):**

- `pass_mask`: **false** — solo **`valid_mask_mismatch: 2`**
- `pass_coords`: **false** — `max_abs_delta_coords`: **≈ 0,01597**; en filas **both_valid** (113 filas): max **≈ 0,0115** (8 filas > ε)
- Filas mask:
  - **i=88:** offline **valid**, C1 **invalid**
  - **i=153:** C1 **valid**, offline **invalid**

**Adjuntos (wudebian):**

1. `gold_report_n2.json` — `/home/angel/TestKanvis/77f5e97045f54f0489e894b93c77ae4f/gold_report_n2.json`
2. `gold_row_diag_*.json` — `/home/angel/TestKanvis/gold_row_diag_77f5e97045f54f0489e894b93c77ae4f.json`
3. Opcional: `clip_buffer.mp4` o confirmación de que vuestro extract usó el mismo fichero

**Pedimos:**

1. Confirmar que **`run_extract.sh`** y **C1 `run_clip_buffer_reinfer_triton_pool`** usan el **mismo** preprocess/postprocess y reglas **`valid_mask`** (≥5 KP, conf 0,5, `keep_all`, sin `strict_body` extra) para explicar **88** y **153**.
2. Valorar **una sola pasada Triton** reproducible (p. ej. extract offline como fuente de verdad y C1 solo persiste, o snapshot de salida Triton en cierre) para cerrar **Δ coords ~0,01–0,02** residual.
3. Criterio de cierre acordado: ¿mantenemos **ε=1e-4** estricto o **piloto** con tolerancia coords **0,02** mientras unificamos infer? En esta sesión solo **2** filas impiden `pass_mask`.

Sesiones anteriores (`67537…` RTSP vs MP4, `167e…` larga con más mask_mm) quedan como histórico; **`77f5…`** es la primera **post-deploy** con producto C2 OK y gold merged **casi** alineado.

Gracias,  
Kanvis C1 / piloto wudebian

---

## Respuesta IAModel (oct-2026)

Recibido **`77f5e970…`** y adjuntos en wudebian. Felicitaciones: primera sesión **post-deploy** con **T=226**, decode **OpenCV** alineado, **C2/p1 OK** y merged **a 2 filas** del `pass_mask` — salto claro respecto a histórico RTSP / nvdec / P0 coords.

### 1) Preprocess, postprocess y `valid_mask` (filas **88** y **153**)

**IAModel `run_extract.sh` / `c1_like`** (referencia gold):

| Etapa | Fuente |
|--------|--------|
| Decode MP4 | OpenCV (`decode_backend=opencv`, ver [`IAMODEL_GOLD_N2_DECODE_PARITY.md`](./IAMODEL_GOLD_N2_DECODE_PARITY.md)) |
| Resize + letterbox | `pose_core.prepare_infer_image` + `letterbox` (`infer_frame_max` 640×360, `tracking_imgsz` 640) desde **`pose.yaml`** del bundle |
| Triton | `yolo26s-pose`, postprocess `decode_end2end` con **`decode_conf = max(0.05, parse_conf × 0.5)`** (`parse_conf` ≈ **0.18** → decode ≈ **0.09**) |
| Tracker | ByteTrack vía `TrackerAdapter` (`track_conf` clip yaml ≈ **0.22**, mismo `tracker_yaml` del bundle) |
| Export fila | `build_masked_pose_from_xyn` con **`min_kp_conf` = 0.5** (`triton_pose_extract_config.json` → `filters.min_kp_conf`) |
| **`valid_mask`** | `keypoint_mask` + **`frame_valid` (≥ 5 KP visibles** según reglas `pose_core.export`) — **sin** `strict_body` extra en gold extract |
| Grid temporal | **`frame_policy=keep_all`** (una fila por frame decodificado) |

**C1 `clip_buffer_reinfer`** debe usar el **mismo bundle** (`pose_bundle_id`, SHA plan, `tracker_yaml`) y umbrales **IAModel full** (**0.5 / 5 KP** vía `pose_export_iamodel_*`), no gates relajados de sesión RTSP (0.35 / 2).

**Filas 88 y 153** encajan con **umbrales en el borde**: un lado tiene **≥5 KP** exportables y el otro **4** (o conf justo bajo 0.5 en un KP) por **caja/track ligeramente distinta** en el mismo frame OpenCV, no por T distinto. Pedimos en **`gold_row_diag`**: en i=88 e i=153, conteo KP con conf ≥0.5 y coords crudas post-Triton en ambos lados.

**Checklist C1 vs offline:** mismo `parse_conf` / perfil reinfer, mismo **`pose_no_coalesce`**, misma persona dominante (equivalente a **`export_primary_user`**), **reset tracker** al inicio del MP4 como `c1_like_extract`.

### 2) Cerrar Δ coords **~0,01–0,02**

Con **decode OpenCV** alineado, el residual suele ser **inferencia + tracker** (FP16 gRPC, orden batch, estado ByteTrack, caja elegida), no normalización KP (P0 ya corrigió el ~0,49).

**Orden de acción (sin relajar ε):**

1. **Cerrar las 2 filas de máscara** — a menudo baja el max Δ global (filas mal clasificadas como both_valid / comparación en bordes).
2. **Paridad de bucle** con `c1_like_extract.py`: batch size **8**, mismo preprocess, **tracker.reset()** en frame 0, mismo `track_conf` / yaml.
3. **Una sola pasada reproducible (medio plazo):** opciones acordadas:
   - **A)** C1 invoca en cierre el **mismo** `run_extract.sh` sobre `clip_buffer` y sube ese npy como merged (gold / auditoría); prod sigue reinfer interno hasta paridad bit a bit.
   - **B)** Módulo compartido en **`kanvis-triton-pose`** (mismo `ClipExtractor` / pool reinfer) — objetivo de [`TRITON_EXTRACT_PERF.md`](./TRITON_EXTRACT_PERF.md).

IAModel **no** cambia Triton/preprocess mientras el diff sea solo C1; sí podemos añadir en compare informe **`max_delta_both_valid`** (ya lo inferís: ~0,0115) en JSON si lo pedís.

### 3) Criterio de cierre **ε = 1e-4** vs piloto **0,02**

| Ámbito | Criterio IAModel |
|--------|------------------|
| **Gold N2 merged** (`compare_c1_golden`, contrato §10) | **ε = 1e-4** en coords **[0,1]** y **`valid_mask` idéntico** en **todas** las filas i↔i — **sin waiver** para dar por cerrado N2 numérico. |
| **Piloto operativo C2** (eventos, p1, chunks) | Puede seguir **GO** con producto OK; el gold es **regresión** aparte. |
| **Sesión 77f5** | A **2 filas** de máscara y Δ~0,016: prioridad **arreglar 88/153** (umbrales/track); re-run compare. Si tras eso Δ both_valid &lt; 1e-4 → **PASS**; si no, seguir tubo único (§2). **No** proponemos cambiar ε del script por defecto solo para piloto. |

### Próximo paso conjunto

1. C1: patch/export para filas 88/153 (log KP visibles + conf en reinfer vs offline row diag).  
2. Re-sync TestKanvis **`77f5…`** + repetir compare (offline OpenCV **sin re-generar** si MP4 igual).  
3. IAModel: archivamos **`77f5…`** como **referencia casi-PASS** en doc N2; actualizamos mensaje de cierre cuando `pass: true`.

Referencia decode: [`IAMODEL_GOLD_N2_DECODE_PARITY.md`](./IAMODEL_GOLD_N2_DECODE_PARITY.md).

---

## Acuerdo piloto — opción **A** (post gate1 reinfer, oct-2026)

**Hecho:** Tras `--reinfer` con tracker reset + batch 8, **77f5…** sin cambio (`mask_mm=2` en 88/153, `max_Δ≈0.016`, `both_valid=113`) → divergencia **persistente** pool C1 vs **`c1_like_extract`**, no artefacto MinIO pre-deploy.

**IAModel — prioridad piloto:** **Sí, opción A** para cerrar N2 merged / intercambio train↔MinIO **mientras** unificamos tubo (opción B).

| | Opción **A** (piloto) | Opción **B** (objetivo) |
|---|------------------------|-------------------------|
| Qué | Al cierre N2, C1 ejecuta **`run_extract.sh`** (o API equivalente) sobre **`clip_buffer.mp4`** y **persiste en MinIO** ese `merged*` / chunks densos (mismo npy que gold offline). | Un solo código de inferencia (vendor / pool = `c1_like`). |
| Gold | `compare_c1_golden`: MinIO vs offline del **mismo** comando → **PASS** (misma fuente lógica; ideal: checksum o copia del artefacto subido). | PASS con reinfer C1 nativo sin subprocess. |
| RTSP live | Sin cambio; solo export **denso al cierre** desde extract. | Paridad total live + fichero. |

**Requisitos mínimos C1 (A):**

1. Subproceso con **`TRITON_EXTRACT_DECODE_BACKEND=opencv`**, `engine=c1_like`, bundle **`yolo26s-pose_v1`**, **`keep_all`**.  
2. Meta sesión/chunk: p. ej. **`pose_export_source=triton_pose_extract_at_close`** (+ hash o ruta del run) además de `clip_buffer_reinfer` si aplica.  
3. Chunks densos = **recorte** sobre la timeline del extract (índices documentados), no cola RTSP histórica.  
4. **`playback_start_at` / chunk_elapsed** cuando publiquéis sync (chunk gold aparte de merged).

**IAModel:** mantenemos **ε=1e-4** en `compare_c1_golden`; con A el cierre N2 merged es **demostrable** en piloto. Seguimos **B** en roadmap (`TRITON_EXTRACT_PERF.md` / `kanvis-triton-pose`).

**Mensaje C1 (una línea):** Aceptamos **priorizar A en piloto**; reinfer pool queda en paralelo para converger a B; **77f5** sigue referencia del gap pool vs extract hasta primer upload 100 % extract.
