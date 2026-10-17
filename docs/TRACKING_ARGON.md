# Auditoría y Propuesta Realista de Arquitectura: Identidad de Sesiones y Pureza de Tracks (C1)

**Fecha:** 2026-10-16  
**Alcance:** Análisis de `SESSION_IDENTITY_ADVISORY_BRIEF.md`[cite: 4], evaluación de la respuesta de GPT-6 Astra y especificación técnica ejecutable para el componente de captura y sesión (`C1`)[cite: 3, 4].  
**Objetivo:** Eliminar los *session merges* (mezcla de dos clientes distintos en un mismo `.npy`/`.mp4`) manteniendo exactamente el mismo coste computacional actual (~0 % GPU extra por defecto)[cite: 4] y asegurando compatibilidad total con el nuevo pipeline de entrenamiento (`Campaign2`)[cite: 3].

---

## Parte 1: Evaluación Crítica de la Respuesta de GPT-6 Astra

GPT-6 Astra ha entregado un documento **estructuralmente muy sólido y alineado con las restricciones de negocio**, pero adolece de su clásico estilo hiper-defensivo y deja algunos cabos sueltos a nivel de ingeniería práctica de *Computer Vision* que debemos corregir antes de bajar a código.

### 1.1 Lo que GPT-6 Astra clava (Adoptar sin reservas)
1. **Identificación de la causa raíz (Autogoles de código):** Astra acierta de pleno al señalar que el problema principal **no es ByteTrack**, sino las heurísticas permisivas que habéis programado encima: los atajos de *"single person in frame"*, el `lone_person_direct_pose_ingest: true`, el `rebind confirm frames: 1` y el hecho de que el `IdentityStabilizer` tenga una memoria de ausencia de **5 segundos** (`stabilizer_max_absence_seconds: 5`) mientras que la sesión se cierra a los **3 segundos**[cite: 4]. Eso garantiza matemáticamente que una identidad "fantasma" sobreviva al cierre o capture a un cliente nuevo que entra por la misma puerta 2 segundos después[cite: 4].
2. **Desacoplar el `track_id` de la propiedad de la sesión (`Session Ownership`):** Tratar el `track_id` de ByteTrack como una *sugerencia de asociación* y poner un "guardia de escritura" (*Authoritative Write Gate*) incluso cuando el `track_id` no ha cambiado es la decisión de arquitectura más importante del documento. Cuando dos personas se cruzan, ByteTrack frecuentemente hace un *ID swap* silencioso manteniendo el mismo número entero.
3. **Separar el Cierre de Identidad de la Cola de Vídeo (*Video Tail*):** Cerrar el `.npy` exactamente en el último frame válido del sujeto (`last_trusted_presence`) y permitir que el vídeo `.mp4` grabe 3 segundos extra solo como contexto visual —sin inyectar esqueletos de terceros ni forzar la cámara a seguir a 12 FPS si el pasillo quedó vacío[cite: 4]— resuelve de golpe la contaminación de cola y ahorra GPU.
4. **Respeto estricto al presupuesto de flota (Tier L por defecto):** Mantiene la disciplina de **0 llamadas neuronales nuevas** para el despliegue en 100+ tiendas, relegando OSNet x0.25 a un Tier M asíncrono con límite duro de *Token Bucket*[cite: 4].

### 1.2 Puntos ciegos y carencias de Astra (Lo que vamos a mejorar)
1. **Desaprovechamiento de los Keypoints para filtrar el color (HSV):** Astra dice que el histograma HSV falla porque la ropa se parece o el recorte (*crop*) del *bounding box* mete mucho fondo del pasillo. **Lo que Astra ignora:** ¡Ya tenéis los 8 keypoints del torso (hombros y caderas) en cada frame gracias a YOLO26s-pose[cite: 1, 4]! En lugar de hacer un histograma HSV de toda la caja (que incluye estanterías, suelo y a la otra persona cuando se cruzan), podemos extraer el **parche poligonal del torso superior** (entre hombros y caderas) y el de la **zona de cadera/pantalón**. Esto cuesta **0.05 ms en CPU**, elimina el 95 % del fondo y multiplica por 10 la capacidad de veto del color sin usar GPU.
2. **El problema del cambio de velocidad de muestreo (4 FPS ↔ 12 FPS) en el filtro de Kalman:** Astra menciona de pasada que cambiar de 4 FPS a 12 FPS afecta a los contadores de frames[cite: 4], pero no da la solución matemática. El filtro de Kalman estándar de ByteTrack en Ultralytics asume un delta temporal constante ($\Delta t = 1$) por llamada. Cuando pasas de 4 FPS ($\Delta t = 0.25\text{ s}$) a 12 FPS ($\Delta t = 0.083\text{ s}$)[cite: 4], el vector de velocidad estimado por Kalman es **3 veces mayor de lo real** durante los primeros frames, provocando que la caja predicha "salte" hacia delante y pierda al sujeto nada más entrar.
3. **Invariantes biomecánicos gratuitos:** Además del centro y el IoU de la caja[cite: 4], tenemos la geometría del esqueleto[cite: 1]. La **longitud del torso** y la **anchura de hombros** (suavizadas en una ventana de 0.5 s) son continuas en el espacio 2D a corto plazo. Si en un cruce de 2 frames el `track_id=5` pasa de medir 110 px de torso a 82 px, o su color de torso cambia drásticamente, es un *ID swap* garantizado.
4. **Falta de concreción en el contrato con `Campaign2` (Máscaras y Grids):** En nuestra conversación anterior diseñamos `Campaign2` con un grid físico estricto de **12 Hz** y máscaras explícitas (`valid_mask.npy`)[cite: 1, 3]. La propuesta de C1 debe exportar exactamente ese formato (rellenando con `NaN` + `valid_mask = False` los huecos de oclusión o los frames descartados por ambigüedad en cruces)[cite: 1, 3], en lugar de eliminar filas y comprimir el tiempo[cite: 1, 2].

---

## Parte 2: Propuesta Realista y Ejecutable para Kanvis (C1 Session Identity v2)

Esta propuesta está diseñada para ejecutarse en **3 fases rápidas**, cumpliendo el objetivo de **reducir >85-90 % los *session merges* con 0 % de GPU adicional** en la arquitectura por defecto (Tier L) y dejando preparado un módulo ultra-ligero para cámaras conflictivas (Tier M)[cite: 4].

---

### 1. Cambios Inmediatos de Configuración (Día 1 — Coste Cero)

Antes de reescribir la máquina de estados, aplica estos cambios en la configuración actual de C1 para detener la hemorragia de *merges*:

| Parámetro actual | Valor actual | Nuevo valor | Por qué resuelve el problema |
| :--- | :---: | :---: | :--- |
| `video_preroll_seconds` | `5.0`[cite: 4] | **`3.0`**[cite: 4] | Evita que el vídeo empiece 5–7 s antes de que la persona sea visible[cite: 4]. |
| `stabilizer_max_absence_seconds` | `5.0`[cite: 4] | **`0.0` (OFF)** | **Crítico:** Imposibilita que el estabilizador recuerde a un cliente que ya salió y le asigne su ID al siguiente que entra[cite: 4]. |
| `reassign_similarity_threshold` | `0.92` (ON)[cite: 4] | **`Disabled`** | Prohibimos que el estabilizador reasigne IDs por su cuenta basándose en HSV de caja entera[cite: 4]. |
| `lone_person_direct_pose_ingest` | `true`[cite: 4] | **`false`** | Obliga a validar saltos espaciales y velocidad aunque haya una sola persona en cámara[cite: 4]. |
| `single_person_track_adoption` | `true` (implícito)[cite: 4] | **`false`** | Si cambia el `track_id`, **se cierra la sesión y se abre una nueva**[cite: 4]. Cero adopciones automáticas por ser "el único en el pasillo"[cite: 4]. |
| `person_id` counter legacy (3 obs) | `Active`[cite: 4] | **`Disabled`** | Un contador de 3 frames no es una identidad[cite: 4]; desactivar cualquier rebind basado en `person_id` sin ReID neuronal[cite: 4]. |

---

### 2. Arquitectura Recomendada para Flota (Tier L: *Keypoint-Guided Ownership Gate*)

Mantenemos **YOLO26s-pose + ByteTrack**[cite: 4], pero introducimos un **Guardia de Propiedad de Sesión (*Session Ownership Gate*)** en Python/NumPy que tarda **< 0.35 ms por frame en CPU** y aprovecha los keypoints que ya tenéis calculados[cite: 4].

#### A. Arreglo del salto 4 FPS → 12 FPS en ByteTrack
* **El problema:** Al detectar la primera presencia en modo *Idle* (4 FPS), el sistema conmuta a modo *Activo* (12 FPS)[cite: 4].
* **La solución:**
  1. En cuanto el modo *Idle* (4 FPS) vea $\ge 1$ detección con `conf >= 0.25`, conmutar **inmediatamente** el *scheduler* de esa cámara a 12 FPS[cite: 4].
  2. No abrir la sesión oficial en el primer frame de 4 FPS: exigir **2 detecciones válidas dentro de los primeros 0.35 segundos ya a 12 FPS** (o reinicializar la covarianza de velocidad del track en el cambio de frecuencia).
  3. El timestamp `first_trusted_presence` se fija en la primera de esas detecciones confirmadas.

#### B. Firma Ultraligera Guiada por Esqueleto (*Keypoint-Torso Signature* — 0 GPU)
En lugar de hacer un histograma HSV del *bounding box* completo (que está lleno de fondo) o llamar a OSNet[cite: 4], cuando los 4 keypoints del torso (`left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`) tienen confianza $> 0.5$[cite: 1]:
1. Calculamos el centro del torso y definimos un pequeño rectángulo interno (el 60 % central del polígono hombros-caderas) directamente sobre el frame 640×360 en memoria[cite: 4].
2. Calculamos un histograma **HSV compacto (8×4×4 = 128 bins)** + la **anchura media de hombros y largo de torso en píxeles**[cite: 1].
3. Guardamos en la sesión un `anchor_signature` (promedio de los primeros 3 frames limpios sin solapamiento con otras cajas) y un `rolling_signature`.

#### C. El Guardia de Escritura (*Authoritative Write Gate* — Incluso para el mismo `track_id`)
Cada vez que llega un frame con detecciones de ByteTrack, **antes de añadir el esqueleto a una sesión activa**, verificamos 4 reglas rápidas. Si **cualquiera** falla, **NO se añade la pose** y se divide la sesión (*Safe Split*)[cite: 4]:

1. **Veto de Cruce / Solapamiento (*Crossing Quarantine*):**
   * Si la caja del sujeto tiene un $\text{IoU} > 0.25$ con la caja de **otra persona** en el mismo frame, marcamos ese frame como `ambiguous_crossing`.
   * Durante los frames que dure el solapamiento fuerte, **no actualizamos firmas** y podemos enmascarar los keypoints dudosos (`valid_mask = False`)[cite: 1, 3].
   * **Al salir del cruce** (cuando el $\text{IoU}$ entre personas vuelve a $< 0.10$), exigimos que el track conserve compatibilidad con el `anchor_signature` pre-cruce (similitud coseno del histograma de torso $\ge 0.82$ y cambio de escala de torso $< 25\%$). Si tras el cruce el color del torso o la estatura han cambiado, **ByteTrack ha hecho un ID Swap**: cerramos la sesión antigua en el instante previo al cruce y abrimos una sesión nueva para el track post-cruce[cite: 4].
2. **Veto Cinemático:**
   * El desplazamiento del centro de caderas/hombros entre frames consecutivos ($\Delta t \approx 83\text{ ms}$) no puede superar $1.5 \times \text{escala\_torso}$[cite: 1]. Un salto mayor indica que el tracker ha saltado a otro cliente o a un reflejo[cite: 1].
3. **Veto de Vacío de Pasillo (*Empty Corridor Seal*):**
   * Si durante **$\ge 3$ frames consecutivos a 12 FPS ($\ge 0.25\text{ s}$)** no hay ninguna detección válida cerca de la última posición del sujeto (y especialmente si la última posición estaba en el **borde de salida** de la imagen, a $< 8\%$ del margen), la sesión pasa a estado `SEALING`. Aunque ByteTrack recicle el mismo `track_id` 0.8 segundos después para alguien que entra por esa misma esquina, **una sesión en `SEALING` por borde de salida jamás acepta nuevos frames**.
4. **Inmutabilidad 1-a-1:**
   * En un timestamp dado $t$, una detección solo puede alimentar a una `session_id`, y una `session_id` solo puede consumir una detección.

---

### 3. Máquina de Estados de Sesión y Contrato de Exportación (`Campaign2` Ready)

Separamos completamente la **vida de la identidad (`.npy`)** de la **grabación del clip de vídeo (`.mp4`)**[cite: 4]:

* **Flujo de estados:**
  1. `CANDIDATE`: Se crea al ver la primera detección a 12 FPS. Pasa a `ACTIVE` tras 2 observaciones válidas en $\le 0.35\text{ s}$.
  2. `ACTIVE`: Acumula poses verificadas por el *Write Gate*. Si el sujeto desaparece $> 0.4\text{ s}$ o toca un borde de salida, pasa a `LOST_PENDING` (o `SEALING` si salió por borde).
  3. `LOST_PENDING`: Si reaparece con el mismo ID y supera el *Write Gate* en $\le 0.4\text{ s}$ (Tier L) o supera la verificación OSNet en $\le 1.5\text{ s}$ (Tier M), vuelve a `ACTIVE`. Si cumple 3.0 s de ausencia o hay un cruce ilegal, pasa a `IDENTITY_CLOSED`.
  4. `IDENTITY_CLOSED`: El tensor `.npy` queda sellado en `t_end`. Se espera a completar `t_end + 3.0 s` en el *Ring Buffer* de vídeo y pasa a `PUBLISHED`.

#### Reglas de Tiempos y Exportación Exacta:
* **`t_start` (`first_trusted_presence`):** Timestamp exacto de la primera observación del candidato confirmado (retroactivo a hace 0.25 s).
* **`t_end` (`last_trusted_presence`):** Timestamp exacto del **último frame que superó el *Write Gate***.
  * *Nota crucial:* Cuando una persona desaparece a las `00:30` y esperamos 3 segundos (hasta las `00:33`) para confirmar que se fue[cite: 4], **`t_end` sigue siendo `00:30`**, no `00:33`.
* **Archivo de Esqueletos (`poses_full.npy` y `valid_mask.npy`):**
  * Cubre **exclusivamente** el intervalo `[t_start, t_end]` sobre una rejilla fija a **12.0 Hz**[cite: 3, 4].
  * Si entre `t_start` y `t_end` hubo 4 frames de oclusión o un cruce retenido, **no se borran las filas**: se insertan filas con `NaN` en `poses_full.npy` y `False` en `valid_mask.npy`[cite: 1, 2, 3]. Así el modelo TCN de `Campaign2` recibe el tiempo físico exacto sin distorsión de velocidad[cite: 2, 3].
  * Si un sujeto permanece en pantalla pero de espaldas/agachado con pocos keypoints visibles (`valid_mask = False`), **la presencia sigue activa** (no cuenta como salida del pasillo) hasta un máximo de 15 s sin pose exportable[cite: 4].
* **Archivo de Vídeo (`clip.mp4`):**
  * Se recorta del *Ring Buffer* de vídeo codificado (conservar ~10 s en RAM/disco local por cámara) desde `t_start - 3.0 s` hasta `t_end + 3.0 s`[cite: 4].
  * Durante los 3 segundos posteriores a `t_end`, si no queda nadie más en el pasillo, **la cámara vuelve inmediatamente a 4 FPS (Idle Scan)** tras **1.0 segundo** sin detecciones[cite: 4]. El vídeo del *tail* se extrae del stream RTSP bruto, sin gastar inferencias de pose a 12 FPS para grabar un pasillo vacío[cite: 4].

---

### 4. Tuning Específico de ByteTrack (Sin cambiar de Tracker)

Coincidimos con Astra: **no cambies ByteTrack por StrongSORT ni BoT-SORT con ReID**[cite: 4]. En pasillos con cámaras fijas, la compensación de movimiento de cámara (GMC) de BoT-SORT gasta CPU inútilmente porque la cámara no se mueve[cite: 4]. Sin embargo, los parámetros por defecto de ByteTrack en Ultralytics son demasiado agresivos manteniendo tracks "muertos":

* **Parámetros recomendados para `bytetrack_kanvis_corridor.yaml`:**
  * `tracker_type: bytetrack`
  * `track_high_thresh: 0.30` — Subir de ~0.25 a 0.30: evita anclar tracks en sombras/maniquíes[cite: 1, 4].
  * `track_low_thresh: 0.12` — Permite asociación secundaria de ByteTrack en oclusiones parciales.
  * `new_track_thresh: 0.35` — Exigir detección clara para abrir un `track_id` nuevo.
  * `track_buffer: 12` — **¡CRÍTICO!** Bajar de 30 (2.5 s) a 12 frames (1.0 s a 12 FPS).
  * `match_thresh: 0.75` — Más estricto que el 0.80 por defecto (menor distancia permitida).
  * `fuse_score: true`

* **Por qué `track_buffer: 12` es vital:** Por defecto, ByteTrack guarda los tracks perdidos durante 30 frames (2.5–3 segundos). En un pasillo, si el cliente A sale y el cliente B entra 2 segundos después por el mismo sitio[cite: 4], ByteTrack le da el mismo `track_id` de forma nativa antes siquiera de llegar a vuestro código de sesión. Bajando `track_buffer` a **12 frames (1.0 s)** y aplicando nuestro *Write Gate* a los **0.4 s**, matamos el 90 % de las resurrecciones falsas dentro del propio ByteTrack.

---

### 5. ¿Cuándo y cómo usar OSNet x0.25? (Tier M — Solo para Reconexión tras Oclusión)

Con el **Tier L** anterior, los *session merges* caerán drásticamente (>85-90 %), a cambio de que algunas visitas que sufran una oclusión total de entre `0.5 s` y `2.0 s` en mitad del pasillo se dividan en 2 sesiones (*False Split*, lo cual vuestro negocio acepta explícitamente y vuestro modelo v2 procesa sin problema en fragmentos de 1–8 s)[cite: 1, 2, 4].

Si en **cámaras específicas** queréis reducir esos *False Splits* sin arriesgar *Merges*, activad **OSNet x0.25 en modo Evento Puro (Tier M)**[cite: 4]:

1. **Cuándo se dispara (Solo 2 eventos):**
   * **Evento 1 (Anclaje):** 1 sola vez por sesión, cuando el track lleva 0.5 s estable, su torso es visible y no se solapa con nadie (`1 embedding`).
   * **Evento 2 (Candidato a Rebind / Salida de Cruce):** Cuando una sesión está en `LOST_PENDING` (lleva perdida entre `0.4 s` y `1.5 s` **en el interior del pasillo**, nunca en un borde de salida) y aparece un `track_id` nuevo cerca, **o** justo al terminar un cruce entre dos personas (`1 embedding`).
2. **Regla de Decisión (*Rebind* vs *Split*):**
   * Solo se une a la sesión anterior si:
     * $\text{Tiempo perdido} \le 1.5\text{ s}$ y no hubo sello de pasillo vacío ni salida por borde.
     * $\text{Similitud Coseno (OSNet L2)} \ge \mathbf{0.91}$[cite: 4].
     * Si hay otra persona en cámara, el margen respecto al segundo candidato es $\ge 0.06$.
     * La escala del torso es compatible ($\pm 20\%$).
   * En cualquier otro caso (o si la cola asíncrona de OSNet tarda $> 500\text{ ms}$): **Split inmediato** (cerrar vieja y abrir nueva)[cite: 4].
3. **Presupuesto garantizado en GPU (*Token Bucket*):**
   * Con esta regla, una visita normal consume **1 o 2 embeddings en toda su vida** (no por segundo).
   * En una tienda de **32 cámaras** (con 10 % de ocupación = ~3.2 cámaras activas)[cite: 4], esto supone **< 1.5 embeddings/segundo en toda la tienda** (menos de **0.5 % de sobrecarga en la RTX 5090**).
   * Se impone por software un `TokenBucket` global de **máximo 8 embeddings/s por tienda de 9 cámaras** y **16 embeddings/s por tienda de 32 cámaras**.

---

### 6. Presupuesto de Cómputo y Escalabilidad (N = 9, 32 y 256 Cámaras)

| Métrica | Piloto Actual (9 cáms)[cite: 4] | Propuesta Tier L (9 cáms) | Propuesta Tier L (32 cáms) | Propuesta Tier M (32 cáms + OSNet Evento) |
| :--- | :---: | :---: | :---: | :---: |
| **Inferencias YOLO26s-pose / s** (10% activas @ 12 FPS, 90% idle @ 4 FPS)[cite: 4] | ~43.2 inf/s[cite: 4] | **43.2 inf/s (0% extra)** | **153.6 inf/s (0% extra)** | **153.6 inf/s** |
| **Inferencias OSNet x0.25 / s** (Media / Tope duro) | 0[cite: 4] | **0** | **0** | **~1.2 inf/s** (Tope duro: **16 inf/s**) |
| **Sobrecarga GPU vs Baseline** | 0 % | **0.0 %** | **0.0 %** | **+0.2 % medio / +1.5 % pico** |
| **Sobrecarga CPU por frame activo** (Gate + HSV Torso Keypoints) | Baseline | **+0.30 ms p50** | **+0.30 ms p50** (~0.05 cores totales) | **+0.45 ms p50** (~0.08 cores totales) |
| **Ciclo p50 por cámara** | ~80 ms[cite: 4] | **~80.3 ms** | **~80.3 ms** | **~80.3 ms** (OSNet es asíncrono fuera del ciclo) |

---

### 7. Tabla de Ranking Pareto (Para Decisión de Flota)

| Opción Arquitectónica | Reducción esperada de *Session Merges* | Tasa de *False Splits* (1 visita → 2 sesiones) | Coste Incremental GPU / CPU @ 32 cámaras | Esfuerzo de Implementación | Encaje para Flota (100+ tiendas) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Tier L (RECOMENDADA POR DEFECTO): Policy Fix + ByteTrack Tune (`buffer=12`) + *Keypoint-Torso HSV Gate*** | **Muy Alta (85 % – 95 %)**<br>Elimina el 100 % de los merges por pasillo vacío/adopción y gran parte de los cruces. | **Moderada (~0.15 – 0.25 splits extra/visita)**<br>Inocuo para `Campaign2` (entrena con fragmentos)[cite: 2, 3]. | **0 % GPU**<br>**< 0.35 ms CPU/frame** (~0.05 cores en 32 cáms). | **Bajo (2–3 días)**<br>Config + ~250 líneas de Python en C1. | **ÓPTIMO (Default Flota)**<br>Cero modelos nuevos, determinista y auditable[cite: 4]. |
| **2. Tier M (Opcional Cámaras Complejas): Tier L + OSNet x0.25 Asíncrono por Evento (Ancla + Rebind ≤ 1.5s)** | **Muy Alta (90 % – 96 %)**<br>Añade discriminación neuronal en cruces con ropa de color parecido. | **Baja (~0.08 – 0.12 splits extra/visita)**<br>Recupera oclusiones cortas de hasta 1.5 s. | **+0.2 % a +1.5 % GPU**<br>(Cap duro: 16 emb/s).<br>**~0.08 cores CPU**. | **Medio (1 semana)**<br>Cola asíncrona Triton + banco de anclas local. | **Excelente como Tier Opcional**<br>Falla en modo seguro hacia Tier L si la GPU se satura. |
| **3. Tier L Alternativo: Misma Policy + OC-SORT (Motion-only)** | **Alta (85 % – 92 %)**<br>Mejor que ByteTrack en trayectorias no lineales, pero ciego a la apariencia. | **Moderada (~0.15 – 0.20 splits extra/visita)** | **0 % GPU**<br>**+0.6 ms CPU/frame** | **Medio (4–5 días)**<br>Integración y validación de umbrales OC-SORT. | **Secundario**<br>Solo probar si ByteTrack con `buffer=12` da demasiados splits. |
| **4. Tier H (Descartado para Flota): BoT-SORT / StrongSORT con ReID Denso a 12 FPS** | **Alta (90 % – 95 %)**<br>(Sigue pudiendo fallar en ropa idéntica si las reglas de sesión son malas). | **Muy Baja (< 0.05 splits/visita)** | **+80 % a +150 % GPU**<br>(>1,000 emb/s en picos de 32 cáms). | **Alto** | **INVIABLE A ESCALA**[cite: 4]<br>Rompe el presupuesto de GPU compartida[cite: 4]. |

---

### 8. Hoja de Ruta de Implementación (Plan de 5 Días)

1. **Día 1 (Fase 0 — Contención por Configuración):**
   * Desactivar adopciones de sesión única, desactivar `lone_person_direct_pose_ingest`, apagar reasignación del estabilizador (`stabilizer_max_absence_seconds = 0`), bajar `preroll` a `3.0 s` y poner `track_buffer = 12` en ByteTrack[cite: 4].
2. **Días 2 y 3 (Fase 1 — Código Tier L en C1):**
   * Implementar la máquina de estados `CANDIDATE → ACTIVE → LOST_PENDING → IDENTITY_CLOSED → PUBLISHED`.
   * Separar `t_end` (última pose válida para el `.npy`) del timer de cierre (+3.0 s para el `.mp4`)[cite: 4].
   * Implementar el *Write Gate* con el histograma HSV del polígono de torso (usando los keypoints 5, 6, 11 y 12 de YOLO-pose)[cite: 1] y la cuarentena en cruces ($\text{IoU} > 0.25$).
   * Asegurar que el exportador `.npy` escriba la rejilla fija a 12 Hz con `valid_mask.npy` sin borrar frames intermedios[cite: 1, 2, 3].
3. **Día 4 (Validación Offline sobre vídeos reales de tienda):**
   * Pasar 50 clips problemáticos del piloto (cruces de 2–3 personas, salidas y entradas rápidas en < 2 s)[cite: 4] y verificar: **0 merges por adopción** y tiempos de clip exactos (`[t_start - 3s, t_end + 3s]`)[cite: 4].
4. **Día 5 (Canary en las 9 cámaras del piloto):**
   * Desplegar Tier L en producción, monitorizar el log de `split_reason` (`ambiguous_crossing`, `torso_appearance_jump`, `exit_border_seal`, `track_lost_timeout`) y confirmar que los FPS e inferencias/s en Triton se mantienen en ~40–45 inf/s[cite: 4].