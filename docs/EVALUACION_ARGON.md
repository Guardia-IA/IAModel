# Evaluación Crítica de Auditoría y Optimización del Plan Maestro (Kanvis Campaign2)

**Fecha:** 16 de octubre de 2026  
**Proyecto:** Kanvis (IAModel — Stage 1 Pose Action Recognition)  
**Objetivo:** Evaluar la cadena de análisis (`TRAINING_AUDIT.md` $\rightarrow$ `TRAINING_EXPERT_RESPONSE.md` $\rightarrow$ `TRAINING2_CAMPAIGN2_MASTER_PLAN.md`), auditar las decisiones técnicas de ambos agentes y proponer mejoras de ingeniería, arquitectura y estrategia de entrenamiento para alcanzar el punto operativo óptimo ($\le 1\text{ FP/día}$ end-to-end, $\le 10\text{ FP/día}$ en Stage 1 con $\ge 66\%$ recall por evento).

---

## 1. Evaluación de los documentos previos (Claude Sonnet 5.5 vs. GPT Astra 6)

El flujo de trabajo que has seguido (usar un modelo para auditar el código y datos en crudo y otro para someter esa auditoría a un escrutinio estadístico y matemático) ha producido una base técnica de altísimo nivel. A continuación, desglosamos el valor real y los puntos ciegos de cada uno.

### 1.1 Evaluación de `TRAINING_AUDIT.md` (Claude Sonnet 5.5)
* **Calificación como auditoría de ingeniería de software y datos:** **Sobresaliente (9.5/10).**
* **Aciertos críticos:**
  * **Detección de bugs silenciosos letales:** Encontrar **H1** (`best_state = model.state_dict()` guardando una referencia mutable en lugar de una copia en memoria, provocando que todos tus modelos previos sean en realidad el último epoch sobreajustado) vale por sí solo toda la auditoría.
  * **Diagnóstico de *Data Leakage* y atajos (H2, M8):** Identificar que el split por UID de clip filtra sesiones/actores entre train y val, y demostrar con métricas AUC que variables espurias (`bbox_aspect_ratio` con $\text{AUC} = 0.775$, *jitter* con $\text{AUC} = 0.73$) separan los robos de los negativos debido a cómo se grabaron las sesiones.
  * **Inconsistencia espacio-temporal (H4, H5):** Detectar que `temporal_resize` destruye la coherencia física entre la posición y el canal de velocidad ($\Delta x \neq v \cdot \Delta t$), y demostrar matemáticamente que las aumentaciones `scale` y `shift` son anuladas por `normalize_sequence`.
  * **Anatomía de los modelos (M1):** Señalar que `out[:, -1, :]` en un BiLSTM/BiGRU ignora toda la secuencia en la rama *backward* salvo el último frame (que además suele ser *padding*), que al Transformer le falta *Positional Encoding*, y que el grafo de ST-GCN conecta articulaciones por índice secuencial en lugar de anatomía humana.
* **Puntos débiles / Excesos:**
  * Extrapoló estadísticas de una muestra local antigua (10.790 clips sin `valid_mask`) como si fueran verdades absolutas de la campaña de octubre (ej. afirmar que `drop_below_50` eliminaba las clases de empleados 7 y 13 en octubre, cuando la tabla C.6.3 demuestra que tenían 458 y 59 clips respectivamente).
  * Propuso un bloqueo de calidad demasiado agresivo (*Quality Gate* en §4.2) que descartaba el $16.9\%$ de los robos reales antes de llegar al modelo, lo cual habría hundido tu *recall* global artificialmente.

### 1.2 Evaluación de `TRAINING_EXPERT_RESPONSE.md` (GPT Astra 6)
* **Calificación como revisor científico, estadístico y de teoría de decisión:** **Excepcional (9.8/10).**
* **Qué aportó sobre la auditoría original:**
  * **Rigor matemático y desmitificación:** Corrigió con precisión quirúrgica los excesos de la auditoría. Por ejemplo: demostró que aunque el bloque TCN en `tcn_lstm` tenga un campo receptivo de 5 frames, la capa recurrente posterior sí ve toda la secuencia; corrigió el sesgo del filtro `single_user_only` demostrando que en validación excluye el **$21.10\%$ de los negativos** ($433/2052$) pero solo el **$0.31\%$ de los positivos** ($1/324$), alterando artificialmente la prevalencia y ocultando las escenas multi-persona donde ocurren muchos FP en producción.
  * **Separación del problema semántico vs. cinemático:** Identificó el mayor cuello de botella conceptual del proyecto: **guardar tu propio móvil en el bolsillo está etiquetado como Categoría 6 (Robo)**. Para un esqueleto de 8 puntos sin información del objeto, guardar un móvil propio y robar una lata de conservas son **cinemáticamente idénticos**. Si entrenas al modelo para dar positivo en "guardar móvil" y luego en la tienda real cuentas a un cliente guardando su móvil como Falso Positivo, estás penalizando al modelo por hacer exactamente lo que le enseñaste. Su propuesta de separar una cabeza auxiliar de *Concealment* (ocultación física) de la cabeza principal de *Theft* (robo confirmado) es brillante.
  * **Teoría de decisión y calibración de colas:** Sustituyó la búsqueda por fuerza bruta en grids de F1 por **estadísticos de orden sobre sesiones negativas completas** (control de riesgo no paramétrico de Neyman-Pearson) y demostró el colapso numérico de `float32` en la función `sigmoid` cuando los márgenes superan $m \ge 17$ ($\sigma(m) \equiv 1.0$).

### 1.3 Evaluación de tu síntesis: `TRAINING2_CAMPAIGN2_MASTER_PLAN.md`
* **Calificación como documento de ingeniería ejecutable:** **Muy Sólido (8.5/10), pero con riesgos operativos en I/O, particionado de datos y diseño de pérdida MIL.**
* Has logrado condensar dos documentos densísimos en un plan de acción limpio, modular (`campaign2/` sin romper `campaign/`), priorizado por fases (P0 a P4) y has añadido una pieza clave de infraestructura: **el orquestador paralelo para aprovechar los 32 GB de la RTX 5090**. Sin embargo, hay trampas técnicas en cómo está planteada la paralelización, los splits y la representación de las poses que debemos ajustar antes de que escribas la primera línea de código.

---

## 2. Análisis y mejoras a las Fases Propuestas en `MASTER_PLAN`

La estructura de Fases (Fase 0 a Fase 4) es **lógicamente impecable** porque impide mezclar cambios de datos con cambios de arquitectura. No obstante, proponemos las siguientes mejoras críticas por fase:

### 2.1 Mejoras a la Sección 4 (Paralelización en RTX 5090): El cuello de botella no es la VRAM, es la CPU/Disco
Tu plan estima meter entre 8 y 10 entrenamientos paralelos de TCN porque cada uno ocupa ~2–4 GB de VRAM en una GPU de 32 GB. **Cuidado aquí:**
1. **Fragmentación y Contextos CUDA:** Cada proceso hijo de PyTorch (`spawn`) inicializa su propio contexto de CUDA y reserva memoria para cuBLAS/cuDNN (~0.8 a 1.2 GB de overhead fijo por proceso independientemente de lo pequeño que sea el modelo). 10 procesos se comerán ~10 GB solo en contextos vacíos.
2. **Cuello de botella de I/O y CPU (*Thrashing*):** Un modelo `Masked TCN` de $0.14\text{M}$ parámetros con secuencias de $[64, 96, 62]$ hace el *forward + backward pass* en la RTX 5090 en **menos de 2 milisegundos**. Si lanzas 8 procesos haciendo lecturas de miles de archivos `.npy` pequeños desde el disco `/media/8TB/...`, aplicando remuestreo a 12 Hz, interpolación, aumentaciones y normalización en CPU al vuelo, **tendrás 8 procesos bloqueados al 0% de GPU esperando al disco y a la CPU**.
3. **Solución de Ingeniería (In-Memory Tensor Cache):**
   * $19.043\text{ clips} \times \sim 150\text{ frames medios} \times 8\text{ joints} \times 2\text{ floats (fp32)}$ ocupan **menos de 200 MB en RAM**. Incluso con todos los metadatos y máscaras pre-remuestreadas a 12 Hz, todo tu dataset cabe en **menos de 1 GB de memoria RAM del sistema**.
   * **Mejora obligatoria en `preflight_campaign2.py`:** Empaqueta todo el dataset pre-remuestreado a 12 Hz en un único archivo binario contiguo en memoria compartida (`mmap` / PyTorch Shared Memory `.pt` o HDF5 en `/dev/shm` RAM disk). Así, los workers de los 8 procesos paralelos no tocarán el disco duro durante el entrenamiento y la RTX 5090 volará al 95% de uso real, reduciendo campañas de horas a **10–15 minutos**.

### 2.2 Mejoras a la Fase 0 (Baseline Honesto)
* **El problema de alimentar ventanas de 8s a los modelos legacy (#41 y #48):**
  * En la Fase 0 propones evaluar `#41` y `#48` en modo ventana de 8s (96 frames a 12 Hz). Recuerda que `#41` fue entrenado con `seq_len=64` sobre clips completos de ~10–15s (donde el gesto de robo ocupaba una fracción central tras hacer `temporal_resize`).
  * Debes ejecutar **dos baselines en P0**:
    1. **P0.A (Clip-level grouped):** Evaluar `#41` y `#48` exactamente como se entrenaron (clip entero), pero corrigiendo el filtro `single_user_only=False` (usando el track objetivo) y aplicando el umbral congelado sobre el nuevo split agrupado. Esto mide cuánto caen las métricas solo por eliminar el *data leakage* y la exclusión de negativos multi-persona.
    2. **P0.B (Sliding-Window replay):** Evaluar `#41` y `#48` sobre ventanas deslizantes de 8s con stride 1s. Esto mide la degradación adicional por el desajuste temporal (*clip vs. window*).

### 2.3 Mejoras a la Fase 1 (Contrato de Datos, Splits y Entrenamiento)
* **Trampa en el Split (`grouped_chronological` en `campaign2_config.json`):**
  * En tu JSON has puesto `"strategy": "grouped_chronological", "ratios": [0.75, 0.125, 0.125]`.
  * **Peligro:** Si la auditoría tiene razón y el $63\%$ de tus robos son grabaciones *staged* (actores del equipo) grabadas en días concretos (ej. el 21 de diciembre) y el resto de días casi no tienen robos, un corte puramente cronológico al $75\% / 12.5\% / 12.5\%$ puede dejar a `val` o `test` con **0 robos** o con el **100% de los robos staged concentrados en un solo split**.
  * **Mejora:** Usa **`StratifiedGroupKFold` híbrido**:
    1. Agrupa por `group_id = connected_component(video_source, session_id, actor_id)`.
    2. Separa explícitamente los grupos en dos estratos: `provenance = staged` vs. `provenance = real_store`.
    3. Para los datos `real_store`, haz un corte cronológico por bloques de días asegurando que `val` y `test` tengan al menos $N$ eventos reales de robo cada uno. Para los datos `staged`, reparte los grupos por **actor/toma** de forma estratificada entre train, val y test (pero informa siempre de las métricas en `val_real` por separado de `val_staged`).
* **Falta el conjunto de Calibración/Umbral ($C / T$) en los ratios:**
  * En la sección 12 dices que elegirás el umbral con estadísticos de orden en el conjunto $T$, pero en `campaign2_config.json` solo defines 3 particiones (`[0.75, 0.125, 0.125]` = Train, Val, Test).
  * Si usas el mismo `val` ($12.5\%$) para hacer *Early Stopping* del modelo en cada época, elegir hiperparámetros, ajustar la calibración de Platt y además fijar el umbral extremo de falsos positivos, volverás a caer en el sesgo de selección (**H3**).
  * **Solución:** Usa un split de 4 vías (`70% Train / 10% Val_Dev / 10% Threshold_Calib (T) / 10% Frozen_Test (U)`) **o**, mejor aún para no perder positivos, usa `Val_Dev` para los positivos, pero inyecta en `Threshold_Calib (T)` **10.000 sesiones negativas reales de producción no etiquetadas** (revisando solo las que superen el umbral, como veremos en la sección 4).

### 2.4 Mejoras a la Fase 2, 3 y 4
* **Adelantar una parte de la Fase 3 a la Fase 1:** La auditoría semántica rápida de los ~219 clips de Categoría 14 ("meter en bolsa propia") y la identificación de qué clips de Categoría 6 son "guardar móvil propio" no debería esperar a la Fase 3. Si entrenas en la Fase 1 con la pérdida de doble cabeza (`theft` + `concealment`), necesitas que esos flags existan ya en el manifest de la Fase 1.
* **Fase 4 (Handoff a SAM3):** Además de pasar `margins` y `masks` a Stage 2, incluye en el JSON de salida de Stage 1 el **`peak_frame_index` y `active_wrist_side` (izquierda/derecha)** donde el modelo detectó el pico de activación temporal dentro de la ventana de 8s. Esto es oro puro para SAM3: en lugar de buscar objetos ocultos a ciegas en 8 segundos de vídeo, SAM3 sabrá exactamente en qué segundo y en qué mano fijarse.

---

## 3. Propuesta de Arquitecturas e Ingeniería de Características (¿Qué haríamos mejor?)

La receta de GPT Astra (`Masked Dilated TCN` de anchura 64 y 5 bloques) es excelente como punto de partida rápido y estable, pero para exprimir la información de **8 puntos corporales** y distinguir un robo de una acción cotidiana tenemos propuestas superiores tanto en representación geométrica como en arquitectura.

### 3.1 Ingeniería de Características: "Kinematic & Bone Representation" (Mucho más potente que solo coordenadas)
En el plan actual se proponen ~62 canales basados en posiciones $(x,y)$, velocidades $(v_x, v_y)$ y 6 distancias muñeca-torso. En reconocimiento de acciones por esqueleto (literatura moderna de *Skeleton Action Recognition*), las redes generalizan mucho mejor cuando les das explícitamente la **geometría articular (huesos y ángulos)** y el **espacio egocéntrico**:

1. **Sistema de Coordenadas Egocéntrico (Invariante al ángulo de las 9 cámaras):**
   * Además de centrar en el punto medio de las caderas $\mathbf{c}_{hip}$ y escalar por la longitud del torso $L_{torso}$ (distancia de $\mathbf{c}_{hip}$ al punto medio de los hombros $\mathbf{c}_{sh}$), calcula el vector unitario del eje del tronco $\mathbf{u}_{spine} = \frac{\mathbf{c}_{sh} - \mathbf{c}_{hip}}{\|\mathbf{c}_{sh} - \mathbf{c}_{hip}\|}$ y el vector de anchura de hombros $\mathbf{v}_{sh} = \mathbf{p}_{R\_sh} - \mathbf{p}_{L\_sh}$.
   * Incluye la razón de aspecto instantánea del torso $\rho_{yaw} = \frac{\|\mathbf{p}_{R\_sh} - \mathbf{p}_{L\_sh}\|}{L_{torso}}$. Cuando el cliente está de frente a la cámara, $\rho_{yaw}$ es alto (~0.6–0.8); cuando se pone **de perfil para ocultar algo con el cuerpo**, $\rho_{yaw} \to 0$. Este único escalar le da al modelo la orientación 3D proyectada de la persona respecto a la cámara.
2. **Vectores de Huesos explícitos (*Bone Vectors*):**
   * Define los vectores dirigidos anatómicos:
     * Hombro $\rightarrow$ Codo (Izq/Der): $\mathbf{b}_{upper\_arm} = \mathbf{p}_{elbow} - \mathbf{p}_{shoulder}$
     * Codo $\rightarrow$ Muñeca (Izq/Der): $\mathbf{b}_{forearm} = \mathbf{p}_{wrist} - \mathbf{p}_{elbow}$
3. **Cosenos y Ángulos Articulares (Invariantes a la escala y perspectiva):**
   * El coseno del ángulo de flexión del codo: $\cos(\theta_{elbow}) = \frac{\mathbf{b}_{upper\_arm} \cdot \mathbf{b}_{forearm}}{\|\mathbf{b}_{upper\_arm}\| \|\mathbf{b}_{forearm}\| + \epsilon}$.
   * En un robo (meter en bolsillo, cintura o bajo la sudadera), hay una firma cinemática clarísima: extensión hacia la estantería ($\cos\theta \approx 1$) seguida de flexión aguda rápida hacia el torso/cadera ($\cos\theta < 0$) y detención breve.
4. **Velocidad Relativa Muñeca-Torso (en lugar de solo velocidad en el marco global):**
   * Calcula la derivada temporal directa de las distancias muñeca-cadera y muñeca-pecho: $\frac{d}{dt}\|\mathbf{p}_{wrist} - \mathbf{p}_{hip}\|$. Mide la **velocidad de aproximación de la mano al cuerpo**, que es completamente inmune a si la persona va andando rápido o está parada.
5. **Codificación de Tiempo Físico Relativo ($\Delta t$ desde la última observación válida):**
   * Cuando hay una oclusión de varios frames (muy común cuando la mano entra dentro de una chaqueta o bolso), añade un canal por muñeca `time_since_last_seen` (en segundos, saturado a $2.0\text{s}$).
   * **Por qué es vital:** En un robo, la muñeca es visible cuando coge el producto, se mueve hacia el torso, **desaparece durante $0.3\text{--}0.8\text{s}$ porque se mete dentro de la ropa/bolsillo** (confianza $\le 0.5 \Rightarrow \text{NaN}$), y vuelve a aparecer saliendo del torso. ¡Esa desaparición localizada junto al torso no es ruido, es una señal fuertísima de ocultación!

### 3.2 Nuevas Arquitecturas Propuestas para `train_v2/models/`

En lugar de usar redes recurrentes pesadas (que sufren con secuencias con huecos) o Transformers puros sin sesgo inductivo, te propongo esta terna para la matriz de la Fase 1/2:

#### A. `MaskedMultiScaleTCN_SE` (Mejora directa sobre la TCN propuesta — ~0.18M params)
La TCN de GPT Astra usa convoluciones dilatadas de $k=3$ simples. El problema de las convoluciones dilatadas puras en series temporales ruidosas es el efecto *gridding* (agujeros entre los pasos que mira el filtro).
* **Mejora 1 (Bloques Inception/Multi-escala 1D):** En cada capa, en lugar de una sola convolución $k=3$, divide los 64 canales en 3 ramas paralelas: una rama con $k=3$ (gesto rápido / *snap*), otra con $k=5$ o dilatación doble (trayectoria media), y un *MaxPool1D* local $k=3$, concatenando y proyectando con $1\times 1$.
* **Mejora 2 (Squeeze-and-Excitation / Channel Attention):** Añade un bloque `SE-1D` ultraligero tras cada bloque residual. Permite a la red apagar dinámicamente los canales del brazo izquierdo si toda la acción ocurre en el brazo derecho.
* **Mejora 3 (Attention-MIL Temporal Pooling):** En lugar de hacer un simple `mean + max` ciego sobre los 96 frames de los 8 segundos (donde 6 segundos pueden ser la persona caminando normal y solo 2 segundos son el robo, lo que diluye el `mean`), usa un **Gated Attention Pooling** (Ilse et al., ICML 2018):
  $$a_t = \frac{\exp\left(\mathbf{w}^\top (\tanh(\mathbf{V}\mathbf{h}_t) \odot \sigma(\mathbf{U}\mathbf{h}_t))\right) \cdot m_t}{\sum_{\tau=1}^T \exp\left(\mathbf{w}^\top (\tanh(\mathbf{V}\mathbf{h}_\tau) \odot \sigma(\mathbf{U}\mathbf{h}_\tau))\right) \cdot m_\tau + \epsilon}$$
  Donde $m_t \in \{0,1\}$ es la máscara de frame válido. Esto produce automáticamente el peso de importancia $a_t$ de cada instante (que además te sirve para decirle a SAM3 en qué segundo exacto ocurrió el robo) y alimenta el clasificador con $\mathbf{z} = \sum_t a_t \mathbf{h}_t \oplus \max_{t: m_t=1} \mathbf{h}_t$.

#### B. `CTR-GCN-Lite` (Channel-wise Topology Refinement GCN para 8 puntos — ~0.12M params)
La auditoría descartó `stgcn` porque el código antiguo tenía un grafo mal programado (conectaba los puntos en fila india `0-1-2-3...`). Pero en la literatura de esqueletos, **CTR-GCN** (Chen et al., ICCV 2021) y **InfoGCN** son el estado del arte muy por encima de las TCN planas.
* Con solo $V=8$ nodos (`LS, RS, LE, RE, LW, RW, LH, RH`), la matriz de adyacencia es minúscula ($8 \times 8$).
* Define 3 subgrafos anatómicos base $\mathbf{A}_{base}$:
  1. Conexiones óseas directas: `(LS-LE, LE-LW, RS-RE, RE-RW, LS-LH, RS-RH, LS-RS, LH-RH)`.
  2. Conexiones funcionales de ocultación (*Cross-body interaction*): conecta explícitamente **ambas muñecas con ambas caderas y ambos hombros opuestos** `(LW-LH, LW-RH, LW-RS, RW-RH, RW-LH, RW-LS)`. ¡Ahí es exactamente donde ocurre el robo!
* Añade un término aprendible $\Delta \mathbf{A}(\mathbf{X})$ dependiente de la entrada (atención ligera entre los 8 nodos). Es extremadamente rápida en GPU, tiene menos parámetros que la TCN y modela de forma nativa las relaciones espaciales entre mano y bolsillo/torso.

#### C. `PatchTST-PoseLite` (Transformer por Parches Temporales — ~0.15M params)
Si pruebas un Transformer en la Fase 2, **no uses un Transformer frame a frame** ($T=96$ tokens ruidosos sin contexto local).
* Agrupa los 96 frames (a 12 Hz) en **parches locales solapados de 8 frames ($0.66\text{s}$) con stride de 4 frames ($0.33\text{s}$)** $\rightarrow$ solo **23 tokens temporales**.
* Añade *Rotary Positional Embeddings (RoPE)* o embeddings sinusoidales fijos, aplica 2 capas Transformer (`d_model=64, nhead=4`) con máscara de atención sobre parches vacíos y el mismo cabezal *Gated Attention Pooling*. Al operar sobre parches de $0.66\text{s}$, cada token ya representa un micro-movimiento coherente y la matriz de atención ($23 \times 23$) es instantánea.

---

## 4. ¿Abordaríamos el entrenamiento de otra manera? (5 Cambios Estratégicos)

Más allá de las arquitecturas, donde realmente vas a ganar la batalla de **0–1 FP/día con alto recall** es en la formulación del aprendizaje. Te propongo cinco mejoras directas sobre el diseño de entrenamiento:

### 4.1 De BCE Estándar a Optimización Directa en Bajo FPR (Smooth AP / Top-Push Loss)
En `MASTER_PLAN` se propone entrenar con `BCEWithLogitsLoss` estándar y un sampler $25\% \text{ pos} / 25\% \text{ hard neg} / 50\% \text{ neg}$.
* **El problema:** La pérdida BCE optimiza la verosimilitud media en todo el rango de probabilidades $[0, 1]$. Un modelo puede reducir muchísimo su BCE mejorando la separación de negativos fáciles (bajando su probabilidad de $0.15$ a $0.01$) mientras deja **5 negativos difíciles con score $0.95$** por encima de la mitad de tus robos. A ti no te importa lo que pase en el $99\%$ inferior de los negativos; **solo te importa el $0.1\%\text{--}1\%$ superior de los negativos más difíciles (la cola extrema)**.
* **Nuestra propuesta (Híbrido BCE + Pairwise Margin / Top-Push Loss en el Batch):**
  En cada batch (donde garantizas por el sampler p.ej. 16 bolsas positivas y 48 negativas), calcula los márgenes de sesión/bolsa $s_i^+$ (positivos) y $s_j^-$ (negativos) y añade un término de **Ranking en la Cola Superior (Top-$k$ Hard Negative Push)**:
  $$\mathcal{L}_{total} = \mathcal{L}_{BCE\_theft} + 0.2 \cdot \mathcal{L}_{BCE\_conceal} + \lambda_{push} \cdot \frac{1}{|P|} \sum_{i \in P} \text{Softplus}\left( \tau \left( \max_{j \in N_{top\_k}} s_j^- - s_i^+ + \delta \right) \right)$$
  Donde $\max_{j \in N_{top\_k}} s_j^-$ es el score de los $k=2$ negativos más altos del batch y $\delta = 1.0$ es el margen deseado.
  * **Efecto directo:** Esto obliga explícitamente al gradiente en cada iteración a empujar los robos por encima del **peor falso positivo del batch**. Alinea matemáticamente la función de pérdida con tu métrica de negocio (`Recall @ Low FPR`).

### 4.2 Corrección del MIL (Multiple Instance Learning) para evitar falsas alarmas crónicas
En `MASTER_PLAN` propones tratar cada clip de robo como una bolsa MIL de ventanas de 8s, y como alternativa probar `ablation_no_mil`.
* **El peligro de `ablation_no_mil` en clips largos:** Si un clip de categoría 6 dura 22 segundos (la persona entra al pasillo, mira 14 segundos, roba en 3 segundos y se va en 5 segundos) y entrenas sin MIL asignando `label=1` a todas las ventanas de 8s extraídas de ese clip, le estás enseñando al modelo que "caminar y mirar durante 8 segundos" es un robo (`label=1`). Eso destruye la precisión en producción.
* **Cómo implementar el MIL de forma limpia y sin coste:**
  * Para cada clip positivo de duración $T_{clip} > 8\text{s}$, extrae una bolsa de $M$ ventanas solapadas (ej. $M=5$ ventanas cubriendo todo el clip).
  * Pasa las $M$ ventanas por la red en paralelo para obtener los logits $[z_1, \dots, z_M]$ y además usa los pesos de atención temporal interna $a_{m,t}$ del *Gated Attention Pooling* (§3.2.A).
  * El logit de la bolsa es $z_{bag} = \text{LogSumExp}_{m=1..M}(z_m / T_{mil}) \cdot T_{mil}$.
  * **Regularización de Esparsidad Temporal (Clave):** Como sabemos que el gesto de ocultar dura entre $1.5\text{s}$ y $4\text{s}$, añade una penalización suave sobre las activaciones dentro del clip positivo para que el modelo no dé score alto en *todas* las ventanas del clip positivo, sino solo en las $1\text{--}2$ ventanas donde ocurre el gesto.
  * **Y para los negativos:** En un clip negativo verificable (categorías 0, 1, 2, 3, 4, 5, 7, 9, 13, 14), **todas** las sub-ventanas son negativas con certeza (`label=0`). Por tanto, el máximo de un clip negativo ($z_{bag}^- = \max_m z_m^-$) debe ser negativo. ¡Esto entrena directamente al agregador `max` que vas a usar en producción!

### 4.3 Explotación de Sesiones de Producción No Etiquetadas ("Zero-Cost Hard Negative Mining")
Para medir y entrenar un sistema que debe dar $\le 1\text{ FP}$ cada $1.000$ sesiones, los $1.619$ clips negativos de validación se quedan cortos. Pero tienes una ventaja enorme en una tienda real: **el $99.9\%$ de los clientes no roban**.
* Toma **10 o 15 días de grabaciones continuas de producción** de las 9 cámaras (~10.000–15.000 sesiones de tracks reales).
* No necesitas etiquetar esas 15.000 sesiones a mano. Pasa tu modelo de la Fase 1 por las 15.000 sesiones.
* Supongamos que el modelo dispara **150 alarmas** en esas 15.000 sesiones. Un humano tarda **menos de 25 minutos** en revisar 150 vídeos de 8 segundos (a 10 segundos por vídeo) y marcar cuáles son los ~145 Falsos Positivos reales y cuáles son los ~5 robos reales (o guardados de móvil).
* **Resultado:** En media hora de trabajo humano has conseguido:
  1. Un conjunto de calibración/umbral ($T$) de **15.000 sesiones reales** con el que puedes certificar estadísticamente el presupuesto de $1\text{ FP/día}$ sin trucos matemáticos.
  2. Los **145 Hard Negatives exactos** en los que tu modelo falla en la vida real para inyectarlos en el entrenamiento de la Fase 3.

### 4.4 ¿Merece la pena re-extraer el dataset con más Keypoints? (Sí, y muy pronto)
El documento `TRAINING_AUDIT` confirma (en C.6.3) que re-extraer los ~19.000 clips con `YOLO26s-pose` en la RTX 5090 tarda solo **~1 a 1.5 días** (y además el cuello de botella era la decodificación FFmpeg en CPU, que puedes paralelizar con `NVDEC` en la GPU de la 5090 bajándolo a **menos de 4 horas**).
* Ahora mismo descartáis los keypoints `0–4` (nariz, ojos, orejas) y `13–16` (rodillas, tobillos), quedándoos solo con `5–12` (hombros, codos, muñecas, caderas), y además **no guardáis el canal de confianza continuo `conf` de cada keypoint** (solo ponéis `NaN` si `conf <= 0.5`).
* **Por qué deberías lanzar un script de re-extracción en segundo plano mientras programas el Sprint 1:**
  1. **Orientación de la cabeza (Keypoints 0–4):** Una de las diferencias humanas más universales entre un cliente guardando su propio móvil/lista de la compra y alguien cometiendo un hurto es el **barrido visual de vigilancia** (girar la cabeza a izquierda/derecha o mirar a cámara antes/durante la ocultación) frente a mirar fijamente el objeto/móvil en su mano. Con el triángulo `Nariz - Oreja Izq - Oreja Der` respecto al vector de los hombros tienes el ángulo de guiñada (*head yaw*) y cabeceo (*pitch*) gratis.
  2. **Rodillas (Keypoints 13–14):** El owner confirmó que hay robos de rodillas/agachado y empleados reponiendo estanterías bajas. Sin las rodillas, cuando alguien se agacha, el torso se acorta por perspectiva o la cadera sube respecto a la imagen, confundiendo la normalización.
  3. **Confianza continua `conf` $\in [0, 1]$:** Pasar de `conf = 0.51` (visible) a `conf = 0.49` (NaN) es un escalón binario brusco. Guardar el tensor `[T, 17, 3]` `(x, y, conf)` permite al modelo saber cuándo una muñeca está empezando a ser ocluida por una prenda o bolso.

### 4.5 Pesos EMA (*Exponential Moving Average*) durante el entrenamiento
En el plan se desactiva `EMA/SWA` inicialmente. Te recomiendo encarecidamente **activar `torch.optim.swa_utils.AveragedModel` (EMA con decay $0.995$)** desde la Fase 1.
* Son literalmente **5 líneas de código**, tiene coste computacional cero, no añade hiperparámetros que afinar, y en modelos pequeños entrenados sobre series temporales con ruido de pose y *hard negatives*, el modelo EMA suaviza drásticamente los picos de sobreconfianza en los logits de cola, reduciendo la varianza entre semillas a la mitad.

---

## 5. Hoja de Ruta Mejorada para `campaign2` (Cambios concretos sobre tu Master Plan)

Para que puedas aplicar estas mejoras directamente en tu implementación, aquí tienes la tabla de ajustes recomendados sobre tu `TRAINING2_CAMPAIGN2_MASTER_PLAN.md`:

| Componente en tu Master Plan | Estado actual en tu documento | Mejora recomendada a implementar | Impacto esperado |
| :--- | :--- | :--- | :--- |
| **Carga de datos en paralelo (§4)** | Cada job lee `.npy` de disco con 4 workers DataLoader | Pre-empaquetar en `preflight_campaign2.py` un tensor único en RAM (`/dev/shm/kanvis_dataset_12hz.pt`) compartido por todos los procesos con `mmap=True` | **Velocidad $\times 5\text{--}10$**, cero cuello de botella de disco, GPU al $>90\%$ real |
| **Estrategia de Split (§6)** | `"grouped_chronological"` puro `[0.75, 0.125, 0.125]` | **Híbrido Estratificado por Grupo:** separar `staged` (agrupado por actor/sesión) de `real` (agrupado por día/bloque), garantizando cuota de robos reales en Val y Test | Evita que `val` o `test` se queden sin robos reales por el sesgo de fechas de grabación |
| **Conjunto de Umbral $T$ (§6)** | No separado de `val` en el JSON de ratios | Añadir un dataset de **Replay Negativo de Producción** (ej. 5.000–10.000 sesiones sin etiquetar donde solo se revisan las alarmas disparadas) como conjunto $T$ | Permite certificar el umbral de $\le 1\text{--}10\text{ FP/día}$ sin sobreajustar `val` |
| **Features de Entrada (§5 Fase 1)** | ~62 canales: `x, y, vx, vy`, máscaras y 6 distancias muñeca-torso | Añadir **vectores de huesos**, **cosenos de ángulos de codo/hombro**, **orientación del torso ($\rho_{yaw}$)**, **velocidad radial muñeca$\to$torso** y **tiempo desde última vista (`dt_occluded`)** | Mayor discriminación cinemática en ocultaciones y robustez ante cambios de cámara |
| **Pooling Temporal (§5 Fase 1)** | `Masked Mean + Max` simple sobre los 96 pasos (8s) | **Gated Attention Pooling + Max** con máscara de validez | Evita que 6s de caminar diluyan un robo de 2s; exporta el instante exacto del robo para SAM3 |
| **Función de Pérdida (§5 Fase 1)** | `BCEWithLogits` (theft) + `0.2 BCE` (concealment) | Añadir término **Top-$k$ Hard-Negative Margin Push** dentro del batch + **EMA (`decay=0.995`)** | Optimiza directamente la cola extrema de Falsos Positivos en lugar de la precisión media |
| **Matriz de Modelos (§5 Fase 2)** | `Masked TCN`, `GRU pooled`, `ST-GCN`, `Transformer` | 1. `MaskedMultiScaleTCN_SE` (con Attention Pooling)<br>2. `CTR-GCN-Lite` (con grafo funcional mano-cadera/hombro)<br>3. `PatchTST-PoseLite` (parches de $0.66\text{s}$) | Arquitecturas del estado del arte específicas para esqueletos con $<0.2\text{M}$ parámetros |
| **Contrato de Handoff C2 $\rightarrow$ SAM3 (§5 Fase 4)** | JSON con márgenes, máscaras y `preprocess_hash` | Añadir **`peak_timestamp`**, **`attention_weights_12hz`** y **`active_wrist` (`left`/`right`/`both`)** derivados de la contribución de canales/atención | Reduce los FP de Stage 2 y ahorra cómputo en SAM3 al focalizar el instante y la mano |

---

## 6. Especificación de Código Lista para Implementar: El Bloque Nuclear de `train_v2`

Para facilitarte el arranque del **Sprint 3 (`train_v2`)**, aquí tienes el diseño de referencia de los dos componentes que mayor salto de calidad te darán sobre el plan base: el **Gated Attention MIL Pooling** (que resuelve el problema de localizar un gesto de 2s dentro de una ventana de 8s con máscaras) y la **Pérdida Híbrida con Top-$k$ Hard-Negative Push**.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedGatedAttentionPooling(nn.Module):
    """
    Agrega una secuencia temporal [B, C, T] con máscara de validez [B, T]
    combinando Masked Max Pooling y Gated Attention Pooling (Ilse et al., 2018).
    Devuelve además los pesos de atención temporal para localizar el instante exacto
    del gesto (Handoff para SAM3).
    """
    def __init__(self, in_channels: int, attn_dim: int = 32):
        super().__init__()
        self.attn_v = nn.Conv1d(in_channels, attn_dim, kernel_size=1)
        self.attn_u = nn.Conv1d(in_channels, attn_dim, kernel_size=1)
        self.attn_w = nn.Conv1d(attn_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: [B, C, T], mask: [B, T] bool o float (1 = válido, 0 = padding/missing)
        mask_f = mask.unsqueeze(1).float()  # [B, 1, T]
        
        # 1. Gated Attention scores
        a_v = torch.tanh(self.attn_v(x))
        a_u = torch.sigmoid(self.attn_u(x))
        scores = self.attn_w(a_v * a_u)  # [B, 1, T]
        
        # Enmascarar frames inválidos con -inf antes del softmax
        scores = scores.masked_fill(mask_f == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1) * mask_f  # [B, 1, T]
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Vector agregado por atención
        feat_attn = (x * attn_weights).sum(dim=-1)  # [B, C]
        
        # 2. Masked Max Pooling
        x_masked_for_max = x.masked_fill(mask_f == 0, -1e9)
        feat_max = x_masked_for_max.max(dim=-1).values  # [B, C]
        # Salvaguarda por si toda la secuencia estuviera vacía
        valid_seq = (mask_f.sum(dim=-1) > 0).float()  # [B, 1]
        feat_max = feat_max * valid_seq
        
        # Concatenar ambos resúmenes [B, 2*C]
        pooled = torch.cat([feat_attn, feat_max], dim=-1)
        return pooled, attn_weights.squeeze(1)


class LowFPRHybridLoss(nn.Module):
    """
    Combina BCE para robo confirmado, BCE auxiliar para ocultación cinemática (ej. móvil),
    y un término Top-k Hard Negative Push que penaliza específicamente los negativos
    con mayor score del batch para despejar la cola extrema de Falsos Positivos.
    """
    def __init__(self, conceal_weight: float = 0.2, push_weight: float = 0.5, 
                 margin: float = 1.0, top_k_neg: int = 3, tau: float = 2.0):
        super().__init__()
        self.conceal_weight = conceal_weight
        self.push_weight = push_weight
        self.margin = margin
        self.top_k_neg = top_k_neg
        self.tau = tau

    def forward(self, theft_logits: torch.Tensor, conceal_logits: torch.Tensor,
                theft_targets: torch.Tensor, conceal_targets: torch.Tensor,
                theft_known_mask: torch.Tensor, conceal_known_mask: torch.Tensor):
        """
        theft_targets: 1.0 (robo confirmado), 0.0 (negativo o móvil legítimo)
        theft_known_mask: 1.0 si el estado de robo es conocido, 0.0 si es ambiguo/unknown crop
        conceal_targets: 1.0 si hay gesto físico de guardar/ocultar (robo O móvil), 0.0 resto
        """
        # 1. Pérdida BCE en cabeza principal (Theft)
        if theft_known_mask.sum() > 0:
            bce_theft = F.binary_cross_entropy_with_logits(
                theft_logits, theft_targets, reduction='none'
            )
            loss_theft = (bce_theft * theft_known_mask).sum() / (theft_known_mask.sum() + 1e-6)
        else:
            loss_theft = theft_logits.sum() * 0.0

        # 2. Pérdida BCE en cabeza auxiliar (Concealment cinemático)
        if conceal_known_mask.sum() > 0:
            bce_conceal = F.binary_cross_entropy_with_logits(
                conceal_logits, conceal_targets, reduction='none'
            )
            loss_conceal = (bce_conceal * conceal_known_mask).sum() / (conceal_known_mask.sum() + 1e-6)
        else:
            loss_conceal = conceal_logits.sum() * 0.0

        # 3. Top-k Hard Negative Push en la cabeza de Theft
        pos_mask = (theft_targets > 0.5) & (theft_known_mask > 0.5)
        neg_mask = (theft_targets < 0.5) & (theft_known_mask > 0.5)
        
        loss_push = theft_logits.sum() * 0.0
        if pos_mask.any() and neg_mask.any():
            pos_logits = theft_logits[pos_mask]
            neg_logits = theft_logits[neg_mask]
            
            k = min(self.top_k_neg, neg_logits.numel())
            hardest_neg_logits = torch.topk(neg_logits, k=k).values.mean()
            
            # Empujar cada positivo por encima de los k peores negativos del batch + margen
            diff = hardest_neg_logits - pos_logits + self.margin
            loss_push = F.softplus(self.tau * diff).mean() / self.tau

        return loss_theft + self.conceal_weight * loss_conceal + self.push_weight * loss_push
```

---

## 7. Veredicto Final y Recomendación Ejecutiva

1. **¿Están bien las fases de `TRAINING2_CAMPAIGN2_MASTER_PLAN.md`?**
   **Sí, están muy bien estructuradas.** El orden **P0 (Medición honesta y corrección de bugs) $\rightarrow$ P1 (Contrato temporal de 8s, normalización e invariancia) $\rightarrow$ P2 (Arquitectura ligera y ensemble de semillas) $\rightarrow$ P3/P4 (Minado de Hard Negatives y Post-procesado)** es exactamente como trabaja un equipo senior de Machine Learning en producción.
2. **¿Cuáles son los 3 ajustes imprescindibles antes de arrancar?**
   * **Ajuste 1 (Infraestructura):** Pre-cargar el dataset a 12 Hz en un archivo compartido en memoria RAM (`/dev/shm`) durante el *preflight* para que los 8 procesos paralelos en la RTX 5090 no colapsen la CPU ni el disco leyendo miles de `.npy`.
   * **Ajuste 2 (Datos y Evaluación):** Asegurar que el split agrupado sea **estratificado por origen (`staged` vs `real`)** para no vaciar de robos reales los conjuntos de validación/test, y usar sesiones negativas reales de tienda como conjunto de calibración de umbral ($T$).
   * **Ajuste 3 (Modelado):** Incorporar las **variables geométricas articulares** (huesos, cosenos de flexión de codo, orientación de hombros $\rho_{yaw}$ y tiempo de oclusión de muñeca), el **Gated Attention Pooling** y la pérdida **Top-$k$ Hard Negative Push** con **EMA**. En paralelo, dejar corriendo la re-extracción a 17 keypoints con `conf` continuo para tener lista la orientación de cabeza y rodillas en la Fase 2/3.