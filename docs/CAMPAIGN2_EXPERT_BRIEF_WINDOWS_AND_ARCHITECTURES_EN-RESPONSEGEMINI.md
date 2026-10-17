## Concrete proposal (binding recommendation)

### Production temporal contract
- window_seconds: 8.0
- stride_seconds: 4.0 (con evaluación en 4 fases de offset 0, 1, 2, 3 s en benchmark offline)
- target_hz: 12.0 (verificando que C1 emita rejilla temporal estricta de 83.33 ms/paso y no FPS nativos)
- seq_len: 96 (= 12 × 8)
- Short-fragment policy (<8 s):
  1. Si la sesión total dura >= 8.0 s: PROHIBIDO emitir un último chunk truncado (<8 s); re-anclar la última ventana a [T_end - 8.0 s, T_end] (96 frames completos).
  2. Si la sesión total dura entre 4.0 s y 7.9 s (48 a 95 frames): rellenar a la derecha hasta 96 con valid_mask=False, evaluar con máscara explícita y exigir un margen adicional de umbral (+0.25 logits calibrado en T).
  3. Si la sesión tiene < 4.0 s (< 48 frames válidos): abstención directa (score = -inf, no dispara alarma).

### Training protocol
- Window lengths used in training: 8.0 s (96 pasos) como base, con Truncation Augmentation en el 25% de los batches (enmascarando aleatoriamente entre 1.0 s y 4.0 s en un extremo para simular sesiones de 4 a 7 s).
- Multi-scale: No muestreo multi-escala en clips débiles; la invariancia a fragmentos de 4–7 s se logra mediante Truncation Augmentation con máscara.
- Stride in training: Muestreo aleatorio continuo del frame de inicio (con padding en bordes de hasta 2.5 s) para que el gesto de 3 s caiga uniformemente al principio, centro o final de la ventana de 8 s; ponderación inversa al número de ventanas por clip.
- Label policy on class-6 clips:
  1. Ventana POSITIVA (y_theft = 1) si solapa >= 1.5 s con el intervalo de ocultación (últimos 3.5 s por defecto o [t0, t1] anotado).
  2. Ventana NEGATIVA DURA (y_theft = 0) si solapa 0.0 s y dista >= 1.0 s del intervalo de ocultación (fases de aproximación/caminata del propio clip de robo).
  3. Ventana IGNORADA (loss_weight = 0 en theft) si el solapamiento es parcial (0.1 s a 1.4 s).
  4. Clips auditados como "móvil/objeto propio": y_theft = 0 (o ignorado en theft) y y_concealment = 1 en la cabeza auxiliar.
- Pose source: Entrenar principalmente sobre `poses_full.npy` + `valid_mask.npy` (que es lo que existe en producción). Descartar `poses.npy` filtrado salvo que C1 replique exactamente ese filtro en tiempo real.

### Evaluation protocol (mandatory before deploy)
- Splits: Split cronológico y agrupado por componente conexa de sesión/video_source/actor con embargo temporal >= 24 h:
  - `train` (~70%)
  - `threshold_calib (T)` (~10% de vídeos etiquetados + >= 1,500 sesiones negativas reales de tienda sin alarma)
  - `val_dev` (~10% de vídeos etiquetados + >= 1,500 sesiones negativas reales)
  - `test_holdout` (~10% final cronológico, intocable hasta elegir el modelo único final)
- Metrics:
  - Primaria de despliegue: Recall de Eventos @ <= 10 sesiones FP por cada 1,000 sesiones negativas en Replay.
  - Primaria de selección de checkpoint (early stopping): Media de Recall @ {5, 10, 20} FP / 1,000 sesiones (evita ruido de discretización en un solo punto).
  - Diagnóstico secundario: Curva hacia <= 1 FP / 1,000 sesiones, retraso desde el gesto hasta la alarma y desglose de FP por categoría (cat 14, 3, 4, 2, fondo).
- Multi-length eval grid: [8.0 s, 5.0 s, 4.0 s, 6.0 s, 3.0 s] usando los mismos pesos entrenados a 8 s, con `seq_len = 12 × segundos` y verificación de equivalencia exacta entre tensor nativo y tensor acolchado a 96 con máscara.
- Production replay: SÍ, obligatorio. Mínimo 3,000 sesiones negativas reales no recortadas a mano + todos los eventos de robo de validación/test, ejecutados con `stride = 4.0 s`, regla de re-anclaje `[T-8, T]`, cooldown de sesión de 30 s y umbral congelado en `T`.

### Architecture plan (max 6 candidates)
1. `masked_tcn64` (Baseline v2, 5 bloques dilatados 1/2/4/8/16, width 64, Readout `Masked Mean + Max + TopK-18`, ~0.14M params) — 3 seeds (17, 42, 103).
2. `gru64_pooled` (BiGRU 1-2 capas width 64 con enmascaramiento estricto y Readout `Masked Mean + Max`, ~0.09M params) — 3 seeds (17, 42, 103).
3. `stgcn_tcn64_hybrid` (2 capas de Grafo Anatómico de 8 articulaciones con matriz de adyacencia aprendible residual + tronco `masked_tcn64`, ~0.18M params) — 3 seeds.
4. `ms_tcn64_se` (TCN Multi-Escala con ramas k=3 y k=5 + Squeeze-and-Excitation sobre los 62 canales, ~0.22M params) — 3 seeds.
5. `lightgbm_tabular_probe` (GBDT en CPU sobre cuantiles cinemáticos y distancias muñeca-cadera/torso por ventana) — 1 run con validación cruzada agrupada.
6. `legacy_48_v2_bridge` (Arquitectura #48 entrenada en el loader v2 con máscaras) — 3 seeds (solo como puente científico de atribución).
- Épocas y optimización: Hasta 100 épocas, AdamW (lr=3e-4, wd=1e-3), warmup 5 épocas + cosine decay, EMA de pesos (decay 0.995) para evaluación estable.
- Ensemble de producción: Media de márgenes (logits calibrados por temperatura en `T`) de las 3 semillas de `masked_tcn64` + 3 semillas de `gru64_pooled` (o solo las 3 semillas de TCN si el combinado no mejora >= 1.5 pp en `val_dev`).

### Explicitly DO NOT do (save compute)
- NO ejecutar grids de 20 a 60 arquitecturas ni combinar hiperparámetros de forma exhaustiva.
- NO usar PoseC3D, MS-G3D, ni Conformers/Transformers pesados.
- NO entrenar multi-escala (2–3 s) heredando a ciegas la etiqueta positiva de todo el clip.
- NO mantener la regla legacy `AND @ 0.80` sobre probabilidades sigmoides saturadas para los modelos v2.
- NO emitir fragmentos cortos (<8 s) al final de sesiones que duran >= 8 s (usar re-anclaje `[T-8, T]`).
- NO duplicar todos los experimentos en `poses_full.npy` vs `poses.npy` filtrado tras la Semana 1; quedarse exclusivamente con `poses_full.npy` + `valid_mask.npy`.

### 4-week execution order
- Week 1 (Infra, Paridad y Replay Baseline):
  1. Verificar timestamps de C1 (`mergedfps.npy`: confirmar 12.0 Hz reales vs 7.5 FPS) y pasar test de paridad de pose Ultralytics vs Triton en 50 clips.
  2. Terminar el run actual de `Campaign2` (`masked_tcn64` + `gru64_pooled`) y ejecutar la evaluación multi-longitud (8, 5, 4, 6, 3 s) y el test de invariancia de máscara (60 nativo vs 96 acolchado).
  3. Construir el evaluador de Session Replay (`stride = 4 s`, re-anclaje `[T-8, T]`, 3,000 sesiones negativas de producción) y medir en él el ensemble legacy `#41 + #48` como línea base real.
- Week 2 (Corrección de Etiquetas Posicionales y Minado de Fondo):
  1. Cambiar `tail_positive` fijo por muestreo de fase aleatoria + etiquetado por solapamiento (con aproximación como negativo duro y zona intermedia ignorada) + Truncation Augmentation (25%).
  2. Triage rápido de los ~2,592 clips de clase 6 para marcar "móvil/objeto propio" y anotar el intervalo `[t0, t1]` en 300 robos reales.
  3. Inyectar 10,000 ventanas negativas de sesiones largas de producción (fondo normal + *hard negatives* donde el modelo de la Semana 1 dio score alto en sesiones sin robo).
  4. Entrenar `lightgbm_tabular_probe` y `legacy_48_v2_bridge`.
- Week 3 (Entrenamiento de Candidatos Finales y Post-procesado):
  1. Entrenar con el dataset enriquecido (Semana 2) las 3 semillas de `masked_tcn64` (con TopK pooling), `gru64_pooled`, `stgcn_tcn64_hybrid` y `ms_tcn64_se` (total 12 jobs = ~2.5 horas en la RTX 5090).
  2. Calibrar temperaturas y umbrales en el split `T` a nivel de sesión y evaluar reglas de post-procesado en Replay (cooldown, penalización de fragmentos <8 s, persistencia suave).
- Week 4 (Congelación, Test Ciego y Shadow Mode):
  1. Elegir el ganador en `val_dev` (TCN 3-seed vs TCN+GRU 6-seed bajo regla Pareto de <= 2 pp).
  2. Ejecutar 1 sola vez sobre `test_holdout` cronológico con todo congelado.
  3. Desplegar en *Shadow Mode* en paralelo con `#41 + #48` durante 7–14 días en las 9 cámaras antes de conmutar las alarmas de producción.