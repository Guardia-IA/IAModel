# Visualización web de `poses.npy` — especificación para frontend

**Objetivo:** el visor web debe reproducir la misma lógica que `experiments/visualize_npy.py` (referencia canónica del repo). No es COCO-17 ni coordenadas en píxeles crudos sin normalizar.

**Referencia Python:** `experiments/visualize_npy.py` (`load_pose_array`, `draw_skeleton_pil`, `CONNECTIONS`).

---

## 1. Formato del array `.npy`

Tras cargar el tensor (API backend o parseo client-side):

| Shape | Significado |
|---|---|
| `[T, J, 2]` | **1 usuario**, T frames, J joints (J=8), coordenadas (x, y) |
| `[T, J, 3]` | 1 usuario + **confianza** por joint |
| `[T, 2, J, 2]` | **2 usuarios** por frame |
| `[T, 2, J, 3]` | 2 usuarios + confianza |

- **T** = índice de frame (0 … T−1)
- **J** = 8 keypoints del torso/brazos (no 17 COCO)
- Última dimensión: `[x, y]` o `[x, y, conf]`

Si el shape no encaja con lo anterior, **no asumir COCO**: son 8 keypoints filtrados del pipeline de entrenamiento.

---

## 2. Sistema de coordenadas (crítico)

Las coordenadas son **normalizadas 0–1** respecto al frame de vídeo:

- `x = 0` → borde izquierdo, `x = 1` → borde derecho
- `y = 0` → **arriba**, `y = 1` → **abajo** (como imagen; **sin invertir Y**)

### Proyección a canvas

Equivalente a `visualize_npy.py` con `W = H = 600`:

```javascript
const px = x * canvasWidth;
const py = y * canvasHeight;  // NO hacer (1 - y)
```

### Auto-normalización si vienen en píxeles

Si `max(x, y) > 1.5` en el clip, dividir cada eje por su máximo antes de dibujar:

```javascript
function normalizeToUnit(points) {
  // points: array de [x, y] o [x, y, conf]
  const xs = points.map(p => p[0]).filter(Number.isFinite);
  const ys = points.map(p => p[1]).filter(Number.isFinite);
  const maxX = Math.max(...xs, 0);
  const maxY = Math.max(...ys, 0);
  if (maxX <= 1.5 && maxY <= 1.5) return points;

  return points.map(([x, y, ...rest]) => [
    maxX > 1 ? x / maxX : x,
    maxY > 1 ? y / maxY : y,
    ...rest,
  ]);
}
```

---

## 3. Keypoints ausentes / inválidos

Un joint **no se dibuja** si cumple cualquiera de:

- `x` o `y` no finitos (`NaN`, `Infinity`)
- `(x === 0 && y === 0)` → marcador de **ausente** en el pipeline
- Si hay canal `conf`: `conf < 0.25` → tratar como ausente (omitir)

```javascript
function isVisible(pt) {
  if (!pt || pt.length < 2) return false;
  const x = Number(pt[0]);
  const y = Number(pt[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
  if (x === 0 && y === 0) return false;
  if (pt.length >= 3 && Number(pt[2]) < 0.25) return false;
  return true;
}
```

**No conectar líneas** entre joints invisibles.

---

## 4. Índices de los 8 joints (orden fijo)

Subconjunto YOLO COCO `KEEP_KPS = [5, 6, 7, 8, 9, 10, 11, 12]`, reindexado localmente 0…7:

| Índice | Parte del cuerpo |
|---:|---|
| 0 | Hombro izquierdo |
| 1 | Hombro derecho |
| 2 | Codo izquierdo |
| 3 | Codo derecho |
| 4 | Muñeca izquierda |
| 5 | Muñeca derecha |
| 6 | Cadera izquierda |
| 7 | Cadera derecha |

**No hay piernas** en este formato.

---

## 5. Conexiones del esqueleto

Dibujar **solo** estas aristas (idéntico a `CONNECTIONS` en Python):

```javascript
const CONNECTIONS = [
  [0, 1], // hombros
  [0, 2], [2, 4], // brazo izquierdo
  [1, 3], [3, 5], // brazo derecho
  [0, 6], [1, 7], // hombro → cadera
  [6, 7], // caderas
];
```

### Orden de dibujo por frame

1. Fondo negro `#000000`
2. Líneas entre pares visibles (grosor ~2 px)
3. Círculos en joints visibles (radio ~4 px)

---

## 6. Multi-usuario `[T, 2, J, 2]`

Por frame `t`:

| Usuario | Array | Colores (RGB) |
|---|---|---|
| Usuario 1 | `poses[t][0]` | Líneas verde `(0, 255, 0)`, puntos `(0, 255, 100)` |
| Usuario 2 | `poses[t][1]` | Líneas naranja `(255, 165, 0)`, puntos `(255, 200, 0)` |

### Un solo usuario `[T, J, 2]`

| Elemento | Color (RGB) |
|---|---|
| Líneas | Amarillo `(255, 255, 0)` |
| Puntos | Azul `(0, 0, 255)` |

---

## 7. Errores típicos del frontend

| Error | Síntoma | Corrección |
|---|---|---|
| Tratar 0–1 como píxeles | Esqueleto minúsculo en una esquina | `px = x * width`, `py = y * height` |
| Invertir Y (`1 - y`) | Persona boca abajo | Usar `y` directo |
| Esqueleto COCO-17 | Conexiones absurdas | Usar las 8 conexiones de la sección 5 |
| Ignorar `(0, 0)` y `NaN` | Líneas al origen / basura | Aplicar `isVisible()` |
| Eje temporal mal | Animación corrupta | Dimensión 0 = T (frames) |
| Confianza no filtrada | Ruido en joints fantasma | Omitir si `conf < 0.25` |

---

## 8. Pseudocódigo de referencia (1 usuario, 1 frame)

```javascript
const CONNECTIONS = [
  [0, 1], [0, 2], [2, 4], [1, 3], [3, 5], [0, 6], [1, 7], [6, 7],
];

function isVisible(pt) {
  if (!pt || pt.length < 2) return false;
  const x = Number(pt[0]), y = Number(pt[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
  if (x === 0 && y === 0) return false;
  if (pt.length >= 3 && Number(pt[2]) < 0.25) return false;
  return true;
}

function drawFrame(ctx, points, width, height) {
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, width, height);

  const toPx = ([x, y]) => [x * width, y * height];

  // Líneas
  ctx.strokeStyle = '#ffff00';
  ctx.lineWidth = 2;
  for (const [a, b] of CONNECTIONS) {
    if (!isVisible(points[a]) || !isVisible(points[b])) continue;
    const [x1, y1] = toPx(points[a]);
    const [x2, y2] = toPx(points[b]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  // Puntos
  ctx.fillStyle = '#0000ff';
  for (const pt of points) {
    if (!isVisible(pt)) continue;
    const [x, y] = toPx(pt);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}
```

### Acceso a un frame

```javascript
// Shape [T, J, 2] — un usuario
const framePoints = poses[frameIndex]; // array de J puntos [x,y] o [x,y,conf]

// Shape [T, 2, J, 2] — dos usuarios
const user0 = poses[frameIndex][0];
const user1 = poses[frameIndex][1];
```

---

## 9. Validación

Comparar frame a frame con el visualizador Python:

```bash
python experiments/visualize_npy.py /ruta/poses.npy --fps 20
```

Si el web coincide en postura, orientación y conexiones, la interpretación es correcta.

---

## 10. Resumen rápido (checklist)

- [ ] Shape correcto: `[T, 8, 2]` o variantes multi-usuario / con conf
- [ ] Coordenadas 0–1 → multiplicar por ancho/alto del canvas
- [ ] **No** invertir eje Y
- [ ] Ocultar `(0,0)`, `NaN` y `conf < 0.25`
- [ ] 8 joints, 8 conexiones (no COCO-17)
- [ ] Fondo negro; líneas antes que puntos
- [ ] Colores distintos si hay 2 usuarios
