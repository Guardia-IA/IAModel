"""
Visualiza un vídeo MP4 en Tkinter con la pose (solo parte superior: hombros, codos, muñecas, cadera).
Opcional: fichero .npy para usar poses precalculadas en lugar de YOLO.
Botón para alternar entre "vídeo + esqueleto" y "solo esqueleto".

Uso:
    python visualize_video_pose.py video.mp4
    python visualize_video_pose.py video.mp4 --all-persons   # todas las personas (YOLO o npy 4D)
    python visualize_video_pose.py video.mp4 poses.npy --all-persons
    python visualize_video_pose.py video.mp4 --npy poses.npy --model yolo11n-pose.pt
    python visualize_video_pose.py video.mp4 --extra   # detecta y pinta móvil en la mano

Nota sobre el .npy: debe corresponder al mismo clip que el vídeo.
  - El .npy viene de pose_extractor (poses.npy o poses_full.npy en user_X/).
  - Las coordenadas están normalizadas 0-1.
  - Si vídeo y .npy no coinciden: comprueba que el vídeo sea el CLIP que generó ese npy,
    no el vídeo original completo. El npy está asociado a un clip cortado (meta.json tiene
    clip_name, t_start, t_end). Usa el clip en temp_clips o regenera el clip para comparar.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
# NOTA: tkinter e ImageTk se importan de forma perezosa dentro de run_app(),
# para poder usar las funciones de este módulo (dibujo de pose, carga de npy)
# desde scripts de terminal/SSH sin entorno gráfico.

# OpenCV solo para lectura de vídeo e inferencia YOLO (no se usa imshow)
import cv2

# Índices COCO: 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
# 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
UPPER_KPS = [5, 6, 7, 8, 9, 10, 11, 12]
# Conexiones entre índices de UPPER_KPS (0..7): hombro-codo-muñeca, hombros, torso, cadera
CONNECTIONS = [(0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (0, 6), (1, 7), (6, 7)]

DISPLAY_W, DISPLAY_H = 640, 480
MIN_CONF = 0.25  # confianza mínima para dibujar un keypoint

# Tamaño de entrada para YOLO (letterbox interno de Ultralytics). Antes se escalaba a 1080p.
YOLO_IMGSZ = 640

# Detección de móvil (--extra): clase COCO "cell phone" y umbrales.
CELL_PHONE_NAME = "cell phone"
PHONE_CONF_THR = 0.25
# Un móvil se considera "en la mano" si su centro está a menos de este porcentaje
# de la dimensión mayor del frame respecto a alguna muñeca.
PHONE_HAND_DIST_RATIO = 0.18
# Índices de las muñecas dentro del array de 8 puntos (UPPER_KPS): 9->4 (izq), 10->5 (der).
WRIST_IDXS = (4, 5)
# Modo --hand-crop: lado del recorte cuadrado alrededor de la muñeca, como fracción
# de la dimensión mayor del frame. Aumenta la resolución efectiva del móvil.
HAND_CROP_RATIO = 0.25


def load_pose_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Se necesita 'ultralytics' para detección de poses. Ejecuta: pip install ultralytics")
        sys.exit(1)
    # 1) Ruta absoluta o relativa que exista tal cual.
    path = Path(model_path)
    if path.exists():
        return YOLO(str(path))
    # 2) Relativa a la carpeta del script (p. ej. engine/ o pesos locales).
    local = Path(__file__).resolve().parent / model_path
    if local.exists():
        return YOLO(str(local))
    # 3) Nombre de modelo estándar de Ultralytics (sin separador de ruta):
    #    se deja que YOLO lo descargue/resuelva automáticamente (p. ej. yolo11n.pt).
    if os.sep not in model_path and (os.altsep is None or os.altsep not in model_path):
        return YOLO(model_path)
    print(f"Modelo no encontrado: {local}")
    sys.exit(1)


def get_upper_poses(results, frame_shape, *, all_persons: bool = False):
    """
    Extrae keypoints de la parte superior del cuerpo.
    Devuelve lista de (points_xy, confs); vacía si no hay detecciones.
    points_xy: np array (8, 2) en coordenadas de imagen.
    """
    if not results or len(results) == 0:
        return []
    r = results[0]
    if r.keypoints is None or r.boxes is None:
        return []
    kpts = r.keypoints.xy.cpu().numpy()   # (N, 17, 2) o (N, 17, 3)
    confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones((kpts.shape[0], 17))
    boxes = r.boxes.xyxy.cpu().numpy()    # (N, 4)

    if kpts.shape[0] == 0:
        return []

    if all_persons:
        indices = list(range(kpts.shape[0]))
    else:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        indices = [int(np.argmax(areas))]

    out = []
    for idx in indices:
        kp = kpts[idx]
        if kp.shape[-1] >= 3:
            kp = kp[:, :2]
        conf = confs[idx] if idx < len(confs) else np.ones(17)
        upper_xy = kp[UPPER_KPS]
        upper_conf = conf[UPPER_KPS]
        out.append((upper_xy, upper_conf))
    return out


def get_upper_pose(results, frame_shape, choose_biggest: bool = True):
    """Compat: una sola persona (la de bbox más grande si hay varias)."""
    poses = get_upper_poses(results, frame_shape, all_persons=False)
    if not poses:
        return None, None
    return poses[0]


# Colores por persona (línea, punto)
PERSON_COLORS = [
    ((0, 255, 0), (0, 255, 100)),       # verde
    ((255, 165, 0), (255, 200, 0)),    # naranja
    ((0, 200, 255), (100, 220, 255)),  # cyan
    ((255, 100, 255), (255, 150, 255)),  # magenta
    ((255, 255, 0), (255, 255, 100)),  # amarillo
]


def draw_upper_skeleton(draw, points_xy, confs, w: int, h: int, color_line=(0, 255, 0), color_pt=(0, 255, 100)):
    """Dibuja solo la parte superior del esqueleto. points_xy (8,2) en píxeles."""
    if points_xy is None or len(points_x= 0:
        return
    pts = np.asarray(points_xy, dtype=float)
    if confs is None:
        confs = np.ones(len(pts))
    r = 5
    for start, end in CONNECTIONS:
        if start < len(pts) and end < len(pts):
            c1, c2 = confs[start], confs[end]
            if c1 >= MIN_CONF and c2 >= MIN_CONF:
                x1, y1 = int(pts[start][0]), int(pts[start][1])
                x2, y2 = int(pts[end][0]), int(pts[end][1])
                draw.line([(x1, y1), (x2, y2)], fill=color_line, width=3)
    for i, pt in enumerate(pts):
        if confs[i] >= MIN_CONF:
            x, y = int(pt[0]), int(pt[1])
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color_pt, outline=color_pt)


def find_phone_class_id(model):
    """Busca el id de la clase 'cell phone' en los names del modelo de detección."""
    names = getattr(model, "names", None)
    if names is None:
        return None
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for k, v in items:
        if str(v).lower() == CELL_PHONE_NAME:
            return int(k)
    return None


def detect_phones(det_results, conf_thr=PHONE_CONF_THR):
    """Devuelve lista de (box_xyxy, conf) de móviles detectados en el frame."""
    phones = []
    if not det_results or len(det_results) == 0:
        return phones
    r = det_results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return phones
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    for box, conf in zip(boxes, confs):
        if conf >= conf_thr:
            phones.append((box, float(conf)))
    return phones


def crop_around(frame, cx, cy, size):
    """Recorta un cuadrado de lado `size` centrado en (cx, cy). Devuelve (crop, ox, oy)."""
    h, w = frame.shape[:2]
    half = int(size) // 2
    x1 = max(0, int(cx) - half)
    y1 = max(0, int(cy) - half)
    x2 = min(w, int(cx) + half)
    y2 = min(h, int(cy) + half)
    if x2 <= x1 or y2 <= y1:
        return None, 0, 0
    return frame[y1:y2, x1:x2], x1, y1


def phone_in_hand(phone_box, wrists_xy, frame_w, frame_h, thr_ratio=PHONE_HAND_DIST_RATIO):
    """True si el centro del móvil está cerca de alguna muñeca."""
    if not wrists_xy:
        return False
    cx = (phone_box[0] + phone_box[2]) / 2.0
    cy = (phone_box[1] + phone_box[3]) / 2.0
    thr = thr_ratio * max(frame_w, frame_h)
    for wx, wy in wrists_xy:
        if np.hypot(cx - wx, cy - wy) <= thr:
            return True
    return False


def draw_phone_box(draw, box, in_hand):
    """Dibuja el bounding box del móvil. Rojo si está en la mano, amarillo si no."""
    x1, y1, x2, y2 = [int(v) for v in box]
    color = (255, 0, 0) if in_hand else (255, 200, 0)
    label = "Movil en mano" if in_hand else "Movil"
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)


def frame_to_pil(bgr_frame):
    """Convierte frame BGR (numpy) a PIL Image RGB."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_npy_poses(npy_path: str, *, all_persons: bool = False):
    """
    Carga poses desde .npy. Devuelve [T, U, 8, 2] con coords normalizadas 0-1 (U = personas).
    Soporta: [T, 8, 2] (un usuario), [T, 2, 8, 2] (varios), [T, 17, 2] (COCO completo).
    """
    data = np.load(npy_path)

    def _to_upper(arr: np.ndarray) -> np.ndarray:
        if arr.shape[-2] == 8:
            return arr[..., :2]
        if arr.shape[-2] >= 13:
            return arr[..., np.array(UPPER_KPS), :2]
        return arr[..., :2]

    if data.ndim == 3:
        upper = _to_upper(data)
        return upper[:, np.newaxis, :, :]
    if data.ndim == 4 and data.shape[1] >= 1:
        upper = _to_upper(data)
        if all_persons:
            return upper
        return upper[:, 0:1, :, :]
    raise ValueError(f"Formato .npy no soportado: shape={data.shape}")


def _norm_to_pixels(pts_norm: np.ndarray, w_orig: int, h_orig: int):
    if pts_norm is None or np.any(np.isnan(pts_norm)):
        return None, np.zeros(8)
    points_xy = pts_norm.copy().astype(float)
    points_xy[:, 0] *= w_orig
    points_xy[:, 1] *= h_orig
    return points_xy, np.ones(8)


def _npy_frame_poses(npy_poses: np.ndarray, frame_idx: int, video_frames: int):
    """Devuelve lista de (8,2) normalizados para un frame de vídeo."""
    npy_len, n_users = npy_poses.shape[0], npy_poses.shape[1]
    if npy_len <= 0:
        return []

    idx = frame_idx % max(1, video_frames or 1)
    if video_frames and npy_len < video_frames:
        offset = video_frames - npy_len
        if idx < offset:
            return []
        j = min(max(0, idx - offset), npy_len - 1)
    elif video_frames and npy_len > video_frames:
        j = int(round(idx * (npy_len - 1) / max(1, video_frames - 1)))
        j = max(0, min(j, npy_len - 1))
    else:
        j = min(idx, npy_len - 1)

    out = []
    for u in range(n_users):
        pts = npy_poses[j, u]
        if not np.any(np.isnan(pts)):
            out.append(pts)
    return out


def run_app(video_path: str, model_path: str = "yolo11n-pose.pt", npy_path: str | None = None,
            detect_phone: bool = False, detect_model_path: str = "yolo11m.pt", imgsz: int = YOLO_IMGSZ,
            sample: int = 1, phone_conf: float = PHONE_CONF_THR,
            hand_crop: bool = False, crop_ratio: float = HAND_CROP_RATIO,
            all_persons: bool = False):
    try:
        import tkinter as tk
        from PIL import ImageTk
    except Exception as e:
        print(f"No se pudo iniciar la interfaz gráfica (tkinter): {e}")
        print("Estás en un entorno sin GUI. Para terminal/SSH usa export_skeleton_video.py.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el vídeo: {video_path}")
        return

    npy_poses = None
    if npy_path:
        npy_poses = load_npy_poses(npy_path, all_persons=all_persons)
        n_users = npy_poses.shape[1]
        print(
            f"Poses cargadas desde {npy_path}: shape={npy_poses.shape} "
            f"({n_users} persona(s)/frame)"
        )

    model = load_pose_model(model_path) if npy_poses is None else None

    detect_model = None
    phone_cls_id = None
    if detect_phone:
        detect_model = load_pose_model(detect_model_path)  # carga genérica de YOLO
        phone_cls_id = find_phone_class_id(detect_model)
        if phone_cls_id is None:
            print(f"Aviso: el modelo {detect_model_path} no tiene la clase '{CELL_PHONE_NAME}'.")
        else:
            modo = f"hand-crop (ratio={crop_ratio})" if hand_crop else "frame completo"
            print(f"Detección de móvil activada (clase {phone_cls_id}) con {detect_model_path}, imgsz={imgsz}, sample={sample}, modo={modo}")
    sample = max(1, int(sample))
    last_phone_boxes = [[]]   # contenedor mutable para reusar entre frames sin nonlocal
    last_crop_regions = [[]]  # zonas de recorte (modo --hand-crop) para dibujarlas
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(1, int(1000 / fps))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    npy_frames = len(npy_poses) if npy_poses is not None else 0

    root = tk.Tk()
    title = "Vídeo + Pose (npy)" if npy_poses is not None else "Vídeo + Pose"
    root.title(title)
    root.resizable(True, True)

    show_video = tk.BooleanVar(value=True)

    label = tk.Label(root)
    label.pack(padx=4, pady=4)

    def toggle():
        show_video.set(not show_video.get())
        # El botón indica qué verás al pulsar: "solo esqueleto" o "vídeo + esqueleto"
        btn.config(text="Ver vídeo + esqueleto" if show_video.get() else "Ver solo esqueleto")

    btn = tk.Button(root, text="Ver solo esqueleto", command=toggle)
    btn.pack(pady=4)

    status = tk.Label(root, text="", font=("", 9))
    status.pack(pady=2)

    current_frame = [0]
    paused = [False]

    def on_key(event):
        if event.keysym == "space":
            paused[0] = not paused[0]
        elif event.keysym == "Escape" or event.char == "q":
            root.quit()
            root.destroy()

    root.bind("<KeyPress>", on_key)

    def update():
        if not cap.isOpened():
            return
        if paused[0]:
            root.after(50, update)
            return

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame[0] = 0
            root.after(delay_ms, update)
            return

        h_orig, w_orig = frame.shape[:2]

        skeletons: list[tuple] = []
        if npy_poses is not None:
            for pts_norm in _npy_frame_poses(npy_poses, current_frame[0], video_frames):
                px, cf = _norm_to_pixels(pts_norm, w_orig, h_orig)
                if px is not None:
                    skeletons.append((px, cf))
        else:
            results = model(frame, imgsz=imgsz, verbose=False)
            for px, cf in get_upper_poses(results, frame.shape, all_persons=all_persons):
                skeletons.append((px, cf if cf is not None else np.ones(8)))

        # Detección de móvil: muñecas de todas las personas visibles
        phone_boxes = last_phone_boxes[0]
        if detect_model is not None and current_frame[0] % sample == 0:
            det_kwargs = {"imgsz": imgsz, "verbose": False}
            if phone_cls_id is not None:
                det_kwargs["classes"] = [phone_cls_id]
            wrists = []
            for points_xy, confs in skeletons:
                if points_xy is None or confs is None:
                    continue
                for wi in WRIST_IDXS:
                    if wi < len(confs) and confs[wi] >= MIN_CONF:
                        wrists.append((points_xy[wi][0], points_xy[wi][1]))
            phone_boxes = []
            crop_regions = []
            if hand_crop and wrists:
                # Recorte alrededor de cada muñeca: el móvil ocupa más píxeles y YOLO lo
                # detecta mejor. Lo detectado aquí ya está "en la mano" por construcción.
                crop_px = max(64, int(crop_ratio * max(w_orig, h_orig)))
                for (wx, wy) in wrists:
                    crop, ox, oy = crop_around(frame, wx, wy, crop_px)
                    if crop is None:
                        continue
                    crop_regions.append([ox, oy, ox + crop.shape[1], oy + crop.shape[0]])
                    det_results = detect_model(crop, **det_kwargs)
                    for box, _conf in detect_phones(det_results, conf_thr=phone_conf):
                        gbox = [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy]
                        phone_boxes.append((gbox, True))
                last_crop_regions[0] = crop_regions
            else:
                # Frame completo: útil cuando no hay pose/muñecas.
                det_results = detect_model(frame, **det_kwargs)
                for box, _conf in detect_phones(det_results, conf_thr=phone_conf):
                    phone_boxes.append((box, phone_in_hand(box, wrists, w_orig, h_orig)))
                last_crop_regions[0] = []
            last_phone_boxes[0] = phone_boxes

        if show_video.get():
            img = frame_to_pil(frame)
        else:
            img = Image.new("RGB", (w_orig, h_orig), (0, 0, 0))

        draw = ImageDraw.Draw(img)
        for pi, (points_xy, confs) in enumerate(skeletons):
            line_c, pt_c = PERSON_COLORS[pi % len(PERSON_COLORS)]
            draw_upper_skeleton(draw, points_xy, confs, w_orig, h_orig, line_c, pt_c)
        if all_persons and len(skeletons) > 1:
            draw.text((8, 8), f"{len(skeletons)} personas", fill=(255, 255, 255))
        for cr in last_crop_regions[0]:
            draw.rectangle([int(cr[0]), int(cr[1]), int(cr[2]), int(cr[3])], outline=(120, 120, 120), width=1)
        for box, in_hand in phone_boxes:
            draw_phone_box(draw, box, in_hand)

        img = img.resize((DISPLAY_W, DISPLAY_H), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo

        current_frame[0] += 1
        src = "npy" if npy_poses is not None else "yolo"
        n_sk = len(skeletons)
        sk_info = f" | personas: {n_sk}" if all_persons or n_sk > 1 else ""
        phone_info = ""
        if detect_model is not None:
            n_hand = sum(1 for _b, ih in phone_boxes if ih)
            phone_info = f" | móviles: {len(phone_boxes)} (en mano: {n_hand})"
        status.config(text=f"Frame {current_frame[0]} | {src}{sk_info}{phone_info} | {'vídeo+esqueleto' if show_video.get() else 'solo esqueleto'} | Esp=pausa Q=salir")
        root.after(delay_ms, update)

    root.after(0, update)
    root.mainloop()
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Visualiza vídeo MP4 con pose (parte superior) en Tkinter")
    parser.add_argument("video", type=str, help="Ruta al archivo MP4")
    parser.add_argument("npy", type=str, nargs="?", default=None, help="Opcional: fichero .npy con poses precalculadas (del mismo clip)")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="Modelo YOLO pose si no se usa .npy")
    parser.add_argument("--extra", action="store_true", help="Detecta si el usuario lleva un móvil en la mano y pinta su bounding box")
    parser.add_argument("--detect-model", dest="detect_model", type=str, default="yolo11n.pt", help="Modelo YOLO de detección de objetos para --extra (default: yolo11n.pt)")
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ, help=f"Tamaño de entrada para YOLO (default: {YOLO_IMGSZ})")
    parser.add_argument("--sample", type=int, default=1, metavar="N", help="Ejecutar la detección de móvil cada N frames (default: 1 = todos). Entre medias reutiliza el último resultado")
    parser.add_argument("--phone-conf", dest="phone_conf", type=float, default=PHONE_CONF_THR, help=f"Confianza mínima para detectar móvil (default: {PHONE_CONF_THR}). Baja el valor si no lo detecta")
    parser.add_argument("--hand-crop", dest="hand_crop", action="store_true", help="Detectar el móvil sobre un recorte alrededor de las muñecas (mejor para objetos pequeños)")
    parser.add_argument("--crop-ratio", dest="crop_ratio", type=float, default=HAND_CROP_RATIO, help=f"Lado del recorte de mano como fracción de la dimensión mayor del frame (default: {HAND_CROP_RATIO})")
    parser.add_argument(
        "--all-persons",
        action="store_true",
        help="Dibuja todas las personas (YOLO: todas las detecciones; npy 4D: todos los usuarios). "
        "Por defecto solo la de bbox más grande / user_0.",
    )
    args = parser.parse_args()

    npy_path = args.npy
    if npy_path and not os.path.exists(npy_path):
        print(f"Fichero .npy no encontrado: {npy_path}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"Archivo no encontrado: {args.video}")
        sys.exit(1)

    run_app(args.video, model_path=args.model, npy_path=npy_path,
            detect_phone=args.extra, detect_model_path=args.detect_model, imgsz=args.imgsz,
            sample=args.sample, phone_conf=args.phone_conf,
            hand_crop=args.hand_crop, crop_ratio=args.crop_ratio,
            all_persons=args.all_persons)


if __name__ == "__main__":
    main()
