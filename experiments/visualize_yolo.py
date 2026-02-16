import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO


def draw_skeleton_and_box(frame, keypoints, bbox, keep_kps):
    """
    Dibuja el bounding box y el esqueleto (usando los mismos puntos que KEEP_KPS).
    - keypoints: (num_kps, 2) en coordenadas XY normalizadas [0,1]
    - bbox: [x1, y1, x2, y2] en píxeles
    """
    h, w = frame.shape[:2]

    # Dibujar bounding box
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Reordenar puntos según KEEP_KPS (igual que en pose_extractor_clean)
    pts = keypoints[keep_kps]  # (N, 2) en [0,1]

    # Definir conexiones (mismo esquema que visualize_npy)
    connections = [
        (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),  # Torso superior y brazos
        (0, 6), (1, 7), (6, 7)                  # Torso a cadera y cadera
    ]

    # Dibujar conexiones
    for start, end in connections:
        x1p = int(pts[start][0] * w)
        y1p = int(pts[start][1] * h)
        x2p = int(pts[end][0] * w)
        y2p = int(pts[end][1] * h)
        cv2.line(frame, (x1p, y1p), (x2p, y2p), (255, 255, 0), 2)

    # Dibujar puntos
    for (x, y) in pts:
        px = int(x * w)
        py = int(y * h)
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)


def visualize_yolo(video_path, model_path="yolo26n-pose.pt"):
    if not os.path.exists(video_path):
        print(f"Vídeo no encontrado: {video_path}")
        return

    print(f"Usando modelo: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el vídeo: {video_path}")
        return

    # Mismos índices que KEEP_KPS en pose_extractor_clean.py
    keep_kps = [5, 6, 7, 8, 9, 10, 11, 12]
    # Mismos puntos críticos que en pose_extractor_clean.py (muñecas y codos)
    critical_kps = [7, 8, 9, 10]
    min_kp_conf = 0.5

    # Acumularemos aquí las poses (Frames, Puntos, 2)
    all_poses = []

    # Usamos el propio modelo para hacer tracking frame a frame
    results = model.track(source=video_path, persist=True, tracker="bytetrack.yaml", stream=True, verbose=False)

    frame_idx = 0
    for r in results:
        success, frame = cap.read()
        if not success:
            break

        if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            kpts = r.keypoints.xyn.cpu().numpy()   # (num_personas, num_kps, 2)
            confs = r.keypoints.conf.cpu().numpy() # (num_personas, num_kps)
            boxes = r.boxes.xyxy.cpu().numpy()     # (num_personas, 4)

            for i, track_id in enumerate(ids):
                draw_skeleton_and_box(frame, kpts[i], boxes[i], keep_kps)
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (int(boxes[i][0]), int(boxes[i][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # Para guardar en .npy: cogemos la primera persona detectada,
            # pero SOLO si todos los puntos críticos tienen buena confianza.
            first_confs = confs[0]
            valid = all(first_confs[idx] > min_kp_conf for idx in critical_kps)
            if valid:
                first_person_kpts = kpts[0][keep_kps]  # (len(keep_kps), 2)
                all_poses.append(first_person_kpts)

        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        frame_idx += 1

        # Reescalar para que la ventana no supere 1080 px en el lado mayor
        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side > 1080:
            scale = 1080.0 / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_to_show = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_to_show = frame

        cv2.imshow("YOLO Pose Debug", frame_to_show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Guardar las poses en un .npy junto al vídeo
    if all_poses:
        poses_array = np.array(all_poses)  # (Frames_detectados, Puntos, 2)
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(video_dir, f"{video_name}_poses.npy")
        np.save(out_path, poses_array)
        print(f"Guardado fichero de poses en: {out_path} con shape {poses_array.shape}")
    else:
        print("No se han detectado poses para guardar en .npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza bounding box y esqueleto YOLO sobre el vídeo.")
    parser.add_argument("video_path", type=str, help="Ruta al vídeo a visualizar")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26x-pose.pt",
        help="Ruta al modelo de pose de YOLO (por defecto: yolo26n-pose.pt)",
    )
    args = parser.parse_args()

    visualize_yolo(args.video_path, args.model)

