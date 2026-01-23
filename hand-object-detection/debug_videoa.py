"""
Debug para videoa.mp4: Ver qué detecta el detector
"""

import cv2
import numpy as np
from ultralytics import YOLO

print("Cargando modelos...")
yolo_pose = YOLO('yolov8n-pose.pt')
yolo_seg = YOLO('yolov8n-seg.pt')

video_path = '/home/debian/Vídeos/videoa.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: No se pudo abrir {video_path}")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total_frames} frames")
print("Analizando frames clave...\n")

frame_count = 0
sample_frames = [10, 20, 30, 40, 50, 60, 70, 80, 90, total_frames // 2]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count in sample_frames or frame_count % 15 == 0:
        print(f"\n=== Frame {frame_count} ===")
        
        # Detectar personas y muñecas
        pose_results = yolo_pose(frame, conf=0.3, verbose=False)
        left_wrist = None
        right_wrist = None
        left_shoulder = None
        right_shoulder = None
        
        if pose_results[0].boxes is not None:
            for i, box in enumerate(pose_results[0].boxes):
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:  # Persona
                    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > i:
                        keypoints = pose_results[0].keypoints[i].xy[0].cpu().numpy()
                        confidences = pose_results[0].keypoints[i].conf[0].cpu().numpy() if hasattr(pose_results[0].keypoints[i], 'conf') else None
                        
                        if len(keypoints) > 10:
                            # Hombros
                            for shoulder_idx in [5, 6]:
                                if (keypoints[shoulder_idx][0] > 0 and keypoints[shoulder_idx][1] > 0 and
                                    (confidences is None or confidences[shoulder_idx] > 0.3)):
                                    shoulder_pos = tuple(map(int, keypoints[shoulder_idx]))
                                    if shoulder_idx == 5:
                                        left_shoulder = shoulder_pos
                                    elif shoulder_idx == 6:
                                        right_shoulder = shoulder_pos
                            
                            # Muñecas
                            for wrist_idx in [9, 10]:
                                if (keypoints[wrist_idx][0] > 0 and keypoints[wrist_idx][1] > 0 and
                                    (confidences is None or confidences[wrist_idx] > 0.3)):
                                    wrist_pos = tuple(map(int, keypoints[wrist_idx]))
                                    if wrist_idx == 9:
                                        left_wrist = wrist_pos
                                    elif wrist_idx == 10:
                                        right_wrist = wrist_pos
                                    print(f"  Muñeca {'IZQUIERDA' if wrist_idx == 9 else 'DERECHA'}: {wrist_pos}")
        
        # Segmentar persona
        seg_results = yolo_seg(frame, conf=0.3, classes=[0], verbose=False)
        person_mask = None
        
        if seg_results[0].masks is not None and len(seg_results[0].masks) > 0:
            h, w = frame.shape[:2]
            mask = seg_results[0].masks.data[0].cpu().numpy()
            person_mask = cv2.resize(mask, (w, h))
            person_mask = (person_mask > 0.5).astype(np.uint8) * 255
            mask_area = np.sum(person_mask > 0)
            print(f"  Máscara de persona: {mask_area} píxeles")
        
        # Calcular tamaño de manos
        if person_mask is not None:
            if left_wrist and left_shoulder:
                wx, wy = left_wrist
                sx, sy = left_shoulder
                dist = np.sqrt((wx - sx)**2 + (wy - sy)**2)
                radius = int(dist * 0.3)
                x1 = max(0, wx - radius)
                y1 = max(0, wy - radius)
                x2 = min(w, wx + radius)
                y2 = min(h, wy + radius)
                hand_region = person_mask[y1:y2, x1:x2]
                hand_area = np.sum(hand_region > 0)
                hand_size = hand_area / (dist + 1)
                print(f"  Mano IZQUIERDA: tamaño={hand_size:.2f}, área={hand_area}, dist_hombro={dist:.1f}")
            
            if right_wrist and right_shoulder:
                wx, wy = right_wrist
                sx, sy = right_shoulder
                dist = np.sqrt((wx - sx)**2 + (wy - sy)**2)
                radius = int(dist * 0.3)
                x1 = max(0, wx - radius)
                y1 = max(0, wy - radius)
                x2 = min(w, wx + radius)
                y2 = min(h, wy + radius)
                hand_region = person_mask[y1:y2, x1:x2]
                hand_area = np.sum(hand_region > 0)
                hand_size = hand_area / (dist + 1)
                print(f"  Mano DERECHA: tamaño={hand_size:.2f}, área={hand_area}, dist_hombro={dist:.1f}")

cap.release()
print("\nAnálisis completado.")
