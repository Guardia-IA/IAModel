"""
Debug: Ver qué detecta YOLO en frames 100-150
"""

import cv2
import numpy as np
from ultralytics import YOLO

print("Cargando modelos YOLO...")
yolo_pose = YOLO('yolov8n-pose.pt')
yolo_objects = YOLO('yolov8n.pt')

video_path = '/home/debian/Vídeos/video0.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
target_frames = list(range(100, 151))

print(f"Analizando frames {target_frames[0]}-{target_frames[-1]}...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count in target_frames:
        print(f"\n=== Frame {frame_count} ===")
        
        # Detectar personas
        pose_results = yolo_pose(frame, conf=0.3, verbose=False)
        wrist_count = 0
        
        if pose_results[0].boxes is not None:
            for i, box in enumerate(pose_results[0].boxes):
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:  # Persona
                    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > i:
                        keypoints = pose_results[0].keypoints[i].xy[0].cpu().numpy()
                        confidences = pose_results[0].keypoints[i].conf[0].cpu().numpy() if hasattr(pose_results[0].keypoints[i], 'conf') else None
                        
                        # Muñecas (9 y 10)
                        for wrist_idx in [9, 10]:
                            if (keypoints[wrist_idx][0] > 0 and keypoints[wrist_idx][1] > 0 and
                                (confidences is None or confidences[wrist_idx] > 0.3)):
                                wrist_count += 1
                                wrist_pos = tuple(map(int, keypoints[wrist_idx]))
                                print(f"  Muñeca {wrist_idx}: {wrist_pos}")
        
        print(f"  Muñecas detectadas: {wrist_count}")
        
        # Detectar objetos
        object_results = yolo_objects(frame, conf=0.2, verbose=False)
        object_count = 0
        
        if object_results[0].boxes is not None:
            for box in object_results[0].boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                if cls != 0:  # No personas
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    obj_area = obj_width * obj_height
                    
                    if 300 < obj_area < 100000:  # Objetos pequeños/medianos
                        object_count += 1
                        print(f"  Objeto {object_count}: clase={cls}, conf={conf:.2f}, bbox=({x1},{y1})-({x2},{y2}), area={obj_area}")
        
        print(f"  Objetos detectados: {object_count}")
        
        # Visualizar
        frame_vis = frame.copy()
        
        # Dibujar muñecas
        if pose_results[0].boxes is not None:
            for i, box in enumerate(pose_results[0].boxes):
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:
                    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > i:
                        keypoints = pose_results[0].keypoints[i].xy[0].cpu().numpy()
                        for wrist_idx in [9, 10]:
                            if (keypoints[wrist_idx][0] > 0 and keypoints[wrist_idx][1] > 0):
                                wrist_pos = tuple(map(int, keypoints[wrist_idx]))
                                cv2.circle(frame_vis, wrist_pos, 20, (0, 255, 0), 5)
        
        # Dibujar objetos
        if object_results[0].boxes is not None:
            for box in object_results[0].boxes:
                cls = int(box.cls[0].cpu().numpy())
                if cls != 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 255, 0), 3)
        
        cv2.putText(frame_vis, f'Frame {frame_count}', (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Redimensionar
        h, w = frame_vis.shape[:2]
        if w > 1920:
            scale = 1920 / w
            frame_vis = cv2.resize(frame_vis, (int(w*scale), int(h*scale)))
        
        cv2.imshow('Debug YOLO', frame_vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\nAnálisis completado.")
