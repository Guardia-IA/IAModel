"""
Debug: Ver qué detecta YOLOv9 en frames 100-150
"""

import cv2
import numpy as np
from ultralytics import YOLO

print("Cargando modelos...")
yolo_pose = YOLO('yolov8n-pose.pt')
yolo_objects = YOLO('yolov9t.pt')

video_path = '/home/debian/Vídeos/video0.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
target_frames = list(range(100, 151))

print(f"Analizando frames {target_frames[0]}-{target_frames[-1]} con YOLOv9...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count in target_frames:
        print(f"\n=== Frame {frame_count} ===")
        
        # Detectar personas y muñecas
        pose_results = yolo_pose(frame, conf=0.3, verbose=False)
        left_wrist = None
        right_wrist = None
        
        if pose_results[0].boxes is not None:
            for i, box in enumerate(pose_results[0].boxes):
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:  # Persona
                    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > i:
                        keypoints = pose_results[0].keypoints[i].xy[0].cpu().numpy()
                        confidences = pose_results[0].keypoints[i].conf[0].cpu().numpy() if hasattr(pose_results[0].keypoints[i], 'conf') else None
                        
                        if len(keypoints) > 10:
                            # Muñeca izquierda (9) y derecha (10)
                            for wrist_idx in [9, 10]:
                                if (keypoints[wrist_idx][0] > 0 and keypoints[wrist_idx][1] > 0 and
                                    (confidences is None or confidences[wrist_idx] > 0.3)):
                                    wrist_pos = tuple(map(int, keypoints[wrist_idx]))
                                    if wrist_idx == 9:
                                        left_wrist = wrist_pos
                                    elif wrist_idx == 10:
                                        right_wrist = wrist_pos
                                    print(f"  Muñeca {'IZQUIERDA' if wrist_idx == 9 else 'DERECHA'}: {wrist_pos}")
        
        # Detectar objetos con YOLOv9 (múltiples umbrales)
        for conf_threshold in [0.1, 0.15, 0.2, 0.25]:
            object_results = yolo_objects(frame, conf=conf_threshold, verbose=False)
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
                        
                        if 100 < obj_area < 200000:  # Objetos pequeños/medianos
                            object_count += 1
                            
                            # Calcular distancia a muñecas
                            obj_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                            dist_to_left = None
                            dist_to_right = None
                            
                            if left_wrist:
                                dist_to_left = np.sqrt((obj_center[0] - left_wrist[0])**2 + 
                                                      (obj_center[1] - left_wrist[1])**2)
                            if right_wrist:
                                dist_to_right = np.sqrt((obj_center[0] - right_wrist[0])**2 + 
                                                       (obj_center[1] - right_wrist[1])**2)
                            
                            dist_left_str = f"{dist_to_left:.1f}" if dist_to_left else "N/A"
                            dist_right_str = f"{dist_to_right:.1f}" if dist_to_right else "N/A"
                            print(f"    Objeto (conf>={conf_threshold:.2f}): clase={cls}, conf={conf:.3f}, "
                                  f"bbox=({x1},{y1})-({x2},{y2}), area={obj_area}, "
                                  f"dist_izq={dist_left_str}, dist_der={dist_right_str}")
            
            if object_count > 0:
                print(f"  → Con conf>={conf_threshold:.2f}: {object_count} objetos detectados")
                break  # Si encontramos objetos, no necesitamos probar umbrales más altos
        
        # Visualizar
        frame_vis = frame.copy()
        
        # Dibujar muñecas
        if left_wrist:
            cv2.circle(frame_vis, left_wrist, 25, (0, 255, 0), 5)
            cv2.putText(frame_vis, 'IZQ', (left_wrist[0]+30, left_wrist[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        if right_wrist:
            cv2.circle(frame_vis, right_wrist, 25, (255, 0, 0), 5)
            cv2.putText(frame_vis, 'DER', (right_wrist[0]+30, right_wrist[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        
        # Dibujar objetos
        object_results = yolo_objects(frame, conf=0.1, verbose=False)
        if object_results[0].boxes is not None:
            for box in object_results[0].boxes:
                cls = int(box.cls[0].cpu().numpy())
                if cls != 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 255, 0), 3)
        
        # No mostrar ventana, solo imprimir
        # cv2.putText(frame_vis, f'Frame {frame_count}', (50, 50),
        #            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        # 
        # # Redimensionar
        # h, w = frame_vis.shape[:2]
        # if w > 1920:
        #     scale = 1920 / w
        #     frame_vis = cv2.resize(frame_vis, (int(w*scale), int(h*scale)))
        # 
        # cv2.imshow('Debug YOLOv9', frame_vis)
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('q'):
        #     break

cap.release()
cv2.destroyAllWindows()

print("\nAnálisis completado.")
