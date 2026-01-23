"""
Debug: Ver qué detecta YOLOv9 en video con bolsas
"""

import cv2
import numpy as np
from ultralytics import YOLO

print("Cargando modelos...")
yolo_pose = YOLO('yolov8n-pose.pt')
yolo_objects = YOLO('yolov9t.pt')

video_path = '/home/debian/Vídeos/videob.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: No se pudo abrir {video_path}")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total_frames} frames")
print("Analizando frames clave...\n")

frame_count = 0
sample_frames = [10, 30, 50, 70, 90, total_frames // 2, total_frames - 10]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count in sample_frames or frame_count % 20 == 0:
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
                            for wrist_idx in [9, 10]:
                                if (keypoints[wrist_idx][0] > 0 and keypoints[wrist_idx][1] > 0 and
                                    (confidences is None or confidences[wrist_idx] > 0.3)):
                                    wrist_pos = tuple(map(int, keypoints[wrist_idx]))
                                    if wrist_idx == 9:
                                        left_wrist = wrist_pos
                                    elif wrist_idx == 10:
                                        right_wrist = wrist_pos
                                    print(f"  Muñeca {'IZQUIERDA' if wrist_idx == 9 else 'DERECHA'}: {wrist_pos}")
        
        # Detectar objetos con YOLOv9 - múltiples umbrales
        for conf_threshold in [0.05, 0.1, 0.15, 0.2]:
            object_results = yolo_objects(frame, conf=conf_threshold, verbose=False)
            objects_near_hands = []
            
            if object_results[0].boxes is not None:
                for box in object_results[0].boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Clases que pueden ser bolsas: 24=backpack, 26=handbag, 28=suitcase, 39=bottle, etc.
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    obj_area = obj_width * obj_height
                    obj_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    # Calcular distancia a muñecas
                    dist_to_left = None
                    dist_to_right = None
                    
                    if left_wrist:
                        dist_to_left = np.sqrt((obj_center[0] - left_wrist[0])**2 + 
                                              (obj_center[1] - left_wrist[1])**2)
                    if right_wrist:
                        dist_to_right = np.sqrt((obj_center[0] - right_wrist[0])**2 + 
                                               (obj_center[1] - right_wrist[1])**2)
                    
                    # Verificar si está cerca de alguna muñeca
                    min_dist = min([d for d in [dist_to_left, dist_to_right] if d is not None], default=float('inf'))
                    
                    if min_dist < 1000:  # Cualquier objeto dentro de 1000 píxeles
                        objects_near_hands.append({
                            'class': cls,
                            'conf': conf,
                            'bbox': [x1, y1, x2, y2],
                            'area': obj_area,
                            'dist_left': dist_to_left,
                            'dist_right': dist_to_right,
                            'min_dist': min_dist
                        })
            
            if objects_near_hands:
                print(f"  Con conf>={conf_threshold:.2f}: {len(objects_near_hands)} objetos cerca de manos")
                for obj in objects_near_hands:
                    dist_str = f"dist={obj['min_dist']:.1f}"
                    if obj['dist_left']:
                        dist_str += f" (izq:{obj['dist_left']:.1f})"
                    if obj['dist_right']:
                        dist_str += f" (der:{obj['dist_right']:.1f})"
                    print(f"    Clase {obj['class']}, conf={obj['conf']:.3f}, area={obj['area']}, {dist_str}")
                break  # Si encontramos objetos, no probar umbrales más altos
        
        # También buscar clases específicas de bolsas
        object_results = yolo_objects(frame, conf=0.1, verbose=False)
        if object_results[0].boxes is not None:
            bag_classes = [24, 26, 28]  # backpack, handbag, suitcase
            for box in object_results[0].boxes:
                cls = int(box.cls[0].cpu().numpy())
                if cls in bag_classes:
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    print(f"  ¡BOLSA detectada! Clase {cls}, conf={conf:.3f}, bbox=({x1},{y1})-({x2},{y2})")

cap.release()
print("\nAnálisis completado.")
