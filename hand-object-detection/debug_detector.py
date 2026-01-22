"""
Script de debug para ver qué está detectando MediaPipe
"""

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage, ImageFormat
import os

# Inicializar MediaPipe
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    import urllib.request
    model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    print("Descargando modelo...")
    urllib.request.urlretrieve(model_url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.1,  # Muy bajo
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

video_path = '/home/debian/Vídeos/video0.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
hands_detected = 0

print("Analizando video para debug...")
print("Presiona 'q' para salir, 's' para guardar frame actual\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)
    detection_result = hand_landmarker.detect(mp_image)
    
    num_hands = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0
    
    if num_hands > 0:
        hands_detected += 1
        print(f"Frame {frame_count}: {num_hands} mano(s) detectada(s)")
        
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Calcular bbox
            x_coords = [int(lm.x * w) for lm in hand_landmarks]
            y_coords = [int(lm.y * h) for lm in hand_landmarks]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            # Detectar si está cerrada
            thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
            index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
            thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
            is_closed = thumb_index_dist < 0.08
            
            print(f"  Mano {i+1}: bbox=({x1},{y1})-({x2},{y2}), cerrada={is_closed}, dist_pulgar_indice={thumb_index_dist:.3f}")
            
            # Dibujar
            for connection in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8)]:
                if connection[0] < len(hand_landmarks) and connection[1] < len(hand_landmarks):
                    pt1 = (int(hand_landmarks[connection[0]].x * w), 
                          int(hand_landmarks[connection[0]].y * h))
                    pt2 = (int(hand_landmarks[connection[1]].x * w), 
                          int(hand_landmarks[connection[1]].y * h))
                    color = (0, 255, 0) if is_closed else (255, 0, 0)
                    cv2.line(frame, pt1, pt2, color, 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f'Mano {i+1}: {"CERRADA" if is_closed else "ABIERTA"}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Mostrar info
    cv2.putText(frame, f'Frame: {frame_count}', (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f'Manos: {num_hands}', (20, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if num_hands > 0 else (0, 0, 255), 2)
    
    # Redimensionar para visualización
    display_w = 1920
    if w > display_w:
        scale = display_w / w
        display_h = int(h * scale)
        frame_display = cv2.resize(frame, (display_w, display_h))
    else:
        frame_display = frame
    
    cv2.imshow('Debug - Deteccion de Manos', frame_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f'debug_frame_{frame_count}.jpg', frame)
        print(f"Frame {frame_count} guardado")

cap.release()
cv2.destroyAllWindows()

print(f"\nResumen:")
print(f"  - Frames totales: {frame_count}")
print(f"  - Frames con manos detectadas: {hands_detected}")
print(f"  - Porcentaje: {hands_detected/frame_count*100:.1f}%")
