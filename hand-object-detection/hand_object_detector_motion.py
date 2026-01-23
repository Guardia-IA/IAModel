"""
Detector por movimiento temporal:
1. Detecta persona con YOLO
2. Detecta pose con YOLO (muñecas)
3. Analiza movimiento dentro del ROI de la persona
4. Identifica objetos que se mueven junto con el cuerpo cerca de las manos
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorMotion:
    """
    Detecta objetos en manos analizando movimiento temporal
    """
    
    def __init__(self, video_path, min_frames=3, display_width=1920):
        self.video_path = video_path
        self.min_frames = min_frames
        self.display_width = display_width
        
        # Cargar YOLO pose
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # Historial de frames para análisis de movimiento
        self.frame_history = deque(maxlen=3)  # Últimos 3 frames (más rápido para detectar movimiento)
        self.left_hand_history = deque(maxlen=min_frames)
        self.right_hand_history = deque(maxlen=min_frames)
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector por movimiento temporal inicializado")
    
    def detect_moving_regions(self, current_frame, person_bbox, prev_frames):
        """
        Detecta regiones que se mueven dentro del ROI de la persona
        Compara el frame actual con frames anteriores para encontrar movimiento
        """
        if len(prev_frames) == 0:
            return None
        
        x1, y1, x2, y2 = person_bbox
        
        # Recortar ROI de la persona
        person_roi = current_frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return None
        
        # Convertir a escala de grises
        current_gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        
        # Usar optical flow para detectar movimiento conjunto
        prev_gray = prev_frames[-1] if len(prev_frames) > 0 else None
        if prev_gray is None or prev_gray.shape != current_gray.shape:
            return None
        
        # Calcular optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calcular magnitud del flujo
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Normalizar magnitud a 0-255
        magnitude_normalized = np.clip(magnitude * 10, 0, 255).astype(np.uint8)
        
        # Aplicar umbral para encontrar regiones con movimiento
        _, motion_mask = cv2.threshold(magnitude_normalized, 5, 255, cv2.THRESH_BINARY)
        
        # También usar diferencia de frames para complementar
        diff = cv2.absdiff(current_gray, prev_gray)
        _, diff_mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        
        # Combinar ambas máscaras
        motion_mask = cv2.bitwise_or(motion_mask, diff_mask)
        
        # Aplicar operaciones morfológicas para limpiar
        kernel = np.ones((7, 7), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return motion_mask, (x1, y1, x2, y2)
    
    def filter_regions_near_hand(self, motion_mask, hand_pos, roi_coords, frame_width, frame_height):
        """
        Filtra regiones en movimiento que están cerca de la mano
        """
        if motion_mask is None or hand_pos is None:
            return None, []
        
        x1, y1, x2, y2 = roi_coords
        wx, wy = hand_pos
        
        # Ajustar coordenadas de la mano al ROI
        wx_roi = wx - x1
        wy_roi = wy - y1
        
        # Verificar que la mano está dentro del ROI
        if not (0 <= wx_roi < motion_mask.shape[1] and 0 <= wy_roi < motion_mask.shape[0]):
            return None, []
        
        # Encontrar contornos en la máscara de movimiento
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos cerca de la mano
        valid_contours = []
        hand_radius = 200  # Radio de búsqueda alrededor de la mano (más grande)
        
        for contour in contours:
            # Calcular centro del contorno
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calcular distancia al punto de la mano
                dist = np.sqrt((cx - wx_roi)**2 + (cy - wy_roi)**2)
                
                # Verificar si está cerca de la mano y tiene área significativa
                area = cv2.contourArea(contour)
                if dist < hand_radius and area > 200:  # Área mínima más baja
                    valid_contours.append(contour)
        
        if valid_contours:
            # Crear máscara solo con contornos válidos
            mask_valid = np.zeros(motion_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask_valid, valid_contours, -1, 255, -1)
            return mask_valid, valid_contours
        
        return None, []
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        
        # 1. Detectar persona con YOLO pose
        pose_results = self.yolo_pose(frame, conf=0.3, verbose=False)
        
        left_wrist = None
        right_wrist = None
        person_bbox = None
        left_mask = None
        right_mask = None
        left_contours = None
        right_contours = None
        
        if pose_results[0].boxes is not None:
            for i, box in enumerate(pose_results[0].boxes):
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:  # Persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    person_bbox = (x1, y1, x2, y2)
                    
                    # Dibujar bounding box de la persona
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, 'Person', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                    
                    # 2. Extraer muñecas del pose de YOLO
                    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > i:
                        keypoints = pose_results[0].keypoints[i].xy[0].cpu().numpy()
                        confidences = pose_results[0].keypoints[i].conf[0].cpu().numpy() if hasattr(pose_results[0].keypoints[i], 'conf') else None
                        
                        # Dibujar esqueleto
                        for connection in self.POSE_CONNECTIONS:
                            pt1_idx, pt2_idx = connection
                            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                                pt1 = tuple(map(int, keypoints[pt1_idx]))
                                pt2 = tuple(map(int, keypoints[pt2_idx]))
                                if confidences is None or (confidences[pt1_idx] > 0.3 and confidences[pt2_idx] > 0.3):
                                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                        
                        # Extraer muñecas (índices 9 y 10 en YOLO pose)
                        if len(keypoints) > 10:
                            for wrist_idx in [9, 10]:
                                if (keypoints[wrist_idx][0] > 0 and keypoints[wrist_idx][1] > 0 and
                                    (confidences is None or confidences[wrist_idx] > 0.3)):
                                    wrist_pos = tuple(map(int, keypoints[wrist_idx]))
                                    
                                    if wrist_idx == 9:
                                        left_wrist = wrist_pos
                                    elif wrist_idx == 10:
                                        right_wrist = wrist_pos
                                    
                                    # Dibujar muñeca
                                    color = (0, 255, 0) if wrist_idx == 9 else (255, 0, 0)
                                    cv2.circle(frame, wrist_pos, 15, color, 4)
                                    cv2.putText(frame, f'{"L" if wrist_idx == 9 else "R"}', 
                                               (wrist_pos[0]+20, wrist_pos[1]), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 3. Analizar movimiento dentro del ROI de la persona
        if person_bbox and len(self.frame_history) >= 2:  # Al menos 2 frames para comparar
            # Detectar regiones en movimiento
            result = self.detect_moving_regions(
                frame, person_bbox, self.frame_history
            )
            
            if result is not None:
                motion_mask, roi_coords = result
            else:
                motion_mask = None
                roi_coords = None
            
            if motion_mask is not None and roi_coords is not None:
                # Filtrar regiones cerca de cada mano
                if left_wrist:
                    left_mask, left_contours = self.filter_regions_near_hand(
                        motion_mask, left_wrist, roi_coords, w, h
                    )
                
                if right_wrist:
                    right_mask, right_contours = self.filter_regions_near_hand(
                        motion_mask, right_wrist, roi_coords, w, h
                    )
                
                # Dibujar máscaras y contornos
                x1, y1, x2, y2 = roi_coords
                
                if left_mask is not None and left_contours:
                    # Crear overlay para mano izquierda
                    overlay = frame.copy()
                    mask_full = np.zeros((h, w), dtype=np.uint8)
                    mask_full[y1:y2, x1:x2] = left_mask
                    overlay[mask_full > 0] = (0, 255, 255)  # Cyan
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    
                    # Dibujar contornos (ajustar coordenadas)
                    for contour in left_contours:
                        contour_global = contour.copy()
                        contour_global[:, :, 0] += x1
                        contour_global[:, :, 1] += y1
                        cv2.drawContours(frame, [contour_global], -1, (0, 255, 255), 3)
                
                if right_mask is not None and right_contours:
                    # Crear overlay para mano derecha
                    overlay = frame.copy()
                    mask_full = np.zeros((h, w), dtype=np.uint8)
                    mask_full[y1:y2, x1:x2] = right_mask
                    overlay[mask_full > 0] = (255, 0, 255)  # Magenta
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    
                    # Dibujar contornos (ajustar coordenadas)
                    for contour in right_contours:
                        contour_global = contour.copy()
                        contour_global[:, :, 0] += x1
                        contour_global[:, :, 1] += y1
                        cv2.drawContours(frame, [contour_global], -1, (255, 0, 255), 3)
        
        # Actualizar historial de frames
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if person_bbox:
            x1, y1, x2, y2 = person_bbox
            person_roi_gray = current_gray[y1:y2, x1:x2]
            self.frame_history.append(person_roi_gray)
        else:
            self.frame_history.append(current_gray)
        
        # Actualizar historial de detecciones
        left_detected = left_mask is not None and len(left_contours) > 0 if left_contours else False
        right_detected = right_mask is not None and len(right_contours) > 0 if right_contours else False
        
        self.left_hand_history.append(left_detected)
        self.right_hand_history.append(right_detected)
        
        # Confirmar detección
        left_confirmed = sum(self.left_hand_history) >= self.min_frames
        right_confirmed = sum(self.right_hand_history) >= self.min_frames
        
        # Visualización
        if left_confirmed:
            cv2.putText(frame, 'OBJETO IZQ!', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
        if right_confirmed:
            cv2.putText(frame, 'OBJETO DER!', (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)
        
        cv2.putText(frame, f'Frame: {frame_number}', (20, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return {
            'frame': frame,
            'left_hand': left_confirmed,
            'right_hand': right_confirmed
        }
    
    def process_video(self):
        """Procesa el video completo"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {self.video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        print("Procesando con análisis de movimiento temporal...\n")
        
        if width > self.display_width:
            scale = self.display_width / width
            display_height = int(height * scale)
        else:
            self.display_width = width
            display_height = height
        
        frame_count = 0
        frames_with_objects = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                result = self.process_frame(frame, frame_count)
                
                if result['left_hand'] or result['right_hand']:
                    frames_with_objects.append({
                        'frame': frame_count,
                        'left': result['left_hand'],
                        'right': result['right_hand']
                    })
                    print(f"  Frame {frame_count}: Izq={'OBJETO' if result['left_hand'] else 'VACIA'}, "
                          f"Der={'OBJETO' if result['right_hand'] else 'VACIA'}")
                
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Procesando: {frame_count}/{total_frames} ({progress:.1f}%)")
                
                if result['frame'].shape[1] > self.display_width:
                    frame_display = cv2.resize(result['frame'], (self.display_width, display_height))
                else:
                    frame_display = result['frame']
                
                cv2.imshow('Motion-based Hand-Object Detection', frame_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\nProcesamiento completado:")
            print(f"  - Frames totales: {frame_count}")
            print(f"  - Frames con objetos: {len(frames_with_objects)}")
            if frames_with_objects:
                print(f"  - Primeros frames: {[f['frame'] for f in frames_with_objects[:10]]}")


def main():
    parser = argparse.ArgumentParser(description='Detector por movimiento temporal')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--min-frames', type=int, default=3, help='Frames mínimos para confirmar')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorMotion(
        video_path=args.video_path,
        min_frames=args.min_frames,
        display_width=args.display_width
    )
    detector.process_video()


if __name__ == '__main__':
    main()
