"""
Detección V5: Sin MediaPipe - Solo YOLO Pose + Tracking Óptico
Enfoque: Usar muñecas de YOLO pose y tracking para detectar cuando la mano
se detiene cerca de un objeto (momento de coger)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorV5:
    """
    Detector usando solo YOLO pose (sin MediaPipe)
    """
    
    def __init__(self, video_path, interaction_threshold=150, min_frames_interaction=5):
        self.video_path = video_path
        self.interaction_threshold = interaction_threshold
        self.min_frames_interaction = min_frames_interaction
        
        # YOLO pose para detectar personas y muñecas
        print("Cargando YOLO pose...")
        self.yolo_model = YOLO('yolov8n-pose.pt')
        
        # Tracking óptico (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Historial
        self.wrist_positions = deque(maxlen=20)
        self.wrist_velocities = deque(maxlen=10)
        self.interaction_history = deque(maxlen=min_frames_interaction)
        self.previous_frame = None
        self.previous_gray = None
        self.feature_points = None
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector V5 inicializado (YOLO Pose + Tracking)")
    
    def detect_objects_near_wrist(self, frame, wrist_pos, previous_frame):
        """
        Detecta objetos cerca de la muñeca usando diferencia de frames
        """
        if previous_frame is None or wrist_pos is None:
            return []
        
        h, w = frame.shape[:2]
        wx, wy = wrist_pos
        
        # Región de búsqueda
        search_size = self.interaction_threshold * 2
        x1 = max(0, wx - search_size)
        y1 = max(0, wy - search_size)
        x2 = min(w, wx + search_size)
        y2 = min(h, wy + search_size)
        
        # Recortar regiones
        current_roi = frame[y1:y2, x1:x2]
        previous_roi = previous_frame[y1:y2, x1:x2]
        
        if current_roi.size == 0 or previous_roi.size == 0:
            return []
        
        # Diferencia
        gray_current = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_roi, cv2.COLOR_BGR2GRAY)
        
        gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)
        gray_previous = cv2.GaussianBlur(gray_previous, (5, 5), 0)
        
        diff = cv2.absdiff(gray_current, gray_previous)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Morfología
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 20000:
                x, y, w_obj, h_obj = cv2.boundingRect(contour)
                obj_x1 = x1 + x
                obj_y1 = y1 + y
                obj_x2 = x1 + x + w_obj
                obj_y2 = y1 + y + h_obj
                
                objects.append({
                    'bbox': [obj_x1, obj_y1, obj_x2, obj_y2],
                    'area': area,
                    'center': ((obj_x1 + obj_x2) // 2, (obj_y1 + obj_y2) // 2)
                })
        
        return objects
    
    def process_frame(self, frame):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar personas con YOLO pose
        results = self.yolo_model(frame, conf=0.3, verbose=False)
        
        wrist_positions = []
        interactions = []
        
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:  # Persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Dibujar persona
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    
                    # Obtener keypoints
                    if results[0].keypoints is not None and len(results[0].keypoints) > i:
                        keypoints = results[0].keypoints[i].xy[0].cpu().numpy()
                        confidences = results[0].keypoints[i].conf[0].cpu().numpy() if hasattr(results[0].keypoints[i], 'conf') else None
                        
                        # Dibujar esqueleto
                        for connection in self.POSE_CONNECTIONS:
                            pt1_idx, pt2_idx = connection
                            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                                pt1 = tuple(map(int, keypoints[pt1_idx]))
                                pt2 = tuple(map(int, keypoints[pt2_idx]))
                                if confidences is None or (confidences[pt1_idx] > 0.3 and confidences[pt2_idx] > 0.3):
                                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                        
                        # Obtener muñecas (índices 9 y 10)
                        if len(keypoints) > 10:
                            # Muñeca izquierda (9) y derecha (10)
                            for wrist_idx in [9, 10]:
                                if keypoints[wrist_idx][0] > 0 and keypoints[wrist_idx][1] > 0:
                                    wrist_pos = tuple(map(int, keypoints[wrist_idx]))
                                    wrist_positions.append(wrist_pos)
                                    
                                    # Dibujar muñeca
                                    cv2.circle(frame, wrist_pos, 15, (0, 255, 0), 3)
                                    cv2.putText(frame, f'Wrist {wrist_idx}', 
                                               (wrist_pos[0]+20, wrist_pos[1]), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    
                                    # Detectar objetos cerca de la muñeca
                                    if self.previous_frame is not None:
                                        objects = self.detect_objects_near_wrist(
                                            frame, wrist_pos, self.previous_frame
                                        )
                                        
                                        for obj in objects:
                                            obj_center = obj['center']
                                            dist = np.sqrt((wrist_pos[0] - obj_center[0])**2 + 
                                                          (wrist_pos[1] - obj_center[1])**2)
                                            
                                            if dist < self.interaction_threshold:
                                                interactions.append({
                                                    'bbox': obj['bbox'],
                                                    'wrist': wrist_pos,
                                                    'distance': dist,
                                                    'confidence': 0.7
                                                })
                                                
                                                # Visualizar
                                                x1_obj, y1_obj, x2_obj, y2_obj = obj['bbox']
                                                cv2.rectangle(frame, (x1_obj, y1_obj), (x2_obj, y2_obj), (0, 255, 255), 4)
                                                cv2.line(frame, wrist_pos, obj_center, (0, 255, 255), 3)
        
        # Actualizar historial
        if wrist_positions:
            self.wrist_positions.append(wrist_positions[0])
        
        # Calcular velocidades
        if len(self.wrist_positions) > 1:
            prev_pos = self.wrist_positions[-2]
            curr_pos = self.wrist_positions[-1]
            velocity = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            self.wrist_velocities.append(velocity)
        
        # Detectar interacción: más estricto
        has_interaction = False
        
        # Solo si hay objetos detectados Y la muñeca se detuvo
        if len(interactions) > 0:
            # Verificar si la muñeca se detuvo
            if len(self.wrist_velocities) > 3:
                recent_vel = np.mean(list(self.wrist_velocities)[-2:])
                older_vel = np.mean(list(self.wrist_velocities)[-4:-2])
                # Se detuvo significativamente
                if older_vel > 15 and recent_vel < 8:
                    has_interaction = True
            # O si hay objetos muy cerca de la muñeca
            elif any(interaction['distance'] < self.interaction_threshold * 0.5 for interaction in interactions):
                has_interaction = True
        
        self.interaction_history.append(has_interaction)
        # Más estricto: necesita más frames consecutivos
        confirmed = sum(self.interaction_history) >= self.min_frames_interaction
        
        # Visualizar confirmación
        if confirmed and interactions:
            for interaction in interactions:
                x1, y1, x2, y2 = interaction['bbox']
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 255), 8)
                cv2.putText(frame, 'OBJETO COGIDO!', (x1, y1-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Debug
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f'Muñecas: {len(wrist_positions)}', (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Objetos detectados: {len(interactions)}', (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Interaccion: {"SI" if has_interaction else "NO"}', (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if has_interaction else (0, 0, 255), 2)
        if confirmed:
            cv2.putText(frame, 'OBJETO COGIDO!', (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        self.previous_frame = frame.copy()
        self.previous_gray = gray.copy()
        
        return frame, interactions, confirmed
    
    def process_video(self, output_path=None, show_video=True, display_width=1920):
        """Procesa el video completo"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {self.video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        if width > display_width:
            scale = display_width / width
            display_height = int(height * scale)
        else:
            display_width = width
            display_height = height
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_interactions = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                frame_annotated, interactions, confirmed = self.process_frame(frame)
                
                if confirmed:
                    total_interactions += 1
                
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Procesando: {frame_count}/{total_frames} ({progress:.1f}%)")
                
                if writer:
                    writer.write(frame_annotated)
                
                if show_video:
                    if frame_annotated.shape[1] > display_width:
                        frame_display = cv2.resize(frame_annotated, (display_width, display_height))
                    else:
                        frame_display = frame_annotated
                    
                    cv2.imshow('Deteccion V5 - YOLO Pose + Tracking', frame_display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_video:
                cv2.destroyAllWindows()
            
            print(f"\nProcesamiento completado:")
            print(f"  - Frames: {frame_count}")
            print(f"  - Interacciones confirmadas: {total_interactions}")


def main():
    parser = argparse.ArgumentParser(description='Deteccion V5: YOLO Pose + Tracking')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--threshold', type=int, default=150, help='Umbral de distancia')
    parser.add_argument('--min-frames', type=int, default=5, help='Frames para confirmar')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualizacion')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorV5(
        video_path=args.video_path,
        interaction_threshold=args.threshold,
        min_frames_interaction=args.min_frames
    )
    
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
