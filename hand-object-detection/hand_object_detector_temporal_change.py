"""
Detector por cambio temporal: Detecta cuando algo aparece cerca de la mano
Enfoque: Comparar frames anteriores - si aparece algo nuevo cerca de la mano, es objeto
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorTemporalChange:
    """
    Detecta objetos por cambios temporales cerca de la mano
    """
    
    def __init__(self, video_path, detection_radius=150, min_frames=3):
        self.video_path = video_path
        self.detection_radius = detection_radius
        self.min_frames = min_frames
        
        # YOLO pose para muñecas
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # Historial de frames y muñecas
        self.frame_history = deque(maxlen=10)  # Últimos 10 frames
        self.wrist_history = {'left': deque(maxlen=10), 'right': deque(maxlen=10)}
        self.left_hand_history = deque(maxlen=min_frames)
        self.right_hand_history = deque(maxlen=min_frames)
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector por cambio temporal inicializado")
    
    def detect_change_near_wrist(self, current_frame, wrist_pos, hand_side):
        """
        Detecta cambios en la región alrededor de la muñeca
        """
        if wrist_pos is None or len(self.frame_history) < 3:
            return False, 0.0
        
        wx, wy = wrist_pos
        h, w = current_frame.shape[:2]
        
        # Definir región alrededor de la muñeca
        x1 = max(0, wx - self.detection_radius)
        y1 = max(0, wy - self.detection_radius)
        x2 = min(w, wx + self.detection_radius)
        y2 = min(h, wy + self.detection_radius)
        
        # Recortar región actual
        current_roi = current_frame[y1:y2, x1:x2]
        if current_roi.size == 0:
            return False, 0.0
        
        # Comparar con frames anteriores
        max_change = 0.0
        has_change = False
        
        for prev_frame in list(self.frame_history)[-5:]:  # Últimos 5 frames
            prev_roi = prev_frame[y1:y2, x1:x2]
            if prev_roi.size == 0:
                continue
            
            # Convertir a escala de grises
            gray_current = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
            gray_previous = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
            
            # Aplicar blur para reducir ruido
            gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)
            gray_previous = cv2.GaussianBlur(gray_previous, (5, 5), 0)
            
            # Calcular diferencia
            diff = cv2.absdiff(gray_current, gray_previous)
            
            # Threshold adaptativo
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Morfología para limpiar
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Calcular ratio de cambio
            change_ratio = np.sum(thresh > 0) / thresh.size
            
            # Si hay cambio significativo
            if change_ratio > 0.12:  # 12% de la región cambió (más sensible)
                has_change = True
                max_change = max(max_change, change_ratio)
        
        # También verificar si la muñeca se movió (puede indicar que cogió algo)
        wrist_moved = False
        wrist_movement = 0.0
        if len(self.wrist_history[hand_side]) >= 2:
            prev_wrist = self.wrist_history[hand_side][-2]
            dist_moved = np.sqrt((wx - prev_wrist[0])**2 + (wy - prev_wrist[1])**2)
            wrist_movement = dist_moved
            if dist_moved > 8:  # Se movió significativamente
                wrist_moved = True
        
        # Si hay cambio Y la muñeca se movió, probablemente tiene objeto
        # O si hay cambio muy grande (objeto grande)
        has_object = (has_change and wrist_moved) or (max_change > 0.25)  # Cambio muy grande
        confidence = min(1.0, max_change * 2.5 + (wrist_movement / 50.0)) if has_object else 0.0
        
        return has_object, confidence
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        
        # Detectar personas con YOLO pose
        pose_results = self.yolo_pose(frame, conf=0.3, verbose=False)
        
        # Extraer muñecas
        left_wrist = None
        right_wrist = None
        
        if pose_results[0].boxes is not None:
            for i, box in enumerate(pose_results[0].boxes):
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:  # Persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Dibujar persona
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
                    # Obtener keypoints
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
                        
                        # Extraer muñecas
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
                                    
                                    # Dibujar región de análisis
                                    cv2.circle(frame, wrist_pos, self.detection_radius, color, 2)
        
        # Detectar cambios cerca de las muñecas
        left_has_object = False
        right_has_object = False
        left_confidence = 0.0
        right_confidence = 0.0
        
        if left_wrist:
            left_has_object, left_confidence = self.detect_change_near_wrist(
                frame, left_wrist, 'left'
            )
            self.wrist_history['left'].append(left_wrist)
        
        if right_wrist:
            right_has_object, right_confidence = self.detect_change_near_wrist(
                frame, right_wrist, 'right'
            )
            self.wrist_history['right'].append(right_wrist)
        
        # Actualizar historial de frames
        self.frame_history.append(frame.copy())
        
        # Actualizar historial de detecciones
        self.left_hand_history.append(left_has_object)
        self.right_hand_history.append(right_has_object)
        
        # Confirmar solo si hay detección consistente
        left_confirmed = sum(self.left_hand_history) >= self.min_frames
        right_confirmed = sum(self.right_hand_history) >= self.min_frames
        
        # Visualizar confirmación
        if left_wrist and left_confirmed:
            cv2.putText(frame, 'OBJETO!', (left_wrist[0]-50, left_wrist[1]-self.detection_radius-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
        
        if right_wrist and right_confirmed:
            cv2.putText(frame, 'OBJETO!', (right_wrist[0]-50, right_wrist[1]-self.detection_radius-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)
        
        # Panel de información
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_number}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mano izquierda
        left_color = (0, 255, 0) if left_confirmed else (150, 150, 150)
        left_bg = (0, 100, 0) if left_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 55), (580, 105), left_bg, -1)
        cv2.rectangle(frame, (15, 55), (580, 105), left_color, 3)
        left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"} (conf: {left_confidence:.2f})'
        cv2.putText(frame, left_text, (25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, left_color, 2)
        
        # Mano derecha
        right_color = (255, 0, 0) if right_confirmed else (150, 150, 150)
        right_bg = (100, 0, 0) if right_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 115), (580, 165), right_bg, -1)
        cv2.rectangle(frame, (15, 115), (580, 165), right_color, 3)
        right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"} (conf: {right_confidence:.2f})'
        cv2.putText(frame, right_text, (25, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, right_color, 2)
        
        # Resumen
        if left_confirmed or right_confirmed:
            summary_text = "OBJETO DETECTADO"
            summary_color = (0, 255, 255)
            if left_confirmed and right_confirmed:
                summary_text = "OBJETOS EN AMBAS MANOS"
            elif left_confirmed:
                summary_text = "✓ OBJETO EN MANO IZQUIERDA"
                summary_color = (0, 255, 0)
            elif right_confirmed:
                summary_text = "✓ OBJETO EN MANO DERECHA"
                summary_color = (255, 0, 0)
            
            text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
            cv2.rectangle(frame, (15, 175), (20 + text_size[0], 195), (0, 0, 0), -1)
            cv2.putText(frame, summary_text, (20, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, summary_color, 4)
        
        return {
            'frame': frame,
            'left_hand': {
                'has_object': left_confirmed,
                'confidence': left_confidence
            },
            'right_hand': {
                'has_object': right_confirmed,
                'confidence': right_confidence
            }
        }
    
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
        frames_with_objects = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                result = self.process_frame(frame, frame_count)
                
                if result['left_hand']['has_object'] or result['right_hand']['has_object']:
                    frames_with_objects.append({
                        'frame': frame_count,
                        'left': result['left_hand']['has_object'],
                        'right': result['right_hand']['has_object']
                    })
                    print(f"  Frame {frame_count}: "
                          f"Izq={'OBJETO' if result['left_hand']['has_object'] else 'VACIA'}, "
                          f"Der={'OBJETO' if result['right_hand']['has_object'] else 'VACIA'}")
                
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Procesando: {frame_count}/{total_frames} ({progress:.1f}%)")
                
                if writer:
                    writer.write(result['frame'])
                
                if show_video:
                    if result['frame'].shape[1] > display_width:
                        frame_display = cv2.resize(result['frame'], (display_width, display_height))
                    else:
                        frame_display = result['frame']
                    
                    cv2.imshow('Temporal Change Detector', frame_display)
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
            print(f"  - Frames totales: {frame_count}")
            print(f"  - Frames con objetos en manos: {len(frames_with_objects)}")
            if frames_with_objects:
                print(f"  - Primeros frames con objetos: {[f['frame'] for f in frames_with_objects[:10]]}")


def main():
    parser = argparse.ArgumentParser(description='Detector por cambio temporal')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--radius', type=int, default=150, help='Radio de análisis alrededor de muñeca')
    parser.add_argument('--min-frames', type=int, default=3, help='Frames mínimos para confirmar')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorTemporalChange(
        video_path=args.video_path,
        detection_radius=args.radius,
        min_frames=args.min_frames
    )
    
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
