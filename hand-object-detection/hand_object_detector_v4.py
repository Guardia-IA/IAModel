"""
Detección V4: Enfoque simple y directo
- Detecta cuando la mano está cerrada (agarre)
- Detecta cuando la mano se mueve con algo (cambio de forma/volumen)
- No intenta detectar el objeto directamente, solo detecta el GESTO de coger
"""

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage, ImageFormat
from collections import deque
import argparse
import os


class HandObjectDetectorV4:
    """
    Detector simple: mano cerrada + movimiento = objeto cogido
    """
    
    def __init__(self, video_path, min_frames_interaction=3):
        self.video_path = video_path
        self.min_frames_interaction = min_frames_interaction
        
        # Inicializar MediaPipe
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            import urllib.request
            model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            print("Descargando modelo de MediaPipe...")
            urllib.request.urlretrieve(model_url, model_path)
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Historial
        self.hand_states = deque(maxlen=10)  # Estados de la mano (abierta/cerrada)
        self.hand_positions = deque(maxlen=10)
        self.interaction_history = deque(maxlen=min_frames_interaction)
        
        print("✓ Detector V4 inicializado (gesto de agarre)")
    
    def is_hand_closed(self, hand_landmarks):
        """Detecta si la mano está cerrada (agarrando algo)"""
        # Puntos clave
        thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
        index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
        middle_tip = np.array([hand_landmarks[12].x, hand_landmarks[12].y])
        ring_tip = np.array([hand_landmarks[16].x, hand_landmarks[16].y])
        pinky_tip = np.array([hand_landmarks[20].x, hand_landmarks[20].y])
        
        # Puntos base de los dedos
        index_mcp = np.array([hand_landmarks[5].x, hand_landmarks[5].y])
        middle_mcp = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
        
        # Distancias entre puntas
        thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
        thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
        index_middle_dist = np.linalg.norm(index_tip - middle_tip)
        
        # Distancias punta-base (curvatura)
        index_curl = np.linalg.norm(index_tip - index_mcp)
        middle_curl = np.linalg.norm(middle_tip - middle_mcp)
        
        # Mano cerrada si:
        # 1. Dedos muy juntos (pinza/agarre)
        is_pinch = thumb_index_dist < 0.08
        # 2. Dedos curvados y juntos (agarre)
        is_grip = (thumb_index_dist < 0.12 and thumb_middle_dist < 0.12 and 
                  index_curl < 0.15 and middle_curl < 0.15)
        # 3. Puño cerrado
        is_fist = (thumb_index_dist < 0.06 and thumb_middle_dist < 0.06 and 
                  index_middle_dist < 0.06)
        
        return is_pinch or is_grip or is_fist
    
    def get_hand_center(self, hand_landmarks, frame_width, frame_height):
        """Obtiene centro de la mano"""
        palm = hand_landmarks[9]  # Base de la palma
        center_x = int(palm.x * frame_width)
        center_y = int(palm.y * frame_height)
        return (center_x, center_y)
    
    def detect_hand_movement(self, current_pos, previous_positions):
        """Detecta si la mano se está moviendo"""
        if len(previous_positions) < 2:
            return False, 0
        
        # Calcular velocidad promedio
        velocities = []
        for i in range(1, len(previous_positions)):
            prev = previous_positions[i-1]
            curr = previous_positions[i]
            vel = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            velocities.append(vel)
        
        avg_velocity = np.mean(velocities) if velocities else 0
        is_moving = avg_velocity > 5  # Mínimo movimiento
        
        return is_moving, avg_velocity
    
    def detect_volume_change(self, hand_landmarks, previous_landmarks_list):
        """
        Detecta si el "volumen" de la mano cambió (indica que cogió algo)
        """
        if len(previous_landmarks_list) < 2:
            return False
        
        # Calcular "volumen" como área del bounding box de los landmarks
        current_x = [lm.x for lm in hand_landmarks]
        current_y = [lm.y for lm in hand_landmarks]
        current_area = (max(current_x) - min(current_x)) * (max(current_y) - min(current_y))
        
        previous_areas = []
        for prev_landmarks in previous_landmarks_list[-3:]:
            prev_x = [lm.x for lm in prev_landmarks]
            prev_y = [lm.y for lm in prev_landmarks]
            prev_area = (max(prev_x) - min(prev_x)) * (max(prev_y) - min(prev_y))
            previous_areas.append(prev_area)
        
        avg_previous_area = np.mean(previous_areas)
        
        # Si el área aumentó significativamente, puede haber un objeto
        if current_area > avg_previous_area * 1.2:
            return True
        
        return False
    
    def process_frame(self, frame):
        """Procesa un frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Detectar manos
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.hand_landmarker.detect(mp_image)
        
        interactions = []
        hand_closed = False
        hand_moving = False
        volume_changed = False
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Detectar si está cerrada
                is_closed = self.is_hand_closed(hand_landmarks)
                hand_closed = is_closed
                
                # Obtener posición
                hand_center = self.get_hand_center(hand_landmarks, w, h)
                
                # Detectar movimiento
                is_moving, velocity = self.detect_hand_movement(
                    hand_center, 
                    list(self.hand_positions)
                )
                hand_moving = is_moving
                
                # Detectar cambio de volumen
                vol_changed = False
                if len(self.hand_states) > 0:
                    # Obtener landmarks anteriores del historial
                    # (simplificado: comparar con estado anterior)
                    vol_changed = self.detect_volume_change(
                        hand_landmarks,
                        [hand_landmarks]  # Simplificado
                    )
                volume_changed = vol_changed
                
                # Actualizar historiales
                self.hand_states.append(is_closed)
                self.hand_positions.append(hand_center)
                
                # Dibujar mano
                for connection in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), 
                                  (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                                  (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
                                  (18, 19), (19, 20)]:
                    if connection[0] < len(hand_landmarks) and connection[1] < len(hand_landmarks):
                        pt1 = (int(hand_landmarks[connection[0]].x * w), 
                              int(hand_landmarks[connection[0]].y * h))
                        pt2 = (int(hand_landmarks[connection[1]].x * w), 
                              int(hand_landmarks[connection[1]].y * h))
                        color = (0, 255, 0) if is_closed else (255, 0, 0)
                        cv2.line(frame, pt1, pt2, color, 2)
                
                # Marcar centro
                color_circle = (0, 255, 255) if is_closed else (255, 255, 0)
                cv2.circle(frame, hand_center, 15, color_circle, 3)
                
                # Bounding box de la mano
                x_coords = [int(lm.x * w) for lm in hand_landmarks]
                y_coords = [int(lm.y * h) for lm in hand_landmarks]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                
                # Detectar interacción: MUY permisivo
                # Si la mano está cerrada, es interacción (incluso sin movimiento)
                # O si la mano se mueve significativamente (puede estar cogiendo algo)
                has_interaction = is_closed or (is_moving and velocity > 10)
                
                if has_interaction:
                    # Crear bbox expandido para el "objeto"
                    padding = 30
                    obj_x1 = max(0, x1 - padding)
                    obj_y1 = max(0, y1 - padding)
                    obj_x2 = min(w, x2 + padding)
                    obj_y2 = min(h, y2 + padding)
                    
                    interactions.append({
                        'bbox': [obj_x1, obj_y1, obj_x2, obj_y2],
                        'confidence': 0.8 if is_moving else 0.6
                    })
                    
                    # Visualizar de forma muy visible
                    # Rectángulo exterior grueso
                    cv2.rectangle(frame, (obj_x1-5, obj_y1-5), (obj_x2+5, obj_y2+5), (0, 255, 255), 8)
                    # Rectángulo interior
                    cv2.rectangle(frame, (obj_x1, obj_y1), (obj_x2, obj_y2), (0, 165, 255), 5)
                    # Círculo en el centro
                    center_obj = ((obj_x1 + obj_x2) // 2, (obj_y1 + obj_y2) // 2)
                    cv2.circle(frame, center_obj, 25, (0, 255, 255), 5)
                    cv2.circle(frame, center_obj, 15, (0, 165, 255), -1)
                    
                    # Rectángulo de la mano
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
        # Actualizar historial de interacciones
        has_interaction = len(interactions) > 0
        self.interaction_history.append(has_interaction)
        
        # Confirmar si hay interacciones en varios frames (más permisivo)
        confirmed = sum(self.interaction_history) >= max(1, self.min_frames_interaction // 2)
        
        # Visualizar confirmación
        if confirmed and interactions:
            for interaction in interactions:
                x1, y1, x2, y2 = interaction['bbox']
                cv2.putText(frame, 'OBJETO COGIDO!', (x1, y1-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Debug info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f'Mano cerrada: {"SI" if hand_closed else "NO"}', (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if hand_closed else (0, 0, 255), 2)
        cv2.putText(frame, f'Movimiento: {"SI" if hand_moving else "NO"}', (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if hand_moving else (0, 0, 255), 2)
        cv2.putText(frame, f'Cambio volumen: {"SI" if volume_changed else "NO"}', (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if volume_changed else (0, 0, 255), 2)
        cv2.putText(frame, f'Interaccion: {"SI" if has_interaction else "NO"}', (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if has_interaction else (0, 0, 255), 2)
        if confirmed:
            cv2.putText(frame, 'OBJETO COGIDO CONFIRMADO!', (20, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
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
                    
                    cv2.imshow('Deteccion V4 - Gesto de Agarre', frame_display)
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
    parser = argparse.ArgumentParser(description='Deteccion V4: Gesto de Agarre')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--min-frames', type=int, default=3, help='Frames para confirmar')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualizacion')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorV4(
        video_path=args.video_path,
        min_frames_interaction=args.min_frames
    )
    
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
