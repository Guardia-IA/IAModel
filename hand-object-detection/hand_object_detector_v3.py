"""
Detección V3: Movimiento conjunto mano-objeto
Enfoque: Si la mano y algo cerca de ella se mueven juntos, es que lo cogió
"""

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage, ImageFormat
from collections import deque
import argparse
import os


class HandObjectDetectorV3:
    """
    Detecta objetos por movimiento conjunto con la mano
    """
    
    def __init__(self, video_path, interaction_threshold=150, min_frames_interaction=5):
        self.video_path = video_path
        self.interaction_threshold = interaction_threshold
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
        
        # Historial de posiciones y frames
        self.hand_positions_history = deque(maxlen=10)
        self.previous_frame = None
        self.object_tracks = {}  # Tracking de objetos
        self.next_track_id = 0
        
        print("✓ Detector V3 inicializado (movimiento conjunto)")
    
    def get_hand_bbox(self, hand_landmarks, frame_width, frame_height):
        """Obtiene bounding box de la mano"""
        x_coords = [int(lm.x * frame_width) for lm in hand_landmarks]
        y_coords = [int(lm.y * frame_height) for lm in hand_landmarks]
        
        min_x, max_x = max(0, min(x_coords) - 50), min(frame_width, max(x_coords) + 50)
        min_y, max_y = max(0, min(y_coords) - 50), min(frame_height, max(y_coords) + 50)
        
        return (min_x, min_y, max_x, max_y)
    
    def detect_moving_objects_near_hand(self, current_frame, hand_bbox, previous_frame):
        """
        Detecta objetos que se mueven cerca de la mano usando diferencia de frames
        """
        if previous_frame is None:
            return []
        
        x1, y1, x2, y2 = hand_bbox
        
        # Expandir región
        padding = self.interaction_threshold
        search_x1 = max(0, x1 - padding)
        search_y1 = max(0, y1 - padding)
        search_x2 = min(current_frame.shape[1], x2 + padding)
        search_y2 = min(current_frame.shape[0], y2 + padding)
        
        # Recortar regiones
        current_roi = current_frame[search_y1:search_y2, search_x1:search_x2]
        previous_roi = previous_frame[search_y1:search_y2, search_x1:search_x2]
        
        if current_roi.size == 0 or previous_roi.size == 0:
            return []
        
        # Calcular diferencia
        gray_current = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_roi, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur para reducir ruido
        gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)
        gray_previous = cv2.GaussianBlur(gray_previous, (5, 5), 0)
        
        # Diferencia absoluta
        diff = cv2.absdiff(gray_current, gray_previous)
        
        # Threshold
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Morfología para limpiar
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 300 < area < 30000:  # Filtrar por tamaño
                x, y, w, h = cv2.boundingRect(contour)
                
                # Convertir a coordenadas globales
                obj_x1 = search_x1 + x
                obj_y1 = search_y1 + y
                obj_x2 = search_x1 + x + w
                obj_y2 = search_y1 + y + h
                
                # Centro del objeto
                center = ((obj_x1 + obj_x2) // 2, (obj_y1 + obj_y2) // 2)
                
                objects.append({
                    'bbox': [obj_x1, obj_y1, obj_x2, obj_y2],
                    'center': center,
                    'area': area
                })
        
        return objects
    
    def track_objects_with_hand(self, current_objects, hand_center):
        """
        Trackea objetos que se mueven junto con la mano
        """
        if not current_objects or hand_center is None:
            return []
        
        # Calcular movimiento de la mano
        hand_velocity = (0, 0)
        if len(self.hand_positions_history) > 1:
            prev_hand = self.hand_positions_history[-2]
            hand_velocity = (
                hand_center[0] - prev_hand[0],
                hand_center[1] - prev_hand[1]
            )
        
        tracked_objects = []
        
        for obj in current_objects:
            obj_center = obj['center']
            
            # Calcular distancia a la mano
            dist_to_hand = np.sqrt(
                (obj_center[0] - hand_center[0])**2 + 
                (obj_center[1] - hand_center[1])**2
            )
            
            # Si está cerca de la mano
            if dist_to_hand < self.interaction_threshold:
                # Intentar emparejar con tracks existentes
                matched = False
                for track_id, track_info in self.object_tracks.items():
                    last_center = track_info['last_center']
                    last_hand_pos = track_info['last_hand_pos']
                    
                    # Calcular distancia al último centro
                    dist_to_track = np.sqrt(
                        (obj_center[0] - last_center[0])**2 +
                        (obj_center[1] - last_center[1])**2
                    )
                    
                    # Si está cerca del track y la mano también se movió
                    if dist_to_track < 50:  # Objeto se movió poco (mismo objeto)
                        # Verificar si la mano y el objeto se mueven juntos
                        hand_movement = np.sqrt(hand_velocity[0]**2 + hand_velocity[1]**2)
                        obj_movement = dist_to_track
                        
                        # Si ambos se mueven o ambos están quietos
                        if (hand_movement > 5 and obj_movement < 30) or (hand_movement < 5 and obj_movement < 10):
                            # Actualizar track
                            track_info['last_center'] = obj_center
                            track_info['last_hand_pos'] = hand_center
                            track_info['frames'] += 1
                            track_info['bbox'] = obj['bbox']
                            matched = True
                            tracked_objects.append({
                                'bbox': obj['bbox'],
                                'track_id': track_id,
                                'frames_tracked': track_info['frames'],
                                'distance': dist_to_hand
                            })
                            break
                
                # Si no hay match, crear nuevo track
                if not matched:
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    self.object_tracks[track_id] = {
                        'last_center': obj_center,
                        'last_hand_pos': hand_center,
                        'frames': 1,
                        'bbox': obj['bbox']
                    }
                    tracked_objects.append({
                        'bbox': obj['bbox'],
                        'track_id': track_id,
                        'frames_tracked': 1,
                        'distance': dist_to_hand
                    })
        
        # Limpiar tracks antiguos
        tracks_to_remove = []
        for track_id, track_info in self.object_tracks.items():
            if track_info['frames'] > 20:  # Track muy largo, probablemente falso positivo
                tracks_to_remove.append(track_id)
        for track_id in tracks_to_remove:
            del self.object_tracks[track_id]
        
        return tracked_objects
    
    def process_frame(self, frame):
        """Procesa un frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Detectar manos
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.hand_landmarker.detect(mp_image)
        
        hand_bboxes = []
        hand_centers = []
        interactions = []
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Obtener bbox y centro
                hand_bbox = self.get_hand_bbox(hand_landmarks, w, h)
                hand_bboxes.append(hand_bbox)
                
                x1, y1, x2, y2 = hand_bbox
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                hand_centers.append(center)
                
                # Dibujar mano
                for connection in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), 
                                  (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12)]:
                    if connection[0] < len(hand_landmarks) and connection[1] < len(hand_landmarks):
                        pt1 = (int(hand_landmarks[connection[0]].x * w), 
                              int(hand_landmarks[connection[0]].y * h))
                        pt2 = (int(hand_landmarks[connection[1]].x * w), 
                              int(hand_landmarks[connection[1]].y * h))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                
                cv2.circle(frame, center, 10, (0, 255, 255), 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                
                # Detectar objetos en movimiento cerca de la mano
                if self.previous_frame is not None:
                    moving_objects = self.detect_moving_objects_near_hand(
                        frame, hand_bbox, self.previous_frame
                    )
                    
                    # Trackear objetos que se mueven con la mano
                    tracked = self.track_objects_with_hand(moving_objects, center)
                    
                    for obj in tracked:
                        # Si el objeto ha sido trackeado varios frames, es interacción
                        if obj['frames_tracked'] >= 3:
                            interactions.append({
                                'bbox': obj['bbox'],
                                'confidence': min(1.0, obj['frames_tracked'] / 5.0),
                                'frames_tracked': obj['frames_tracked']
                            })
                            
                            # Visualizar
                            x1_obj, y1_obj, x2_obj, y2_obj = obj['bbox']
                            cv2.rectangle(frame, (x1_obj, y1_obj), (x2_obj, y2_obj), (0, 255, 255), 4)
                            cv2.putText(frame, f'OBJETO! ({obj["frames_tracked"]} frames)', 
                                       (x1_obj, y1_obj-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Actualizar historial
        if hand_centers:
            self.hand_positions_history.append(hand_centers[0])
        
        self.previous_frame = frame.copy()
        
        # Confirmar interacciones
        confirmed = len(interactions) > 0
        
        # Debug info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f'Manos: {len(hand_centers)}', (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Objetos trackeados: {len(self.object_tracks)}', (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Interacciones: {len(interactions)}', (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if confirmed else (0, 0, 255), 2)
        
        if confirmed:
            cv2.putText(frame, 'OBJETO COGIDO!', (20, 120),
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
                    
                    cv2.imshow('Deteccion V3 - Movimiento Conjunto', frame_display)
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
            print(f"  - Interacciones detectadas: {total_interactions}")


def main():
    parser = argparse.ArgumentParser(description='Deteccion V3: Movimiento Conjunto')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--threshold', type=int, default=150, help='Umbral de distancia')
    parser.add_argument('--min-frames', type=int, default=5, help='Frames para confirmar')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualizacion')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorV3(
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
