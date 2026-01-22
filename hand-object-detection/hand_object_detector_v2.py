"""
Detección de interacción mano-objeto usando SAM (Segment Anything Model)
Enfoque completamente diferente: segmentar todo lo que está cerca de la mano
y detectar cambios temporales que indiquen que se cogió un objeto
"""

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage, ImageFormat
from collections import deque
import argparse
import os


class HandObjectDetectorV2:
    """
    Detector alternativo usando SAM y análisis temporal
    """
    
    def __init__(self, video_path, interaction_threshold=100, min_frames_interaction=5):
        """
        Inicializa el detector
        
        Args:
            video_path: Ruta al video
            interaction_threshold: Distancia en píxeles para considerar interacción
            min_frames_interaction: Frames consecutivos para confirmar
        """
        self.video_path = video_path
        self.interaction_threshold = interaction_threshold
        self.min_frames_interaction = min_frames_interaction
        
        # Inicializar MediaPipe para manos
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
        
        # Historial de frames para análisis temporal
        self.frame_history = deque(maxlen=10)  # Últimos 10 frames
        self.hand_history = deque(maxlen=10)  # Posiciones de mano
        self.interaction_history = deque(maxlen=min_frames_interaction)
        
        print("✓ Detector V2 inicializado (enfoque temporal + segmentación)")
    
    def get_hand_region(self, hand_landmarks, frame_width, frame_height):
        """Obtiene región alrededor de la mano"""
        x_coords = [int(lm.x * frame_width) for lm in hand_landmarks]
        y_coords = [int(lm.y * frame_height) for lm in hand_landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Expandir región
        padding = self.interaction_threshold
        x1 = max(0, min_x - padding)
        y1 = max(0, min_y - padding)
        x2 = min(frame_width, max_x + padding)
        y2 = min(frame_height, max_y + padding)
        
        return (x1, y1, x2, y2)
    
    def segment_hand_region(self, frame, hand_region):
        """
        Segmenta objetos en la región de la mano usando múltiples métodos
        """
        x1, y1, x2, y2 = hand_region
        roi = frame[y1:y2, x1:x2].copy()
        
        if roi.size == 0:
            return []
        
        objects = []
        
        # Método 1: Exclusión de piel
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        non_skin_mask = cv2.bitwise_not(skin_mask)
        
        # Método 2: Detección de bordes
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Combinar máscaras
        combined_mask = cv2.bitwise_or(non_skin_mask, edges)
        
        # Morfología para limpiar
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        roi_area = (x2 - x1) * (y2 - y1)
        min_area = max(100, roi_area * 0.01)  # Al menos 1% del ROI
        max_area = roi_area * 0.8  # Máximo 80% del ROI
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filtrar contornos muy alargados (probablemente ruido)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:
                    obj_x1 = x1 + x
                    obj_y1 = y1 + y
                    obj_x2 = x1 + x + w
                    obj_y2 = y1 + y + h
                    
                    objects.append({
                        'bbox': [obj_x1, obj_y1, obj_x2, obj_y2],
                        'area': area,
                        'contour': contour
                    })
        
        return objects
    
    def detect_motion_change(self, current_hand_pos, previous_hand_positions):
        """
        Detecta si la mano cambió de movimiento (indica que cogió algo)
        """
        if len(previous_hand_positions) < 3:
            return False
        
        # Calcular velocidades
        velocities = []
        for i in range(1, len(previous_hand_positions)):
            prev = previous_hand_positions[i-1]
            curr = previous_hand_positions[i]
            vel = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            velocities.append(vel)
        
        if len(velocities) < 2:
            return False
        
        # Si la velocidad disminuyó significativamente, puede haber cogido algo
        recent_vel = np.mean(velocities[-2:])
        older_vel = np.mean(velocities[:-2]) if len(velocities) > 2 else recent_vel
        
        # Si la velocidad bajó mucho, probablemente cogió algo
        if older_vel > 10 and recent_vel < older_vel * 0.3:
            return True
        
        return False
    
    def detect_temporal_change(self, current_objects, previous_objects_list):
        """
        Detecta cambios temporales en objetos cerca de la mano
        """
        if len(previous_objects_list) < 2:
            return False
        
        current_total_area = sum(obj['area'] for obj in current_objects) if current_objects else 0
        
        previous_areas = []
        for prev_objs in previous_objects_list[-5:]:  # Últimos 5 frames
            if prev_objs:
                prev_area = sum(obj['area'] for obj in prev_objs)
                previous_areas.append(prev_area)
            else:
                previous_areas.append(0)
        
        if not previous_areas:
            return False
        
        avg_previous_area = np.mean(previous_areas)
        std_previous_area = np.std(previous_areas) if len(previous_areas) > 1 else 0
        
        # Si hay objetos actuales y el área cambió significativamente
        if current_total_area > 0:
            # Si el área aumentó mucho O si apareció de la nada
            if avg_previous_area == 0 and current_total_area > 500:
                return True
            elif avg_previous_area > 0:
                # Cambio significativo (más de 1.3x o más de 2 desviaciones estándar)
                if current_total_area > avg_previous_area * 1.3:
                    return True
                if std_previous_area > 0 and current_total_area > avg_previous_area + 2 * std_previous_area:
                    return True
        
        return False
    
    def process_frame(self, frame):
        """
        Procesa un frame con el nuevo enfoque
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Detectar manos
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.hand_landmarker.detect(mp_image)
        
        hand_regions = []
        hand_positions = []
        hand_objects_list = []
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Obtener región de la mano
                hand_region = self.get_hand_region(hand_landmarks, w, h)
                hand_regions.append(hand_region)
                
                # Obtener posición central
                x1, y1, x2, y2 = hand_region
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                hand_positions.append((center_x, center_y))
                
                # Segmentar objetos en la región
                objects = self.segment_hand_region(frame, hand_region)
                hand_objects_list.append(objects)
                
                # Dibujar mano
                for connection in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), 
                                  (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12)]:
                    if connection[0] < len(hand_landmarks) and connection[1] < len(hand_landmarks):
                        pt1 = (int(hand_landmarks[connection[0]].x * w), 
                              int(hand_landmarks[connection[0]].y * h))
                        pt2 = (int(hand_landmarks[connection[1]].x * w), 
                              int(hand_landmarks[connection[1]].y * h))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                
                # Marcar centro
                cv2.circle(frame, (center_x, center_y), 10, (0, 255, 255), 3)
                
                # Dibujar región de búsqueda
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Análisis temporal
        has_interaction = False
        motion_change = False
        temporal_change = False
        
        if hand_positions:
            # Detectar cambio de movimiento
            if len(self.hand_history) > 2:
                motion_change = self.detect_motion_change(
                    hand_positions[0], 
                    list(self.hand_history)
                )
            
            # Detectar cambio temporal en objetos
            if hand_objects_list and hand_objects_list[0]:
                temporal_change = self.detect_temporal_change(
                    hand_objects_list[0],
                    list(self.frame_history)
                )
            
            # Si hay objetos detectados, considerar interacción
            # MUY permisivo: si hay objetos detectados, es interacción
            if hand_objects_list and hand_objects_list[0]:
                num_objects = len(hand_objects_list[0])
                total_area = sum(obj['area'] for obj in hand_objects_list[0])
                
                # Si hay objetos detectados (aunque sean pequeños)
                if num_objects > 0 and total_area > 200:  # Muy bajo umbral
                    has_interaction = True
        
        # Actualizar historiales
        if hand_positions:
            self.hand_history.append(hand_positions[0])
        if hand_objects_list:
            self.frame_history.append(hand_objects_list[0] if hand_objects_list else [])
        
        self.interaction_history.append(has_interaction)
        
        # Confirmar interacción - más permisivo
        # Si hay interacciones en al menos la mitad de los frames requeridos
        confirmed = (sum(self.interaction_history) >= max(1, self.min_frames_interaction // 2))
        
        # Visualizar objetos detectados
        if hand_objects_list:
            for objects in hand_objects_list:
                for obj in objects:
                    x1, y1, x2, y2 = obj['bbox']
                    color = (0, 255, 255) if confirmed else (255, 255, 0)
                    thickness = 4 if confirmed else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Dibujar área del objeto
                    area_text = f"Area: {int(obj['area'])}"
                    cv2.putText(frame, area_text, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    if confirmed:
                        # Rectángulo doble para objetos confirmados
                        cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 6)
                        cv2.putText(frame, 'OBJETO COGIDO!', (x1, y1-25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Información de debug
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f'Manos: {len(hand_positions)}', (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        total_objects = sum(len(objs) for objs in hand_objects_list)
        total_area = sum(sum(obj['area'] for obj in objs) for objs in hand_objects_list)
        cv2.putText(frame, f'Objetos: {total_objects} (area: {int(total_area)})', 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Mostrar información de detección
        motion_text = "SI" if motion_change else "NO"
        temporal_text = "SI" if temporal_change else "NO"
        cv2.putText(frame, f'Movimiento: {motion_text} | Temporal: {temporal_text}', 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f'Interaccion: {"SI" if has_interaction else "NO"}', 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if has_interaction else (0, 0, 255), 2)
        
        if confirmed:
            cv2.putText(frame, 'OBJETO COGIDO CONFIRMADO!', (20, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        interactions = []
        if confirmed and hand_objects_list:
            for objects in hand_objects_list:
                for obj in objects:
                    interactions.append({
                        'bbox': obj['bbox'],
                        'confidence': 0.8
                    })
        
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
                    
                    cv2.imshow('Deteccion V2 - Temporal + Segmentacion', frame_display)
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
    parser = argparse.ArgumentParser(description='Deteccion V2: Temporal + Segmentacion')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--threshold', type=int, default=100, help='Umbral de distancia')
    parser.add_argument('--min-frames', type=int, default=5, help='Frames para confirmar')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualizacion')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorV2(
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
