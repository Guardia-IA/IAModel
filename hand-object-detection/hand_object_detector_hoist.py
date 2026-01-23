"""
Detector inspirado en HOIST-Former: Análisis de forma de mano + atención espacial
Enfoque: MediaPipe Hands para landmarks + análisis de forma + detección de volumen
"""

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage, ImageFormat
from collections import deque
import argparse
import os


class HandObjectDetectorHOIST:
    """
    Detector inspirado en HOIST-Former usando análisis de forma de mano
    """
    
    def __init__(self, video_path, min_frames=3):
        self.video_path = video_path
        self.min_frames = min_frames
        
        # Inicializar MediaPipe Hands
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
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Historial
        self.left_hand_history = deque(maxlen=min_frames)
        self.right_hand_history = deque(maxlen=min_frames)
        self.hand_shapes_history = deque(maxlen=10)
        
        print("✓ Detector HOIST-style inicializado")
    
    def calculate_hand_volume(self, landmarks, frame_width, frame_height):
        """
        Calcula el "volumen" o área ocupada por la mano
        """
        x_coords = [lm.x * frame_width for lm in landmarks]
        y_coords = [lm.y * frame_height for lm in landmarks]
        
        # Bounding box
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Área del bounding box
        bbox_area = (max_x - min_x) * (max_y - min_y)
        
        # Área convexa (envolvente convexa de los landmarks)
        points = np.array([[x, y] for x, y in zip(x_coords, y_coords)])
        if len(points) > 2:
            hull = cv2.convexHull(points)
            convex_area = cv2.contourArea(hull)
        else:
            convex_area = bbox_area
        
        return bbox_area, convex_area, (min_x, min_y, max_x, max_y)
    
    def analyze_hand_shape(self, landmarks):
        """
        Analiza la forma de la mano para detectar si está sosteniendo algo
        """
        # Puntos clave
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        index_tip = np.array([landmarks[8].x, landmarks[8].y])
        middle_tip = np.array([landmarks[12].x, landmarks[12].y])
        ring_tip = np.array([landmarks[16].x, landmarks[16].y])
        pinky_tip = np.array([landmarks[20].x, landmarks[20].y])
        
        thumb_mcp = np.array([landmarks[2].x, landmarks[2].y])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y])
        middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
        
        # Distancias entre puntas de dedos
        thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
        thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
        index_middle_dist = np.linalg.norm(index_tip - middle_tip)
        
        # Distancias punta-base (curvatura de dedos)
        index_curl = np.linalg.norm(index_tip - index_mcp)
        middle_curl = np.linalg.norm(middle_tip - middle_mcp)
        
        # Área del triángulo formado por pulgar, índice y medio
        triangle_area = 0.5 * abs(
            (thumb_tip[0] * (index_tip[1] - middle_tip[1]) +
             index_tip[0] * (middle_tip[1] - thumb_tip[1]) +
             middle_tip[0] * (thumb_tip[1] - index_tip[1]))
        )
        
        # Análisis de forma
        is_closed = (
            thumb_index_dist < 0.08 and
            thumb_middle_dist < 0.08 and
            index_middle_dist < 0.06
        )
        
        is_gripping = (
            thumb_index_dist < 0.12 and
            thumb_middle_dist < 0.12 and
            index_curl < 0.15 and
            middle_curl < 0.15
        )
        
        # Si la mano está cerrada o agarrando, puede tener objeto
        hand_state = "closed" if is_closed else ("gripping" if is_gripping else "open")
        
        return {
            'state': hand_state,
            'thumb_index_dist': thumb_index_dist,
            'thumb_middle_dist': thumb_middle_dist,
            'index_curl': index_curl,
            'middle_curl': middle_curl,
            'triangle_area': triangle_area,
            'is_closed': is_closed,
            'is_gripping': is_gripping
        }
    
    def detect_object_by_volume_change(self, current_volume, history_volumes):
        """
        Detecta objetos comparando el volumen actual con el histórico
        """
        if len(history_volumes) < 3:
            return False, 0.0
        
        # Volumen promedio histórico
        avg_historical = np.mean(history_volumes)
        
        # Si el volumen actual es significativamente mayor, hay objeto
        volume_increase = current_volume / avg_historical if avg_historical > 0 else 1.0
        
        has_object = volume_increase > 1.3  # 30% más grande
        confidence = min(1.0, (volume_increase - 1.0) * 2.0)
        
        return has_object, confidence
    
    def detect_object_by_shape_analysis(self, shape_info, history_shapes):
        """
        Detecta objetos analizando cambios en la forma de la mano
        """
        if len(history_shapes) < 2:
            return False, 0.0
        
        # Si la mano está cerrada o agarrando, es probable que tenga objeto
        if shape_info['is_closed'] or shape_info['is_gripping']:
            # Verificar si antes estaba abierta
            previous_states = [s['state'] for s in history_shapes[-3:]]
            was_open = any(s == 'open' for s in previous_states)
            
            if was_open:
                # Cambió de abierta a cerrada = probablemente cogió algo
                return True, 0.8
            else:
                # Ha estado cerrada = puede tener objeto
                return True, 0.6
        
        return False, 0.0
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Detectar manos con MediaPipe
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.hand_landmarker.detect(mp_image)
        
        left_hand_info = None
        right_hand_info = None
        left_has_object = False
        right_has_object = False
        left_confidence = 0.0
        right_confidence = 0.0
        
        if detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Determinar si es mano izquierda o derecha
                handedness = detection_result.handedness[idx][0] if detection_result.handedness else None
                is_left = handedness is None or handedness.category_name == "Left"
                
                # Analizar forma de la mano
                shape_info = self.analyze_hand_shape(hand_landmarks)
                
                # Calcular volumen
                bbox_area, convex_area, bbox = self.calculate_hand_volume(hand_landmarks, w, h)
                
                # Detectar objeto por volumen
                volume_history = [v['convex_area'] for v in self.hand_shapes_history if v['is_left'] == is_left]
                has_object_volume, conf_volume = self.detect_object_by_volume_change(
                    convex_area, volume_history
                )
                
                # Detectar objeto por forma
                shape_history = [s for s in self.hand_shapes_history if s['is_left'] == is_left]
                has_object_shape, conf_shape = self.detect_object_by_shape_analysis(
                    shape_info, shape_history
                )
                
                # Combinar detecciones
                has_object = has_object_volume or has_object_shape
                confidence = max(conf_volume, conf_shape)
                
                # Guardar en historial
                self.hand_shapes_history.append({
                    'is_left': is_left,
                    'convex_area': convex_area,
                    'bbox_area': bbox_area,
                    'shape': shape_info
                })
                
                if is_left:
                    left_hand_info = {
                        'landmarks': hand_landmarks,
                        'shape': shape_info,
                        'bbox': bbox,
                        'volume': convex_area
                    }
                    left_has_object = has_object
                    left_confidence = confidence
                else:
                    right_hand_info = {
                        'landmarks': hand_landmarks,
                        'shape': shape_info,
                        'bbox': bbox,
                        'volume': convex_area
                    }
                    right_has_object = has_object
                    right_confidence = confidence
                
                # Dibujar landmarks
                color = (0, 255, 0) if is_left else (255, 0, 0)
                for connection in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), 
                                  (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                                  (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
                                  (18, 19), (19, 20)]:
                    if connection[0] < len(hand_landmarks) and connection[1] < len(hand_landmarks):
                        pt1 = (int(hand_landmarks[connection[0]].x * w), 
                              int(hand_landmarks[connection[0]].y * h))
                        pt2 = (int(hand_landmarks[connection[1]].x * w), 
                              int(hand_landmarks[connection[1]].y * h))
                        cv2.line(frame, pt1, pt2, color, 2)
                
                # Dibujar bbox
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Marcar si tiene objeto
                if has_object:
                    cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), 20, color, 5)
                    cv2.putText(frame, 'OBJETO!', (int(x1), int(y1)-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        # Actualizar historial
        self.left_hand_history.append(left_has_object)
        self.right_hand_history.append(right_has_object)
        
        # Confirmar solo si hay detección consistente
        left_confirmed = sum(self.left_hand_history) >= self.min_frames
        right_confirmed = sum(self.right_hand_history) >= self.min_frames
        
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
        left_state = left_hand_info['shape']['state'] if left_hand_info else "N/A"
        left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"} (forma: {left_state}, conf: {left_confidence:.2f})'
        cv2.putText(frame, left_text, (25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, left_color, 2)
        
        # Mano derecha
        right_color = (255, 0, 0) if right_confirmed else (150, 150, 150)
        right_bg = (100, 0, 0) if right_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 115), (580, 165), right_bg, -1)
        cv2.rectangle(frame, (15, 115), (580, 165), right_color, 3)
        right_state = right_hand_info['shape']['state'] if right_hand_info else "N/A"
        right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"} (forma: {right_state}, conf: {right_confidence:.2f})'
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
                'confidence': left_confidence,
                'shape': left_hand_info['shape']['state'] if left_hand_info else None
            },
            'right_hand': {
                'has_object': right_confirmed,
                'confidence': right_confidence,
                'shape': right_hand_info['shape']['state'] if right_hand_info else None
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
                          f"Izq={'OBJETO' if result['left_hand']['has_object'] else 'VACIA'} "
                          f"(forma: {result['left_hand']['shape']}), "
                          f"Der={'OBJETO' if result['right_hand']['has_object'] else 'VACIA'} "
                          f"(forma: {result['right_hand']['shape']})")
                
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
                    
                    cv2.imshow('HOIST-Style Detector', frame_display)
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
    parser = argparse.ArgumentParser(description='Detector HOIST-style: Análisis de forma de mano')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--min-frames', type=int, default=3, help='Frames mínimos para confirmar')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorHOIST(
        video_path=args.video_path,
        min_frames=args.min_frames
    )
    
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
