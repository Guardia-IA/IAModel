"""
Detector usando MediaPipe Interactive Segmentation
1. Detecta persona con YOLO (solo para bounding box)
2. Detecta esqueleto con MediaPipe Pose (más puntos que YOLO)
3. Detecta manos con MediaPipe Hand Landmarks
4. Usa segmentación interactiva de MediaPipe (magic_touch) para segmentar objetos cerca de las manos
"""

import cv2
import numpy as np
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage, ImageFormat
import argparse
import os
from collections import deque


class HandObjectDetectorMediaPipeSeg:
    """
    Detector usando segmentación interactiva de MediaPipe
    """
    
    def __init__(self, video_path, min_frames=3, display_width=1920):
        self.video_path = video_path
        self.min_frames = min_frames
        self.display_width = display_width
        
        # Cargar YOLO solo para detectar personas (bounding box)
        print("Cargando YOLO para detección de personas...")
        self.yolo = YOLO('yolov8n.pt')
        
        # Inicializar MediaPipe Pose para esqueleto (más puntos que YOLO)
        print("Cargando MediaPipe Pose...")
        pose_model_path = 'pose_landmarker_heavy.task'
        if not os.path.exists(pose_model_path):
            import urllib.request
            model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
            print("Descargando modelo de MediaPipe Pose...")
            urllib.request.urlretrieve(model_url, pose_model_path)
        
        base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            output_segmentation_masks=True,
            num_poses=1
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)
        
        # Inicializar MediaPipe Hand Landmarker para manos
        print("Cargando MediaPipe Hand Landmarker...")
        hand_model_path = 'hand_landmarker.task'
        if not os.path.exists(hand_model_path):
            import urllib.request
            model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            print("Descargando modelo de MediaPipe Hand...")
            urllib.request.urlretrieve(model_url, hand_model_path)
        
        base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            num_hands=2,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
        
        # Inicializar MediaPipe Interactive Segmenter (magic_touch)
        print("Cargando MediaPipe Interactive Segmenter (magic_touch)...")
        seg_model_path = 'magic_touch.tflite'
        if not os.path.exists(seg_model_path):
            import urllib.request
            model_url = 'https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite'
            print("Descargando modelo magic_touch.tflite...")
            try:
                urllib.request.urlretrieve(model_url, seg_model_path)
                print("✓ Modelo descargado")
            except Exception as e:
                print(f"Error descargando modelo: {e}")
                seg_model_path = None
        
        if seg_model_path and os.path.exists(seg_model_path):
            try:
                base_options_seg = python.BaseOptions(model_asset_path=seg_model_path)
                options_seg = vision.InteractiveSegmenterOptions(
                    base_options=base_options_seg,
                    output_category_mask=True
                )
                self.interactive_segmenter = vision.InteractiveSegmenter.create_from_options(options_seg)
                print("✓ MediaPipe Interactive Segmenter (magic_touch) cargado")
            except Exception as e:
                print(f"Error cargando Interactive Segmenter: {e}")
                self.interactive_segmenter = None
        else:
            self.interactive_segmenter = None
        
        # Historial
        self.left_hand_history = deque(maxlen=min_frames)
        self.right_hand_history = deque(maxlen=min_frames)
        
        print("✓ Detector MediaPipe Segmentación inicializado")
    
    def get_hand_region(self, hand_landmarks, frame_width, frame_height):
        """Obtiene la región alrededor de la mano"""
        # Obtener puntos clave de la mano
        x_coords = [lm.x * frame_width for lm in hand_landmarks]
        y_coords = [lm.y * frame_height for lm in hand_landmarks]
        
        # Calcular bounding box con padding
        x_min = max(0, int(min(x_coords) - 50))
        y_min = max(0, int(min(y_coords) - 50))
        x_max = min(frame_width, int(max(x_coords) + 50))
        y_max = min(frame_height, int(max(y_coords) + 50))
        
        return (x_min, y_min, x_max, y_max)
    
    def segment_around_hand(self, frame, hand_landmarks, frame_width, frame_height):
        """
        Segmenta objetos alrededor de la mano usando MediaPipe Interactive Segmenter
        Usa los puntos de la mano como puntos de interés para la segmentación interactiva
        """
        if self.interactive_segmenter is None:
            return None, None
        
        # Obtener región de la mano con padding para capturar objetos
        x_min, y_min, x_max, y_max = self.get_hand_region(hand_landmarks, frame_width, frame_height)
        
        # Expandir región
        padding = 150
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame_width, x_max + padding)
        y_max = min(frame_height, y_max + padding)
        
        # Recortar región alrededor de la mano
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None, None
        
        # Convertir ROI a formato MediaPipe
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=roi_rgb)
        
        # Obtener puntos clave de la mano (ajustados al ROI)
        # Usar varios puntos de la mano como puntos de interés
        roi_width = x_max - x_min
        roi_height = y_max - y_min
        
        # Crear puntos de interés desde los landmarks de la mano
        # Usar muñeca (0), índice (5, 8), medio (9, 12), pulgar (4)
        key_points = [
            (0, hand_landmarks[0]),  # Muñeca
            (8, hand_landmarks[8]),  # Punta índice
            (12, hand_landmarks[12]),  # Punta medio
            (4, hand_landmarks[4]),  # Punta pulgar
        ]
        
        # Convertir puntos a coordenadas del ROI y formato MediaPipe
        try:
            # Crear puntos normalizados (0-1) en el ROI
            normalized_points = []
            for idx, landmark in key_points:
                # Convertir coordenadas globales a coordenadas del ROI
                px_global = landmark.x * frame_width
                py_global = landmark.y * frame_height
                px_roi = (px_global - x_min) / roi_width
                py_roi = (py_global - y_min) / roi_height
                
                # Asegurar que están en rango [0, 1]
                px_roi = max(0.0, min(1.0, px_roi))
                py_roi = max(0.0, min(1.0, py_roi))
                
                normalized_points.append((px_roi, py_roi))
            
            # Intentar segmentación interactiva
            # InteractiveSegmenter usa RegionOfInterest con NormalizedKeypoint
            try:
                # Crear RegionOfInterest usando los puntos de la mano
                # Usar el punto de la muñeca como punto central
                wrist_point = normalized_points[0]
                
                # Importar NormalizedKeypoint
                from mediapipe.tasks.python.components.containers import keypoint as keypoint_module
                
                # Crear NormalizedKeypoint
                normalized_keypoint = keypoint_module.NormalizedKeypoint(
                    x=wrist_point[0],
                    y=wrist_point[1]
                )
                
                # Crear ROI
                roi = vision.InteractiveSegmenterRegionOfInterest(
                    format=vision.InteractiveSegmenterRegionOfInterest.Format.KEYPOINT,
                    keypoint=normalized_keypoint
                )
                
                # Segmentar usando el ROI
                segmentation_result = self.interactive_segmenter.segment(mp_image, roi)
            except Exception as e:
                print(f"Error en segmentación interactiva con ROI: {e}")
                # Intentar sin ROI (segmentación general)
                try:
                    segmentation_result = self.interactive_segmenter.segment(mp_image)
                except Exception as e2:
                    print(f"Error en segmentación general: {e2}")
                    return None, None
            
            if segmentation_result.category_mask is not None:
                # Obtener máscara
                mask = segmentation_result.category_mask.numpy_view()
                
                # Crear máscara binaria
                if len(mask.shape) == 2:
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                else:
                    # Si es multi-categoría, usar la categoría más cercana a los puntos de la mano
                    binary_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
                    # Usar la categoría en el punto de la muñeca
                    wrist_x_roi = int(normalized_points[0][0] * roi_width)
                    wrist_y_roi = int(normalized_points[0][1] * roi_height)
                    if 0 <= wrist_y_roi < mask.shape[0] and 0 <= wrist_x_roi < mask.shape[1]:
                        hand_category = mask[wrist_y_roi, wrist_x_roi]
                        binary_mask[mask == hand_category] = 255
                
                # Encontrar contornos
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filtrar contornos significativos
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Área mínima
                        valid_contours.append(contour)
                
                if valid_contours:
                    # Crear máscara final
                    mask_valid = np.zeros((roi_height, roi_width), dtype=np.uint8)
                    cv2.drawContours(mask_valid, valid_contours, -1, 255, -1)
                    
                    # Ajustar coordenadas al frame completo
                    mask_full = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    mask_full[y_min:y_max, x_min:x_max] = mask_valid
                    
                    return mask_full, valid_contours
        except Exception as e:
            print(f"Error en segmentación interactiva: {e}")
        
        return None, None
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)
        
        # 1. Detectar persona con YOLO (solo para bounding box)
        yolo_results = self.yolo(frame, conf=0.3, verbose=False)
        
        # 2. Detectar esqueleto con MediaPipe Pose (más puntos)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)
        pose_result = self.pose_landmarker.detect(mp_image)
        
        left_hand_detected = False
        right_hand_detected = False
        left_mask = None
        right_mask = None
        left_contours = None
        right_contours = None
        
        # Dibujar bounding boxes de personas detectadas
        if yolo_results[0].boxes is not None:
            for box in yolo_results[0].boxes:
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:  # Persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Dibujar bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, 'Person', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
        
        # Dibujar esqueleto con MediaPipe Pose (más puntos que YOLO)
        if pose_result.pose_landmarks:
            # MediaPipe Pose tiene 33 puntos
            landmarks = pose_result.pose_landmarks[0]
            
            # Conexiones de MediaPipe Pose (33 puntos)
            POSE_CONNECTIONS = [
                (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
                (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
                (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
                (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
            ]
            
            # Dibujar conexiones
            for connection in POSE_CONNECTIONS:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
                    pt1 = (int(landmarks[pt1_idx].x * w), int(landmarks[pt1_idx].y * h))
                    pt2 = (int(landmarks[pt2_idx].x * w), int(landmarks[pt2_idx].y * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
            
            # Dibujar puntos clave
            for landmark in landmarks:
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
        
        # 3. Detectar manos con MediaPipe Hand Landmarker
        hand_result = self.hand_landmarker.detect(mp_image)
        
        if hand_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                # Determinar si es mano izquierda o derecha
                hand_label = hand_result.handedness[idx][0].category_name if hand_result.handedness else "Unknown"
                
                # Dibujar landmarks de la mano
                for lm in hand_landmarks:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
                
                # 4. Segmentar objetos alrededor de la mano usando Interactive Segmenter
                mask, contours = self.segment_around_hand(frame, hand_landmarks, w, h)
                
                if mask is not None and contours:
                    # Determinar mano
                    is_left = hand_label == "Left" or idx == 0
                    
                    if is_left:
                        left_hand_detected = True
                        left_mask = mask
                        left_contours = contours
                    else:
                        right_hand_detected = True
                        right_mask = mask
                        right_contours = contours
                    
                    # Dibujar máscara y contornos
                    color = (0, 255, 255) if is_left else (255, 0, 255)
                    
                    # Crear overlay con color
                    overlay = frame.copy()
                    overlay[mask > 0] = color
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    
                    # Dibujar contornos (ajustar coordenadas)
                    for contour in contours:
                        # Los contornos están en coordenadas del ROI, ajustar
                        x_min, y_min, x_max, y_max = self.get_hand_region(hand_landmarks, w, h)
                        contour_global = contour.copy()
                        contour_global[:, :, 0] += x_min
                        contour_global[:, :, 1] += y_min
                        cv2.drawContours(frame, [contour_global], -1, color, 3)
        
        # Actualizar historial
        self.left_hand_history.append(left_hand_detected)
        self.right_hand_history.append(right_hand_detected)
        
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
        print("Procesando con segmentación interactiva de MediaPipe...\n")
        
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
                
                cv2.imshow('MediaPipe Interactive Segmentation', frame_display)
                
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
    parser = argparse.ArgumentParser(description='Detector con MediaPipe Interactive Segmentation')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--min-frames', type=int, default=3, help='Frames mínimos para confirmar')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorMediaPipeSeg(
        video_path=args.video_path,
        min_frames=args.min_frames,
        display_width=args.display_width
    )
    detector.process_video()


if __name__ == '__main__':
    main()
