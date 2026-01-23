"""
Detector por segmentación: Detecta objetos en manos usando segmentación
ROI de mano + padding -> segmentar mano/fondo -> detectar contornos restantes
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorSegmentation:
    """
    Detecta objetos en manos usando segmentación
    """
    
    def __init__(self, video_path, padding=80, min_contour_area=500):
        self.video_path = video_path
        self.padding = padding
        self.min_contour_area = min_contour_area
        
        # YOLO pose para muñecas
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # Historial para estabilizar
        self.left_hand_history = deque(maxlen=5)
        self.right_hand_history = deque(maxlen=5)
        self.background_model = None  # Modelo de fondo (primeros frames)
        self.background_frames = []
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector por segmentación inicializado")
    
    def segment_hand_region(self, roi):
        """
        Segmenta la mano en el ROI usando detección de piel (múltiples métodos)
        """
        # Método 1: HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Rangos de color de piel (HSV) - más amplios
        lower_skin1 = np.array([0, 20, 50], dtype=np.uint8)
        upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
        
        lower_skin2 = np.array([0, 30, 40], dtype=np.uint8)
        upper_skin2 = np.array([25, 255, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask_hsv = cv2.bitwise_or(mask1, mask2)
        
        # Método 2: YCrCb (mejor para detección de piel)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        skin_mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combinar máscaras
        skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)
        
        # Morfología para limpiar
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return skin_mask
    
    def segment_background(self, roi, background_roi):
        """
        Segmenta el fondo comparando con modelo de fondo
        """
        if background_roi is None or background_roi.size == 0:
            return None
        
        # Convertir a escala de grises
        gray_current = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_background = cv2.cvtColor(background_roi, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur
        gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)
        gray_background = cv2.GaussianBlur(gray_background, (5, 5), 0)
        
        # Diferencia
        diff = cv2.absdiff(gray_current, gray_background)
        _, bg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Invertir: queremos lo que NO es fondo
        bg_mask = cv2.bitwise_not(bg_mask)
        
        return bg_mask
    
    def detect_object_in_hand_roi(self, frame, wrist_pos, frame_number):
        """
        Detecta objetos en el ROI de la mano usando segmentación
        """
        if wrist_pos is None:
            return False, 0.0
        
        wx, wy = wrist_pos
        h, w = frame.shape[:2]
        
        # Crear ROI con padding
        x1 = max(0, wx - self.padding)
        y1 = max(0, wy - self.padding)
        x2 = min(w, wx + self.padding)
        y2 = min(h, wy + self.padding)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, 0.0
        
        # Segmentar mano (piel)
        hand_mask = self.segment_hand_region(roi)
        
        # Obtener modelo de fondo si no existe
        if len(self.background_frames) < 10 and frame_number < 20:
            self.background_frames.append(roi.copy())
            # Si aún no tenemos suficientes frames, usar análisis estático
            if len(self.background_frames) < 5:
                # Análisis simple: buscar contornos que no sean piel
                hand_mask = self.segment_hand_region(roi)
                # Invertir: lo que no es piel
                non_hand_mask = cv2.bitwise_not(hand_mask)
                # Limpiar
                kernel = np.ones((5, 5), np.uint8)
                non_hand_mask = cv2.morphologyEx(non_hand_mask, cv2.MORPH_CLOSE, kernel)
                # Contornos
                contours, _ = cv2.findContours(non_hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                significant = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
                if len(significant) > 0:
                    roi_area = roi.shape[0] * roi.shape[1]
                    total_area = sum(cv2.contourArea(c) for c in significant)
                    ratio = total_area / roi_area if roi_area > 0 else 0
                    has_obj = ratio > 0.02
                    conf = min(1.0, len(significant) * 0.3 + ratio * 15)
                    return has_obj, conf, significant, non_hand_mask
            return False, 0.0, [], None
        
        # Crear modelo de fondo (promedio de primeros frames)
        if self.background_model is None and len(self.background_frames) >= 5:
            self.background_model = np.mean(self.background_frames, axis=0).astype(np.uint8)
        
        # Segmentar fondo
        bg_mask = None
        if self.background_model is not None:
            bg_mask = self.segment_background(roi, self.background_model)
        
        # Combinar máscaras: queremos lo que NO es mano NI fondo
        # Objeto = todo - mano - fondo
        combined_mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255
        
        # Restar mano
        combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(hand_mask))
        
        # Restar fondo si está disponible
        if bg_mask is not None:
            combined_mask = cv2.bitwise_and(combined_mask, bg_mask)
        
        # Limpiar máscara
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos significativos
        significant_contours = []
        total_object_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                # Verificar que no esté en los bordes (probablemente ruido)
                x, y, w_cont, h_cont = cv2.boundingRect(contour)
                margin = 5  # Margen más pequeño
                
                # Verificar posición relativa al centro (donde está la muñeca)
                center_x, center_y = roi.shape[1] // 2, roi.shape[0] // 2
                contour_center_x = x + w_cont // 2
                contour_center_y = y + h_cont // 2
                dist_to_center = np.sqrt((contour_center_x - center_x)**2 + 
                                        (contour_center_y - center_y)**2)
                
                # Aceptar si está cerca del centro O si no está en los bordes
                if (dist_to_center < self.padding * 0.6 or 
                    (x > margin and y > margin and 
                     x + w_cont < roi.shape[1] - margin and 
                     y + h_cont < roi.shape[0] - margin)):
                    significant_contours.append(contour)
                    total_object_area += area
        
        # Calcular ratio de área de objeto vs área total del ROI
        roi_area = roi.shape[0] * roi.shape[1]
        object_ratio = total_object_area / roi_area if roi_area > 0 else 0
        
        # Si hay contornos significativos y ocupan suficiente área
        # Umbral más bajo para detectar objetos pequeños
        has_object = len(significant_contours) > 0 and object_ratio > 0.02
        
        # Confianza basada en número de contornos y área
        confidence = min(1.0, len(significant_contours) * 0.3 + object_ratio * 15)
        
        return has_object, confidence, significant_contours, combined_mask
    
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
        
        # Analizar cada mano
        left_has_object = False
        right_has_object = False
        left_confidence = 0.0
        right_confidence = 0.0
        left_contours = []
        right_contours = []
        
        if left_wrist:
            left_has_object, left_confidence, left_contours, left_mask = self.detect_object_in_hand_roi(
                frame, left_wrist, frame_number
            )
            
            # Dibujar ROI y contornos
            wx, wy = left_wrist
            x1 = max(0, wx - self.padding)
            y1 = max(0, wy - self.padding)
            x2 = min(w, wx + self.padding)
            y2 = min(h, wy + self.padding)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar contornos de objetos
            for contour in left_contours:
                # Ajustar coordenadas al frame completo
                contour_global = contour.copy()
                contour_global[:, :, 0] += x1
                contour_global[:, :, 1] += y1
                cv2.drawContours(frame, [contour_global], -1, (0, 255, 0), 3)
        
        if right_wrist:
            right_has_object, right_confidence, right_contours, right_mask = self.detect_object_in_hand_roi(
                frame, right_wrist, frame_number
            )
            
            # Dibujar ROI y contornos
            wx, wy = right_wrist
            x1 = max(0, wx - self.padding)
            y1 = max(0, wy - self.padding)
            x2 = min(w, wx + self.padding)
            y2 = min(h, wy + self.padding)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Dibujar contornos de objetos
            for contour in right_contours:
                # Ajustar coordenadas al frame completo
                contour_global = contour.copy()
                contour_global[:, :, 0] += x1
                contour_global[:, :, 1] += y1
                cv2.drawContours(frame, [contour_global], -1, (255, 0, 0), 3)
        
        # Actualizar historial (solo si realmente hay contornos)
        # No contar como positivo si no hay contornos en el frame actual
        left_has_object_real = left_has_object and len(left_contours) > 0
        right_has_object_real = right_has_object and len(right_contours) > 0
        
        self.left_hand_history.append(left_has_object_real)
        self.right_hand_history.append(right_has_object_real)
        
        # Confirmar solo si hay detección consistente Y hay contornos en frame actual
        left_confirmed = sum(self.left_hand_history) >= 2 and len(left_contours) > 0
        right_confirmed = sum(self.right_hand_history) >= 2 and len(right_contours) > 0
        
        # Visualizar confirmación
        if left_wrist and left_confirmed:
            wx, wy = left_wrist
            cv2.putText(frame, 'OBJETO!', (wx-50, wy-self.padding-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
        
        if right_wrist and right_confirmed:
            wx, wy = right_wrist
            cv2.putText(frame, 'OBJETO!', (wx-50, wy-self.padding-30),
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
        left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"} (conf: {left_confidence:.2f}, contornos: {len(left_contours)})'
        cv2.putText(frame, left_text, (25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, left_color, 2)
        
        # Mano derecha
        right_color = (255, 0, 0) if right_confirmed else (150, 150, 150)
        right_bg = (100, 0, 0) if right_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 115), (580, 165), right_bg, -1)
        cv2.rectangle(frame, (15, 115), (580, 165), right_color, 3)
        right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"} (conf: {right_confidence:.2f}, contornos: {len(right_contours)})'
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
                'contours': len(left_contours)
            },
            'right_hand': {
                'has_object': right_confirmed,
                'confidence': right_confidence,
                'contours': len(right_contours)
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
        print("Construyendo modelo de fondo (primeros frames)...")
        
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
                          f"(contornos: {result['left_hand']['contours']}), "
                          f"Der={'OBJETO' if result['right_hand']['has_object'] else 'VACIA'} "
                          f"(contornos: {result['right_hand']['contours']})")
                
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
                    
                    cv2.imshow('Deteccion por Segmentacion', frame_display)
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
    parser = argparse.ArgumentParser(description='Detección por segmentación de objetos en manos')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--padding', type=int, default=80, help='Padding alrededor de muñeca (pixels)')
    parser.add_argument('--min-area', type=int, default=500, help='Área mínima de contorno')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorSegmentation(
        video_path=args.video_path,
        padding=args.padding,
        min_contour_area=args.min_area
    )
    
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
