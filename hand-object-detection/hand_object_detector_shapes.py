"""
Detector por formas geométricas: Detecta polígonos/figuras en el área de las manos
Enfoque: Segmentación + detección de formas (rectángulos, círculos, polígonos)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorShapes:
    """
    Detecta objetos identificando formas geométricas en el área de las manos
    """
    
    def __init__(self, video_path, detection_radius=150, min_frames=3):
        self.video_path = video_path
        self.detection_radius = detection_radius
        self.min_frames = min_frames
        
        # YOLO pose para muñecas
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # Historial
        self.left_hand_history = deque(maxlen=min_frames)
        self.right_hand_history = deque(maxlen=min_frames)
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector por formas geométricas inicializado")
    
    def detect_shapes_in_region(self, roi):
        """
        Detecta formas geométricas y bultos en una región
        Usa múltiples métodos: bordes, segmentación por color, y análisis de densidad
        """
        if roi.size == 0:
            return []
        
        shapes = []
        
        # MÉTODO 1: Detección por bordes (para objetos con bordes claros)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar bordes (múltiples métodos)
        edges1 = cv2.Canny(blurred, 50, 150)
        edges2 = cv2.Canny(blurred, 30, 100)  # Más sensible
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Dilatar bordes para conectar líneas
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                continue
            
            # Aproximar contorno a polígono
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Clasificar forma
            shape_type = "unknown"
            vertices = len(approx)
            
            if vertices == 3:
                shape_type = "triangle"
            elif vertices == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.8 <= aspect_ratio <= 1.2:
                    shape_type = "square"
                else:
                    shape_type = "rectangle"
            elif vertices == 5:
                shape_type = "pentagon"
            elif vertices == 6:
                shape_type = "hexagon"
            elif vertices > 6:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circle_area = np.pi * radius * radius
                if abs(area - circle_area) / circle_area < 0.3:
                    shape_type = "circle"
                else:
                    shape_type = f"polygon_{vertices}"
            else:
                shape_type = "irregular"
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
            
            shapes.append({
                'type': shape_type,
                'vertices': vertices,
                'area': area,
                'bbox': [x, y, x+w, y+h],
                'contour': approx,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'method': 'edges'
            })
        
        # MÉTODO 2: Segmentación por umbral adaptativo (para bultos/objetos blandos)
        # Convertir a HSV para mejor segmentación
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calcular umbral adaptativo en el canal V (brillo) - más sensible
        v_channel = hsv[:, :, 2]
        adaptive_thresh = cv2.adaptiveThreshold(
            v_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 5  # Ventana más grande, constante más alta
        )
        
        # También usar umbral Otsu
        _, otsu_thresh = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Umbral simple adicional (más permisivo)
        _, simple_thresh = cv2.threshold(v_channel, np.percentile(v_channel, 40), 255, cv2.THRESH_BINARY_INV)
        
        # Combinar todos los umbrales
        combined_thresh = cv2.bitwise_or(adaptive_thresh, otsu_thresh)
        combined_thresh = cv2.bitwise_or(combined_thresh, simple_thresh)
        
        # Limpiar ruido (operaciones morfológicas más agresivas)
        kernel_clean = np.ones((7, 7), np.uint8)
        combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
        combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_clean, iterations=1)
        
        # Encontrar contornos en la segmentación
        contours_seg, _ = cv2.findContours(combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_seg:
            area = cv2.contourArea(contour)
            if area < 400:  # Área mínima más baja para bultos
                continue
            
            # Calcular características del bulto
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # Filtrar formas muy alargadas o muy pequeñas
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue
            
            # Aproximar contorno
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            # Clasificar como bulto si es irregular pero significativo (más permisivo)
            if extent > 0.2:  # Al menos 20% del bbox está ocupado
                if vertices > 8:
                    shape_type = "bulge"
                elif vertices > 4:
                    shape_type = f"bulge_{vertices}"
                else:
                    shape_type = "bulge_compact"
                
                # Verificar si ya existe una forma similar (evitar duplicados)
                # Para bultos, ser más permisivo (no filtrar si es un bulto)
                is_duplicate = False
                for existing_shape in shapes:
                    ex_bbox = existing_shape['bbox']
                    # Si los bboxes se solapan significativamente, es duplicado
                    overlap_x = max(0, min(ex_bbox[2], x+w) - max(ex_bbox[0], x))
                    overlap_y = max(0, min(ex_bbox[3], y+h) - max(ex_bbox[1], y))
                    overlap_area = overlap_x * overlap_y
                    # Si el existente es un bulto y este también, no es duplicado (puede haber múltiples bultos)
                    if 'bulge' in existing_shape['type'] and extent > 0.2:
                        continue  # Permitir múltiples bultos
                    if overlap_area > min(area, existing_shape['area']) * 0.5:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    shapes.append({
                        'type': shape_type,
                        'vertices': vertices,
                        'area': area,
                        'bbox': [x, y, x+w, y+h],
                        'contour': approx,
                        'aspect_ratio': aspect_ratio,
                        'extent': extent,
                        'method': 'segmentation'
                    })
        
        # MÉTODO 3: Análisis de densidad de píxeles (para detectar regiones compactas)
        # Calcular gradiente para detectar cambios de textura
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        if gradient_magnitude.max() > 0:
            gradient_magnitude = np.uint8(255 * gradient_magnitude / gradient_magnitude.max())
        else:
            gradient_magnitude = np.uint8(gradient_magnitude)
        
        # Umbralizar gradiente para encontrar regiones con textura (más sensible)
        _, grad_thresh = cv2.threshold(gradient_magnitude, 30, 255, cv2.THRESH_BINARY)  # Umbral más bajo
        
        # Encontrar regiones conectadas con alta densidad de gradiente
        contours_grad, _ = cv2.findContours(grad_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_grad:
            area = cv2.contourArea(contour)
            if area < 600:  # Área mínima más baja para regiones de textura
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
            
            if aspect_ratio > 4 or aspect_ratio < 0.25:
                continue
            
            # Verificar si ya existe una forma similar
            is_duplicate = False
            for existing_shape in shapes:
                ex_bbox = existing_shape['bbox']
                overlap_x = max(0, min(ex_bbox[2], x+w) - max(ex_bbox[0], x))
                overlap_y = max(0, min(ex_bbox[3], y+h) - max(ex_bbox[1], y))
                overlap_area = overlap_x * overlap_y
                if overlap_area > min(area, existing_shape['area']) * 0.4:
                    is_duplicate = True
                    break
            
            if not is_duplicate and extent > 0.25:
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                vertices = len(approx)
                
                shapes.append({
                    'type': f"texture_{vertices}",
                    'vertices': vertices,
                    'area': area,
                    'bbox': [x, y, x+w, y+h],
                    'contour': approx,
                    'aspect_ratio': aspect_ratio,
                    'extent': extent,
                    'method': 'gradient'
                })
        
        # MÉTODO 4: Detección de bultos por diferencia de color con el fondo
        # Calcular el color promedio del fondo (bordes de la ROI)
        h_roi, w_roi = roi.shape[:2]
        border_width = min(20, h_roi // 10, w_roi // 10)
        
        # Obtener muestras del borde
        top_border = roi[0:border_width, :]
        bottom_border = roi[h_roi-border_width:h_roi, :]
        left_border = roi[:, 0:border_width]
        right_border = roi[:, w_roi-border_width:w_roi]
        
        # Calcular color promedio del borde
        border_pixels = np.vstack([
            top_border.reshape(-1, 3),
            bottom_border.reshape(-1, 3),
            left_border.reshape(-1, 3),
            right_border.reshape(-1, 3)
        ])
        bg_color = np.mean(border_pixels, axis=0)
        
        # Calcular diferencia de color con el fondo
        roi_bgr = roi.astype(np.float32)
        bg_diff = np.sqrt(np.sum((roi_bgr - bg_color)**2, axis=2))
        bg_diff = np.uint8(255 * bg_diff / bg_diff.max() if bg_diff.max() > 0 else bg_diff)
        
        # Umbralizar para encontrar regiones diferentes al fondo (más sensible)
        # Usar percentil en lugar de umbral fijo para adaptarse mejor
        threshold_value = max(30, np.percentile(bg_diff, 60))  # 60% de los píxeles
        _, bg_thresh = cv2.threshold(bg_diff, int(threshold_value), 255, cv2.THRESH_BINARY)
        
        # Limpiar
        kernel_bg = np.ones((9, 9), np.uint8)
        bg_thresh = cv2.morphologyEx(bg_thresh, cv2.MORPH_CLOSE, kernel_bg, iterations=2)
        bg_thresh = cv2.morphologyEx(bg_thresh, cv2.MORPH_OPEN, kernel_bg, iterations=1)
        
        # Encontrar contornos
        contours_bg, _ = cv2.findContours(bg_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_bg:
            area = cv2.contourArea(contour)
            if area < 400:  # Área mínima
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # Filtrar formas muy alargadas
            if aspect_ratio > 4 or aspect_ratio < 0.25:
                continue
            
            # Verificar si ya existe una forma similar
            # Para bultos, ser más permisivo
            is_duplicate = False
            for existing_shape in shapes:
                ex_bbox = existing_shape['bbox']
                overlap_x = max(0, min(ex_bbox[2], x+w) - max(ex_bbox[0], x))
                overlap_y = max(0, min(ex_bbox[3], y+h) - max(ex_bbox[1], y))
                overlap_area = overlap_x * overlap_y
                # Si ambos son bultos, permitir múltiples
                if 'bulge' in existing_shape.get('type', '') and extent > 0.15:
                    continue
                if overlap_area > min(area, existing_shape['area']) * 0.4:  # Umbral más alto para bultos
                    is_duplicate = True
                    break
            
            if not is_duplicate and extent > 0.12:  # Muy permisivo para bultos (12%)
                epsilon = 0.06 * cv2.arcLength(contour, True)  # Más suave para capturar bultos
                approx = cv2.approxPolyDP(contour, epsilon, True)
                vertices = len(approx)
                
                # Clasificar como bulto (siempre como "bulge" para objetos blandos)
                if area > 800:  # Bultos grandes siempre son "bulge"
                    shape_type = "bulge"
                elif vertices > 5:
                    shape_type = "bulge"
                else:
                    shape_type = f"bulge_{vertices}"
                
                shapes.append({
                    'type': shape_type,
                    'vertices': vertices,
                    'area': area,
                    'bbox': [x, y, x+w, y+h],
                    'contour': approx,
                    'aspect_ratio': aspect_ratio,
                    'extent': extent,
                    'method': 'background_diff'
                })
        
        return shapes
    
    def analyze_hand_region(self, frame, wrist_pos):
        """
        Analiza el área alrededor de la muñeca para detectar formas
        """
        if wrist_pos is None:
            return False, [], 0.0
        
        wx, wy = wrist_pos
        h, w = frame.shape[:2]
        
        # Definir región circular alrededor de la muñeca
        x1 = max(0, wx - self.detection_radius)
        y1 = max(0, wy - self.detection_radius)
        x2 = min(w, wx + self.detection_radius)
        y2 = min(h, wy + self.detection_radius)
        
        # Recortar región
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, [], 0.0
        
        # Detectar formas en la región
        shapes = self.detect_shapes_in_region(roi)
        
        # Filtrar formas que probablemente son objetos (no ruido)
        significant_shapes = []
        for shape in shapes:
            # Filtrar por área mínima y máxima (más permisivo para bultos)
            min_area = 300 if shape.get('method') == 'edges' else 500
            max_area = 50000
            
            if min_area < shape['area'] < max_area:
                # Para bultos y texturas, ser más permisivo con extent
                min_extent = 0.2 if shape.get('method') in ['segmentation', 'gradient'] else 0.3
                if shape['extent'] > min_extent:
                    significant_shapes.append(shape)
        
        # Si hay formas significativas, probablemente hay objeto
        has_object = len(significant_shapes) > 0
        
        # Calcular confianza basada en número y tipo de formas
        confidence = 0.0
        if significant_shapes:
            # Formas regulares (cuadrado, rectángulo, círculo) tienen más peso
            regular_shapes = [s for s in significant_shapes if s['type'] in ['square', 'rectangle', 'circle']]
            confidence = min(1.0, len(regular_shapes) * 0.4 + len(significant_shapes) * 0.2)
        
        return has_object, significant_shapes, confidence
    
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
                                    
                                    # Dibujar área de análisis (círculo)
                                    cv2.circle(frame, wrist_pos, self.detection_radius, color, 2)
        
        # Analizar formas en cada mano
        left_has_object = False
        right_has_object = False
        left_shapes = []
        right_shapes = []
        left_confidence = 0.0
        right_confidence = 0.0
        
        if left_wrist:
            left_has_object, left_shapes, left_confidence = self.analyze_hand_region(frame, left_wrist)
            
            # Dibujar formas detectadas
            wx, wy = left_wrist
            x1 = max(0, wx - self.detection_radius)
            y1 = max(0, wy - self.detection_radius)
            
            for shape in left_shapes:
                # Ajustar coordenadas al frame completo
                shape_bbox = shape['bbox']
                global_bbox = [shape_bbox[0] + x1, shape_bbox[1] + y1, 
                              shape_bbox[2] + x1, shape_bbox[3] + y1]
                
                # Ajustar contorno al frame completo
                contour_global = shape['contour'].copy()
                contour_global[:, :, 0] += x1
                contour_global[:, :, 1] += y1
                
                # Crear overlay para relleno semi-transparente
                overlay = frame.copy()
                
                # Color llamativo para mano izquierda (amarillo/cyan)
                fill_color = (0, 255, 255)  # Cyan brillante
                contour_color = (0, 200, 200)  # Cyan más oscuro para contorno
                
                # Dibujar relleno
                cv2.fillPoly(overlay, [contour_global], fill_color)
                # Aplicar transparencia
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                
                # Dibujar contorno de la forma (más grueso y visible)
                cv2.drawContours(frame, [contour_global], -1, contour_color, 4)
                
                # Etiquetar forma
                x, y = global_bbox[0], global_bbox[1]
                cv2.putText(frame, f"{shape['type']}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
        
        if right_wrist:
            right_has_object, right_shapes, right_confidence = self.analyze_hand_region(frame, right_wrist)
            
            # Dibujar formas detectadas
            wx, wy = right_wrist
            x1 = max(0, wx - self.detection_radius)
            y1 = max(0, wy - self.detection_radius)
            
            for shape in right_shapes:
                # Ajustar coordenadas al frame completo
                shape_bbox = shape['bbox']
                global_bbox = [shape_bbox[0] + x1, shape_bbox[1] + y1, 
                              shape_bbox[2] + x1, shape_bbox[3] + y1]
                
                # Ajustar contorno al frame completo
                contour_global = shape['contour'].copy()
                contour_global[:, :, 0] += x1
                contour_global[:, :, 1] += y1
                
                # Crear overlay para relleno semi-transparente
                overlay = frame.copy()
                
                # Color llamativo para mano derecha (magenta/rosa)
                fill_color = (255, 0, 255)  # Magenta brillante
                contour_color = (200, 0, 200)  # Magenta más oscuro para contorno
                
                # Dibujar relleno
                cv2.fillPoly(overlay, [contour_global], fill_color)
                # Aplicar transparencia
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                
                # Dibujar contorno de la forma (más grueso y visible)
                cv2.drawContours(frame, [contour_global], -1, contour_color, 4)
                
                # Etiquetar forma
                x, y = global_bbox[0], global_bbox[1]
                cv2.putText(frame, f"{shape['type']}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 3)
        
        # Actualizar historial
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
        cv2.rectangle(overlay, (10, 10), (700, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_number}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mano izquierda
        left_color = (0, 255, 0) if left_confirmed else (150, 150, 150)
        left_bg = (0, 100, 0) if left_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 55), (680, 105), left_bg, -1)
        cv2.rectangle(frame, (15, 55), (680, 105), left_color, 3)
        
        if left_shapes:
            shapes_str = ", ".join([s['type'] for s in left_shapes[:3]])
            left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"} - Formas: {shapes_str} ({len(left_shapes)})'
        else:
            left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"} - Sin formas detectadas'
        cv2.putText(frame, left_text, (25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, left_color, 2)
        
        # Mano derecha
        right_color = (255, 0, 0) if right_confirmed else (150, 150, 150)
        right_bg = (100, 0, 0) if right_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 115), (680, 165), right_bg, -1)
        cv2.rectangle(frame, (15, 115), (680, 165), right_color, 3)
        
        if right_shapes:
            shapes_str = ", ".join([s['type'] for s in right_shapes[:3]])
            right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"} - Formas: {shapes_str} ({len(right_shapes)})'
        else:
            right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"} - Sin formas detectadas'
        cv2.putText(frame, right_text, (25, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, right_color, 2)
        
        # Información de formas
        cv2.putText(frame, f'Formas detectadas - Izq: {len(left_shapes)}, Der: {len(right_shapes)}', (20, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
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
            cv2.rectangle(frame, (15, 210), (20 + text_size[0], 240), (0, 0, 0), -1)
            cv2.putText(frame, summary_text, (20, 235),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, summary_color, 4)
        
        return {
            'frame': frame,
            'left_hand': {
                'has_object': left_confirmed,
                'shapes': left_shapes,
                'confidence': left_confidence
            },
            'right_hand': {
                'has_object': right_confirmed,
                'shapes': right_shapes,
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
        print("Detectando formas geométricas en áreas de manos...\n")
        
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
        all_shape_types = set()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                result = self.process_frame(frame, frame_count)
                
                # Registrar tipos de formas
                for shape in result['left_hand']['shapes']:
                    all_shape_types.add(shape['type'])
                for shape in result['right_hand']['shapes']:
                    all_shape_types.add(shape['type'])
                
                if result['left_hand']['has_object'] or result['right_hand']['has_object']:
                    frames_with_objects.append({
                        'frame': frame_count,
                        'left': result['left_hand']['has_object'],
                        'right': result['right_hand']['has_object'],
                        'left_shapes': [s['type'] for s in result['left_hand']['shapes']],
                        'right_shapes': [s['type'] for s in result['right_hand']['shapes']]
                    })
                    
                    left_shapes_list = [s['type'] for s in result['left_hand']['shapes'][:3]] if result['left_hand']['shapes'] else []
                    right_shapes_list = [s['type'] for s in result['right_hand']['shapes'][:3]] if result['right_hand']['shapes'] else []
                    left_shapes_str = ", ".join(left_shapes_list) if left_shapes_list else "ninguna"
                    right_shapes_str = ", ".join(right_shapes_list) if right_shapes_list else "ninguna"
                    
                    print(f"  Frame {frame_count}: "
                          f"Izq={'OBJETO' if result['left_hand']['has_object'] else 'VACIA'} "
                          f"(formas: {left_shapes_str}), "
                          f"Der={'OBJETO' if result['right_hand']['has_object'] else 'VACIA'} "
                          f"(formas: {right_shapes_str})")
                
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
                    
                    cv2.imshow('Shape Detector', frame_display)
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
            print(f"  - Tipos de formas detectadas: {sorted(all_shape_types)}")
            if frames_with_objects:
                print(f"  - Primeros frames con objetos: {[f['frame'] for f in frames_with_objects[:10]]}")


def main():
    parser = argparse.ArgumentParser(description='Detector por formas geométricas')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--radius', type=int, default=150, help='Radio del área alrededor de muñeca')
    parser.add_argument('--min-frames', type=int, default=3, help='Frames mínimos para confirmar')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorShapes(
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
