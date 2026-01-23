"""
Detector por segmentación de cuerpo: Si la mano es más grande de lo normal, tiene objeto
Enfoque: Segmentar persona + analizar tamaño de mano relativo al cuerpo
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorBodySeg:
    """
    Detecta objetos analizando el tamaño de la mano relativo al cuerpo
    """
    
    def __init__(self, video_path, min_frames=3, size_threshold=1.3):
        self.video_path = video_path
        self.min_frames = min_frames
        self.size_threshold = size_threshold  # Si la mano es 30% más grande, tiene objeto
        
        # YOLO pose para muñecas y personas
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # YOLO con segmentación para personas
        print("Cargando YOLO-Seg para segmentación de personas...")
        self.yolo_seg = YOLO('yolov8n-seg.pt')
        
        # Historial
        self.left_hand_history = deque(maxlen=min_frames)
        self.right_hand_history = deque(maxlen=min_frames)
        self.hand_size_baseline = {'left': None, 'right': None}  # Tamaño base de mano vacía
        self.hand_size_history = {'left': deque(maxlen=10), 'right': deque(maxlen=10)}
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector por segmentación de cuerpo inicializado")
    
    def calculate_hand_size_from_body(self, person_mask, wrist_pos, shoulder_pos, frame_number):
        """
        Calcula el tamaño de la mano relativo al cuerpo usando segmentación (mejorado)
        """
        if wrist_pos is None or shoulder_pos is None or person_mask is None:
            return None, None
        
        wx, wy = wrist_pos
        sx, sy = shoulder_pos
        
        # Calcular distancia hombro-muñeca (referencia de tamaño corporal)
        shoulder_wrist_dist = np.sqrt((wx - sx)**2 + (wy - sy)**2)
        
        if shoulder_wrist_dist < 50:  # Distancia muy pequeña, probablemente error
            return None, None
        
        # Definir región alrededor de la muñeca (más grande para capturar objetos)
        hand_radius = int(shoulder_wrist_dist * 0.4)  # 40% de la distancia (más grande)
        hand_radius = max(hand_radius, 50)  # Mínimo 50 píxeles
        hand_radius = min(hand_radius, 200)  # Máximo 200 píxeles
        
        h, w = person_mask.shape[:2]
        x1 = max(0, wx - hand_radius)
        y1 = max(0, wy - hand_radius)
        x2 = min(w, wx + hand_radius)
        y2 = min(h, wy + hand_radius)
        
        # Recortar región de la mano
        hand_region = person_mask[y1:y2, x1:x2]
        if hand_region.size == 0:
            return None, None
        
        # Calcular área de la mano en la máscara (píxeles del cuerpo en esa región)
        hand_area = np.sum(hand_region > 0)
        
        # Área total de la región
        region_area = hand_region.shape[0] * hand_region.shape[1]
        
        # Ratio de ocupación
        hand_occupation_ratio = hand_area / region_area if region_area > 0 else 0
        
        # Tamaño relativo (área de mano / distancia de referencia)
        # Usar área absoluta también como indicador
        hand_size_relative = hand_area / (shoulder_wrist_dist + 1)  # +1 para evitar división por 0
        
        # Si el área absoluta es muy grande, probablemente hay objeto
        # Normalizar por área de región para ser más robusto
        if hand_area > 10000:  # Área absoluta grande
            hand_size_relative = max(hand_size_relative, hand_area / 100.0)  # Aumentar tamaño relativo
        
        return hand_size_relative, hand_occupation_ratio
    
    def detect_object_by_hand_size(self, current_size, hand_side, frame_number):
        """
        Detecta objeto comparando tamaño actual con baseline (más robusto y adaptativo)
        """
        # Establecer baseline en primeros frames (mano vacía)
        if frame_number < 30:  # Más frames para calibración
            if current_size is not None and current_size > 0:
                self.hand_size_history[hand_side].append(current_size)
            
            # Calcular baseline después de varios frames (usar mediana para ser más robusto)
            if len(self.hand_size_history[hand_side]) >= 5 and self.hand_size_baseline[hand_side] is None:
                sizes = list(self.hand_size_history[hand_side])
                # Filtrar valores extremos (ceros y valores muy grandes)
                sizes_filtered = [s for s in sizes if s > 0 and s < 200]
                if len(sizes_filtered) >= 3:
                    sizes_sorted = sorted(sizes_filtered)
                    # Usar percentil 50 (mediana) y percentil 25-75 para baseline más robusto
                    median = np.median(sizes_sorted)
                    q25 = np.percentile(sizes_sorted, 25)
                    q75 = np.percentile(sizes_sorted, 75)
                    # Baseline = mediana, pero ajustado si hay mucha variación
                    self.hand_size_baseline[hand_side] = median
                    print(f"  Baseline {hand_side}: {self.hand_size_baseline[hand_side]:.2f} (mediana de {len(sizes_filtered)} valores, rango: {q25:.2f}-{q75:.2f})")
            
            return False, 0.0
        
        # Comparar con baseline Y con historial
        has_object = False
        confidence = 0.0
        size_ratio = 1.0
        
        # Método 1: Comparar con baseline si existe
        if current_size is not None and current_size > 0 and self.hand_size_baseline[hand_side] is not None:
            size_ratio = current_size / self.hand_size_baseline[hand_side] if self.hand_size_baseline[hand_side] > 0 else 1.0
            if size_ratio > self.size_threshold:
                has_object = True
                confidence = min(1.0, (size_ratio - 1.0) * 1.5)
        
        # Método 2: Tamaño absoluto grande
        if current_size > 50:  # Tamaño absoluto grande
            has_object = True
            if self.hand_size_baseline[hand_side] is not None:
                size_ratio = max(size_ratio, current_size / (self.hand_size_baseline[hand_side] + 1))
            confidence = max(confidence, min(1.0, current_size / 100.0))
        
        # Método 3: Comparar con historial reciente (más robusto)
        if len(self.hand_size_history[hand_side]) >= 3:
            recent_sizes = list(self.hand_size_history[hand_side])[-5:]
            recent_avg = np.mean([s for s in recent_sizes if s > 0])
            if recent_avg > 0 and current_size > recent_avg * 1.3:  # 30% mayor que promedio reciente
                has_object = True
                size_ratio = max(size_ratio, current_size / recent_avg)
                confidence = max(confidence, min(1.0, (current_size / recent_avg - 1.0) * 2.0))
        
        # Actualizar historial siempre
        if current_size is not None and current_size > 0:
            self.hand_size_history[hand_side].append(current_size)
        
        # Actualizar baseline adaptativamente solo si no hay objeto detectado
        if not has_object and self.hand_size_baseline[hand_side] is not None and len(self.hand_size_history[hand_side]) > 0:
            recent_avg = np.mean([s for s in list(self.hand_size_history[hand_side])[-5:] if s > 0])
            if recent_avg > 0 and current_size < recent_avg * 1.5:  # Si no es mucho mayor, puede ser mano vacía
                # Actualizar baseline con promedio móvil (peso bajo para cambios graduales)
                self.hand_size_baseline[hand_side] = self.hand_size_baseline[hand_side] * 0.95 + current_size * 0.05
        
        return has_object, confidence
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        
        # Detectar personas con YOLO pose
        pose_results = self.yolo_pose(frame, conf=0.3, verbose=False)
        
        # Segmentar personas con YOLO-Seg
        seg_results = self.yolo_seg(frame, conf=0.3, classes=[0], verbose=False)  # Solo personas
        
        # Extraer muñecas y hombros
        left_wrist = None
        right_wrist = None
        left_shoulder = None
        right_shoulder = None
        person_mask = None
        
        # Obtener máscara de persona
        if seg_results[0].masks is not None and len(seg_results[0].masks) > 0:
            # Usar la primera persona detectada
            mask = seg_results[0].masks.data[0].cpu().numpy()
            person_mask = cv2.resize(mask, (w, h))
            person_mask = (person_mask > 0.5).astype(np.uint8) * 255
        
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
                        
                        # Extraer muñecas (9, 10) y hombros (5, 6)
                        if len(keypoints) > 10:
                            # Hombros
                            for shoulder_idx in [5, 6]:
                                if (keypoints[shoulder_idx][0] > 0 and keypoints[shoulder_idx][1] > 0 and
                                    (confidences is None or confidences[shoulder_idx] > 0.3)):
                                    shoulder_pos = tuple(map(int, keypoints[shoulder_idx]))
                                    if shoulder_idx == 5:
                                        left_shoulder = shoulder_pos
                                    elif shoulder_idx == 6:
                                        right_shoulder = shoulder_pos
                            
                            # Muñecas
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
        
        # Analizar tamaño de manos
        left_has_object = False
        right_has_object = False
        left_confidence = 0.0
        right_confidence = 0.0
        left_size = None
        right_size = None
        
        if person_mask is not None:
            # Mano izquierda
            if left_wrist and left_shoulder:
                left_size, left_occupation = self.calculate_hand_size_from_body(
                    person_mask, left_wrist, left_shoulder, frame_number
                )
                if left_size is not None:
                    left_has_object, left_confidence = self.detect_object_by_hand_size(
                        left_size, 'left', frame_number
                    )
                    
                    # Dibujar región de análisis
                    wx, wy = left_wrist
                    sx, sy = left_shoulder
                    dist = np.sqrt((wx - sx)**2 + (wy - sy)**2)
                    radius = int(dist * 0.3)
                    cv2.circle(frame, left_wrist, radius, (0, 255, 0), 2)
            
            # Mano derecha
            if right_wrist and right_shoulder:
                right_size, right_occupation = self.calculate_hand_size_from_body(
                    person_mask, right_wrist, right_shoulder, frame_number
                )
                if right_size is not None:
                    right_has_object, right_confidence = self.detect_object_by_hand_size(
                        right_size, 'right', frame_number
                    )
                    
                    # Dibujar región de análisis
                    wx, wy = right_wrist
                    sx, sy = right_shoulder
                    dist = np.sqrt((wx - sx)**2 + (wy - sy)**2)
                    radius = int(dist * 0.3)
                    cv2.circle(frame, right_wrist, radius, (255, 0, 0), 2)
        
        # Actualizar historial
        self.left_hand_history.append(left_has_object)
        self.right_hand_history.append(right_has_object)
        
        # Confirmar solo si hay detección consistente (más permisivo)
        # Si hay detección en frame actual, ser más permisivo
        min_frames_required = self.min_frames
        if left_has_object or right_has_object:
            min_frames_required = max(1, self.min_frames - 1)  # Más permisivo si hay detección actual
        
        left_confirmed = sum(self.left_hand_history) >= min_frames_required
        right_confirmed = sum(self.right_hand_history) >= min_frames_required
        
        # Visualizar confirmación
        if left_wrist and left_confirmed:
            cv2.putText(frame, 'OBJETO!', (left_wrist[0]-50, left_wrist[1]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
        
        if right_wrist and right_confirmed:
            cv2.putText(frame, 'OBJETO!', (right_wrist[0]-50, right_wrist[1]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)
        
        # Panel de información
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_number}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mano izquierda
        left_color = (0, 255, 0) if left_confirmed else (150, 150, 150)
        left_bg = (0, 100, 0) if left_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 55), (640, 105), left_bg, -1)
        cv2.rectangle(frame, (15, 55), (640, 105), left_color, 3)
        baseline_str = f"baseline: {self.hand_size_baseline['left']:.2f}" if self.hand_size_baseline['left'] else "calibrando..."
        size_str = f"tamaño: {left_size:.2f}" if left_size else "N/A"
        left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"} ({size_str}, {baseline_str})'
        cv2.putText(frame, left_text, (25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, left_color, 2)
        
        # Mano derecha
        right_color = (255, 0, 0) if right_confirmed else (150, 150, 150)
        right_bg = (100, 0, 0) if right_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 115), (640, 165), right_bg, -1)
        cv2.rectangle(frame, (15, 115), (640, 165), right_color, 3)
        baseline_str = f"baseline: {self.hand_size_baseline['right']:.2f}" if self.hand_size_baseline['right'] else "calibrando..."
        size_str = f"tamaño: {right_size:.2f}" if right_size else "N/A"
        right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"} ({size_str}, {baseline_str})'
        cv2.putText(frame, right_text, (25, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, right_color, 2)
        
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
            cv2.rectangle(frame, (15, 175), (20 + text_size[0], 210), (0, 0, 0), -1)
            cv2.putText(frame, summary_text, (20, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, summary_color, 4)
        
        return {
            'frame': frame,
            'left_hand': {
                'has_object': left_confirmed,
                'confidence': left_confidence,
                'size': left_size
            },
            'right_hand': {
                'has_object': right_confirmed,
                'confidence': right_confidence,
                'size': right_size
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
        print("Calibrando baseline de manos vacías (primeros 20 frames)...")
        
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
                    left_size_str = f"{result['left_hand']['size']:.2f}" if result['left_hand']['size'] else "N/A"
                    right_size_str = f"{result['right_hand']['size']:.2f}" if result['right_hand']['size'] else "N/A"
                    print(f"  Frame {frame_count}: "
                          f"Izq={'OBJETO' if result['left_hand']['has_object'] else 'VACIA'} "
                          f"(tamaño: {left_size_str}), "
                          f"Der={'OBJETO' if result['right_hand']['has_object'] else 'VACIA'} "
                          f"(tamaño: {right_size_str})")
                
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
                    
                    cv2.imshow('Body Segmentation Detector', frame_display)
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
    parser = argparse.ArgumentParser(description='Detector por segmentación de cuerpo')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--min-frames', type=int, default=3, help='Frames mínimos para confirmar')
    parser.add_argument('--threshold', type=float, default=1.3, help='Umbral de tamaño (1.3 = 30% más grande)')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorBodySeg(
        video_path=args.video_path,
        min_frames=args.min_frames,
        size_threshold=args.threshold
    )
    
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
