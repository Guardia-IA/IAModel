"""
Detector usando YOLO con segmentación: Detecta objetos segmentados cerca de muñecas
Enfoque: YOLO pose para muñecas + YOLO-Seg para objetos + validación espacial
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorYOLOSeg:
    """
    Detecta objetos usando YOLO con segmentación
    """
    
    def __init__(self, video_path, detection_threshold=150, min_frames=3, min_confidence=0.25):
        self.video_path = video_path
        self.detection_threshold = detection_threshold
        self.min_frames = min_frames
        self.min_confidence = min_confidence
        
        # YOLO pose para muñecas
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # YOLO con segmentación para objetos
        print("Cargando YOLO con segmentación...")
        try:
            self.yolo_seg = YOLO('yolov8n-seg.pt')  # Modelo con segmentación
            print("✓ YOLO-Seg cargado")
        except:
            print("⚠ YOLO-Seg no disponible, usando YOLO normal")
            self.yolo_seg = YOLO('yolov8n.pt')
        
        # Historial
        self.left_hand_history = deque(maxlen=min_frames)
        self.right_hand_history = deque(maxlen=min_frames)
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector YOLO-Seg inicializado")
    
    def calculate_iou_mask(self, mask1, mask2):
        """Calcula IoU entre dos máscaras"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
    
    def is_object_near_wrist(self, wrist_pos, obj_mask, obj_bbox):
        """Verifica si un objeto segmentado está cerca de la muñeca"""
        if wrist_pos is None:
            return False, 0.0
        
        wx, wy = wrist_pos
        x1, y1, x2, y2 = obj_bbox
        
        # Centro del objeto
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        # Distancia al centro
        dist_to_center = np.sqrt((wx - obj_center_x)**2 + (wy - obj_center_y)**2)
        
        # Verificar si la muñeca está dentro del bbox
        wrist_inside_bbox = (x1 <= wx <= x2 and y1 <= wy <= y2)
        
        # Verificar si la muñeca está dentro de la máscara
        wrist_in_mask = False
        if wrist_inside_bbox:
            # Coordenadas relativas al bbox
            rel_x = int(wx - x1)
            rel_y = int(wy - y1)
            if 0 <= rel_y < obj_mask.shape[0] and 0 <= rel_x < obj_mask.shape[1]:
                wrist_in_mask = obj_mask[rel_y, rel_x] > 0
        
        # Calcular confianza
        if wrist_in_mask:
            confidence = 1.0
        elif wrist_inside_bbox:
            confidence = 0.8
        elif dist_to_center < self.detection_threshold:
            confidence = max(0.3, 1.0 - (dist_to_center / self.detection_threshold))
        else:
            confidence = 0.0
        
        has_object = confidence > 0.3
        
        return has_object, confidence
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        
        # Detectar personas con YOLO pose
        pose_results = self.yolo_pose(frame, conf=0.3, verbose=False)
        
        # Detectar objetos con YOLO-Seg
        seg_results = self.yolo_seg(frame, conf=self.min_confidence, verbose=False)
        
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
        
        # Procesar objetos segmentados
        left_has_object = False
        right_has_object = False
        left_confidence = 0.0
        right_confidence = 0.0
        left_best_obj = None
        right_best_obj = None
        
        if seg_results[0].masks is not None and seg_results[0].boxes is not None:
            for i, box in enumerate(seg_results[0].boxes):
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Filtrar personas
                if cls == 0:
                    continue
                
                # Obtener máscara
                if i < len(seg_results[0].masks.data):
                    mask = seg_results[0].masks.data[i].cpu().numpy()
                    # Redimensionar máscara al tamaño del frame
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                else:
                    # Si no hay máscara, usar bbox
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    mask_binary = np.zeros((h, w), dtype=np.uint8)
                    mask_binary[y1:y2, x1:x2] = 255
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                obj_bbox = [x1, y1, x2, y2]
                
                # Recortar máscara al bbox para eficiencia
                mask_cropped = mask_binary[y1:y2, x1:x2] if mask_binary is not None else None
                
                # Verificar proximidad a muñecas (más permisivo)
                # Aumentar umbral de distancia para detectar mejor
                original_threshold = self.detection_threshold
                self.detection_threshold = original_threshold * 1.5  # 50% más permisivo
                
                if left_wrist:
                    has_obj, conf = self.is_object_near_wrist(
                        left_wrist, mask_cropped, obj_bbox
                    )
                    if has_obj and conf > left_confidence:
                        left_has_object = True
                        left_confidence = conf
                        left_best_obj = {
                            'bbox': obj_bbox,
                            'mask': mask_cropped,
                            'class': cls,
                            'confidence': conf
                        }
                
                if right_wrist:
                    has_obj, conf = self.is_object_near_wrist(
                        right_wrist, mask_cropped, obj_bbox
                    )
                    if has_obj and conf > right_confidence:
                        right_has_object = True
                        right_confidence = conf
                        right_best_obj = {
                            'bbox': obj_bbox,
                            'mask': mask_cropped,
                            'class': cls,
                            'confidence': conf
                        }
                
                # Restaurar umbral
                self.detection_threshold = original_threshold
                
                # Dibujar objeto (sutil)
                if mask_binary is not None:
                    # Crear overlay de color
                    overlay = frame.copy()
                    color = (0, 255, 255) if cls in [24, 26, 28] else (255, 255, 0)  # Bolsas en cyan
                    overlay[mask_binary > 0] = color
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
        # Actualizar historial
        self.left_hand_history.append(left_has_object)
        self.right_hand_history.append(right_has_object)
        
        # Confirmar solo si hay detección consistente (más permisivo al inicio)
        # Si es al principio del video, ser más estricto para evitar falsos positivos
        min_frames_required = self.min_frames if frame_number > 20 else max(4, self.min_frames + 1)
        left_confirmed = sum(self.left_hand_history) >= min_frames_required
        right_confirmed = sum(self.right_hand_history) >= min_frames_required
        
        # Visualizar objetos confirmados
        if left_confirmed and left_best_obj:
            x1, y1, x2, y2 = left_best_obj['bbox']
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 0), 5)
            if left_wrist:
                cv2.line(frame, left_wrist, ((x1+x2)//2, (y1+y2)//2), (0, 255, 0), 3)
        
        if right_confirmed and right_best_obj:
            x1, y1, x2, y2 = right_best_obj['bbox']
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255, 0, 0), 5)
            if right_wrist:
                cv2.line(frame, right_wrist, ((x1+x2)//2, (y1+y2)//2), (255, 0, 0), 3)
        
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
                    
                    cv2.imshow('YOLO-Seg Detector', frame_display)
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
    parser = argparse.ArgumentParser(description='Detector YOLO-Seg: Segmentación de objetos')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--threshold', type=int, default=150, help='Distancia máxima muñeca-objeto')
    parser.add_argument('--min-frames', type=int, default=3, help='Frames mínimos para confirmar')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confianza mínima para objetos')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorYOLOSeg(
        video_path=args.video_path,
        detection_threshold=args.threshold,
        min_frames=args.min_frames,
        min_confidence=args.confidence
    )
    
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
