"""
Detector con YOLOv9: Detección binaria de objetos en cada mano
Usa YOLOv9 (PGI) para detectar objetos pequeños y YOLO pose para muñecas
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorYOLOv9:
    """
    Detecta si hay objeto en cada mano (binario)
    """
    
    def __init__(self, video_path, detection_threshold=300, min_confidence=0.05):
        self.video_path = video_path
        self.detection_threshold = detection_threshold  # Distancia máxima muñeca-objeto
        self.min_confidence = min_confidence
        
        # YOLO pose para muñecas
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # YOLOv9 para objetos pequeños (mejor detección gracias a PGI)
        print("Cargando YOLOv9...")
        try:
            # Intentar cargar YOLOv9 (puede ser yolov9t.pt, yolov9s.pt, etc.)
            # Si no está disponible, usar YOLOv8 pero con mejor configuración
            self.yolo_objects = YOLO('yolov9t.pt')  # Tiny version, más rápido
            print("✓ YOLOv9 cargado")
        except:
            print("⚠ YOLOv9 no disponible, usando YOLOv8 con configuración optimizada")
            self.yolo_objects = YOLO('yolov8n.pt')
        
        # Historial para estabilizar detecciones
        self.left_hand_history = deque(maxlen=5)
        self.right_hand_history = deque(maxlen=5)
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector YOLOv9 inicializado")
    
    def calculate_distance(self, point1, point2):
        """Calcula distancia euclidiana"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_object_near_wrist(self, wrist_pos, obj_bbox):
        """Verifica si un objeto está cerca de la muñeca"""
        if wrist_pos is None:
            return False
        
        wx, wy = wrist_pos
        x1, y1, x2, y2 = obj_bbox
        
        # Centro del objeto
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        # Distancia al centro
        dist_to_center = self.calculate_distance(wrist_pos, (obj_center_x, obj_center_y))
        
        # Verificar si la muñeca está dentro del bbox expandido
        margin = max((x2 - x1), (y2 - y1)) * 0.8  # Margen más amplio para bolsas
        wrist_inside = (x1 - margin <= wx <= x2 + margin and 
                       y1 - margin <= wy <= y2 + margin)
        
        # O muy cerca del centro (umbral más permisivo)
        # Para objetos grandes (bolsas), usar umbral más amplio
        obj_size = max((x2 - x1), (y2 - y1))
        threshold = self.detection_threshold if obj_size < 200 else self.detection_threshold * 1.5
        
        return wrist_inside or dist_to_center < threshold
    
    def detect_objects_in_hand(self, wrist_pos, objects, hand_side="left"):
        """
        Detecta si hay objeto en una mano específica
        Retorna: (has_object, closest_object, confidence)
        """
        if wrist_pos is None or not objects:
            return False, None, 0.0
        
        closest_obj = None
        min_distance = float('inf')
        
        for obj in objects:
            if self.is_object_near_wrist(wrist_pos, obj['bbox']):
                # Calcular distancia
                obj_center = ((obj['bbox'][0] + obj['bbox'][2]) / 2,
                             (obj['bbox'][1] + obj['bbox'][3]) / 2)
                distance = self.calculate_distance(wrist_pos, obj_center)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_obj = obj
        
        # Si hay objeto cerca, calcular confianza
        if closest_obj is not None:
            # Confianza basada en distancia y confianza del objeto
            distance_score = max(0, 1.0 - (min_distance / self.detection_threshold))
            confidence = closest_obj['confidence'] * 0.7 + distance_score * 0.3
            return True, closest_obj, confidence
        
        return False, None, 0.0
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        
        # Detectar personas con YOLO pose
        pose_results = self.yolo_pose(frame, conf=0.3, verbose=False)
        
        # Detectar objetos con YOLOv9 (o YOLOv8)
        object_results = self.yolo_objects(frame, conf=self.min_confidence, verbose=False)
        
        # Extraer muñecas
        left_wrist = None   # Índice 9
        right_wrist = None  # Índice 10
        
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
                            # Muñeca izquierda (9) y derecha (10)
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
        
        # Extraer objetos detectados
        detected_objects = []
        if object_results[0].boxes is not None:
            for box in object_results[0].boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                obj_width = x2 - x1
                obj_height = y2 - y1
                obj_area = obj_width * obj_height
                
                # Clases de bolsas: 24=backpack, 26=handbag, 28=suitcase
                bag_classes = [24, 26, 28]
                is_bag = cls in bag_classes
                
                # Aceptar:
                # 1. Objetos no-persona con confianza suficiente
                # 2. Bolsas con confianza muy baja (0.05)
                # 3. Objetos clase 0 (persona) si están MUY cerca de muñecas (puede ser bolsa detectada como persona)
                should_include = False
                
                if cls != 0 and conf > self.min_confidence:
                    should_include = True
                elif is_bag and conf > 0.05:  # Bolsas con confianza muy baja
                    should_include = True
                elif cls == 0 and conf > 0.3:  # Objetos "persona" con alta confianza cerca de muñecas
                    # Verificar si está cerca de alguna muñeca (más permisivo para bolsas)
                    obj_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    for wrist_pos in [left_wrist, right_wrist]:
                        if wrist_pos is not None:
                            dist = np.sqrt((obj_center[0] - wrist_pos[0])**2 + 
                                          (obj_center[1] - wrist_pos[1])**2)
                            # Para objetos grandes (posibles bolsas), umbral más amplio
                            threshold_dist = 300 if obj_area > 50000 else 200
                            if dist < threshold_dist:  # Muy cerca de muñeca
                                should_include = True
                                break
                
                if should_include:
                    # Rango amplio de tamaños (incluye bolsas grandes)
                    if 100 < obj_area < 500000:  # Acepta objetos más grandes (bolsas)
                        detected_objects.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls,
                            'area': obj_area,
                            'is_bag': is_bag
                        })
                        
                        # Dibujar objeto (más visible si es bolsa)
                        color = (0, 255, 255) if is_bag else (255, 255, 0)
                        thickness = 3 if is_bag else 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        if is_bag:
                            cv2.putText(frame, f'BAG {conf:.2f}', (x1, y1-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Detectar objetos en cada mano
        left_has_object, left_obj, left_conf = self.detect_objects_in_hand(
            left_wrist, detected_objects, "left"
        )
        right_has_object, right_obj, right_conf = self.detect_objects_in_hand(
            right_wrist, detected_objects, "right"
        )
        
        # Actualizar historial para estabilizar
        self.left_hand_history.append(left_has_object)
        self.right_hand_history.append(right_has_object)
        
        # Confirmar solo si hay detección consistente (más permisivo)
        left_confirmed = sum(self.left_hand_history) >= 2
        right_confirmed = sum(self.right_hand_history) >= 2
        
        # Visualizar objetos en manos
        if left_has_object and left_obj:
            x1, y1, x2, y2 = left_obj['bbox']
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 0), 5)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)
            if left_wrist:
                cv2.line(frame, left_wrist, ((x1+x2)//2, (y1+y2)//2), (0, 255, 0), 3)
        
        if right_has_object and right_obj:
            x1, y1, x2, y2 = right_obj['bbox']
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255, 0, 0), 5)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 0), 3)
            if right_wrist:
                cv2.line(frame, right_wrist, ((x1+x2)//2, (y1+y2)//2), (255, 0, 0), 3)
        
        # Panel de información mejorado
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Información de frame
        cv2.putText(frame, f'Frame: {frame_number}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mano izquierda - más visible
        left_color = (0, 255, 0) if left_confirmed else (150, 150, 150)
        left_bg_color = (0, 100, 0) if left_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 55), (580, 105), left_bg_color, -1)
        cv2.rectangle(frame, (15, 55), (580, 105), left_color, 3)
        left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"}'
        if left_has_object and left_obj:
            left_text += f' (conf: {left_conf:.2f})'
        cv2.putText(frame, left_text, (25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, left_color, 3)
        
        # Mano derecha - más visible
        right_color = (255, 0, 0) if right_confirmed else (150, 150, 150)
        right_bg_color = (100, 0, 0) if right_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 115), (580, 165), right_bg_color, -1)
        cv2.rectangle(frame, (15, 115), (580, 165), right_color, 3)
        right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"}'
        if right_has_object and right_obj:
            right_text += f' (conf: {right_conf:.2f})'
        cv2.putText(frame, right_text, (25, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, right_color, 3)
        
        # Objetos detectados
        cv2.putText(frame, f'Objetos YOLOv9 detectados: {len(detected_objects)}', (20, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Resumen binario grande y visible
        if left_confirmed or right_confirmed:
            summary_text = "OBJETO DETECTADO"
            summary_color = (0, 255, 255)
            if left_confirmed and right_confirmed:
                summary_text = "OBJETOS EN AMBAS MANOS"
                summary_color = (0, 255, 255)
            elif left_confirmed:
                summary_text = "✓ OBJETO EN MANO IZQUIERDA"
                summary_color = (0, 255, 0)
            elif right_confirmed:
                summary_text = "✓ OBJETO EN MANO DERECHA"
                summary_color = (255, 0, 0)
            
            # Fondo para el texto
            text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
            cv2.rectangle(frame, (15, 200), (20 + text_size[0], 245), (0, 0, 0), -1)
            cv2.putText(frame, summary_text, (20, 235),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, summary_color, 4)
        
        return {
            'frame': frame,
            'left_hand': {
                'has_object': left_confirmed,
                'object': left_obj,
                'confidence': left_conf
            },
            'right_hand': {
                'has_object': right_confirmed,
                'object': right_obj,
                'confidence': right_conf
            },
            'total_objects': len(detected_objects)
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
        print(f"Analizando frames 100-150 para detección de objetos en manos...\n")
        
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
                
                # Registrar frames con objetos
                if result['left_hand']['has_object'] or result['right_hand']['has_object']:
                    frames_with_objects.append({
                        'frame': frame_count,
                        'left': result['left_hand']['has_object'],
                        'right': result['right_hand']['has_object']
                    })
                    if 100 <= frame_count <= 150:
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
                    
                    cv2.imshow('YOLOv9 - Objetos en Manos', frame_display)
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
                print(f"  - Frames con objetos (100-150): "
                      f"{[f['frame'] for f in frames_with_objects if 100 <= f['frame'] <= 150]}")


def main():
    parser = argparse.ArgumentParser(description='Detección binaria de objetos en manos con YOLOv9')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--threshold', type=int, default=80, help='Distancia máxima muñeca-objeto (pixels)')
    parser.add_argument('--confidence', type=float, default=0.2, help='Confianza mínima para objetos')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorYOLOv9(
        video_path=args.video_path,
        detection_threshold=args.threshold,
        min_confidence=args.confidence
    )
    
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
