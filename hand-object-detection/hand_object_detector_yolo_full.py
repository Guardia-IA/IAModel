"""
Detector YOLO completo: Detecta TODOS los objetos y muestra qué hay cerca de las manos
Implementación limpia con YOLO más reciente
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorYOLOFull:
    """
    Detecta todos los objetos con YOLO y muestra qué hay cerca de cada mano
    """
    
    def __init__(self, video_path, detection_threshold=200, min_frames=3, min_confidence=0.15):
        self.video_path = video_path
        self.detection_threshold = detection_threshold
        self.min_frames = min_frames
        self.min_confidence = min_confidence
        
        # YOLO pose para muñecas
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # YOLO para objetos - usar la mejor versión disponible
        print("Cargando YOLO para objetos (todas las clases)...")
        try:
            # Intentar YOLOv9 primero (mejor para objetos pequeños)
            self.yolo_objects = YOLO('yolov9t.pt')
            print("✓ YOLOv9 cargado")
        except:
            try:
                # Si no, usar YOLOv8
                self.yolo_objects = YOLO('yolov8n.pt')
                print("✓ YOLOv8 cargado")
            except:
                raise Exception("No se pudo cargar ningún modelo YOLO")
        
        # Obtener nombres de clases
        self.class_names = self.yolo_objects.names
        
        # Historial
        self.left_hand_history = deque(maxlen=min_frames)
        self.right_hand_history = deque(maxlen=min_frames)
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print(f"✓ Detector YOLO completo inicializado ({len(self.class_names)} clases disponibles)")
    
    def calculate_distance(self, point1, point2):
        """Calcula distancia euclidiana"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def find_objects_near_wrist(self, wrist_pos, objects):
        """Encuentra objetos cerca de la muñeca con información detallada"""
        if wrist_pos is None:
            return []
        
        wx, wy = wrist_pos
        nearby_objects = []
        
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            obj_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Distancia al centro del objeto
            dist_to_center = self.calculate_distance(wrist_pos, obj_center)
            
            # Verificar si la muñeca está dentro del bbox expandido
            obj_width = x2 - x1
            obj_height = y2 - y1
            margin = max(obj_width, obj_height) * 0.5
            wrist_inside = (x1 - margin <= wx <= x2 + margin and 
                           y1 - margin <= wy <= y2 + margin)
            
            # Priorizar objetos pequeños (más probables de estar en la mano)
            is_small_object = obj['area'] < 30000  # Objetos pequeños/medianos
            is_medium_object = 30000 <= obj['area'] < 80000  # Objetos medianos
            is_very_close = dist_to_center < self.detection_threshold * 0.7  # Muy cerca
            
            # Si está cerca
            if wrist_inside or dist_to_center < self.detection_threshold:
                # Filtrar objetos muy grandes a menos que estén muy cerca
                if obj['area'] > 80000 and dist_to_center > self.detection_threshold * 0.5:
                    continue  # Saltar objetos muy grandes lejos
                
                nearby_objects.append({
                    'bbox': obj['bbox'],
                    'class': obj['class'],
                    'class_name': obj['class_name'],
                    'confidence': obj['confidence'],
                    'distance': dist_to_center,
                    'wrist_inside': wrist_inside,
                    'area': obj['area'],
                    'score': obj['confidence'] * (1.0 if is_small_object else (0.8 if is_medium_object else 0.5)) * (1.0 if is_very_close else 0.7)
                })
        
        # Ordenar por score (confianza * tamaño * distancia)
        nearby_objects.sort(key=lambda x: x['score'], reverse=True)
        
        return nearby_objects
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        
        # Detectar personas con YOLO pose
        pose_results = self.yolo_pose(frame, conf=0.3, verbose=False)
        
        # Detectar TODOS los objetos con YOLO (sin filtrar clases)
        object_results = self.yolo_objects(frame, conf=self.min_confidence, verbose=False)
        
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
                                    
                                    # Dibujar región de búsqueda
                                    cv2.circle(frame, wrist_pos, self.detection_threshold, color, 2)
        
        # Extraer TODOS los objetos detectados
        all_objects = []
        if object_results[0].boxes is not None:
            for box in object_results[0].boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                obj_width = x2 - x1
                obj_height = y2 - y1
                obj_area = obj_width * obj_height
                
                # Filtrar objetos: excluir personas y objetos muy grandes (probablemente fondo)
                # Objetos muy grandes probablemente son de fondo (neveras, estanterías, etc.)
                max_area = 50000  # Área máxima para objetos en manos
                
                if cls != 0 and 100 < obj_area < max_area:  # Excluir personas y objetos muy grandes
                    class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                    
                    all_objects.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': cls,
                        'class_name': class_name,
                        'confidence': conf,
                        'area': obj_area
                    })
                    
                    # Dibujar objeto con color según clase
                    color_obj = (255, 255, 0)  # Amarillo por defecto
                    if cls in [24, 26, 28]:  # backpack, handbag, suitcase
                        color_obj = (0, 255, 255)  # Cyan
                    elif cls == 39:  # bottle
                        color_obj = (255, 0, 255)  # Magenta
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_obj, 2)
                    cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_obj, 1)
        
        # Encontrar objetos cerca de cada muñeca
        left_nearby = self.find_objects_near_wrist(left_wrist, all_objects)
        right_nearby = self.find_objects_near_wrist(right_wrist, all_objects)
        
        # Determinar si hay objeto en cada mano
        left_has_object = len(left_nearby) > 0
        right_has_object = len(right_nearby) > 0
        
        # Visualizar objetos cerca de manos
        if left_wrist and left_nearby:
            for obj in left_nearby[:3]:  # Mostrar los 3 más cercanos
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 0), 4)
                cv2.line(frame, left_wrist, ((x1+x2)//2, (y1+y2)//2), (0, 255, 0), 3)
                cv2.putText(frame, f"{obj['class_name']} {obj['confidence']:.2f}", 
                           (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if right_wrist and right_nearby:
            for obj in right_nearby[:3]:  # Mostrar los 3 más cercanos
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (255, 0, 0), 4)
                cv2.line(frame, right_wrist, ((x1+x2)//2, (y1+y2)//2), (255, 0, 0), 3)
                cv2.putText(frame, f"{obj['class_name']} {obj['confidence']:.2f}", 
                           (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Actualizar historial
        self.left_hand_history.append(left_has_object)
        self.right_hand_history.append(right_has_object)
        
        # Confirmar solo si hay detección consistente
        left_confirmed = sum(self.left_hand_history) >= self.min_frames
        right_confirmed = sum(self.right_hand_history) >= self.min_frames
        
        # Visualizar confirmación
        if left_wrist and left_confirmed:
            cv2.putText(frame, 'OBJETO!', (left_wrist[0]-50, left_wrist[1]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
        
        if right_wrist and right_confirmed:
            cv2.putText(frame, 'OBJETO!', (right_wrist[0]-50, right_wrist[1]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)
        
        # Panel de información detallado
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (700, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_number}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mano izquierda
        left_color = (0, 255, 0) if left_confirmed else (150, 150, 150)
        left_bg = (0, 100, 0) if left_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 55), (680, 105), left_bg, -1)
        cv2.rectangle(frame, (15, 55), (680, 105), left_color, 3)
        
        if left_nearby:
            obj_info = ", ".join([f"{obj['class_name']}({obj['confidence']:.2f})" for obj in left_nearby[:2]])
            left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"} - {obj_info}'
        else:
            left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"} - Sin objetos detectados'
        cv2.putText(frame, left_text, (25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, left_color, 2)
        
        # Mano derecha
        right_color = (255, 0, 0) if right_confirmed else (150, 150, 150)
        right_bg = (100, 0, 0) if right_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 115), (680, 165), right_bg, -1)
        cv2.rectangle(frame, (15, 115), (680, 165), right_color, 3)
        
        if right_nearby:
            obj_info = ", ".join([f"{obj['class_name']}({obj['confidence']:.2f})" for obj in right_nearby[:2]])
            right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"} - {obj_info}'
        else:
            right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"} - Sin objetos detectados'
        cv2.putText(frame, right_text, (25, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, right_color, 2)
        
        # Información de objetos totales
        cv2.putText(frame, f'Objetos totales detectados: {len(all_objects)}', (20, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Mostrar clases únicas detectadas
        unique_classes = set([obj['class_name'] for obj in all_objects])
        if unique_classes:
            classes_str = ", ".join(list(unique_classes)[:5])
            cv2.putText(frame, f'Clases: {classes_str}', (20, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
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
            cv2.rectangle(frame, (15, 240), (20 + text_size[0], 275), (0, 0, 0), -1)
            cv2.putText(frame, summary_text, (20, 265),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, summary_color, 4)
        
        return {
            'frame': frame,
            'left_hand': {
                'has_object': left_confirmed,
                'objects': left_nearby
            },
            'right_hand': {
                'has_object': right_confirmed,
                'objects': right_nearby
            },
            'all_objects': all_objects
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
        print(f"Detectando todas las clases de objetos...\n")
        
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
        all_detected_classes = set()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                result = self.process_frame(frame, frame_count)
                
                # Registrar clases detectadas
                for obj in result['all_objects']:
                    all_detected_classes.add(obj['class_name'])
                
                if result['left_hand']['has_object'] or result['right_hand']['has_object']:
                    frames_with_objects.append({
                        'frame': frame_count,
                        'left': result['left_hand']['has_object'],
                        'right': result['right_hand']['has_object'],
                        'left_objects': [obj['class_name'] for obj in result['left_hand']['objects']],
                        'right_objects': [obj['class_name'] for obj in result['right_hand']['objects']]
                    })
                    
                    left_objs = [obj['class_name'] for obj in result['left_hand']['objects'][:2]]
                    right_objs = [obj['class_name'] for obj in result['right_hand']['objects'][:2]]
                    left_objs_str = ", ".join(left_objs) if left_objs else "ninguno"
                    right_objs_str = ", ".join(right_objs) if right_objs else "ninguno"
                    
                    print(f"  Frame {frame_count}: "
                          f"Izq={'OBJETO' if result['left_hand']['has_object'] else 'VACIA'} "
                          f"({len(result['left_hand']['objects'])} objetos: {left_objs_str}), "
                          f"Der={'OBJETO' if result['right_hand']['has_object'] else 'VACIA'} "
                          f"({len(result['right_hand']['objects'])} objetos: {right_objs_str})")
                
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
                    
                    cv2.imshow('YOLO Full Detector', frame_display)
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
            print(f"  - Clases de objetos detectadas: {sorted(all_detected_classes)}")
            if frames_with_objects:
                print(f"  - Primeros frames con objetos: {[f['frame'] for f in frames_with_objects[:10]]}")


def main():
    parser = argparse.ArgumentParser(description='Detector YOLO completo - Todas las clases')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--threshold', type=int, default=200, help='Distancia máxima muñeca-objeto')
    parser.add_argument('--min-frames', type=int, default=3, help='Frames mínimos para confirmar')
    parser.add_argument('--confidence', type=float, default=0.15, help='Confianza mínima para objetos')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorYOLOFull(
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
