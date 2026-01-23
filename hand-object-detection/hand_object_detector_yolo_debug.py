"""
Detector YOLO Debug: Muestra TODO lo que YOLO detecta sin filtrar
Para ver qué detecta realmente YOLO cerca de las manos
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse


class HandObjectDetectorYOLODebug:
    """
    Muestra todo lo que YOLO detecta para análisis
    """
    
    def __init__(self, video_path, min_confidence=0.1):
        self.video_path = video_path
        self.min_confidence = min_confidence
        
        # YOLO pose para muñecas
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # YOLO para objetos - mejor versión
        print("Cargando YOLO para objetos...")
        try:
            self.yolo_objects = YOLO('yolov9t.pt')
            print("✓ YOLOv9 cargado")
        except:
            self.yolo_objects = YOLO('yolov8n.pt')
            print("✓ YOLOv8 cargado")
        
        # Obtener nombres de clases
        self.class_names = self.yolo_objects.names
        print(f"✓ {len(self.class_names)} clases disponibles")
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
    
    def calculate_distance(self, point1, point2):
        """Calcula distancia euclidiana"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        
        # Detectar personas con YOLO pose
        pose_results = self.yolo_pose(frame, conf=0.3, verbose=False)
        
        # Detectar TODOS los objetos con YOLO (confianza muy baja para ver todo)
        object_results = self.yolo_objects(frame, conf=self.min_confidence, verbose=False)
        
        # Extraer muñecas
        left_wrist = None
        right_wrist = None
        
        if pose_results[0].boxes is not None:
            for i, box in enumerate(pose_results[0].boxes):
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:  # Persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
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
                                    
                                    color = (0, 255, 0) if wrist_idx == 9 else (255, 0, 0)
                                    cv2.circle(frame, wrist_pos, 20, color, 5)
                                    cv2.putText(frame, f'{"L" if wrist_idx == 9 else "R"}', 
                                               (wrist_pos[0]+25, wrist_pos[1]), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                                    
                                    # Dibujar región de búsqueda
                                    cv2.circle(frame, wrist_pos, 300, color, 2)
        
        # Procesar TODOS los objetos detectados
        all_objects = []
        objects_near_left = []
        objects_near_right = []
        
        if object_results[0].boxes is not None:
            for box in object_results[0].boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                obj_width = x2 - x1
                obj_height = y2 - y1
                obj_area = obj_width * obj_height
                obj_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                
                obj_info = {
                    'bbox': [x1, y1, x2, y2],
                    'class': cls,
                    'class_name': class_name,
                    'confidence': conf,
                    'area': obj_area,
                    'center': obj_center
                }
                
                all_objects.append(obj_info)
                
                # Verificar proximidad a muñecas (excluir personas)
                if cls != 0:  # No incluir personas
                    if left_wrist:
                        dist = self.calculate_distance(left_wrist, obj_center)
                        if dist < 300:  # Dentro de 300 píxeles
                            objects_near_left.append({
                                **obj_info,
                                'distance': dist
                            })
                    
                    if right_wrist:
                        dist = self.calculate_distance(right_wrist, obj_center)
                        if dist < 300:  # Dentro de 300 píxeles
                            objects_near_right.append({
                                **obj_info,
                                'distance': dist
                            })
                
                # Dibujar objeto con color según clase
                color_obj = (255, 255, 0)  # Amarillo por defecto
                if cls in [24, 26, 28]:  # backpack, handbag, suitcase
                    color_obj = (0, 255, 255)  # Cyan
                elif cls == 39:  # bottle
                    color_obj = (255, 0, 255)  # Magenta
                elif cls == 0:  # person
                    color_obj = (128, 128, 128)  # Gris
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_obj, 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_obj, 2)
        
        # Resaltar objetos cerca de manos
        if left_wrist and objects_near_left:
            for obj in objects_near_left:
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 0), 4)
                cv2.line(frame, left_wrist, tuple(map(int, obj['center'])), (0, 255, 0), 3)
                cv2.putText(frame, f"{obj['class_name']} {obj['confidence']:.2f} (dist: {obj['distance']:.0f})", 
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if right_wrist and objects_near_right:
            for obj in objects_near_right:
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (255, 0, 0), 4)
                cv2.line(frame, right_wrist, tuple(map(int, obj['center'])), (255, 0, 0), 3)
                cv2.putText(frame, f"{obj['class_name']} {obj['confidence']:.2f} (dist: {obj['distance']:.0f})", 
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Panel de información completo
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (800, 350), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_number}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Mano izquierda
        cv2.putText(frame, f'Mano IZQUIERDA:', (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if objects_near_left:
            y_offset = 100
            for i, obj in enumerate(objects_near_left[:5]):  # Mostrar hasta 5
                text = f"  {i+1}. {obj['class_name']} (conf: {obj['confidence']:.2f}, dist: {obj['distance']:.0f}px, area: {obj['area']})"
                cv2.putText(frame, text, (25, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                y_offset += 30
        else:
            cv2.putText(frame, "  Sin objetos detectados cerca", (25, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 2)
        
        # Mano derecha
        right_y = 250 if objects_near_left else 100
        cv2.putText(frame, f'Mano DERECHA:', (20, right_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if objects_near_right:
            y_offset = right_y + 25
            for i, obj in enumerate(objects_near_right[:5]):  # Mostrar hasta 5
                text = f"  {i+1}. {obj['class_name']} (conf: {obj['confidence']:.2f}, dist: {obj['distance']:.0f}px, area: {obj['area']})"
                cv2.putText(frame, text, (25, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
                y_offset += 30
        else:
            cv2.putText(frame, "  Sin objetos detectados cerca", (25, right_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 2)
        
        # Resumen de objetos totales
        cv2.putText(frame, f'Objetos totales detectados: {len(all_objects)}', (20, 320),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return {
            'frame': frame,
            'all_objects': all_objects,
            'objects_near_left': objects_near_left,
            'objects_near_right': objects_near_right
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
        print("Mostrando TODO lo que YOLO detecta...\n")
        
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
        all_detected_classes = set()
        frames_with_objects_near_hands = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                result = self.process_frame(frame, frame_count)
                
                # Registrar clases
                for obj in result['all_objects']:
                    all_detected_classes.add(obj['class_name'])
                
                # Imprimir información de objetos cerca de manos
                if result['objects_near_left'] or result['objects_near_right']:
                    frames_with_objects_near_hands.append(frame_count)
                    print(f"\nFrame {frame_count}:")
                    if result['objects_near_left']:
                        print(f"  Mano IZQUIERDA ({len(result['objects_near_left'])} objetos):")
                        for obj in result['objects_near_left']:
                            print(f"    - {obj['class_name']} (conf: {obj['confidence']:.3f}, dist: {obj['distance']:.0f}px, area: {obj['area']})")
                    if result['objects_near_right']:
                        print(f"  Mano DERECHA ({len(result['objects_near_right'])} objetos):")
                        for obj in result['objects_near_right']:
                            print(f"    - {obj['class_name']} (conf: {obj['confidence']:.3f}, dist: {obj['distance']:.0f}px, area: {obj['area']})")
                
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
                    
                    cv2.imshow('YOLO Debug - Todo lo detectado', frame_display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_video:
                cv2.destroyAllWindows()
            
            print(f"\n{'='*60}")
            print(f"Resumen completo:")
            print(f"  - Frames totales: {frame_count}")
            print(f"  - Frames con objetos cerca de manos: {len(frames_with_objects_near_hands)}")
            print(f"  - Clases de objetos detectadas: {sorted(all_detected_classes)}")
            if frames_with_objects_near_hands:
                print(f"  - Frames con objetos cerca: {frames_with_objects_near_hands[:20]}")


def main():
    parser = argparse.ArgumentParser(description='YOLO Debug: Muestra todo lo detectado')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--confidence', type=float, default=0.1, help='Confianza mínima para objetos')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorYOLODebug(
        video_path=args.video_path,
        min_confidence=args.confidence
    )
    
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
