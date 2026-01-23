"""
Detección V6: Enfoque estricto y robusto
- YOLO Pose para muñecas (más confiable que MediaPipe a distancia)
- YOLO para objetos
- Análisis temporal estricto para minimizar falsos positivos
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorV6:
    """
    Detector estricto: solo confirma cuando hay evidencia clara
    """
    
    def __init__(self, video_path, interaction_threshold=100, min_frames_interaction=8):
        self.video_path = video_path
        self.interaction_threshold = interaction_threshold
        self.min_frames_interaction = min_frames_interaction
        
        # YOLO pose para personas y muñecas
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # YOLO para objetos
        print("Cargando YOLO objetos...")
        self.yolo_objects = YOLO('yolov8n.pt')
        
        # Historial estricto
        self.wrist_positions = deque(maxlen=30)
        self.object_detections = deque(maxlen=30)
        self.interaction_candidates = deque(maxlen=min_frames_interaction * 2)
        self.confirmed_interactions = []
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector V6 inicializado (estricto)")
    
    def calculate_iou(self, box1, box2):
        """Calcula Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def is_wrist_near_object(self, wrist_pos, obj_bbox):
        """Verifica si la muñeca está cerca del objeto"""
        if wrist_pos is None:
            return False
        
        wx, wy = wrist_pos
        x1, y1, x2, y2 = obj_bbox
        
        # Centro del objeto
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        # Distancia al centro
        dist_to_center = np.sqrt((wx - obj_center_x)**2 + (wy - obj_center_y)**2)
        
        # Tamaño del objeto
        obj_width = x2 - x1
        obj_height = y2 - y1
        obj_size = max(obj_width, obj_height)
        
        # La muñeca debe estar dentro o muy cerca del objeto
        # (dentro del bbox expandido)
        margin = obj_size * 0.3
        if (x1 - margin <= wx <= x2 + margin and 
            y1 - margin <= wy <= y2 + margin):
            return True
        
        # O muy cerca del centro
        if dist_to_center < self.interaction_threshold:
            return True
        
        return False
    
    def detect_interaction_temporal(self, current_wrists, current_objects):
        """
        Detecta interacciones: simplificado pero estricto
        """
        if not current_wrists or not current_objects:
            return []
        
        interactions = []
        
        # Para cada muñeca
        for wrist_pos in current_wrists:
            # Buscar objetos cerca
            for obj in current_objects:
                if self.is_wrist_near_object(wrist_pos, obj['bbox']):
                    # Verificar consistencia temporal simple
                    consistent_frames = 1  # Este frame cuenta
                    
                    # Revisar últimos 5 frames
                    for i in range(1, min(6, len(self.wrist_positions), len(self.object_detections))):
                        if i < len(self.wrist_positions) and i < len(self.object_detections):
                            prev_wrists = self.wrist_positions[-i]
                            prev_objects = self.object_detections[-i]
                            
                            # Verificar si había muñeca y objeto cerca
                            found_match = False
                            for prev_wrist in prev_wrists:
                                for prev_obj in prev_objects:
                                    if self.is_wrist_near_object(prev_wrist, prev_obj['bbox']):
                                        # Verificar que es el mismo objeto (IOU)
                                        iou = self.calculate_iou(obj['bbox'], prev_obj['bbox'])
                                        if iou > 0.2:  # Mismo objeto
                                            consistent_frames += 1
                                            found_match = True
                                            break
                                if found_match:
                                    break
                    
                    # Solo confirmar si hay consistencia en varios frames
                    if consistent_frames >= 3:
                        interactions.append({
                            'bbox': obj['bbox'],
                            'wrist': wrist_pos,
                            'confidence': min(1.0, consistent_frames / 5.0),
                            'consistent_frames': consistent_frames
                        })
        
        return interactions
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        
        # Detectar personas con YOLO pose
        pose_results = self.yolo_pose(frame, conf=0.4, verbose=False)
        
        # Detectar objetos con YOLO
        object_results = self.yolo_objects(frame, conf=0.25, verbose=False)
        
        wrist_positions = []
        object_detections = []
        interactions = []
        
        # Procesar personas y muñecas
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
                        
                        # Obtener muñecas (índices 9 y 10)
                        if len(keypoints) > 10:
                            for wrist_idx in [9, 10]:
                                if (keypoints[wrist_idx][0] > 0 and keypoints[wrist_idx][1] > 0 and
                                    (confidences is None or confidences[wrist_idx] > 0.3)):
                                    wrist_pos = tuple(map(int, keypoints[wrist_idx]))
                                    wrist_positions.append(wrist_pos)
                                    
                                    # Dibujar muñeca
                                    cv2.circle(frame, wrist_pos, 12, (0, 255, 0), 3)
        
        # Procesar objetos
        if object_results[0].boxes is not None:
            for box in object_results[0].boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Filtrar solo objetos pequeños/medianos (no personas, coches, etc.)
                # Aceptamos cualquier objeto con confianza suficiente
                if cls != 0 and conf > 0.2:  # No personas, confianza mínima más baja
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Filtrar objetos muy grandes (probablemente falsos positivos)
                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    obj_area = obj_width * obj_height
                    
                    # Solo objetos pequeños/medianos (más permisivo)
                    if 300 < obj_area < 100000:
                        object_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls,
                            'area': obj_area
                        })
                        
                        # Dibujar objeto (sutil)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
        # Detectar interacciones usando análisis temporal
        if wrist_positions and object_detections:
            interactions = self.detect_interaction_temporal(wrist_positions, object_detections)
        
        # Actualizar historiales
        self.wrist_positions.append(wrist_positions)
        self.object_detections.append(object_detections)
        
        # Confirmar interacciones solo si son consistentes
        confirmed = False
        if interactions:
            # Verificar si esta interacción ha sido consistente en varios frames
            self.interaction_candidates.append({
                'frame': frame_number,
                'interactions': interactions
            })
            
            # Confirmar solo si hay interacciones consistentes en varios frames
            if len(self.interaction_candidates) >= self.min_frames_interaction:
                # Contar cuántos frames tienen interacciones similares
                consistent_count = 0
                confirmed_interaction = None
                
                for candidate in list(self.interaction_candidates)[-self.min_frames_interaction:]:
                    if candidate['interactions']:
                        consistent_count += 1
                        if confirmed_interaction is None:
                            confirmed_interaction = candidate['interactions'][0]
                
                if consistent_count >= self.min_frames_interaction:
                    confirmed = True
                    # Guardar interacción confirmada
                    if confirmed_interaction:
                        self.confirmed_interactions.append({
                            'frame': frame_number,
                            'bbox': confirmed_interaction['bbox']
                        })
        
        # Visualizar interacciones confirmadas
        if confirmed and interactions:
            for interaction in interactions:
                x1, y1, x2, y2 = interaction['bbox']
                wrist_pos = interaction['wrist']
                
                # Visualización muy visible
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 255), 8)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 5)
                cv2.circle(frame, wrist_pos, 20, (0, 255, 255), 5)
                cv2.line(frame, wrist_pos, ((x1+x2)//2, (y1+y2)//2), (0, 255, 255), 4)
                cv2.putText(frame, 'OBJETO COGIDO!', (x1, y1-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
        
        # Debug info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_number}', (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Muñecas: {len(wrist_positions)}', (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Objetos: {len(object_detections)}', (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Interacciones: {len(interactions)}', (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if interactions else (0, 0, 255), 2)
        if confirmed:
            cv2.putText(frame, 'OBJETO COGIDO CONFIRMADO!', (20, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
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
        print(f"Buscando interacciones entre frames 100-150...")
        
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
                frame_annotated, interactions, confirmed = self.process_frame(frame, frame_count)
                
                if confirmed:
                    total_interactions += 1
                    print(f"  → Interacción confirmada en frame {frame_count}")
                
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
                    
                    cv2.imshow('Deteccion V6 - Estricto', frame_display)
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
            if self.confirmed_interactions:
                print(f"  - Frames con interacciones: {[i['frame'] for i in self.confirmed_interactions]}")


def main():
    parser = argparse.ArgumentParser(description='Deteccion V6: Estricto y Robusto')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--threshold', type=int, default=100, help='Umbral de distancia')
    parser.add_argument('--min-frames', type=int, default=8, help='Frames para confirmar (estricto)')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualizacion')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorV6(
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
