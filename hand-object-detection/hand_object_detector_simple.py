"""
Detector simple y directo: Gesto de mano cerrada + objetos YOLO cerca
Enfoque: Si la mano está cerrada (gesto) Y hay objetos cerca, tiene objeto
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorSimple:
    """
    Detector simple: gesto + objetos cerca
    """
    
    def __init__(self, video_path, detection_threshold=200, min_frames=3, min_confidence=0.2):
        self.video_path = video_path
        self.detection_threshold = detection_threshold
        self.min_frames = min_frames
        self.min_confidence = min_confidence
        
        # YOLO pose para muñecas y gestos
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # YOLO para objetos
        print("Cargando YOLO objetos...")
        self.yolo_objects = YOLO('yolov8n.pt')
        
        # Historial
        self.left_hand_history = deque(maxlen=min_frames)
        self.right_hand_history = deque(maxlen=min_frames)
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector simple inicializado")
    
    def is_hand_closed_gesture(self, keypoints, wrist_idx):
        """
        Detecta si la mano está cerrada analizando la posición de los dedos
        """
        if len(keypoints) < 17:
            return False
        
        # Índices de keypoints relevantes
        # Hombro, codo, muñeca
        if wrist_idx == 9:  # Muñeca izquierda
            shoulder_idx = 5
            elbow_idx = 7
        else:  # Muñeca derecha
            shoulder_idx = 6
            elbow_idx = 8
        
        wrist = keypoints[wrist_idx]
        elbow = keypoints[elbow_idx] if elbow_idx < len(keypoints) else None
        shoulder = keypoints[shoulder_idx] if shoulder_idx < len(keypoints) else None
        
        if elbow is None or shoulder is None:
            return False
        
        # Calcular ángulo brazo (para determinar orientación)
        # Vector hombro-codo y codo-muñeca
        arm_vec1 = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        arm_vec2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
        
        # Si la muñeca está más abajo que el codo (brazo extendido hacia abajo)
        # y la muñeca está cerca del cuerpo, puede estar cerrada
        wrist_below_elbow = wrist[1] > elbow[1]
        
        # Distancia muñeca-cuerpo (aproximada por distancia a hombro)
        dist_to_shoulder = np.sqrt((wrist[0] - shoulder[0])**2 + (wrist[1] - shoulder[1])**2)
        dist_elbow_shoulder = np.sqrt((elbow[0] - shoulder[0])**2 + (elbow[1] - shoulder[1])**2)
        
        # Si la muñeca está cerca del cuerpo (relativo al brazo), puede estar cerrada
        if dist_elbow_shoulder > 0:
            relative_dist = dist_to_shoulder / dist_elbow_shoulder
            # Si está cerca (ratio bajo) y abajo, probablemente cerrada
            hand_closed = relative_dist < 1.5 and wrist_below_elbow
        else:
            hand_closed = False
        
        return hand_closed
    
    def find_objects_near_wrist(self, wrist_pos, objects):
        """Encuentra objetos cerca de la muñeca"""
        if wrist_pos is None:
            return []
        
        wx, wy = wrist_pos
        nearby_objects = []
        
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            obj_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Distancia al centro del objeto
            dist = np.sqrt((wx - obj_center[0])**2 + (wy - obj_center[1])**2)
            
            # Verificar si la muñeca está dentro del bbox expandido
            margin = max((x2 - x1), (y2 - y1)) * 0.5
            wrist_inside = (x1 - margin <= wx <= x2 + margin and 
                           y1 - margin <= wy <= y2 + margin)
            
            if wrist_inside or dist < self.detection_threshold:
                nearby_objects.append({
                    'bbox': obj['bbox'],
                    'distance': dist,
                    'confidence': obj['confidence']
                })
        
        return nearby_objects
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        
        # Detectar personas con YOLO pose
        pose_results = self.yolo_pose(frame, conf=0.3, verbose=False)
        
        # Detectar objetos con YOLO
        object_results = self.yolo_objects(frame, conf=self.min_confidence, verbose=False)
        
        # Extraer muñecas y analizar gestos
        left_wrist = None
        right_wrist = None
        left_hand_closed = False
        right_hand_closed = False
        
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
                        
                        # Extraer muñecas y detectar gestos
                        if len(keypoints) > 10:
                            for wrist_idx in [9, 10]:
                                if (keypoints[wrist_idx][0] > 0 and keypoints[wrist_idx][1] > 0 and
                                    (confidences is None or confidences[wrist_idx] > 0.3)):
                                    wrist_pos = tuple(map(int, keypoints[wrist_idx]))
                                    
                                    if wrist_idx == 9:
                                        left_wrist = wrist_pos
                                        left_hand_closed = self.is_hand_closed_gesture(keypoints, wrist_idx)
                                    elif wrist_idx == 10:
                                        right_wrist = wrist_pos
                                        right_hand_closed = self.is_hand_closed_gesture(keypoints, wrist_idx)
                                    
                                    # Dibujar muñeca
                                    color = (0, 255, 0) if wrist_idx == 9 else (255, 0, 0)
                                    if (wrist_idx == 9 and left_hand_closed) or (wrist_idx == 10 and right_hand_closed):
                                        cv2.circle(frame, wrist_pos, 20, color, 5)  # Más grande si cerrada
                                    else:
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
                
                # Filtrar personas
                if cls != 0 and conf > self.min_confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Filtrar por tamaño
                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    obj_area = obj_width * obj_height
                    
                    if 300 < obj_area < 200000:  # Objetos pequeños/medianos
                        detected_objects.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls,
                            'area': obj_area
                        })
                        
                        # Dibujar objeto (sutil)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
        # Detectar objetos en cada mano
        left_has_object = False
        right_has_object = False
        
        if left_wrist:
            nearby_objects = self.find_objects_near_wrist(left_wrist, detected_objects)
            # Si la mano está cerrada Y hay objetos cerca, tiene objeto
            if left_hand_closed and len(nearby_objects) > 0:
                left_has_object = True
                # Visualizar
                for obj in nearby_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 0), 4)
                    cv2.line(frame, left_wrist, ((x1+x2)//2, (y1+y2)//2), (0, 255, 0), 3)
        
        if right_wrist:
            nearby_objects = self.find_objects_near_wrist(right_wrist, detected_objects)
            if right_hand_closed and len(nearby_objects) > 0:
                right_has_object = True
                # Visualizar
                for obj in nearby_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255, 0, 0), 4)
                    cv2.line(frame, right_wrist, ((x1+x2)//2, (y1+y2)//2), (255, 0, 0), 3)
        
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
        
        # Panel de información
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_number}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mano izquierda
        left_color = (0, 255, 0) if left_confirmed else (150, 150, 150)
        left_bg = (0, 100, 0) if left_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 55), (630, 105), left_bg, -1)
        cv2.rectangle(frame, (15, 55), (630, 105), left_color, 3)
        left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"} (gesto: {"CERRADA" if left_hand_closed else "ABIERTA"})'
        cv2.putText(frame, left_text, (25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, left_color, 2)
        
        # Mano derecha
        right_color = (255, 0, 0) if right_confirmed else (150, 150, 150)
        right_bg = (100, 0, 0) if right_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 115), (630, 165), right_bg, -1)
        cv2.rectangle(frame, (15, 115), (630, 165), right_color, 3)
        right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"} (gesto: {"CERRADA" if right_hand_closed else "ABIERTA"})'
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
                'gesture_closed': left_hand_closed
            },
            'right_hand': {
                'has_object': right_confirmed,
                'gesture_closed': right_hand_closed
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
                          f"(gesto: {'CERRADA' if result['left_hand']['gesture_closed'] else 'ABIERTA'}), "
                          f"Der={'OBJETO' if result['right_hand']['has_object'] else 'VACIA'} "
                          f"(gesto: {'CERRADA' if result['right_hand']['gesture_closed'] else 'ABIERTA'})")
                
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
                    
                    cv2.imshow('Detector Simple', frame_display)
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
    parser = argparse.ArgumentParser(description='Detector simple: Gesto + objetos')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--threshold', type=int, default=200, help='Distancia máxima muñeca-objeto')
    parser.add_argument('--min-frames', type=int, default=3, help='Frames mínimos para confirmar')
    parser.add_argument('--confidence', type=float, default=0.2, help='Confianza mínima para objetos')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorSimple(
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
