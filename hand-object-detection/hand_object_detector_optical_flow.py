"""
Detector por Optical Flow: Detecta objetos cuando se mueven junto con la mano
Enfoque: Tracking óptico de la mano + detección de movimiento conjunto
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class HandObjectDetectorOpticalFlow:
    """
    Detecta objetos usando optical flow para detectar movimiento conjunto
    """
    
    def __init__(self, video_path, min_frames=5):
        self.video_path = video_path
        self.min_frames = min_frames
        
        # YOLO pose para muñecas
        print("Cargando YOLO pose...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # Parámetros de optical flow (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Historial
        self.left_hand_history = deque(maxlen=min_frames)
        self.right_hand_history = deque(maxlen=min_frames)
        self.previous_gray = None
        self.hand_tracks = {'left': None, 'right': None}  # Puntos de tracking
        self.object_tracks = {'left': [], 'right': []}  # Objetos que se mueven con la mano
        
        # Conexiones del esqueleto
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        print("✓ Detector por Optical Flow inicializado")
    
    def detect_moving_objects_near_hand(self, frame, gray, wrist_pos, hand_side):
        """
        Detecta objetos que se mueven cerca de la mano usando optical flow
        """
        if wrist_pos is None or self.previous_gray is None:
            return []
        
        wx, wy = wrist_pos
        h, w = gray.shape
        
        # Definir región de búsqueda alrededor de la muñeca
        search_radius = 200
        x1 = max(0, wx - search_radius)
        y1 = max(0, wy - search_radius)
        x2 = min(w, wx + search_radius)
        y2 = min(h, wy + search_radius)
        
        # Recortar regiones
        roi_current = gray[y1:y2, x1:x2]
        roi_previous = self.previous_gray[y1:y2, x1:x2]
        
        if roi_current.size == 0 or roi_previous.size == 0:
            return []
        
        # Calcular optical flow en la región
        # Detectar esquinas (puntos de interés)
        corners = cv2.goodFeaturesToTrack(roi_previous, maxCorners=100, 
                                          qualityLevel=0.01, minDistance=10)
        
        if corners is None or len(corners) == 0:
            return []
        
        # Calcular flujo óptico
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_previous, roi_current, 
                                                corners, None, **self.lk_params)
        
        # Filtrar puntos buenos
        good_new = p1[st == 1]
        good_old = corners[st == 1]
        
        if len(good_new) == 0:
            return []
        
        # Calcular movimiento de la mano (usar tracking si existe)
        hand_velocity = (0, 0)
        if self.hand_tracks[hand_side] is not None:
            # Calcular velocidad promedio de los puntos de tracking de la mano
            if len(self.hand_tracks[hand_side]) > 1:
                velocities = []
                for i in range(1, len(self.hand_tracks[hand_side])):
                    prev = self.hand_tracks[hand_side][i-1]
                    curr = self.hand_tracks[hand_side][i]
                    vel = (curr[0] - prev[0], curr[1] - prev[1])
                    velocities.append(vel)
                if velocities:
                    hand_velocity = (np.mean([v[0] for v in velocities]), 
                                   np.mean([v[1] for v in velocities]))
        
        # Analizar movimiento de puntos cerca de la mano
        objects = []
        wrist_relative = (wx - x1, wy - y1)  # Posición de muñeca relativa al ROI
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Calcular movimiento del punto
            dx = new[0][0] - old[0][0]
            dy = new[0][1] - old[0][1]
            magnitude = np.sqrt(dx**2 + dy**2)
            
            # Distancia del punto a la muñeca
            point_pos = (new[0][0] + x1, new[0][1] + y1)  # Coordenadas globales
            dist_to_wrist = np.sqrt((point_pos[0] - wx)**2 + (point_pos[1] - wy)**2)
            
            # Si el punto está cerca de la muñeca Y se mueve
            if dist_to_wrist < search_radius and magnitude > 2:
                # Verificar si se mueve junto con la mano
                hand_movement = np.sqrt(hand_velocity[0]**2 + hand_velocity[1]**2)
                
                # Si la mano se mueve, verificar si el punto se mueve en dirección similar
                if hand_movement > 3:
                    # Calcular ángulo entre movimiento del punto y movimiento de la mano
                    point_angle = np.arctan2(dy, dx)
                    hand_angle = np.arctan2(hand_velocity[1], hand_velocity[0])
                    angle_diff = abs(point_angle - hand_angle)
                    # Normalizar ángulo
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    
                    # Si el punto se mueve en dirección similar a la mano, es un objeto
                    if angle_diff < np.pi / 3:  # 60 grados
                        objects.append({
                            'position': point_pos,
                            'movement': (dx, dy),
                            'magnitude': magnitude,
                            'distance': dist_to_wrist,
                            'angle_diff': angle_diff
                        })
                else:
                    # Si la mano no se mueve mucho, cualquier movimiento cerca es sospechoso
                    if magnitude > 5:
                        objects.append({
                            'position': point_pos,
                            'movement': (dx, dy),
                            'magnitude': magnitude,
                            'distance': dist_to_wrist,
                            'angle_diff': 0
                        })
        
        return objects
    
    def process_frame(self, frame, frame_number):
        """Procesa un frame"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
        
        # Actualizar tracking de manos
        if self.previous_gray is not None:
            if left_wrist:
                if self.hand_tracks['left'] is None:
                    # Inicializar tracking
                    self.hand_tracks['left'] = deque(maxlen=10)
                self.hand_tracks['left'].append(left_wrist)
            
            if right_wrist:
                if self.hand_tracks['right'] is None:
                    self.hand_tracks['right'] = deque(maxlen=10)
                self.hand_tracks['right'].append(right_wrist)
        
        # Detectar objetos que se mueven con las manos
        left_has_object = False
        right_has_object = False
        left_objects = []
        right_objects = []
        
        if self.previous_gray is not None:
            if left_wrist:
                left_objects = self.detect_moving_objects_near_hand(
                    frame, gray, left_wrist, 'left'
                )
                # Si hay varios objetos moviéndose con la mano, probablemente tiene algo
                if len(left_objects) >= 3:
                    left_has_object = True
                    # Dibujar objetos
                    for obj in left_objects:
                        cv2.circle(frame, tuple(map(int, obj['position'])), 5, (0, 255, 0), 2)
            
            if right_wrist:
                right_objects = self.detect_moving_objects_near_hand(
                    frame, gray, right_wrist, 'right'
                )
                if len(right_objects) >= 3:
                    right_has_object = True
                    # Dibujar objetos
                    for obj in right_objects:
                        cv2.circle(frame, tuple(map(int, obj['position'])), 5, (255, 0, 0), 2)
        
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
        cv2.rectangle(overlay, (10, 10), (600, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_number}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mano izquierda
        left_color = (0, 255, 0) if left_confirmed else (150, 150, 150)
        left_bg = (0, 100, 0) if left_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 55), (580, 105), left_bg, -1)
        cv2.rectangle(frame, (15, 55), (580, 105), left_color, 3)
        left_text = f'Mano IZQUIERDA: {"✓ OBJETO" if left_confirmed else "✗ VACIA"} (puntos: {len(left_objects)})'
        cv2.putText(frame, left_text, (25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, left_color, 2)
        
        # Mano derecha
        right_color = (255, 0, 0) if right_confirmed else (150, 150, 150)
        right_bg = (100, 0, 0) if right_confirmed else (50, 50, 50)
        cv2.rectangle(frame, (15, 115), (580, 165), right_bg, -1)
        cv2.rectangle(frame, (15, 115), (580, 165), right_color, 3)
        right_text = f'Mano DERECHA: {"✓ OBJETO" if right_confirmed else "✗ VACIA"} (puntos: {len(right_objects)})'
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
        
        # Guardar frame para siguiente iteración
        self.previous_gray = gray.copy()
        
        return {
            'frame': frame,
            'left_hand': {
                'has_object': left_confirmed,
                'points': len(left_objects)
            },
            'right_hand': {
                'has_object': right_confirmed,
                'points': len(right_objects)
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
                          f"(puntos: {result['left_hand']['points']}), "
                          f"Der={'OBJETO' if result['right_hand']['has_object'] else 'VACIA'} "
                          f"(puntos: {result['right_hand']['points']})")
                
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
                    
                    cv2.imshow('Optical Flow Detector', frame_display)
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
    parser = argparse.ArgumentParser(description='Detector por Optical Flow')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--min-frames', type=int, default=5, help='Frames mínimos para confirmar')
    parser.add_argument('--output', type=str, default=None, help='Video de salida')
    parser.add_argument('--no-show', action='store_true', help='No mostrar video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    detector = HandObjectDetectorOpticalFlow(
        video_path=args.video_path,
        min_frames=args.min_frames
    )
    
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
