"""
Detección de interacción mano-objeto para videos de supermercado
Combina MediaPipe (detección de manos) con YOLO (detección de objetos)
Optimizado para distancias de ~5 metros
"""

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage, ImageFormat
from ultralytics import YOLO
from collections import deque
import argparse
from scipy import ndimage


class HandObjectDetector:
    def __init__(self, video_path, yolo_model='yolov8n-pose.pt', interaction_threshold=80, 
                 min_frames_interaction=3, confidence_hands=0.5, confidence_objects=0.3, confidence_person=0.5,
                 use_ddshan=False, ddshan_model_path=None):
        """
        Inicializa el detector de interacción mano-objeto
        
        Args:
            video_path: Ruta al video a procesar
            yolo_model: Modelo YOLO a usar (recomendado: 'yolov8n-pose.pt' para pose, o 'yolov8n.pt' para objetos)
            interaction_threshold: Distancia en píxeles para considerar interacción (ajustar según resolución)
            min_frames_interaction: Mínimo de frames consecutivos para confirmar interacción
            confidence_hands: Confianza mínima para detección de manos (0.0-1.0)
            confidence_objects: Confianza mínima para detección de objetos (0.0-1.0)
            confidence_person: Confianza mínima para detección de personas (0.0-1.0)
        """
        self.video_path = video_path
        self.use_ddshan = use_ddshan
        
        # Intentar cargar detector de ddshan si se solicita
        self.ddshan_detector = None
        if use_ddshan:
            try:
                from ddshan_wrapper import DdshanDetector, is_available
                if is_available():
                    print("Inicializando detector de ddshan...")
                    self.ddshan_detector = DdshanDetector(
                        model_path=ddshan_model_path,
                        thresh_hand=confidence_hands,
                        thresh_obj=confidence_objects
                    )
                    print("✓ Detector de ddshan inicializado correctamente")
                else:
                    print("⚠ Detector de ddshan no disponible, usando método alternativo")
                    self.use_ddshan = False
            except Exception as e:
                print(f"⚠ Error al inicializar detector de ddshan: {e}")
                print("  Usando método alternativo (MediaPipe + YOLO)")
                self.use_ddshan = False
        
        # Inicializar MediaPipe para detección de manos (nueva API 0.10.x)
        # Buscar el modelo en el directorio actual o descargarlo si no existe
        import os
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            # Si no existe, intentar descargarlo
            import urllib.request
            model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            print(f"Descargando modelo de MediaPipe...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Modelo descargado correctamente")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        # Bajar mucho los umbrales para distancias largas
        min_hand_conf = min(0.2, confidence_hands)  # Máximo 0.2 para distancias largas
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=min_hand_conf,
            min_hand_presence_confidence=min_hand_conf,
            min_tracking_confidence=min_hand_conf
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        print(f"MediaPipe configurado con confianza mínima: {min_hand_conf}")
        
        # Inicializar YOLO para detección de personas (pose) y objetos
        print(f"Cargando modelo YOLO: {yolo_model}...")
        self.yolo_model = YOLO(yolo_model)
        self.has_pose = 'pose' in yolo_model.lower()
        print(f"Modelo YOLO cargado correctamente (Pose: {self.has_pose})")
        
        # Si no es modelo pose, cargar también modelo de objetos
        if not self.has_pose:
            print("Cargando modelo adicional para objetos...")
            self.yolo_objects = YOLO('yolov8n.pt')
        else:
            self.yolo_objects = None
        
        # Parámetros de detección
        self.interaction_threshold = interaction_threshold
        self.min_frames_interaction = min_frames_interaction
        self.confidence_hands = confidence_hands
        self.confidence_objects = confidence_objects
        self.confidence_person = confidence_person
        
        # Tracking temporal para reducir falsos positivos
        self.interaction_history = deque(maxlen=min_frames_interaction)
        
        # Conexiones del esqueleto humano (COCO pose format)
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Cabeza
            (5, 6),  # Hombros
            (5, 7), (7, 9),  # Brazo izquierdo
            (6, 8), (8, 10),  # Brazo derecho
            (5, 11), (6, 12),  # Torso
            (11, 13), (13, 15),  # Pierna izquierda
            (12, 14), (14, 16),  # Pierna derecha
            (11, 12)  # Cadera
        ]
        
    def get_hand_center(self, hand_landmarks, frame_width, frame_height):
        """
        Obtiene el centro de la mano (punto de referencia para interacción)
        Usa el landmark 9 (base de la palma) como referencia principal
        """
        # Landmark 9 es la base de la palma (mejor punto de referencia)
        palm = hand_landmarks[9]
        # También consideramos el punto medio entre índice y pulgar para mayor precisión
        index_tip = hand_landmarks[8]
        thumb_tip = hand_landmarks[4]
        
        # Centro ponderado
        center_x = (palm.x * 0.5 + index_tip.x * 0.25 + thumb_tip.x * 0.25) * frame_width
        center_y = (palm.y * 0.5 + index_tip.y * 0.25 + thumb_tip.y * 0.25) * frame_height
        
        return int(center_x), int(center_y)
    
    def is_hand_closed(self, hand_landmarks):
        """
        Detecta si la mano está cerrada (posible agarre de objeto)
        Compara distancias entre dedos
        """
        # Puntos clave de los dedos
        thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
        index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
        middle_tip = np.array([hand_landmarks[12].x, hand_landmarks[12].y])
        ring_tip = np.array([hand_landmarks[16].x, hand_landmarks[16].y])
        
        # Calcular distancias entre dedos
        thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
        thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
        
        # Si las distancias son pequeñas, la mano está cerrada
        return thumb_index_dist < 0.05 and thumb_middle_dist < 0.05
    
    def detect_object_by_hand_shape(self, hand_landmarks):
        """
        Detecta si hay un objeto comparando la forma de la mano con una plantilla de mano normal
        Si la forma no encaja bien con una mano típica, probablemente hay un objeto
        
        Returns:
            (has_object, confidence): Tupla con booleano y confianza (0-1)
        """
        # Obtener todos los landmarks normalizados
        landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks])
        
        # Plantilla de mano abierta típica (proporciones relativas)
        # Basado en la estructura típica de una mano humana
        open_hand_template = np.array([
            [0.5, 0.5],      # 0: muñeca
            [0.5, 0.4],     # 1: base pulgar
            [0.55, 0.35],   # 2: medio pulgar
            [0.6, 0.3],     # 3: pulgar
            [0.65, 0.25],   # 4: punta pulgar
            [0.45, 0.3],    # 5: base índice
            [0.45, 0.2],    # 6: medio índice
            [0.45, 0.1],    # 7: índice
            [0.45, 0.0],    # 8: punta índice
            [0.5, 0.3],     # 9: base medio
            [0.5, 0.15],    # 10: medio medio
            [0.5, 0.05],    # 11: medio
            [0.5, -0.05],   # 12: punta medio
            [0.55, 0.3],    # 13: base anular
            [0.55, 0.15],   # 14: medio anular
            [0.55, 0.05],   # 15: anular
            [0.55, -0.05],  # 16: punta anular
            [0.6, 0.3],     # 17: base meñique
            [0.6, 0.2],     # 18: medio meñique
            [0.6, 0.1],     # 19: meñique
            [0.6, 0.0]      # 20: punta meñique
        ])
        
        # Normalizar landmarks actuales al mismo rango
        landmarks_normalized = landmarks.copy()
        # Centrar en la muñeca (landmark 0)
        wrist = landmarks[0]
        landmarks_normalized = landmarks_normalized - wrist
        
        # Escalar para que el índice (landmark 8) tenga aproximadamente la misma longitud
        if len(landmarks) > 8:
            current_index_length = np.linalg.norm(landmarks[8] - landmarks[0])
            template_index_length = np.linalg.norm(open_hand_template[8] - open_hand_template[0])
            if current_index_length > 0:
                scale = template_index_length / current_index_length
                landmarks_normalized = landmarks_normalized * scale
        
        # Calcular distancia promedio entre landmarks actuales y plantilla
        if len(landmarks_normalized) == len(open_hand_template):
            distances = np.linalg.norm(landmarks_normalized - open_hand_template, axis=1)
            avg_distance = np.mean(distances)
            
            # Si la distancia promedio es alta, la forma no encaja (probablemente hay objeto)
            # Umbral ajustable: valores altos = forma diferente = posible objeto
            threshold = 0.15  # Ajustar según pruebas
            
            has_object = avg_distance > threshold
            confidence = min(1.0, avg_distance / 0.3)  # Normalizar a 0-1
            
            return has_object, confidence
        
        return False, 0.0
    
    def detect_grasp_gesture(self, hand_landmarks):
        """
        Detecta si la mano está en posición de agarre/coger objeto
        Basado en gestos de agarre típicos
        
        Returns:
            (is_grasping, confidence): Tupla con booleano y confianza (0-1)
        """
        # Obtener puntos clave
        thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
        index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
        middle_tip = np.array([hand_landmarks[12].x, hand_landmarks[12].y])
        ring_tip = np.array([hand_landmarks[16].x, hand_landmarks[16].y])
        pinky_tip = np.array([hand_landmarks[20].x, hand_landmarks[20].y])
        
        # Puntos base de los dedos
        index_mcp = np.array([hand_landmarks[5].x, hand_landmarks[5].y])
        middle_mcp = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
        
        # Calcular distancias entre puntas de dedos
        thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
        thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
        index_middle_dist = np.linalg.norm(index_tip - middle_tip)
        
        # Calcular si los dedos están curvados (distancia punta-base)
        index_curl = np.linalg.norm(index_tip - index_mcp)
        middle_curl = np.linalg.norm(middle_tip - middle_mcp)
        
        # Gestos de agarre típicos:
        # 1. Pinza: pulgar cerca de índice (distancia pequeña)
        # 2. Agarre: varios dedos curvados y cerca del pulgar
        # 3. Puño: todos los dedos muy cerca
        
        is_pinch = thumb_index_dist < 0.08  # Pinza
        is_grip = (thumb_index_dist < 0.12 and thumb_middle_dist < 0.12 and 
                  index_curl < 0.15 and middle_curl < 0.15)  # Agarre con dedos curvados
        is_fist = (thumb_index_dist < 0.06 and thumb_middle_dist < 0.06 and 
                  index_middle_dist < 0.06)  # Puño cerrado
        
        is_grasping = is_pinch or is_grip or is_fist
        
        # Calcular confianza basada en qué tan "cerrada" está la mano
        if is_grasping:
            # Confianza basada en qué tan cerca están los dedos
            min_dist = min(thumb_index_dist, thumb_middle_dist, index_middle_dist)
            confidence = 1.0 - (min_dist / 0.12)  # Normalizar
            confidence = max(0.5, min(1.0, confidence))
        else:
            confidence = 0.0
        
        return is_grasping, confidence
    
    def detect_object_in_hand(self, hand_landmarks):
        """
        Detecta si hay un objeto en la mano combinando:
        1. Comparación de forma con plantilla de mano normal
        2. Detección de gestos de agarre
        
        Returns:
            (has_object, confidence, method): Tupla con booleano, confianza y método usado
        """
        # Método 1: Comparación de forma
        shape_has_object, shape_confidence = self.detect_object_by_hand_shape(hand_landmarks)
        
        # Método 2: Detección de gesto de agarre
        grasp_detected, grasp_confidence = self.detect_grasp_gesture(hand_landmarks)
        
        # Combinar ambos métodos
        # Si ambos indican objeto, alta confianza
        # Si solo uno indica, confianza media
        if shape_has_object and grasp_detected:
            confidence = (shape_confidence + grasp_confidence) / 2
            method = "ambos"
            return True, confidence, method
        elif shape_has_object:
            # Forma diferente pero no necesariamente agarre
            confidence = shape_confidence * 0.7
            method = "forma"
            return True, confidence, method
        elif grasp_detected:
            # Gesto de agarre detectado
            confidence = grasp_confidence * 0.8
            method = "gesto"
            return True, confidence, method
        else:
            return False, 0.0, "ninguno"
    
    def calculate_distance(self, point1, point2):
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_objects_near_hand(self, frame, hand_positions, hand_landmarks_list):
        """
        Detecta objetos cerca de las manos usando procesamiento de imagen
        Encuentra contornos y regiones de interés cerca de la mano que no sean la mano misma
        """
        h, w = frame.shape[:2]
        detected_objects = []
        
        for hand_idx, hand_pos in enumerate(hand_positions):
            hx, hy = hand_pos
            
            # Crear máscara de la mano
            hand_mask = np.zeros((h, w), dtype=np.uint8)
            hand_mask_dilated = np.zeros((h, w), dtype=np.uint8)
            
            if hand_landmarks_list and hand_idx < len(hand_landmarks_list):
                # Crear polígono de la mano usando landmarks
                hand_points = []
                for lm in hand_landmarks_list[hand_idx]:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    hand_points.append([px, py])
                
                if len(hand_points) > 0:
                    hand_points = np.array(hand_points, dtype=np.int32)
                    cv2.fillPoly(hand_mask, [hand_points], 255)
                    
                    # Dilatar la máscara para incluir área cercana
                    kernel = np.ones((30, 30), np.uint8)
                    hand_mask_dilated = cv2.dilate(hand_mask, kernel, iterations=2)
            
            # Área de búsqueda alrededor de la mano
            search_radius = self.interaction_threshold * 2
            search_x1 = max(0, hx - search_radius)
            search_y1 = max(0, hy - search_radius)
            search_x2 = min(w, hx + search_radius)
            search_y2 = min(h, hy + search_radius)
            
            # Recortar región de interés
            roi = frame[search_y1:search_y2, search_x1:search_x2].copy()
            
            if roi.size == 0:
                continue
            
            # Convertir a escala de grises
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Aplicar filtro para detectar bordes
            edges = cv2.Canny(gray_roi, 50, 150)
            
            # Si hay máscara de mano, excluirla
            if hand_mask_dilated is not None and np.sum(hand_mask_dilated) > 0:
                roi_mask = hand_mask_dilated[search_y1:search_y2, search_x1:search_x2]
                edges = cv2.bitwise_and(edges, cv2.bitwise_not(roi_mask))
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por tamaño y posición
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtrar contornos muy pequeños o muy grandes
                if area < 200 or area > 50000:
                    continue
                
                # Obtener bounding box del contorno
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Ajustar coordenadas al frame completo
                abs_x = x + search_x1
                abs_y = y + search_y1
                abs_x2 = abs_x + cw
                abs_y2 = abs_y + ch
                
                # Verificar que el centro del objeto esté cerca de la mano
                obj_center_x = abs_x + cw // 2
                obj_center_y = abs_y + ch // 2
                dist_to_hand = self.calculate_distance(hand_pos, (obj_center_x, obj_center_y))
                
                if dist_to_hand < self.interaction_threshold * 1.5:
                    # Calcular características del objeto para validar
                    obj_roi = frame[abs_y:abs_y2, abs_x:abs_x2]
                    
                    if obj_roi.size > 0:
                        # Verificar que no sea principalmente piel (color de la mano)
                        hsv_roi = cv2.cvtColor(obj_roi, cv2.COLOR_BGR2HSV)
                        # Rango de color de piel en HSV
                        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                        skin_mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)
                        skin_ratio = np.sum(skin_mask > 0) / (cw * ch) if cw * ch > 0 else 0
                        
                        # Si menos del 40% es piel, probablemente es un objeto
                        if skin_ratio < 0.4:
                            detected_objects.append({
                                'bbox': [abs_x, abs_y, abs_x2, abs_y2],
                                'confidence': 0.6,  # Confianza media para objetos detectados por contornos
                                'class': -2,  # Clase especial para objetos detectados por contornos
                                'distance': dist_to_hand,
                                'area': area
                            })
        
        return detected_objects
    
    def detect_interactions(self, hand_positions, object_boxes, person_boxes, frame_shape):
        """
        Detecta interacciones entre manos y objetos basándose en proximidad
        Solo cuenta objetos REALES detectados (no estimados)
        
        Returns:
            Lista de interacciones detectadas [(hand_pos, obj_box, distance)]
        """
        interactions = []
        h, w = frame_shape[:2]
        
        # Filtrar objetos reales: YOLO (clase >= 0), contornos (clase == -2), o forma de mano (clase == -3)
        real_objects = [obj for obj in object_boxes if (obj['class'] >= 0 or obj['class'] == -2 or obj['class'] == -3) and obj['confidence'] > 0.2]
        
        # Si hay manos detectadas, buscar objetos cercanos
        for hand_pos in hand_positions:
            hand_x, hand_y = hand_pos
            
            for obj_box in real_objects:
                x1, y1, x2, y2 = obj_box['bbox']
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2
                
                # Calcular distancia entre mano y centro del objeto
                distance = self.calculate_distance(hand_pos, (obj_center_x, obj_center_y))
                
                # Verificar si la mano está dentro del bounding box
                hand_inside_box = x1 <= hand_x <= x2 and y1 <= hand_y <= y2
                
                # Calcular distancia a cualquier borde del bbox
                min_dist_to_box = min(
                    abs(hand_x - x1), abs(hand_x - x2),
                    abs(hand_y - y1), abs(hand_y - y2)
                )
                
                # Ajustar umbral según tamaño del objeto
                obj_area = (x2 - x1) * (y2 - y1)
                threshold = self.interaction_threshold
                
                # Para objetos pequeños, ser más estricto
                if obj_area < 2000:  # Objetos muy pequeños
                    threshold = self.interaction_threshold * 0.6
                elif obj_area < 10000:  # Objetos medianos
                    threshold = self.interaction_threshold * 0.8
                
                # Solo considerar interacción si:
                # 1. La mano está dentro del bbox, O
                # 2. Está muy cerca del bbox (dentro del umbral ajustado)
                if hand_inside_box:
                    interactions.append({
                        'hand': hand_pos,
                        'object': (obj_center_x, obj_center_y),
                        'bbox': obj_box['bbox'],
                        'distance': 0,  # Dentro del bbox
                        'confidence': obj_box['confidence']
                    })
                elif min_dist_to_box < threshold:
                    # Solo si está realmente cerca del bbox
                    interactions.append({
                        'hand': hand_pos,
                        'object': (obj_center_x, obj_center_y),
                        'bbox': obj_box['bbox'],
                        'distance': min_dist_to_box,
                        'confidence': obj_box['confidence']
                    })
        
        return interactions
    
    def process_frame(self, frame):
        """
        Procesa un frame y detecta interacciones mano-objeto
        
        Returns:
            frame_annotated: Frame con anotaciones visuales
            interactions: Lista de interacciones detectadas
        """
        # Si se usa el detector de ddshan, procesar con él
        if self.use_ddshan and self.ddshan_detector is not None:
            return self._process_frame_ddshan(frame)
        
        # Método original (MediaPipe + YOLO + detección por forma)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # 1. DETECCIÓN DE PERSONAS Y POSE (YOLO)
        person_boxes = []
        person_keypoints = []
        
        if self.has_pose:
            # Usar modelo pose para detectar personas y esqueleto
            results_pose = self.yolo_model(frame, conf=self.confidence_person, verbose=False)
            
            if results_pose[0].boxes is not None:
                for i, box in enumerate(results_pose[0].boxes):
                    cls = int(box.cls[0].cpu().numpy())
                    if cls == 0:  # Persona
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        
                        person_boxes.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })
                        
                        # Dibujar bounding box de persona
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(frame, f'Persona: {conf:.2f}', (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        
                        # Dibujar esqueleto si hay keypoints
                        if results_pose[0].keypoints is not None and len(results_pose[0].keypoints) > i:
                            keypoints = results_pose[0].keypoints[i].xy[0].cpu().numpy()
                            confidences = results_pose[0].keypoints[i].conf[0].cpu().numpy() if hasattr(results_pose[0].keypoints[i], 'conf') else None
                            
                            # Dibujar conexiones del esqueleto
                            for connection in self.POSE_CONNECTIONS:
                                pt1_idx, pt2_idx = connection
                                if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                                    pt1 = tuple(map(int, keypoints[pt1_idx]))
                                    pt2 = tuple(map(int, keypoints[pt2_idx]))
                                    
                                    # Solo dibujar si ambos puntos tienen confianza suficiente
                                    if confidences is None or (confidences[pt1_idx] > 0.3 and confidences[pt2_idx] > 0.3):
                                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                            
                            # Dibujar puntos clave
                            for j, kp in enumerate(keypoints):
                                if confidences is None or confidences[j] > 0.3:
                                    cv2.circle(frame, tuple(map(int, kp)), 4, (0, 255, 0), -1)
                            
                            person_keypoints.append(keypoints)
        else:
            # Si no es modelo pose, detectar personas con modelo normal
            results_person = self.yolo_model(frame, conf=self.confidence_person, verbose=False)
            if results_person[0].boxes is not None:
                for box in results_person[0].boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    if cls == 0:  # Persona
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        
                        person_boxes.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })
                        
                        # Dibujar bounding box de persona
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(frame, f'Persona: {conf:.2f}', (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # 2. DETECCIÓN DE MANOS (MediaPipe) - Con umbrales muy bajos
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.hand_landmarker.detect(mp_image)
        
        hand_positions = []
        hand_closed_states = []
        hand_landmarks_list = []  # Guardar landmarks para detección de objetos
        
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
            (0, 5), (5, 6), (6, 7), (7, 8),  # Índice
            (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
            (0, 13), (13, 14), (14, 15), (15, 16),  # Anular
            (0, 17), (17, 18), (18, 19), (19, 20),  # Meñique
            (5, 9), (9, 13), (13, 17)  # Conexiones entre dedos
        ]
        
        hand_objects = []  # Objetos detectados por forma de mano
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Guardar landmarks
                hand_landmarks_list.append(hand_landmarks)
                
                # Dibujar conexiones de la mano
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    pt1 = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                    pt2 = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                
                # Dibujar puntos de landmarks
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
                # Obtener posición de la mano
                hand_pos = self.get_hand_center(hand_landmarks, w, h)
                hand_positions.append(hand_pos)
                
                # Detectar si la mano está cerrada
                is_closed = self.is_hand_closed(hand_landmarks)
                hand_closed_states.append(is_closed)
                
                # NUEVO: Detectar objeto por forma de mano y gestos
                has_object, obj_confidence, method = self.detect_object_in_hand(hand_landmarks)
                
                if has_object:
                    # Crear bbox alrededor de la mano cuando tiene objeto
                    # Obtener bounding box de todos los landmarks
                    x_coords = [int(lm.x * w) for lm in hand_landmarks]
                    y_coords = [int(lm.y * h) for lm in hand_landmarks]
                    
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    
                    # Expandir bbox un poco para incluir el objeto
                    padding = 40
                    obj_x1 = max(0, min_x - padding)
                    obj_y1 = max(0, min_y - padding)
                    obj_x2 = min(w, max_x + padding)
                    obj_y2 = min(h, max_y + padding)
                    
                    hand_objects.append({
                        'bbox': [obj_x1, obj_y1, obj_x2, obj_y2],
                        'confidence': obj_confidence,
                        'class': -3,  # Clase especial para objetos detectados por forma
                        'method': method,
                        'hand_pos': hand_pos
                    })
                    
                    # Dibujar bbox del objeto detectado por forma (magenta)
                    cv2.rectangle(frame, (obj_x1, obj_y1), (obj_x2, obj_y2), (255, 0, 255), 3)
                    cv2.putText(frame, f'Objeto ({method}): {obj_confidence:.2f}', 
                               (obj_x1, obj_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Marcar posición de la mano con círculo más grande
                color = (0, 255, 0) if not has_object else (255, 0, 255)  # Verde si no tiene objeto, magenta si tiene
                cv2.circle(frame, hand_pos, 10, color, 3)
        
        # 2b. DETECCIÓN ALTERNATIVA: Usar keypoints del esqueleto para estimar posición de manos
        if len(hand_positions) == 0 and len(person_keypoints) > 0:
            for keypoints in person_keypoints:
                # Índices de COCO pose: 9=muñeca izquierda, 10=muñeca derecha
                if len(keypoints) > 10:
                    # Muñeca izquierda (índice 9)
                    if keypoints[9][0] > 0 and keypoints[9][1] > 0:
                        wrist_left = tuple(map(int, keypoints[9]))
                        hand_positions.append(wrist_left)
                        cv2.circle(frame, wrist_left, 15, (255, 255, 0), 3)
                        cv2.putText(frame, 'Mano L', (wrist_left[0]+20, wrist_left[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Muñeca derecha (índice 10)
                    if keypoints[10][0] > 0 and keypoints[10][1] > 0:
                        wrist_right = tuple(map(int, keypoints[10]))
                        hand_positions.append(wrist_right)
                        cv2.circle(frame, wrist_right, 15, (255, 255, 0), 3)
                        cv2.putText(frame, 'Mano R', (wrist_right[0]+20, wrist_right[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 3. DETECCIÓN DE OBJETOS (YOLO) - Con umbral MUY bajo y múltiples intentos
        object_boxes = []
        
        # Intentar con diferentes umbrales de confianza
        confidence_levels = [0.1, 0.15, 0.2, 0.25] if self.confidence_objects < 0.2 else [self.confidence_objects]
        
        for conf_level in confidence_levels:
            # Si es modelo pose, usar modelo adicional para objetos, si no, usar el mismo modelo
            if self.has_pose and self.yolo_objects:
                results_objects = self.yolo_objects(frame, conf=conf_level, verbose=False)
            else:
                results_objects = self.yolo_model(frame, conf=conf_level, verbose=False)
            
            if results_objects[0].boxes is not None:
                for box in results_objects[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Excluir personas (clase 0)
                    if cls != 0:
                        # Evitar duplicados (objetos muy cercanos)
                        is_duplicate = False
                        for existing in object_boxes:
                            ex1, ey1, ex2, ey2 = existing['bbox']
                            # Si los bboxes se solapan mucho, es duplicado
                            overlap = max(0, min(x2, ex2) - max(x1, ex1)) * max(0, min(y2, ey2) - max(y1, ey1))
                            area1 = (x2-x1) * (y2-y1)
                            area2 = (ex2-ex1) * (ey2-ey1)
                            if overlap > 0.5 * min(area1, area2):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            object_boxes.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class': cls
                            })
                            
                            # Dibujar bounding box del objeto (verde) - SIEMPRE mostrar
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Mostrar clase COCO si es conocida
                            class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                                         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                                         'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                                         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                                         'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                                         'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                                         'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                                         'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                                         'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                                         'toothbrush']
                            class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
                            cv2.putText(frame, f'{class_name}: {conf:.2f}', (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 3b. Añadir objetos detectados por forma de mano
        for hand_obj in hand_objects:
            # Evitar duplicados
            is_duplicate = False
            hx1, hy1, hx2, hy2 = hand_obj['bbox']
            
            for existing in object_boxes:
                ex1, ey1, ex2, ey2 = existing['bbox']
                overlap = max(0, min(hx2, ex2) - max(hx1, ex1)) * max(0, min(hy2, ey2) - max(hy1, ey1))
                area1 = (hx2-hx1) * (hy2-hy1)
                area2 = (ex2-ex1) * (ey2-ey1)
                if overlap > 0.3 * min(area1, area2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                object_boxes.append(hand_obj)
        
        # 3c. DETECCIÓN ALTERNATIVA: Detectar objetos cerca de las manos usando procesamiento de imagen
        if len(hand_positions) > 0 and len(hand_landmarks_list) > 0:
            contour_objects = self.detect_objects_near_hand(frame, hand_positions, hand_landmarks_list)
            
            # Añadir objetos detectados por contornos (evitando duplicados)
            for contour_obj in contour_objects:
                is_duplicate = False
                cx1, cy1, cx2, cy2 = contour_obj['bbox']
                
                for existing in object_boxes:
                    ex1, ey1, ex2, ey2 = existing['bbox']
                    # Verificar solapamiento
                    overlap = max(0, min(cx2, ex2) - max(cx1, ex1)) * max(0, min(cy2, ey2) - max(cy1, ey1))
                    area1 = (cx2-cx1) * (cy2-cy1)
                    area2 = (ex2-ex1) * (ey2-ey1)
                    if overlap > 0.3 * min(area1, area2):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    object_boxes.append(contour_obj)
                    # Dibujar objeto detectado por contornos (azul)
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
                    cv2.putText(frame, f'Objeto (contorno): {contour_obj["confidence"]:.2f}', (cx1, cy1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 3b. DETECCIÓN ALTERNATIVA: Solo mostrar áreas de búsqueda, NO crear objetos falsos
        # (Comentado para evitar falsos positivos)
        # if len(object_boxes) == 0 and len(hand_positions) > 0:
        #     # Solo dibujar áreas de búsqueda para debug, pero NO añadirlas como objetos
        #     for hand_pos in hand_positions:
        #         hx, hy = hand_pos
        #         search_size = self.interaction_threshold * 2
        #         est_x1 = max(0, hx - search_size)
        #         est_y1 = max(0, hy - search_size)
        #         est_x2 = min(w, hx + search_size)
        #         est_y2 = min(h, hy + search_size)
        #         
        #         # Solo dibujar área de búsqueda (amarillo) para visualización, NO como objeto
        #         cv2.rectangle(frame, (est_x1, est_y1), (est_x2, est_y2), (0, 255, 255), 1)
        #         cv2.putText(frame, 'Area busqueda', (est_x1, est_y1-10),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 4. DETECTAR INTERACCIONES MANO-OBJETO (mejorado)
        interactions = self.detect_interactions(hand_positions, object_boxes, person_boxes, frame.shape)
        
        # Actualizar historial - Solo contar interacciones con objetos reales
        # Filtrar interacciones con confianza suficiente
        valid_interactions = [i for i in interactions if i['confidence'] > 0.25]
        has_interaction = len(valid_interactions) > 0
        
        self.interaction_history.append(has_interaction)
        
        # Confirmar solo si hay interacciones válidas en varios frames consecutivos
        confirmed_interaction = (sum(self.interaction_history) >= self.min_frames_interaction 
                                and len(self.interaction_history) == self.min_frames_interaction
                                and len(valid_interactions) > 0)
        
        # Usar solo interacciones válidas para visualización
        interactions = valid_interactions
        
        # 5. VISUALIZAR INTERACCIONES (objeto cogido)
        grabbed_object = None
        for interaction in interactions:
            hand_pos = interaction['hand']
            obj_pos = interaction['object']
            bbox = interaction['bbox']
            
            # Dibujar línea de conexión mano-objeto
            color = (0, 255, 255) if confirmed_interaction else (255, 255, 0)  # Cian para confirmado
            thickness = 5 if confirmed_interaction else 2
            cv2.line(frame, hand_pos, obj_pos, color, thickness)
            
            # Resaltar objeto cogido con color muy visible
            if confirmed_interaction:
                x1, y1, x2, y2 = bbox
                
                # Rectángulo exterior grueso en CIAN (BGR: 255, 255, 0)
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (255, 255, 0), 6)
                
                # Rectángulo interior en NARANJA (BGR: 0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 4)
                
                # Círculo grande en el centro
                cv2.circle(frame, obj_pos, 20, (255, 255, 0), 5)
                cv2.circle(frame, obj_pos, 10, (0, 165, 255), -1)
                
                # Texto grande y visible
                text = "OBJETO COGIDO!"
                font_scale = 1.2
                thickness_text = 3
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)
                
                # Fondo para el texto
                cv2.rectangle(frame, (x1, y1 - text_height - 20), (x1 + text_width + 10, y1), (255, 255, 0), -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_text)
                
                grabbed_object = bbox
        
        # 6. MOSTRAR INFORMACIÓN DE DEBUG EN PANTALLA (siempre visible)
        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Información de debug
        y_offset = 30
        cv2.putText(frame, f'Personas: {len(person_boxes)}', (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_offset += 30
        cv2.putText(frame, f'Manos: {len(hand_positions)}', (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f'Objetos: {len(object_boxes)}', (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(frame, f'Interacciones: {len(interactions)}', (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_offset += 30
        if confirmed_interaction:
            cv2.putText(frame, 'OBJETO COGIDO!', (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        
        # Mostrar interacciones aunque no estén confirmadas (para debug)
        for i, interaction in enumerate(interactions):
            if not confirmed_interaction:
                bbox = interaction['bbox']
                x1, y1, x2, y2 = bbox
                # Dibujar línea amarilla para interacciones no confirmadas
                cv2.line(frame, interaction['hand'], interaction['object'], (255, 255, 0), 2)
                # Rectángulo amarillo delgado
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        return frame, interactions, confirmed_interaction
    
    def process_video(self, output_path=None, show_video=True, display_width=1920):
        """
        Procesa el video completo
        
        Args:
            output_path: Si se proporciona, guarda el video procesado
            show_video: Si True, muestra el video en tiempo real con cv2.imshow
            display_width: Ancho máximo para visualización (default: 1920 para 1080p)
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {self.video_path}")
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Calcular dimensiones de visualización manteniendo aspect ratio
        if width > display_width:
            scale = display_width / width
            display_height = int(height * scale)
            print(f"Visualización redimensionada a: {display_width}x{display_height}")
        else:
            display_width = width
            display_height = height
        
        # Configurar video writer si es necesario
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
                
                # Procesar frame
                frame_annotated, interactions, confirmed = self.process_frame(frame)
                
                if confirmed:
                    total_interactions += 1
                
                # Mostrar progreso
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Procesando: {frame_count}/{total_frames} frames ({progress:.1f}%)")
                
                # Guardar frame si es necesario
                if writer:
                    writer.write(frame_annotated)
                
                # Mostrar video (redimensionado para visualización)
                if show_video:
                    # Redimensionar solo para visualización, mantener procesamiento en resolución original
                    if frame_annotated.shape[1] > display_width:
                        frame_display = cv2.resize(frame_annotated, (display_width, display_height), 
                                                  interpolation=cv2.INTER_LINEAR)
                    else:
                        frame_display = frame_annotated
                    
                    cv2.imshow('Deteccion Mano-Objeto', frame_display)
                    
                    # Presionar 'q' para salir, 'p' para pausar
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Procesamiento interrumpido por el usuario")
                        break
                    elif key == ord('p'):
                        print("Pausado. Presiona cualquier tecla para continuar...")
                        cv2.waitKey(0)
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_video:
                cv2.destroyAllWindows()
            # MediaPipe 0.10.x no requiere close() explícito
            
            print(f"\nProcesamiento completado:")
            print(f"  - Frames procesados: {frame_count}")
            print(f"  - Interacciones confirmadas: {total_interactions}")
            if output_path:
                print(f"  - Video guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Deteccion de interaccion mano-objeto en videos de supermercado'
    )
    parser.add_argument('video_path', type=str, help='Ruta al video a procesar')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt',
                       help='Modelo YOLO a usar (default: yolov8n-pose.pt para pose, o yolov8n.pt para objetos)')
    parser.add_argument('--threshold', type=int, default=150,
                       help='Umbral de distancia para interaccion en pixeles (default: 150, aumentado para distancias largas)')
    parser.add_argument('--min-frames', type=int, default=3,
                       help='Minimo de frames consecutivos para confirmar interaccion (default: 3)')
    parser.add_argument('--conf-hands', type=float, default=0.3,
                       help='Confianza minima para deteccion de manos (default: 0.3, bajo para distancias largas)')
    parser.add_argument('--conf-objects', type=float, default=0.15,
                       help='Confianza minima para deteccion de objetos (default: 0.15, bajo para objetos pequeños)')
    parser.add_argument('--conf-person', type=float, default=0.5,
                       help='Confianza minima para deteccion de personas (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Ruta para guardar el video procesado (opcional)')
    parser.add_argument('--no-show', action='store_true',
                       help='No mostrar el video en tiempo real')
    parser.add_argument('--display-width', type=int, default=1920,
                       help='Ancho máximo para visualización en píxeles (default: 1920 para 1080p)')
    parser.add_argument('--use-ddshan', action='store_true',
                       help='Usar detector de ddshan (requiere modelo pre-entrenado y compilación CUDA)')
    parser.add_argument('--ddshan-model', type=str, default=None,
                       help='Ruta al modelo de ddshan (si no se especifica, intenta encontrarlo automáticamente)')
    
    args = parser.parse_args()
    
    # Crear detector
    detector = HandObjectDetector(
        video_path=args.video_path,
        yolo_model=args.model,
        interaction_threshold=args.threshold,
        min_frames_interaction=args.min_frames,
        confidence_hands=args.conf_hands,
        confidence_objects=args.conf_objects,
        confidence_person=args.conf_person,
        use_ddshan=args.use_ddshan,
        ddshan_model_path=args.ddshan_model
    )
    
    # Procesar video
    detector.process_video(
        output_path=args.output,
        show_video=not args.no_show,
        display_width=args.display_width
    )


if __name__ == '__main__':
    main()
