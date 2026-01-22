import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import argparse

# Inicializar modelos pre-entrenados
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
yolo_model = YOLO('yolov8n.pt')  # Modelo pre-entrenado en COCO (nano para velocidad)

# Función para calcular IOU entre dos bounding boxes [x1,y1,x2,y2]
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Función principal para procesar vídeo
def process_video(video_path, output_path='output.mp4'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error abriendo el vídeo.")
        return

    # Obtener propiedades del vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    interaction_count = 0
    consecutive_overlap = 0
    iou_threshold = 0.3  # Ajusta para sensibilidad (más alto = menos falsos positivos)
    min_consecutive = 5  # Frames seguidos para confirmar "coger"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar manos con MediaPipe
        hand_results = hands.process(rgb_frame)
        hand_boxes = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Calcular bounding box de la mano
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * width
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * height
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * width
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * height
                hand_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                # Dibujar landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Detectar objetos con YOLO (excluir 'person' para enfocarnos en objetos)
        yolo_results = yolo_model(frame, verbose=False)
        object_boxes = []
        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != 0:  # 0 es 'person' en COCO, ignora
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    object_boxes.append([x1, y1, x2, y2])
                    # Dibujar box de objeto
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Chequear interacciones (IOU entre manos y objetos)
        has_overlap = False
        for h_box in hand_boxes:
            cv2.rectangle(frame, (h_box[0], h_box[1]), (h_box[2], h_box[3]), (255, 0, 0), 2)
            for o_box in object_boxes:
                iou = calculate_iou(h_box, o_box)
                if iou > iou_threshold:
                    has_overlap = True
                    cv2.putText(frame, f'IOU: {iou:.2f}', (h_box[0], h_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    break

        if has_overlap:
            consecutive_overlap += 1
            if consecutive_overlap >= min_consecutive:
                interaction_count += 1
                cv2.putText(frame, 'Posible COGER detectado!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f'Frame {frame_count}: Posible interaccion mano-objeto (IOU > {iou_threshold})')
        else:
            consecutive_overlap = 0

        # Escribir frame anotado
        out.write(frame)

    cap.release()
    out.release()
    print(f'Procesado completado. Detecciones de "coger": {interaction_count}. Vídeo de salida: {output_path}')

# Argumentos de línea de comandos
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detectar interacciones mano-objeto en vídeo')
    parser.add_argument('--video', type=str, required=True, help='Ruta al clip de vídeo de entrada')
    args = parser.parse_args()
    process_video(args.video)