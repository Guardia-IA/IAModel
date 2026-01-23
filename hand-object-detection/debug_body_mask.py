"""
Script para detectar persona, extraer contorno del cuerpo y pintarlo de negro
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse


def process_video(video_path, display_width=1920):
    """Procesa el video detectando personas y pintando el cuerpo de negro"""
    
    # Cargar YOLO pose para detectar personas y esqueleto
    print("Cargando YOLO pose...")
    yolo_pose = YOLO('yolov8n-pose.pt')
    
    # También cargar YOLO seg para segmentación del cuerpo
    print("Cargando YOLO seg...")
    yolo_seg = YOLO('yolov8n-seg.pt')
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    print("Procesando...\n")
    
    if width > display_width:
        scale = display_width / width
        display_height = int(height * scale)
    else:
        display_width = width
        display_height = height
    
    frame_count = 0
    
    # Conexiones del esqueleto para dibujar
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_original = frame.copy()
            
            # Detectar personas con YOLO pose
            pose_results = yolo_pose(frame, conf=0.3, verbose=False)
            
            # Detectar personas con YOLO seg para obtener máscaras
            seg_results = yolo_seg(frame, conf=0.3, verbose=False)
            
            # Procesar cada persona detectada
            if pose_results[0].boxes is not None:
                for i, box in enumerate(pose_results[0].boxes):
                    cls = int(box.cls[0].cpu().numpy())
                    if cls == 0:  # Persona
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # Dibujar bounding box de la persona
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                        
                        # Dibujar esqueleto
                        if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > i:
                            keypoints = pose_results[0].keypoints[i].xy[0].cpu().numpy()
                            confidences = pose_results[0].keypoints[i].conf[0].cpu().numpy() if hasattr(pose_results[0].keypoints[i], 'conf') else None
                            
                            # Dibujar conexiones del esqueleto
                            for connection in POSE_CONNECTIONS:
                                pt1_idx, pt2_idx = connection
                                if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                                    pt1 = tuple(map(int, keypoints[pt1_idx]))
                                    pt2 = tuple(map(int, keypoints[pt2_idx]))
                                    if confidences is None or (confidences[pt1_idx] > 0.3 and confidences[pt2_idx] > 0.3):
                                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                            
                            # Dibujar keypoints
                            for j, kp in enumerate(keypoints):
                                if (confidences is None or confidences[j] > 0.3) and kp[0] > 0 and kp[1] > 0:
                                    cv2.circle(frame, tuple(map(int, kp)), 5, (0, 255, 0), -1)
            
            # Crear frame para mostrar lo que está fuera del cuerpo (dentro del bbox)
            frame_outside_body = frame_original.copy()
            
            # Procesar segmentación para obtener máscaras del cuerpo
            if seg_results[0].masks is not None and pose_results[0].boxes is not None:
                # Obtener máscaras de todas las personas
                masks = seg_results[0].masks.data.cpu().numpy()
                boxes = pose_results[0].boxes
                
                # Procesar cada persona
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0].cpu().numpy())
                    if cls == 0:  # Persona
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Asegurar que las coordenadas están dentro del frame
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        
                        # Obtener la máscara correspondiente a esta persona
                        if i < len(masks):
                            mask = masks[i]
                            # Redimensionar máscara al tamaño del frame
                            mask_resized = cv2.resize(mask, (width, height))
                            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                            
                            # Crear máscara solo para el bounding box
                            bbox_mask = np.zeros((height, width), dtype=np.uint8)
                            bbox_mask[y1:y2, x1:x2] = 255
                            
                            # Máscara del cuerpo dentro del bbox
                            body_mask_in_bbox = cv2.bitwise_and(mask_binary, bbox_mask)
                            
                            # Máscara inversa: todo lo que está en el bbox pero NO es cuerpo
                            outside_body_mask = cv2.bitwise_and(bbox_mask, cv2.bitwise_not(body_mask_in_bbox))
                            
                            # Pintar de rojo todo lo que está fuera del cuerpo (dentro del bbox)
                            frame_outside_body[outside_body_mask > 0] = [0, 0, 255]  # Rojo en BGR
                            
                            # Dibujar contornos del cuerpo
                            contours_body, _ = cv2.findContours(body_mask_in_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(frame, contours_body, -1, (0, 255, 0), 2)  # Verde para contorno del cuerpo
                            
                            # Dibujar contornos de lo que está fuera del cuerpo
                            contours_outside, _ = cv2.findContours(outside_body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(frame, contours_outside, -1, (0, 0, 255), 2)  # Rojo para contornos fuera del cuerpo
                            
                            # Información de debug
                            area_body = np.sum(body_mask_in_bbox > 0)
                            area_outside = np.sum(outside_body_mask > 0)
                            area_bbox = (x2 - x1) * (y2 - y1)
                            
                            cv2.putText(frame, f'Body area: {area_body}, Outside: {area_outside}, BBox: {area_bbox}', 
                                       (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar información
            cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Mostrar ambos frames lado a lado
            if seg_results[0].masks is not None and pose_results[0].boxes is not None:
                frame_combined = np.hstack([frame, frame_outside_body])
            else:
                frame_combined = frame
                cv2.putText(frame_combined, f'Frame: {frame_count}/{total_frames} - Sin segmentacion', (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Procesando: {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Redimensionar para visualización
            if frame_combined.shape[1] > display_width:
                frame_display = cv2.resize(frame_combined, (display_width, display_height))
            else:
                frame_display = frame_combined
            
            cv2.imshow('Body Detection and Masking', frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcesamiento completado: {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description='Detectar persona y pintar cuerpo de negro')
    parser.add_argument('video_path', type=str, help='Ruta al video')
    parser.add_argument('--display-width', type=int, default=1920, help='Ancho de visualización')
    
    args = parser.parse_args()
    
    process_video(args.video_path, args.display_width)


if __name__ == '__main__':
    main()
