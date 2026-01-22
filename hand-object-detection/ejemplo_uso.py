"""
Ejemplo simple de uso del detector de interacción mano-objeto
"""

from hand_object_detector import HandObjectDetector

# Ejemplo 1: Uso básico
def ejemplo_basico():
    """Ejemplo más simple - solo pasar el path del video"""
    detector = HandObjectDetector(
        video_path='/home/debian/Vídeos/video1.mp4'
    )
    detector.process_video(show_video=True)

# Ejemplo 2: Ajustado para distancias largas (5 metros)
def ejemplo_distancia_larga():
    """Configuración optimizada para cámaras de supermercado a 5 metros"""
    detector = HandObjectDetector(
        video_path='/home/debian/Vídeos/video1.mp4',
        yolo_model='yolov8m.pt',  # Modelo mediano para mejor precisión
        interaction_threshold=100,  # Ajustar según resolución
        min_frames_interaction=5,  # Más frames para mayor robustez
        confidence_hands=0.4,  # Más permisivo para distancias largas
        confidence_objects=0.25  # Más permisivo para objetos pequeños
    )
    detector.process_video(show_video=True)

# Ejemplo 3: Procesar y guardar resultado
def ejemplo_con_guardado():
    """Procesa el video y guarda el resultado"""
    detector = HandObjectDetector(
        video_path='/home/debian/Vídeos/video1.mp4',
        yolo_model='yolov8n.pt',  # Modelo ligero para velocidad
        interaction_threshold=80,
        min_frames_interaction=3
    )
    detector.process_video(
        output_path='/home/debian/Vídeos/video_procesado.mp4',
        show_video=True
    )

# Ejemplo 4: Procesar frame por frame (para integración)
def ejemplo_frame_por_frame():
    """Procesa frame por frame para integrar con tu sistema"""
    import cv2
    
    detector = HandObjectDetector(
        video_path='/home/debian/Vídeos/video1.mp4',
        yolo_model='yolov8n.pt'
    )
    
    cap = cv2.VideoCapture('/home/debian/Vídeos/video1.mp4')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar frame
        frame_annotated, interactions, confirmed = detector.process_frame(frame)
        
        # Aquí puedes integrar con tu modelo de pose tracking
        # if confirmed:
        #     print("Interacción confirmada - posible robo")
        
        cv2.imshow('Frame', frame_annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Descomenta el ejemplo que quieras usar:
    
    # ejemplo_basico()
    # ejemplo_distancia_larga()
    # ejemplo_con_guardado()
    ejemplo_frame_por_frame()
