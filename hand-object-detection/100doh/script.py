import cv2
import os
import sys
from inference_sdk import InferenceHTTPClient
import supervision as sv

# --- CONFIGURACIÓN ---
# ID del modelo encontrado en Roboflow Universe. 
# Se puede cambiar por 'osmando/hands-sghts' si se busca más especificidad.
MODEL_ID = "hand-holding-and-empty/1" 
# Es necesario obtener una API Key gratuita en app.roboflow.com
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "rf_Md9BDXI2teYlgt1o28rzsuYiG2Z2")

def main(video_path=None):
    # 1. Inicializar el Cliente de Inferencia
    # Si se ejecuta un servidor local Docker (recomendado para baja latencia),
    # cambiar api_url a "http://localhost:9001".
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=API_KEY
    )

    # 2. Configurar la fuente de vídeo (archivo MP4)
    if video_path is None:
        # Si no se proporciona como argumento, intentar usar un archivo por defecto
        video_path = "video.mp4"  # Cambia esto por la ruta de tu video
    
    if not os.path.exists(video_path):
        print(f"Error: No se encontró el archivo de video: {video_path}")
        print("Uso: python script.py [ruta_al_video.mp4]")
        return
    
    print(f"Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video: {video_path}")
        return
    
    # Obtener información del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames totales")
    
    # Crear nombre del archivo de salida
    video_dir = os.path.dirname(video_path) if os.path.dirname(video_path) else "."
    video_name = os.path.basename(video_path)
    video_name_no_ext, video_ext = os.path.splitext(video_name)
    output_video_path = os.path.join(video_dir, f"{video_name_no_ext}_100doh{video_ext}")
    
    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print(f"Guardando video procesado en: {output_video_path}\n")
    
    # 3. Configurar anotadores de Supervision para visualización profesional
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.ColorPalette.DEFAULT
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=10
    )

    print(f"Iniciando inferencia en modelo: {MODEL_ID}...")
    print("Presiona 'q' para salir o cerrar la ventana\n")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\nFin del video alcanzado.")
            break
        
        frame_count += 1

        # Controlar velocidad de reproducción según FPS del video
        delay = max(1, int(1000 / fps)) if fps > 0 else 1

        # 4. Inferencia
        # La API acepta numpy arrays directamente en versiones recientes,
        # o rutas de archivos temporales. Para rendimiento óptimo en producción
        # se recomienda usar el servidor local de Roboflow (Docker).
        
        # Guardamos frame temporalmente para enviar a la API (método compatible universal)
        # En despliegue real, usar inferencia local para evitar latencia de red.
        temp_file = "temp_frame.jpg"
        cv2.imwrite(temp_file, frame)

        annotated_frame = frame.copy()  # Frame por defecto si hay error
        try:
            # Llamada a la API
            result = client.infer(temp_file, model_id=MODEL_ID)
            
            # 5. Convertir respuesta JSON a objeto Detections de Supervision
            detections = sv.Detections.from_inference(result)

            # Filtrar por confianza si es necesario
            if len(detections) > 0:
                detections = detections[detections.confidence > 0.5]

            # 6. Lógica de Negocio: Colorear según estado
            # Aquí personalizamos las etiquetas basadas en la clase detectada
            labels = []
            if len(detections) > 0:
                # Obtener los nombres de las clases de las detecciones
                # La estructura puede variar según la versión de supervision
                try:
                    # Intentar obtener class_name del resultado de inferencia
                    if isinstance(result, dict) and 'predictions' in result:
                        predictions = result['predictions']
                        class_names = [pred.get('class', pred.get('class_name', 'Unknown')) for pred in predictions]
                    elif hasattr(detections, 'data') and isinstance(detections.data, dict):
                        class_names = detections.data.get('class_name', [])
                    else:
                        # Si no hay class_name disponible, usar class_id
                        class_names = [f"Class_{id}" for id in detections.class_id] if hasattr(detections, 'class_id') else []
                    
                    # Crear etiquetas para cada detección
                    for i, confidence in enumerate(detections.confidence):
                        if i < len(class_names):
                            class_name = str(class_names[i])
                            state = "INTERACCIÓN" if "holding" in class_name.lower() else "LIBRE"
                            labels.append(f"{state} ({confidence:.2f})")
                        else:
                            labels.append(f"Objeto ({confidence:.2f})")
                except Exception as e:
                    # Si hay error al obtener class_names, usar confianza solamente
                    print(f"Advertencia al procesar etiquetas: {e}")
                    labels = [f"Conf: {conf:.2f}" for conf in detections.confidence]

            # 7. Anotar la imagen
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

            # Agregar información del frame en la imagen
            frame_info = f"Frame: {frame_count}/{total_frames} | FPS: {fps}"
            cv2.putText(annotated_frame, frame_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            error_str = str(e)
            # Detectar errores específicos de la API
            if "403" in error_str or "Forbidden" in error_str:
                print(f"Error 403 - Acceso denegado (frame {frame_count}):")
                print("  - Verifica que tu API_KEY sea válida")
                print("  - Verifica que tengas acceso al modelo: " + MODEL_ID)
                print("  - Puede que necesites suscribirte o verificar permisos en Roboflow")
            else:
                print(f"Error en inferencia (frame {frame_count}): {e}")
            
            # Si hay error, mostrar el frame original con mensaje de error
            annotated_frame = frame.copy()
            error_msg = f"Error API - Frame: {frame_count}/{total_frames}"
            cv2.putText(annotated_frame, error_msg, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Guardar frame procesado en el video de salida
        out.write(annotated_frame)
        
        # Mostrar resultado (tanto si hubo éxito como error)
        cv2.imshow("Detección de Estado de Mano (100DOH)", annotated_frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q') or key == 27:  # 'q' o ESC para salir
            break

    cap.release()
    out.release()  # Cerrar el escritor de video
    cv2.destroyAllWindows()
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")
    
    print(f"\n✓ Video procesado guardado exitosamente en: {output_video_path}")

if __name__ == "__main__":
    if API_KEY == "TU_API_KEY_AQUI":
        print("ALERTA: Debes configurar tu ROBOFLOW_API_KEY.")
        print("Puedes hacerlo con: export ROBOFLOW_API_KEY='tu_api_key'")
    else:
        # Obtener la ruta del video desde los argumentos de línea de comandos
        video_path = sys.argv[1] if len(sys.argv) > 1 else None
        main(video_path)