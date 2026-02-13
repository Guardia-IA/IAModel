import numpy as np
import cv2
import time

def visualize_skeleton(npy_path):
    # Cargar los datos (Frames, Puntos, 2)
    data = np.load(npy_path)
    print(f"Visualizando: {npy_path}")
    print(f"Dimensiones: {data.shape} (Frames, Puntos, XY)")

    # Definir conexiones para el esqueleto (basado en los 8 puntos que guardamos)
    # Puntos guardados (KEEP_KPS): 5, 6, 7, 8, 9, 10, 11, 12
    # En el array .npy, estos son los índices 0, 1, 2, 3, 4, 5, 6, 7
    # Conexiones: (0-1 hombros), (0-2 hombro-codo izq), (2-4 codo-muñeca izq), etc.
    connections = [
        (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), # Torso superior y brazos
        (0, 6), (1, 7), (6, 7)                  # Torso a cadera y cadera
    ]

    # Crear una imagen negra para dibujar
    h, w = 600, 600
    
    for frame_idx in range(len(data)):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        points = data[frame_idx]

        # Dibujar conexiones
        for start, end in connections:
            pt1 = (int(points[start][0] * w), int(points[start][1] * h))
            pt2 = (int(points[end][0] * w), int(points[end][1] * h))
            cv2.line(canvas, pt1, pt2, (255, 255, 0), 2)

        # Dibujar puntos
        for pt in points:
            x, y = int(pt[0] * w), int(pt[1] * h)
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)

        cv2.putText(canvas, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Verificador de Esqueletos", canvas)
        
        # Salir con 'q' o esperar entre frames
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Cambia esto por la ruta de un archivo generado
    RUTA_EJEMPLO = "dataset_final_limpio/2/video_ejemplo_p1/poses.npy"
    if Path(RUTA_EJEMPLO).exists():
        visualize_skeleton(RUTA_EJEMPLO)
    else:
        print("Primero genera los datos con el script anterior.")