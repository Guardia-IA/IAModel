import argparse
import os
import numpy as np
import cv2
import time

def visualize_skeleton(npy_path, fps: float = 20.0):
    # Cargar los datos (Frames, Puntos, 2)
    data = np.load(npy_path)
    print(f"Visualizando: {npy_path}")
    print(f"Dimensiones: {data.shape} (Frames, Puntos, XY)")

    # Conexiones: hombros, codos, muñecas, cadera (sin piernas)
    connections = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7)]

    # Crear una imagen negra para dibujar
    h, w = 600, 600

    # Tiempo entre frames en ms según fps deseados
    delay_ms = int(1000 / fps) if fps > 0 else 50
    
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

        # Salir con 'q' o avanzar según fps
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza poses desde un archivo .npy")
    parser.add_argument("npy_path", type=str, help="Ruta al archivo poses.npy")
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="FPS a los que reproducir la secuencia (por defecto: 20)",
    )
    args = parser.parse_args()

    if os.path.exists(args.npy_path):
        visualize_skeleton(args.npy_path, fps=args.fps)
    else:
        print(f"Archivo no encontrado: {args.npy_path}")
        exit(1)