import os
import zipfile
import argparse
import pandas as pd

def comprimir_carpetas_aleatorias(csv_path, destino_dir, zip_name, num_filas=50):
    # 1. Leer el archivo CSV
    if not os.path.exists(csv_path):
        print(f"Error: El archivo CSV '{csv_path}' no existe.")
        return
        
    df = pd.read_csv(csv_path)

    # 2. Seleccionar filas aleatorias de la primera columna
    num_filas_a_seleccionar = min(num_filas, len(df))
    df_aleatorio = df.sample(n=num_filas_a_seleccionar)

    # .iloc[:, 0] selecciona la primera columna sin importar su nombre
    rutas_carpetas = df_aleatorio.iloc[:, 0].tolist()

    # 3. Asegurarse de que el directorio de destino exista
    os.makedirs(destino_dir, exist_ok=True)
    
    # Asegurar que el nombre termine en .zip
    if not zip_name.endswith('.zip'):
        zip_name += '.zip'
        
    ruta_completa_zip = os.path.join(destino_dir, zip_name)

    # 4. Crear el archivo ZIP y comprimir las carpetas
    print(f"Iniciando la compresión de {len(rutas_carpetas)} carpetas aleatorias...")

    count_exito = 0
    with zipfile.ZipFile(ruta_completa_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for ruta_carpeta in rutas_carpetas:
            ruta_carpeta = str(ruta_carpeta).strip()
            
            if os.path.exists(ruta_carpeta) and os.path.isdir(ruta_carpeta):
                print(f"Comprimiendo: {ruta_carpeta}")
                count_exito += 1
                
                for raiz, carpetas, archivos in os.walk(ruta_carpeta):
                    for archivo in archivos:
                        ruta_completa_archivo = os.path.join(raiz, archivo)
                        # Mantener estructura interna relativa de la carpeta
                        ruta_relativa_en_zip = os.path.relpath(ruta_completa_archivo, os.path.dirname(ruta_carpeta))
                        zipf.write(ruta_completa_archivo, arcname=ruta_relativa_en_zip)
            else:
                print(f"Advertencia: Se saltó '{ruta_carpeta}' (no existe o no es una carpeta válida).")

    print(f"\n¡Proceso completado!")
    print(f"Se procesaron con éxito {count_exito} de {len(rutas_carpetas)} carpetas.")
    print(f"Archivo ZIP guardado en: {ruta_completa_zip}")

if __name__ == "__main__":
    # Configuración de los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Selecciona carpetas aleatorias de un CSV y las comprime en un archivo ZIP.")
    
    # Argumentos obligatorios (posicionales)
    parser.add_argument("csv", help="Ruta al archivo CSV que contiene las rutas de las carpetas.")
    parser.add_argument("destino", help="Directorio donde se guardará el archivo ZIP resultante.")
    parser.add_argument("nombre_zip", help="Nombre que tendrá el archivo ZIP (ej. mis_carpetas.zip).")
    
    # Argumento opcional (por si en el futuro quieres cambiar el número de 50 a otro número)
    parser.add_argument("-n", "--numero", type=int, default=50, help="Número de carpetas aleatorias a elegir (por defecto: 50).")

    # Parsear los argumentos pasados por consola
    args = parser.parse_args()

    # Ejecutar la función principal con los argumentos recibidos
    comprimir_carpetas_aleatorias(
        csv_path=args.csv, 
        destino_dir=args.destino, 
        zip_name=args.nombre_zip,
        num_filas=args.numero
    )

#python comprimir.py "datos.csv" "/ruta/del/directorio/destino" "archivo_final.zip"