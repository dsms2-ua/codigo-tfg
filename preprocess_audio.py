import os
import numpy as np
from tqdm import tqdm
# Importamos la función de tu repositorio para asegurar consistencia
from my_utils.preprocessing import preprocess_audio

# --- CONFIGURACIÓN ---
# Carpeta donde tienes el dataset original (con los .wav)
ORIGEN = "/home/davidetsm/Escritorio/TFG/real-a2s/real_a2s_sax_dataset" 
# Carpeta nueva donde se guardarán los procesados
DESTINO = "/home/davidetsm/Escritorio/TFG/real-a2s/dataset_processed"

def procesar_dataset():
    # 1. Crear lista de todos los archivos .wav
    archivos_audio = []
    for root, dirs, files in os.walk(ORIGEN):
        # Ignoramos la carpeta de etiquetas 'krn' si está dentro de 'dataset'
        if 'krn' in root:
            continue
        for f in files:
            if f.endswith(".wav"):
                archivos_audio.append(os.path.join(root, f))

    if not archivos_audio:
        print(f"❌ No se encontraron archivos .wav en {ORIGEN}")
        return

    print(f"Se han encontrado {len(archivos_audio)} archivos. Iniciando proceso...")

    # 2. Bucle de procesamiento
    for path_wav in tqdm(archivos_audio, desc="Procesando"):
        try:
            # Calcular la ruta relativa para mantener la estructura (ej: real/tenor/archivo.wav)
            ruta_relativa = os.path.relpath(path_wav, ORIGEN)
            
            # Definir la ruta de salida cambiando .wav por .npy
            path_npy = os.path.join(DESTINO, ruta_relativa).replace(".wav", ".npy")
            
            # Crear las carpetas necesarias si no existen (ej: real/tenor/)
            os.makedirs(os.path.dirname(path_npy), exist_ok=True)

            # 3. Preprocesar usando madmom (vía tu código oficial)
            # Usamos training=False para obtener solo el espectrograma
            espectrograma = preprocess_audio(path_wav, training=False, width_reduction=1)

            # 4. Guardar en formato binario de numpy
            np.save(path_npy, espectrograma)

        except Exception as e:
            print(f"\n[!] Error en {path_wav}: {e}")

if __name__ == "__main__":
    procesar_dataset()
    print(f"\n✅ ¡Listo! Dataset procesado en: {DESTINO}")