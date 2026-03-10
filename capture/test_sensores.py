from pykinect2 import PyKinectV2, PyKinectRuntime
import time

# Inicializamos con los tres flujos
kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Color | 
    PyKinectV2.FrameSourceTypes_Depth | 
    PyKinectV2.FrameSourceTypes_Infrared
)

print("--- DIAGNÓSTICO DE SENSORES KINECT V2 ---")
print("Presiona Ctrl+C para detener\n")

try:
    while True:
        # Verificamos disponibilidad de frames
        has_color = kinect.has_new_color_frame()
        has_depth = kinect.has_new_depth_frame()
        has_ir = kinect.has_new_infrared_frame()

        # Creamos el mensaje de estado
        status = []
        status.append("[COLOR: OK]" if has_color else "[COLOR: None]")
        status.append("[DEPTH: OK]" if has_depth else "[DEPTH: None]")
        status.append("[INFRARED: OK]" if has_ir else "[IR: None]")

        # Imprimimos en la misma línea para ver el parpadeo
        print(f"\r{' | '.join(status)}", end="", flush=True)
        
        # Si detectamos un frame, lo "limpiamos" para que el sensor pida el siguiente
        if has_color: kinect.get_last_color_frame()
        if has_depth: kinect.get_last_depth_frame()
        if has_ir: kinect.get_last_infrared_frame()

        time.sleep(0.1) # Pausa mínima para no saturar el CPU
except KeyboardInterrupt:
    kinect.close()
    print("\n\nPrueba finalizada.")