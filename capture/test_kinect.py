from pykinect2 import PyKinectV2, PyKinectRuntime
import cv2
import numpy as np

# Inicializar Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

print("Buscando señal del Kinect... Presiona 'q' en la ventana de video para salir.")

while True:
    if kinect.has_new_color_frame():
        # Obtener frame
        frame = kinect.get_last_color_frame()
        # Darle formato (1920x1080, 4 canales BGRA)
        img = frame.reshape((1080, 1920, 4)).astype(np.uint8)
        # Redimensionar para que quepa en la pantalla
        img_small = cv2.resize(img, (960, 540))
        
        # MOSTRAR VENTANA
        cv2.imshow('PRUEBA DE CONEXION', img_small)

    # El número 1 indica que espera 1ms. 
    # ESTA LÍNEA ES LA QUE PERMITE QUE LA VENTANA SE DIBUJE
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()