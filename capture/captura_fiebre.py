import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pykinect2 import PyKinectV2, PyKinectRuntime
import numpy as np
import cv2
import mediapipe as mp
from datetime import datetime
from config import (KINECT_COLOR_WIDTH, KINECT_COLOR_HEIGHT,
                    KINECT_IR_WIDTH,    KINECT_IR_HEIGHT)

# ── Configuración ─────────────────────────────────────
DATASET_DIR        = 'dataset'
CALIBRATION_FRAMES = 30
CLASE              = 'fever'

# Umbral IR que indica calor suficiente para capturar
# (evita capturar si la compresa aún no calentó bien)
IR_MIN_MEAN        = 140   # de 0-255, ajustable

# Crear carpetas
for split in ['train', 'val']:
    os.makedirs(f'{DATASET_DIR}/{split}/{CLASE}', exist_ok=True)

counters = {'train': 0, 'val': 0}

# ── Helpers IR ────────────────────────────────────────
baseline  = None
calib_buf = []
calibrado = False

def agregar_calibracion(ir_frame):
    global baseline, calibrado
    calib_buf.append(ir_frame.astype(np.float32))
    if len(calib_buf) >= CALIBRATION_FRAMES:
        baseline  = np.mean(calib_buf, axis=0)
        calibrado = True
        print('\n✅ IR Calibrado. ¡Listo para capturar!')

def procesar_ir(ir_frame, depth_frame=None):
    frame = ir_frame.astype(np.float32)
    if depth_frame is not None:
        d     = np.clip(depth_frame / 1000.0, 0.3, 8.0)
        frame = frame / np.power(d, 3.41)
    if calibrado:
        frame = np.clip(frame - baseline, 0, None)
    p_lo  = np.percentile(frame, 1)
    p_hi  = np.percentile(frame, 99)
    frame = np.clip((frame - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)
    return (frame * 255).astype(np.uint8)

def estimar_temp(ir_roi):
    mean_ir = float(np.mean(ir_roi))
    return round(35.0 + (mean_ir / 255.0) * 4.5, 1)

def calor_suficiente(ir_roi):
    """Verifica que la ROI tiene suficiente calor para ser clase fiebre."""
    return float(np.mean(ir_roi)) >= IR_MIN_MEAN

def guardar_par(rgb_roi, ir_roi, split):
    ts    = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    fname = f'{DATASET_DIR}/{split}/{CLASE}/{ts}'
    cv2.imwrite(f'{fname}_rgb.jpg', cv2.resize(
        cv2.cvtColor(rgb_roi, cv2.COLOR_BGRA2BGR), (224, 224)))
    cv2.imwrite(f'{fname}_ir.jpg',  cv2.resize(ir_roi, (224, 224)))
    counters[split] += 1
    print(f'  🔥 [{CLASE.upper()} / {split.upper()}] '
          f'Guardado #{counters[split]} '
          f'| Temp: {estimar_temp(ir_roi)}°C')

# ── Inicializar ───────────────────────────────────────
print('\n' + '='*50)
print('   CAPTURA CLASE: FIEBRE (simulada)')
print('='*50)
print('Inicializando Kinect v2...')

kinect   = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Color   |
    PyKinectV2.FrameSourceTypes_Depth   |
    PyKinectV2.FrameSourceTypes_Infrared
)
mp_face  = mp.solutions.face_detection
face_det = mp_face.FaceDetection(
    min_detection_confidence=0.75, model_selection=0)

print('✅ Kinect listo.')
print('\n📋 INSTRUCCIONES:')
print('  • Aplica compresa tibia (40°C) o taza de agua caliente')
print('    cerca del rostro durante 2-3 minutos ANTES de capturar')
print('  • El indicador de calor debe estar en VERDE para capturar')
print('  • Mantén el encuadre libre al inicio (calibración IR)')
print('\nControles:')
print('  [ESPACIO] Capturar   [T] Split Train')
print('  [V] Split Val        [Q] Salir\n')

split = 'train'

# ── Bucle ─────────────────────────────────────────────
while True:
    color_frame = depth_frame = ir_frame = None

    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame().reshape(
            (KINECT_COLOR_HEIGHT, KINECT_COLOR_WIDTH, 4)).astype(np.uint8)
    if kinect.has_new_depth_frame():
        depth_frame = kinect.get_last_depth_frame().reshape(
            (KINECT_IR_HEIGHT, KINECT_IR_WIDTH)).astype(np.float32)
    if kinect.has_new_infrared_frame():
        ir_frame = kinect.get_last_infrared_frame().reshape(
            (KINECT_IR_HEIGHT, KINECT_IR_WIDTH)).astype(np.float32)

    if color_frame is None or ir_frame is None:
        continue

    # Calibración
    if not calibrado:
        agregar_calibracion(ir_frame)
        prog = int(len(calib_buf) / CALIBRATION_FRAMES * 100)
        print(f'\r  Calibrando IR: {prog}%  ', end='', flush=True)
        continue

    # Procesar IR
    ir_proc    = procesar_ir(ir_frame, depth_frame)
    ir_colored = cv2.applyColorMap(ir_proc, cv2.COLORMAP_INFERNO)
    ir_show    = cv2.resize(ir_colored, (320, 240))

    # Detección facial
    rgb_for_mp = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2RGB)
    results    = face_det.process(rgb_for_mp)
    display    = cv2.resize(
        cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR), (960, 540))

    rois_rgb, rois_ir, calor_ok = [], [], []

    if results.detections:
        H0, W0 = color_frame.shape[:2]
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x  = int(bb.xmin * W0);  y  = int(bb.ymin * H0)
            w  = int(bb.width * W0); h  = int(bb.height * H0)
            if w <= 0 or h <= 0:
                continue

            pad     = 30
            roi_rgb = color_frame[
                max(0,y-pad):min(H0,y+h+pad),
                max(0,x-pad):min(W0,x+w+pad)]

            sx, sy  = 512/W0, 424/H0
            xi, yi  = int(x*sx), int(y*sy)
            wi, hi  = int(w*sx), int(h*sy)
            p2      = 15
            roi_ir  = ir_proc[
                max(0,yi-p2):min(424,yi+hi+p2),
                max(0,xi-p2):min(512,xi+wi+p2)]

            if roi_rgb.size > 0 and roi_ir.size > 0:
                rois_rgb.append(roi_rgb)
                rois_ir.append(roi_ir)
                temp   = estimar_temp(roi_ir)
                ok     = calor_suficiente(roi_ir)
                calor_ok.append(ok)

                # Bbox — verde si calor ok, naranja si insuficiente
                color_bbox = (0, 200, 0) if ok else (0, 140, 255)
                sx_d, sy_d = 960/W0, 540/H0
                xd = int(x*sx_d); yd = int(y*sy_d)
                wd = int(w*sx_d); hd = int(h*sy_d)
                cv2.rectangle(display,(xd,yd),(xd+wd,yd+hd), color_bbox, 2)

                # Temperatura + indicador calor
                estado_calor = '🔥 Listo' if ok else '⏳ Calienta...'
                cv2.putText(display,
                            f'{temp}C  {estado_calor}',
                            (xd+5, yd-8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color_bbox, 2)

    # Indicador global de calor
    todo_listo = len(calor_ok) > 0 and all(calor_ok)
    ind_color  = (0, 200, 0) if todo_listo else (0, 140, 255)
    ind_texto  = '🔥 CALOR SUFICIENTE — puede capturar' \
                  if todo_listo else \
                  '⏳ Esperando calor... aplica compresa'

    # Panel UI
    H, W = display.shape[:2]
    cv2.rectangle(display, (0,0), (W,70), (20,20,20), -1)
    cv2.putText(display, f'MODO: FIEBRE SIMULADA  |  Split: {split.upper()}',
                (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(display,
                f'Train: {counters["train"]}   Val: {counters["val"]}   '
                f'Rostros: {len(rois_rgb)}',
                (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(display, ind_texto,
                (10,68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ind_color, 1)
    cv2.rectangle(display, (0,H-40), (W,H), (20,20,20), -1)
    cv2.putText(display,
                '[ESPACIO] Capturar  [T] Train  [V] Val  [Q] Salir',
                (10,H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow('Captura FIEBRE — RGB', display)
    cv2.imshow('Captura FIEBRE — IR',  ir_show)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('t'):
        split = 'train'
        print(f'\n▶ Split: TRAIN')
    elif key == ord('v'):
        split = 'val'
        print(f'\n▶ Split: VAL')
    elif key == ord(' '):
        if len(rois_rgb) == 0:
            print('\n⚠️  No se detectó rostro.')
        elif not all(calor_ok):
            print('\n⚠️  Calor insuficiente. Espera a que el indicador sea VERDE.')
        else:
            for rgb_roi, ir_roi in zip(rois_rgb, rois_ir):
                guardar_par(rgb_roi, ir_roi, split)

# Cierre
kinect.close()
cv2.destroyAllWindows()
print('\n' + '='*50)
print(f'  TOTAL FIEBRE — Train: {counters["train"]} | Val: {counters["val"]}')
print('='*50)