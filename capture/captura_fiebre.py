import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parche Python 3.8
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

from pykinect2 import PyKinectV2, PyKinectRuntime
import numpy as np
import cv2
import mediapipe as mp
from datetime import datetime
import winsound

# ── Dimensiones ───────────────────────────────────────
K_W, K_H   = 1920, 1080
IR_W, IR_H = 512, 424

# ── Dataset ───────────────────────────────────────────
DATASET_DIR = 'dataset'
CLASE       = 'fever'
IR_MIN_MEAN = 140   # Umbral mínimo IR para clase fiebre

for s in ['train', 'val']:
    os.makedirs(f'{DATASET_DIR}/{s}/{CLASE}', exist_ok=True)

counters    = {'train': 0, 'val': 0}
split       = 'train'
flash_timer = 0

# ── Inicializar Kinect ────────────────────────────────
print('Inicializando Kinect v2...')
try:
    kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color    |
        PyKinectV2.FrameSourceTypes_Infrared |
        PyKinectV2.FrameSourceTypes_Depth
    )
    print('✅ Kinect inicializado.')
except Exception as e:
    print(f'❌ Error Kinect: {e}')
    input('Presiona Enter para salir...')
    sys.exit()

# ── MediaPipe ─────────────────────────────────────────
mp_face  = mp.solutions.face_detection
face_det = mp_face.FaceDetection(
    min_detection_confidence=0.5, model_selection=0)

print('\nControles: [ESPACIO] Capturar | [T] Train | [V] Val | [Q] Salir\n')
print('⚠️  Aplica compresa tibia 2-3 min ANTES de capturar\n')

# ── Buffers persistentes ──────────────────────────────
last_ir    = None
last_depth = None
last_color = None

# ── Esperar sensor IR con ventana visible ─────────────
print('Esperando sensor IR...')
timeout = 0
while last_ir is None:
    if kinect.has_new_infrared_frame():
        last_ir = kinect.get_last_infrared_frame().reshape(
            (IR_H, IR_W)).astype(np.float32)
    if kinect.has_new_color_frame():
        last_color = kinect.get_last_color_frame().reshape(
            (K_H, K_W, 4)).astype(np.uint8)

    if last_color is not None:
        waiting = cv2.resize(
            cv2.cvtColor(last_color, cv2.COLOR_BGRA2BGR), (640, 360))
        cv2.rectangle(waiting, (0, 0), (640, 40), (20, 20, 20), -1)
        cv2.putText(waiting,
                    'Iniciando sensor IR... espera unos segundos',
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.imshow('Captura FIEBRE — Kinect v2', waiting)

    cv2.waitKey(1)
    timeout += 1
    if timeout > 300:
        print('⚠️  IR tardando, continuando sin esperar.')
        break

print('✅ Sensor IR listo. ¡Puedes capturar!')

# ── Bucle principal ───────────────────────────────────
while True:

    if kinect.has_new_color_frame():
        last_color = kinect.get_last_color_frame().reshape(
            (K_H, K_W, 4)).astype(np.uint8)
    if kinect.has_new_infrared_frame():
        last_ir = kinect.get_last_infrared_frame().reshape(
            (IR_H, IR_W)).astype(np.float32)
    if kinect.has_new_depth_frame():
        last_depth = kinect.get_last_depth_frame().reshape(
            (IR_H, IR_W)).astype(np.float32)

    if last_color is None:
        continue

    display = cv2.resize(
        cv2.cvtColor(last_color, cv2.COLOR_BGRA2BGR), (640, 360))

    # ── Procesar IR y verificar calor ─────────────────
    calor_ok = False
    ir_mean  = 0
    if last_ir is not None:
        ir_norm  = (np.clip(last_ir / 65535.0, 0, 1) * 255).astype(np.uint8)
        ir_mini  = cv2.resize(
            cv2.applyColorMap(ir_norm, cv2.COLORMAP_INFERNO), (150, 140))
        display[210:350, 480:630] = ir_mini
        ir_mean  = float(np.mean(ir_norm))
        calor_ok = ir_mean >= IR_MIN_MEAN
        ir_texto = f'IR: {ir_mean:.0f}/255'
        ir_color = (0, 220, 0) if calor_ok else (0, 140, 255)
    else:
        ir_texto = 'IR no disponible'
        ir_color = (0, 0, 255)

    # ── Detección facial ──────────────────────────────
    small_rgb = cv2.cvtColor(
        cv2.resize(last_color, (640, 360)), cv2.COLOR_BGRA2RGB)
    res    = face_det.process(small_rgb)
    rostros = 0

    if res.detections:
        rostros = len(res.detections)
        for det in res.detections:
            b      = det.location_data.relative_bounding_box
            x1     = int(b.xmin * 640)
            y1     = int(b.ymin * 360)
            x2     = int((b.xmin + b.width)  * 640)
            y2     = int((b.ymin + b.height) * 360)
            color_bbox = (0, 200, 0) if calor_ok else (0, 140, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color_bbox, 2)
            estado = '🔥 Listo' if calor_ok else 'Calienta...'
            cv2.putText(display, estado,
                        (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color_bbox, 2)

    # ── Flash ─────────────────────────────────────────
    if flash_timer > 0:
        cv2.rectangle(display, (0, 0), (640, 360), (0, 0, 255), 12)
        flash_timer -= 1

    # ── Panel UI ──────────────────────────────────────
    cv2.rectangle(display, (0, 0), (640, 38), (20, 20, 20), -1)
    cv2.putText(display,
                f'FIEBRE | {split.upper()} | '
                f'Train:{counters["train"]}  Val:{counters["val"]}',
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Indicador calor
    ind_texto = '🔥 CALOR OK — puede capturar' \
                if calor_ok else '⏳ Esperando calor... aplica compresa'
    ind_color = (0, 200, 0) if calor_ok else (0, 140, 255)
    cv2.putText(display, ind_texto,
                (8, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ind_color, 1)

    cv2.putText(display, ir_texto,
                (482, 206),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ir_color, 1)

    cv2.rectangle(display, (0, 338), (640, 360), (20, 20, 20), -1)
    cv2.putText(display,
                f'Rostros:{rostros}  '
                f'[ESPACIO] Capturar  [T] Train  [V] Val  [Q] Salir',
                (8, 354),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    cv2.imshow('Captura FIEBRE — Kinect v2', display)

    # ── Teclado ───────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('t'):
        split = 'train'
        print('▶ Split: TRAIN')
    elif key == ord('v'):
        split = 'val'
        print('▶ Split: VAL')
    elif key == ord(' '):
        if last_color is None:
            print('⚠️  Sin frame de color.')
        elif rostros == 0:
            print('⚠️  No se detectó rostro.')
        elif not calor_ok:
            print(f'⚠️  Calor insuficiente ({ir_mean:.0f}/255). '
                  f'Necesitas al menos {IR_MIN_MEAN}. '
                  f'Aplica compresa y espera.')
        else:
            ts      = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            rgb_224 = cv2.resize(
                cv2.cvtColor(last_color, cv2.COLOR_BGRA2BGR), (224, 224))
            ir_224  = cv2.resize(ir_norm, (224, 224))

            cv2.imwrite(
                f'{DATASET_DIR}/{split}/{CLASE}/{ts}_rgb.jpg', rgb_224)
            cv2.imwrite(
                f'{DATASET_DIR}/{split}/{CLASE}/{ts}_ir.jpg',  ir_224)

            counters[split] += 1
            flash_timer = 8
            winsound.Beep(1500, 80)
            print(f'🔥 [{split.upper()}] RGB+IR '
                  f'#{counters[split]} | IR:{ir_mean:.0f}')

# ── Cierre ────────────────────────────────────────────
kinect.close()
cv2.destroyAllWindows()
print(f'\nRESUMEN FIEBRE — Train:{counters["train"]} | Val:{counters["val"]}')