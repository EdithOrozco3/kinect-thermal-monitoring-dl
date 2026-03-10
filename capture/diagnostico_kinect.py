import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Parche Python 3.8
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

import cv2
import numpy as np

print('='*50)
print('  DIAGNÓSTICO KINECT v2 — Python 3.8')
print('='*50)

# ── 1. Imports ────────────────────────────────────────
print('\n[1/4] Verificando imports...')
try:
    from pykinect2 import PyKinectV2, PyKinectRuntime
    print('  ✅ pykinect2 OK')
except Exception as e:
    print(f'  ❌ pykinect2: {e}')
    input('\nPresiona Enter para salir...')
    sys.exit()

try:
    import mediapipe as mp
    print(f'  ✅ mediapipe {mp.__version__} OK')
except Exception as e:
    print(f'  ❌ mediapipe: {e}')

# ── 2. Conexión Kinect ────────────────────────────────
print('\n[2/4] Conectando sensor Kinect v2...')
print('  ⚠️  Asegúrate de cerrar Kinect Studio antes')
try:
    kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color    |
        PyKinectV2.FrameSourceTypes_Infrared |
        PyKinectV2.FrameSourceTypes_Depth
    )
    print('  ✅ Kinect inicializado correctamente')
except Exception as e:
    print(f'  ❌ Error al conectar: {e}')
    input('\nPresiona Enter para salir...')
    sys.exit()

# ── 3. Recibir frames ─────────────────────────────────
print('\n[3/4] Esperando frames (15 segundos)...')
print('  Colócate frente al Kinect ahora\n')

color_frame = ir_frame = depth_frame = None
t_inicio    = time.time()
intentos    = 0

while time.time() - t_inicio < 15:
    intentos += 1

    if kinect.has_new_color_frame() and color_frame is None:
        color_frame = kinect.get_last_color_frame().reshape(
            (1080, 1920, 4)).astype(np.uint8)
        print(f'  ✅ Color  recibido (intento {intentos})')

    if kinect.has_new_infrared_frame() and ir_frame is None:
        ir_frame = kinect.get_last_infrared_frame().reshape(
            (424, 512)).astype(np.float32)
        print(f'  ✅ IR     recibido (intento {intentos})')

    if kinect.has_new_depth_frame() and depth_frame is None:
        depth_frame = kinect.get_last_depth_frame().reshape(
            (424, 512)).astype(np.float32)
        print(f'  ✅ Depth  recibido (intento {intentos})')

    if color_frame is not None and ir_frame is not None and depth_frame is not None:
        print('\n  ✅ Los 3 streams funcionan correctamente')
        break

    time.sleep(0.033)

# ── 4. Análisis de valores ────────────────────────────
print('\n[4/4] Análisis de valores...')

if color_frame is not None:
    print(f'\n  COLOR (1920x1080x4):')
    print(f'    Media R: {color_frame[:,:,2].mean():.1f}')
    print(f'    Media G: {color_frame[:,:,1].mean():.1f}')
    print(f'    Media B: {color_frame[:,:,0].mean():.1f}')
else:
    print('  ❌ Sin frame de Color')

if ir_frame is not None:
    ir_norm = (np.clip(ir_frame / 65535.0, 0, 1) * 255).astype(np.uint8)
    temp_proxy = 35.0 + (ir_norm.mean() / 255.0) * 4.5
    print(f'\n  IR (512x424):')
    print(f'    Min     : {ir_frame.min():.0f}')
    print(f'    Max     : {ir_frame.max():.0f}')
    print(f'    Media   : {ir_frame.mean():.0f}')
    print(f'    IR norm : {ir_norm.mean():.1f} / 255')
    print(f'    Temp proxy estimada: {temp_proxy:.1f} °C')
else:
    print('  ❌ Sin frame IR — cierra Kinect Studio e intenta de nuevo')

if depth_frame is not None:
    depth_valido = depth_frame[depth_frame > 0]
    print(f'\n  DEPTH (512x424):')
    print(f'    Min válido: {depth_valido.min():.0f} mm')
    print(f'    Max válido: {depth_valido.max():.0f} mm')
    print(f'    Media     : {depth_valido.mean():.0f} mm (~{depth_valido.mean()/1000:.2f} m)')
else:
    print('  ❌ Sin frame Depth')

# ── Mostrar imágenes ──────────────────────────────────
if color_frame is not None or ir_frame is not None:
    print('\nMostrando imágenes 10 segundos...')
    print('[Q] para cerrar antes')

    t_show = time.time()
    while time.time() - t_show < 10:

        if color_frame is not None:
            display = cv2.resize(
                cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR), (640, 360))
            cv2.putText(display, 'COLOR OK',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 220, 0), 2)
            cv2.imshow('Diagnostico — COLOR', display)

        if ir_frame is not None:
            ir_norm    = (np.clip(ir_frame / 65535.0, 0, 1) * 255).astype(np.uint8)
            ir_colored = cv2.applyColorMap(ir_norm, cv2.COLORMAP_INFERNO)
            ir_show    = cv2.resize(ir_colored, (512, 424))
            temp_proxy = 35.0 + (ir_norm.mean() / 255.0) * 4.5
            cv2.putText(ir_show,
                        f'IR OK — Temp proxy: {temp_proxy:.1f} C',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.imshow('Diagnostico — IR (INFERNO)', ir_show)

        if depth_frame is not None:
            depth_norm = cv2.normalize(
                depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_col  = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_show = cv2.resize(depth_col, (512, 424))
            cv2.putText(depth_show, 'DEPTH OK',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.imshow('Diagnostico — DEPTH', depth_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ── Resumen final ─────────────────────────────────────
print('\n' + '='*50)
print('  RESUMEN FINAL')
print('='*50)
print(f'  Color : {"✅ OK" if color_frame is not None else "❌ No recibido"}')
print(f'  IR    : {"✅ OK" if ir_frame    is not None else "❌ No recibido — cierra Kinect Studio"}')
print(f'  Depth : {"✅ OK" if depth_frame is not None else "❌ No recibido"}')

if ir_frame is not None and depth_frame is not None and color_frame is not None:
    print('\n  🎉 Kinect v2 funcionando al 100%')
    print('  Puedes correr captura_normal.py y captura_fiebre.py')
else:
    print('\n  ⚠️  Algunos streams no funcionaron')
    print('  → Cierra Kinect Studio completamente')
    print('  → Desconecta y reconecta el Kinect')
    print('  → Vuelve a correr este diagnóstico')

print('='*50)

kinect.close()
cv2.destroyAllWindows()
input('\nPresiona Enter para salir...')