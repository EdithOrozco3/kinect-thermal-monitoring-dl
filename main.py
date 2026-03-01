import cv2
from datetime import datetime
from capture.kinect_stream          import KinectCapture
from detection.face_detector        import FaceDetector
from preprocessing.ir_thermal_proxy import IRThermalProxy
from model.predict                  import FeverPredictor
from alerts.sms_alert               import SMSAlert

def draw_overlay(frame, bbox, has_fever, prob, temperatura):
    x, y, w, h = bbox
    color = (0, 0, 255) if has_fever else (0, 220, 0)
    estado = 'ANOMALIA' if has_fever else 'Normal'
    now   = datetime.now().strftime('%H:%M:%S')

    # Bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Fondo etiqueta superior
    cv2.rectangle(frame, (x, y-50), (x+w, y), color, -1)

    # Texto: estado y probabilidad
    cv2.putText(frame,
                f'{estado}  {prob:.0%}',
                (x+5, y-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Texto: temperatura y hora
    cv2.putText(frame,
                f'{temperatura:.1f} C  {now}',
                (x+5, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    return frame

def log_deteccion(has_fever, prob, temperatura):
    """Registra cada detección en consola con timestamp."""
    now    = datetime.now()
    estado = '🚨 ANOMALIA' if has_fever else '✅ Normal  '
    print(f'[{now.strftime("%d/%m/%Y %H:%M:%S")}] '
          f'{estado} | '
          f'Temp: {temperatura:.1f}°C | '
          f'Prob: {prob:.1%}')

def main():
    kinect    = KinectCapture()
    detector  = FaceDetector()
    ir_proxy  = IRThermalProxy()
    predictor = FeverPredictor()
    sms       = SMSAlert()

    # ── Calibración IR ──────────────────────────────────
    print('\nCalibrando sensor IR...')
    print('Mantenga el encuadre libre de personas.\n')
    while not ir_proxy.calibrated:
        ir_frame = kinect.get_ir_frame()
        if ir_frame is not None:
            ir_proxy.add_calibration_frame(ir_frame)

    print('✅ Calibración completada.')
    print('Iniciando monitoreo... [Q para salir]\n')

    # ── Bucle principal ─────────────────────────────────
    while True:
        color_frame = kinect.get_color_frame()
        depth_frame = kinect.get_depth_frame()
        ir_frame    = kinect.get_ir_frame()

        if color_frame is None:
            continue

        display = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
        faces   = detector.detect(color_frame)

        for bbox in faces:
            rgb_roi = detector.extract_roi(color_frame, bbox)
            if rgb_roi is None:
                continue

            ir_roi = None
            if ir_frame is not None:
                ir_bbox = detector.scale_bbox_to_ir(bbox)
                ir_roi  = ir_proxy.extract_face_ir(
                    ir_frame, depth_frame, ir_bbox
                )

            if ir_roi is not None:
                has_fever, prob = predictor.predict(rgb_roi, ir_roi)
                temperatura     = ir_proxy.estimate_temperature(ir_roi)

                # Log en consola siempre
                log_deteccion(has_fever, prob, temperatura)

                # Overlay visual
                display = draw_overlay(
                    display, bbox, has_fever, prob, temperatura
                )

                # SMS solo si hay anomalía
                if has_fever:
                    sms.send_alert(prob, temperatura)

        cv2.imshow('Fever Monitor — Kinect v2  [Q: salir]', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    kinect.close()
    cv2.destroyAllWindows()
    print('Sistema detenido.')

if __name__ == '__main__':
    main()
