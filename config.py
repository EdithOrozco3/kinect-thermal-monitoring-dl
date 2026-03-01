# ── Kinect v2 ──────────────────────────────────────────
KINECT_COLOR_WIDTH    = 1920
KINECT_COLOR_HEIGHT   = 1080
KINECT_IR_WIDTH       = 512
KINECT_IR_HEIGHT      = 424

# ── Detección facial ───────────────────────────────────
FACE_MIN_CONFIDENCE   = 0.75
ROI_PADDING           = 30

# ── Modelo Deep Learning ───────────────────────────────
MODEL_INPUT_SIZE      = (224, 224)
MODEL_PATH            = 'model/saved/fever_model_best.h5'
FEVER_THRESHOLD       = 0.65

# ── Proxy térmico IR ───────────────────────────────────
IR_CALIBRATION_FRAMES = 30
IR_GAMMA_CORRECTION   = True

# ── Temperatura de referencia ──────────────────────────
# Rango normal de temperatura superficial facial (°C)
TEMP_NORMAL_MIN       = 35.0
TEMP_NORMAL_MAX       = 37.1
TEMP_FEVER_MIN        = 37.2   # Umbral de anomalía

# ── Alertas SMS Twilio ─────────────────────────────────
TWILIO_ACCOUNT_SID    = 'TU_ACCOUNT_SID'
TWILIO_AUTH_TOKEN     = 'TU_AUTH_TOKEN'
TWILIO_FROM_NUMBER    = '+1XXXXXXXXXX'
ALERT_TO_NUMBER       = '+52XXXXXXXXXX'
SMS_COOLDOWN_SECONDS  = 60