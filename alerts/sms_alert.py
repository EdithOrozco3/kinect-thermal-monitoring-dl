import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twilio.rest import Client
from datetime import datetime
from config import (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
                    TWILIO_FROM_NUMBER, ALERT_TO_NUMBER,
                    SMS_COOLDOWN_SECONDS, FEVER_THRESHOLD,
                    TEMP_FEVER_MIN)

class SMSAlert:
    def __init__(self):
        self._client     = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        self._last_sent  = None
        self._sent_count = 0

    def _in_cooldown(self):
        if self._last_sent is None:
            return False
        elapsed = (datetime.now() - self._last_sent).total_seconds()
        return elapsed < SMS_COOLDOWN_SECONDS

    def send_alert(self, probability, temperatura):
        """
        Envía SMS solo si:
        - probability >= FEVER_THRESHOLD  (modelo DL)
        - temperatura >= TEMP_FEVER_MIN   (proxy IR)
        - No está en período de cooldown
        """
        # Verificar umbrales
        if probability < FEVER_THRESHOLD:
            print(f'[SMS] Prob {probability:.1%} < umbral, no se envía.')
            return False

        if temperatura < TEMP_FEVER_MIN:
            print(f'[SMS] Temp {temperatura}°C < {TEMP_FEVER_MIN}°C, no se envía.')
            return False

        if self._in_cooldown():
            elapsed = (datetime.now() - self._last_sent).total_seconds()
            restante = int(SMS_COOLDOWN_SECONDS - elapsed)
            print(f'[SMS] Cooldown activo — {restante}s restantes.')
            return False

        # Construir mensaje
        now    = datetime.now()
        estado = 'FIEBRE DETECTADA' if temperatura >= 38.0 else 'TEMPERATURA ELEVADA'

        mensaje = (
            f'🚨 ALERTA: {estado}\n'
            f'──────────────────\n'
            f'🌡  Temperatura : {temperatura:.1f} °C\n'
            f'📊 Probabilidad: {probability:.1%}\n'
            f'🕐 Hora        : {now.strftime("%H:%M:%S")}\n'
            f'📅 Fecha       : {now.strftime("%d/%m/%Y")}\n'
            f'──────────────────\n'
            f'Sistema: Kinect v2 Fever Monitor'
        )

        try:
            msg = self._client.messages.create(
                body=mensaje,
                from_=TWILIO_FROM_NUMBER,
                to=ALERT_TO_NUMBER
            )
            self._last_sent  = now
            self._sent_count += 1
            print(f'[SMS] ✅ Enviado #{self._sent_count} | '
                  f'Temp:{temperatura}°C | '
                  f'Prob:{probability:.1%} | '
                  f'SID:{msg.sid}')
            return True

        except Exception as e:
            print(f'[SMS] ❌ Error al enviar: {e}')
            return False