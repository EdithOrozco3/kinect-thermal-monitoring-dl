import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twilio.rest import Client
from datetime import datetime
from config import (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
                    TWILIO_FROM_NUMBER, ALERT_TO_NUMBER,
                    SMS_COOLDOWN_SECONDS)

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
        Envía SMS con hora, fecha, temperatura estimada
        y probabilidad de anomalía.
        """
        if self._in_cooldown():
            print('[SMS] En cooldown, alerta suprimida.')
            return False

        now    = datetime.now()
        estado = 'FIEBRE DETECTADA' if temperatura >= 37.2 else 'TEMPERATURA ELEVADA'

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

"""
        try:
            msg = self._client.messages.create(
                body=mensaje,
                from_=TWILIO_FROM_NUMBER,
                to=ALERT_TO_NUMBER
            )
            self._last_sent  = now
            self._sent_count += 1
            print(f'[SMS] ✅ Enviado | Temp: {temperatura}°C | '
                  f'Prob: {probability:.1%} | SID: {msg.sid}')
            return True
        except Exception as e:
            print(f'[SMS] ❌ Error: {e}')
            return False
"""