# Modelos de voz

Voz neuronal offline Kokoro v1.0 (licencia Apache-2.0). Descárgala una vez con:

    python generar_video.py --descargar-voz

Se guardan `kokoro-v1.0.onnx` y `voices-v1.0.bin` (≈350 MB, no se suben a git). Si el paquete `onnx` está instalado, se crea además `kokoro-v1.0-tiempos.onnx`, que expone la duración de cada fonema para sincronizar los subtítulos palabra a palabra al milisegundo.
