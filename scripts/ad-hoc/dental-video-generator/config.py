"""
Configuración global del generador de video «24 horas sin cepillarse».

Todo lo que normalmente querrás ajustar (rutas, formato, paleta, tipografías,
marca del consultorio, niveles de audio, motor de voz) vive aquí.
Los textos, tiempos y escenas están en `guion.py`.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto (todas relativas a esta carpeta: no hay rutas absolutas)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
DIR_ASSETS = RAIZ / "assets"
DIR_IMAGENES = DIR_ASSETS / "imagenes"
DIR_VIDEOS = DIR_ASSETS / "videos"
DIR_AUDIOS = DIR_ASSETS / "audios"
DIR_NARRACION = DIR_AUDIOS / "narracion"
DIR_MUSICA = DIR_AUDIOS / "musica"
DIR_SFX = DIR_AUDIOS / "sfx"
DIR_FUENTES = DIR_ASSETS / "fuentes"
DIR_SUBTITULOS = RAIZ / "subtitulos"
DIR_TEMP = RAIZ / "temp"
DIR_EXPORT = RAIZ / "export"

ARCHIVO_SALIDA = "24_horas_sin_cepillarse.mp4"

# ---------------------------------------------------------------------------
# Formato de video (TikTok / Reels / Shorts)
# ---------------------------------------------------------------------------
ANCHO = 1080
ALTO = 1920
FPS = 30
DURACION_TOTAL = 48.0  # segundos; debe coincidir con el final del último bloque del guion

# Perfiles de codificación H.264. CRF más bajo = más calidad y más peso.
CALIDADES = {
    "final": {"preset": "slow", "crf": 17},
    "borrador": {"preset": "veryfast", "crf": 24},
}

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
AUDIO_SR = 44100
AUDIO_BITRATE = "192k"
LOUDNESS_OBJETIVO_LUFS = -14.0  # estándar habitual de plataformas de video corto

# Ganancias en dB relativas (0 dB = narración normalizada).
NIVELES_DB = {
    "narracion": 0.0,
    "musica_libre": -15.0,  # música cuando nadie habla
    "musica_bajo_voz": -24.0,  # música mientras habla la dentista (ducking)
    "sfx": -9.0,
}

# ---------------------------------------------------------------------------
# Narración / texto a voz
# ---------------------------------------------------------------------------
# "auto" prueba en este orden: grabación propia > edge-tts (en línea) > espeak-ng (offline)
# > silencio (el video se genera igual, con subtítulos, para montar la voz después).
MOTOR_TTS = "auto"  # "auto" | "edge" | "espeak" | "silencio"
VOZ_EDGE = "es-MX-DaliaNeural"  # voz femenina cálida (requiere internet)
VOCES_ESPEAK = ["mb-es3", "es-419", "es"]  # mb-es3 = voz femenina MBROLA (paquete mbrola-es3)
VELOCIDAD_ESPEAK = 168  # palabras por minuto aproximadas
TEMPO_MAXIMO = 1.45  # aceleración máxima permitida para que un bloque quepa en su tiempo
MARGEN_INICIO_BLOQUE = 0.06  # s de aire antes de que empiece a hablar en cada bloque
MARGEN_FIN_BLOQUE = 0.10  # s de aire al final de cada bloque

# Sustituciones solo para la voz sintética (los subtítulos conservan el texto original).
PRONUNCIACION = {
    "pH": "pe hache",
    "…": ",",
}

# ---------------------------------------------------------------------------
# Identidad visual (documento de producción, Fase 6)
# ---------------------------------------------------------------------------
PALETA = {
    "esmalte": (247, 245, 240),  # #F7F5F0
    "azul": (14, 42, 71),  # #0E2A47 fondos, noche
    "menta": (63, 209, 181),  # #3FD1B5 limpio, reloj, marca
    "ambar": (255, 176, 32),  # #FFB020 ácido, alerta
    "coral": (255, 107, 91),  # #FF6B5B bacterias, negaciones ✗
    "blanco": (255, 255, 255),
    "tinta": (8, 22, 38),  # contorno de textos
    "turquesa": (46, 196, 182),  # estreptococos
    "encia": (226, 120, 136),
}

# Tipografías: se buscan primero en assets/fuentes/ y si no existen se usan
# alternativas del sistema. Cada entrada: (lista de archivos candidatos, peso).
FUENTES = {
    "titulo": (
        [
            DIR_FUENTES / "Montserrat-ExtraBold.ttf",
            DIR_FUENTES / "Montserrat[wght].ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ],
        800,
    ),
    "dato": (
        [
            DIR_FUENTES / "Inter-Medium.ttf",
            DIR_FUENTES / "Inter[opsz,wght].ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ],
        500,
    ),
    "dato_negrita": (
        [
            DIR_FUENTES / "Inter-Bold.ttf",
            DIR_FUENTES / "Inter[opsz,wght].ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ],
        700,
    ),
    "reloj": (
        [
            DIR_FUENTES / "JetBrainsMono-Bold.ttf",
            DIR_FUENTES / "JetBrainsMono[wght].ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
            "C:/Windows/Fonts/consolab.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ],
        700,
    ),
    "simbolos": (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/seguisym.ttf",
            "/System/Library/Fonts/Apple Symbols.ttf",
        ],
        700,
    ),
}
FUENTE_EMOJI = [
    DIR_FUENTES / "NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "C:/Windows/Fonts/seguiemj.ttf",
]

# Descarga opcional (--descargar-fuentes): fuentes libres OFL del repositorio de Google Fonts.
URLS_FUENTES = {
    "Montserrat[wght].ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Inter[opsz,wght].ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz,wght%5D.ttf",
    "JetBrainsMono[wght].ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
}

# ---------------------------------------------------------------------------
# Marca del consultorio (aparece discretamente en el cierre)
# ---------------------------------------------------------------------------
MARCA = {
    "consultorio": "Tu Consultorio Dental",  # cámbialo por el nombre real
    "usuario": "@tuconsultorio",  # cámbialo por el @ real
    "nombre_dentista": "Dra.",  # texto de la placa de la bata en la ilustración
}

# ---------------------------------------------------------------------------
# Subtítulos (2–3 palabras, palabra activa en menta, contorno oscuro de 4 px,
# entre el 55 % y el 70 % de la altura)
# ---------------------------------------------------------------------------
SUBTITULOS = {
    "max_palabras": 3,
    "max_caracteres": 18,
    "tamano": 84,
    "contorno": 4,
    "centro_y": 0.625,  # fracción de la altura (62,5 % → dentro de la franja 55–70 %)
    "centro_x": 520,  # algo a la izquierda del centro: evita los botones laterales de las apps
    "mayusculas": True,
}
