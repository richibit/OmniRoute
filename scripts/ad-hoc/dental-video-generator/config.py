"""
Configuración global del generador de video «24 horas sin cepillarse».

Estilo: fórmula de explicador vertical (gancho + negación, una idea por frase,
subtítulos karaoke arriba, etiquetas y tarjetas de datos, voz sola y cierre
con golpe grave sobre el logo). Todo lo ajustable (rutas, formato, paleta,
tipografías, marca, voz y niveles de audio) vive aquí.
Los textos y escenas están en `guion.py`.
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
DIR_MODELOS_VOZ = DIR_ASSETS / "modelos_voz"
DIR_SUBTITULOS = RAIZ / "subtitulos"
DIR_TEMP = RAIZ / "temp"
DIR_EXPORT = RAIZ / "export"

ARCHIVO_SALIDA = "24_horas_sin_cepillarse.mp4"  # además se exporta *_sin_voz.mp4

# ---------------------------------------------------------------------------
# Formato de video (TikTok / Reels / Shorts)
# ---------------------------------------------------------------------------
ANCHO = 1080
ALTO = 1920
FPS = 30
# La duración NO es fija: sale de la narración real (voz + cierre). Estos son los márgenes.
INICIO_VOZ = 0.10  # la voz entra casi en el primer fotograma (el gancho no espera)
PAUSA_ENTRE_BLOQUES = 0.16  # la fórmula es densa: casi sin silencios entre frases
CIERRE_SILENCIO = 1.20  # silencio tras la última palabra antes del golpe del logo
CIERRE_COLA = 2.40  # duración del cierre después del golpe
DURACION_MAXIMA = 75.0  # aviso si el video se alarga más (los de referencia duran 55–72 s)

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
LOUDNESS_OBJETIVO_LUFS = -14.0  # igual que los videos de referencia
NIVELES_DB = {
    "narracion": 0.0,
    "sfx": -20.0,  # efectos muy discretos: la fórmula es voz sola
    "golpe_final": -2.0,  # el «boom» del cierre suena más fuerte que la voz (como en las referencias)
    "musica": -24.0,  # solo si MUSICA_DE_FONDO = True
}
MUSICA_DE_FONDO = False  # la fórmula no lleva música bajo la voz; True para añadir assets/audios/musica/musica_fondo.*

# ---------------------------------------------------------------------------
# Narración / texto a voz
# ---------------------------------------------------------------------------
# "auto" prueba: grabación propia (bloque_XX.wav) > Kokoro (neuronal, offline) > edge-tts (en línea)
# > espeak-ng (robótica) > silencio (el video se genera igual, con subtítulos).
MOTOR_TTS = "auto"  # "auto" | "kokoro" | "edge" | "espeak" | "silencio"
VOZ_KOKORO = "ef_dora"  # femenina (la dentista). Alternativa masculina: "em_alex" (tono de los videos de referencia)
VELOCIDAD_KOKORO = 1.10  # ≈ 3 palabras/s, el ritmo medido en los videos de referencia
IDIOMA_KOKORO = "es-419"  # español latinoamericano (seseo). "es" = pronunciación de España
ARCHIVOS_KOKORO = {
    "modelo": "kokoro-v1.0.onnx",
    "modelo_tiempos": "kokoro-v1.0-tiempos.onnx",  # versión que expone la duración de cada fonema
    "voces": "voices-v1.0.bin",
}
URL_KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
VOZ_EDGE = "es-MX-DaliaNeural"
VOCES_ESPEAK = ["mb-es3", "es-419", "es"]
VELOCIDAD_ESPEAK = 168
TEMPO_MAXIMO = 1.25  # aceleración máxima si una grabación propia es demasiado lenta

# Sustituciones solo para voces sintéticas antiguas (espeak); Kokoro lee bien el texto original.
PRONUNCIACION = {"pH": "pe hache", "…": ","}

# ---------------------------------------------------------------------------
# Identidad visual
# ---------------------------------------------------------------------------
PALETA = {
    "esmalte": (247, 245, 240),  # #F7F5F0
    "azul": (14, 42, 71),  # #0E2A47
    "noche": (6, 10, 22),  # fondo casi negro de la fórmula
    "menta": (63, 209, 181),  # #3FD1B5 limpio, reloj, marca
    "ambar": (255, 176, 32),  # #FFB020 ácido, alerta
    "coral": (255, 107, 91),  # #FF6B5B bacterias, negaciones ✗
    "blanco": (255, 255, 255),
    "gris_texto": (178, 184, 196),  # palabras ya dichas en los subtítulos
    "tinta": (8, 14, 26),
    "seleccion": (59, 130, 246),  # recuadro azul de «selección» de la palabra clave
    "turquesa": (46, 196, 182),
    "encia": (226, 120, 136),
    "morado": (167, 139, 250),
    "oro": (232, 186, 92),
}

# Tipografías: se buscan primero en assets/fuentes/ y si no existen se usan
# alternativas del sistema. Cada entrada: (lista de archivos candidatos, peso).
_MONTSERRAT = [
    DIR_FUENTES / "Montserrat[wght].ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
FUENTES = {
    "titulo": ([DIR_FUENTES / "Montserrat-ExtraBold.ttf"] + _MONTSERRAT, 800),
    "sub_activa": ([DIR_FUENTES / "Montserrat-Bold.ttf"] + _MONTSERRAT, 700),
    "sub_pasada": ([DIR_FUENTES / "Montserrat-SemiBold.ttf"] + _MONTSERRAT, 600),
    "dato": (
        [
            DIR_FUENTES / "Inter[opsz,wght].ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ],
        500,
    ),
    "dato_negrita": (
        [
            DIR_FUENTES / "Inter[opsz,wght].ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ],
        700,
    ),
    "condensada": (
        [
            DIR_FUENTES / "Oswald[wght].ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "C:/Windows/Fonts/arialnb.ttf",
        ],
        600,
    ),
    "reloj": (
        [
            DIR_FUENTES / "JetBrainsMono[wght].ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
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
    "Oswald[wght].ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
}

# ---------------------------------------------------------------------------
# Marca del consultorio (marca de agua y cierre)
# ---------------------------------------------------------------------------
MARCA = {
    "consultorio": "Tu Consultorio Dental",  # cámbialo por el nombre real
    "usuario": "@tuconsultorio",  # cámbialo por el @ real (aparece en la marca de agua y en el cierre)
}

# ---------------------------------------------------------------------------
# Subtítulos karaoke (fórmula): arriba, la palabra que se dice en blanco y negrita,
# las ya dichas en gris, y una palabra clave por frase con recuadro azul de selección.
# ---------------------------------------------------------------------------
SUBTITULOS = {
    "tamano": 64,
    "y_primera_linea": 345,  # línea base de la primera línea (≈18 % de la altura)
    "interlineado": 80,
    "ancho_max": 860,
    "max_lineas": 2,
    "adelanto": 0.04,  # la palabra aparece 40 ms antes de oírse (se percibe sincronizada)
}

# Marca de agua (icono de Instagram + @usuario) a la derecha, bajo los subtítulos.
MARCA_AGUA = {"x_icono": 965, "y_icono": 488, "tam_icono": 58, "y_usuario": 556, "x_derecha": 992}
