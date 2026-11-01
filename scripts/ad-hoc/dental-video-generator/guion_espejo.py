"""
Guion de «La última vez que tu mamá te cepilló los dientes» (contado por el
espejo del baño), tal como viene en el PDF «Guion de video: salud dental».

Fórmula de los videos de referencia: cámara fija, contador a la izquierda,
subtítulos cortos en minúscula, salto de tiempo, giro final y cierre con un
dato. Sin voz en off: texto, música suave y efectos.

Tiempos en segundos absolutos. Cada subtítulo se muestra desde el inicio de su
escena + 0,25 s hasta el final − 0,25 s (medio segundo sin texto entre uno y otro).
"""

DURACION = 75.0

# Si el protagonista es niña: «yo solita» y «la que cuenta» (ver ESCENAS 9 y 15).
NOMBRE_HIJO = "Mateo"  # aparece en la cajita de lata
NOMBRE_MAMA = "Ana"  # aparece en su tarjeta de cita

TITULO = ["La última vez que tu mamá", "te cepilló los dientes"]
PALABRA_DORADA = "cepilló"
ETIQUETA = "CONTADO POR EL ESPEJO DEL BAÑO"
FIN_PORTADA = 3.0

ROTULO_VECES = "VECES QUE TE CEPILLÓ LOS DIENTES"
ROTULO_DESDE = "DESDE LA ÚLTIMA VEZ QUE TE LOS CEPILLÓ"

# contador: (rótulo, número_inicial, número_final, [(línea, dorada)]); None = oculto.
# El número puede ser un entero (se anima de inicial a final) o un texto fijo («8 AÑOS»).
ESCENAS = [
    {"id": 1, "inicio": 0.0, "fin": 3.0, "luz": "noche", "subtitulo": None,
     "contador": (ROTULO_VECES, 1, 1, [("TIENES 8 MESES · 1 DIENTE", False)]),
     "sonidos": [("cepillado_suave", 1.35, 0.6)]},
    {"id": 2, "inicio": 3.0, "fin": 7.0, "luz": "alterna", "subtitulo": "mañana y noche, ella te cepillaba los dientes.",
     "contador": (ROTULO_VECES, 1, 974, [("TIENES 2 AÑOS", False)]),
     "sonidos": [("cepillado_suave", 0.4, 0.5), ("cepillado_suave", 2.2, 0.5)]},
    {"id": 3, "inicio": 7.0, "fin": 11.0, "luz": "noche", "subtitulo": "contaba hasta 120. dos minutos exactos.",
     "contador": (ROTULO_VECES, 974, 1120, [("TIENES 2 AÑOS", False)]),
     "sonidos": [("cepillado_suave", 0.5, 0.5)]},
    {"id": 4, "inicio": 11.0, "fin": 15.0, "luz": "noche", "subtitulo": "a veces solo mojabas el cepillo.",
     "contador": (ROTULO_VECES, 2434, 2434, [("TIENES 4 AÑOS · 20 DIENTES", False)]),
     "sonidos": [("agua", 0.52, 0.7), ("vaso", 0.95, 0.5), ("vaso", 2.88, 0.8)]},
    {"id": 5, "inicio": 15.0, "fin": 19.0, "luz": "noche", "subtitulo": "ella sabía. te los cepillaba de todos modos.",
     "contador": (ROTULO_VECES, 2434, 2434, [("TIENES 4 AÑOS · 20 DIENTES", False)]),
     "sonidos": [("pasos", 0.0, 0.5), ("vaso", 1.75, 0.5), ("cepillado_suave", 3.3, 0.6)]},
    {"id": 6, "inicio": 19.0, "fin": 23.0, "luz": "manana", "subtitulo": "te llevaba al dentista aunque no te doliera nada.",
     "contador": (ROTULO_VECES, 3166, 3166, [("TIENES 5 AÑOS", False)]),
     "sonidos": [("papel", 2.45, 0.8)]},
    {"id": 7, "inicio": 23.0, "fin": 27.0, "luz": "manana", "subtitulo": "se te cayó el primer diente. ella lo guardó.",
     "contador": (ROTULO_VECES, 3896, 3896, [("TIENES 6 AÑOS", False)]),
     "sonidos": [("papel", 1.65, 0.7), ("lata", 2.98, 0.9)]},
    {"id": 8, "inicio": 27.0, "fin": 31.0, "luz": "noche", "subtitulo": "tú no te acuerdas. yo lo vi todo.",
     "contador": (ROTULO_VECES, 3896, 4500, [("TIENES 6 AÑOS", False)]),
     "zoom": {"foco": (594, 1335), "hasta": 1.4},
     "sonidos": [("monedas", 0.5, 0.8), ("lapiz", 1.5, 0.7)]},
    {"id": 9, "inicio": 31.0, "fin": 35.0, "luz": "noche", "subtitulo": "una noche dijiste: «yo solito».",
     "contador": (ROTULO_VECES, 4626, 4626, [("TIENES 7 AÑOS", False), ("LA ÚLTIMA", True)]),
     "sonidos": [("cepillado_suave", 1.4, 0.5)]},
    {"id": 10, "inicio": 35.0, "fin": 39.0, "luz": "noche", "subtitulo": "ninguno de los dos lo vio venir. yo tampoco.",
     "contador": (ROTULO_VECES, 4626, 4626, [("LA ÚLTIMA", True)]),
     "zoom": {"foco": (730, 820), "hasta": 1.22},
     "sonidos": []},
    {"id": 11, "inicio": 39.0, "fin": 40.0, "luz": "noche", "subtitulo": None,
     "contador": (ROTULO_DESDE, "1 DÍA", "1 DÍA", []),
     "sonidos": []},
    {"id": 12, "inicio": 40.0, "fin": 44.0, "luz": "manana", "subtitulo": "después, solo preguntaba desde la cocina.",
     "contador": (ROTULO_DESDE, "8 AÑOS", "8 AÑOS", [("TIENES 15 · ELLA 41", False)]),
     "notificacion": {"en": 1.6, "app": "MENSAJES", "de": "Mamá", "texto": "¿ya te lavaste los dientes? 🦷"},
     "sonidos": [("cepillado_suave", 0.3, 0.4), ("notificacion", 1.6, 1.0)]},
    {"id": 13, "inicio": 44.0, "fin": 48.0, "luz": "noche", "subtitulo": "sus citas siempre podían esperar.",
     "contador": (ROTULO_DESDE, "12 AÑOS", "12 AÑOS", [("TIENES 19 · ELLA 45", False)]),
     "zoom": {"foco": (590, 1180), "hasta": 1.22},
     "sonidos": [("papel", 1.5, 0.7), ("lapiz", 2.2, 0.8)]},
    {"id": 14, "inicio": 48.0, "fin": 50.0, "luz": "manana", "subtitulo": None, "contador": None,
     "salto": ["25 AÑOS", "DESPUÉS"],
     "sonidos": []},
    {"id": 15, "inicio": 50.0, "fin": 54.0, "luz": "noche", "subtitulo": "ahora eres tú el que cuenta hasta 120.",
     "contador": (ROTULO_DESDE, "25 AÑOS", "25 AÑOS", [("TIENES 32 · ELLA 58", False)]),
     "sonidos": [("cepillado_suave", 0.6, 0.5)]},
    {"id": 16, "inicio": 54.0, "fin": 58.0, "luz": "noche", "subtitulo": "ella cuidó cada uno de tus dientes.",
     "contador": (ROTULO_DESDE, "25 AÑOS", "25 AÑOS", [("TIENES 32 · ELLA 58", False), ("ELLA GUARDÓ LOS 20", True)]),
     "zoom": {"foco": (528, 1200), "hasta": 1.25},
     "sonidos": [("cajon", 0.2, 0.8), ("lata", 1.55, 1.0)]},
    {"id": 17, "inicio": 58.0, "fin": 62.0, "luz": "noche", "subtitulo": "los suyos siguen esperando.",
     "contador": (ROTULO_DESDE, "25 AÑOS", "25 AÑOS", [("TIENES 32 · ELLA 58", False), ("ELLA GUARDÓ LOS 20", True)]),
     "sonidos": []},
    {"id": 18, "inicio": 62.0, "fin": 66.0, "luz": "noche", "subtitulo": "esta vez, la llevas tú.",
     "contador": (ROTULO_DESDE, "25 AÑOS", "25 AÑOS", [("TIENES 32 · ELLA 58", False), ("SU CITA: JUEVES · 10:00 AM", True)]),
     "sonidos": [("papel", 0.8, 0.9)]},
]

# Tarjetas metidas en el marco del espejo (esquina superior derecha).
TARJETA_HIJO = {"titulo": "Próxima cita", "lineas": ["14 de marzo, 5 pm"], "aparece": 21.2}
TARJETA_MAMA = {
    "titulo": f"Cita dental · {NOMBRE_MAMA}",
    "tachadas": ["3 de abril", "9 de junio", "20 de sept."],
    "nueva": "15 de enero",
}
TARJETA_NUEVA = {"titulo": "Cita dental · Mamá", "lineas": ["jueves 10:00 am", "yo te llevo"], "aparece": 63.0}
NOTA_RATON = "Del Ratón de los Dientes"

# Cierre (1:06–1:12) sobre el último cuadro desenfocado y oscuro.
CIERRE_INICIO = 66.0
CIERRE_LINEAS = [
    ("titulo", "La última vez que tu mamá\nte cepilló los dientes"),
    ("dato", "LOS NIÑOS APRENDEN A CEPILLARSE SOLOS\nALREDEDOR DE LOS 7 AÑOS. ANTES, LO HACE\nUN ADULTO. (SOCIEDAD ESPAÑOLA DE\nODONTOPEDIATRÍA Y AEP)"),
    ("dato", "ELLA LO HIZO 4,626 VECES. AGÉNDALE SU CITA."),
    ("dorado", "¿CUÁNDO FUE LA ÚLTIMA VEZ\nQUE ELLA FUE AL DENTISTA?"),
]
# Dato alternativo del guion: «CASI LA MITAD DE LA POBLACIÓN MUNDIAL TIENE ALGUNA
# ENFERMEDAD BUCODENTAL. (OMS, 2022)».
LOGO_INICIO = 72.0  # logo y @usuario al centro (sin marca de agua durante el video)

# Música: caja de música y piano suaves; 1 s de silencio antes de «25 AÑOS DESPUÉS» y la música sube ahí.
SILENCIO_ANTES_DEL_SALTO = (47.0, 48.0)
SUBE_MUSICA = 48.0
