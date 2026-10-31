"""
Guion de «24 horas sin cepillarse» adaptado a la fórmula de los explicadores
verticales de referencia (los datos científicos son los del documento de
producción y su anexo de verificación).

Estructura de la fórmula:
  1. Gancho dirigido a «tú» con imagen literal.
  2. Negación que abre la intriga («no es suciedad…») + promesa.
  3. Mecanismo paso a paso, una idea por frase, con metáforas simples.
  4. Dato numérico grande y comparación.
  5. Dato curioso que rompe una creencia (Svatun y Attramadal, 1978).
  6. Giro: lo que esperas (caries) no pasa.
  7. Prueba histórica (experimento de Löe, 1965).
  8. Frase que reencuadra el gancho («Un día no… la costumbre, sí»).
  9. Llamada a compartir con destinatario concreto + cierre con logo.

Este archivo es el único que hay que tocar para cambiar textos, palabras
resaltadas, etiquetas, sonidos o el reloj. La duración NO se fija aquí: cada
bloque dura lo que dura su narración (voz real o sintética).

Referencias de tiempo dentro de un bloque:
  * número  → segundos desde el inicio del bloque (``0.0`` = cuando empieza a hablar)
  * ``("palabra", desplazamiento)`` → instante en que se dice esa palabra en el mismo bloque
  * ``("palabra", bloque, desplazamiento)`` → palabra de otro bloque
"""

BLOQUES = [
    {
        "id": 1,
        "nombre": "Gancho",
        "escena": "gancho",
        "narracion": "Te acabas de cepillar… y tus dientes ya se están cubriendo otra vez.",
        "resaltar": ["cubriendo"],
        "sonidos": [
            {"nombre": "brillo", "en": 0.15},
            {"nombre": "whoosh", "en": ("cubriendo", -0.15)},
        ],
    },
    {
        "id": 2,
        "nombre": "Negación + promesa",
        "escena": "negacion",
        "narracion": "Y no es suciedad: es tu propia saliva. Esto pasa si no te cepillas en 24 horas.",
        "resaltar": ["saliva"],
        "textos": {"etiqueta_saliva": "saliva"},
        "sonidos": [
            {"nombre": "burbujas", "en": ("saliva", -0.3)},
            {"nombre": "clic", "en": ("24", 0.0)},
        ],
    },
    {
        "id": 3,
        "nombre": "Minuto 1: la película",
        "escena": "pelicula",
        "narracion": "En un minuto, tu saliva pega una capa de proteínas al diente. Funciona como velcro.",
        "resaltar": ["velcro"],
        # En la tarjeta de velcro: los ganchos son las bacterias y los bucles, las proteínas de la película.
        "textos": {"etiqueta": "película salival", "tarjeta": "VELCRO", "gancho": "bacterias", "bucle": "proteínas"},
        "sonidos": [
            {"nombre": "whoosh", "en": 0.0},
            {"nombre": "pop", "en": ("capa", 0.0)},
            {"nombre": "velcro", "en": ("velcro", 0.0)},
        ],
    },
    {
        "id": 4,
        "nombre": "Hora 4: las pioneras",
        "escena": "pioneras",
        "narracion": "A las cuatro horas ya hay bacterias pegadas: casi todas, estreptococos… y la mayoría, inofensivas.",
        "resaltar": ["inofensivas"],
        "textos": {
            "dato": "60–90 %",
            "dato_sub": "SON ESTREPTOCOCOS",
            "especies": ["S. mitis", "S. oralis"],
            "comensales": "comensales ✓",
        },
        "sonidos": [
            {"nombre": "plop", "en": ("bacterias", 0.2)},
            {"nombre": "plop", "en": ("pegadas", 0.1)},
            {"nombre": "pop", "en": ("estreptococos", 0.0)},
            {"nombre": "pop", "en": ("inofensivas", 0.0)},
        ],
    },
    {
        "id": 5,
        "nombre": "Hora 8: colonias y ácido",
        "escena": "colonias",
        "narracion": "A las ocho forman colonias. Y cada vez que comes azúcar, lo convierten en ácido en minutos.",
        "resaltar": ["ácido"],
        "textos": {"umbral": "5,5 · umbral del esmalte"},
        "sonidos": [
            {"nombre": "crujido", "en": ("azúcar", 0.05)},
            {"nombre": "efervescencia", "en": ("convierten", 0.0)},
            {"nombre": "whoosh", "en": ("ácido", -0.25)},
        ],
    },
    {
        "id": 6,
        "nombre": "Dato clave",
        "escena": "dato_clave",
        "narracion": "Y lo más curioso: sin placa, el azúcar casi no baja el pH. El ácido lo fabrican ellas.",
        "resaltar": ["ellas"],
        "textos": {
            "titulo": "pH TRAS COMER AZÚCAR",
            "fila_1": "SIN PLACA",
            "fila_2": "CON PLACA",
            "fuente": "Svatun y Attramadal, 1978",
            "etiqueta_ellas": "bacterias de la placa",
        },
        "sonidos": [
            {"nombre": "riser", "en": ("curioso", 0.0), "hasta": ("ácido", -0.15)},
            {"nombre": "pop", "en": ("ellas", 0.0)},
        ],
    },
    {
        "id": 7,
        "nombre": "La noche",
        "escena": "noche",
        "narracion": "De noche tu saliva casi se detiene y las bacterias de tu lengua sueltan gases de azufre. Buenos días, aliento.",
        "resaltar": ["aliento"],
        "textos": {"rio": "saliva ≈ 0", "gases": "compuestos sulfurados volátiles"},
        "sonidos": [
            {"nombre": "grillos", "en": 0.0, "duracion": 3.0},
            {"nombre": "comico", "en": ("Buenos", 0.0)},
        ],
    },
    {
        "id": 8,
        "nombre": "Hora 24: la revelación",
        "escena": "revelacion",
        "narracion": "A las 24 horas ya tienes una capa pegajosa. ¿Caries? No. ¿Gingivitis? Tampoco: tarda de 10 a 21 días.",
        "resaltar": ["Tampoco"],
        "textos": {"placa": "biopelícula", "caries": "CARIES", "gingivitis": "GINGIVITIS"},
        "sonidos": [
            {"nombre": "impacto", "en": ("Caries", 0.0)},
            {"nombre": "ding_error", "en": ("No", 0.0)},
            {"nombre": "ding_error", "en": ("Tampoco", 0.0)},
        ],
    },
    {
        "id": 9,
        "nombre": "Prueba histórica",
        "escena": "historia",
        "narracion": "En 1965, unos voluntarios dejaron de cepillarse a propósito. Todos tuvieron gingivitis… y con una semana de cepillado, desapareció.",
        "resaltar": ["desapareció"],
        "textos": {
            "anio": "1965",
            "anio_sub": "VOLUNTARIOS SIN CEPILLARSE",
            "fuente": "Löe et al., 1965",
            "gingivitis": "gingivitis",
            "sanos": "encías sanas ✓",
            "dias": "día",
        },
        "sonidos": [
            {"nombre": "clic", "en": 0.0},
            {"nombre": "pop", "en": ("gingivitis", 0.0)},
            {"nombre": "brillo", "en": ("desapareció", 0.0)},
        ],
    },
    {
        "id": 10,
        "nombre": "Reencuadre",
        "escena": "costumbre",
        "narracion": "Un día no te arruina la boca. La costumbre, sí. Y todo eso se va con un buen cepillado.",
        "resaltar": ["costumbre"],
        "textos": {"pasta": "pasta con flúor"},
        "sonidos": [
            {"nombre": "cepillado", "en": ("todo", 0.0)},
            {"nombre": "brillo", "en": ("cepillado", 0.2)},
        ],
    },
    {
        "id": 11,
        "nombre": "Llamada a compartir",
        "escena": "compartir",
        "narracion": "Y mándaselo a quien se duerme sin cepillarse.",  # la «Y» suaviza el arranque de la voz sintética
        "resaltar": ["cepillarse"],
        "textos": {
            "enviar_a": "Enviar a",
            "contactos": ["Mi pareja", "Mamá", "Andrés", "Vale"],
            "boton": "Enviar",
            "enviado": "Enviado ✓",
        },
        "sonidos": [{"nombre": "pop", "en": ("quien", 0.1)}, {"nombre": "clic", "en": ("cepillarse", 0.1)}],
    },
]

# ---------------------------------------------------------------------------
# Reloj de 24 h (píldora fija bajo los subtítulos, a la izquierda).
# (referencia, horas). El cambio de hora dura RELOJ_TRANSICION segundos (efecto time-lapse).
# ---------------------------------------------------------------------------
RELOJ_APARECE = ("24", 2, -0.05)
RELOJ_TRANSICION = 0.6
RELOJ_CLAVES = [
    (("24", 2, 0.0), 0.0),
    (("minuto", 3, -0.1), 1 / 60),  # 00:01
    (("cuatro", 4, -0.1), 4.0),
    (("ocho", 5, -0.1), 8.0),
    (("noche", 7, -0.1), 16.0),  # horas transcurridas, no hora del día
    (("24", 8, -0.1), 24.0),
    (("cepillado", 10, 0.0), 0.0),  # giro inverso a 00:00 cuando se cepilla
]
RELOJ_NOCHE = (("noche", 7, -0.1), ("A", 8, -0.1))
RELOJ_OCULTO = [(("En", 9, -0.15), ("Un", 10, -0.15))]  # durante la historia de 1965

# Textos legales y de rigor (pequeños, abajo a la izquierda).
TEXTOS_LEGALES = [
    {"texto": "Tiempos aproximados: varían entre personas.", "desde": ("En", 3, -0.1), "hasta": ("En", 9, -0.15)},
    {"texto": "Ilustración esquemática · no a escala", "desde": ("En", 3, -0.1), "hasta": ("Y", 6, -0.1)},
]

# Música opcional (desactivada en la fórmula): ver config.MUSICA_DE_FONDO.
BPM = 105
