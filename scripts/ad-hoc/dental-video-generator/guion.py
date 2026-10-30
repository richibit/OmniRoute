"""
Guion definitivo de «24 horas sin cepillarse» (documento de producción, Fases 3 y 4).

Este archivo es el único que hay que tocar para cambiar textos, tiempos,
sonidos o recursos. Reglas de las referencias de tiempo:

* Un número (p. ej. ``12.5``) es un instante absoluto en segundos del video.
* Una tupla ``("palabra", bloque)`` es el instante en que la narradora dice esa
  palabra dentro del bloque indicado (se calcula a partir del audio real), con
  un tercer elemento opcional de desplazamiento: ``("caries", 8, -0.1)``.
  Así las imágenes, los textos y los sonidos quedan sincronizados con la voz
  aunque cambies la narración o la grabes tú.
* ``None`` en ``hasta`` significa «hasta el final del bloque».
"""

# ---------------------------------------------------------------------------
# Bloques narrativos = escenas. Deben ser contiguos y cubrir 0–48 s.
# ---------------------------------------------------------------------------
BLOQUES = [
    {
        "id": 1,
        "nombre": "Gancho",
        "inicio": 0.0,
        "fin": 3.0,
        "narracion": "Te acabas de cepillar… y tus dientes ya se están cubriendo otra vez.",
        "palabras_clave": ["cubriendo"],
        "textos": [
            {"texto": "TE ACABAS DE CEPILLAR…", "desde": 0.0, "hasta": 1.5, "estilo": "titulo"},
        ],
        "sonidos": [
            {"nombre": "squeak", "en": 0.05},
            {"nombre": "whoosh", "en": ("cubriendo", 1, -0.12)},
        ],
    },
    {
        "id": 2,
        "nombre": "Promesa",
        "inicio": 3.0,
        "fin": 6.0,
        "narracion": "Esto pasa en tu boca si no te cepillas en 24 horas.",
        "palabras_clave": ["24", "horas"],
        "textos": [
            {"texto": "24 HORAS\nSIN CEPILLARTE", "desde": 3.25, "hasta": None, "estilo": "titulo"},
        ],
        "sonidos": [{"nombre": "clic", "en": 3.0}],
    },
    {
        "id": 3,
        "nombre": "Minuto 1: la película",
        "inicio": 6.0,
        "fin": 10.0,
        "narracion": "Minuto uno: la saliva pega una capa de proteínas al diente. Funciona como velcro.",
        "palabras_clave": ["proteínas", "velcro"],
        "textos": [
            {"texto": "00:01 · Película salival", "desde": 6.2, "hasta": None, "estilo": "dato"},
            {"texto": "= VELCRO", "desde": ("velcro", 3, -0.05), "hasta": None, "estilo": "badge_menta", "y": 600},
        ],
        "sonidos": [
            {"nombre": "burbujas", "en": 6.25},
            {"nombre": "velcro", "en": ("velcro", 3, -0.05)},
        ],
    },
    {
        "id": 4,
        "nombre": "Hora 4: los pioneros",
        "inicio": 10.0,
        "fin": 15.0,
        "narracion": "Hora cuatro: ya hay bacterias pegadas. Casi todas estreptococos… y la mayoría, inofensivas.",
        "palabras_clave": ["bacterias", "estreptococos", "inofensivas"],
        "textos": [
            {"texto": "04:00 · 60–90 % estreptococos", "desde": 10.4, "hasta": None, "estilo": "dato"},
            {
                "texto": "Sorpresa: la mayoría son comensales",
                "desde": ("inofensivas", 4, -0.05),
                "hasta": None,
                "estilo": "badge_menta",
                "y": 640,
            },
        ],
        "sonidos": [
            {"nombre": "ticks_rapidos", "en": 10.0},
            {"nombre": "plop", "en": 11.35},
            {"nombre": "plop", "en": 11.9},
            {"nombre": "plop", "en": 12.5},
            {"nombre": "plop", "en": 13.15},
            {"nombre": "plop", "en": 13.8},
        ],
    },
    {
        "id": 5,
        "nombre": "Hora 8: colonias y ácido",
        "inicio": 15.0,
        "fin": 20.0,
        "narracion": "Hora ocho: forman colonias. Y cada vez que comes azúcar, lo convierten en ácido en minutos.",
        "palabras_clave": ["colonias", "azúcar", "ácido"],
        "textos": [
            {"texto": "08:00 · AZÚCAR → ÁCIDO", "desde": 15.3, "hasta": None, "estilo": "dato"},
        ],
        "sonidos": [
            {"nombre": "ticks_rapidos", "en": 15.0},
            {"nombre": "crujido", "en": ("azúcar", 5, 0.0)},
            {"nombre": "whoosh", "en": ("convierten", 5, -0.25)},
            {"nombre": "efervescencia", "en": ("convierten", 5, 0.05)},
        ],
    },
    {
        "id": 6,
        "nombre": "Dato clave",
        "inicio": 20.0,
        "fin": 25.0,
        "narracion": "Dato clave: sin placa, el azúcar casi no baja el pH. El ácido lo fabrican ellas.",
        "palabras_clave": ["placa", "ellas"],
        "textos": [
            {"texto": "Svatun y Attramadal, 1978", "desde": 20.6, "hasta": 24.8, "estilo": "fuente", "y": 1440},
        ],
        "sonidos": [
            {"nombre": "whoosh", "en": 20.0},
            # el riser sube y se corta en seco justo antes de «El ácido lo fabrican ellas»
            {"nombre": "riser", "en": 21.0, "hasta": ("ácido", 6, -0.3)},
        ],
    },
    {
        "id": 7,
        "nombre": "La noche",
        "inicio": 25.0,
        "fin": 31.0,
        "narracion": "De noche tu saliva casi se detiene. Las bacterias de la lengua sueltan gases de azufre. Buenos días, aliento.",
        "palabras_clave": ["noche", "azufre", "aliento"],
        "textos": [
            {"texto": "16:00 · NOCHE · Saliva ≈ 0", "desde": 25.35, "hasta": ("Buenos", 7, -0.1), "estilo": "dato"},
            {
                "texto": "Compuestos sulfurados volátiles",
                "desde": ("bacterias", 7, 0.0),
                "hasta": ("Buenos", 7, -0.1),
                "estilo": "badge_ambar",
                "y": 600,
            },
            {"texto": "BUENOS DÍAS,\nALIENTO", "desde": ("Buenos", 7, 0.0), "hasta": None, "estilo": "titulo"},
        ],
        "sonidos": [
            {"nombre": "grillos", "en": 25.2, "duracion": 4.0},
            {"nombre": "rio", "en": 25.1, "duracion": 3.2},
            {"nombre": "comico", "en": ("Buenos", 7, 0.0)},
        ],
    },
    {
        "id": 8,
        "nombre": "Hora 24: la revelación",
        "inicio": 31.0,
        "fin": 37.0,
        "narracion": "Hora veinticuatro: una capa pegajosa. ¿Ya tienes caries? No. ¿Gingivitis? Tampoco: tarda de 10 a 21 días.",
        "palabras_clave": ["pegajosa", "caries", "No", "Tampoco", "21"],
        "textos": [
            {"texto": "24:00", "desde": 31.2, "hasta": ("caries", 8, 0.2), "estilo": "titulo"},
            {"texto": "¿CARIES? ✗", "desde": ("No", 8, 0.0), "hasta": None, "estilo": "badge_coral", "y": 400},
            {
                "texto": "¿GINGIVITIS? ✗ (10–21 días)",
                "desde": ("Tampoco", 8, 0.0),
                "hasta": None,
                "estilo": "badge_coral",
                "y": 510,
            },
        ],
        "sonidos": [
            {"nombre": "impacto", "en": ("caries", 8, 0.0)},
            {"nombre": "ding_error", "en": ("No", 8, 0.05)},
            {"nombre": "ding_error", "en": ("Tampoco", 8, 0.05)},
        ],
    },
    {
        "id": 9,
        "nombre": "Conclusión",
        "inicio": 37.0,
        "fin": 43.0,
        "narracion": "Un día no te arruina la boca. La costumbre, sí. Y todo esto se va con un buen cepillado.",
        "palabras_clave": ["costumbre", "cepillado"],
        "textos": [
            {"texto": "UN DÍA NO.\nLA COSTUMBRE, SÍ.", "desde": 37.15, "hasta": 39.45, "estilo": "destacado"},
        ],
        "sonidos": [
            {"nombre": "cepillado", "en": ("todo", 9, 0.0)},
            {"nombre": "brillo", "en": ("cepillado", 9, 0.35)},
            {"nombre": "whoosh", "en": 42.55},
        ],
    },
    {
        "id": 10,
        "nombre": "Cierre y bucle",
        "inicio": 43.0,
        "fin": 48.0,
        "narracion": "¿Qué es lo máximo que has pasado sin cepillarte? Confiesa en comentarios. Aquí no juzgamos.",
        "palabras_clave": ["máximo", "comentarios"],
        "textos": [
            {"texto": "¿TU RÉCORD\nSIN CEPILLARTE? 👇", "desde": 43.35, "hasta": 47.2, "estilo": "titulo"},
        ],
        "sonidos": [
            {"nombre": "pop", "en": 43.35},
            {"nombre": "pop", "en": 44.6},
            {"nombre": "pop", "en": 45.4},
        ],
    },
]

# Burbujas de comentarios de ejemplo de la escena 10 (texto, instante de aparición).
COMENTARIOS_EJEMPLO = [
    ("2 días 😬", 44.6),
    ("nunca, lo juro", 45.4),
]

# Instante del corte final al macro del diente (bucle de repetición).
CORTE_BUCLE = 47.25

# ---------------------------------------------------------------------------
# Textos permanentes
# ---------------------------------------------------------------------------
TEXTOS_GLOBALES = [
    # Texto legal (6–37 s, esquina inferior, tamaño pequeño)
    {"texto": "Tiempos aproximados: varían entre personas.", "desde": 6.0, "hasta": 37.0, "estilo": "legal"},
    # Aviso de rigor en las vistas microscópicas
    {"texto": "Ilustración esquemática · no a escala", "desde": 6.0, "hasta": 20.0, "estilo": "legal2"},
]

# ---------------------------------------------------------------------------
# Reloj de 24 h (esquina superior derecha). Fotogramas clave: (instante, horas).
# Entre dos claves el reloj avanza con aceleración tipo time-lapse.
# ---------------------------------------------------------------------------
RELOJ_APARECE = 3.0  # pop-in con rebote en la escena 2
RELOJ_CLAVES = [
    (0.0, 0.0),
    (6.0, 0.0),
    (6.35, 1 / 60),  # 00:01
    (10.0, 1 / 60),
    (10.75, 4.0),  # salto a 04:00
    (15.0, 4.0),
    (15.6, 8.0),  # 08:00
    (25.0, 8.0),
    (25.7, 16.0),  # 16:00 (horas transcurridas, no hora del día)
    (31.0, 16.0),
    (31.8, 24.0),  # completa la vuelta y brilla
    (41.9, 24.0),
    (42.6, 0.0),  # giro inverso rápido a 00:00
    (48.0, 0.0),
]
RELOJ_NOCHE = (25.0, ("Buenos", 7, 0.0))  # fondo oscuro + luna

# ---------------------------------------------------------------------------
# Música generada (105 BPM → 21 compases exactos en 48 s: bucle limpio).
# capas: tic (pulso de reloj), bombo, bajo, pad, hats.
# armonia: "base" (Lam–Fa–Do–Sol), "oscura" (Lam–Rem–Sib–Mi), "mayor" (Do–Fa–Sol–Do)
# ---------------------------------------------------------------------------
BPM = 105
CAPAS_COMPLETAS = ["tic", "bombo", "bajo", "pad"]
MUSICA_SECCIONES = [
    {"inicio": 0.0, "fin": 0.5, "capas": [], "ganancia": 0.0},  # silencio inicial
    {"inicio": 0.5, "fin": 3.0, "capas": ["tic"], "ganancia": 1.0},  # entra el pulso
    {"inicio": 3.0, "fin": 10.0, "capas": CAPAS_COMPLETAS, "ganancia": 1.0},  # entra el bajo
    {"inicio": 10.0, "fin": 15.0, "capas": CAPAS_COMPLETAS + ["hats"], "ganancia": 1.15},  # subida
    {"inicio": 15.0, "fin": 20.0, "capas": CAPAS_COMPLETAS + ["hats"], "ganancia": 1.1, "armonia": "oscura"},
    {"inicio": 20.0, "fin": 25.0, "capas": ["tic", "pad", "bajo"], "ganancia": 0.3, "armonia": "oscura"},
    {"inicio": 25.0, "fin": 31.0, "capas": ["tic", "pad", "bajo"], "ganancia": 0.7, "noche": True},
    {"inicio": 31.0, "fin": ("caries", 8, 0.0), "capas": CAPAS_COMPLETAS, "ganancia": 1.0},
    {"inicio": ("caries", 8, 0.0), "fin": ("Tampoco", 8, 0.0), "capas": [], "ganancia": 0.0},  # silencio
    {"inicio": ("Tampoco", 8, 0.0), "fin": 37.0, "capas": CAPAS_COMPLETAS, "ganancia": 1.0},
    {"inicio": 37.0, "fin": 43.0, "capas": CAPAS_COMPLETAS, "ganancia": 1.0, "armonia": "mayor"},
    {"inicio": 43.0, "fin": 48.0, "capas": ["tic", "pad"], "ganancia": 0.9, "armonia": "mayor"},
]

# ---------------------------------------------------------------------------
# Recursos opcionales grabados. Si el archivo existe se usa; si no, el programa
# dibuja una alternativa animada. Basta con copiar el archivo con ese nombre
# (cualquier extensión de la lista) dentro de assets/imagenes o assets/videos.
# ---------------------------------------------------------------------------
EXTENSIONES_VIDEO = [".mp4", ".mov", ".m4v", ".webm", ".mkv"]
EXTENSIONES_IMAGEN = [".png", ".jpg", ".jpeg", ".webp"]
RECURSOS = {
    "dentista_espejo": "Escena 1 (inicio): la dentista termina de cepillarse frente al espejo y sonríe.",
    "velcro": "Escena 3: primer plano de una tira de velcro real (inserto de 0,5 s).",
    "persona_despierta": "Escena 7 (final): persona que despierta y se tapa la boca con humor.",
    "dentista_no": "Escena 8: la dentista a cámara en el consultorio responde «No» con calma.",
    "dentista_cierre": "Escena 10: la dentista a cámara con sonrisa cómplice.",
    "logo": "Escena 10: logotipo del consultorio (PNG con fondo transparente recomendado).",
}
