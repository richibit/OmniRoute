"""
Escenas de «La última vez que tu mamá te cepilló los dientes», contado por el
espejo del baño: cámara fija a la altura del espejo (vemos el lavabo, a quien
se cepilla y la puerta del fondo), personajes 3D tipo juguete, contador a la
izquierda, subtítulos en itálica minúscula, salto de 25 años y cierre con dato.

Cada fotograma es una función pura del tiempo `t`: el video se renderiza en
paralelo por tramos sin que cambie el resultado.
"""

import functools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import config
import graficos as G
import guion_espejo as GE
import juguete3d as J
import personajes as PJ
from utilidades import (
    disco_suave,
    fuente,
    gradiente_radial,
    lerp,
    limitar,
    pegar,
    progreso,
    render_lineas,
    salida_cubica,
    suave,
)

W, H, FPS = config.ANCHO, config.ALTO, config.FPS
CREMA = (255, 250, 243)
DORADO = (232, 190, 98)  # #DDB664 del guion, un punto más vivo para leerse sobre fondos claros
LINEA = 1500  # borde de la encimera: lo que hay detrás queda oculto por debajo
PISO = 1330  # línea del suelo en la pared del fondo (pies de quien está en la puerta)
E = 1.4  # escala de quien está frente al espejo (cerca de la cámara)
PUERTA = (815, 770, 985)  # x0, y0, x1 del vano de la puerta (el suelo está en PISO)
VASO = (935, 1530)  # boca del vaso de los cepillos
GRIFO = (520, 1800)  # donde cae el agua
MARCO = (885, 520)  # esquina del espejo donde se meten las tarjetas
# Posición de cada tarjeta en el marco: la del hijo, la vieja de mamá y la nueva (en el borde de arriba,
# para no tapar la cara de mamá cuando la lee desde la puerta).
POS_TARJETAS = {0: (885, 480), 1: (859, 610), 2: (650, 300)}


# ---------------------------------------------------------------------------
# Utilidades de animación
# ---------------------------------------------------------------------------
def kf(t, claves):
    """Interpola fotogramas clave [(t, valor)] con suavizado; los valores pueden ser números o tuplas."""
    if t <= claves[0][0]:
        return claves[0][1]
    for (t0, v0), (t1, v1) in zip(claves, claves[1:]):
        if t <= t1:
            k = suave(progreso(t, t0, t1))
            if v0 is None or v1 is None:
                return v1 if k > 0.5 else v0
            if isinstance(v0, (tuple, list)):
                return tuple(a + (b - a) * k for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * k
    return claves[-1][1]


def parpadeo(t, base, desfase=0.0):
    """Sustituye la expresión por un parpadeo breve cada ~3,4 s."""
    return "parpadeo" if ((t + desfase) % 3.4) < 0.12 and base in ("neutral", "sonrisa", "abierta") else base


def cepillado(centro, t, amplitud=13, frecuencia=4.2):
    """Movimiento de vaivén de una mano que cepilla alrededor de `centro`."""
    x, y, z = centro
    return (x + math.sin(t * 2 * math.pi * frecuencia) * amplitud, y + math.cos(t * 2 * math.pi * frecuencia * 2) * 3, z)


def respiracion(t, desfase=0.0):
    return math.sin((t + desfase) * 2 * math.pi * 0.22) * 2.5


# ---------------------------------------------------------------------------
# Reparto
# ---------------------------------------------------------------------------
SUETER = (214, 160, 52)  # suéter mostaza de mamá (no cambia en 30 años)


@functools.lru_cache(maxsize=None)
def mama(edad=26):
    pelo = (206, 202, 198) if edad >= 55 else (84, 56, 42) if edad >= 40 else (92, 58, 40)
    return PJ.Personaje("adulto", SUETER, (92, 74, 64), pelo, "mono", arrugas=edad >= 55)


@functools.lru_cache(maxsize=None)
def hijo(edad, pegatina=False):
    if edad < 1:
        return PJ.Personaje("bebe", (176, 208, 238), (176, 208, 238), (70, 48, 34), "bebe")
    if edad < 3:
        return PJ.Personaje("pequeno", (128, 174, 230), (128, 174, 230), (64, 42, 30), "corto")
    if edad < 10:
        return PJ.Personaje("nino", (86, 128, 212), (86, 128, 212), (60, 40, 28), "corto", pegatina=(250, 200, 70) if pegatina else None)
    if edad < 25:
        return PJ.Personaje("adolescente", (150, 153, 160), (64, 70, 92), (54, 38, 30), "capucha")
    return PJ.Personaje("adulto", (40, 58, 98), (62, 62, 70), (54, 38, 30), "corto")


@functools.lru_cache(maxsize=None)
def nieta():
    return PJ.Personaje("pequeno", (240, 142, 152), (240, 142, 152), (72, 46, 34), "coletas")


def limitar_cuerpo(prims):
    """Oculta el cuerpo por debajo de la encimera; los brazos y las manos sí pueden pasar por delante."""
    for p in prims:
        if getattr(p, "parte", "cuerpo") != "brazo":
            p.limite_y = LINEA
    return prims


# ---------------------------------------------------------------------------
# Utilería 3D
# ---------------------------------------------------------------------------
def cepillo(mano, punta, color, pasta=False):
    """Cepillo de dientes de la mano a la punta (mango de color, cabezal blanco, pasta opcional)."""
    m, p = np.asarray(mano, np.float32), np.asarray(punta, np.float32)
    u = (p - m) / (np.linalg.norm(p - m) + 1e-6)
    prims = [J.Capsula(tuple(m - u * 12), tuple(p - u * 16), 7, color, brillo=0.35)]
    prims.append(J.Capsula(tuple(p - u * 18), tuple(p), 8.5, (236, 242, 248), brillo=0.1))
    if pasta:
        prims.append(J.Elipsoide(tuple(p - u * 9 + np.array([0, -7, 6])), (9, 6, 7), (250, 252, 252), brillo=0.2))
    return prims


def telefono(mano, inclinacion=0.0):
    x, y, z = mano
    return [J.Capsula((x - 4, y - 46, z + 10), (x + 4, y + 30, z + 10), 24, (36, 38, 46), aplanado=0.3, brillo=0.5)]


def lata(centro, abierta=False):
    x, y, z = centro
    prims = [J.Capsula((x - 60, y, z), (x + 60, y, z), 34, (186, 70, 64), aplanado=0.55, brillo=0.45)]
    if not abierta:
        prims.append(J.Capsula((x - 64, y - 26, z + 4), (x + 64, y - 26, z + 4), 14, (205, 90, 80), aplanado=0.5, brillo=0.5))
    return prims


def tarjeta3d(mano):
    x, y, z = mano
    return [J.Capsula((x - 44, y - 34, z + 14), (x + 44, y - 34, z + 14), 26, (250, 246, 236), aplanado=0.12, brillo=0.05)]


def diente_pequeno(mano):
    x, y, z = mano
    return [J.Elipsoide((x, y - 22, z + 18), (9, 11, 8), (252, 250, 240), brillo=0.4)]


# ---------------------------------------------------------------------------
# Decorado: el baño visto desde el espejo
# ---------------------------------------------------------------------------
def _luz_mapa(noche):
    """Mapa de iluminación del decorado (multiplicador por píxel)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if noche:  # lámpara sobre el espejo (arriba, hacia la cámara) + resplandor del pasillo
        base = 0.40 + 0.78 * np.exp(-(((xx - 520) / 700) ** 2 + ((yy - 560) / 900) ** 2))
        base += 0.5 * np.exp(-(((xx - 900) / 120) ** 2 + ((yy - 1050) / 300) ** 2))
    else:  # luz de la ventana (izquierda)
        base = 0.62 + 0.5 * np.exp(-(((xx - 280) / 560) ** 2 + ((yy - 560) / 760) ** 2))
    return base[..., None]


@functools.lru_cache(maxsize=2)
def fondo_bano(nombre):
    noche = nombre == "noche"
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    bx0, bx1, by0, zocalo = 120, 1000, 140, 960
    pared = (190, 142, 104) if noche else (204, 208, 196)
    azulejo = (230, 220, 202) if noche else (232, 238, 236)
    junta = (188, 172, 150) if noche else (190, 200, 198)
    d.polygon([(0, 0), (W, 0), (bx1, by0), (bx0, by0)], fill=(140, 104, 78) if noche else (182, 186, 180))  # techo
    d.polygon([(0, 0), (bx0, by0), (bx0, PISO), (0, H)], fill=tuple(int(c * 0.8) for c in pared))
    d.polygon([(W, 0), (bx1, by0), (bx1, PISO), (W, H)], fill=tuple(int(c * 0.74) for c in pared))
    d.polygon([(bx0, PISO), (bx1, PISO), (W, H), (0, H)], fill=(118, 94, 78) if noche else (160, 160, 152))
    d.rectangle([bx0, by0, bx1, zocalo], fill=pared)
    d.rectangle([bx0, zocalo, bx1, PISO], fill=azulejo)
    for y in range(zocalo, PISO, 56):
        d.line([(bx0, y), (bx1, y)], fill=junta, width=3)
    for x in range(bx0, bx1, 56):
        d.line([(x, zocalo), (x, PISO)], fill=junta, width=3)
    for lado, x_borde in ((-1, bx0), (1, bx1)):
        ext = 0 if lado < 0 else W
        d.polygon([(x_borde, zocalo), (x_borde, PISO), (ext, H), (ext, zocalo + 40)], fill=tuple(int(c * (0.84 if lado < 0 else 0.78)) for c in azulejo))
    d.rectangle([bx0, zocalo - 8, bx1, zocalo + 6], fill=tuple(int(c * 0.9) for c in azulejo))
    # ventana (izquierda)
    wx0, wy0, wx1, wy1 = 165, 370, 385, 690
    marco = (236, 230, 220) if noche else (244, 246, 246)
    d.rectangle([wx0 - 14, wy0 - 14, wx1 + 14, wy1 + 14], fill=marco)
    d.rectangle([wx0, wy0, wx1, wy1], fill=(58, 66, 92) if noche else (226, 240, 252))
    d.line([((wx0 + wx1) / 2, wy0), ((wx0 + wx1) / 2, wy1)], fill=marco, width=10)
    d.line([(wx0, (wy0 + wy1) / 2), (wx1, (wy0 + wy1) / 2)], fill=marco, width=10)
    # aplique de luz en la pared
    d.rounded_rectangle([625, 520, 695, 600], radius=26, fill=(255, 236, 196) if noche else (236, 232, 222))
    d.rectangle([652, 600, 668, 630], fill=(150, 130, 110))
    # toallero con toalla a rayas
    d.rounded_rectangle([555, 790, 705, 802], radius=6, fill=(188, 188, 194))
    d.rounded_rectangle([570, 798, 690, 1050], radius=20, fill=(222, 110, 96) if noche else (212, 122, 112))
    for y in (870, 910):
        d.rectangle([570, y, 690, y + 14], fill=(244, 230, 214))
    # puerta (derecha) con el pasillo iluminado
    px0, py0, px1 = PUERTA
    d.rectangle([px0 - 18, py0 - 18, px1 + 18, PISO], fill=(244, 236, 224) if noche else (238, 240, 238))
    pasillo = gradiente_radial(px1 - px0, PISO - py0, ((px1 - px0) / 2, 160), 520, (255, 212, 150) if noche else (220, 216, 206), (140, 90, 56) if noche else (150, 150, 144))
    dp = ImageDraw.Draw(pasillo)
    dp.rectangle([40, 90, 130, 170], outline=(120, 80, 50) if noche else (130, 126, 120), width=6)
    dp.polygon([(0, PISO - py0 - 110), (px1 - px0, PISO - py0 - 110), (px1 - px0, PISO - py0), (0, PISO - py0)], fill=(166, 106, 68) if noche else (156, 150, 140))
    img.paste(pasillo.convert("RGB").filter(ImageFilter.GaussianBlur(3)), (px0, py0))
    arr = np.asarray(img, np.float32) * _luz_mapa(noche)
    if noche:
        arr += np.asarray(gradiente_radial(W, H, (900, 1060), 300, (60, 34, 12), (0, 0, 0)).convert("RGB"), np.float32)
        arr += np.asarray(gradiente_radial(W, H, (660, 560), 220, (90, 60, 26), (0, 0, 0)).convert("RGB"), np.float32)
    else:
        rayos = Image.new("L", (W, H), 0)
        ImageDraw.Draw(rayos).polygon([(wx0, wy0), (wx1, wy0 + 40), (920, 1500), (320, 1500)], fill=52)
        arr += np.asarray(rayos.filter(ImageFilter.GaussianBlur(40)), np.float32)[..., None] * np.array([1.0, 0.97, 0.9])
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(1.2))
    return out.convert("RGBA")


@functools.lru_cache(maxsize=2)
def encimera(nombre):
    """Encimera en primer plano con lavabo, grifo cromado, vaso de cepillos y pasta (capa con transparencia arriba)."""
    noche = nombre == "noche"
    luz = J.NOCHE if noche else J.MANANA
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sup = (224, 210, 190) if noche else (226, 228, 224)
    arr = np.zeros((H, W, 4), np.float32)
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    k = np.clip((yy - LINEA) / (H - LINEA), 0, 1)
    for i in range(3):
        arr[..., i] = sup[i] * (0.98 - 0.16 * k)
    arr[..., 3] = (yy >= LINEA) * 255
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(img)
    d.rectangle([0, LINEA - 3, W, LINEA + 5], fill=tuple(min(255, c + 22) for c in sup) + (255,))
    cx, cy, rx, ry = GRIFO[0], 1830, 330, 150
    d.ellipse([cx - rx - 16, cy - ry - 12, cx + rx + 16, cy + ry + 12], fill=(250, 250, 248, 255))
    cuenco = gradiente_radial(rx * 2, ry * 2, (rx, ry * 0.75), rx * 1.1, (212, 218, 222), (146, 152, 160))
    m = Image.new("L", (rx * 2, ry * 2), 0)
    ImageDraw.Draw(m).ellipse([0, 0, rx * 2 - 1, ry * 2 - 1], fill=255)
    cuenco.putalpha(m)
    img.alpha_composite(cuenco, (cx - rx, cy - ry))
    d.ellipse([cx - 16, cy + 6, cx + 16, cy + 24], fill=(116, 120, 128, 255))
    vx, vy = VASO
    utileria = [
        J.Capsula((vx, vy - 10, 0), (vx, vy + 110, 0), 52, (150, 210, 196), brillo=0.3),  # vaso
        J.Capsula((vx - 24, vy, 40), (vx - 52, vy - 150, 30), 7, (238, 196, 70), brillo=0.35),  # cepillo de mamá
        J.Capsula((vx - 52, vy - 150, 30), (vx - 53, vy - 172, 30), 9, (236, 242, 248)),
    ]
    spr, x0, y0 = J.render(utileria, luz, margen=8)
    sh, pad = J.sombra_de(spr, (10, 8), 14, 0.35)
    img.alpha_composite(sh, (x0 - pad, y0 - pad))
    img.alpha_composite(spr, (x0, y0))
    tubo = sprite_pasta()
    sh, pad = J.sombra_de(tubo, (8, 8), 10, 0.35)
    img.alpha_composite(sh, (150 - pad, 1660 - pad))
    img.alpha_composite(tubo, (150, 1660))
    grifo = sprite_grifo(noche)
    img.alpha_composite(grifo, (cx - grifo.width // 2, H - grifo.height))
    return img


@functools.lru_cache(maxsize=1)
def sprite_pasta():
    """Tubo de pasta dental (blanco con franja azul, tapa roja), un poco girado."""
    ss = 3
    w, h = 190, 70
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(10 * ss, 14 * ss), (150 * ss, 8 * ss), (150 * ss, 58 * ss), (10 * ss, 52 * ss)], fill=(244, 246, 246, 255))
    d.rectangle([4 * ss, 12 * ss, 16 * ss, 54 * ss], fill=(226, 228, 230, 255))  # pliegue
    d.polygon([(60 * ss, 11 * ss), (110 * ss, 9 * ss), (110 * ss, 57 * ss), (60 * ss, 55 * ss)], fill=(66, 140, 210, 255))
    d.rounded_rectangle([148 * ss, 16 * ss, 184 * ss, 50 * ss], radius=8 * ss, fill=(210, 60, 66, 255))
    brillo = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(brillo).rectangle([14 * ss, 16 * ss, 146 * ss, 24 * ss], fill=(255, 255, 255, 120))
    img.alpha_composite(brillo.filter(ImageFilter.GaussianBlur(3 * ss)))
    return img.resize((w, h), Image.LANCZOS).rotate(8, expand=True, resample=Image.BICUBIC)


@functools.lru_cache(maxsize=2)
def sprite_grifo(noche):
    """Grifo cromado que asoma desde abajo (degradado metálico con reflejos)."""
    ss = 3
    w, h = 120, 210
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    m = Image.new("L", img.size, 0)
    dm = ImageDraw.Draw(m)
    dm.rounded_rectangle([30 * ss, 60 * ss, 90 * ss, h * ss], radius=30 * ss, fill=255)
    dm.rounded_rectangle([40 * ss, 0, 80 * ss, 90 * ss], radius=20 * ss, fill=255)
    xx = np.linspace(0, 1, w * ss)[None, :]
    base = 0.45 + 0.55 * np.abs(np.sin(xx * np.pi * 1.6 + 0.4))
    tono = np.array([214, 222, 236] if noche else [200, 212, 230], np.float32)
    rgb = np.repeat((base[..., None] * tono[None, None, :]), h * ss, axis=0)
    if noche:
        rgb = rgb * np.array([1.0, 0.93, 0.84])
    arr = np.dstack([np.clip(rgb, 0, 255), np.asarray(m, np.float32)]).astype(np.uint8)
    img = Image.fromarray(arr, "RGBA")
    return img.resize((w, h), Image.LANCZOS)


@functools.lru_cache(maxsize=4)
def sprite_lata(abierta=False):
    """Cajita de lata roja con el nombre del niño (vista sobre la encimera)."""
    ss = 2
    w, h = 220, 130
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 34 * ss, w * ss - 1, h * ss - 1], radius=26 * ss, fill=(176, 62, 58, 255))
    d.rounded_rectangle([10 * ss, 44 * ss, (w - 10) * ss, (h - 12) * ss], radius=20 * ss, outline=(214, 120, 100, 255), width=3 * ss)
    if abierta:
        d.rounded_rectangle([10 * ss, 0, (w - 10) * ss, 46 * ss], radius=18 * ss, fill=(206, 92, 80, 255))
    else:
        d.rounded_rectangle([0, 18 * ss, w * ss - 1, 56 * ss], radius=18 * ss, fill=(206, 92, 80, 255))
        d.rounded_rectangle([60 * ss, 70 * ss, (w - 60) * ss, 104 * ss], radius=6 * ss, fill=(250, 244, 228, 255))
        d.text((74 * ss, 68 * ss), GE.NOMBRE_HIJO, font=fuente("manuscrita", 28 * ss), fill=(150, 50, 46, 255))
    return img.resize((w, h), Image.LANCZOS)


def cepillo_en_vaso(nombre, color=(90, 140, 230)):
    """Cepillo del niño dentro del vaso (aparte, porque a veces está en su mano)."""
    luz = J.NOCHE if nombre == "noche" else J.MANANA
    return _cepillo_vaso(nombre, color, luz)


@functools.lru_cache(maxsize=8)
def _cepillo_vaso(nombre, color, luz):
    vx, vy = VASO
    spr, x0, y0 = J.render([J.Capsula((vx + 18, vy, 50), (vx + 50, vy - 152, 40), 7, color, brillo=0.35), J.Capsula((vx + 50, vy - 152, 40), (vx + 52, vy - 174, 40), 9, (236, 242, 248))], luz, margen=6)
    return spr, x0, y0


def chorro(img, t, intensidad=1.0):
    """Chorro de agua del grifo al lavabo."""
    if intensidad <= 0:
        return
    capa = Image.new("RGBA", (60, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    for k in range(4):
        x = 30 + math.sin(t * 40 + k) * 3
        d.line([(x, 0), (x + math.sin(t * 25 + k * 2) * 2, 90)], fill=(220, 236, 255, int(110 * intensidad)), width=6 - k)
    pegar(img, capa.filter(ImageFilter.GaussianBlur(1.2)), GRIFO[0], GRIFO[1] + 12)


@functools.lru_cache(maxsize=1)
def vineta_suave():
    """Oscurecimiento suave de los bordes (aspecto de cámara y mejor lectura de los textos)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.72)) ** 2 + ((yy - H * 0.5) / (H * 0.66)) ** 2)
    a = np.clip((d - 0.62) / 0.6, 0, 1) ** 1.6 * 150
    img = np.zeros((H, W, 4), np.uint8)
    img[..., 3] = a.astype(np.uint8)
    return Image.fromarray(img, "RGBA")


# ---------------------------------------------------------------------------
# Textos en pantalla
# ---------------------------------------------------------------------------
def _sombra_texto(capa, radio=8, fuerza=0.65, dy=3):
    s = Image.new("RGBA", (capa.width + radio * 4, capa.height + radio * 4), (0, 0, 0, 0))
    a = Image.new("L", s.size, 0)
    a.paste(capa.getchannel("A"), (radio * 2, radio * 2 + dy))
    s.putalpha(a.filter(ImageFilter.GaussianBlur(radio)).point(lambda v: int(v * fuerza)))
    s.alpha_composite(capa, (radio * 2, radio * 2))
    return s


@functools.lru_cache(maxsize=64)
def sprite_titulo():
    lineas = []
    for linea in GE.TITULO:
        segs = []
        for i, palabra in enumerate(linea.split(" ")):
            texto = palabra + (" " if i < len(linea.split(" ")) - 1 else "")
            segs.append((texto, DORADO if palabra == GE.PALABRA_DORADA else CREMA))
        lineas.append(segs)
    return _sombra_texto(render_lineas(lineas, "serif", 70, interlineado=1.08), 10, 0.6)


@functools.lru_cache(maxsize=8)
def sprite_etiqueta(texto):
    t = G.texto_espaciado(texto, "versalitas", 24, CREMA, 0.22)
    img = Image.new("RGBA", (t.width + 40, t.height + 14), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=6, fill=(24, 18, 14, 150))
    img.alpha_composite(t, (20, 6))
    return img


def _envolver_rotulo(texto, max_chars=22):
    palabras, lineas, actual = texto.split(), [], ""
    for p in palabras:
        if actual and len(actual) + 1 + len(p) > max_chars:
            lineas.append(actual)
            actual = p
        else:
            actual = (actual + " " + p).strip()
    return lineas + [actual]


@functools.lru_cache(maxsize=256)
def sprite_contador(rotulo, numero, lineas):
    """Contador de la izquierda: rótulo en versalitas, número grande en serif y líneas de edad (dorado si toca)."""
    partes = [G.texto_espaciado(l, "versalitas", 23, CREMA, 0.16) for l in _envolver_rotulo(rotulo)]
    num = render_lineas([numero], "serif", 72, color=CREMA)
    extras = [G.texto_espaciado(texto, "versalitas", 21, DORADO if dorada else CREMA, 0.16) for texto, dorada in lineas]
    ancho = max([p.width for p in partes] + [num.width] + [e.width for e in extras] + [10])
    alto = 30 * len(partes) + num.height + 30 * len(extras) + 50
    img = Image.new("RGBA", (ancho + 10, alto), (0, 0, 0, 0))
    y = 0
    for p in partes:  # rótulo en una o dos líneas
        img.alpha_composite(p, (0, y))
        y += 30
    img.alpha_composite(num, (-2, y + 12))
    y += num.height + 16
    for e in extras:  # edad y línea dorada
        img.alpha_composite(e, (0, y))
        y += 30
    # halo oscuro muy suave detrás del bloque: se lee igual sobre la pared, los azulejos o la ropa
    pad = 40
    halo = Image.new("L", (img.width + pad * 2, img.height + pad * 2), 0)
    halo.paste(img.getchannel("A"), (pad, pad))
    halo = halo.filter(ImageFilter.GaussianBlur(26)).point(lambda v: min(255, int(v * 1.6)))
    fondo = Image.new("RGBA", halo.size, (14, 9, 6, 0))
    fondo.putalpha(halo.point(lambda v: int(v * 0.7)))
    fondo.alpha_composite(_sombra_texto(img, 7, 0.85), (pad - 14, pad - 14))
    return fondo


@functools.lru_cache(maxsize=64)
def sprite_subtitulo(texto):
    t = render_lineas([texto], "serif_italica", 46, color=CREMA)
    return _sombra_texto(t, 9, 0.75, 2)


@functools.lru_cache(maxsize=4)
def sprite_salto(lineas):
    a = render_lineas([lineas[0]], "serif", 96, color=CREMA)
    b = render_lineas([lineas[1]], "serif", 64, color=(236, 228, 214))
    img = Image.new("RGBA", (max(a.width, b.width), a.height + b.height - 22), (0, 0, 0, 0))
    img.alpha_composite(a, (0, 0))
    img.alpha_composite(b, (4, a.height - 22))
    return _sombra_texto(img, 10, 0.7)


@functools.lru_cache(maxsize=16)
def sprite_tarjeta(titulo, lineas, tachadas=(), nueva=None, ancho=300, color=(252, 248, 236)):
    """Tarjeta de cita escrita a mano (con fechas tachadas si toca)."""
    ss = 2
    filas = 1 + len(lineas) + len(tachadas) + (1 if nueva else 0)
    alto = 70 + filas * 46
    img = Image.new("RGBA", (ancho * ss, alto * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, ancho * ss - 1, alto * ss - 1], radius=10 * ss, fill=color + (255,))
    d.rectangle([0, 0, ancho * ss, 16 * ss], fill=(150, 210, 196, 255))
    f_t = fuente("manuscrita", 40 * ss)
    tinta = (40, 50, 80, 255)
    y = 26 * ss
    d.text((22 * ss, y), titulo, font=f_t, fill=tinta)
    y += 50 * ss
    for l in lineas:
        d.text((22 * ss, y), l, font=f_t, fill=tinta)
        y += 46 * ss
    for l in tachadas:
        d.text((22 * ss, y), l, font=f_t, fill=(90, 96, 120, 255))
        w = f_t.getlength(l)
        d.line([(18 * ss, y + 26 * ss), (26 * ss + w, y + 20 * ss)], fill=(200, 60, 60, 255), width=4 * ss)
        y += 46 * ss
    if nueva:
        d.text((22 * ss, y), nueva, font=f_t, fill=tinta)
    return img.resize((ancho, alto), Image.LANCZOS)


@functools.lru_cache(maxsize=4)
def sprite_notificacion(app, de, texto):
    ss = 2
    w, h = 780, 170
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=34 * ss, fill=(246, 246, 248, 240))
    d.rounded_rectangle([26 * ss, 24 * ss, 62 * ss, 60 * ss], radius=9 * ss, fill=(52, 199, 89, 255))
    d.text((78 * ss, 28 * ss), app, font=fuente("versalitas", 24 * ss), fill=(120, 120, 128, 255))
    d.text(((w - 100) * ss, 28 * ss), "ahora", font=fuente("dato", 24 * ss), fill=(120, 120, 128, 255))
    d.text((30 * ss, 70 * ss), de, font=fuente("dato_negrita", 34 * ss), fill=(20, 20, 24, 255))
    t = render_lineas([texto], "dato", 32 * ss, color=(30, 30, 34))
    img.alpha_composite(t, (24 * ss, 108 * ss))
    return _sombra_texto(img.resize((w, h), Image.LANCZOS), 14, 0.45, 6)


@functools.lru_cache(maxsize=2)
def sprite_nota():
    ss = 2
    w, h = 300, 140
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w * ss - 1, h * ss - 1], fill=(255, 250, 220, 255))
    f = fuente("manuscrita", 38 * ss)
    palabras = GE.NOTA_RATON.split(" ")
    d.text((18 * ss, 18 * ss), " ".join(palabras[:2]), font=f, fill=(80, 60, 120, 255))
    d.text((18 * ss, 66 * ss), " ".join(palabras[2:]), font=f, fill=(80, 60, 120, 255))
    return img.resize((w, h), Image.LANCZOS).rotate(-8, expand=True, resample=Image.BICUBIC)


@functools.lru_cache(maxsize=2)
def sprite_lata_abierta():
    """Interior de la cajita de lata: los 20 dientes de leche, cada uno con su fecha."""
    ss = 2
    w, h = 470, 330
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=36 * ss, fill=(186, 70, 64, 255))
    d.rounded_rectangle([18 * ss, 18 * ss, (w - 18) * ss, (h - 18) * ss], radius=24 * ss, fill=(240, 226, 206, 255))
    f = fuente("manuscrita", 17 * ss)
    meses = ["mar", "may", "jun", "ago", "oct", "nov", "ene", "feb", "abr", "jun"]
    for i in range(20):
        fila, col = divmod(i, 5)
        x = (52 + col * 84) * ss
        y = (50 + fila * 66) * ss
        d.ellipse([x - 14 * ss, y - 16 * ss, x + 14 * ss, y + 12 * ss], fill=(252, 250, 242, 255), outline=(200, 190, 170, 255), width=2 * ss)
        d.ellipse([x - 6 * ss, y - 10 * ss, x + 2 * ss, y - 4 * ss], fill=(255, 255, 255, 255))
        d.text((x - 26 * ss, y + 14 * ss), f"{(i * 7) % 27 + 1} {meses[i % 10]}", font=f, fill=(90, 70, 60, 255))
    t = fuente("manuscrita", 28 * ss)
    d.text((w * ss - 160 * ss, (h - 16) * ss - 34 * ss), GE.NOMBRE_HIJO, font=t, fill=(150, 50, 46, 255))
    return img.resize((w, h), Image.LANCZOS)


@functools.lru_cache(maxsize=8)
def sprite_cierre(clave):
    estilo, texto = clave
    lineas = texto.split("\n")
    if estilo == "titulo":
        return _sombra_texto(render_lineas(lineas, "serif", 54, color=CREMA, interlineado=1.1), 8, 0.6)
    color = DORADO if estilo == "dorado" else (232, 226, 216)
    tam = 30 if estilo == "dorado" else 24
    partes = [G.texto_espaciado(l, "versalitas", tam, color, 0.12) for l in lineas]
    ancho = max(p.width for p in partes)
    img = Image.new("RGBA", (ancho, sum(p.height for p in partes) - 6 * (len(partes) - 1)), (0, 0, 0, 0))
    y = 0
    for p in partes:
        img.alpha_composite(p, ((ancho - p.width) // 2, y))
        y += p.height - 6
    return _sombra_texto(img, 7, 0.6)


# ---------------------------------------------------------------------------
# Renderizador
# ---------------------------------------------------------------------------
class Renderizador:
    def __init__(self, plan=None):
        self.escenas = {e["id"]: e for e in GE.ESCENAS}
        self.orden = [e["id"] for e in GE.ESCENAS]

    def escena_en(self, t):
        for e in GE.ESCENAS:
            if e["inicio"] <= t < e["fin"]:
                return e
        return GE.ESCENAS[-1]

    # ------------------------------------------------------------ fotograma
    def fotograma(self, t):
        if t >= GE.CIERRE_INICIO:
            img = self.cierre(t)
        else:
            e = self.escena_en(t)
            img = self.cuadro(t, e)
            img.alpha_composite(vineta_suave())
            self.capa_textos(img, t, e)
        return np.asarray(img.convert("RGB"))

    def cuadro(self, t, e):
        """Imagen de la escena (sin textos), con el acercamiento lento si la escena lo pide."""
        img = getattr(self, f"escena_{e['id']}")(t, e)
        zoom = e.get("zoom")
        if zoom:
            z = 1.0 + (zoom["hasta"] - 1.0) * suave(progreso(t, e["inicio"], e["fin"]))
            fx, fy = zoom["foco"]
            img = img.transform((W, H), Image.AFFINE, (1 / z, 0, fx - fx / z, 0, 1 / z, fy - fy / z), resample=Image.BILINEAR)
        return img

    # ---------------------------------------------------------- composición
    def base(self, luz_nombre, k_manana=None):
        if k_manana is None:
            return fondo_bano(luz_nombre).copy(), encimera(luz_nombre)
        a = Image.blend(fondo_bano("noche"), fondo_bano("manana"), k_manana)
        b = Image.blend(encimera("noche"), encimera("manana"), k_manana)
        return a, b

    def luz(self, nombre, k_manana=None):
        if k_manana is not None:
            return J.NOCHE.mezclar(J.MANANA, k_manana)
        return J.NOCHE if nombre == "noche" else J.MANANA

    def componer(self, t, luz_nombre, prims, extras=None, k_manana=None, cepillo_vaso=True, agua=0.0, tarjetas=None, encima=None):
        """Decorado + encimera + personajes (z-buffer conjunto) + agua + tarjetas del marco."""
        fondo, enc = self.base(luz_nombre, k_manana)
        fondo.alpha_composite(enc)
        nombre = luz_nombre if k_manana is None else ("manana" if k_manana > 0.5 else "noche")
        if cepillo_vaso:
            spr, x0, y0 = cepillo_en_vaso(nombre)
            fondo.alpha_composite(spr, (x0, y0))
        if extras:
            extras(fondo)
        if prims:
            spr, x0, y0 = J.render(prims, self.luz(luz_nombre, k_manana))
            if spr is not None:
                sh, pad = J.sombra_de(spr, (26, 10), 26, 0.32)
                fondo.alpha_composite(sh, (x0 - pad, y0 - pad))
                fondo.alpha_composite(spr, (x0, y0))
        if encima:
            encima(fondo)
        chorro(fondo, t, agua)
        for tarjeta in tarjetas or []:
            self.tarjeta_marco(fondo, *tarjeta)
        return fondo

    def tarjeta_marco(self, img, sprite, alfa=1.0, indice=0):
        """Tarjeta metida en el marco del espejo (esquina superior derecha, un poco girada)."""
        if alfa <= 0.01:
            return
        rot = sprite.rotate([-6, 5, -3][indice % 3], expand=True, resample=Image.BICUBIC)
        sh = _sombra_texto(rot, 10, 0.4, 6)
        x, y = POS_TARJETAS.get(indice, MARCO)
        pegar(img, sh, x, y, alfa=alfa)

    # --------------------------------------------------------------- textos
    def capa_textos(self, img, t, e):
        if t < GE.FIN_PORTADA + 0.3:  # portada: título y etiqueta los primeros 3 s
            a = 1 - progreso(t, GE.FIN_PORTADA - 0.3, GE.FIN_PORTADA)
            tit = sprite_titulo()
            pegar(img, tit, W / 2, 300, alfa=a)
            et = sprite_etiqueta(GE.ETIQUETA)
            pegar(img, et, W / 2, 300 + tit.height / 2 + 26, alfa=a)
        self.capa_contador(img, t, e)
        if e.get("subtitulo"):
            a = min(progreso(t, e["inicio"] + 0.25, e["inicio"] + 0.55), 1 - progreso(t, e["fin"] - 0.55, e["fin"] - 0.25))
            if a > 0:
                pegar(img, sprite_subtitulo(e["subtitulo"]), W / 2, 1190, alfa=a)
        if e.get("salto"):
            a = min(progreso(t, e["inicio"] + 0.1, e["inicio"] + 0.5), 1 - progreso(t, e["fin"] - 0.3, e["fin"]))
            spr = sprite_salto(tuple(e["salto"]))
            pegar(img, spr, 60, 1210, alfa=a, ancla="esquina")
        notif = e.get("notificacion")
        if notif:
            t0 = e["inicio"] + notif["en"]
            a = min(progreso(t, t0, t0 + 0.25), 1 - progreso(t, e["fin"] - 0.6, e["fin"] - 0.3))
            if a > 0:
                y = 230 - 60 * (1 - salida_cubica(progreso(t, t0, t0 + 0.35)))
                pegar(img, sprite_notificacion(notif["app"], notif["de"], notif["texto"]), W / 2, y, alfa=a)

    def capa_contador(self, img, t, e):
        c = e.get("contador")
        if not c:
            return
        rotulo, ini, fin, lineas = c
        if isinstance(ini, int):
            if ini == fin:
                n = ini
            else:  # el número corre (más rápido en la escena 8)
                k = progreso(t, e["inicio"] + 0.2, e["fin"] - 0.4)
                n = int(round(lerp(ini, fin, k if e["id"] == 8 else salida_cubica(k))))
            numero = f"{n:,}"
        else:
            numero = ini
        spr = sprite_contador(rotulo, numero, tuple(tuple(x) for x in lineas))
        # fundido corto cuando cambian rótulo/líneas respecto a la escena anterior
        previa = self.escenas.get(e["id"] - 1)
        a = 1.0
        if previa is None or previa.get("contador") is None or previa["contador"][0] != rotulo or previa["contador"][3] != lineas:
            a = progreso(t, e["inicio"], e["inicio"] + 0.3)
        pegar(img, spr, 44 - 40, 700 - 40, alfa=0.95 * a, ancla="esquina")

    # ================================================================ ESCENAS
    POS_MAMA = (470, 760, -170)
    POS_PUERTA = (900, 883)  # cabeza de quien mira desde la puerta (pies en el suelo del fondo)
    ESC_PUERTA = 0.7
    Z_PUERTA = -520
    POS_NINO = (745, 945)
    POS_PEQUENO = (745, 990)

    def _mama(self, t, edad, pose, pos=None, escala=E):
        x, y, z = pos or self.POS_MAMA
        y += respiracion(t) * escala / E
        pose = dict(pose)
        pose["cara"] = parpadeo(t, pose.get("cara", "sonrisa"))
        pose.setdefault("hasta_y", LINEA + 80)
        prims, info = mama(edad).construir(x, y, pose, escala=escala, z0=z)
        return limitar_cuerpo(prims), info

    def _mama_puerta(self, t, edad, pose, avance=0.0, destino=None, escala_destino=E):
        """Mamá en la puerta (avance 0) o caminando hacia `destino` (avance 1)."""
        destino = destino or self.POS_MAMA
        x = lerp(self.POS_PUERTA[0], destino[0], avance)
        y = lerp(self.POS_PUERTA[1], destino[1], avance)
        z = lerp(self.Z_PUERTA, destino[2], avance)
        esc = lerp(self.ESC_PUERTA, escala_destino, avance)
        pose = dict(pose)
        pose.setdefault("piernas", "de_pie")
        pose.setdefault("hasta_y", 0)
        return self._mama(t, edad, pose, (x, y, z), esc)

    def _nino(self, t, edad, pose, pos, pegatina=False, z=0.0, escala=E, limitar=True):
        x, y = pos
        y += respiracion(t, 1.3)
        pose = dict(pose)
        pose["cara"] = parpadeo(t, pose.get("cara", "neutral"), 1.7)
        if limitar:
            pose.setdefault("hasta_y", LINEA + 80)
        prims, info = hijo(edad, pegatina).construir(x, y, pose, escala=escala, z0=z)
        return (limitar_cuerpo(prims) if limitar else prims), info

    def _nieta(self, t, pose, pos):
        x, y = pos
        pose = dict(pose)
        pose["cara"] = parpadeo(t, pose.get("cara", "neutral"), 0.6)
        pose.setdefault("hasta_y", LINEA + 80)
        prims, info = nieta().construir(x, y + respiracion(t, 0.8), pose, escala=E)
        return limitar_cuerpo(prims), info

    def _adulto(self, t, pose, pos=None):
        x, y, z = pos or self.POS_MAMA
        pose = dict(pose)
        pose["cara"] = parpadeo(t, pose.get("cara", "neutral"), 2.2)
        pose.setdefault("hasta_y", LINEA + 80)
        prims, info = hijo(32).construir(x, y + respiracion(t, 0.4), pose, escala=E, z0=z)
        return limitar_cuerpo(prims), info

    @staticmethod
    def boca(info, dz=40):
        cx, cy = info["cabeza"]
        R = info["R"]
        return (cx - info["giro"] * R * 0.3, cy + R * 0.34, R * 0.96 + dz)

    def objetivo_cepillar(self, t, info_boca, lado=-1):
        """(mano objetivo, punta del cepillo) para cepillar la boca de `info_boca` desde el lado `lado`."""
        b = self.boca(info_boca)
        return cepillado((b[0] + lado * 46, b[1] + 8, b[2] + 34), t), (b[0] + lado * 4, b[1] - 2, b[2] + 10)

    @staticmethod
    def mano(info, lado="der"):
        return info["manos"][lado]

    def _tarjeta_hijo(self, alfa=1.0):
        return (sprite_tarjeta(GE.TARJETA_HIJO["titulo"], tuple(GE.TARJETA_HIJO["lineas"])), alfa, 0)

    def _tarjeta_mama_vieja(self):
        tachadas = tuple(GE.TARJETA_MAMA["tachadas"]) + (GE.TARJETA_MAMA["nueva"], "2 de mayo", "8 de nov.")
        return sprite_tarjeta(GE.TARJETA_MAMA["titulo"], (), tachadas, None)

    POS_MAMA_BEBE = (540, 885, -170)  # inclinada hacia el bebé sentado en la encimera
    POS_BEBE = (700, 1290)

    # 1 · Portada: mamá sienta al bebé en el lavabo y le pone pasta del tamaño de un grano de arroz
    def escena_1(self, t, e):
        lt = t - e["inicio"]
        abre = lt > 1.5
        pb, ib = self._nino(t, 0.7, {"cara": "abierta" if abre else "sonrisa", "piernas": "sentado", "mano_izq": (620, 1510, 90), "mano_der": (780, 1510, 90)}, self.POS_BEBE, escala=1.45, limitar=False)
        objetivo, punta = self.objetivo_cepillar(t, ib)
        if not abre:  # acerca el cepillo con la pasta
            b = self.boca(ib)
            objetivo = kf(lt, [(0.0, (560, 1440, 120)), (1.5, (b[0] - 80, b[1] + 50, b[2] + 40))])
            punta = (objetivo[0] + 34, objetivo[1] - 50, objetivo[2] + 10)
        pm, im = self._mama(t, 26, {"cara": "sonrisa", "giro": 0.35, "cabeceo": 0.45, "mano_der": objetivo, "mano_izq": (600, 1450, 60)}, self.POS_MAMA_BEBE)
        cep = cepillo(self.mano(im), punta, (238, 196, 70), pasta=True)
        return self.componer(t, "noche", pm + pb + cep, agua=1 - progreso(lt, 0.8, 1.2))

    # 2 · Mañana y noche: el bebé crece hasta los 2 años en el mismo encuadre
    def escena_2(self, t, e):
        lt = t - e["inicio"]
        paso = 0.66
        i = int(lt / paso)
        k_trans = progreso(lt - i * paso, paso - 0.2, paso)
        img = self._fase_2(t, i)
        if k_trans > 0 and lt < e["fin"] - e["inicio"] - paso * 0.5:
            img = Image.blend(img, self._fase_2(t, i + 1), k_trans)
        return img

    def _fase_2(self, t, i):
        luz = "manana" if i % 2 == 0 else "noche"
        edad = [0.7, 1.0, 1.4, 1.7, 2.0, 2.0, 2.0][min(i, 6)]
        if edad < 1.2:
            pb, ib = self._nino(t, 0.7, {"cara": "abierta", "piernas": "sentado", "mano_izq": (620, 1510, 90), "mano_der": (780, 1510, 90)}, self.POS_BEBE, escala=1.45 + 0.1 * (edad - 0.7), limitar=False)
            pos_m = self.POS_MAMA_BEBE
        else:
            k = (edad - 1.4) / 0.6
            pb, ib = self._nino(t, 2.0, {"cara": "abierta"}, (self.POS_PEQUENO[0], self.POS_PEQUENO[1] + 30 * (1 - k)), escala=E * (0.88 + 0.12 * k))
            pos_m = self.POS_MAMA
        objetivo, punta = self.objetivo_cepillar(t, ib)
        pm, im = self._mama(t, 26, {"cara": "sonrisa", "giro": 0.35, "cabeceo": 0.3, "mano_der": objetivo}, pos_m)
        return self.componer(t, luz, pm + pb + cepillo(self.mano(im), punta, (238, 196, 70)))

    # 3 · El niño aprieta la boca; mamá hace caras al espejo y cuenta con los dedos
    def escena_3(self, t, e):
        lt = t - e["inicio"]
        abre = lt > 2.2
        pb, ib = self._nino(t, 2.0, {"cara": "abierta" if abre else "apretada"}, self.POS_PEQUENO)
        objetivo, punta = self.objetivo_cepillar(t, ib)
        if not abre:
            b = self.boca(ib)
            objetivo = (b[0] - 80, b[1] + 50, b[2] + 40)
            punta = (objetivo[0] + 34, objetivo[1] - 50, objetivo[2] + 10)
        dedos = 1 + int(limitar((lt - 0.6) / 0.55, 0, 4))
        pm, im = self._mama(t, 26, {"cara": "mueca" if not abre else "sonrisa", "giro": -0.1, "mano_der": objetivo, "mano_izq": (330, 720 + math.sin(t * 6) * 6, 80), "dedos_izq": dedos})
        return self.componer(t, "noche", pm + pb + cepillo(self.mano(im), punta, (238, 196, 70)))

    # 4 · El niño (4) solo: moja el cepillo seco, mira a la puerta y lo devuelve al vaso
    def escena_4(self, t, e):
        lt = t - e["inicio"]
        vx, vy = VASO
        objetivo = kf(lt, [(0.0, (vx + 30, vy - 40, 80)), (0.6, (GRIFO[0] + 50, GRIFO[1] - 70, 150)), (1.8, (GRIFO[0] + 50, GRIFO[1] - 70, 150)), (2.4, (800, 1360, 110)), (3.2, (vx + 30, vy - 30, 80)), (4.0, (vx + 30, vy - 30, 80))])
        giro = kf(lt, [(0.0, 0.0), (1.9, 0.0), (2.2, 0.9), (2.9, 0.9), (3.2, 0.0)])
        pb, ib = self._nino(t, 4, {"cara": "neutral", "giro": giro, "mano_der": objetivo}, self.POS_NINO)
        en_mano = lt < 3.25
        m = self.mano(ib)
        prims = pb + (cepillo(m, (m[0] - 10, m[1] - 130, m[2] + 10), (90, 140, 230)) if en_mano else [])
        return self.componer(t, "noche", prims, cepillo_vaso=not en_mano, agua=1.0 if 0.5 < lt < 1.9 else 0.0)

    # 5 · Mamá entra, toca las cerdas, lo mira por el espejo, sonríe y se los cepilla otra vez
    def escena_5(self, t, e):
        lt = t - e["inicio"]
        pb, ib = self._nino(t, 4, {"cara": "neutral" if lt < 2.0 else ("abierta" if lt > 2.7 else "sonrisa"), "giro": 0.5 if lt < 1.4 else 0.0}, self.POS_NINO)
        if lt < 1.3:  # entra caminando desde la puerta
            pm, _ = self._mama_puerta(t, 26, {"cara": "neutral", "piernas": "caminando", "paso": lt * 9, "giro": -0.4}, suave(progreso(lt, 0.0, 1.3)))
            return self.componer(t, "noche", pb + pm)
        vx, vy = VASO
        b = self.boca(ib)
        if lt < 2.7:
            objetivo = kf(lt, [(1.3, (vx - 40, vy - 60, 60)), (1.7, (vx + 30, vy - 130, 60)), (2.2, (b[0] - 80, b[1] + 60, b[2] + 40)), (2.7, (b[0] - 46, b[1] + 8, b[2] + 34))])
            punta = None
        else:
            objetivo, punta = self.objetivo_cepillar(t, ib)
        giro = kf(lt, [(1.3, 0.3), (1.8, 0.0), (2.1, 0.0), (2.4, 0.35)])
        pm, im = self._mama(t, 26, {"cara": "neutral" if lt < 1.8 else "sonrisa", "giro": giro, "mano_der": objetivo})
        m = self.mano(im)
        cep = []
        if lt > 1.75:
            cep = cepillo(m, punta or (m[0] + 28, m[1] - 70, m[2] + 10), (90, 140, 230), pasta=lt > 2.1)
        return self.componer(t, "noche", pm + pb + cep, cepillo_vaso=lt < 1.75)

    # 6 · De regreso del dentista: estampa «paciente valiente», enseña los dientes; mamá mete la tarjeta en el marco
    def escena_6(self, t, e):
        lt = t - e["inicio"]
        pb, _ = self._nino(t, 5, {"cara": "dientes", "mano_izq": (650, 1200, 90), "mano_der": (840, 1200, 90)}, self.POS_NINO, pegatina=True)
        objetivo = kf(lt, [(0.0, (560, 1330, 0)), (1.6, (560, 1330, 0)), (2.3, (MARCO[0], MARCO[1] + 70, 330)), (2.9, (MARCO[0] + 10, MARCO[1] + 40, 340)), (3.5, (560, 1330, 0))])
        pm, im = self._mama(t, 26, {"cara": "sonrisa", "giro": kf(lt, [(1.5, 0.3), (2.2, 0.75), (3.2, 0.3)]), "mano_der": objetivo})
        prims = pm + pb + (tarjeta3d(self.mano(im)) if 1.6 < lt < 2.75 else [])
        return self.componer(t, "manana", prims, tarjetas=[self._tarjeta_hijo(progreso(lt, 2.6, 2.9))])

    # 7 · Se le cae el primer diente: mamá lo envuelve en papel y lo guarda en la cajita de lata
    def escena_7(self, t, e):
        lt = t - e["inicio"]
        obj_n = kf(lt, [(0.0, (720, 1180, 160)), (1.4, (720, 1180, 160)), (1.9, (680, 1230, 140))])
        pb, ib = self._nino(t, 6, {"cara": "dientes", "hueco": True, "mano_izq": obj_n}, self.POS_NINO)
        obj_m = kf(lt, [(0.0, (560, 1340, 0)), (1.2, (560, 1340, 0)), (1.8, (670, 1220, 170)), (2.5, (360, 1450, 160)), (4.0, (360, 1450, 160))])
        pm, im = self._mama(t, 26, {"cara": "sonrisa", "giro": 0.35, "cabeceo": 0.3, "mano_der": obj_m})
        prims = pm + pb
        if lt < 1.85:
            prims += diente_pequeno(self.mano(ib, "izq"))
        elif lt < 2.6:
            m = self.mano(im)
            prims += [J.Elipsoide((m[0], m[1] - 20, m[2] + 20), (18, 15, 13), (250, 248, 240))]

        def mesa(img):
            pegar(img, sprite_lata(abierta=lt < 3.0), 330, 1590)

        return self.componer(t, "manana", prims, extras=mesa, tarjetas=[self._tarjeta_hijo()])

    # 8 · De noche, mamá sola saca monedas y escribe una notita del Ratón de los Dientes (acercamiento lento)
    def escena_8(self, t, e):
        lt = t - e["inicio"]
        escribe = cepillado((590, 1530, 150), t, amplitud=10, frecuencia=2.5) if lt > 1.5 else (560, 1520, 150)
        pm, im = self._mama(t, 26, {"cara": "sonrisa", "cabeceo": 0.9, "mano_der": escribe, "mano_izq": (440, 1530, 140)}, (540, 790, -170))
        m = self.mano(im)
        prims = pm + [J.Capsula((m[0], m[1] - 4, m[2] + 6), (m[0] + 34, m[1] - 70, m[2] + 14), 6, (230, 190, 70), brillo=0.4)]

        def mesa(img):
            pegar(img, sprite_nota(), 610, 1620, alfa=progreso(lt, 0.8, 1.2))
            for k, (mx, my) in enumerate(((380, 1600), (425, 1625), (400, 1572))):
                pegar(img, disco_suave(24, (232, 190, 80), 0.8, 255), mx, my, alfa=progreso(lt, 0.3 + k * 0.2, 0.5 + k * 0.2))

        return self.componer(t, "noche", prims, encima=mesa, tarjetas=[self._tarjeta_hijo()])

    def _nino_cepillandose(self, t, edad, pos, cara="abierta", extra=None):
        """El niño se cepilla solo (mano derecha)."""
        _, ib = self._nino(t, edad, {}, pos)
        b = self.boca(ib)
        objetivo = cepillado((b[0] + 46, b[1] + 8, b[2] + 34), t)
        pose = {"cara": cara, "mano_der": objetivo}
        pose.update(extra or {})
        pb, ib = self._nino(t, edad, pose, pos)
        return pb + cepillo(self.mano(ib), (b[0] + 4, b[1] - 2, b[2] + 10), (90, 140, 230)), ib

    # 9 · El niño (7) le quita el cepillo: ella lo suelta y da un paso atrás, hasta la puerta
    def escena_9(self, t, e):
        lt = t - e["inicio"]
        _, ib = self._nino(t, 7, {}, self.POS_NINO)
        b = self.boca(ib)
        if lt < 1.4:
            obj_n = kf(lt, [(0.0, (880, 1330, 80)), (0.7, (b[0] - 60, b[1] + 30, b[2] + 34)), (1.4, (b[0] + 46, b[1] + 8, b[2] + 34))])
            pb, ibn = self._nino(t, 7, {"cara": "neutral", "mano_der": obj_n}, self.POS_NINO)
            prims = pb + cepillo(self.mano(ibn), (b[0] + 4, b[1] - 2, b[2] + 10), (90, 140, 230))
        else:
            prims, _ = self._nino_cepillandose(t, 7, self.POS_NINO)
        retro = suave(progreso(lt, 1.2, 3.4))
        obj_m = (b[0] - 60, b[1] + 30, b[2] + 34) if lt < 0.75 else None
        pm, _ = self._mama_puerta(t, 26, {"cara": "sonrisa", "giro": 0.3, "piernas": "caminando" if 1.2 < lt < 3.4 else "de_pie", "paso": lt * 7, "mano_der": obj_m}, 1 - retro)
        return self.componer(t, "noche", prims + pm)

    # 10 · Mamá lo mira desde la puerta; levanta la mano para ayudarle y la baja (acercamiento lento)
    def escena_10(self, t, e):
        lt = t - e["inicio"]
        prims, _ = self._nino_cepillandose(t, 7, self.POS_NINO)
        x, y = self.POS_PUERTA
        R = 74 * self.ESC_PUERTA
        abajo = (x - R * 0.9, y + R * 4.5, self.Z_PUERTA)
        sube = (x - R * 1.6, y + R * 1.4, self.Z_PUERTA + R)
        mano = kf(lt, [(0.0, None), (1.0, None), (1.01, abajo), (1.6, sube), (2.4, sube), (3.0, abajo), (3.01, None)])
        pm, _ = self._mama_puerta(t, 26, {"cara": "sonrisa" if lt < 2.4 else "neutral", "giro": -0.3, "mano_izq": mano})
        return self.componer(t, "noche", prims + pm)

    # 11 · Solo cambia el contador
    def escena_11(self, t, e):
        prims, _ = self._nino_cepillandose(t, 7, self.POS_NINO)
        return self.componer(t, "noche", prims)

    # 12 · Adolescente (15) se cepilla 20 segundos con el celular en la mano; llega la notificación
    def escena_12(self, t, e):
        pos = (600, 765)
        _, ib = self._nino(t, 15, {"cabeceo": 0.5, "giro": 0.2}, pos)
        b = self.boca(ib)
        obj_c = cepillado((b[0] + 46, b[1] + 8, b[2] + 34), t, 9, 3.0)
        obj_t = (450, 1230 + math.sin(t) * 3, 190)
        pb, ib = self._nino(t, 15, {"cara": "abierta", "cabeceo": 0.5, "giro": 0.2, "mano_der": obj_c, "mano_izq": obj_t}, pos)
        prims = pb + cepillo(self.mano(ib), (b[0] + 4, b[1] - 2, b[2] + 10), (90, 140, 230)) + telefono(self.mano(ib, "izq"))
        return self.componer(t, "manana", prims, tarjetas=[self._tarjeta_hijo()])

    # 13 · De noche, mamá sola se toca la mejilla con dolor; tacha la fecha de su tarjeta y escribe otra
    def escena_13(self, t, e):
        lt = t - e["inicio"]
        x, y, z = self.POS_MAMA
        mejilla = (x - 100, y + 30, 120)
        marco = (MARCO[0] - 10, MARCO[1] + 170, 330)
        if lt < 1.3:
            pose = {"cara": "dolor", "giro": -0.2, "mano_izq": mejilla}
        else:
            obj = kf(lt, [(1.3, (x + 80, y + 560, 0)), (1.9, marco)])
            if lt > 2.1:
                obj = cepillado(marco, t, 10, 2.2)
            pose = {"cara": "neutral", "giro": 0.6, "mano_der": obj}
        pm, im = self._mama(t, 45, pose)
        prims = pm
        if lt > 1.9:
            m = self.mano(im)
            prims = prims + [J.Capsula((m[0], m[1] - 4, m[2] + 6), (m[0] + 30, m[1] - 64, m[2] + 14), 6, (230, 190, 70), brillo=0.4)]
        tarjeta = sprite_tarjeta(GE.TARJETA_MAMA["titulo"], (), tuple(GE.TARJETA_MAMA["tachadas"]), GE.TARJETA_MAMA["nueva"] if lt > 2.8 else None)
        return self.componer(t, "noche", prims, tarjetas=[self._tarjeta_hijo(), (tarjeta, 1.0, 1)])

    # 14 · «25 AÑOS DESPUÉS» (el baño vacío)
    def escena_14(self, t, e):
        return self.componer(t, "manana", [], tarjetas=[(self._tarjeta_mama_vieja(), 1.0, 1)])

    # 15 · Tú (32) le cepillas los dientes a tu hija (3) y cuentas con los dedos; mamá (58) mira desde la puerta
    def escena_15(self, t, e):
        lt = t - e["inicio"]
        pn, inf = self._nieta(t, {"cara": "abierta"}, self.POS_PEQUENO)
        objetivo, punta = self.objetivo_cepillar(t, inf)
        dedos = 1 + int(limitar((lt - 0.5) / 0.6, 0, 4))
        ph, ih = self._adulto(t, {"cara": "sonrisa", "giro": 0.35, "mano_der": objetivo, "mano_izq": (330, 730 + math.sin(t * 5) * 5, 80), "dedos_izq": dedos})
        pm, _ = self._mama_puerta(t, 58, {"cara": "sonrisa", "giro": -0.3})
        prims = ph + pn + cepillo(self.mano(ih), punta, (240, 120, 150)) + pm
        return self.componer(t, "noche", prims, tarjetas=[(self._tarjeta_mama_vieja(), 1.0, 1)])

    # 16 · Abres el cajón buscando hilo dental y encuentras la cajita: tus 20 dientes de leche (acercamiento)
    def escena_16(self, t, e):
        lt = t - e["inicio"]
        manos_y = kf(lt, [(0.0, 1600), (0.8, 1600), (1.6, 1300)])
        ph, ih = self._adulto(t, {"cara": "sorpresa" if lt > 1.6 else "neutral", "cabeceo": kf(lt, [(0.0, 0.8), (1.6, 0.6)]), "mano_izq": (390, manos_y, 150), "mano_der": (600, manos_y, 150)})
        pm, _ = self._mama_puerta(t, 58, {"cara": "sonrisa", "giro": -0.3})
        centro = ((self.mano(ih, "izq")[0] + self.mano(ih)[0]) / 2, (self.mano(ih, "izq")[1] + self.mano(ih)[1]) / 2)

        def caja(img):
            a = progreso(lt, 1.4, 1.9)
            if a > 0:
                pegar(img, sprite_lata_abierta(), centro[0], centro[1] - 70, escala=0.85, alfa=a)

        return self.componer(t, "noche", ph + pm, encima=caja, tarjetas=[(self._tarjeta_mama_vieja(), 1.0, 1)])

    # 17 · Mamá se ríe desde la puerta tapándose la boca; en el marco sigue su tarjeta vieja
    def escena_17(self, t, e):
        lt = t - e["inicio"]
        x, y = self.POS_PUERTA
        R = 74 * self.ESC_PUERTA
        boca_m = (x - R * 0.1, y + R * 0.35, self.Z_PUERTA + R * 1.4)
        abajo = (x - R, y + R * 4.4, self.Z_PUERTA)
        mano = kf(lt, [(0.0, None), (0.01, abajo), (0.4, boca_m), (3.4, boca_m), (3.9, abajo), (3.91, None)])
        pm, _ = self._mama_puerta(t, 58, {"cara": "tapada", "giro": -0.2, "mano_izq": mano})
        ph, ih = self._adulto(t, {"cara": "sonrisa", "giro": 0.5, "mano_izq": (400, 1300, 150), "mano_der": (590, 1300, 150)})
        centro = ((self.mano(ih, "izq")[0] + self.mano(ih)[0]) / 2, (self.mano(ih, "izq")[1] + self.mano(ih)[1]) / 2)

        def caja(img):
            pegar(img, sprite_lata_abierta(), centro[0], centro[1] - 70, escala=0.85)

        return self.componer(t, "noche", ph + pm, encima=caja, tarjetas=[(self._tarjeta_mama_vieja(), 0.85 + 0.15 * math.sin(lt * 3), 1)])

    # 18 · Metes en el marco una tarjeta nueva; ella la lee y sonríe enseñando los dientes por primera vez
    def escena_18(self, t, e):
        lt = t - e["inicio"]
        marco = (POS_TARJETAS[2][0], POS_TARJETAS[2][1] + 90, 340)
        obj = kf(lt, [(0.0, (560, 1330, 0)), (0.4, (560, 1330, 0)), (1.0, marco), (1.5, marco), (2.0, (560, 1330, 0))])
        ph, ih = self._adulto(t, {"cara": "sonrisa", "giro": kf(lt, [(0.3, 0.2), (0.9, 0.75), (2.0, 0.4)]), "mano_der": obj})
        acerca = suave(progreso(lt, 1.6, 2.8)) * 0.35
        cara = "neutral" if lt < 2.9 else "dientes"
        pm, _ = self._mama_puerta(t, 58, {"cara": cara, "giro": 0.5 if lt < 2.9 else 0.1, "cabeceo": -0.2 if lt < 2.9 else 0.0, "piernas": "caminando" if 1.6 < lt < 2.8 else "de_pie", "paso": lt * 7}, acerca, destino=(760, 800, -300))
        prims = ph + pm + (tarjeta3d(self.mano(ih)) if 0.3 < lt < 1.4 else [])
        nueva = sprite_tarjeta(GE.TARJETA_NUEVA["titulo"], tuple(GE.TARJETA_NUEVA["lineas"]), color=(255, 246, 228))
        return self.componer(t, "noche", prims, tarjetas=[(self._tarjeta_mama_vieja(), 1.0, 1), (nueva, progreso(lt, 1.2, 1.5), 2)])

    # ================================================================ CIERRE
    @functools.lru_cache(maxsize=1)
    def _fondo_cierre(self):
        e = self.escenas[18]
        img = self.cuadro(e["fin"] - 1 / FPS, e)
        img = img.filter(ImageFilter.GaussianBlur(18))
        img.alpha_composite(Image.new("RGBA", (W, H), (8, 6, 4, 165)))
        return img

    def cierre(self, t):
        img = self._fondo_cierre().copy()
        k_texto = progreso(t, GE.CIERRE_INICIO + 0.2, GE.CIERRE_INICIO + 0.9)
        k_logo = progreso(t, GE.LOGO_INICIO, GE.LOGO_INICIO + 0.4)
        oscuro = progreso(t, GE.LOGO_INICIO - 0.2, GE.LOGO_INICIO + 0.4)
        if oscuro > 0:
            img.alpha_composite(Image.new("RGBA", (W, H), (4, 3, 2, int(150 * oscuro))))
        y = 720
        for i, clave in enumerate(GE.CIERRE_LINEAS):
            spr = sprite_cierre(clave)
            a = progreso(t, GE.CIERRE_INICIO + 0.2 + i * 0.35, GE.CIERRE_INICIO + 0.7 + i * 0.35) * (1 - 0.75 * oscuro)
            pegar(img, spr, W / 2, y + spr.height / 2, alfa=a)
            y += spr.height + (26 if i else 34)
        if k_logo > 0:
            color = t < GE.LOGO_INICIO + 1.3
            pegar(img, G.icono_instagram(130, color), W / 2, 930, alfa=k_logo, escala=0.9 + 0.1 * salida_cubica(k_logo))
            usuario = G.texto_espaciado(config.MARCA["usuario"].upper(), "condensada", 40, (255, 255, 255), 0.04)
            pegar(img, usuario, W / 2, 1040, alfa=k_logo)
        del k_texto
        return img
