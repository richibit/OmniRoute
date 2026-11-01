"""
Utilidades compartidas: errores, curvas de animación, tipografías, render de
texto (con símbolos y emoji) y operaciones básicas de composición de imágenes.
"""

import functools
import math
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config


class ErrorProduccion(Exception):
    """Error con un mensaje pensado para la persona que produce el video."""


# ---------------------------------------------------------------------------
# Curvas de animación
# ---------------------------------------------------------------------------
def limitar(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def progreso(t, t0, t1):
    """0 antes de t0, 1 después de t1 y lineal entre ambos."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return limitar((t - t0) / (t1 - t0))


def suave(x):
    x = limitar(x)
    return x * x * (3 - 2 * x)


def salida_cubica(x):
    x = limitar(x)
    return 1 - (1 - x) ** 3


def entrada_cubica(x):
    x = limitar(x)
    return x**3


def salida_rebote(x, s=1.70158):
    """Ease-out-back: se pasa un poco y vuelve (efecto «pop»)."""
    x = limitar(x) - 1
    return x * x * ((s + 1) * x + s) + 1


def mezclar_color(c1, c2, t):
    t = limitar(t)
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def aleatorio(semilla):
    """Generador determinista: cada fotograma depende solo del tiempo, nunca del orden de render."""
    return np.random.default_rng(semilla)


# ---------------------------------------------------------------------------
# Normalización de palabras (para buscar «palabras ancla» en la narración)
# ---------------------------------------------------------------------------
def normalizar_palabra(p):
    p = unicodedata.normalize("NFD", p.lower())
    p = "".join(c for c in p if unicodedata.category(c) != "Mn")
    return "".join(c for c in p if c.isalnum())


# ---------------------------------------------------------------------------
# Tipografías
# ---------------------------------------------------------------------------
def _fijar_peso(fuente, peso):
    """En fuentes variables (p. ej. Montserrat[wght].ttf) fija el eje de peso."""
    try:
        ejes = fuente.get_variation_axes()
    except Exception:
        return fuente  # fuente estática
    valores = []
    for eje in ejes:
        nombre = eje.get("name", b"")
        nombre = nombre.decode() if isinstance(nombre, bytes) else str(nombre)
        if nombre.lower().startswith("weight") or nombre.lower() == "wght":
            valores.append(limitar(peso, eje["minimum"], eje["maximum"]))
        elif nombre.lower().startswith("optical"):
            valores.append(eje["maximum"])  # tamaño óptico de titular
        else:
            valores.append(eje["default"])
    try:
        fuente.set_variation_by_axes(valores)
    except Exception:
        pass
    return fuente


@functools.lru_cache(maxsize=None)
def ruta_fuente(rol):
    candidatos, _ = config.FUENTES[rol]
    for c in candidatos:
        try:
            ImageFont.truetype(str(c), 20)
            return str(c)
        except OSError:
            continue
    return None


@functools.lru_cache(maxsize=None)
def fuente(rol, tam):
    ruta = ruta_fuente(rol)
    if ruta is None:
        # Último recurso: fuente integrada de Pillow (fea pero siempre disponible).
        return ImageFont.load_default(size=tam)
    f = ImageFont.truetype(ruta, tam)
    return _fijar_peso(f, config.FUENTES[rol][1])


@functools.lru_cache(maxsize=None)
def ruta_emoji():
    for c in config.FUENTE_EMOJI:
        try:
            ImageFont.truetype(str(c), 109)
            return str(c)
        except OSError:
            continue
    return None


SIMBOLOS = set("✓✗→↓↑≈₂")
SUSTITUTOS_EMOJI = {"👇": "↓", "😬": ""}


def es_emoji(ch):
    return ord(ch) >= 0x1F000


@functools.lru_cache(maxsize=256)
def sprite_emoji(ch, alto):
    """Emoji a color (Noto Color Emoji solo existe a 109 px: se dibuja ahí y se escala)."""
    ruta = ruta_emoji()
    if ruta is None:
        return None
    try:
        f = ImageFont.truetype(ruta, 109)
        img = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((10, 10), ch, font=f, embedded_color=True)
        caja = img.getbbox()
        if not caja:
            return None
        img = img.crop(caja)
        ancho = max(1, int(img.width * alto / img.height))
        return img.resize((ancho, alto), Image.LANCZOS)
    except Exception:
        return None


def _tramos(texto):
    """Divide un texto en tramos según la fuente necesaria: normal, símbolo o emoji."""
    tramos = []
    for ch in texto:
        if ch == "️":
            continue
        if es_emoji(ch):
            tipo = "emoji"
        elif ch in SIMBOLOS:
            tipo = "simbolo"
        else:
            tipo = "normal"
        if tramos and tramos[-1][0] == tipo and tipo != "emoji":
            tramos[-1][1] += ch
        else:
            tramos.append([tipo, ch])
    return tramos


def medir_texto(texto, rol, tam):
    """Ancho en píxeles de un texto de una línea, contando símbolos y emoji."""
    ancho = 0.0
    for tipo, t in _tramos(texto):
        if tipo == "emoji":
            spr = sprite_emoji(t, int(tam * 0.95))
            ancho += spr.width + tam * 0.1 if spr else fuente("simbolos", tam).getlength(SUSTITUTOS_EMOJI.get(t, ""))
        else:
            f = fuente("simbolos" if tipo == "simbolo" else rol, tam)
            ancho += f.getlength(t)
    return ancho


def envolver(texto, rol, tam, max_ancho):
    """Reparte un texto en líneas que no superen max_ancho (respeta los saltos manuales)."""
    lineas = []
    for parrafo in texto.split("\n"):
        actual = ""
        for palabra in parrafo.split(" "):
            prueba = (actual + " " + palabra).strip()
            if actual and medir_texto(prueba, rol, tam) > max_ancho:
                lineas.append(actual)
                actual = palabra
            else:
                actual = prueba
        lineas.append(actual)
    return lineas


def render_lineas(
    lineas,
    rol,
    tam,
    color=(255, 255, 255),
    contorno=0,
    color_contorno=(0, 0, 0),
    sombra=0,
    interlineado=1.12,
    alineacion="centro",
):
    """
    Dibuja texto multilínea en una imagen RGBA recortada.

    `lineas` es una lista donde cada línea es un str o una lista de segmentos
    (texto, color) para colorear palabras distintas (subtítulos con palabra activa).
    """
    f = fuente(rol, tam)
    asc, desc = f.getmetrics()
    alto_linea = int((asc + desc) * interlineado)
    segs_por_linea = [[(l, color)] if isinstance(l, str) else l for l in lineas]
    anchos = [sum(medir_texto(s, rol, tam) for s, _ in segs) for segs in segs_por_linea]
    pad = contorno + sombra * 2 + 6
    # espacio real por encima de la línea base de la primera línea: los acentos de las mayúsculas
    # (Ñ, É, Á…) sobresalen de la altura nominal de la fuente y no deben recortarse
    medidor = ImageDraw.Draw(Image.new("L", (1, 1)))
    alto_real = asc
    for texto, _ in segs_por_linea[0] if segs_por_linea else []:
        for tipo, t in _tramos(texto):
            if tipo != "emoji" and t:
                caja = medidor.textbbox((0, 0), t, font=fuente("simbolos" if tipo == "simbolo" else rol, tam), anchor="ls")
                alto_real = max(alto_real, -caja[1])
    extra_arriba = int(math.ceil(alto_real - asc))
    W = int(max(anchos + [1]) + pad * 2)
    H = int(alto_linea * len(lineas) + pad * 2 + desc + extra_arriba)
    capa = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    emojis = []
    for i, segs in enumerate(segs_por_linea):
        if alineacion == "centro":
            x = pad + (W - 2 * pad - anchos[i]) / 2
        else:
            x = pad
        base_y = pad + extra_arriba + i * alto_linea + asc
        for texto, col in segs:
            for tipo, t in _tramos(texto):
                if tipo == "emoji":
                    spr = sprite_emoji(t, int(tam * 0.95))
                    if spr is not None:
                        emojis.append((spr, int(x + tam * 0.05), int(base_y - asc * 0.88)))
                        x += spr.width + tam * 0.1
                        continue
                    t = SUSTITUTOS_EMOJI.get(t, "")
                    tipo = "simbolo"
                ff = fuente("simbolos" if tipo == "simbolo" else rol, tam)
                d.text(
                    (x, base_y),
                    t,
                    font=ff,
                    fill=tuple(col) + ((255,) if len(col) == 3 else ()),
                    anchor="ls",
                    stroke_width=contorno,
                    stroke_fill=tuple(color_contorno),
                )
                x += ff.getlength(t)
    if sombra:
        alfa = capa.getchannel("A").filter(ImageFilter.GaussianBlur(sombra))
        alfa = alfa.point(lambda v: int(v * 0.55))
        sombra_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sombra_img.putalpha(alfa)
        fondo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        fondo.alpha_composite(sombra_img, (0, int(sombra * 0.8)))
        fondo.alpha_composite(capa)
        capa = fondo
    for spr, ex, ey in emojis:
        capa.alpha_composite(spr, (max(0, ex), max(0, ey)))
    return capa


# ---------------------------------------------------------------------------
# Composición de imágenes
# ---------------------------------------------------------------------------
def con_alfa(img, alfa):
    """Copia de una imagen RGBA con su transparencia multiplicada por `alfa` (0–1)."""
    if alfa >= 0.999:
        return img
    img = img.copy()
    a = img.getchannel("A").point(lambda v: int(v * alfa))
    img.putalpha(a)
    return img


def pegar(base, spr, cx, cy, escala=1.0, alfa=1.0, ancla="centro", remuestreo=Image.BILINEAR):
    """
    Compone `spr` (RGBA) sobre `base` (RGBA) con su centro en (cx, cy).
    Recorta automáticamente lo que queda fuera del lienzo.
    """
    if spr is None or alfa <= 0.003 or escala <= 0.001:
        return
    if abs(escala - 1.0) > 1e-3:
        w = max(1, int(round(spr.width * escala)))
        h = max(1, int(round(spr.height * escala)))
        spr = spr.resize((w, h), remuestreo)
    spr = con_alfa(spr, alfa)
    if ancla == "centro":
        x0, y0 = int(round(cx - spr.width / 2)), int(round(cy - spr.height / 2))
    else:  # esquina superior izquierda
        x0, y0 = int(round(cx)), int(round(cy))
    x1, y1 = x0 + spr.width, y0 + spr.height
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(base.width, x1), min(base.height, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
        spr = spr.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0))
    base.alpha_composite(spr, (cx0, cy0))


def gradiente_vertical(w, h, c1, c2, curva=1.0):
    t = (np.linspace(0, 1, h) ** curva)[:, None, None]
    a = np.array(c1, dtype=np.float32)[None, None, :3]
    b = np.array(c2, dtype=np.float32)[None, None, :3]
    arr = a + (b - a) * t
    arr = np.repeat(arr, w, axis=1)
    return Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")


def gradiente_radial(w, h, centro, radio, c_dentro, c_fuera):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - centro[0]) ** 2 + (yy - centro[1]) ** 2) / radio
    t = np.clip(d, 0, 1)[..., None]
    a = np.array(c_dentro, dtype=np.float32)[:3]
    b = np.array(c_fuera, dtype=np.float32)[:3]
    arr = a + (b - a) * (t * t * (3 - 2 * t))
    return Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")


@functools.lru_cache(maxsize=64)
def disco_suave(radio, color, dureza=0.5, alfa=255):
    """Círculo con borde difuminado (bokeh, brillos, resplandores)."""
    r = max(1, int(radio))
    tam = r * 2 + 4
    yy, xx = np.mgrid[0:tam, 0:tam].astype(np.float32)
    d = np.sqrt((xx - tam / 2) ** 2 + (yy - tam / 2) ** 2) / r
    borde = max(0.02, 1 - dureza)
    a = np.clip((1 - d) / borde, 0, 1)
    a = a * a * (3 - 2 * a) * alfa
    img = np.zeros((tam, tam, 4), np.uint8)
    img[..., :3] = color[:3]
    img[..., 3] = a.astype(np.uint8)
    return Image.fromarray(img, "RGBA")


def desenfoque_movimiento(img, desplazamiento, pasos=7):
    """Desenfoque horizontal barato (whip pan): promedia copias desplazadas."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    acum = np.zeros_like(arr)
    for i in range(pasos):
        dx = int(round(desplazamiento * (i / (pasos - 1) - 0.5)))
        acum += np.roll(arr, dx, axis=1)
    return Image.fromarray((acum / pasos).astype(np.uint8), "RGB").convert("RGBA")


def punto_en_circulo(cx, cy, r, angulo_grados):
    a = math.radians(angulo_grados - 90)
    return cx + r * math.cos(a), cy + r * math.sin(a)
