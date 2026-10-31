"""
Biblioteca de gráficos procedurales (sin imágenes externas) con la estética
de la fórmula: fondos oscuros con foco de luz, partículas, viñeta y resplandor;
dientes, placa, película salival, bacterias, cristales de azúcar, medidor de pH,
lengua, persona despertando, grupo de 1965, calendario, cepillo, teléfono con
menú «Enviar a», y los elementos gráficos de los explicadores (icono de
Instagram, píldoras con línea guía, tarjetas de dato, barras comparativas,
dibujos de línea que se trazan solos, insignias ✗/✓ y línea de tiempo).

Cada sprite se dibuja a 2–4× su tamaño y se reduce con LANCZOS para obtener
bordes suaves. Los sprites estáticos se guardan en caché.
"""

import functools
import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

import config
from utilidades import (
    aleatorio,
    con_alfa,
    disco_suave,
    fuente,
    gradiente_radial,
    gradiente_vertical,
    limitar,
    mezclar_color,
    pegar,
    punto_en_circulo,
    render_lineas,
)

P = config.PALETA
W, H = config.ANCHO, config.ALTO


def _reducir(img, tam):
    return img.resize((max(1, int(tam[0])), max(1, int(tam[1]))), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Fondos cinematográficos
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=16)
def fondo_foco(c_centro, c_borde, cx=540, cy=980, radio=1350):
    """Fondo oscuro con un foco de luz suave (base de casi todas las escenas)."""
    return gradiente_radial(W, H, (cx, cy), radio, c_centro, c_borde)


@functools.lru_cache(maxsize=1)
def vineta():
    """Oscurecimiento de bordes que concentra la mirada en el centro."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.78)) ** 2 + ((yy - H * 0.52) / (H * 0.62)) ** 2)
    a = np.clip((d - 0.55) / 0.65, 0, 1) ** 1.5 * 215
    img = np.zeros((H, W, 4), np.uint8)
    img[..., 3] = a.astype(np.uint8)
    return Image.fromarray(img, "RGBA")


@functools.lru_cache(maxsize=8)
def _particulas(n, semilla):
    rng = aleatorio(semilla)
    out = []
    for _ in range(n):
        z = rng.uniform(0.3, 1.0)  # profundidad: las cercanas son más grandes y rápidas
        out.append(
            (
                rng.uniform(0, W),
                rng.uniform(0, H),
                max(2, int(2 + 9 * z**2 + rng.uniform(0, 3))),
                rng.uniform(-6, 6) * z,
                rng.uniform(-22, -6) * z,
                rng.uniform(0, 6.28),
                rng.uniform(0.25, 0.75) * z,
                z,
            )
        )
    return tuple(out)


def polvo(lienzo, t, color=(190, 215, 255), n=55, semilla=1, intensidad=1.0, deriva=(0.0, 0.0)):
    """Partículas de polvo/luz que flotan (dan profundidad y vida a los fondos oscuros)."""
    for x0, y0, r, vx, vy, fase, a, z in _particulas(n, semilla):
        x = (x0 + vx * t + deriva[0] * z) % (W + 80) - 40
        y = (y0 + vy * t + deriva[1] * z) % (H + 80) - 40
        alfa = a * (0.55 + 0.45 * math.sin(t * 1.3 + fase)) * intensidad
        pegar(lienzo, disco_suave(r, color, 0.3 if r > 6 else 0.6, 255), x, y, alfa=alfa)


_GLOWS = {}


def resplandor(clave, sprite, radio, color, fuerza=1.0):
    """Halo de luz del color dado con la forma del sprite (en caché por clave)."""
    k = (clave, radio, color, round(fuerza, 2))
    if k not in _GLOWS:
        pad = radio * 3
        base = Image.new("L", (sprite.width + pad * 2, sprite.height + pad * 2), 0)
        base.paste(sprite.getchannel("A"), (pad, pad))
        a = base.filter(ImageFilter.GaussianBlur(radio)).point(lambda v: int(min(255, v * fuerza)))
        img = Image.new("RGBA", base.size, color + (0,))
        img.putalpha(a)
        _GLOWS[k] = img
    return _GLOWS[k]


# ---------------------------------------------------------------------------
# Diente (incisivo central superior, vista frontal)
# ---------------------------------------------------------------------------
def contorno_diente(w, h, n=180, x0=0.0, y0=0.0):
    """Superelipse estrechada hacia el cuello: corona de incisivo más ancha en el borde incisal."""
    pts = []
    p = 3.2
    for i in range(n):
        th = 2 * math.pi * i / n
        c, s = math.cos(th), math.sin(th)
        x = math.copysign(abs(c) ** (2 / p), c)
        y = math.copysign(abs(s) ** (2 / p), s)
        v = (y + 1) / 2
        pts.append((x0 + (0.5 + x * 0.40 * (0.80 + 0.20 * v)) * w, y0 + (0.56 + y * 0.44) * h))
    return pts


@functools.lru_cache(maxsize=8)
def diente(ancho, alto):
    """Devuelve (sprite RGBA, máscara L del esmalte visible)."""
    ss = 2
    Ws, Hs = ancho * ss, alto * ss
    m = Image.new("L", (Ws, Hs), 0)
    ImageDraw.Draw(m).polygon(contorno_diente(Ws, Hs), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(1.2 * ss))
    yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
    u, v = xx / Ws, yy / Hs
    esmalte = np.array(P["esmalte"], np.float32)
    lateral = 1 - 0.34 * np.clip(np.abs(u - 0.5) / 0.42, 0, 1) ** 2.2
    col = esmalte[None, None, :] * lateral[..., None]
    calido = np.array([238, 224, 196], np.float32)
    k_c = (np.clip((0.48 - v) / 0.25, 0, 1) * 0.6)[..., None]
    col = col * (1 - k_c) + calido * lateral[..., None] * k_c
    incisal = np.array([175, 198, 222], np.float32)
    k_i = (np.clip((v - 0.80) / 0.18, 0, 1) ** 1.5 * 0.7)[..., None]
    col = col * (1 - k_i) + incisal * k_i
    col *= (1 - 0.02 * (np.sin(v * Hs / (5.0 * ss)) > 0.7))[..., None]
    brillo = Image.new("L", (Ws, Hs), 0)
    d = ImageDraw.Draw(brillo)
    d.ellipse([Ws * 0.26, Hs * 0.30, Ws * 0.39, Hs * 0.80], fill=210)
    d.ellipse([Ws * 0.58, Hs * 0.36, Ws * 0.66, Hs * 0.62], fill=120)
    d.ellipse([Ws * 0.30, Hs * 0.86, Ws * 0.62, Hs * 0.92], fill=90)
    b = np.asarray(brillo.filter(ImageFilter.GaussianBlur(Ws * 0.03)), np.float32)[..., None] / 255
    col = col + (255 - col) * b * 0.85
    alfa_e = np.asarray(m, np.float32) / 255
    limite_encia = 0.21 + 0.16 * ((u - 0.5) / 0.48) ** 2
    dentro = (((u - 0.5) / 0.47) ** 2 + ((v - 0.14) / 0.17) ** 2) < 1
    mg = Image.fromarray(((v < limite_encia) * dentro * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4 * ss))
    mg = np.asarray(mg, np.float32) / 255
    encia = np.array(P["encia"], np.float32)
    col_encia = encia * (0.72 + 0.28 * np.clip(v / 0.35, 0, 1)[..., None])
    banda = np.exp(-(((v - limite_encia + 0.035) / 0.02) ** 2))[..., None]
    col_encia = col_encia + (255 - col_encia) * banda * 0.35
    rgb = col * (alfa_e * (1 - mg))[..., None] + col_encia * mg[..., None]
    alfa = np.maximum(alfa_e, mg * (v < 0.42))
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(alfa[..., None] > 0, rgb / np.maximum(alfa[..., None], 1e-3), 0)
    arr = np.dstack([np.clip(rgb, 0, 255), alfa * 255]).astype(np.uint8)
    sprite = _reducir(Image.fromarray(arr, "RGBA"), (ancho, alto))
    mascara = _reducir(Image.fromarray(((alfa_e * (1 - mg)) * 255).astype(np.uint8), "L"), (ancho, alto))
    return sprite, mascara


@functools.lru_cache(maxsize=8)
def placa(ancho, alto):
    """Biopelícula: más gruesa junto a la encía y en los bordes, opaca y brillante."""
    _, mascara = diente(ancho, alto)
    rng = aleatorio(7)
    ruido = Image.fromarray((rng.random((alto // 14 + 2, ancho // 14 + 2)) * 255).astype(np.uint8))
    ruido = np.asarray(ruido.resize((ancho, alto), Image.BICUBIC), np.float32) / 255
    yy, xx = np.mgrid[0:alto, 0:ancho].astype(np.float32)
    u, v = xx / ancho, yy / alto
    grosor = 0.45 + 0.5 * np.clip((0.55 - v) / 0.3, 0, 1) + 0.3 * np.clip((np.abs(u - 0.5) - 0.25) / 0.15, 0, 1)
    grosor = np.clip(grosor + (ruido - 0.5) * 0.5, 0, 1)
    rgb = np.array([238, 222, 170], np.float32) * (1 - grosor[..., None] * 0.6) + np.array([212, 186, 118], np.float32) * grosor[..., None] * 0.6
    brillo = Image.new("L", (ancho, alto), 0)
    d = ImageDraw.Draw(brillo)
    for _ in range(14):
        x, y = rng.uniform(0.2, 0.8) * ancho, rng.uniform(0.25, 0.85) * alto
        r = rng.uniform(0.015, 0.04) * ancho
        d.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6], fill=int(rng.uniform(120, 230)))
    b = np.asarray(brillo.filter(ImageFilter.GaussianBlur(ancho * 0.012)), np.float32)[..., None] / 255
    rgb = rgb + (255 - rgb) * b * 0.8
    alfa = np.asarray(mascara, np.float32) / 255 * (0.55 + 0.4 * grosor)
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alfa * 255]).astype(np.uint8), "RGBA")


def pelicula(ancho, alto, radio_rel, centro=(0.5, 0.40), alfa=0.45):
    """Película adquirida: capa azul translúcida que se extiende como una gota sobre el esmalte."""
    _, mascara = diente(ancho, alto)
    peq = Image.new("L", (ancho // 4, alto // 4), 0)
    cx, cy = centro[0] * ancho / 4, centro[1] * alto / 4
    r = radio_rel * max(ancho, alto) / 4
    ImageDraw.Draw(peq).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    anillo = Image.new("L", peq.size, 0)
    ImageDraw.Draw(anillo).ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=3)
    peq = peq.resize((ancho, alto), Image.BILINEAR)
    anillo = anillo.resize((ancho, alto), Image.BILINEAR)
    cuerpo = ImageChops.multiply(peq, mascara).point(lambda x: int(x * alfa))
    borde = ImageChops.multiply(anillo, mascara).point(lambda x: int(x * 0.8))
    capa = Image.new("RGBA", (ancho, alto), (130, 200, 255, 0))
    capa.putalpha(ImageChops.lighter(cuerpo, borde))
    return capa


# ---------------------------------------------------------------------------
# Destellos, esferas, bacterias, azúcar
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=16)
def destello(tam, color=(255, 255, 255)):
    ss = 3
    s = tam * ss
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = s / 2
    for ang, largo, grosor in ((0, 0.5, 0.07), (90, 0.5, 0.07), (45, 0.28, 0.05), (135, 0.28, 0.05)):
        a = math.radians(ang)
        dx, dy, px, py = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
        d.polygon(
            [
                (c + dx * largo * s, c + dy * largo * s),
                (c + px * grosor * s, c + py * grosor * s),
                (c - dx * largo * s, c - dy * largo * s),
                (c - px * grosor * s, c - py * grosor * s),
            ],
            fill=color + (255,),
        )
    img = _reducir(img, (tam, tam))
    pegar(img, disco_suave(tam // 3, color, 0.1, 160), tam / 2, tam / 2)
    return img


@functools.lru_cache(maxsize=64)
def esfera(r, color, alfa=255, cara=False):
    """Esfera brillante sombreada (proteínas salivales, cocos)."""
    ss = 3
    rs = r * ss
    tam = rs * 2 + 4
    yy, xx = np.mgrid[0:tam, 0:tam].astype(np.float32)
    nx, ny = (xx - tam / 2) / rs, (yy - tam / 2) / rs
    d2 = nx**2 + ny**2
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    luz = np.array([-0.45, -0.6, 0.66])
    luz /= np.linalg.norm(luz)
    sombreado = 0.40 + 0.60 * np.clip(nx * luz[0] + ny * luz[1] + nz * luz[2], 0, 1)
    especular = np.exp(-(((nx + 0.38) ** 2 + (ny + 0.42) ** 2) / 0.035))
    rim = np.clip((np.sqrt(d2) - 0.75) / 0.25, 0, 1) ** 2 * (np.sqrt(d2) <= 1)  # luz de contorno
    c = np.array(color, np.float32)
    rgb = c * sombreado[..., None]
    rgb = rgb + (255 - rgb) * (especular[..., None] * 0.85 + rim[..., None] * 0.25)
    borde = np.clip((1 - np.sqrt(d2)) * rs, 0, 1)
    img = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), borde * alfa]).astype(np.uint8), "RGBA")
    if cara:
        d = ImageDraw.Draw(img)
        oscuro = (12, 50, 56, 255)
        ojo = rs * 0.13
        for sx in (-0.32, 0.32):
            ex, ey = tam / 2 + sx * rs, tam / 2 - 0.08 * rs
            d.ellipse([ex - ojo, ey - ojo * 1.25, ex + ojo, ey + ojo * 1.25], fill=oscuro)
        d.line(
            [(tam / 2 - rs * 0.22, tam / 2 + rs * 0.34), (tam / 2 + rs * 0.22, tam / 2 + rs * 0.34)],
            fill=oscuro,
            width=max(2, int(rs * 0.09)),
        )
    return _reducir(img, (r * 2 + 2, r * 2 + 2))


@functools.lru_cache(maxsize=32)
def cadena_cocos(n, r, semilla, color=None):
    """Cadena de estreptococos (cocos en fila) con una cara neutra, nunca monstruosa."""
    rng = aleatorio(semilla)
    base = color or P["turquesa"]
    paso = r * 1.72
    ancho = int(paso * (n - 1) + r * 2 + 8)
    alto = int(r * 2 + 8 + r * 1.2)
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    fase = rng.uniform(0, 6)
    tonos = [mezclar_color(base, (120, 230, 215), rng.uniform(0, 0.45)) for _ in range(n)]
    for i in range(n):
        pegar(img, esfera(r, tonos[i], 255, cara=(i == n // 2)), r + 4 + i * paso, alto / 2 + math.sin(fase + i * 0.7) * r * 0.45)
    return img


@functools.lru_cache(maxsize=8)
def paracaidas(ancho):
    alto = int(ancho * 0.9)
    ss = 2
    img = Image.new("RGBA", (ancho * ss, alto * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.pieslice([0, 0, ancho * ss, alto * ss * 0.7], 180, 360, fill=(255, 255, 255, 150))
    for k in range(1, 4):
        x = ancho * ss * k / 4
        d.line([(x, alto * ss * 0.35), (x, alto * ss * 0.12)], fill=P["menta"] + (180,), width=3 * ss)
    for x in (0.03, 0.5, 0.97):
        d.line([(ancho * ss * x, alto * ss * 0.35), (ancho * ss * 0.5, alto * ss)], fill=(255, 255, 255, 130), width=2 * ss)
    return _reducir(img, (ancho, alto))


@functools.lru_cache(maxsize=48)
def cubo_azucar(tam, angulo_cuantizado):
    """Cristal de azúcar (cubo isométrico translúcido)."""
    ss = 3
    s = tam * ss
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c, a = s / 2, s * 0.32
    arriba = [(c, c - a), (c + a, c - a / 2), (c, c), (c - a, c - a / 2)]
    izq = [(c - a, c - a / 2), (c, c), (c, c + a), (c - a, c + a / 2)]
    der = [(c, c), (c + a, c - a / 2), (c + a, c + a / 2), (c, c + a)]
    d.polygon(arriba, fill=(255, 255, 255, 245))
    d.polygon(izq, fill=(214, 232, 245, 235))
    d.polygon(der, fill=(186, 210, 230, 235))
    for cara in (arriba, izq, der):
        d.line(cara + [cara[0]], fill=(255, 255, 255, 255), width=ss)
    return _reducir(img.rotate(angulo_cuantizado, resample=Image.BICUBIC), (tam, tam))


# ---------------------------------------------------------------------------
# Texturas microscópicas (versión oscura, tipo microscopio electrónico coloreado)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def textura_esmalte_superior(ancho=1300, alto=2200):
    """Superficie del esmalte vista al microscopio: relieve de prismas con luz de contorno cian."""
    img = Image.new("RGB", (ancho, alto), (120, 128, 140))
    d = ImageDraw.Draw(img)
    rng = aleatorio(21)
    celda = 64
    for fila in range(-1, alto // int(celda * 0.86) + 2):
        for col in range(-1, ancho // celda + 2):
            x = col * celda + (celda / 2 if fila % 2 else 0) + rng.uniform(-5, 5)
            y = fila * celda * 0.86 + rng.uniform(-5, 5)
            r = celda * 0.47
            tono = int(rng.uniform(150, 175))
            d.ellipse([x - r, y - r * 0.9, x + r, y + r * 0.9], fill=(tono, tono + 4, tono + 10), outline=(70, 80, 96), width=6)
            d.ellipse([x - r * 0.55, y + r * 0.25, x + r * 0.55, y + r * 0.95], fill=(tono - 30, tono - 26, tono - 18))
    img = img.filter(ImageFilter.GaussianBlur(2.2))
    arr = np.asarray(img, np.float32)
    gy, gx = np.gradient(arr.mean(axis=2))
    relieve = np.clip(-(gx + gy) * 2.4, -50, 60)
    arr = arr * 0.42 + relieve[..., None] * np.array([0.55, 0.95, 1.1])  # luz rasante cian
    arr = arr + np.array([8, 22, 38])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


@functools.lru_cache(maxsize=1)
def mundo_lateral(alto_mundo=2700, superficie=2160):
    """Corte lateral oscuro: saliva arriba (con partículas), película luminosa y esmalte (prismas) abajo."""
    img = gradiente_vertical(W, alto_mundo, (4, 12, 26), (10, 44, 62), curva=0.8)
    rng = aleatorio(31)
    for _ in range(70):
        r = int(rng.uniform(3, 18))
        pegar(img, disco_suave(r, (120, 220, 230), 0.25, int(rng.uniform(25, 80))), rng.uniform(0, W), rng.uniform(0, superficie))
    esmalte = Image.new("RGBA", (W, alto_mundo - superficie), (60, 66, 78, 255))
    d = ImageDraw.Draw(esmalte)
    x = 0
    while x < W:
        ancho = rng.uniform(22, 30)
        tono = int(rng.uniform(70, 96))
        d.rectangle([x, 0, x + ancho - 3, esmalte.height], fill=(tono, tono + 6, tono + 16, 255))
        x += ancho
    esmalte = esmalte.filter(ImageFilter.GaussianBlur(1.5))
    sombra = Image.new("RGBA", esmalte.size, (0, 0, 0, 0))
    sombra.putalpha(Image.fromarray((np.linspace(0, 170, esmalte.height)[:, None] * np.ones((1, W))).astype(np.uint8)))
    esmalte.alpha_composite(sombra)
    img.alpha_composite(esmalte, (0, superficie))
    brillo = Image.new("RGBA", (W, 60), (0, 0, 0, 0))
    ImageDraw.Draw(brillo).rectangle([0, 22, W, 40], fill=(190, 160, 255, 200))
    img.alpha_composite(brillo.filter(ImageFilter.GaussianBlur(7)), (0, superficie - 40))
    for i in range(70):
        pegar(img, esfera(7, (205, 185, 255)), (i + 0.5) * W / 70 + rng.uniform(-4, 4), superficie - 12 + rng.uniform(-5, 5))
    return img


@functools.lru_cache(maxsize=1)
def textura_lengua():
    """Dorso de la lengua con papilas, iluminado por luz de luna."""
    ancho, alto = 1400, 1300
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([0, 120, ancho, alto + 700], fill=(150, 92, 128, 255))
    rng = aleatorio(51)
    papilas = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    dp = ImageDraw.Draw(papilas)
    for _ in range(2600):
        x, y = rng.uniform(0, ancho), rng.uniform(150, alto)
        r = rng.uniform(5, 11)
        dp.ellipse([x - r, y - r, x + r, y + r], fill=mezclar_color((176, 116, 152), (214, 160, 190), rng.uniform(0, 1)) + (255,))
        dp.ellipse([x - r * 0.4, y - r * 0.6, x + r * 0.2, y - r * 0.1], fill=(235, 200, 225, 200))
    papilas.putalpha(ImageChops.multiply(papilas.getchannel("A"), img.getchannel("A")))
    img.alpha_composite(papilas)
    surco = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    ImageDraw.Draw(surco).line([(ancho / 2, 200), (ancho / 2, alto)], fill=(90, 50, 80, 120), width=26)
    img.alpha_composite(surco.filter(ImageFilter.GaussianBlur(10)))
    luna = gradiente_vertical(ancho, alto, (60, 100, 200), (10, 16, 50))
    luna.putalpha(img.getchannel("A").point(lambda v: int(v * 0.58)))
    img.alpha_composite(luna)
    return img.filter(ImageFilter.GaussianBlur(0.8))


# ---------------------------------------------------------------------------
# Boca estilizada
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def boca(ancho=2400):
    """Boca abierta estilizada con arcada superior e inferior (grande para poder hacer zoom)."""
    alto = int(ancho * 0.62)
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, ancho, alto], fill=(200, 86, 104, 255))
    m = ancho * 0.07
    d.ellipse([m, m * 1.1, ancho - m, alto - m * 1.1], fill=(60, 14, 30, 255))
    d.ellipse([ancho * 0.18, alto * 0.55, ancho * 0.82, alto * 1.05], fill=(186, 84, 114, 255))
    d.rectangle([m, m * 1.1, ancho - m, alto * 0.33], fill=(214, 108, 124, 255))
    centro = ancho / 2
    posiciones, x_der = [], centro
    for i, a in enumerate([0.115, 0.095, 0.085, 0.075]):
        posiciones.append((x_der, a * ancho, i))
        x_der += a * ancho * 0.98
    for x_ini, w_d, i in posiciones:
        for lado in (1, -1):
            x0 = x_ini if lado == 1 else 2 * centro - x_ini - w_d
            h_d = w_d * (1.35 - i * 0.12)
            y0 = alto * 0.21 + i * alto * 0.012
            d.polygon(contorno_diente(w_d * 0.98, h_d, 80, x0, y0), fill=P["esmalte"] + (255,))
            d.ellipse([x0 + w_d * 0.25, y0 + h_d * 0.35, x0 + w_d * 0.38, y0 + h_d * 0.8], fill=(255, 255, 255, 255))
    d.rectangle([m, m * 1.1, ancho - m, alto * 0.25], fill=(214, 108, 124, 255))
    for x_ini, w_d, i in posiciones:
        for lado in (1, -1):
            x0 = x_ini if lado == 1 else 2 * centro - x_ini - w_d
            y0 = alto * 0.70 - i * alto * 0.01
            d.rounded_rectangle([x0 + w_d * 0.08, y0, x0 + w_d * 0.92, y0 + w_d * 0.8], radius=w_d * 0.25, fill=(236, 232, 222, 255))
    d.ellipse([0, 0, ancho, alto], outline=(200, 86, 104, 255), width=int(m * 1.15))
    return img.filter(ImageFilter.GaussianBlur(1.0))


# ---------------------------------------------------------------------------
# Medidor de pH
# ---------------------------------------------------------------------------
def color_ph(valor):
    return mezclar_color(P["ambar"], P["menta"], (valor - 5.5) / 1.5)


def medidor_ph(valor, alto=620, ancho=150, etiqueta_umbral=None):
    """Tubo vertical con escala 4–8; se llena hasta el pH actual; línea roja en 5,5."""
    ss = 2
    margen_der = 430 if etiqueta_umbral else 70
    Ws, Hs = (ancho + margen_der + 70) * ss, (alto + 40) * ss
    img = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 70 * ss, 20 * ss, (70 + ancho) * ss, (20 + alto) * ss

    def y_de(ph):
        return y1 - (ph - 4.0) / 4.0 * (y1 - y0)

    d.rounded_rectangle([x0, y0, x1, y1], radius=ancho * ss // 2, fill=(8, 16, 30, 230), outline=(255, 255, 255, 220), width=4 * ss)
    nivel = y_de(limitar(valor, 4.0, 8.0))
    relleno = Image.new("L", (Ws, Hs), 0)
    ImageDraw.Draw(relleno).rounded_rectangle([x0 + 12 * ss, y0 + 12 * ss, x1 - 12 * ss, y1 - 12 * ss], radius=(ancho - 24) * ss // 2, fill=255)
    corte = Image.new("L", (Ws, Hs), 0)
    ImageDraw.Draw(corte).rectangle([0, nivel, Ws, Hs], fill=255)
    capa = Image.new("RGBA", (Ws, Hs), color_ph(valor) + (255,))
    capa.putalpha(ImageChops.multiply(relleno, corte))
    img.alpha_composite(capa)
    for ph in range(4, 9):
        y = y_de(ph)
        d.line([(x0 - 26 * ss, y), (x0 - 6 * ss, y)], fill=(255, 255, 255, 200), width=3 * ss)
        txt = render_lineas([str(ph)], "dato_negrita", 30 * ss, color=(220, 228, 240))
        img.alpha_composite(txt, (max(0, int(x0 - 34 * ss - txt.width)), int(y - txt.height / 2)))
    yu = y_de(5.5)
    for xs in range(int(x0 - 20 * ss), int(x1 + (40 if etiqueta_umbral else 20) * ss), 22 * ss):
        d.line([(xs, yu), (xs + 13 * ss, yu)], fill=(255, 80, 80, 255), width=5 * ss)
    if etiqueta_umbral:
        txt = render_lineas(etiqueta_umbral.split(" · ", 1), "dato_negrita", 30 * ss, color=(255, 130, 120), alineacion="izq")
        img.alpha_composite(txt, (int(x1 + 46 * ss), int(yu - txt.height / 2)))
    return _reducir(img, (Ws // ss, Hs // ss))


# ---------------------------------------------------------------------------
# Elementos de la fórmula
# ---------------------------------------------------------------------------
_IG_GRADIENTE = [(254, 218, 117), (250, 126, 30), (214, 41, 118), (150, 47, 191), (79, 91, 213)]


@functools.lru_cache(maxsize=16)
def icono_instagram(tam, a_color=False):
    """Glifo de cámara de Instagram (trazo blanco, o con el degradado de la marca)."""
    ss = 4
    s = tam * ss
    m = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(m)
    g = int(s * 0.085)
    d.rounded_rectangle([g / 2, g / 2, s - g / 2, s - g / 2], radius=int(s * 0.28), outline=255, width=g)
    r = s * 0.22
    d.ellipse([s / 2 - r, s / 2 - r, s / 2 + r, s / 2 + r], outline=255, width=g)
    p = s * 0.055
    d.ellipse([s * 0.75 - p, s * 0.25 - p, s * 0.75 + p, s * 0.25 + p], fill=255)
    if a_color:
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        k = np.clip(((xx / s) + (1 - yy / s)) / 2, 0, 1) * (len(_IG_GRADIENTE) - 1)
        i0 = np.floor(k).astype(int)
        i1 = np.minimum(i0 + 1, len(_IG_GRADIENTE) - 1)
        f = (k - i0)[..., None]
        paleta = np.array(_IG_GRADIENTE, np.float32)
        rgb = paleta[i0] * (1 - f) + paleta[i1] * f
        img = Image.fromarray(np.dstack([rgb, np.asarray(m, np.float32)]).astype(np.uint8), "RGBA")
    else:
        img = Image.new("RGBA", (s, s), (255, 255, 255, 0))
        img.putalpha(m)
    return _reducir(img, (tam, tam))


ESTILOS_PILDORA = {
    "blanca": {"fondo": (255, 255, 255, 242), "texto": P["tinta"], "borde": None},
    "menta": {"fondo": P["menta"] + (242,), "texto": P["tinta"], "borde": None},
    "coral": {"fondo": P["coral"] + (242,), "texto": (255, 255, 255), "borde": None},
    "ambar": {"fondo": (38, 24, 4, 205), "texto": P["ambar"], "borde": P["ambar"] + (255,)},
    "morada": {"fondo": P["morado"] + (240,), "texto": P["tinta"], "borde": None},
    "oscura": {"fondo": (8, 16, 30, 215), "texto": (255, 255, 255), "borde": (255, 255, 255, 70)},
}


@functools.lru_cache(maxsize=128)
def pildora(texto, estilo="blanca", tam=32):
    """Etiqueta tipo píldora (rectángulo redondeado) como las de los explicadores."""
    e = ESTILOS_PILDORA[estilo]
    t = render_lineas([texto], "dato_negrita", tam, color=e["texto"])
    pad_x, pad_y = int(tam * 0.5), int(tam * 0.18)
    img = Image.new("RGBA", (t.width + pad_x * 2, t.height + pad_y * 2), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle(
        [1, 1, img.width - 2, img.height - 2], radius=int(img.height * 0.28), fill=e["fondo"], outline=e["borde"], width=3 if e["borde"] else 0
    )
    img.alpha_composite(t, (pad_x, pad_y))
    return img


def linea_guia(lienzo, origen, destino, alfa=1.0, color=(255, 255, 255)):
    """Línea fina que une una etiqueta con el objeto señalado (con punto en el extremo)."""
    if alfa <= 0.01:
        return
    x0, y0 = origen
    x1, y1 = destino
    caja = (int(min(x0, x1) - 10), int(min(y0, y1) - 10), int(max(x0, x1) + 10), int(max(y0, y1) + 10))
    capa = Image.new("RGBA", (caja[2] - caja[0], caja[3] - caja[1]), (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    a = (x0 - caja[0], y0 - caja[1])
    b = (x1 - caja[0], y1 - caja[1])
    d.line([a, b], fill=color + (int(235 * alfa),), width=3)
    d.ellipse([b[0] - 6, b[1] - 6, b[0] + 6, b[1] + 6], fill=color + (int(255 * alfa),))
    pegar(lienzo, capa, caja[0], caja[1], ancla="esquina")


def etiqueta_con_guia(lienzo, texto, estilo, pos, destino, alfa=1.0, tam=32):
    """Píldora en `pos` con línea guía hasta `destino` (aparece con un pequeño rebote)."""
    if alfa <= 0.01:
        return
    spr = pildora(texto, estilo, tam)
    linea_guia(lienzo, (pos[0], pos[1] + (spr.height / 2 if destino[1] > pos[1] else -spr.height / 2)), destino, alfa)
    pegar(lienzo, spr, pos[0], pos[1], escala=0.85 + 0.15 * alfa, alfa=alfa)


def texto_espaciado(texto, rol, tam, color, espaciado=0.18):
    """Texto en versalitas con separación entre letras (rótulos de tarjetas)."""
    f = fuente(rol, tam)
    ancho = sum(f.getlength(c) for c in texto) + espaciado * tam * (len(texto) - 1)
    img = Image.new("RGBA", (int(ancho) + 8, int(tam * 1.4)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = 4
    for c in texto:
        d.text((x, tam * 1.05), c, font=f, fill=tuple(color) + (255,), anchor="ls")
        x += f.getlength(c) + espaciado * tam
    return img


@functools.lru_cache(maxsize=16)
def tarjeta_dato(numero, unidad, sub, color_numero=(255, 255, 255)):
    """Tarjeta oscura con un número grande (+ unidad) y un rótulo en versalitas debajo."""
    n = render_lineas([[(numero, color_numero)]], "titulo", 128)
    u = render_lineas([unidad], "titulo", 54) if unidad else None
    s = texto_espaciado(sub, "dato_negrita", 30, (215, 222, 235)) if sub else None
    ancho_linea = n.width + (u.width + 6 if u else 0)
    w = max(ancho_linea, s.width if s else 0) + 90
    h = n.height + (s.height + 10 if s else 0) + 40
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([1, 1, w - 2, h - 2], radius=30, fill=(8, 14, 28, 205), outline=(255, 255, 255, 55), width=2)
    x = (w - ancho_linea) // 2
    img.alpha_composite(n, (x, 14))
    if u:
        img.alpha_composite(u, (x + n.width + 6, 14 + n.height - u.height - 18))
    if s:
        img.alpha_composite(s, ((w - s.width) // 2, 14 + n.height - 8))
    return img


def tarjeta_barras(titulo, filas, ancho=880):
    """
    Tarjeta comparativa: título en versalitas y filas con etiqueta, valor y barra.
    filas = [(etiqueta, valor_texto, fraccion 0–1, color, alfa 0–1)]
    """
    tit = texto_espaciado(titulo, "dato_negrita", 32, P["oro"], 0.2)
    alto = 70 + tit.height + len(filas) * 120
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([1, 1, ancho - 2, alto - 2], radius=30, fill=(8, 14, 28, 200), outline=(255, 255, 255, 50), width=2)
    img.alpha_composite(tit, ((ancho - tit.width) // 2, 28))
    y = 50 + tit.height
    for etiqueta, valor, frac, color, alfa in filas:
        if alfa <= 0.01:
            y += 120
            continue
        capa = Image.new("RGBA", (ancho, 120), (0, 0, 0, 0))
        dc = ImageDraw.Draw(capa)
        e = texto_espaciado(etiqueta, "dato_negrita", 30, (235, 240, 248))
        v = render_lineas([valor], "titulo", 42, color=color)
        capa.alpha_composite(e, (60, 10))
        capa.alpha_composite(v, (ancho - 60 - v.width, 0))
        dc.rounded_rectangle([60, 70, ancho - 60, 92], radius=11, fill=(255, 255, 255, 34))
        x1 = 60 + (ancho - 120) * limitar(frac)
        if x1 > 72:
            barra = Image.new("RGBA", (ancho, 120), (0, 0, 0, 0))
            ImageDraw.Draw(barra).rounded_rectangle([60, 70, x1, 92], radius=11, fill=color + (255,))
            halo = barra.filter(ImageFilter.GaussianBlur(9))
            capa.alpha_composite(halo)
            capa.alpha_composite(barra)
        img.alpha_composite(con_alfa(capa, alfa), (0, y))
        y += 120
    return img


@functools.lru_cache(maxsize=4)
def marco_tarjeta(ancho, alto):
    """Tarjeta oscura con marco dorado fino (para dibujos de línea)."""
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, ancho - 1, alto - 1], fill=(10, 16, 32, 225))
    d.rectangle([10, 10, ancho - 11, alto - 11], outline=P["oro"] + (255,), width=4)
    for cx, cy in ((10, 10), (ancho - 11, 10), (10, alto - 11), (ancho - 11, alto - 11)):
        d.rectangle([cx - 9, cy - 9, cx + 9, cy + 9], outline=P["oro"] + (255,), width=4)
    return img


def trazos_parciales(trazos, progreso):
    """Recorta una lista de polilíneas a la fracción `progreso` de su longitud total (efecto de dibujo)."""
    largos = [sum(math.dist(a, b) for a, b in zip(t, t[1:])) for t in trazos]
    restante = sum(largos) * limitar(progreso)
    salida = []
    for t, largo in zip(trazos, largos):
        if restante <= 0:
            break
        if restante >= largo:
            salida.append(t)
            restante -= largo
            continue
        parcial = [t[0]]
        for a, b in zip(t, t[1:]):
            seg = math.dist(a, b)
            if restante >= seg:
                parcial.append(b)
                restante -= seg
            else:
                f = restante / seg if seg else 0
                parcial.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
                restante = 0
                break
        salida.append(parcial)
        break
    return salida


def _gancho(x, y, alto, sentido):
    """Polilínea en forma de J (gancho de velcro) apuntando hacia arriba (-1) o abajo (+1)."""
    pts = [(x, y), (x, y + sentido * alto * 0.7)]
    for k in range(0, 181, 20):
        a = math.radians(k)
        pts.append((x + alto * 0.18 - alto * 0.18 * math.cos(a), y + sentido * (alto * 0.7 + alto * 0.18 * math.sin(a))))
    return pts


def _bucle(x, y, alto, sentido):
    return [(x - alto * 0.22 * math.sin(math.radians(k)) * (1 if k < 180 else 1), y + sentido * alto * 0.5 * (1 - math.cos(math.radians(k)))) for k in range(0, 361, 15)]


def velcro_lineal(progreso, union, textos):
    """Dibujo de línea del velcro: ganchos (bacterias) arriba y bucles (proteínas) abajo que se enganchan."""
    ancho, alto = 820, 640
    ss = 2
    img = marco_tarjeta(ancho, alto).resize((ancho * ss, alto * ss), Image.NEAREST)
    d = ImageDraw.Draw(img)
    titulo = texto_espaciado(textos["tarjeta"], "dato_negrita", 34 * ss, P["oro"], 0.3)
    img.alpha_composite(titulo, ((ancho * ss - titulo.width) // 2, 34 * ss))
    y_arriba = (250 + union * 46) * ss
    y_abajo = 470 * ss
    trazos = [[(110 * ss, y_arriba), (710 * ss, y_arriba)], [(110 * ss, y_abajo), (710 * ss, y_abajo)]]
    for i in range(9):
        x = (140 + i * 68) * ss
        trazos.append(_gancho(x, y_arriba, 120 * ss, 1))
    for i in range(10):
        x = (130 + i * 64) * ss
        trazos.append(_bucle(x, y_abajo, 110 * ss, -1))
    for t in trazos_parciales(trazos, progreso):
        if len(t) > 1:
            d.line(t, fill=(240, 244, 250, 255), width=5 * ss, joint="curve")
    if progreso > 0.85:
        a = int(255 * limitar((progreso - 0.85) / 0.15))
        for txt, y in ((textos["gancho"], y_arriba - 70 * ss), (textos["bucle"], y_abajo + 30 * ss)):
            e = render_lineas([txt], "dato_negrita", 32 * ss, color=P["menta"])
            img.alpha_composite(con_alfa(e, a / 255), ((ancho * ss - e.width) // 2, int(y)))
    return _reducir(img, (ancho, alto))


@functools.lru_cache(maxsize=8)
def insignia(tipo, tam=150):
    """Círculo coral con ✗ o menta con ✓."""
    ss = 3
    s = tam * ss
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = P["coral"] if tipo == "x" else P["menta"]
    d.ellipse([0, 0, s - 1, s - 1], fill=color + (255,))
    g = int(s * 0.11)
    if tipo == "x":
        d.line([(s * 0.32, s * 0.32), (s * 0.68, s * 0.68)], fill=(255, 255, 255, 255), width=g)
        d.line([(s * 0.68, s * 0.32), (s * 0.32, s * 0.68)], fill=(255, 255, 255, 255), width=g)
    else:
        d.line([(s * 0.27, s * 0.52), (s * 0.44, s * 0.68), (s * 0.74, s * 0.34)], fill=P["tinta"] + (255,), width=g, joint="curve")
    return _reducir(img, (tam, tam))


def linea_tiempo(k, ancho=900):
    """Escala de días: hoy (24 h) frente a la gingivitis (día 10–21)."""
    img = Image.new("RGBA", (ancho, 190), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([1, 1, ancho - 2, 188], radius=34, fill=(8, 14, 28, int(210 * limitar(k * 3))), outline=(255, 255, 255, 50), width=2)
    y = 95
    x0, x1 = 70, ancho - 70

    def x_dia(dd):
        return x0 + (x1 - x0) * (dd - 1) / 20

    largo = x0 + (x1 - x0) * limitar(k) ** 0.7
    d.line([(x0, y), (largo, y)], fill=(255, 255, 255, 255), width=8)
    if largo > x_dia(10):
        d.line([(x_dia(10), y), (min(largo, x_dia(21)), y)], fill=P["coral"] + (255,), width=16)
    d.ellipse([x_dia(1) - 14, y - 14, x_dia(1) + 14, y + 14], fill=P["menta"] + (255,))
    for dd, txt, col in ((1, "hoy · 24 h", P["menta"]), (10, "día 10", P["coral"]), (21, "día 21", P["coral"])):
        if largo >= x_dia(dd) - 1:
            s = render_lineas([txt], "dato_negrita", 30, color=col)
            img.alpha_composite(s, (int(limitar(x_dia(dd) - s.width / 2, 10, ancho - s.width - 10)), y + 20))
            d.line([(x_dia(dd), y - 18), (x_dia(dd), y + 18)], fill=col + (255,), width=5)
    if k > 0.75:
        s = render_lineas(["gingivitis"], "dato_negrita", 32, color=P["coral"])
        img.alpha_composite(con_alfa(s, limitar((k - 0.75) / 0.25)), (int((x_dia(10) + x_dia(21)) / 2 - s.width / 2), y - 70))
    return img


@functools.lru_cache(maxsize=256)
def pildora_reloj(minutos, noche=False, brillo_q=0):
    """Reloj compacto: anillo de progreso de 24 h + HH:MM, en una píldora oscura."""
    ss = 2
    w, h = 250, 74
    img = Image.new("RGBA", ((w + 40) * ss, (h + 40) * ss), (0, 0, 0, 0))
    if brillo_q:
        halo = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(halo).rounded_rectangle([20 * ss, 20 * ss, (w + 20) * ss, (h + 20) * ss], radius=h * ss // 2, fill=P["menta"] + (int(60 * brillo_q),))
        img.alpha_composite(halo.filter(ImageFilter.GaussianBlur(14 * ss)))
    d = ImageDraw.Draw(img)
    acento = P["morado"] if noche else P["menta"]
    d.rounded_rectangle([20 * ss, 20 * ss, (w + 20) * ss, (h + 20) * ss], radius=h * ss // 2, fill=(8, 14, 28, 220), outline=acento + (255,), width=3 * ss)
    cx, cy, r = (20 + 40) * ss, (20 + h / 2) * ss, 22 * ss
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(70, 90, 120, 255), width=6 * ss)
    horas = minutos / 60
    if horas > 0.001:
        d.arc([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * min(horas, 24) / 24, fill=acento + (255,), width=6 * ss)
    px, py = punto_en_circulo(cx, cy, r * 0.62, horas / 24 * 360)
    d.line([(cx, cy), (px, py)], fill=(255, 255, 255, 255), width=3 * ss)
    if noche:
        d.ellipse([cx - 8 * ss, cy - 8 * ss, cx + 8 * ss, cy + 8 * ss], fill=(255, 236, 170, 255))
        d.ellipse([cx - 3 * ss, cy - 12 * ss, cx + 12 * ss, cy + 4 * ss], fill=(8, 14, 28, 255))
    hh, mm = divmod(int(round(minutos)), 60)
    t = render_lineas([f"{hh:02d}:{mm:02d}"], "reloj", 40 * ss, color=(255, 255, 255))
    img.alpha_composite(t, (int(cx + r + 18 * ss), int(cy - t.height / 2)))
    return _reducir(img, (img.width // ss, img.height // ss))


# ---------------------------------------------------------------------------
# Personajes y objetos
# ---------------------------------------------------------------------------
PIEL = (232, 178, 146)
PIEL_SOMBRA = (205, 148, 118)
PELO = (44, 30, 24)


@functools.lru_cache(maxsize=1)
def dormitorio():
    """Persona que despierta en la cama y se tapa la boca (estilo plano, de noche/amanecer)."""
    ss = 2
    img = gradiente_vertical(W * ss, H * ss, (10, 16, 40), (54, 44, 70))
    d = ImageDraw.Draw(img)

    def S(*v):
        return [x * ss for x in v]

    d.rectangle(S(90, 640, 380, 900), fill=(36, 54, 96, 255), outline=(70, 86, 130, 255), width=14 * ss)  # ventana
    d.ellipse(S(250, 680, 330, 760), fill=(250, 236, 190, 255))
    d.ellipse(S(232, 664, 306, 738), fill=(36, 54, 96, 255))
    d.line(S(235, 640, 235, 900), fill=(70, 86, 130, 255), width=8 * ss)
    d.rounded_rectangle(S(90, 930, 990, 1220), radius=40 * ss, fill=(96, 62, 54, 255))
    d.ellipse(S(210, 1090, 870, 1330), fill=(226, 230, 244, 255))
    d.ellipse(S(372, 820, 708, 1160), fill=PELO + (255,))
    for k in range(7):
        d.ellipse(S(360 + k * 50, 790 + (k % 2) * 30, 430 + k * 50, 870 + (k % 2) * 30), fill=PELO + (255,))
    d.rounded_rectangle(S(170, 1310, 910, 2000), radius=200 * ss, fill=(84, 104, 190, 255))
    d.ellipse(S(400, 880, 680, 1190), fill=PIEL + (255,))
    for ex in (480, 600):
        d.ellipse(S(ex - 32, 975, ex + 32, 1040), fill=(255, 255, 255, 255))
        d.ellipse(S(ex - 12, 995, ex + 12, 1020), fill=(30, 20, 15, 255))
    d.rounded_rectangle(S(430, 1055, 650, 1155), radius=46 * ss, fill=PIEL + (255,))
    for k in range(3):
        d.line(S(470 + k * 50, 1060, 470 + k * 50, 1115), fill=PIEL_SOMBRA + (255,), width=5 * ss)
    d.rounded_rectangle(S(470, 1135, 620, 1410), radius=40 * ss, fill=(84, 104, 190, 255))
    d.rectangle(S(0, 1500, W, H), fill=(196, 202, 224, 255))
    d.polygon(S(0, 1500, W, 1450, W, H, 0, H), fill=(70, 96, 170, 255))
    for a in (-60, -30, 30, 60):
        d.line([punto_en_circulo(540 * ss, 1010 * ss, 240 * ss, a), punto_en_circulo(540 * ss, 1010 * ss, 310 * ss, a)], fill=(255, 255, 255, 255), width=10 * ss)
    return _reducir(img, (W, H))


@functools.lru_cache(maxsize=1)
def nube_aliento():
    """Nube verde de dibujo animado (aliento matutino), con contorno y líneas de olor."""
    ss = 2
    w, h = 420, 340
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bolas = [(150, 210, 80), (240, 190, 95), (320, 230, 70), (200, 260, 70), (110, 255, 55), (280, 270, 60)]
    for x, y, r in bolas:  # contorno
        d.ellipse([(x - r - 6) * ss, (y - r - 6) * ss, (x + r + 6) * ss, (y + r + 6) * ss], fill=(70, 120, 60, 255))
    for x, y, r in bolas:
        d.ellipse([(x - r) * ss, (y - r) * ss, (x + r) * ss, (y + r) * ss], fill=(176, 226, 120, 255))
    for x, y, r in bolas[:3]:
        d.ellipse([(x - r * 0.5) * ss, (y - r * 0.7) * ss, (x - r * 0.1) * ss, (y - r * 0.35) * ss], fill=(220, 248, 190, 255))
    for k in range(3):  # líneas de olor onduladas
        x0 = 150 + k * 70
        pts = [((x0 + 14 * math.sin(i * 0.9)) * ss, (110 - i * 10) * ss) for i in range(10)]
        d.line(pts, fill=(176, 226, 120, 255), width=8 * ss, joint="curve")
    return _reducir(img, (w, h))


@functools.lru_cache(maxsize=24)
def persona_busto(variante, encia_q):
    """Busto plano sonriente; `encia_q` 0–10 mezcla el color de encía de sano (menta) a inflamado (coral)."""
    ss = 2
    w, h = 220, 300
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def S(*v):
        return [x * ss for x in v]

    ropa = [(84, 104, 190), (200, 120, 90), (90, 150, 120), (150, 100, 170), (190, 160, 80)][variante % 5]
    pelo = [PELO, (90, 60, 40), (30, 30, 34), (120, 80, 50), (60, 40, 30)][variante % 5]
    piel = [PIEL, (196, 140, 106), (240, 200, 170), (170, 120, 90), (226, 170, 136)][variante % 5]
    d.rounded_rectangle(S(20, 200, 200, 330), radius=70 * ss, fill=ropa + (255,))
    d.ellipse(S(45, 30, 175, 175), fill=pelo + (255,))
    d.ellipse(S(55, 50, 165, 190), fill=piel + (255,))
    if variante % 2:
        d.chord(S(50, 35, 170, 140), 180, 360, fill=pelo + (255,))
    for ex in (88, 132):
        d.ellipse(S(ex - 7, 105, ex + 7, 119), fill=(30, 20, 15, 255))
    encia = mezclar_color(P["encia"], (240, 70, 80), encia_q / 10)
    d.chord(S(78, 128, 142, 178), 0, 180, fill=(110, 30, 44, 255))
    d.rectangle(S(80, 150, 140, 160), fill=encia + (255,))
    d.rectangle(S(84, 156, 136, 166), fill=(250, 250, 248, 255))
    return _reducir(img, (w, h))


def hoja_calendario(dia, ancho=380, alto=420, etiqueta="DÍA"):
    ss = 2
    img = Image.new("RGBA", (ancho * ss, alto * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, ancho * ss, alto * ss], radius=34 * ss, fill=(255, 255, 255, 255))
    d.rounded_rectangle([0, 0, ancho * ss, 110 * ss], radius=34 * ss, fill=P["coral"] + (255,))
    d.rectangle([0, 70 * ss, ancho * ss, 110 * ss], fill=P["coral"] + (255,))
    t = render_lineas([etiqueta], "titulo", 56 * ss, color=(255, 255, 255))
    img.alpha_composite(t, (int(ancho * ss / 2 - t.width / 2), int(55 * ss - t.height / 2)))
    n = render_lineas([str(dia)], "titulo", 190 * ss, color=P["azul"])
    img.alpha_composite(n, (int(ancho * ss / 2 - n.width / 2), int(265 * ss - n.height / 2)))
    for x in (0.3, 0.7):
        d.ellipse([ancho * ss * x - 14 * ss, -14 * ss, ancho * ss * x + 14 * ss, 14 * ss], fill=(60, 70, 80, 255))
    return _reducir(img, (ancho, alto))


@functools.lru_cache(maxsize=2)
def cepillo(largo=1000):
    """Cepillo de lado con las cerdas hacia abajo y una franja de pasta con flúor."""
    ss = 2
    alto = 300
    img = Image.new("RGBA", (largo * ss, alto * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def S(*v):
        return [x * ss for x in v]

    d.rounded_rectangle(S(0, 40, largo - 230, 100), radius=30 * ss, fill=P["menta"] + (255,))
    d.rounded_rectangle(S(60, 52, largo - 400, 70), radius=9 * ss, fill=(255, 255, 255, 140))
    d.rounded_rectangle(S(largo - 280, 50, largo, 100), radius=24 * ss, fill=(36, 170, 150, 255))
    for k in range(17):
        x = largo - 262 + k * 15
        d.rounded_rectangle(S(x, 98, x + 9, 190), radius=3 * ss, fill=(232, 242, 252, 255))
    d.rounded_rectangle(S(largo - 270, 180, largo - 6, 236), radius=26 * ss, fill=(255, 255, 255, 255))
    for k in range(5):
        x = largo - 250 + k * 52
        d.rounded_rectangle(S(x, 192, x + 26, 224), radius=10 * ss, fill=P["menta"] + (255,))
    return _reducir(img, (largo, alto))


@functools.lru_cache(maxsize=2)
def mano():
    """Mano plana con el índice extendido (para el gesto de compartir)."""
    ss = 3
    w, h = 260, 420
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def S(*v):
        return [x * ss for x in v]

    d.rounded_rectangle(S(92, 10, 146, 230), radius=27 * ss, fill=PIEL + (255,))  # índice
    d.ellipse(S(98, 14, 140, 52), fill=(240, 200, 180, 255))  # uña
    d.rounded_rectangle(S(60, 170, 230, 400), radius=70 * ss, fill=PIEL + (255,))  # palma
    for x0 in (146, 180):
        d.rounded_rectangle(S(x0, 160, x0 + 46, 270), radius=23 * ss, fill=PIEL_SOMBRA + (255,))
    d.rounded_rectangle(S(30, 220, 90, 330), radius=30 * ss, fill=PIEL_SOMBRA + (255,))  # pulgar
    return _reducir(img, (w, h))


def telefono(miniatura, hoja=0.0, seleccion=0.0, enviado=0.0, textos=None):
    """
    Teléfono con la imagen del video y el menú «Enviar a» que sube desde abajo.
    hoja: 0–1 (subida del menú) · seleccion: 0–1 (check en el primer contacto) · enviado: 0–1
    """
    textos = textos or {}
    w, h = 560, 1080
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=70, fill=(16, 18, 24, 255), outline=(70, 76, 90, 255), width=4)
    pantalla = Image.new("RGBA", (w - 36, h - 36), (6, 10, 22, 255))
    pegar(pantalla, miniatura, pantalla.width / 2, pantalla.height * 0.38)
    if hoja > 0:
        alto_hoja = 430
        y = int(pantalla.height - alto_hoja * limitar(hoja))
        capa = Image.new("RGBA", (pantalla.width, alto_hoja), (0, 0, 0, 0))
        dc = ImageDraw.Draw(capa)
        dc.rounded_rectangle([0, 0, capa.width - 1, alto_hoja + 40], radius=40, fill=(30, 33, 42, 250))
        dc.rounded_rectangle([capa.width / 2 - 40, 14, capa.width / 2 + 40, 22], radius=4, fill=(110, 116, 130, 255))
        t = render_lineas([textos.get("enviar_a", "Enviar a")], "dato_negrita", 34, color=(255, 255, 255))
        capa.alpha_composite(t, ((capa.width - t.width) // 2, 40))
        colores = [(255, 140, 150), (110, 170, 255), (110, 210, 150), (250, 196, 80)]
        for i, (nombre, col) in enumerate(zip(textos.get("contactos", []), colores)):
            cx = 70 + i * ((capa.width - 140) / 3)
            dc.ellipse([cx - 46, 120, cx + 46, 212], fill=col + (255,))
            inicial = render_lineas([nombre[:1]], "titulo", 40, color=(255, 255, 255))
            capa.alpha_composite(inicial, (int(cx - inicial.width / 2), int(166 - inicial.height / 2)))
            if i == 0 and seleccion > 0:
                pegar(capa, insignia("v", 40), cx + 34, 200, escala=limitar(seleccion) * 1.0)
            n = render_lineas([nombre], "dato_negrita", 24, color=(220, 225, 235))
            capa.alpha_composite(n, (int(cx - n.width / 2), 224))
        verde = mezclar_color((150, 90, 240), (40, 190, 110), limitar(enviado))
        dc.rounded_rectangle([40, 300, capa.width - 40, 370], radius=20, fill=verde + (255,))
        etiqueta = textos.get("enviado", "Enviado ✓") if enviado >= 0.5 else textos.get("boton", "Enviar")
        b = render_lineas([etiqueta], "dato_negrita", 32, color=(255, 255, 255))
        capa.alpha_composite(b, ((capa.width - b.width) // 2, int(335 - b.height / 2)))
        pantalla.alpha_composite(capa, (0, y))
    mascara = Image.new("L", pantalla.size, 0)
    ImageDraw.Draw(mascara).rounded_rectangle([0, 0, pantalla.width - 1, pantalla.height - 1], radius=54, fill=255)
    pantalla.putalpha(ImageChops.multiply(pantalla.getchannel("A"), mascara))
    img.alpha_composite(pantalla, (18, 18))
    d.rounded_rectangle([w / 2 - 70, 34, w / 2 + 70, 62], radius=14, fill=(16, 18, 24, 255))  # isla de la cámara
    return img
