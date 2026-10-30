"""
Biblioteca de gráficos procedurales (sin imágenes externas): dientes, placa,
película salival, bacterias, cristales de azúcar, medidor de pH, reloj de 24 h,
dentista ilustrada, lengua, calendario, cepillo, etc.

Cada sprite se dibuja a 2–3× su tamaño y se reduce con LANCZOS para obtener
bordes suaves (antialiasing). Los sprites estáticos se guardan en caché.
"""

import functools
import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

import config
from utilidades import (
    aleatorio,
    disco_suave,
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
    return img.resize(tam, Image.LANCZOS)


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
        estrechamiento = 0.80 + 0.20 * v
        pts.append((x0 + (0.5 + x * 0.40 * estrechamiento) * w, y0 + (0.56 + y * 0.44) * h))
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
    lateral = 1 - 0.30 * np.clip(np.abs(u - 0.5) / 0.42, 0, 1) ** 2.2
    col = esmalte[None, None, :] * lateral[..., None]
    calido = np.array([238, 224, 196], np.float32)
    k_c = (np.clip((0.48 - v) / 0.25, 0, 1) * 0.6)[..., None]
    col = col * (1 - k_c) + calido * lateral[..., None] * k_c
    incisal = np.array([190, 210, 228], np.float32)
    k_i = (np.clip((v - 0.80) / 0.18, 0, 1) ** 1.5 * 0.7)[..., None]
    col = col * (1 - k_i) + incisal * k_i
    col *= (1 - 0.02 * (np.sin(v * Hs / (5.0 * ss)) > 0.7))[..., None]  # perikymata sutiles
    brillo = Image.new("L", (Ws, Hs), 0)
    d = ImageDraw.Draw(brillo)
    d.ellipse([Ws * 0.26, Hs * 0.30, Ws * 0.39, Hs * 0.80], fill=210)
    d.ellipse([Ws * 0.58, Hs * 0.36, Ws * 0.66, Hs * 0.62], fill=120)
    d.ellipse([Ws * 0.30, Hs * 0.86, Ws * 0.62, Hs * 0.92], fill=90)
    b = np.asarray(brillo.filter(ImageFilter.GaussianBlur(Ws * 0.03)), np.float32)[..., None] / 255
    col = col + (255 - col) * b * 0.85
    alfa_e = np.asarray(m, np.float32) / 255

    limite_encia = 0.21 + 0.16 * ((u - 0.5) / 0.48) ** 2
    # Encía con contorno redondeado (no rectangular) que se funde por arriba.
    dentro = (((u - 0.5) / 0.47) ** 2 + ((v - 0.14) / 0.17) ** 2) < 1
    mg = Image.fromarray(((v < limite_encia) * dentro * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4 * ss))
    mg = np.asarray(mg, np.float32) / 255
    encia = np.array(P["encia"], np.float32)
    k = np.clip(v / 0.35, 0, 1)[..., None]
    col_encia = encia * (0.78 + 0.22 * k)
    banda = np.exp(-(((v - limite_encia + 0.035) / 0.02) ** 2))[..., None]
    col_encia = col_encia + (255 - col_encia) * banda * 0.35  # brillo húmedo del margen

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
    """Capa de placa (biopelícula): más gruesa junto a la encía y en los bordes, opaca y brillante."""
    _, mascara = diente(ancho, alto)
    rng = aleatorio(7)
    ruido = Image.fromarray((rng.random((alto // 14 + 2, ancho // 14 + 2)) * 255).astype(np.uint8))
    ruido = np.asarray(ruido.resize((ancho, alto), Image.BICUBIC), np.float32) / 255
    yy, xx = np.mgrid[0:alto, 0:ancho].astype(np.float32)
    u, v = xx / ancho, yy / alto
    grosor = 0.45 + 0.5 * np.clip((0.55 - v) / 0.3, 0, 1) + 0.3 * np.clip((np.abs(u - 0.5) - 0.25) / 0.15, 0, 1)
    grosor = np.clip(grosor + (ruido - 0.5) * 0.5, 0, 1)
    base = np.array([238, 222, 170], np.float32)
    oscuro = np.array([212, 186, 118], np.float32)
    rgb = base * (1 - grosor[..., None] * 0.6) + oscuro * grosor[..., None] * 0.6
    brillo = Image.new("L", (ancho, alto), 0)
    d = ImageDraw.Draw(brillo)
    for _ in range(14):
        x, y = rng.uniform(0.2, 0.8) * ancho, rng.uniform(0.25, 0.85) * alto
        r = rng.uniform(0.015, 0.04) * ancho
        d.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6], fill=int(rng.uniform(120, 230)))
    b = np.asarray(brillo.filter(ImageFilter.GaussianBlur(ancho * 0.012)), np.float32)[..., None] / 255
    rgb = rgb + (255 - rgb) * b * 0.8
    alfa = np.asarray(mascara, np.float32) / 255 * (0.55 + 0.4 * grosor)
    arr = np.dstack([np.clip(rgb, 0, 255), alfa * 255]).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def pelicula(ancho, alto, radio_rel, centro=(0.5, 0.48), alfa=0.42):
    """Película adquirida: capa azul translúcida que se extiende como una gota sobre el esmalte."""
    _, mascara = diente(ancho, alto)
    peq = Image.new("L", (ancho // 4, alto // 4), 0)
    d = ImageDraw.Draw(peq)
    cx, cy = centro[0] * ancho / 4, centro[1] * alto / 4
    r = radio_rel * max(ancho, alto) / 4
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    anillo = Image.new("L", peq.size, 0)
    ImageDraw.Draw(anillo).ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=3)
    peq = peq.resize((ancho, alto), Image.BILINEAR)
    anillo = anillo.resize((ancho, alto), Image.BILINEAR)
    cuerpo = ImageChops.multiply(peq, mascara).point(lambda x: int(x * alfa))
    borde = ImageChops.multiply(anillo, mascara).point(lambda x: int(x * 0.75))
    a = ImageChops.lighter(cuerpo, borde)
    capa = Image.new("RGBA", (ancho, alto), (150, 210, 255, 0))
    capa.putalpha(a)
    return capa


@functools.lru_cache(maxsize=4)
def fondo_macro():
    """Fondo de los planos macro: azul profundo con bokeh y dientes vecinos desenfocados."""
    img = gradiente_radial(W, H, (W / 2, H * 0.45), H * 0.75, (24, 74, 104), P["azul"])
    vecino, _ = diente(560, 790)
    vecino = vecino.filter(ImageFilter.GaussianBlur(14))
    pegar(img, vecino, -150, 960, alfa=0.55)
    pegar(img, vecino, W + 150, 960, alfa=0.55)
    rng = aleatorio(3)
    for _ in range(26):
        r = int(rng.uniform(10, 46))
        pegar(img, disco_suave(r, (180, 235, 230), 0.3, 70), rng.uniform(0, W), rng.uniform(0, H))
    return img


# ---------------------------------------------------------------------------
# Destellos, gotas, esferas
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
        dx, dy = math.cos(a), math.sin(a)
        px, py = -dy, dx
        pts = [
            (c + dx * largo * s, c + dy * largo * s),
            (c + px * grosor * s, c + py * grosor * s),
            (c - dx * largo * s, c - dy * largo * s),
            (c - px * grosor * s, c - py * grosor * s),
        ]
        d.polygon(pts, fill=color + (255,))
    img = _reducir(img, (tam, tam))
    glow = disco_suave(tam // 3, color, 0.1, 160)
    pegar(img, glow, tam / 2, tam / 2)
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
    lambert = np.clip(nx * luz[0] + ny * luz[1] + nz * luz[2], 0, 1)
    sombreado = 0.42 + 0.58 * lambert
    especular = np.exp(-(((nx + 0.38) ** 2 + (ny + 0.42) ** 2) / 0.035))
    c = np.array(color, np.float32)
    rgb = c * sombreado[..., None]
    rgb = rgb + (255 - rgb) * especular[..., None] * 0.85
    borde = np.clip((1 - np.sqrt(d2)) * rs, 0, 1)
    arr = np.dstack([np.clip(rgb, 0, 255), borde * alfa]).astype(np.uint8)
    img = Image.fromarray(arr, "RGBA")
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
def cadena_cocos(n, r, semilla):
    """Cadena de estreptococos (cocos en fila) con una cara neutra, nunca monstruosa."""
    rng = aleatorio(semilla)
    paso = r * 1.72
    ancho = int(paso * (n - 1) + r * 2 + 8)
    alto = int(r * 2 + 8 + r * 1.2)
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    fase = rng.uniform(0, 6)
    cara_idx = n // 2
    tonos = [mezclar_color(P["turquesa"], (90, 220, 205), rng.uniform(0, 0.5)) for _ in range(n)]
    for i in range(n):
        x = r + 4 + i * paso
        y = alto / 2 + math.sin(fase + i * 0.7) * r * 0.45
        pegar(img, esfera(r, tonos[i], 255, cara=(i == cara_idx)), x, y)
    return img


@functools.lru_cache(maxsize=8)
def paracaidas(ancho):
    alto = int(ancho * 0.9)
    ss = 2
    img = Image.new("RGBA", (ancho * ss, alto * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    domo = [0, 0, ancho * ss, alto * ss * 0.7]
    d.pieslice(domo, 180, 360, fill=(255, 255, 255, 170))
    for k in range(1, 4):
        x = ancho * ss * k / 4
        d.line([(x, alto * ss * 0.35), (x, alto * ss * 0.12)], fill=(63, 209, 181, 180), width=3 * ss)
    for x in (0.03, 0.5, 0.97):
        d.line([(ancho * ss * x, alto * ss * 0.35), (ancho * ss * 0.5, alto * ss)], fill=(255, 255, 255, 150), width=2 * ss)
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
    img = img.rotate(angulo_cuantizado, resample=Image.BICUBIC)
    return _reducir(img, (tam, tam))


# ---------------------------------------------------------------------------
# Texturas microscópicas
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=2)
def textura_esmalte_superior(ancho=1300, alto=2100):
    """Superficie del esmalte vista al microscopio: relieve de prismas (patrón de ojo de cerradura)."""
    img = Image.new("RGB", (ancho, alto), (224, 228, 230))
    d = ImageDraw.Draw(img)
    rng = aleatorio(21)
    celda = 64
    for fila in range(-1, alto // int(celda * 0.86) + 2):
        for col in range(-1, ancho // celda + 2):
            x = col * celda + (celda / 2 if fila % 2 else 0) + rng.uniform(-5, 5)
            y = fila * celda * 0.86 + rng.uniform(-5, 5)
            r = celda * 0.47
            tono = int(rng.uniform(232, 246))
            d.ellipse([x - r, y - r * 0.9, x + r, y + r * 0.9], fill=(tono, tono - 2, tono - 8), outline=(176, 184, 192), width=5)
            d.ellipse([x - r * 0.55, y + r * 0.25, x + r * 0.55, y + r * 0.95], fill=(tono - 14, tono - 14, tono - 16))
    img = img.filter(ImageFilter.GaussianBlur(2.2))
    arr = np.asarray(img, np.float32)
    lum = arr.mean(axis=2)
    gy, gx = np.gradient(lum)
    relieve = np.clip(-(gx + gy) * 2.2, -40, 40)
    arr = arr + relieve[..., None]
    tinte = np.array([196, 222, 238], np.float32)
    arr = arr * 0.72 + tinte * 0.28
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


@functools.lru_cache(maxsize=2)
def mundo_lateral(alto_mundo=2700, superficie=2160):
    """Corte lateral: saliva arriba, película y esmalte (prismas verticales) abajo. Base de las escenas 4–5."""
    img = gradiente_vertical(W, alto_mundo, (12, 46, 74), (40, 128, 150), curva=0.9)
    rng = aleatorio(31)
    for _ in range(60):
        r = int(rng.uniform(4, 22))
        pegar(img, disco_suave(r, (170, 230, 235), 0.25, int(rng.uniform(30, 90))), rng.uniform(0, W), rng.uniform(0, superficie))
    esmalte = Image.new("RGBA", (W, alto_mundo - superficie), (236, 232, 222, 255))
    d = ImageDraw.Draw(esmalte)
    x = 0
    while x < W:
        ancho = rng.uniform(22, 30)
        tono = int(rng.uniform(222, 244))
        d.rectangle([x, 0, x + ancho - 3, esmalte.height], fill=(tono, tono - 4, tono - 12, 255))
        x += ancho
    esmalte = esmalte.filter(ImageFilter.GaussianBlur(1.5))
    sombra = gradiente_vertical(W, esmalte.height, (0, 0, 0), (0, 0, 0))
    sombra.putalpha(Image.fromarray((np.linspace(0, 120, esmalte.height)[:, None] * np.ones((1, W))).astype(np.uint8)))
    esmalte.alpha_composite(sombra)
    img.alpha_composite(esmalte, (0, superficie))
    pel = Image.new("RGBA", (W, 26), (180, 160, 235, 150))
    img.alpha_composite(pel, (0, superficie - 22))
    for i in range(70):
        pegar(img, esfera(7, (200, 180, 250)), (i + 0.5) * W / 70 + rng.uniform(-4, 4), superficie - 14 + rng.uniform(-5, 5))
    return img


@functools.lru_cache(maxsize=2)
def textura_velcro():
    """Primer plano de velcro: ganchos sobre tejido (alternativa si no hay foto real)."""
    img = gradiente_vertical(W, H, (20, 40, 66), (10, 24, 40))
    ss = 2
    capa = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    rng = aleatorio(41)
    paso = 46 * ss
    for y in range(0, H * ss + paso, paso):
        for x in range(0, W * ss + paso, paso):
            px, py = x + rng.uniform(-14, 14) * ss, y + rng.uniform(-14, 14) * ss
            r = rng.uniform(13, 19) * ss
            color = mezclar_color((63, 209, 181), (220, 240, 238), rng.uniform(0, 0.6)) + (255,)
            d.line([(px, py + r * 1.6), (px, py)], fill=color, width=4 * ss)
            d.arc([px, py - r, px + 2 * r, py + r], 180, 360 + 40, fill=color, width=4 * ss)
    capa = _reducir(capa, (W, H))
    img.alpha_composite(capa)
    return img


@functools.lru_cache(maxsize=2)
def textura_lengua():
    """Dorso de la lengua con papilas, iluminado por luz de luna."""
    ancho, alto = 1400, 1300
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 120, ancho, alto + 700], fill=(150, 92, 128, 255))
    rng = aleatorio(51)
    papilas = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    dp = ImageDraw.Draw(papilas)
    for _ in range(2600):
        x, y = rng.uniform(0, ancho), rng.uniform(150, alto)
        r = rng.uniform(5, 11)
        tono = rng.uniform(0, 1)
        col = mezclar_color((176, 116, 152), (214, 160, 190), tono) + (255,)
        dp.ellipse([x - r, y - r, x + r, y + r], fill=col)
        dp.ellipse([x - r * 0.4, y - r * 0.6, x + r * 0.2, y - r * 0.1], fill=(235, 200, 225, 200))
    mascara = img.getchannel("A")
    papilas.putalpha(ImageChops.multiply(papilas.getchannel("A"), mascara))
    img.alpha_composite(papilas)
    surco = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    ImageDraw.Draw(surco).line([(ancho / 2, 200), (ancho / 2, alto)], fill=(90, 50, 80, 120), width=26)
    img.alpha_composite(surco.filter(ImageFilter.GaussianBlur(10)))
    luna = gradiente_vertical(ancho, alto, (70, 110, 200), (20, 30, 70))
    luna.putalpha(img.getchannel("A").point(lambda v: int(v * 0.45)))
    img.alpha_composite(luna)
    return img.filter(ImageFilter.GaussianBlur(0.8))


# ---------------------------------------------------------------------------
# Medidor de pH
# ---------------------------------------------------------------------------
def color_ph(valor):
    return mezclar_color(P["ambar"], P["menta"], (valor - 5.5) / 1.5)


def medidor_ph(valor, alto=620, ancho=150, etiqueta_umbral=True):
    """Tubo vertical con escala 4–8; se llena hasta el pH actual; línea roja en 5,5."""
    ss = 2
    margen_der = 420 if etiqueta_umbral else 70
    Ws, Hs = (ancho + margen_der + 70) * ss, (alto + 40) * ss
    img = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 70 * ss, 20 * ss, (70 + ancho) * ss, (20 + alto) * ss

    def y_de(ph):
        return y1 - (ph - 4.0) / 4.0 * (y1 - y0)

    d.rounded_rectangle([x0, y0, x1, y1], radius=ancho * ss // 2, fill=(8, 24, 42, 230), outline=(255, 255, 255, 255), width=5 * ss)
    nivel = y_de(limitar(valor, 4.0, 8.0))
    relleno = Image.new("L", (Ws, Hs), 0)
    ImageDraw.Draw(relleno).rounded_rectangle(
        [x0 + 12 * ss, y0 + 12 * ss, x1 - 12 * ss, y1 - 12 * ss], radius=(ancho - 24) * ss // 2, fill=255
    )
    corte = Image.new("L", (Ws, Hs), 0)
    ImageDraw.Draw(corte).rectangle([0, nivel, Ws, Hs], fill=255)
    capa = Image.new("RGBA", (Ws, Hs), color_ph(valor) + (255,))
    capa.putalpha(ImageChops.multiply(relleno, corte))
    img.alpha_composite(capa)
    f_tam = 30 * ss
    for ph in range(4, 9):
        y = y_de(ph)
        d.line([(x0 - 26 * ss, y), (x0 - 6 * ss, y)], fill=(255, 255, 255, 230), width=3 * ss)
        txt = render_lineas([str(ph)], "dato_negrita", f_tam, color=(255, 255, 255))
        img.alpha_composite(txt, (max(0, int(x0 - 34 * ss - txt.width)), int(y - txt.height / 2)))
    yu = y_de(5.5)
    for xs in range(int(x0 - 20 * ss), int(x1 + (40 if etiqueta_umbral else 20) * ss), 22 * ss):
        d.line([(xs, yu), (xs + 13 * ss, yu)], fill=(255, 70, 70, 255), width=5 * ss)
    if etiqueta_umbral:
        txt = render_lineas(["5,5 · umbral", "del esmalte"], "dato_negrita", 30 * ss, color=(255, 120, 110), alineacion="izq")
        img.alpha_composite(txt, (int(x1 + 46 * ss), int(yu - txt.height / 2)))
    return _reducir(img, (Ws // ss, Hs // ss))


# ---------------------------------------------------------------------------
# Reloj de 24 h (esquina superior derecha)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=128)
def reloj(minutos_totales, noche=False, brillo_q=0):
    """Reloj circular: anillo de progreso menta, manecilla de horas (dial de 24 h) y de minutos."""
    horas = minutos_totales / 60.0
    ss = 2
    D = 230
    tam = (D + 80) * ss
    img = Image.new("RGBA", (tam, tam + 50 * ss), (0, 0, 0, 0))
    c = tam / 2
    r = D / 2 * ss
    if brillo_q:
        glow = disco_suave(int(r * 1.35), P["menta"], 0.05, int(40 * brillo_q))
        pegar(img, glow, c, c)
    d = ImageDraw.Draw(img)
    fondo = (4, 12, 26, 235) if noche else (14, 42, 71, 225)
    d.ellipse([c - r, c - r, c + r, c + r], fill=fondo, outline=(255, 255, 255, 255), width=4 * ss)
    rr = r - 16 * ss
    d.arc([c - rr, c - rr, c + rr, c + rr], 0, 360, fill=(60, 90, 120, 255), width=12 * ss)
    if horas > 0.001:
        d.arc([c - rr, c - rr, c + rr, c + rr], -90, -90 + 360 * min(horas, 24) / 24, fill=P["menta"] + (255,), width=12 * ss)
    for k in range(24):
        a = k * 15
        largo = 14 if k % 6 == 0 else 6
        p0 = punto_en_circulo(c, c, rr - 14 * ss, a)
        p1 = punto_en_circulo(c, c, rr - (14 + largo) * ss, a)
        d.line([p0, p1], fill=(255, 255, 255, 200), width=(3 if k % 6 == 0 else 2) * ss)
    ang_min = (minutos_totales % 60) / 60 * 360
    d.line([(c, c), punto_en_circulo(c, c, rr * 0.62, ang_min)], fill=(160, 190, 210, 255), width=4 * ss)
    ang_h = horas / 24 * 360
    d.line([(c, c), punto_en_circulo(c, c, rr * 0.5, ang_h)], fill=(255, 255, 255, 255), width=7 * ss)
    px, py = punto_en_circulo(c, c, rr * 0.5, ang_h)
    d.ellipse([px - 6 * ss, py - 6 * ss, px + 6 * ss, py + 6 * ss], fill=P["menta"] + (255,))
    d.ellipse([c - 9 * ss, c - 9 * ss, c + 9 * ss, c + 9 * ss], fill=(255, 255, 255, 255))
    h, m = divmod(int(round(minutos_totales)), 60)
    txt = render_lineas([f"{h:02d}:{m:02d}"], "reloj", 40 * ss, color=(255, 255, 255))
    img.alpha_composite(txt, (int(c - txt.width / 2), int(c + r * 0.22)))
    if noche:
        lx, ly = c, c - r * 0.45
        d.ellipse([lx - 18 * ss, ly - 18 * ss, lx + 18 * ss, ly + 18 * ss], fill=(255, 236, 170, 255))
        d.ellipse([lx - 8 * ss, ly - 24 * ss, lx + 26 * ss, ly + 12 * ss], fill=fondo)
    etiqueta = render_lineas(["horas transcurridas"], "dato_negrita", 21 * ss, color=(255, 255, 255), contorno=3 * ss, color_contorno=P["tinta"])
    img.alpha_composite(etiqueta, (int(c - etiqueta.width / 2), int(c + r + 4 * ss)))
    return _reducir(img, (img.width // ss, img.height // ss))


# ---------------------------------------------------------------------------
# Boca estilizada (escena 2)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=2)
def boca(ancho=2400):
    """Boca abierta estilizada con arcada superior e inferior. Se dibuja grande para poder hacer zoom."""
    alto = int(ancho * 0.62)
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, ancho, alto], fill=(200, 86, 104, 255))  # labios
    m = ancho * 0.07
    d.ellipse([m, m * 1.1, ancho - m, alto - m * 1.1], fill=(70, 18, 34, 255))  # interior
    d.ellipse([ancho * 0.18, alto * 0.55, ancho * 0.82, alto * 1.05], fill=(196, 90, 120, 255))  # lengua
    d.rectangle([m, m * 1.1, ancho - m, alto * 0.33], fill=(214, 108, 124, 255))  # encía superior
    centro = ancho / 2
    anchos = [0.115, 0.095, 0.085, 0.075]
    x_der = centro
    posiciones = []
    for i, a in enumerate(anchos):
        w_d = a * ancho
        posiciones.append((x_der, w_d, i))
        x_der += w_d * 0.98
    for x_ini, w_d, i in posiciones:
        for lado in (1, -1):
            x0 = x_ini if lado == 1 else 2 * centro - x_ini - w_d
            h_d = w_d * (1.35 - i * 0.12)
            y0 = alto * 0.21 + i * alto * 0.012
            pts = contorno_diente(w_d * 0.98, h_d, 80, x0, y0)
            d.polygon(pts, fill=P["esmalte"] + (255,))
            d.ellipse([x0 + w_d * 0.25, y0 + h_d * 0.35, x0 + w_d * 0.38, y0 + h_d * 0.8], fill=(255, 255, 255, 255))
    d.rectangle([m, m * 1.1, ancho - m, alto * 0.25], fill=(214, 108, 124, 255))
    for x_ini, w_d, i in posiciones:
        for lado in (1, -1):
            x0 = x_ini if lado == 1 else 2 * centro - x_ini - w_d
            h_d = w_d * 0.8
            y0 = alto * 0.70 - i * alto * 0.01
            d.rounded_rectangle([x0 + w_d * 0.08, y0, x0 + w_d * 0.92, y0 + h_d], radius=w_d * 0.25, fill=(236, 232, 222, 255))
    d.ellipse([0, 0, ancho, alto], outline=(200, 86, 104, 255), width=int(m * 1.15))  # labios por encima
    return img.filter(ImageFilter.GaussianBlur(1.0))


# ---------------------------------------------------------------------------
# Dentista ilustrada (alternativa si no hay video grabado)
# ---------------------------------------------------------------------------
PIEL = (232, 178, 146)
PIEL_SOMBRA = (205, 148, 118)
PELO = (54, 36, 28)


@functools.lru_cache(maxsize=2)
def fondo_consultorio():
    img = gradiente_vertical(W, H, (236, 248, 245), (190, 222, 218))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 300, 420, 900], fill=(255, 255, 255, 255))  # ventana
    d.rectangle([60, 300, 420, 900], outline=(205, 225, 222, 255), width=18)
    d.line([(240, 300), (240, 900)], fill=(205, 225, 222, 255), width=12)
    d.rectangle([700, 420, 960, 600], fill=(250, 250, 248, 255), outline=(170, 150, 120, 255), width=10)  # diploma
    d.ellipse([760, 150, 1010, 300], fill=(250, 250, 250, 255))  # lámpara dental
    d.line([(885, 300), (980, 600)], fill=(170, 190, 190, 255), width=16)
    d.rectangle([0, 1450, W, H], fill=(170, 205, 200, 255))  # mueble
    d.rectangle([0, 1440, W, 1470], fill=(150, 185, 180, 255))
    return img.filter(ImageFilter.GaussianBlur(9))


@functools.lru_cache(maxsize=2)
def cuerpo_dentista():
    """Cuerpo y cabeza sin rasgos móviles (ojos y boca se dibujan en cada fotograma)."""
    ss = 2
    img = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def S(*v):
        return [x * ss for x in v]

    d.ellipse(S(325, 520, 755, 1110), fill=PELO + (255,))  # pelo trasero
    d.rectangle(S(488, 900, 592, 1080), fill=PIEL_SOMBRA + (255,))  # cuello
    d.rounded_rectangle(S(140, 1050, 940, 2200), radius=230 * ss, fill=(250, 251, 252, 255))  # bata
    d.polygon(S(410, 1050, 670, 1050, 540, 1290), fill=(36, 160, 140, 255))  # pijama menta oscuro
    d.polygon(S(410, 1050, 540, 1290, 470, 1430, 330, 1150), fill=(238, 241, 244, 255), outline=(200, 206, 212, 255))
    d.polygon(S(670, 1050, 540, 1290, 610, 1430, 750, 1150), fill=(238, 241, 244, 255), outline=(200, 206, 212, 255))
    d.rectangle(S(640, 1440, 800, 1520), outline=(200, 206, 212, 255), width=4 * ss)  # bolsillo
    d.rounded_rectangle(S(700, 1400, 716, 1470), radius=6 * ss, fill=P["menta"] + (255,))  # bolígrafo
    d.rounded_rectangle(S(290, 1390, 480, 1450), radius=14 * ss, fill=P["menta"] + (255,))  # placa
    txt = render_lineas([config.MARCA["nombre_dentista"]], "titulo", 34 * ss, color=P["azul"])
    img.alpha_composite(txt, (int(385 * ss - txt.width / 2), int(1420 * ss - txt.height / 2)))
    d.ellipse(S(352, 735, 392, 815), fill=PIEL_SOMBRA + (255,))  # orejas
    d.ellipse(S(688, 735, 728, 815), fill=PIEL_SOMBRA + (255,))
    d.ellipse(S(372, 560, 708, 965), fill=PIEL + (255,))  # cara
    d.chord(S(360, 520, 720, 830), 180, 360, fill=PELO + (255,))  # flequillo
    d.polygon(S(380, 680, 470, 600, 610, 590, 520, 640, 420, 760), fill=PELO + (255,))
    d.ellipse(S(372, 640, 410, 800), fill=PELO + (255,))
    d.ellipse(S(670, 640, 708, 800), fill=PELO + (255,))
    for sx in (455, 625):  # rubor
        pegar(img, disco_suave(34 * ss, (240, 140, 140), 0.1, 90), sx * ss, 850 * ss)
    d.arc(S(522, 770, 558, 830), 20, 160, fill=PIEL_SOMBRA + (255,), width=5 * ss)  # nariz
    for x0 in (440, 572):  # cejas
        d.arc(S(x0, 682, x0 + 76, 722), 200, 340, fill=PELO + (255,), width=8 * ss)
    return _reducir(img, (W, H))


def cara_dentista(apertura=0.0, parpadeo=False, sonrisa=0.5, cepillo=None):
    """
    Rasgos móviles de la dentista sobre una capa transparente de 1080×1920.
    apertura: 0–1 (boca hablando); sonrisa: 0–1; cepillo: desplazamiento horizontal o None.
    """
    ss = 2
    x0, y0, x1, y1 = 380, 680, 700, 1000  # zona de la cara que se redibuja
    img = Image.new("RGBA", ((x1 - x0) * ss, (y1 - y0) * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def S(*v):
        out = []
        for i, x in enumerate(v):
            out.append((x - (x0 if i % 2 == 0 else y0)) * ss)
        return out

    for ex in (478, 602):
        ey = 752
        if parpadeo:
            d.arc(S(ex - 28, ey - 10, ex + 28, ey + 14), 10, 170, fill=PELO + (255,), width=6 * ss)
        else:
            d.ellipse(S(ex - 28, ey - 18, ex + 28, ey + 18), fill=(255, 255, 255, 255))
            d.ellipse(S(ex - 15, ey - 15, ex + 15, ey + 15), fill=(70, 46, 32, 255))
            d.ellipse(S(ex - 7, ey - 7, ex + 7, ey + 7), fill=(15, 10, 8, 255))
            d.ellipse(S(ex - 11, ey - 12, ex - 3, ey - 4), fill=(255, 255, 255, 255))
            d.arc(S(ex - 30, ey - 22, ex + 30, ey + 16), 200, 340, fill=PELO + (255,), width=5 * ss)
    cx, cy = 540, 868
    ancho = 62 + 22 * sonrisa
    alto = 6 + 30 * apertura + 12 * sonrisa
    if alto < 12 and sonrisa < 0.3:
        d.arc(S(cx - ancho / 2, cy - 18, cx + ancho / 2, cy + 10), 20, 160, fill=(160, 60, 70, 255), width=6 * ss)
    else:
        # Media elipse inferior: labio superior recto, sonrisa curva abajo.
        caja = S(cx - ancho / 2, cy - alto, cx + ancho / 2, cy + alto)
        d.chord(caja, 0, 180, fill=(120, 32, 48, 255))
        dientes = S(cx - ancho * 0.40, cy, cx + ancho * 0.40, cy + min(13, alto * 0.38))
        d.rounded_rectangle(dientes, radius=4 * ss, fill=(252, 252, 250, 255))
        if apertura > 0.3:
            d.ellipse(S(cx - ancho * 0.22, cy + alto * 0.55, cx + ancho * 0.22, cy + alto * 0.95), fill=(214, 100, 120, 255))
        d.arc(caja, 0, 180, fill=(170, 70, 80, 255), width=4 * ss)
        d.line(S(cx - ancho / 2, cy, cx + ancho / 2, cy), fill=(170, 70, 80, 255), width=4 * ss)
    if cepillo is not None:
        bx = cx + cepillo
        d.rounded_rectangle(S(bx - 10, cy - 8, bx + 260, cy + 12), radius=8 * ss, fill=P["menta"] + (255,))
        d.rounded_rectangle(S(bx - 60, cy - 14, bx + 10, cy + 2), radius=5 * ss, fill=(245, 250, 255, 255))
        for k in range(6):
            d.line(S(bx - 56 + k * 11, cy + 2, bx - 56 + k * 11, cy + 16), fill=(230, 240, 250, 255), width=4 * ss)
        d.ellipse(S(bx + 120, cy - 30, bx + 230, cy + 60), fill=PIEL + (255,))  # mano
        for k in range(3):
            d.arc(S(bx + 120, cy - 22 + k * 22, bx + 170, cy + 6 + k * 22), 90, 270, fill=PIEL_SOMBRA + (255,), width=4 * ss)
    capa = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    capa.alpha_composite(_reducir(img, (x1 - x0, y1 - y0)), (x0, y0))
    return capa


# ---------------------------------------------------------------------------
# Persona despertando (escena 7, alternativa al video real)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=2)
def dormitorio():
    ss = 2
    img = gradiente_vertical(W * ss, H * ss, (38, 58, 110), (230, 170, 140))
    d = ImageDraw.Draw(img)

    def S(*v):
        return [x * ss for x in v]

    d.rectangle(S(640, 260, 1000, 720), fill=(255, 196, 120, 255), outline=(240, 230, 220, 255), width=16 * ss)
    d.ellipse(S(760, 560, 900, 700), fill=(255, 240, 180, 255))  # sol saliendo
    d.line(S(820, 260, 820, 720), fill=(240, 230, 220, 255), width=10 * ss)
    d.rounded_rectangle(S(80, 900, 1000, 1200), radius=40 * ss, fill=(120, 80, 70, 255))  # cabecero
    d.ellipse(S(200, 1080, 880, 1330), fill=(250, 250, 252, 255))  # almohada
    d.ellipse(S(372, 800, 708, 1150), fill=PELO + (255,))  # pelo despeinado
    for k in range(7):
        d.ellipse(S(360 + k * 50, 760 + (k % 2) * 30, 430 + k * 50, 840 + (k % 2) * 30), fill=PELO + (255,))
    d.rounded_rectangle(S(160, 1300, 920, 2000), radius=200 * ss, fill=(90, 120, 200, 255))  # pijama
    d.ellipse(S(400, 860, 680, 1170), fill=PIEL + (255,))  # cara
    for ex in (480, 600):  # ojos muy abiertos
        d.ellipse(S(ex - 30, 960, ex + 30, 1020), fill=(255, 255, 255, 255))
        d.ellipse(S(ex - 12, 978, ex + 12, 1002), fill=(30, 20, 15, 255))
    d.rounded_rectangle(S(430, 1040, 650, 1140), radius=46 * ss, fill=PIEL + (255,))  # mano tapando la boca
    for k in range(3):
        d.line(S(470 + k * 50, 1045, 470 + k * 50, 1100), fill=PIEL_SOMBRA + (255,), width=5 * ss)
    d.rounded_rectangle(S(470, 1120, 620, 1400), radius=40 * ss, fill=(90, 120, 200, 255))  # brazo
    d.rectangle(S(0, 1500, W, H), fill=(240, 240, 248, 255))  # sábana
    d.polygon(S(0, 1500, W, 1450, W, H, 0, H), fill=(110, 150, 220, 255))  # manta
    for a in (-60, -30, 30, 60):  # líneas cómicas de sorpresa
        p0 = punto_en_circulo(540 * ss, 1000 * ss, 230 * ss, a)
        p1 = punto_en_circulo(540 * ss, 1000 * ss, 300 * ss, a)
        d.line([p0, p1], fill=(255, 255, 255, 255), width=10 * ss)
    return _reducir(img, (W, H))


# ---------------------------------------------------------------------------
# Calendario, cepillo, burbujas de comentario, logotipo
# ---------------------------------------------------------------------------
def hoja_calendario(dia, ancho=380, alto=420):
    ss = 2
    img = Image.new("RGBA", (ancho * ss, alto * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, ancho * ss, alto * ss], radius=34 * ss, fill=(255, 255, 255, 255))
    d.rounded_rectangle([0, 0, ancho * ss, 110 * ss], radius=34 * ss, fill=P["coral"] + (255,))
    d.rectangle([0, 70 * ss, ancho * ss, 110 * ss], fill=P["coral"] + (255,))
    t = render_lineas(["DÍA"], "titulo", 56 * ss, color=(255, 255, 255))
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


def burbuja_comentario(texto, color_avatar):
    ss = 2
    t = render_lineas([texto], "dato_negrita", 46 * ss, color=P["azul"])
    u = render_lineas(["@usuario"], "dato", 28 * ss, color=(120, 130, 140))
    ancho = max(t.width, u.width) + 150 * ss
    alto = t.height + u.height + 30 * ss
    img = Image.new("RGBA", (ancho, alto + 30 * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, ancho, alto], radius=40 * ss, fill=(255, 255, 255, 245))
    d.polygon([(60 * ss, alto - 2), (110 * ss, alto - 2), (50 * ss, alto + 28 * ss)], fill=(255, 255, 255, 245))
    d.ellipse([22 * ss, alto / 2 - 38 * ss, 98 * ss, alto / 2 + 38 * ss], fill=color_avatar + (255,))
    img.alpha_composite(u, (120 * ss, 12 * ss))
    img.alpha_composite(t, (120 * ss, 8 * ss + u.height))
    return _reducir(img, (img.width // ss, img.height // ss))


@functools.lru_cache(maxsize=2)
def logo_alternativo():
    """Logotipo provisional (círculo menta con diente) si no existe assets/imagenes/logo.png."""
    ss = 2
    tam = 150
    img = Image.new("RGBA", (tam * ss, tam * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, tam * ss, tam * ss], fill=P["menta"] + (255,))
    d.polygon(contorno_diente(tam * ss * 0.5, tam * ss * 0.62, 90, tam * ss * 0.25, tam * ss * 0.17), fill=(255, 255, 255, 255))
    return _reducir(img, (tam, tam))
