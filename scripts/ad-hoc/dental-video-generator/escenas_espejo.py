"""
Escenas de «La última vez que tu mamá te cepilló los dientes», contado por el
espejo del baño: una cámara en perspectiva detrás del espejo (vemos la
encimera con el lavabo, a quien se cepilla y la puerta del fondo), personajes
3D tipo juguete, contador a la izquierda, subtítulos en itálica minúscula,
salto de 25 años y cierre con dato.

Todo lo que se toca se toca de verdad: las manos llegan a la boca, al vaso,
al grifo o al marco con brazos de largo real; los objetos se colocan por su
punto natural (cabeza del cepillo, punta del lápiz, centro de la tarjeta) y
la mano se calcula a partir de ellos, así que van siempre juntos. Las
trayectorias son curvas suaves (sin paradas bruscas) y los pies se apoyan en
el suelo al caminar.

Cada fotograma es una función pura del tiempo `t`: el video se renderiza en
paralelo por tramos sin que cambie el resultado.
"""

import functools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import config
import escenario as S
import graficos as G
import guion_espejo as GE
import juguete3d as J
import personajes as PJ
import utileria as U
from utilidades import fuente, lerp, limitar, pegar, progreso, render_lineas, salida_cubica, suave

W, H, FPS = config.ANCHO, config.ALTO, config.FPS
Y_SUBTITULO = 1262  # justo encima del borde de la encimera: no tapa caras
CREMA = (255, 250, 243)
DORADO = (232, 190, 98)  # #DDB664 del guion, un punto más vivo para leerse sobre fondos claros
N = J._norm


# ---------------------------------------------------------------------------
# Utilidades de animación
# ---------------------------------------------------------------------------
def kf(t, claves):
    """Interpola fotogramas clave [(t, valor)] con suavizado (para y desde reposo); valores número o tupla."""
    if t <= claves[0][0]:
        return claves[0][1]
    for (t0, v0), (t1, v1) in zip(claves, claves[1:]):
        if t <= t1:
            k = suave(progreso(t, t0, t1))
            if v0 is None or v1 is None:
                return v1 if k > 0.5 else v0
            if isinstance(v0, (tuple, list, np.ndarray)):
                return tuple(float(a) + (float(b) - float(a)) * k for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * k
    return claves[-1][1]


def curva(t, claves):
    """
    Trayectoria suave (Catmull-Rom en el tiempo) por los puntos clave [(t, punto)]:
    velocidad continua, sin pararse en cada clave; arranca y frena en la primera y la última.
    """
    ts = [c[0] for c in claves]
    ps = [np.asarray(c[1], np.float64) for c in claves]
    if t <= ts[0]:
        return tuple(ps[0])
    if t >= ts[-1]:
        return tuple(ps[-1])
    i = max(0, min(len(ts) - 2, int(np.searchsorted(ts, t, side="right")) - 1))
    t0, t1 = ts[i], ts[i + 1]
    p0, p1 = ps[i], ps[i + 1]
    dt = t1 - t0

    def tangente(j):
        if j == 0 or j == len(ts) - 1:
            return np.zeros_like(ps[j])
        return (ps[j + 1] - ps[j - 1]) / (ts[j + 1] - ts[j - 1])

    m0, m1 = tangente(i) * dt, tangente(i + 1) * dt
    u = (t - t0) / dt
    h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
    h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
    return tuple(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)


def parpadeo(t, base, desfase=0.0):
    """Sustituye la expresión por un parpadeo breve cada ~3,4 s."""
    return "parpadeo" if ((t + desfase) % 3.4) < 0.11 and base in ("neutral", "sonrisa", "abierta", "concentrada") else base


def respiracion(t, desfase=0.0):
    return math.sin((t + desfase) * 2 * math.pi * 0.22) * 2.2


def pulso(t, centros, ancho=0.08):
    """Suma de pulsos gaussianos (gestos breves) en los instantes `centros`."""
    return sum(math.exp(-(((t - c) / ancho) ** 2)) for c in centros)


def escalon(t, centros, dur=0.16):
    """Cuenta cuántos instantes de `centros` han pasado, con transición suave de `dur` s."""
    return sum(suave(progreso(t, c, c + dur)) for c in centros)


def pen_encimera(pts, radios):
    """Cuánto se mete cada esfera (punto, radio) en el mueble de la encimera (fuera del hueco del lavabo)."""
    pts = np.asarray(pts, np.float32)
    dt = S.ENCIMERA_Y - pts[:, 1]  # >0: por encima de la superficie
    df = S.ENCIMERA_Z0 - pts[:, 2]  # >0: por delante del canto
    fuera = (dt > 0) & (df > 0)
    d = np.where(fuera, np.sqrt(np.maximum(dt, 0) ** 2 + np.maximum(df, 0) ** 2), np.maximum(dt, df))
    L = S.LAVABO
    en_lavabo = (((pts[:, 0] - L["x"]) / L["rx"]) ** 2 + ((pts[:, 2] - L["z"]) / L["rz"]) ** 2) < 0.9
    return np.where(en_lavabo & (dt < 0), 0.0, np.maximum(np.asarray(radios, np.float32) - d, 0.0)).astype(np.float32)


@functools.lru_cache(maxsize=1)
def _utileria_obstaculos():
    return tuple(lista_utileria(("amarillo", "azul"), True, True))


def utileria_obstaculos():
    return list(_utileria_obstaculos())


def punto_mano(spec, clave, tipo, lado, escala=1.0):
    """Dónde queda el punto `clave` (yemas, palma, pinza…) de una mano pedida con otra clave (para enlazar trayectorias sin saltos)."""
    sx = -1 if lado == "izq" else 1
    L = PJ.TIPOS[tipo]["mano"] * PJ.TIPOS[tipo]["R"] * escala
    vieja = next(k for k in ("palma", "yemas", "pinza", "indice", "centro") if k in spec)
    pr = PJ.mezclar_formas(*spec["mezcla"]) if "mezcla" in spec else PJ.params_forma(spec.get("forma", "abierta"))
    d, nrm, lat = PJ.marco_mano(spec["dir"], spec["nrm"], sx)
    a, b, c = PJ._desfase_punto(PJ._redondear(pr), sx, vieja)
    W = np.asarray(spec[vieja], np.float32) - (d * a + lat * b + nrm * c) * L
    a, b, c = PJ._desfase_punto(PJ._redondear(pr), sx, clave)
    return W + (d * a + lat * b + nrm * c) * L


def y_cabeza(tipo, banco=0.0, escala=1.0):
    """Altura del centro de la cabeza de alguien de pie en el suelo (o en un banquito de `banco` px)."""
    return S.SUELO - banco - PJ.altura_pies(tipo, escala)


# ---------------------------------------------------------------------------
# Reparto y posiciones
# ---------------------------------------------------------------------------
SUETER = (214, 160, 52)  # suéter mostaza de mamá (no cambia en 30 años)


@functools.lru_cache(maxsize=None)
def mama(edad=26):
    pelo = (206, 202, 198) if edad >= 55 else (84, 56, 42) if edad >= 40 else (92, 58, 40)
    return PJ.Personaje("adulta", SUETER, (92, 74, 64), pelo, "mono", arrugas=edad >= 55, nombre="mama")


@functools.lru_cache(maxsize=None)
def hijo(edad, pegatina=False):
    if edad < 1:
        return PJ.Personaje("bebe", (176, 208, 238), (176, 208, 238), (70, 48, 34), "bebe", nombre="hijo")
    if edad < 3:
        return PJ.Personaje("pequeno", (128, 174, 230), (128, 174, 230), (64, 42, 30), "corto", nombre="hijo")
    if edad < 5:
        return PJ.Personaje("nino4", (86, 128, 212), (86, 128, 212), (60, 40, 28), "corto", nombre="hijo")
    if edad < 6:  # a los 5, otra camiseta (otro día): la pegatina del dentista va en el dorso de la mano
        return PJ.Personaje("nino4", (204, 86, 76), (64, 84, 132), (60, 40, 28), "corto", nombre="hijo")
    if edad < 6.5:
        return PJ.Personaje("nino4", (84, 158, 118), (64, 84, 132), (60, 40, 28), "corto", nombre="hijo")
    if edad < 10:
        return PJ.Personaje("nino7", (86, 128, 212), (86, 128, 212), (60, 40, 28), "corto", nombre="hijo")
    if edad < 25:
        return PJ.Personaje("adolescente", (150, 153, 160), (64, 70, 92), (54, 38, 30), "capucha", nombre="hijo")
    return PJ.Personaje("adulto", (40, 58, 98), (62, 62, 70), (54, 38, 30), "corto", nombre="hijo")


@functools.lru_cache(maxsize=None)
def nieta():
    return PJ.Personaje("pequeno", (240, 142, 152), (240, 142, 152), (72, 46, 34), "coletas", nombre="nieta")


YM = y_cabeza("adulta")
# Quien se cepilla está al centro, frente al lavabo. Para cepillar a un niño, el adulto se pone detrás
# de él (como se hace de verdad: una mano con el cepillo y la otra sujetando la cabeza); cuando solo
# lo acompaña, está a su derecha (del lado de la puerta, el vaso y el marco).
MAMA = (825.0, YM, -5.0)
NINO_AL_LADO = 505.0  # x del niño cuando mamá está a su lado (no a su espalda)
MAMA_DETRAS = (562.0, YM, -108.0)
EJE_SOLO = (-0.88, 0.28, 0.38)  # quien se cepilla solo: el mango sale hacia su mano derecha, la mano lejos de la mejilla
MAMA_SOLA = (545.0, YM, -5.0)
MAMA_BEBE = (562.0, YM, -32.0)
NINO4 = (548.0, y_cabeza("nino4", 215), 22.0)
PEQ = (548.0, y_cabeza("pequeno", 330), 22.0)
NINO7 = (548.0, y_cabeza("nino7", 150), 22.0)
TEEN = (548.0, y_cabeza("adolescente"), -5.0)
HIJO_DETRAS = (562.0, y_cabeza("adulto"), -112.0)
HIJO_SOLO = (505.0, y_cabeza("adulto"), -30.0)  # sin meter la cadera en el canto de la encimera
NIETA = (548.0, y_cabeza("pequeno", 340), 22.0)
BEBE = (548.0, S.ENCIMERA_Y - (PJ.TIPOS["bebe"]["cadera_y"] + 1.15) * PJ.TIPOS["bebe"]["R"], 150.0)  # sentado en el borde del lavabo
PUERTA_XZ = (1190.0, -780.0)  # quien mira desde la puerta (pies en el suelo del fondo)


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
    for p in partes:
        img.alpha_composite(p, (0, y))
        y += 30
    img.alpha_composite(num, (-2, y + 12))
    y += num.height + 16
    for e in extras:
        img.alpha_composite(e, (0, y))
        y += 30
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
# Papeles con texto (texturas de tarjetas y de la nota)
# ---------------------------------------------------------------------------
TINTA = (40, 50, 80, 255)
TINTA_ROJA = (200, 60, 60, 255)


@functools.lru_cache(maxsize=96)
def sprite_tarjeta(titulo, lineas, tachadas=(), nueva=None, ancho=300, color=(252, 248, 236), k_tachar=1.0, k_nueva=1.0, ss=2):
    """
    Tarjeta de cita escrita a mano. `k_tachar` (0–1) dibuja la raya de la última fecha tachada
    y `k_nueva` (0–1) escribe la fecha nueva de izquierda a derecha (para animar la escritura).
    """
    filas = 1 + len(lineas) + len(tachadas) + (1 if nueva else 0)
    alto = 70 + filas * 46
    f_med = fuente("manuscrita", 40)
    ancho = max(ancho, int(math.ceil(max(f_med.getlength(x) for x in (titulo,) + tuple(lineas) + tuple(tachadas) + ((nueva,) if nueva else ())) + 48)))  # el texto nunca se corta
    img = Image.new("RGBA", (ancho * ss, alto * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, ancho * ss - 1, alto * ss - 1], radius=10 * ss, fill=color + (255,))
    d.rectangle([0, 0, ancho * ss, 16 * ss], fill=(150, 210, 196, 255))
    f_t = fuente("manuscrita", 40 * ss)
    y = 26 * ss
    d.text((22 * ss, y), titulo, font=f_t, fill=TINTA)
    y += 50 * ss
    for l in lineas:
        d.text((22 * ss, y), l, font=f_t, fill=TINTA)
        y += 46 * ss
    for i, l in enumerate(tachadas):
        d.text((22 * ss, y), l, font=f_t, fill=(90, 96, 120, 255))
        w = f_t.getlength(l)
        k = k_tachar if i == len(tachadas) - 1 else 1.0
        if k > 0:
            x0, y0, x1, y1 = 18 * ss, y + 26 * ss, 26 * ss + w, y + 20 * ss
            d.line([(x0, y0), (x0 + (x1 - x0) * k, y0 + (y1 - y0) * k)], fill=TINTA_ROJA, width=4 * ss)
        y += 46 * ss
    if nueva and k_nueva > 0:
        capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(capa).text((22 * ss, y), nueva, font=f_t, fill=TINTA)
        w = f_t.getlength(nueva)
        corte = int(22 * ss + w * k_nueva)
        mascara = Image.new("L", img.size, 0)
        ImageDraw.Draw(mascara).rectangle([0, 0, corte, img.height], fill=255)
        capa.putalpha(Image.fromarray(np.minimum(np.asarray(capa.getchannel("A")), np.asarray(mascara))))
        img.alpha_composite(capa)
    return img


def tex_tarjeta(*args, **kw):
    return _tex_tarjeta(args, tuple(sorted(kw.items())))


@functools.lru_cache(maxsize=96)
def _tex_tarjeta(args, kw):
    return U.textura(sprite_tarjeta(*args, **dict(kw)))


NOTA_LINEAS = (("Del Ratón", 0.09, 0.36), ("de los Dientes", 0.09, 0.74))  # texto, u inicial, v de la línea
NOTA_PX = (400, 250)
NOTA_FUENTE = 62  # cabe entero «de los Dientes» (con su s final)


@functools.lru_cache(maxsize=64)
def _metricas_nota():
    f = fuente("manuscrita", NOTA_FUENTE)
    return [f.getlength(t) / NOTA_PX[0] for t, _, _ in NOTA_LINEAS]


@functools.lru_cache(maxsize=128)
def tex_nota(k):
    """Nota del Ratón de los Dientes con el texto escrito hasta `k` (0–1, por líneas)."""
    w, h = NOTA_PX
    img = Image.new("RGBA", (w, h), (255, 250, 222, 255))
    d = ImageDraw.Draw(img)
    f = fuente("manuscrita", NOTA_FUENTE)
    largos = _metricas_nota()
    total = sum(largos)
    hecho = k * total
    for (texto, u0, v), largo in zip(NOTA_LINEAS, largos):
        if hecho <= 0:
            break
        capa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(capa).text((u0 * w, v * h - 44), texto, font=f, fill=(80, 60, 120, 255))
        corte = int((u0 + min(hecho, largo)) * w)
        mascara = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mascara).rectangle([0, 0, corte, h], fill=255)
        capa.putalpha(Image.fromarray(np.minimum(np.asarray(capa.getchannel("A")), np.asarray(mascara))))
        img.alpha_composite(capa)
        hecho -= largo
    del d
    return U.textura(img)


def punta_nota(k, t):
    """(u, v) de la punta del lápiz cuando el texto va por `k`: avanza por la línea con pequeños bucles de letra."""
    largos = _metricas_nota()
    total = sum(largos)
    hecho = k * total
    for (texto, u0, v), largo in zip(NOTA_LINEAS, largos):
        if hecho <= largo or (texto, u0, v) == NOTA_LINEAS[-1]:
            u = u0 + min(hecho, largo)
            return u, v - 0.06 + 0.07 * math.sin(t * 2 * math.pi * 6.5) + 0.02 * math.sin(t * 2 * math.pi * 11)
        hecho -= largo
    return NOTA_LINEAS[-1][1], NOTA_LINEAS[-1][2]


# ---------------------------------------------------------------------------
# Tarjetas en el marco del espejo (3D, en el plano del espejo)
# ---------------------------------------------------------------------------
# centro en pantalla con el encuadre base y giro (grados, como PIL: + antihorario)
RANURAS = {0: ((760.0, 490.0), -5.0), 1: ((205.0, 520.0), 4.0), 2: ((770.0, 486.0), -3.0)}  # la 0 queda al alcance de mamá sin cruzarle la cara; la 1 no tapa la puerta
Z_MARCO = S.ESPEJO_Z - 5.0


def ranura(indice, ancho_px, alto_px):
    """(centro, ancho, alto, rot) en el mundo para que la tarjeta se vea en la ranura `indice` del marco."""
    (X, Y), ang = RANURAS[indice]
    cam = S.CAM
    Xv, Yv = cam.a_virtual(X, Y)
    s = cam.D / (cam.D - Z_MARCO)
    k = 1.0 / (s * cam.zoom)
    centro = (cam.cx + (Xv - cam.cx) / s, cam.cy + (Yv - cam.cy) / s, Z_MARCO)
    return np.array(centro, np.float32), ancho_px * k, alto_px * k, J.rot_z(-math.radians(ang))


TARJETAS = {
    "hijo": dict(args=(GE.TARJETA_HIJO["titulo"], tuple(GE.TARJETA_HIJO["lineas"])), kw={}),
    "mama_vieja": dict(args=(GE.TARJETA_MAMA["titulo"], (), tuple(GE.TARJETA_MAMA["tachadas"]) + (GE.TARJETA_MAMA["nueva"], "2 de mayo", "8 de nov.")), kw={}),
    "nueva": dict(args=(GE.TARJETA_NUEVA["titulo"], tuple(GE.TARJETA_NUEVA["lineas"])), kw={"color": (255, 246, 228)}),
}


def medidas_tarjeta(nombre):
    spr = sprite_tarjeta(*TARJETAS[nombre]["args"], **TARJETAS[nombre]["kw"])
    return spr.width / 2, spr.height / 2  # tamaño en pantalla (ss=2)


def tarjeta_en_ranura(nombre, indice):
    w, h = medidas_tarjeta(nombre)
    c, aw, ah, rot = ranura(indice, w, h)
    tex = tex_tarjeta(*TARJETAS[nombre]["args"], **TARJETAS[nombre]["kw"])
    return U.tarjeta(c, aw, ah, tex, rot)


# ---------------------------------------------------------------------------
# Encimera con las sombras de lo que está siempre encima (vaso, pasta)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=2)
def _encimera_con_sombras(nombre):
    img = S.capa_encimera(nombre).copy()
    capa = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(capa)
    cam = S.CAM
    vx, _, vz = U.VASO
    tx, tz = 172.0 + 20, 112.0  # tubo de pasta (ver utileria.tubo_pasta)
    for x, z, rx, rz, f in ((vx + 6, vz - 6, 52, 40, 0.75), (tx, tz, 78, 22, 0.55), (S.GRIFO["x"], S.GRIFO["z"], 44, 30, 0.5)):
        X, Y = cam.proyectar(x, S.ENCIMERA_Y, z)
        Xa, _ = cam.proyectar(x + rx, S.ENCIMERA_Y, z)
        _, Yf = cam.proyectar(x, S.ENCIMERA_Y, z - rz)
        _, Yn = cam.proyectar(x, S.ENCIMERA_Y, z + rz)
        ex = abs(float(Xa) - float(X))
        d.ellipse([float(X) - ex, float(Yf), float(X) + ex, float(Yn)], fill=int(255 * f))
    capa = capa.filter(ImageFilter.GaussianBlur(10))
    negro = Image.new("RGBA", (W, H), (30, 18, 10, 0))
    negro.putalpha(Image.fromarray((np.asarray(capa, np.float32) * 0.55 * (np.asarray(img.getchannel("A"), np.float32) / 255)).astype(np.uint8)))
    img.alpha_composite(negro)
    return img


def lista_utileria(vaso_cepillos, pasta, con_grifo=True, permisos=()):
    """Vaso con sus cepillos, tubo de pasta y grifo (lo que está siempre sobre la encimera); `permisos`: quién puede tocar los cepillos del vaso."""
    prims = U.vaso()
    colores = {"amarillo": U.AMARILLO, "azul": U.AZUL, "rosa": U.ROSA}
    for k, c in enumerate(vaso_cepillos):
        if c:
            prims += U.permitir(U.cepillo_en_vaso(U.VASO, k, colores[c]), ("vaso", "objeto"), *permisos)
    if pasta:
        prims += U.tubo_pasta()[0]
    if con_grifo:
        prims += S.grifo(0.0)
    return prims


@functools.lru_cache(maxsize=16)
def capa_utileria(nombre, vaso_cepillos, pasta, con_grifo):
    """La utilería fija renderizada una vez con el encuadre base, con su t por subpíxel (tapa lo que pase por detrás)."""
    luz = J.NOCHE if nombre == "noche" else J.MANANA
    img, x0, y0, tm = J.render(lista_utileria(vaso_cepillos, pasta, con_grifo), luz, S.CAM, oclusor=S.oclusor, devolver_t="completo")
    return img, x0, y0, tm


# ---------------------------------------------------------------------------
# Viñeta
# ---------------------------------------------------------------------------
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
# Caminar con los pies apoyados
# ---------------------------------------------------------------------------
def caminata(t, t0, t1, a, b, tipo, escala=1.0, paso_rel=0.5, cierre=0.3):
    """
    Camina de a=(x, z) a b=(x, z) entre t0 y t1. Devuelve (x, z) del cuerpo, los pies
    {"izq": (x, y, z), "der": …} plantados sin resbalar, el balanceo de los brazos y la
    altura de la cadera necesaria para que las piernas lleguen al suelo (rebote natural).
    """
    P = PJ.TIPOS[tipo]
    R = P["R"] * escala
    ax, az = a
    bx, bz = b
    L = math.hypot(bx - ax, bz - az)
    ux, uz = ((bx - ax) / L, (bz - az) / L) if L > 1e-6 else (0.0, 1.0)
    paso = paso_rel * (P["muslo"] + P["pierna"]) * R
    n = max(1, int(round(L / paso))) if L > 1e-6 else 0
    paso = L / n if n else 1.0
    k = progreso(t, t0, t1)
    s = L * suave(k)
    ancho = P["cadera"] * R * 0.45
    y_tob = S.SUELO - 0.42 * R
    pies, alturas = {}, {}
    for f, (lado, sx) in enumerate((("der", 1), ("izq", -1))):
        pos, lift = 0.0, 0.0
        for j in range(1 + f, n + 1, 2):
            ini, fin = (j - 1) * paso, j * paso
            desde, hasta = max(0, j - 2) * paso, j * paso
            if s >= fin:
                pos = hasta
                continue
            if s > ini:
                w = (s - ini) / paso
                pos = desde + (hasta - desde) * suave(w)
                lift = math.sin(math.pi * w) * 0.3 * R
            break
        if n and k >= 1.0 and pos < L - 1:  # paso final para juntar los pies
            w = progreso(t, t1, t1 + cierre)
            pos, lift = pos + (L - pos) * suave(w), math.sin(math.pi * w) * 0.18 * R
        pies[lado] = (ax + ux * pos + sx * ancho, y_tob - lift, az + uz * pos)
        alturas[lado] = lift
    cx, cz = ax + ux * s, az + uz * s
    largo = 0.985 * (P["muslo"] + P["pierna"]) * R
    y_cadera = -1e9
    for lado, sx in (("der", 1), ("izq", -1)):
        px, py, pz = pies[lado]
        dxz = math.hypot(px - (cx + sx * ancho), pz - (cz - 0.36 * R))
        y_cadera = max(y_cadera, py - math.sqrt(max(largo**2 - dxz**2, 1.0)))
    y_cadera = min(y_cadera, y_tob - largo * 0.999)  # de pie: piernas casi estiradas
    fase = s / paso * math.pi if n else 0.0
    balanceo = 0.32 * math.sin(fase) * min(1.0, k * 6, (1 - k) * 6 + 0.2)
    return {"x": cx, "z": cz, "pies": pies, "balanceo": balanceo, "y_cadera": y_cadera, "R": R}


def cabeza_desde_cadera(tipo, y_cadera, escala=1.0):
    P = PJ.TIPOS[tipo]
    return y_cadera - (P["cadera_y"] + 0.25) * P["R"] * escala


# ---------------------------------------------------------------------------
# Renderizador
# ---------------------------------------------------------------------------
class Renderizador:
    def __init__(self, plan=None):
        self.escenas = {e["id"]: e for e in GE.ESCENAS}
        self.orden = [e["id"] for e in GE.ESCENAS]
        self.avisos = []
        self.cam = S.CAM
        self.modo_revision = False

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

    def camara(self, t, e):
        """Cámara de la escena: encuadre base y, si la escena lo pide, un acercamiento lento."""
        zoom = e.get("zoom")
        if not zoom:
            return S.CAM, 1.0, (W / 2, H / 2)
        z2 = 1.0 + (zoom["hasta"] - 1.0) * suave(progreso(t, e["inicio"], e["fin"]))
        f2 = zoom["foco"]
        zb, fb = S.CAM.zoom, S.CAM.foco
        z = zb * z2
        f = tuple((f2[i] * (1 - z2) + fb[i] * z2 * (1 - zb)) / (1 - z) for i in range(2))
        return J.Camara(S.CAM.D, S.CAM.cx, S.CAM.cy, z, f), z2, f2

    def cuadro(self, t, e):
        self.cam, self.z2, self.f2 = self.camara(t, e)
        self.t_escena = t - e["inicio"]
        self.e = e
        self.palanca = 0.0
        self.grifo_movil = False
        self.permisos_vaso = ()  # la mano que acaba de soltar un cepillo en el vaso todavía lo rodea
        self.permisos_grifo = ()  # los dedos que empujan la palanca
        self.utileria = (("amarillo", "azul"), True)
        return getattr(self, f"escena_{e['id']}")(t, e)

    # ---------------------------------------------------------- composición
    def capas_base(self, luz_nombre, k_manana=None):
        if k_manana is None:
            fondo, enc = S.fondo_bano(luz_nombre), _encimera_con_sombras(luz_nombre)
        else:
            fondo = Image.blend(S.fondo_bano("noche"), S.fondo_bano("manana"), k_manana)
            enc = Image.blend(_encimera_con_sombras("noche"), _encimera_con_sombras("manana"), k_manana)
        if self.z2 > 1.0001:
            return S.acercar(fondo, self.z2, self.f2), S.acercar(enc, self.z2, self.f2)
        return fondo.copy(), enc

    def luz(self, nombre, k_manana=None):
        if k_manana is not None:
            return J.NOCHE.mezclar(J.MANANA, k_manana)
        return J.NOCHE if nombre == "noche" else J.MANANA

    def componer(self, t, luz_nombre, prims, k_manana=None, agua=0.0, sombras=(), marco=(), marco_sombra=1.0):
        """
        Pared del fondo → sombra de los personajes → encimera → sombras de contacto →
        personajes y objetos (z-buffer conjunto, ocultos por la encimera) → agua → tarjetas del marco.
        """
        if self.modo_revision:  # solo las primitivas, para revisar choques
            vaso, pasta = getattr(self, "utileria", (("amarillo", "azul"), True))
            lista = list(prims) + list(marco or []) + lista_utileria(vaso, pasta, not self.grifo_movil, self.permisos_vaso)
            if self.grifo_movil:
                lista += U.permitir(S.grifo(self.palanca), *self.permisos_grifo, solo="palanca")
            return lista
        cam = self.cam
        luz = self.luz(luz_nombre, k_manana)
        fondo, enc = self.capas_base(luz_nombre, k_manana)
        vaso, pasta = getattr(self, "utileria", (("amarillo", "azul"), True))
        nombre = luz_nombre if k_manana is None else ("manana" if k_manana > 0.5 else "noche")
        fija, fx0, fy0, ftm = capa_utileria(nombre, vaso, pasta, not self.grifo_movil)
        zoom2 = (self.z2, self.f2[0], self.f2[1]) if self.z2 > 1.0001 else None
        prims = list(prims)
        if self.grifo_movil:
            prims += S.grifo(self.palanca)
        spr, x0, y0, tmap = J.render(prims, luz, cam, oclusor=S.oclusor, devolver_t=True, mapa_t=(ftm, fx0, fy0, 2, zoom2) if ftm is not None else None)
        if spr is not None:
            sh, pad = J.sombra_de(spr, (26, 34), 36, 0.26)
            fondo.alpha_composite(sh, (x0 - pad, y0 - pad))
        fondo.alpha_composite(enc)
        if sombras:
            self.sombras_contacto(fondo, sombras)
        if fija is not None:
            capa = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            capa.alpha_composite(fija, (fx0, fy0))
            fondo.alpha_composite(S.acercar(capa, self.z2, self.f2) if zoom2 else capa)
        if spr is not None:
            fondo.alpha_composite(spr, (x0, y0))
        if agua > 0.01:
            self.dibujar_agua(fondo, t, agua, luz, spr, x0, y0, tmap)
        if marco:
            spm, mx0, my0 = J.render(list(marco), luz, cam)
            if spm is not None:
                if marco_sombra > 0:
                    sh, pad = J.sombra_de(spm, (8, 12), 12, 0.42 * marco_sombra)
                    fondo.alpha_composite(sh, (mx0 - pad, my0 - pad))
                fondo.alpha_composite(spm, (mx0, my0))
        return fondo

    def dibujar_agua(self, img, t, intensidad, luz, spr, x0, y0, tmap):
        prims = S.chorro(t, intensidad)
        if not prims:
            return
        sa, ax0, ay0, ta = J.render(prims, luz, self.cam, oclusor=S.oclusor, devolver_t=True)
        if sa is None:
            return
        a = np.asarray(sa, np.float32).copy()
        if spr is not None:  # lo que está delante del chorro (la mano, el cepillo) lo tapa
            h, w = ta.shape
            tm = np.full((h, w), np.inf, np.float32)
            ox, oy = x0 - ax0, y0 - ay0
            ys0, xs0 = max(0, oy), max(0, ox)
            ys1, xs1 = min(h, oy + tmap.shape[0]), min(w, ox + tmap.shape[1])
            if ys1 > ys0 and xs1 > xs0:
                tm[ys0:ys1, xs0:xs1] = tmap[ys0 - oy : ys1 - oy, xs0 - ox : xs1 - ox]
            a[..., 3] *= ta < tm
        a[..., 3] *= 0.78 * intensidad
        a[..., :3] = np.clip(a[..., :3] * 1.08 + 12, 0, 255)
        img.alpha_composite(Image.fromarray(a.astype(np.uint8), "RGBA"), (ax0, ay0))

    def sombras_contacto(self, img, sombras):
        """Sombras suaves sobre la encimera bajo lo que se apoya o está muy cerca: (x, z, rx, rz, fuerza)."""
        capa = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(capa)
        for x, z, rx, rz, fuerza in sombras:
            if fuerza <= 0.01:
                continue
            y = S.ENCIMERA_Y
            X, Y = self.cam.proyectar(x, y, z)
            Xa, _ = self.cam.proyectar(x + rx, y, z)
            _, Yf = self.cam.proyectar(x, y, z - rz)
            _, Yn = self.cam.proyectar(x, y, z + rz)
            ex = abs(float(Xa) - float(X))
            d.ellipse([float(X) - ex, float(Yf), float(X) + ex, float(Yn)], fill=int(255 * min(1.0, fuerza)))
        capa = capa.filter(ImageFilter.GaussianBlur(9))
        negro = Image.new("RGBA", (W, H), (30, 18, 10, 0))
        negro.putalpha(capa.point(lambda v: int(v * 0.55)))
        img.alpha_composite(negro)

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
                pegar(img, sprite_subtitulo(e["subtitulo"]), W / 2, Y_SUBTITULO, alfa=a)
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
        previa = self.escenas.get(e["id"] - 1)
        a = 1.0
        if previa is None or previa.get("contador") is None or previa["contador"][0] != rotulo or previa["contador"][3] != lineas:
            a = progreso(t, e["inicio"], e["inicio"] + 0.3)
        pegar(img, spr, 44 - 40, 700 - 40, alfa=0.95 * a, ancla="esquina")

    # ======================================================= construcción
    def persona(self, t, quien, pose, pos, escala=1.0, desfase=0.0, respira=True, vaiven=True, evitar=None):
        """
        Personaje con respiración, parpadeo, leve vaivén de cabeza y del peso; registra si alguna mano no llega.
        `evitar`: primitivas (otra persona, objetos) que sus brazos no deben atravesar; con ellas, la encimera y la
        utilería fija, cada codo busca solo su mejor posición.
        """
        x, y, z = pos
        if evitar is not None:
            pose = dict(pose, evitar=J.obstaculo(list(evitar) + utileria_obstaculos(), extra=pen_encimera))
        if respira:
            y += respiracion(t, desfase)
        if vaiven:  # el peso pasa muy despacio de un pie al otro
            x += 2.5 * math.sin(t * 2 * math.pi * 0.17 + desfase * 1.7)
        pose = dict(pose)
        pose["cara"] = parpadeo(t, pose.get("cara", "neutral"), desfase)
        pose["giro"] = pose.get("giro", 0.0) + 0.035 * math.sin(t * 0.9 + desfase * 2.1)
        prims, info = quien.construir(x, y, pose, escala=escala, z0=z, camara=S.CAM)
        for lado, err in info["error"].items():
            if err > 6 and pose.get(f"mano_{lado}") is not None:
                self.avisos.append((round(t, 2), quien.tipo, lado, round(err, 1)))
        return prims, info

    def info(self, t, quien, pose, pos, escala=1.0, desfase=0.0, respira=True, vaiven=True):
        """Solo la cabeza, la boca y los hombros (para preparar agarres antes de construir); misma postura que `persona`."""
        x, y, z = pos
        if respira:
            y += respiracion(t, desfase)
        if vaiven:
            x += 2.5 * math.sin(t * 2 * math.pi * 0.17 + desfase * 1.7)
        pose = dict(pose, solo_hombros=True)
        pose["giro"] = pose.get("giro", 0.0) + 0.035 * math.sin(t * 0.9 + desfase * 2.1)
        return quien.construir(x, y, pose, escala=escala, z0=z, camara=S.CAM)[1]

    def resolver_mano(self, t, quien, pose, pos, lado, spec, escala=1.0, desfase=0.0):
        """La mano que pide `spec` (agarre, yemas, pinza, palma…) como muñeca + marco + forma, con la postura del cuerpo de `pose`."""
        ib = self.info(t, quien, pose, pos, escala=escala, desfase=desfase)
        P, R = quien.p, ib["R"]
        sx = -1 if lado == "izq" else 1
        H = np.asarray(ib["hombros"][lado], np.float32)
        l1, l2 = P["brazo"][0] * R, P["brazo"][1] * R
        muneca, d, nrm, params, polo = quien._mano_objetivo({f"mano_{lado}": spec}, lado, sx, H, l1, l2, P["mano"] * R, R)
        if params is None:  # colgando: la mano sigue al antebrazo (como en `construir`)
            codo, mu = PJ.ik(H, muneca, l1, l2, polo)
            muneca, d, nrm, params = mu, N(mu - codo), np.array([-sx, 0.0, 0.25], np.float32), PJ.params_forma("relajada")
        return {"muneca": tuple(float(v) for v in muneca), "dir": tuple(float(v) for v in d), "nrm": tuple(float(v) for v in nrm), "forma": params, "polo": tuple(float(v) for v in polo)}

    def transicion(self, t, quien, pose, pos, lado, spec_a, spec_b, k, escala=1.0, desfase=0.0, arco=(0.0, 0.0, 0.0)):
        """Paso continuo de una petición de mano a otra (sin saltos al cambiar de agarre a yemas, etc.); k de 0 a 1; `arco` levanta el camino."""
        if k <= 0.0:
            return spec_a
        if k >= 1.0:
            return spec_b
        a = self.resolver_mano(t, quien, pose, pos, lado, spec_a, escala, desfase)
        b = self.resolver_mano(t, quien, pose, pos, lado, spec_b, escala, desfase)
        mez = lambda u, v: tuple(float(x) + (float(y) - float(x)) * k for x, y in zip(u, v))  # noqa: E731
        muneca = np.asarray(mez(a["muneca"], b["muneca"]), np.float32) + np.asarray(arco, np.float32) * math.sin(math.pi * k)
        return {"muneca": tuple(float(v) for v in muneca), "dir": tuple(N(mez(a["dir"], b["dir"]))), "nrm": tuple(N(mez(a["nrm"], b["nrm"]))),
                "forma": PJ.mezclar_formas(a["forma"], b["forma"], k), "polo": mez(a["polo"], b["polo"])}

    @staticmethod
    def sostenido(prims, dueno, lado):
        """El objeto puede tocar la mano (y la muñeca) que lo sostiene."""
        return U.permitir(prims, (dueno, f"mano_{lado}"), (dueno, f"brazo_{lado}_2"))

    @staticmethod
    def cepillo_en_boca(t, info, eje, color, amp=8.0, f=2.8, fase=0.0, pasta=0.0, hondo=7.0, desplaza=(0.0, 0.0, 0.0), agarre=U.AGARRE_CEPILLO):
        """
        Cepillo con la cabeza dentro de la boca de `info` y el mango hacia `eje`: vaivén corto a lo
        largo de los dientes con un leve giro de muñeca. Devuelve (primitivas, agarre para la mano).
        """
        hx, hy, hz = info["cabeza"]
        m = np.asarray(info["boca"], np.float32)
        e = N(eje)
        horiz = N((e[0], 0.0, e[2] * 0.4))
        v = math.sin(2 * math.pi * f * t + fase)
        off = horiz * amp * v + np.array([0, 2.2 * math.sin(4 * math.pi * f * t + fase), 0], np.float32)
        hacia = N(np.array([hx, hy, hz], np.float32) - m)
        cab = m + hacia * hondo + off + np.asarray(desplaza, np.float32)
        giro_muneca = 0.06 * math.sin(2 * math.pi * f * t + fase + 1.2)
        e = N(e + np.array([0, giro_muneca, 0], np.float32))
        prims, ag = U.cepillo(cab, e, hacia, color, pasta, agarre=agarre)
        U.permitir(prims, (info["dueno"], "cabeza"), solo="cabezal")
        return prims, {"agarre": tuple(ag), "eje": tuple(-e)}

    @staticmethod
    def mano_en_cabeza(info, lado="der"):
        """Palma apoyada con suavidad en la parte de atrás de la cabeza de otra persona (la sostiene mientras la cepilla)."""
        hx, hy, hz = info["cabeza"]
        R = info["R"]
        sx = -1 if lado == "izq" else 1
        fuera = N((sx * 0.42, -0.6, -0.68))
        r = np.array([R, R * 1.02, R * 0.96], np.float32)
        punto = np.array([hx, hy, hz], np.float32) + fuera * r * 1.0 + fuera * 12.0
        arriba = np.array([0.0, -1.0, 0.0], np.float32)
        d = N(arriba - fuera * float(arriba @ fuera))
        # dedos apenas curvados: siguen la curva de la cabeza sin hundirse
        return {"palma": tuple(punto), "dir": tuple(d), "nrm": tuple(-fuera), "mezcla": ("abierta", "relajada", 0.15), "polo": (sx * 1.0, 0.3, 0.2)}

    @staticmethod
    def pegatina_en_mano(info, lado, quien):
        """Pegatina «paciente valiente» pegada en el dorso de la mano (disco con dibujo, sigue a la mano)."""
        pts = info["puntos"].get(lado)
        if not pts:
            return []
        d, nrm, lat = pts["marco"]
        L = quien.p["mano"] * info["R"]
        c = np.asarray(pts["centro"], np.float32) - nrm * (0.17 * L + 1.0) + d * (0.06 * L)
        afuera = -nrm  # el dorso
        eje_x = N(lat - afuera * float(lat @ afuera))
        rot = np.stack([eje_x, -afuera, np.cross(eje_x, -afuera)], 1)
        disco = J.Cilindro(tuple(c), 0.25 * L, 1.0, (250, 250, 248), rot=rot, brillo=0.3, textura_tapa=PJ.textura_pegatina())
        return U.etiqueta([disco], "pegatina", ignorar=((info["dueno"], f"mano_{lado}"), (info["dueno"], f"brazo_{lado}_2")))

    @staticmethod
    def mano_en_coronilla(info, lado="der"):
        """Palma apoyada encima de la cabeza de otra persona (desde atrás, sin taparle la cara), los dedos hacia delante."""
        hx, hy, hz = info["cabeza"]
        R = info["R"]
        sx = -1 if lado == "izq" else 1
        fuera = N((sx * 0.3, -0.93, 0.05))
        r = np.array([R, R * 1.02, R * 0.96], np.float32)
        punto = np.array([hx, hy, hz], np.float32) + fuera * r + fuera * 16.0
        adelante = np.array([-sx * 0.15, 0.25, 1.0], np.float32)
        d = N(adelante - fuera * float(adelante @ fuera))
        return {"palma": tuple(punto), "dir": tuple(d), "nrm": tuple(-fuera), "mezcla": ("abierta", "relajada", 0.25), "polo": (sx * 1.0, -0.35, 0.25)}

    def cepillar_desde_atras(self, t, adulto, pos_a, inf_n, color, amp=7.0, f=2.7, fase=0.0, pasta=0.0, hondo=7.0, desplaza=(0, 0, 0), mano_der="cabeza", extra=None, desfase=0.0, polo=(-1.0, 0.25, -0.35)):
        """El adulto, detrás del niño, le cepilla con la mano dominante (pantalla izquierda) y con la otra le sostiene la cabeza."""
        eje = (-0.80, 0.22, 0.56)
        cep, spec = self.cepillo_en_boca(t, inf_n, eje, color, amp, f, fase, pasta, hondo, desplaza, agarre=108.0)
        self.sostenido(cep, adulto.nombre, "izq")
        spec["polo"] = polo
        pose = {"cara": "sonrisa", "mano_izq": spec, "mirar": tuple(inf_n["boca"]), "cabeceo": 0.0}
        if mano_der == "cabeza":
            pose["mano_der"] = self.mano_en_cabeza(inf_n, "der")
        elif mano_der is not None:
            pose["mano_der"] = mano_der
        pose.update(extra or {})
        pa, ia = self.persona(t, adulto, pose, pos_a, desfase=desfase)
        return pa + cep, ia

    @staticmethod
    def contar(info, lado, n, pulsos=(), alto=1.15):
        """Mano levantada junto a la cara mostrando `n` dedos al espejo (con un pequeño golpe en cada número)."""
        hx, hy, hz = info["cabeza"]
        R = info["R"]
        sx = -1 if lado == "izq" else 1
        golpe = 9.0 * sum(pulsos)
        centro = (hx + sx * 1.02 * R, hy + alto * R + golpe, hz + 1.35 * R)
        return {"centro": centro, "forma": "contar", "dir": (sx * 0.12, -1.0, 0.12), "nrm": (-sx * 0.2, 0.05, 1.0), "dedos": PJ.cuenta_dedos(n), "polo": (sx * 0.5, 1.0, 0.35)}

    def mama_puerta(self, t, edad, pose, desfase=0.4, xz=None):
        """Mamá de pie en la puerta, mirando al lavabo."""
        x, z = xz or PUERTA_XZ
        pose = dict(pose)
        pose.setdefault("piernas", "de_pie")
        return self.persona(t, mama(edad), pose, (x, YM, z), desfase=desfase, vaiven=False)

    def caminando(self, t, quien, tipo, t0, t1, a, b, pose, giro_cuerpo=0.0, cierre=0.3):
        """Personaje caminando de a=(x, z) a b=(x, z) entre t0 y t1 (tiempos de la escena), pies apoyados."""
        w = caminata(self.t_escena, t0, t1, a, b, tipo, cierre=cierre)
        pose = dict(pose)
        pose["piernas"] = {"pies": w["pies"]}
        pose["balanceo"] = w["balanceo"]
        pose.setdefault("giro_cuerpo", giro_cuerpo)
        y = cabeza_desde_cadera(tipo, w["y_cadera"])
        return self.persona(t, quien, pose, (w["x"], y, w["z"]), respira=False, vaiven=False)

    def mama_caminando(self, t, edad, t0, t1, a, b, pose, giro_cuerpo=0.0):
        return self.caminando(t, mama(edad), "adulta", t0, t1, a, b, pose, giro_cuerpo)

    # utilería fija de la encimera (se renderiza una vez y tapa lo que pase por detrás)
    def encimera_objetos(self, t, vaso_cepillos=("amarillo", "azul"), pasta=True):
        self.utileria = (tuple(vaso_cepillos), pasta)
        return []

    # ------------------------------------------------------------ apoyos
    @staticmethod
    def mano_en_hombro(info_otro, lado_otro="izq"):
        """Mano apoyada encima del hombro de otra persona."""
        hx, hy, hz = info_otro["hombros"][lado_otro]
        R = info_otro["R"]
        sx = -1 if lado_otro == "izq" else 1
        return {"palma": (hx + sx * 4, hy - 0.46 * R, hz + 4), "dir": (-sx * 0.55, 0.1, 0.83), "nrm": (0.0, 1.0, 0.1), "forma": "relajada"}

    def mano_en_borde(self, x, lado="der"):
        """Niño pequeño agarrado al borde de la encimera (le llega al pecho): la muñeca queda detrás del borde."""
        sx = -1 if lado == "izq" else 1
        return {"palma": (x, S.ENCIMERA_Y - 11, S.ENCIMERA_Z0 + 14), "dir": (-sx * 0.12, 0.14, 1.0), "nrm": (0, 1, 0), "forma": "abierta", "polo": (sx * 0.55, 1.0, -0.7)}

    def mano_en_encimera(self, x, lado="der", z=64.0):
        """Mano apoyada plana sobre la encimera: el codo va hacia atrás y afuera, nunca dentro del mueble."""
        sx = -1 if lado == "izq" else 1
        return {"palma": (x, S.ENCIMERA_Y - 10, z), "dir": (-sx * 0.12, 0.12, 1.0), "nrm": (0, 1, 0), "forma": "abierta", "polo": (sx * 1.0, 0.1, -0.7)}

    def tarjeta_mano(self, nombre, pinza, rot, dueno, lado, esquina="der", tex=None, polo=(0.0, 0.9, 0.45), forma=None):
        """Tarjeta sujeta en pinza por una esquina de abajo (la mano y la tarjeta van juntas)."""
        w, h = medidas_tarjeta(nombre)
        _, aw, ah, _ = ranura(0, w, h)
        R = np.asarray(rot, np.float32)
        sxe = 1 if esquina == "der" else -1
        local = np.array([sxe * (aw / 2 - 13), ah / 2 - 15, -7.5], np.float32)
        centro = np.asarray(pinza, np.float32) - R @ local
        tex = tex if tex is not None else tex_tarjeta(*TARJETAS[nombre]["args"], **TARJETAS[nombre]["kw"])
        # la mano va por detrás de la tarjeta (del lado de quien la sostiene): la palma hacia la tarjeta, el pulgar asoma por la esquina
        mano = {"pinza": tuple(pinza), "dir": tuple(R @ np.array([-sxe * 0.45, -0.88, 0.0], np.float32)), "nrm": tuple(R @ np.array([0.0, 0.0, 1.0], np.float32)), "polo": polo}
        if forma is not None:
            mano["mezcla"] = forma
        prims = U.tarjeta(centro, aw, ah, tex, R)
        self.sostenido(prims, dueno, lado)
        return prims, mano

    def meter_en_marco(self, lt, nombre, indice, t_mira, t_sube, t_llega, t_suelta, pinza_ini, dueno, lado, esquina="der", inclinacion=0.3, polo=(0.0, 0.9, 0.45), reposo=None):
        """
        Lleva una tarjeta de la mano a su ranura del marco con un movimiento natural: primero mira el
        sitio, sube la mano en arco por delante del hombro (sin cruzar caras), se inclina un poco hacia
        el espejo, la mete con un empujoncito, la suelta abriendo los dedos y la mano baja por el mismo
        camino. Devuelve (prims de la tarjeta en la mano o None, spec de la mano o None, inclinación,
        mirar, tarjeta del marco o None, fuerza de su sombra).
        """
        w, h = medidas_tarjeta(nombre)
        c_r, aw, ah, R_r = ranura(indice, w, h)
        sxe = 1 if esquina == "der" else -1
        local = np.array([sxe * (aw / 2 - 13), ah / 2 - 15, -7.5], np.float32)
        pinza_r = c_r + R_r @ local
        ang_r = math.atan2(float(R_r[1, 0]), float(R_r[0, 0]))
        p_ini = np.asarray(pinza_ini, np.float32)
        p_rep = np.asarray(reposo if reposo is not None else p_ini + np.array([0, 90, -30], np.float32), np.float32)
        # punto de paso: delante del hombro, a media altura (el codo va abajo, la tarjeta no tapa la cara)
        p_via = p_ini * 0.35 + pinza_r * 0.65 + np.array([0, 55, -95], np.float32)
        incl = inclinacion * suave(progreso(lt, t_sube + 0.1, t_llega)) * (1 - suave(progreso(lt, t_suelta + 0.15, t_suelta + 0.85)))
        mirar = tuple(c_r) if t_mira <= lt <= t_suelta + 0.3 else None
        if lt < t_suelta:
            p = np.array(curva(lt, [(t_sube, tuple(p_ini)), (t_sube + (t_llega - t_sube) * 0.5, tuple(p_via)), (t_llega - 0.08, tuple(pinza_r + np.array([0, 14, -6], np.float32))), (t_llega + 0.1, tuple(pinza_r + np.array([0, -4, 0], np.float32))), (t_suelta, tuple(pinza_r))]), np.float32)
            k = suave(progreso(lt, t_sube, t_llega))
            R = J.rot_z(0.06 * (1 - k) + ang_r * k) @ J.rot_x(-0.32 * (1 - k))
            prims, mano = self.tarjeta_mano(nombre, p, R, dueno, lado, esquina, polo=polo)
            return prims, mano, incl, mirar, None, 0.0
        marco = tarjeta_en_ranura(nombre, indice)
        t_fin = t_suelta + 0.85
        mano = None
        if lt < t_fin:
            abre = suave(progreso(lt, t_suelta, t_suelta + 0.14))
            p = np.array(curva(lt, [(t_suelta, tuple(pinza_r)), (t_suelta + 0.18, tuple(pinza_r + np.array([0, 30, -60], np.float32))), (t_suelta + 0.5, tuple(p_via)), (t_fin, tuple(p_rep))]), np.float32)
            d = R_r @ np.array([-sxe * 0.45, -0.88, 0.0], np.float32)
            mano = {"pinza": tuple(p), "dir": tuple(d), "nrm": tuple(R_r @ np.array([0.0, 0, 1.0], np.float32)), "mezcla": ("pinza", "relajada", abre), "polo": polo}
            if lt < t_suelta + 0.3:  # los dedos todavía rozan la tarjeta al soltarla
                U.permitir(marco, (dueno, f"mano_{lado}"))
        return None, mano, incl, mirar, marco, suave(progreso(lt, t_suelta + 0.15, t_suelta + 0.45))

    # ================================================================ ESCENAS
    def bebe_y_mama(self, t, lt, escala=1.0, fase=0.0, trae=None):
        """Bebé sentado en la encimera; mamá, detrás, lo sostiene por la espalda y le cepilla su primer diente."""
        ib = self.info(t, hijo(0.7), {"cara": "abierta"}, BEBE, escala=escala, desfase=1.1, vaiven=False)
        mb = np.asarray(ib["boca"], np.float32)
        eje = N((-0.80, 0.26, 0.54))
        if trae is not None and lt < trae:  # trae el cepillo con su grano de pasta en arco, desde delante de ella
            cab_final = mb + N(np.asarray(ib["cabeza"]) - mb) * 7
            cab = np.array(curva(lt, [(0.0, (395, 1080, 190)), (trae * 0.55, (425, 1040, 245)), (trae, tuple(cab_final))]), np.float32)
            cep, ag = U.cepillo(cab, eje, (0.3, 0, -1), U.AMARILLO, pasta=0.35, agarre=108.0)
            U.permitir(cep, ("hijo", "cabeza"), solo="cabezal")
            mano_izq = {"agarre": tuple(ag), "eje": tuple(-eje)}
        else:
            cep, mano_izq = self.cepillo_en_boca(t, ib, eje, U.AMARILLO, amp=5.0, f=2.4, fase=fase, pasta=0.35, agarre=108.0)
        self.sostenido(cep, "mama", "izq")
        mano_izq["polo"] = (-1.0, 0.3, -0.3)
        xb, yb, zb = BEBE
        R = ib["R"]
        abre = trae is None or lt > trae - 0.05
        mira_b = None
        if trae is not None and lt < trae + 0.3:  # sigue el cepillo con la mirada, sin agachar del todo la cabeza
            cc = np.asarray(cep[0].c, np.float32)
            mira_b = (float(cc[0]), float(min(cc[1], yb + 60)), float(cc[2]) + 300)
        pb, _ = self.persona(t, hijo(0.7), {"cara": "abierta" if abre else "sonrisa", "piernas": "sentado",
                                            "asiento_y": S.ENCIMERA_Y - 8,
                                            "mano_izq": {"palma": (xb - 36, S.ENCIMERA_Y - 100, zb + 64), "dir": (0.1, 0.2, 1.0), "nrm": (0.1, 1, 0), "forma": "relajada", "polo": (-1.0, 0.2, 0.1)},  # las manitas sobre los muslos
                                            "mano_der": {"palma": (xb + 36, S.ENCIMERA_Y - 100, zb + 64), "dir": (-0.1, 0.2, 1.0), "nrm": (-0.1, 1, 0), "forma": "relajada", "polo": (1.0, 0.2, 0.1)},
                                            "mirar": mira_b}, BEBE, escala=escala, desfase=1.1, vaiven=False, evitar=cep)
        espalda = zb - 0.36 * R - PJ.TIPOS["bebe"]["prof"] * PJ.TIPOS["bebe"]["pecho"] * R
        sosten = {"palma": (xb + 0.22 * R, yb + 1.9 * R, espalda - 3), "dir": (-0.1, -1.0, 0.1), "nrm": (0.0, 0.0, 1.0), "forma": "abierta", "polo": (0.9, 0.6, -0.4)}
        pm, _ = self.persona(t, mama(26), {"cara": "sonrisa", "inclinacion": 0.08, "mano_izq": mano_izq, "mano_der": sosten,
                                           "mirar": tuple(ib["boca"]) if lt < 2.2 else None}, MAMA_BEBE, evitar=pb)
        sombras = [(xb, zb + 30, 70, 45, 0.5)]
        return pm + pb + cep, sombras

    # 1 · Portada: mamá sostiene al bebé sentado en la encimera y le cepilla su primer diente
    def escena_1(self, t, e):
        lt = t - e["inicio"]
        prims, sombras = self.bebe_y_mama(t, lt, trae=1.3)
        return self.componer(t, "noche", prims + self.encimera_objetos(t), sombras=sombras)

    # 2 · Mañana y noche: el bebé crece hasta los 2 años en el mismo encuadre
    def escena_2(self, t, e):
        lt = t - e["inicio"]
        paso = 0.66
        i = int(lt / paso)
        k_trans = progreso(lt - i * paso, paso - 0.2, paso)
        img = self._fase_2(t, lt, i)
        if self.modo_revision:
            return img
        if k_trans > 0 and lt < e["fin"] - e["inicio"] - paso * 0.5:
            img = Image.blend(img, self._fase_2(t, lt, i + 1), k_trans)
        return img

    def _fase_2(self, t, lt, i):
        luz = "manana" if i % 2 == 0 else "noche"
        edad = [0.7, 1.0, 1.4, 1.7, 2.0, 2.0, 2.0][min(i, 6)]
        if edad < 1.2:
            prims, sombras = self.bebe_y_mama(t, 5.0, escala=1.0 + 0.08 * (edad - 0.7), fase=i)
        else:
            k = (edad - 1.4) / 0.6
            esc = 0.9 + 0.1 * k
            pos = (PEQ[0], y_cabeza("pequeno", 330 + 40 * (1 - k), esc), PEQ[2])
            ib = self.info(t, hijo(2.0), {"cara": "abierta"}, pos, escala=esc, desfase=1.7)
            pb, _ = self.persona(t, hijo(2.0), {"cara": "abierta", "mano_izq": self.mano_en_encimera(pos[0] - 92, "izq", 78), "mano_der": self.mano_en_encimera(pos[0] + 92, "der", 78)}, pos, escala=esc, desfase=1.7, evitar=[])
            hombro = self.mano_en_hombro(ib, "der")
            hombro["palma"] = tuple(np.asarray(hombro["palma"], np.float32) + np.array([0.0, -8.0, 4.0], np.float32))
            hombro["polo"] = (0.8, 0.6, -0.5)
            pa, _ = self.cepillar_desde_atras(t, mama(26), MAMA_DETRAS, ib, U.AMARILLO, amp=7.0, f=2.7, fase=i, pasta=0.5, mano_der=hombro,
                                              extra={"mirar": tuple(ib["boca"]) if i % 2 else None, "evitar": J.obstaculo(pb + utileria_obstaculos(), extra=pen_encimera)})
            prims, sombras = pa + pb, [(pos[0] - 62, 60, 34, 20, 0.35), (pos[0] + 62, 60, 34, 20, 0.35)]
        return self.componer(t, luz, prims + self.encimera_objetos(t), sombras=sombras)

    # 3 · El niño aprieta la boca; mamá hace caras al espejo y cuenta con los dedos
    def escena_3(self, t, e):
        lt = t - e["inicio"]
        abre = lt > 2.25
        cara_n = "apretada" if lt < 2.0 else ("feliz" if lt < 2.25 else "abierta")
        ib = self.info(t, hijo(2.0), {"cara": cara_n}, PEQ, desfase=1.7)
        pb, _ = self.persona(t, hijo(2.0), {"cara": cara_n, "mano_izq": self.mano_en_borde(PEQ[0] - 92, "izq"), "mano_der": self.mano_en_borde(PEQ[0] + 92, "der")}, PEQ, desfase=1.7)
        cuentas = [1.0, 1.6, 2.2, 2.8, 3.4]
        n = escalon(lt, cuentas)
        im0 = self.info(t, mama(26), {}, MAMA_DETRAS)
        sube = suave(progreso(lt, 0.25, 0.75))
        mano_der = None
        if sube > 0.01:
            mano_der = self.contar(im0, "der", n, [pulso(lt, [c + 0.06], 0.07) for c in cuentas])
            if sube < 0.999:  # la mano sube por fuera del brazo del niño hasta junto a la cara
                mano_der = self.transicion(t, mama(26), {}, MAMA_DETRAS, "der", None, mano_der, sube, arco=(70.0, -40.0, -30.0))
        if not abre:  # espera junto a los labios cerrados, moviéndolo un poco
            kw = dict(amp=2.5, f=1.1, hondo=-16, desplaza=(-14, 4, 0))
        else:
            entra = suave(progreso(lt, 2.25, 2.55))
            kw = dict(amp=7.0 * entra, f=2.7, hondo=-16 + 23 * entra, desplaza=(-14 * (1 - entra), 4 * (1 - entra), 0))
        cara_m = "mueca" if 0.3 < lt < 2.1 else "sonrisa"
        pa, _ = self.cepillar_desde_atras(t, mama(26), MAMA_DETRAS, ib, U.AMARILLO, pasta=0.5, mano_der=mano_der,
                                          extra={"cara": cara_m, "mirar": None, "evitar": J.obstaculo(pb + utileria_obstaculos(), extra=pen_encimera)}, **kw)
        return self.componer(t, "noche", pa + pb + self.encimera_objetos(t), sombras=[(PEQ[0] - 62, 60, 34, 20, 0.35), (PEQ[0] + 62, 60, 34, 20, 0.35)])

    # 4 · El niño (4) solo: se estira para abrir el grifo, moja el cepillo seco, mira a la puerta, lo devuelve y cierra
    def escena_4(self, t, e):
        lt = t - e["inicio"]
        self.grifo_movil = True
        palanca = kf(lt, [(0.0, 0.0), (0.46, 0.0), (0.66, 1.0), (3.14, 1.0), (3.34, 0.0)])
        self.palanca = palanca
        agua = limitar((palanca - 0.35) / 0.4)
        # el cuerpo acompaña a la mano: se corre de lado sobre el banquito, se inclina y se pone de puntitas
        quieto, palanca_c, vaso_c, chorro_c = (548.0, 0.0, 0.0, 0.0), (470.0, 0.3, 30.0, 0.0), (588.0, 0.1, 22.0, -0.2), (556.0, 0.14, 16.0, 0.0)
        cierre_c = (470.0, 0.36, 34.0, 0.0)  # para empujar la palanca abierta (más alta) desde arriba se estira un poco más
        xc, incl, punt, giro_c = kf(lt, [(0.06, quieto), (0.42, palanca_c), (0.7, palanca_c), (0.96, vaso_c), (1.18, vaso_c), (1.46, chorro_c), (2.2, chorro_c),
                                         (2.6, vaso_c), (2.94, vaso_c), (3.14, cierre_c), (3.36, palanca_c), (3.72, quieto)])
        pos = (xc, NINO4[1] - punt, NINO4[2])
        pose = {"cara": "neutral", "inclinacion": incl, "giro_cuerpo": giro_c}
        cab_vaso, e_vaso, s_vaso = U.pose_en_vaso(U.VASO, 1)
        arriba = -e_vaso  # sale del vaso a lo largo de su propio eje (el mango roza el borde, nunca lo atraviesa)
        fuera = cab_vaso + arriba * 134.0
        sx_, sy_, sz_ = S.SALIDA_AGUA
        bajo_chorro = np.array([sx_ + 4, sy_ + 64, sz_ - 2], np.float32)
        e_chorro = N((0.92, 0.12, -0.36))  # agarre de puño: la cabeza sale por el pulgar hacia el agua, el mango hacia la derecha
        s_chorro = np.array([0.0, -1.0, 0.0], np.float32)
        mira_puerta = suave(progreso(lt, 1.56, 1.8)) * (1 - suave(progreso(lt, 2.12, 2.36)))
        if mira_puerta > 0:  # ¿viene mamá?
            pose["mirar"] = tuple(np.array(curva(mira_puerta, [(0, tuple(bajo_chorro)), (1, (PUERTA_XZ[0], YM, PUERTA_XZ[1]))])))
        elif lt < 0.7 or 2.94 < lt < 3.4:
            pose["mirar"] = tuple(S.punta_palanca(palanca))
        elif lt < 1.2 or 2.4 < lt <= 2.94:
            pose["mirar"] = tuple(cab_vaso + arriba * 30)
        elif lt < 2.4:
            pose["mirar"] = tuple(bajo_chorro)
        estira = suave(progreso(lt, 0.15, 0.34)) * (1 - suave(progreso(lt, 3.44, 3.62)))  # el codo sale solo cuando la mano ya subió
        polo_d = tuple(np.array([0.55, 1.0, -0.7], np.float32) * (1 - estira) + np.array([1.0, 0.15, -0.35], np.float32) * estira)
        reposo_d = dict(self.mano_en_borde(648.0, "der"))
        yemas_reposo = tuple(punto_mano(reposo_d, "yemas", "nino4", "der"))
        tip0, tip1 = S.punta_palanca(0.0), S.punta_palanca(1.0)
        bajo_palanca = lambda a: S.punta_palanca(a) + np.array([0.0, 13.0, 3.0], np.float32)  # noqa: E731  yemas bajo la punta
        sobre_palanca = lambda a: S.punta_palanca(a) + np.array([0.0, -12.0, 8.0], np.float32)  # noqa: E731  las yemas encima, un poco pasada la punta
        ag_v = cab_vaso + e_vaso * 38.0
        prims_cep = []
        en_mano = 0.96 <= lt < 2.9
        desf = 1.7

        def sube_palanca(tt):  # yemas por debajo de la punta, la palma hacia arriba
            if tt < 0.46:  # primero sube la mano junto al cuerpo (el codo no baja a la encimera) y luego la estira
                p = curva(tt, [(0.06, yemas_reposo), (0.17, (652.0, 1098.0, 104.0)), (0.3, (602.0, 1078.0, 250.0)), (0.39, tuple(bajo_palanca(0.0) + np.array([4.0, 4.0, -36.0], np.float32))), (0.46, tuple(bajo_palanca(0.0)))])
            else:
                p = tuple(bajo_palanca(palanca))
            giro_mano = suave(progreso(tt, 0.1, 0.4))
            nrm = kf(giro_mano, [(0.0, (0.0, 1.0, 0.0)), (0.5, (-1.0, 0.0, 0.0)), (1.0, (0.0, -1.0, 0.0))])
            return {"yemas": tuple(p), "dir": (-0.1, 0.0, 1.0), "nrm": nrm, "mezcla": ("abierta", "relajada", 0.25 * giro_mano), "polo": polo_d}

        def baja_palanca(tt):  # yemas encima de la punta, la palma hacia abajo
            if tt < 3.14:
                p = curva(tt, [(2.98, tuple(ag_v + np.array([-10.0, -40.0, -10.0], np.float32))), (3.08, tuple(sobre_palanca(1.0) + np.array([0.0, -18.0, -34.0], np.float32))), (3.14, tuple(sobre_palanca(1.0)))])
            elif tt < 3.34:
                p = tuple(sobre_palanca(palanca))
            else:
                p = curva(tt, [(3.34, tuple(sobre_palanca(0.0))), (3.48, (602.0, 1078.0, 250.0)), (3.6, (652.0, 1098.0, 104.0)), (3.72, yemas_reposo)])
            return {"yemas": tuple(p), "dir": (-0.1, 0.22, 1.0), "nrm": (0.0, 1.0, 0.0), "forma": "abierta", "polo": polo_d}

        def agarre_vaso(tt, ida=True):  # la mano abierta que llega al cuello del cepillo (o se va tras soltarlo)
            if ida:
                p = curva(tt, [(0.7, tuple(bajo_palanca(1.0) + np.array([20.0, 0.0, -60.0], np.float32))), (0.86, tuple(ag_v + np.array([-34.0, -30.0, -26.0], np.float32))), (0.96, tuple(ag_v))])
                k = progreso(tt, 0.86, 0.96)
                return {"agarre": tuple(p), "eje": tuple(-e_vaso), "mezcla": ("relajada", "agarre", k), "polo": polo_d}
            return {"agarre": tuple(ag_v), "eje": tuple(-e_vaso), "mezcla": ("agarre", "abierta", progreso(tt, 2.9, 3.0)), "polo": polo_d}

        if lt < 0.06:
            mano_der = reposo_d
        elif lt < 0.7:  # deja el borde, va por debajo de la palanca y la levanta
            mano_der = self.transicion(t, hijo(4), pose, pos, "der", reposo_d, sube_palanca(lt), suave(progreso(lt, 0.06, 0.16)), desfase=desf)
            self.permisos_grifo = (("hijo", "mano_der"),)
        elif lt < 0.96:  # suelta la palanca y va al cuello del cepillo
            if lt > 0.86:
                self.permisos_vaso = (("hijo", "mano_der"),)  # los dedos se cierran alrededor del cuello
            mano_der = self.transicion(t, hijo(4), pose, pos, "der", sube_palanca(0.66), agarre_vaso(lt), suave(progreso(lt, 0.7, 0.86)), desfase=desf)
        elif en_mano:  # lo saca del vaso, lo moja bajo el chorro (mirando si viene mamá), lo sacude y lo devuelve
            sacude = 10 * math.sin((lt - 1.98) * 2 * math.pi * 7) * suave(progreso(lt, 1.98, 2.04)) * (1 - suave(progreso(lt, 2.18, 2.24)))
            cab = np.array(curva(lt, [(0.96, tuple(cab_vaso)), (1.16, tuple(fuera)), (1.34, tuple(bajo_chorro + np.array([60.0, -90.0, -40.0], np.float32))), (1.48, tuple(bajo_chorro)),
                                      (1.98, tuple(bajo_chorro)), (2.24, tuple(bajo_chorro + np.array([40.0, -70.0, -50.0], np.float32))), (2.66, tuple(fuera)), (2.9, tuple(cab_vaso))]), np.float32)
            cab = cab + np.array([2.0 * math.sin(lt * 2 * math.pi * 2.5) * progreso(lt, 1.5, 1.6) * (1 - progreso(lt, 1.9, 1.98)), sacude, 0.0], np.float32)
            k_e = suave(progreso(lt, 1.16, 1.44)) * (1 - suave(progreso(lt, 2.3, 2.66)))
            eje = N(e_vaso * (1 - k_e) + e_chorro * k_e)
            cerdas = N(s_vaso * (1 - k_e) + s_chorro * k_e)
            prims_cep, ag = U.cepillo(cab, eje, cerdas, U.AZUL, agarre=38.0)
            self.sostenido(prims_cep, "hijo", "der")
            U.permitir(prims_cep, ("vaso", "objeto"))
            mano_der = {"agarre": tuple(ag), "eje": tuple(-eje), "polo": polo_d}
        elif lt < 3.14:  # lo suelta en el vaso y va a la palanca por arriba
            self.permisos_vaso = (("hijo", "mano_der"),)
            mano_der = self.transicion(t, hijo(4), pose, pos, "der", agarre_vaso(lt, ida=False), baja_palanca(lt), suave(progreso(lt, 2.96, 3.1)), desfase=desf)
        elif lt < 3.72:  # baja la palanca con las yemas y vuelve al borde
            if lt < 3.36:
                self.permisos_grifo = (("hijo", "mano_der"),)
            mano_der = self.transicion(t, hijo(4), pose, pos, "der", baja_palanca(lt), reposo_d, suave(progreso(lt, 3.6, 3.72)), desfase=desf)
        else:
            mano_der = reposo_d
        pose["mano_der"] = mano_der
        apoyo = self.mano_en_borde(xc - 100.0, "izq")  # la otra mano se desliza por el borde y, al inclinarse, se apoya más adentro
        k_ap = incl / 0.3
        apoyo["palma"] = (apoyo["palma"][0] - 6 * k_ap, apoyo["palma"][1] - 2.0, apoyo["palma"][2] + 64 * k_ap)
        apoyo["polo"] = tuple(np.array(apoyo["polo"], np.float32) * (1 - k_ap) + np.array([-1.0, -0.6, -0.3], np.float32) * k_ap)
        pose["mano_izq"] = apoyo
        pb, _ = self.persona(t, hijo(4), pose, pos, desfase=desf)
        prims = pb + prims_cep + self.encimera_objetos(t, ("amarillo", None if en_mano else "azul"))
        return self.componer(t, "noche", prims, agua=agua, sombras=[(pos[0] - 72, 52, 34, 16, 0.35)])

    # 5 · Mamá entra, toma el cepillo del vaso, toca las cerdas secas, lo mira por el espejo, sonríe, lo rodea por detrás y se los cepilla
    def escena_5(self, t, e):
        lt = t - e["inicio"]
        pos_n, desf_n = NINO4, 1.7
        cara_n = "neutral" if lt < 2.3 else ("sonrisa" if lt < 2.95 else "abierta")
        pose_n = {"cara": cara_n, "cabeceo": -0.08 * suave(progreso(lt, 3.0, 3.3))}  # al final echa la cabeza un poco atrás, contra mamá
        ib = self.info(t, hijo(4), pose_n, pos_n, desfase=desf_n)
        pb, _ = self.persona(t, hijo(4), dict(pose_n, mano_izq=self.mano_en_borde(pos_n[0] - 100, "izq"), mano_der=self.mano_en_borde(pos_n[0] + 100, "der")), pos_n, desfase=desf_n)
        sombras = [(pos_n[0] - 70, 62, 36, 20, 0.4), (pos_n[0] + 70, 62, 36, 20, 0.4)]
        LADO = (900.0, YM, -130.0)  # junto al vaso, un paso atrás del niño (sin tocar sus codos)
        DETRAS = (562.0, YM, -160.0)  # detrás, a un palmo: ni la cabeza ni los codos del niño se meten en ella
        if lt < 1.2:  # entra caminando desde la puerta
            pm, _ = self.mama_caminando(t, 26, 0.0, 1.2, PUERTA_XZ, (LADO[0], LADO[2]), {"cara": "neutral", "mirar": tuple(ib["cabeza"])}, giro_cuerpo=-0.15)
            return self.componer(t, "noche", pm + pb + self.encimera_objetos(t, ("amarillo", "azul")), sombras=sombras)
        # lo rodea por detrás con dos pasos cortos, por fuera de los codos del niño
        k_paso = progreso(lt, 2.55, 3.06)
        mx, mz = curva(lt, [(2.55, (LADO[0], LADO[2])), (2.8, (744.0, -224.0)), (3.06, (DETRAS[0], DETRAS[2]))]) if lt > 2.55 else (LADO[0], LADO[2])
        pos_m = (mx, YM + 2.5 * (1 - math.cos(4 * math.pi * k_paso)), mz)
        pose_m = {"cara": "neutral", "giro_cuerpo": -0.15 * (1 - suave(progreso(lt, 1.2, 1.6))), "inclinacion": 0.1 * suave(progreso(lt, 2.9, 3.3))}
        cab_v, e_v, s_v = U.pose_en_vaso(U.VASO, 1)
        ag_v = cab_v + e_v * 38.0
        delante = np.array([858.0, 968.0, 4.0], np.float32)  # la cabeza del cepillo delante de su pecho, inclinado
        e_del = N((-0.55, 0.8, -0.12))
        s_del = N((1.0, 0.2, -0.35))  # las cerdas hacia su otra mano (la del pulgar que las toca)
        cep, mano_der, mano_izq = [], None, None
        mira = tuple(ib["cabeza"])
        corona = self.mano_en_hombro(ib, "der")  # la otra mano, en su hombro (lo sostiene con suavidad)
        corona["palma"] = tuple(np.asarray(corona["palma"], np.float32) + np.array([0.0, -4.0, 4.0], np.float32))
        corona["polo"] = (0.8, 0.6, -0.5)
        if lt < 1.62:  # va al vaso con su mano derecha (la que usa para cepillar) y agarra el cepillo del niño por el cuello
            mira = tuple(cab_v)
            llega = {"agarre": tuple(curva(lt, [(1.2, (770.0, 1120.0, -40.0)), (1.48, tuple(ag_v + np.array([-24.0, 12.0, -40.0], np.float32))), (1.62, tuple(ag_v))])),
                     "eje": tuple(-e_v), "mezcla": ("relajada", "agarre", progreso(lt, 1.48, 1.62)), "polo": (-0.4, 0.85, -0.45)}
            mano_izq = self.transicion(t, mama(26), pose_m, pos_m, "izq", None, llega, suave(progreso(lt, 1.2, 1.4)))
            if lt > 1.48:
                self.permisos_vaso = (("mama", "mano_izq"),)
        elif lt < 2.95:
            k_e = suave(progreso(lt, 1.76, 1.96))
            if lt < 2.55:  # lo saca a lo largo de su eje y lo pone delante de su pecho
                cab = np.array(curva(lt, [(1.62, tuple(cab_v)), (1.8, tuple(cab_v - e_v * 134.0)), (1.96, tuple(delante))]), np.float32) if lt < 1.96 else delante.copy()
                eje, cerdas = N(e_v * (1 - k_e) + e_del * k_e), N(s_v * (1 - k_e) + s_del * k_e)
            else:  # lo lleva arriba, de su lado izquierdo de la pantalla, mientras rodea al niño
                k_l = suave(progreso(lt, 2.55, 2.85))
                arriba_izq = np.array([mx - 150.0, 880.0, mz + 100.0], np.float32)
                cab = delante * (1 - k_l) + arriba_izq * k_l
                eje, cerdas = N(e_del * (1 - k_l) + np.array([0.0, 1.0, -0.15], np.float32) * k_l), s_del
            # acomoda el agarre: mientras la otra mano sujeta la cabeza, la suya baja por el mango
            largo = 38.0 + 70.0 * suave(progreso(lt, 2.24, 2.46))
            afloja = math.sin(math.pi * progreso(lt, 2.24, 2.46))
            cep, ag_i = U.cepillo(cab, eje, cerdas, U.AZUL, agarre=largo)
            U.permitir(cep, ("vaso", "objeto"))
            self.sostenido(cep, "mama", "izq")
            mano_izq = {"agarre": tuple(ag_i), "eje": tuple(-eje), "mezcla": ("agarre", "relajada", 0.45 * afloja), "polo": (-0.4, 0.85, -0.45)}
            # la otra mano: toca las cerdas con el pulgar (están secas), sujeta la cabeza mientras ella reacomoda y se va
            toca = suave(progreso(lt, 1.98, 2.14))
            p_cerdas = cab + cerdas * 10.0 + eje * (-10.0 + 3.0 * math.sin(lt * 24) * (1 - progreso(lt, 2.2, 2.3)))
            pinza = {"pinza": tuple(p_cerdas), "dir": (-0.6, -0.5, 0.6), "nrm": (-0.3, 0.2, -1.0), "mezcla": ("relajada", "pinza", 0.6 + 0.4 * toca), "polo": (0.9, 0.5, -0.3)}
            if lt < 1.96:
                mano_der = None
            elif lt < 2.46:
                mano_der = self.transicion(t, mama(26), pose_m, pos_m, "der", None, pinza, suave(progreso(lt, 1.96, 2.12)))
                U.permitir(cep, ("mama", "mano_der"))
            elif lt < 2.8:
                mano_der = self.transicion(t, mama(26), pose_m, pos_m, "der", pinza, None, suave(progreso(lt, 2.46, 2.72)))
                if lt < 2.56:
                    U.permitir(cep, ("mama", "mano_der"))
            else:  # va a apoyarse en su hombro
                mano_der = self.transicion(t, mama(26), pose_m, pos_m, "der", None, corona, suave(progreso(lt, 2.85, 3.2)), arco=(20.0, -170.0, -16.0))
            mira = tuple(cab) if lt < 2.2 else (tuple(ib["cabeza"]) if lt > 2.62 else None)  # mira las cerdas y luego al niño por el espejo
            pose_m["cara"] = "neutral" if lt < 2.22 else "sonrisa"
        else:  # le lleva el cepillo a la boca por delante de la mejilla y, ya detrás, se los cepilla
            entra = suave(progreso(lt, 2.95, 3.35))
            corona_m = self.transicion(t, mama(26), pose_m, pos_m, "der", None, corona, suave(progreso(lt, 2.85, 3.2)), arco=(20.0, -170.0, -16.0))
            if lt < 3.35:
                cab0 = np.array([DETRAS[0] - 150.0, 880.0, DETRAS[2] + 100.0], np.float32)
                _, spec_b = self.cepillo_en_boca(t, ib, (-0.80, 0.22, 0.56), U.AZUL, amp=0.0, agarre=108.0)
                e_b = -np.asarray(spec_b["eje"], np.float32)
                cab_b = np.asarray(spec_b["agarre"], np.float32) - e_b * 108.0
                hacia = N(np.asarray(ib["cabeza"], np.float32) - np.asarray(ib["boca"], np.float32))
                cab = np.array(curva(entra, [(0.0, tuple(cab0)), (0.55, (ib["cabeza"][0] - 120.0, ib["cabeza"][1] - 20.0, ib["cabeza"][2] + 150.0)), (1.0, tuple(cab_b))]), np.float32)
                eje = N(np.array([0.0, 1.0, -0.15], np.float32) * (1 - entra) + e_b * entra)
                cep, ag = U.cepillo(cab, eje, N(s_del * (1 - entra) + hacia * entra), U.AZUL, agarre=112.0 - 4.0 * entra)
                U.permitir(cep, ("hijo", "cabeza"), solo="cabezal")
                self.sostenido(cep, "mama", "izq")
                mano_izq = {"agarre": tuple(ag), "eje": tuple(-eje), "polo": tuple(np.array([-0.4, 0.85, -0.45], np.float32) * (1 - entra) + np.array([-0.9, 0.5, -0.6], np.float32) * entra)}
                mano_der = corona_m
                mira = tuple(ib["boca"])
                pose_m["cara"] = "sonrisa"
            else:
                pa, _ = self.cepillar_desde_atras(t, mama(26), pos_m, ib, U.AZUL, amp=7.0 * suave(progreso(lt, 3.35, 3.55)), f=2.7, mano_der=corona_m,
                                                  extra={"inclinacion": pose_m["inclinacion"]}, polo=(-0.9, 0.5, -0.6))
                return self.componer(t, "noche", pa + pb + self.encimera_objetos(t, ("amarillo", None)), sombras=sombras)
        pose_m.update({"mano_der": mano_der, "mano_izq": mano_izq, "mirar": mira})
        pm, _ = self.persona(t, mama(26), pose_m, pos_m)
        prims = pm + pb + cep + self.encimera_objetos(t, ("amarillo", "azul" if lt < 1.62 else None))
        return self.componer(t, "noche", prims, sombras=sombras)

    # 6 · De regreso del dentista: presume la pegatina de «paciente valiente» en el dorso de la mano; mamá mete la tarjeta en el marco
    def escena_6(self, t, e):
        lt = t - e["inicio"]
        esc = 1.03
        quien_n = hijo(5)
        pos_n = (452.0, y_cabeza("nino4", 215, esc), NINO4[2])
        pose_n = {"cara": "dientes"}
        ib = self.info(t, quien_n, pose_n, pos_n, escala=esc, desfase=1.7)
        hx, hy, hz = ib["cabeza"]
        R = ib["R"]
        # levanta su mano derecha (la de la izquierda de la pantalla, lejos de mamá) junto a la cara, con el dorso
        # hacia el espejo: la pegatina a la vista; luego la vuelve a apoyar
        ensena = suave(progreso(lt, 0.15, 0.5)) * (1 - suave(progreso(lt, 1.2, 1.55)))
        reposo = self.mano_en_borde(pos_n[0] - 100, "izq")
        alto = {"centro": (hx - 1.18 * R, hy + 0.45 * R + 4 * math.sin(lt * 5.0), hz + 0.75 * R), "dir": (-0.15, -1.0, 0.12), "nrm": (0.35, 0.0, -1.0), "forma": "relajada", "polo": (-0.55, 0.7, -0.6)}
        mano_izq = self.transicion(t, quien_n, pose_n, pos_n, "izq", reposo, alto, ensena, escala=esc, desfase=1.7, arco=(-10.0, -70.0, -16.0))
        pb, ibn = self.persona(t, quien_n, dict(pose_n, mano_izq=mano_izq, mano_der=self.mano_en_borde(pos_n[0] + 100, "der")), pos_n, escala=esc, desfase=1.7)
        pb += self.pegatina_en_mano(ibn, "izq", quien_n)
        # mamá, a su lado (sin pegarse), con la tarjeta de la próxima cita; da un paso hacia el centro para llegar al marco
        x_m = kf(lt, [(1.3, 880.0), (1.75, 814.0), (2.85, 814.0), (3.35, 880.0)])
        pos_m = (x_m, YM + 3.0 * math.sin(math.pi * progreso(lt, 1.3, 1.75)) + 3.0 * math.sin(math.pi * progreso(lt, 2.85, 3.35)), -20.0)
        pinza_ini = (pos_m[0] - 110, pos_m[1] + 310 + 3 * math.sin(lt * 2), pos_m[2] + 130)  # a un lado, por delante (ni su cadera ni los cepillos del vaso)
        prims_t, mano, incl, mirar, marco, f_sombra = self.meter_en_marco(lt, "hijo", 0, 1.3, 1.5, 2.45, 2.7, pinza_ini, "mama", "izq", esquina="der", inclinacion=0.5, polo=(-0.15, 0.9, -0.4), reposo=pinza_ini)
        puntas = 46.0 * incl / 0.5  # se pone de puntitas para llegar al marco
        pos_mm = (pos_m[0], pos_m[1] - puntas, pos_m[2])
        pose_m = {"cara": "sonrisa", "inclinacion": incl, "mano_der": self.mano_en_encimera(pos_m[0] + 150, "der", 60)}
        if mano is None:
            hombro = self.mano_en_hombro(ib, "der")  # al final apoya la mano en su hombro
            hombro["palma"] = tuple(np.asarray(hombro["palma"], np.float32) + np.array([0.0, -8.0, 4.0], np.float32))
            espera = {"pinza": pinza_ini, "dir": (-0.3, -0.85, 0.4), "nrm": (-0.2, 0.3, -1.0), "forma": "relajada", "polo": (-0.3, 0.9, -0.3)}
            mano = self.transicion(t, mama(26), pose_m, pos_mm, "izq", espera, hombro, suave(progreso(lt, 3.55, 3.9)), arco=(0.0, -40.0, 0.0))
        pose_m["mano_izq"] = mano
        pose_m["mirar"] = mirar if mirar is not None else (tuple(ib["cabeza"]) if lt < 1.3 or lt > 3.4 else None)
        pm, _ = self.persona(t, mama(26), pose_m, pos_mm)
        prims = pm + pb + (prims_t or []) + self.encimera_objetos(t)
        sombras = [(pos_n[0] - 72, 64, 36, 20, 0.4 * (1 - ensena)), (pos_n[0] + 72, 64, 36, 20, 0.4), (pos_m[0] + 150, 60, 36, 20, 0.4)]
        return self.componer(t, "manana", prims, marco=marco or (), marco_sombra=f_sombra, sombras=sombras)

    # 7 · Se le cae el primer diente: lo deja en el pañuelito que mamá tiene en la palma; ella lo envuelve y lo guarda en la cajita
    def escena_7(self, t, e):
        lt = t - e["inicio"]
        esc = 1.06
        quien_n = hijo(6)
        pos_n = (470.0, y_cabeza("nino4", 215, esc), NINO4[2])
        pos_m = (880.0, YM, -20.0)
        lata_c = np.array([760.0, S.ENCIMERA_Y - U.LATA_H[1], 108.0], np.float32)  # delante de su lado izquierdo de la pantalla, sin tocar el vaso
        palma_m = np.array([600.0, 1102.0, 132.0], np.float32)  # mamá acerca la palma bajo la mano del niño (su brazo pasa por delante, abajo)
        sobre_lata = lata_c + np.array([-6.0, -U.LATA_H[1] - 84.0, -2.0], np.float32)  # alto: el antebrazo pasa por encima de la tapa abierta
        tapa = 1.0 - suave(progreso(lt, 2.74, 3.04))
        prims_lata, borde_tapa = U.lata(tapa, centro=lata_c, llena=False, bisagra="delante")
        prims = list(prims_lata)
        # --- el niño: enseña el diente entre los dedos, lo lleva sobre la palma de mamá y lo suelta
        mira_n = None
        pose_n = {"cara": "dientes", "hueco": True, "cabeceo": 0.1 * suave(progreso(lt, 0.9, 1.2)) * (1 - suave(progreso(lt, 1.9, 2.2)))}
        ib = self.info(t, quien_n, pose_n, pos_n, escala=esc, desfase=1.7)
        hx, hy, hz = ib["cabeza"]
        R = ib["R"]
        ensena = np.array([hx + 0.95 * R, hy + 0.35 * R, hz + 0.85 * R], np.float32)  # lo enseña en alto, junto a la cara
        suelta_p = palma_m + np.array([0.0, -54.0, 0.0], np.float32)
        p_diente = np.array(curva(lt, [(0.0, tuple(ensena)), (0.9, tuple(ensena + np.array([0.0, -4.0, 0.0], np.float32))), (1.42, tuple(suelta_p))]), np.float32)
        p_diente = p_diente + np.array([0.0, 3.0 * math.sin(lt * 3.0) * (1 - progreso(lt, 0.9, 1.1)), 0.0], np.float32)
        gira = suave(progreso(lt, 0.9, 1.4))  # al soltarlo sobre la palma de mamá, los dedos apuntan hacia abajo (la mano queda arriba)
        dir_n = kf(gira, [(0.0, (0.35, -0.75, 0.5)), (1.0, (0.15, 0.45, 0.88))])  # al soltarlo, la mano apunta adelante y un poco abajo
        polo_n = kf(gira, [(0.0, (0.5, 0.8, -0.3)), (1.0, (-0.1, 1.0, -0.55))])  # el codo abajo, pegado a su costado: el antebrazo va hacia delante
        pinza_n = {"pinza": tuple(p_diente), "dir": dir_n, "nrm": (-0.6, 0.1, 0.8), "forma": "pinza", "polo": polo_n}
        reposo_n = self.mano_en_borde(pos_n[0] + 100, "der")
        if lt < 1.5:
            mano_n = pinza_n
        else:  # abre los dedos y vuelve a apoyarse
            pinza_n["mezcla"] = ("pinza", "relajada", progreso(lt, 1.5, 1.6))
            medio = {"centro": (pos_n[0] + 92, 1060.0, 72.0), "dir": (0.25, 0.5, 0.83), "nrm": (-0.4, -0.2, 0.9), "forma": "relajada", "polo": (0.8, 0.8, -0.2)}
            if lt < 1.7:  # primero la retira hacia arriba y hacia sí (lejos de la mano de mamá) y luego la apoya
                mano_n = self.transicion(t, quien_n, pose_n, pos_n, "der", pinza_n, medio, suave(progreso(lt, 1.52, 1.7)), escala=esc, desfase=1.7, arco=(0.0, -24.0, 0.0))
            else:
                mano_n = self.transicion(t, quien_n, pose_n, pos_n, "der", medio, reposo_n, suave(progreso(lt, 1.7, 1.98)), escala=esc, desfase=1.7)
        if lt > 0.95:  # mira hacia la mano de mamá sin agachar del todo la cabeza (se le sigue viendo la cara)
            obj = palma_m if lt < 2.2 else lata_c
            mira_n = (float(obj[0]), float(obj[1]) - 150.0, float(obj[2]) + 320.0)
        if mira_n is not None:
            pose_n["mirar"] = mira_n
        pose_nb = dict(pose_n, mano_der=mano_n, mano_izq=self.mano_en_borde(pos_n[0] - 100, "izq"))
        pb, _ = self.persona(t, quien_n, pose_nb, pos_n, escala=esc, desfase=1.7, evitar=prims_lata)
        # --- mamá: palma abierta con el pañuelito, lo envuelve, lo lleva a la cajita, lo suelta, cierra la tapa y le da dos palmaditas
        pose_m = {"cara": "sonrisa", "mano_der": self.mano_en_encimera(pos_m[0] + 150, "der", 60)}
        recibe = {"palma": tuple(palma_m), "dir": (-0.88, 0.0, 0.47), "nrm": (0.0, -1.0, 0.0), "forma": "abierta", "polo": (-0.15, 1.0, 0.05)}
        envuelve = suave(progreso(lt, 1.74, 2.02))
        if lt < 1.66:  # sube la mano junto al cuerpo (por fuera de la encimera) y la adelanta con la palma hacia arriba
            alzada = {"palma": (742.0, 1110.0, -30.0), "dir": (-0.3, -0.85, 0.42), "nrm": (-0.3, -0.4, 0.9), "forma": "relajada", "polo": (-0.15, 1.0, 0.05)}
            mano_m = self.transicion(t, mama(26), pose_m, pos_m, "izq", None, alzada, suave(progreso(lt, 0.2, 0.55)))
            mano_m = self.transicion(t, mama(26), pose_m, pos_m, "izq", mano_m, recibe, suave(progreso(lt, 0.5, 0.95)))
        elif lt < 2.5:  # cierra la mano sobre el pañuelo y la lleva, palma abajo, sobre la cajita
            lleva = suave(progreso(lt, 1.98, 2.46))
            p = np.array(curva(lt, [(1.98, tuple(palma_m)), (2.25, tuple(palma_m * 0.5 + sobre_lata * 0.5 + np.array([0.0, -26.0, 0.0], np.float32))), (2.46, tuple(sobre_lata))]), np.float32) if lt > 1.98 else palma_m
            nrm = kf(lleva, [(0.0, (0.0, -1.0, 0.0)), (0.5, (-0.9, 0.0, 0.3)), (1.0, (0.0, 1.0, 0.15))])
            mano_m = {"palma": tuple(p), "dir": (-0.75, 0.25 * lleva, 0.6), "nrm": nrm, "mezcla": ("abierta", "agarre", 0.82 * envuelve), "polo": (-0.15, 1.0, 0.05)}
        elif lt < 2.74:  # abre la mano: la bolita cae dentro
            mano_m = {"palma": tuple(sobre_lata), "dir": (-0.75, 0.25, 0.6), "nrm": (0.0, 1.0, 0.15), "mezcla": ("agarre", "abierta", 0.82 * (1 - progreso(lt, 2.5, 2.62))), "polo": (-0.15, 1.0, 0.05)}
        elif lt < 3.06:  # empuja la tapa con las yemas
            yemas = np.asarray(borde_tapa, np.float32) + np.array([0.0, -4.0, -6.0], np.float32)
            empuja = {"yemas": tuple(yemas), "dir": (0.0, 0.35, 1.0), "nrm": (-0.15, 1.0, 0.05), "forma": "abierta", "polo": (-0.15, 1.0, 0.05)}
            abierta = {"palma": tuple(sobre_lata), "dir": (-0.75, 0.25, 0.6), "nrm": (0.0, 1.0, 0.15), "forma": "abierta", "polo": (-0.15, 1.0, 0.05)}
            mano_m = self.transicion(t, mama(26), pose_m, pos_m, "izq", abierta, empuja, suave(progreso(lt, 2.74, 2.86)))
            U.permitir(prims_lata, ("mama", "mano_izq"))
        else:  # la mano descansa sobre la tapa y le da dos palmaditas
            pal = 6.0 * pulso(lt, [3.3, 3.62], 0.07)
            descansa = {"palma": tuple(lata_c + np.array([0.0, -U.LATA_H[1] - 13.0 - pal, 0.0], np.float32)), "dir": (-0.15, 0.2, 1.0), "nrm": (0.0, 1.0, 0.0), "forma": "relajada", "polo": (-0.15, 1.0, 0.05)}
            yemas = np.asarray(borde_tapa, np.float32) + np.array([0.0, -4.0, -6.0], np.float32)
            empuja = {"yemas": tuple(yemas), "dir": (0.0, 0.35, 1.0), "nrm": (-0.15, 1.0, 0.05), "forma": "abierta", "polo": (-0.15, 1.0, 0.05)}
            mano_m = self.transicion(t, mama(26), pose_m, pos_m, "izq", empuja, descansa, suave(progreso(lt, 3.06, 3.2)))
            U.permitir(prims_lata, ("mama", "mano_izq"))
        pose_m["mano_izq"] = mano_m
        pose_m["mirar"] = tuple(p_diente) if lt < 1.6 else (tuple(lata_c) if lt < 3.2 else tuple(ib["cabeza"]))
        pm, im = self.persona(t, mama(26), pose_m, pos_m, evitar=pb + prims_lata)
        pb, _ = self.persona(t, quien_n, pose_nb, pos_n, escala=esc, desfase=1.7, evitar=pm + prims_lata)  # segunda pasada: el niño también esquiva a mamá
        # el pañuelito (y el diente) van con la mano de mamá
        pts = im["puntos"].get("izq")
        if lt < 2.62 and pts:
            d_m, n_m, _ = pts["marco"]
            if lt < 1.98:  # lo trae hecho bolita en la mano y lo extiende sobre la palma; luego lo envuelve
                extiende = suave(progreso(lt, 0.62, 0.95))
                k_bola = max(1.0 - extiende, envuelve)
                c_pap = np.asarray(pts["palma"], np.float32) + n_m * (3.0 + 8.0 * k_bola) if extiende > 0.02 else np.asarray(pts["agarre"], np.float32)
                pp = U.papelito(c_pap, k_bola, J.rot_y(-0.3) if k_bola < 0.5 else None)
            elif lt < 2.55:
                c_pap = np.asarray(pts["agarre"], np.float32)
                pp = U.papelito(c_pap, 1.0)
            else:
                k = progreso(lt, 2.55, 2.66)
                c0 = np.asarray(pts["agarre"], np.float32)
                fondo = lata_c + np.array([0.0, U.LATA_H[1] - 15.0, 0.0], np.float32)
                c_pap = c0 + (fondo - c0) * (k * k)
                pp = U.papelito(c_pap, 1.0)
                U.permitir(pp, ("lata", "objeto"))
            self.sostenido(pp, "mama", "izq")
            prims += pp
            if lt < 1.66:  # el diente: en los dedos del niño, cae y queda sobre el pañuelo
                cae = progreso(lt, 1.46, 1.58)
                d_pos = p_diente + (c_pap + np.array([0.0, -8.0, 0.0], np.float32) - p_diente) * (cae * cae)
                d_prims = U.diente(d_pos)
                U.permitir(d_prims, ("hijo", "mano_der"), ("mama", "mano_izq"), ("papel", "objeto"))
                prims += d_prims
        prims += pm + pb + self.encimera_objetos(t)
        sombras = [(760, 108, 62, 42, 0.6), (pos_n[0] - 72, 64, 36, 20, 0.35), (pos_n[0] + 72, 64, 36, 20, 0.35 if lt > 2.0 else 0.0), (pos_m[0] + 150, 60, 36, 20, 0.4)]
        return self.componer(t, "manana", prims, marco=tarjeta_en_ranura("hijo", 0), sombras=sombras)

    # 8 · De noche, mamá sola deja unas monedas y escribe la nota del Ratón de los Dientes (acercamiento lento)
    def escena_8(self, t, e):
        lt = t - e["inicio"]
        nota_c = np.array([520.0, S.ENCIMERA_Y - 0.9, 112.0], np.float32)
        pos_m = (MAMA_SOLA[0], MAMA_SOLA[1], -72.0)  # un paso atrás: se inclina sobre la nota sin que su pecho la tape
        ancho, fondo = 150.0, 94.0
        R_nota = J.rot_y(math.pi)  # el texto mira hacia ella (desde el espejo se ve al revés)
        largos = _metricas_nota()
        f1 = largos[0] / sum(largos)
        if lt < 1.45:
            k_txt, viaje = 0.0, 0.0
        elif lt < 2.55:
            k_txt, viaje = f1 * progreso(lt, 1.45, 2.55), 0.0
        elif lt < 2.78:
            k_txt, viaje = f1, progreso(lt, 2.55, 2.78)
        else:
            k_txt, viaje = f1 + (1 - f1) * progreso(lt, 2.78, 3.7), 0.0
        tex = tex_nota(round(k_txt * 60) / 60)
        prims = U.nota_en_encimera(nota_c, ancho, fondo, tex, giro=math.pi)

        def en_nota(u, v, alto=0.0):
            return nota_c + R_nota @ np.array([(u - 0.5) * ancho, -1.0 - alto, (v - 0.5) * fondo], np.float32)

        if viaje > 0:  # levanta el lápiz y lo lleva al principio de la segunda línea
            u0, v0 = NOTA_LINEAS[0][1] + largos[0], NOTA_LINEAS[0][2]
            u1, v1 = NOTA_LINEAS[1][1], NOTA_LINEAS[1][2]
            punta = en_nota(u0 + (u1 - u0) * suave(viaje), v0 + (v1 - v0) * suave(viaje), 3.5 + 12 * math.sin(math.pi * viaje))
        elif 1.45 <= lt < 3.7:
            u, v = punta_nota(k_txt, t)
            punta = en_nota(u, v, 3.5 + 1.5 * max(0.0, math.sin(lt * 2 * math.pi * 3.2)) ** 6)
        elif lt < 1.45:
            ini = en_nota(NOTA_LINEAS[0][1], NOTA_LINEAS[0][2], 3.5)
            punta = np.array(curva(lt, [(0.0, tuple(ini + np.array([-90, -70, -10], np.float32))), (1.0, tuple(ini + np.array([-50, -40, 0], np.float32))), (1.45, tuple(ini))]), np.float32)
        else:
            fin = en_nota(*punta_nota(1.0, 3.7), 3.5)
            punta = fin + np.array([-18, -40, -10], np.float32) * suave(progreso(lt, 3.7, 4.0))
        eje_l = N((-0.32, -0.8, -0.5))  # el lápiz se inclina hacia su mano derecha (pantalla izquierda)
        lap = U.lapiz(punta, eje_l)
        self.sostenido(lap, "mama", "izq")
        prims += lap
        mano_izq = {"pinza": tuple(punta + eje_l * 30), "dir": (0.2, 0.62, 0.76), "nrm": (0.72, 0.1, -0.2), "mezcla": ("lapiz", "lapiz", 1.0), "polo": (-0.35, 0.7, -0.6)}
        sitio = np.array([660.0, S.ENCIMERA_Y, 118.0], np.float32)
        p_mon = np.array(curva(lt, [(0.0, (690, 1090, 110)), (0.5, tuple(sitio + np.array([0, -14, 0], np.float32)))]), np.float32)
        if lt < 0.55:
            mano_der = {"pinza": tuple(p_mon), "dir": (-0.2, 0.75, 0.6), "nrm": (-0.6, 0.3, -0.6), "polo": (0.35, 0.7, -0.6)}
            for k in range(3):
                mon = U.moneda(p_mon + np.array([0, k * 3.6 - 4, 0], np.float32))
                self.sostenido(mon, "mama", "der")
                prims += mon
        else:
            for k, (dx_, dz_) in enumerate(((0, 0), (15, 9), (-10, 14))):
                prims += U.moneda(sitio + np.array([dx_, -1.9 - 3.6 * (k == 0) * 0, dz_], np.float32))
            sujeta = en_nota(0.03, 0.55, 0) + np.array([32, -9, 0], np.float32)
            p = np.array(curva(lt, [(0.55, tuple(sitio + np.array([0, -44, 0], np.float32))), (0.8, tuple((sitio + sujeta) / 2 + np.array([0, -46, 0], np.float32))), (1.05, tuple(sujeta))]), np.float32)
            mano_der = {"palma": tuple(p), "dir": (-0.45, 0.12, 0.88), "nrm": (0, 1, 0), "mezcla": ("pinza", "abierta", progreso(lt, 0.55, 0.8)), "polo": (0.35, 0.7, -0.6)}
            U.permitir(prims[:1], ("mama", "mano_der"))
            if lt < 0.75:  # los dedos acaban de soltar las monedas
                for q in prims:
                    if getattr(q, "dueno", "") == "moneda":
                        U.permitir([q], ("mama", "mano_der"))
        cara = "concentrada" if lt < 3.75 else "sonrisa"
        pm, _ = self.persona(t, mama(26), {"cara": cara, "mirar": tuple(punta), "inclinacion": 0.4, "mano_der": mano_der, "mano_izq": mano_izq}, pos_m, evitar=prims)  # inclinada sobre la nota, los codos por fuera
        prims += pm + self.encimera_objetos(t)
        sombras = [(520, 112, 78, 48, 0.25), (660, 122, 28, 20, 0.4 if lt > 0.5 else 0.0)]
        return self.componer(t, "noche", prims, marco=tarjeta_en_ranura("hijo", 0), sombras=sombras)

    def nino_cepillandose(self, t, edad, pos, cara="abierta", amp=9.0, f=2.6, agarre=84.0, extra=None, escala=1.0, evitar=None, eje=EJE_SOLO):
        """El niño se cepilla solo con la mano derecha (pantalla izquierda; el mango sale hacia ese lado, la mano lejos de la mejilla)."""
        quien = hijo(edad)
        base = {"cara": cara}
        base.update(extra or {})
        ib = self.info(t, quien, base, pos, escala=escala, desfase=1.7)
        cep, spec = self.cepillo_en_boca(t, ib, eje, U.AZUL, amp=amp, f=f, agarre=agarre)
        self.sostenido(cep, "hijo", "izq")
        spec["polo"] = (-0.55, 1.0, 0.3)
        pose = dict(base)
        pose["mano_izq"] = spec
        pose.setdefault("mano_der", self.mano_en_borde(pos[0] + 100, "der"))
        pb, ib = self.persona(t, quien, pose, pos, escala=escala, desfase=1.7, evitar=[] if evitar is None else evitar)
        return pb + cep, ib

    # 9 · El niño (7) agarra el cepillo («yo solito»): ella lo suelta y da pasos atrás hasta la puerta
    def escena_9(self, t, e):
        lt = t - e["inicio"]
        pos_n = NINO7
        detras = (562.0, YM, -150.0)
        if lt < 1.3:  # mamá cepilla desde atrás (la otra mano en su hombro); él sube la mano y agarra el mango entre la boca y la mano de ella
            cara_n = "neutral" if lt < 0.9 else "abierta"
            pose_n = {"cara": cara_n}
            ib = self.info(t, hijo(7), pose_n, pos_n, desfase=1.7)
            para = suave(progreso(lt, 0.45, 0.75))
            cep, mano_m = self.cepillo_en_boca(t, ib, (-0.80, 0.22, 0.56), U.AZUL, amp=7.0 * (1 - para), f=2.7, agarre=112.0)
            self.sostenido(cep, "mama", "izq")
            e_m = -np.asarray(mano_m["eje"], np.float32)
            cab = np.asarray(mano_m["agarre"], np.float32) - e_m * 112.0
            ag_n = cab + e_m * 64.0
            sube = suave(progreso(lt, 0.42, 0.88))
            mano_n = self.mano_en_borde(pos_n[0] - 100, "izq")
            if sube > 0.0:
                agarra = {"agarre": tuple(ag_n), "eje": tuple(-e_m), "mezcla": ("relajada", "agarre", progreso(lt, 0.7, 0.88)), "polo": (-0.9, 0.7, 0.1)}  # el codo abierto: el antebrazo no roza su mejilla
                mano_n = self.transicion(t, hijo(7), pose_n, pos_n, "izq", mano_n, agarra, sube, desfase=1.7, arco=(-20.0, -50.0, 10.0))
                if lt > 0.7:
                    self.sostenido(cep, "hijo", "izq")
            pose_nb = dict(pose_n, mano_izq=mano_n, mano_der=self.mano_en_borde(pos_n[0] + 100, "der"))
            hombro = self.mano_en_hombro(ib, "der")
            hombro["palma"] = tuple(np.asarray(hombro["palma"], np.float32) + np.array([0.0, -8.0, 4.0], np.float32))
            hombro["polo"] = (0.8, 0.6, -0.5)
            suelta = suave(progreso(lt, 0.84, 1.02))
            mano_m["mezcla"] = ("agarre", "relajada", suelta)
            mano_m["polo"] = (-0.9, 0.5, -0.6)
            if suelta > 0:  # abre la mano y la aparta del mango, hacia atrás
                mano_m["agarre"] = tuple(np.asarray(mano_m["agarre"], np.float32) + np.array([-50.0, -34.0, 22.0], np.float32) * suelta)  # sale por el extremo del mango, lejos de la mano del niño
            if lt > 0.95:
                U.permitir(cep, ("mama", "mano_izq"))
            if lt < 0.95:
                mano_der_m = hombro
            else:  # la levanta de su hombro hacia atrás y luego la baja a su costado
                alza = dict(hombro, palma=tuple(np.asarray(hombro["palma"], np.float32) + np.array([50.0, -70.0, -110.0], np.float32)))
                mano_der_m = self.transicion(t, mama(26), {}, detras, "der", hombro, alza, suave(progreso(lt, 0.95, 1.12)))
                if lt > 1.08:
                    mano_der_m = self.transicion(t, mama(26), {}, detras, "der", alza, None, suave(progreso(lt, 1.08, 1.3)))
            pose_m = {"cara": "sonrisa", "mano_izq": mano_m, "mano_der": mano_der_m,
                      "mirar": tuple(ib["boca"]), "inclinacion": 0.08}
            pb, _ = self.persona(t, hijo(7), pose_nb, pos_n, desfase=1.7, evitar=[])
            pm, _ = self.persona(t, mama(26), pose_m, detras, evitar=pb)
            pb, _ = self.persona(t, hijo(7), pose_nb, pos_n, desfase=1.7, evitar=pm)
            if 0.68 < lt < 1.04:  # en el relevo las dos manos se tocan sobre el mango
                U.permitir([q for q in pb if getattr(q, "grupo", "") == "mano_izq"], ("mama", "mano_izq"))
            prims = pb + pm + cep
        else:
            k_eje = suave(progreso(lt, 1.3, 1.7))  # el cepillo pasa del ángulo de mamá al suyo sin saltos
            eje_n = tuple(N(np.array((-0.80, 0.22, 0.56), np.float32) * (1 - k_eje) + np.array(EJE_SOLO, np.float32) * k_eje))
            prims, ib = self.nino_cepillandose(t, 7, pos_n, amp=8.0 * suave(progreso(lt, 1.3, 1.7)), f=2.5, agarre=64.0 + 20.0 * k_eje, eje=eje_n)
            if lt < 1.65:  # la mano de mamá, ya abierta, vuelve a su costado por detrás
                ib1 = self.info(t, hijo(7), {"cara": "abierta"}, pos_n, desfase=1.7)
                _, spec1 = self.cepillo_en_boca(t, ib1, (-0.80, 0.22, 0.56), U.AZUL, amp=0.0, agarre=112.0)
                suelta_spec = {"agarre": tuple(np.asarray(spec1["agarre"], np.float32) + np.array([-50.0, -34.0, 22.0], np.float32)), "eje": spec1["eje"], "forma": "relajada", "polo": (-0.9, 0.5, -0.6)}
                mano = self.transicion(t, mama(26), {}, detras, "izq", suelta_spec, None, suave(progreso(lt, 1.3, 1.65)), arco=(-20.0, -60.0, -110.0))
                pm, _ = self.persona(t, mama(26), {"cara": "sonrisa", "mano_izq": mano, "mirar": tuple(ib["cabeza"])}, detras, evitar=prims)
            else:
                pm, _ = self.mama_caminando(t, 26, 1.6, 3.7, (detras[0], detras[2]), PUERTA_XZ, {"cara": "sonrisa", "mirar": tuple(ib["cabeza"])}, giro_cuerpo=-0.1)
            prims = prims + pm
        return self.componer(t, "noche", prims + self.encimera_objetos(t, ("amarillo", None)), marco=tarjeta_en_ranura("hijo", 0), sombras=[(pos_n[0] + 72, 62, 36, 20, 0.4)])

    # 10 · Mamá lo mira desde la puerta; levanta la mano para ayudarle y la baja (acercamiento lento)
    def escena_10(self, t, e):
        lt = t - e["inicio"]
        prims, ib = self.nino_cepillandose(t, 7, NINO7, amp=9.0 + 3 * math.sin(lt * 1.3), f=2.3)
        x, z = PUERTA_XZ
        sube = suave(progreso(lt, 1.0, 1.6)) * (1 - suave(progreso(lt, 2.4, 3.0)))
        im0 = self.info(t, mama(26), {}, (x, YM, z), vaiven=False, desfase=0.4)
        hxs, hys, hzs = im0["hombros"]["izq"]
        R = im0["R"]
        abajo = np.array([hxs - 0.15 * R, hys + 4.0 * R, hzs + 0.3 * R], np.float32)
        arriba = np.array([hxs - 1.3 * R, hys + 1.9 * R, hzs + 2.4 * R], np.float32)
        mano = None
        if sube > 0.01:
            p = abajo * (1 - sube) + arriba * sube
            mano = {"palma": tuple(p), "dir": (-0.45, 0.15, 0.9), "nrm": (0.2, 0.8, 0.4), "mezcla": ("relajada", "abierta", sube)}
        cara = "sonrisa" if lt < 2.5 else "neutral"
        pm, _ = self.mama_puerta(t, 26, {"cara": cara, "mano_izq": mano, "inclinacion": 0.06 * sube, "mirar": tuple(ib["cabeza"])})
        return self.componer(t, "noche", prims + pm + self.encimera_objetos(t, ("amarillo", None)), marco=tarjeta_en_ranura("hijo", 0), sombras=[(NINO7[0] + 72, 62, 36, 20, 0.4)])

    # 11 · Solo cambia el contador
    def escena_11(self, t, e):
        prims, _ = self.nino_cepillandose(t, 7, NINO7, f=2.4)
        return self.componer(t, "noche", prims + self.encimera_objetos(t, ("amarillo", None)), marco=tarjeta_en_ranura("hijo", 0), sombras=[(NINO7[0] + 72, 62, 36, 20, 0.4)])

    # 12 · Adolescente (15) se cepilla con desgana mirando el celular; llega la notificación
    def escena_12(self, t, e):
        lt = t - e["inicio"]
        quien = hijo(15)
        tel_c = np.array([TEEN[0] + 74, TEEN[1] + 196 + 3 * math.sin(lt * 1.3), 176.0], np.float32)  # a la altura del pecho, delante, como se lee el celular
        ib0 = self.info(t, quien, {}, TEEN)
        hacia_ojos = N(np.asarray(ib0["cabeza"], np.float32) - tel_c)
        arriba = np.array([0.0, -1.0, 0.0], np.float32)
        rot = J.base_desde(-(arriba - hacia_ojos * float(arriba @ hacia_ojos)), -hacia_ojos)  # la pantalla mira a sus ojos
        lee = suave(progreso(lt, 1.7, 1.9)) * (1 - suave(progreso(lt, 2.45, 2.75)))
        pose = {"cara": "abierta", "mirar": tuple(tel_c)}
        ib = self.info(t, quien, pose, TEEN)
        cep, mano_izq = self.cepillo_en_boca(t, ib, EJE_SOLO, U.AZUL, amp=6.0 * (1 - 0.9 * lee), f=2.0, agarre=94.0)
        self.sostenido(cep, "hijo", "izq")
        mano_izq["polo"] = (-0.5, 1.0, 0.3)
        pulgar = pulso(lt, [2.05, 2.28], 0.05)
        # la mano sostiene el celular por detrás (la palma contra la tapa), dedos alrededor del borde
        mano_der = {"palma": tuple(tel_c + rot @ np.array([4, 22, 14], np.float32)), "dir": tuple(rot @ np.array([-0.15, -1.0, 0.0], np.float32)),
                    "nrm": tuple(rot @ np.array([0, 0, -1.0], np.float32)), "mezcla": ("relajada", "agarre", 0.45 + 0.15 * pulgar), "polo": (0.7, 1.0, 0.1)}
        pose.update({"mano_der": mano_der, "mano_izq": mano_izq})
        tel = U.telefono(tel_c, rot)
        pt, _ = self.persona(t, quien, pose, TEEN, evitar=[])
        self.sostenido(tel, "hijo", "der")
        prims = pt + cep + tel + self.encimera_objetos(t, ("amarillo", None))
        return self.componer(t, "manana", prims, marco=tarjeta_en_ranura("hijo", 0))

    # 13 · De noche, mamá (45) se toca la mejilla con dolor; toma su tarjeta, tacha la fecha y escribe otra
    def escena_13(self, t, e):
        lt = t - e["inicio"]
        quien = mama(45)
        pos_m = (MAMA_SOLA[0], MAMA_SOLA[1], -42.0)  # sin pegarse al canto de la encimera
        im0 = self.info(t, quien, {"giro": 0.15}, pos_m)
        hx, hy, hz = im0["cabeza"]
        R = im0["R"]
        datos = GE.TARJETA_MAMA
        tach = tuple(datos["tachadas"])
        k_tachar = progreso(lt, 2.2, 2.55)
        k_nueva = progreso(lt, 2.7, 3.65)
        tex = tex_tarjeta(datos["titulo"], (), tach, datos["nueva"], k_tachar=round(k_tachar * 20) / 20, k_nueva=round(k_nueva * 40) / 40)
        spr = sprite_tarjeta(datos["titulo"], (), tach, datos["nueva"])
        w, h = spr.width / 2, spr.height / 2
        _, aw, ah, _ = ranura(1, w, h)
        aw, ah = aw * 1.12, ah * 1.12
        en_encimera = np.array([640.0, S.ENCIMERA_Y - 1.4, 86.0], np.float32)  # el borde de su lado sobresale un poco del canto
        en_mano = np.array([578.0, 1036.0, 150.0], np.float32)  # delante del pecho: se ve y los brazos quedan libres
        sube = suave(progreso(lt, 1.55, 2.0))
        c = np.array(curva(lt, [(1.55, tuple(en_encimera)), (1.78, tuple(en_encimera * 0.5 + en_mano * 0.5 + np.array([0, -40, 0], np.float32))), (2.0, tuple(en_mano))]), np.float32) if lt < 2.0 else en_mano
        # tumbada (texto hacia ella), casi plana, un poco inclinada hacia su cara
        Rc = J.rot_y(math.pi) @ J.rot_x(math.radians(90) - math.radians(18) * sube)
        prims = U.tarjeta(c, aw, ah, tex, Rc)
        normal = Rc @ np.array([0, 0, 1.0], np.float32)

        def en_tarjeta(px, py, alto=0.0):
            return c + Rc @ np.array([(px / w - 0.5) * aw, (py / h - 0.5) * ah, 1.3 + alto], np.float32)

        # mano derecha (pantalla): la palma plana en la mejilla → toma la tarjeta por su borde y la sostiene
        giro_c = 0.15 if lt < 1.4 else 0.0
        n_mej = N(np.array([0.74, 0.42, 0.54], np.float32))  # mejilla y mandíbula (bajo el pelo)
        n_mej = np.array([n_mej[0] * math.cos(giro_c) + n_mej[2] * math.sin(giro_c), n_mej[1], -n_mej[0] * math.sin(giro_c) + n_mej[2] * math.cos(giro_c)], np.float32)
        mejilla = {"palma": tuple(np.array([hx, hy, hz], np.float32) + n_mej * np.array([R, R * 1.02, R * 0.96], np.float32) + n_mej * 21.0),
                   "dir": (-0.15, -0.55, 0.82), "nrm": tuple(-n_mej), "mezcla": ("abierta", "relajada", 0.12), "polo": (0.25, 1.0, 0.25)}  # el codo abajo, junto al cuerpo
        borde = {"pinza": tuple(en_tarjeta(w * 0.1, h - 3, 0.0)), "dir": (-0.35, 0.1, 0.93), "nrm": (0.05, 0.85, -0.5), "mezcla": ("relajada", "pinza", 1.0), "polo": (0.7, 1.0, -0.2)}  # el borde de su lado (sobresale del canto)
        if lt < 1.15:
            mano_der = self.transicion(t, quien, {}, pos_m, "der", None, mejilla, suave(progreso(lt, 0.0, 0.32)), arco=(40.0, -30.0, 34.0))
        elif lt < 1.55:  # deja la mejilla y va al borde de la tarjeta (los dedos la pellizcan)
            borde_ida = dict(borde, mezcla=("relajada", "pinza", progreso(lt, 1.42, 1.55)))
            mano_der = self.transicion(t, quien, {}, pos_m, "der", mejilla, borde_ida, suave(progreso(lt, 1.15, 1.5)), arco=(64.0, -20.0, 6.0))
            if lt > 1.42:
                self.sostenido(prims, "mama", "der")
        else:
            mano_der = borde
            self.sostenido(prims, "mama", "der")
        # mano izquierda (pantalla) con el bolígrafo: la punta toca la tarjeta sin atravesarla
        f_t = fuente("manuscrita", 40)
        y_ult = 76 + 46 * (len(tach) - 1)
        w_ult = f_t.getlength(tach[-1])
        y_nueva = 76 + 46 * len(tach)
        w_nueva = f_t.getlength(datos["nueva"])
        roce = 3.2  # radio de la punta: apoya sobre el papel
        if lt < 2.15:
            reposo = np.array([hx - 1.6 * R, S.ENCIMERA_Y - 32, 70], np.float32)
            punta = np.array(curva(lt, [(0.0, tuple(reposo)), (1.7, tuple(reposo + np.array([10, -20, 20], np.float32))), (2.15, tuple(en_tarjeta(18, y_ult + 23, 8)))]), np.float32)
        elif lt < 2.55:
            k = progreso(lt, 2.2, 2.55)
            punta = en_tarjeta(18 + (w_ult + 8) * k, y_ult + 26 - 6 * k, roce)
        elif lt < 2.7:
            k = suave(progreso(lt, 2.55, 2.7))
            punta = en_tarjeta(26 + w_ult - (w_ult + 4) * k, y_ult + 22 + (y_nueva - y_ult) * k, roce + 10 * math.sin(math.pi * k))
        elif lt < 3.65:
            k = progreso(lt, 2.7, 3.65)
            punta = en_tarjeta(22 + w_nueva * k, y_nueva + 20 + 7 * math.sin(lt * 2 * math.pi * 6.5), roce)
        else:
            punta = en_tarjeta(22 + w_nueva, y_nueva + 18, roce + 22 * suave(progreso(lt, 3.65, 3.95)))
        eje_l = N(normal * 0.75 + np.array([-0.45, -0.3, -0.35], np.float32))  # el lápiz sale hacia arriba, nunca por debajo del papel
        lap = U.lapiz(punta, eje_l, color=(150, 60, 60))
        self.sostenido(lap, "mama", "izq")
        U.permitir(lap, ("tarjeta", "objeto"))
        U.permitir(prims, ("lapiz", "objeto"))
        mano_izq = {"pinza": tuple(punta + eje_l * 30), "dir": (0.25, 0.6, 0.76), "nrm": (0.72, 0.1, -0.2), "mezcla": ("lapiz", "lapiz", 1.0), "polo": (-0.35, 0.7, -0.6)}
        cara = "dolor" if lt < 1.4 else ("concentrada" if lt < 3.7 else "neutral")
        pm, _ = self.persona(t, quien, {"cara": cara, "giro": giro_c, "mano_izq": mano_izq, "mano_der": mano_der, "mirar": None if lt < 1.4 else tuple(c), "inclinacion": 0.06}, pos_m, evitar=prims + lap)
        prims += lap + pm + self.encimera_objetos(t, ("amarillo", None))
        return self.componer(t, "noche", prims, marco=tarjeta_en_ranura("hijo", 0), sombras=[(640, 86, 70, 50, 0.35 * (1 - sube))])

    # 14 · «25 AÑOS DESPUÉS» (el baño vacío; la tarjeta vieja de mamá en el marco)
    def escena_14(self, t, e):
        return self.componer(t, "manana", self.encimera_objetos(t, ("amarillo", "rosa")), marco=tarjeta_en_ranura("mama_vieja", 1))

    # 15 · Tú (32), detrás de tu hija (3), le cepillas los dientes y cuentas con los dedos; mamá (58) mira desde la puerta
    def escena_15(self, t, e):
        lt = t - e["inicio"]
        inf = self.info(t, nieta(), {"cara": "abierta"}, NIETA, desfase=0.6)
        pn, _ = self.persona(t, nieta(), {"cara": "abierta", "mano_izq": self.mano_en_borde(NIETA[0] - 92, "izq"), "mano_der": self.mano_en_borde(NIETA[0] + 92, "der")}, NIETA, desfase=0.6)
        cuentas = [0.6, 1.2, 1.8, 2.4, 3.0]
        ih0 = self.info(t, hijo(32), {}, HIJO_DETRAS, desfase=2.2)
        sube = suave(progreso(lt, 0.0, 0.45))
        mano_der = self.contar(ih0, "der", escalon(lt, cuentas), [pulso(lt, [c + 0.06], 0.07) for c in cuentas])
        if sube < 0.999:  # la mano sube por fuera del brazo de la niña hasta junto a su cara
            mano_der = self.transicion(t, hijo(32), {}, HIJO_DETRAS, "der", None, mano_der, sube, desfase=2.2, arco=(70.0, -40.0, -30.0))
        ph, _ = self.cepillar_desde_atras(t, hijo(32), HIJO_DETRAS, inf, U.ROSA, amp=6.5, f=2.6, pasta=0.4, mano_der=mano_der, extra={"mirar": None, "evitar": J.obstaculo(pn + utileria_obstaculos(), extra=pen_encimera)}, desfase=2.2)
        pm, _ = self.mama_puerta(t, 58, {"cara": "sonrisa", "mirar": tuple(inf["cabeza"])})
        prims = ph + pn + pm + self.encimera_objetos(t, ("amarillo", None))
        return self.componer(t, "noche", prims, marco=tarjeta_en_ranura("mama_vieja", 1), sombras=[(NIETA[0] - 62, 60, 34, 20, 0.35), (NIETA[0] + 62, 60, 34, 20, 0.35)])

    def lata_en_manos(self, lt):
        """
        Cajita que el hijo saca del cajón de arriba (delante del mueble, bajo el canto): se agacha, la saca,
        se endereza, la sube a la altura del vientre, da un paso al frente, la abre y la inclina hacia el espejo.
        """
        cajon = np.array([HIJO_SOLO[0] + 22, S.ENCIMERA_Y + 84, -12.0], np.float32)
        vientre = np.array([HIJO_SOLO[0] + 22, S.ENCIMERA_Y - 92, -12.0], np.float32)
        delante = np.array([HIJO_SOLO[0] + 25, 1085.0, 162.0], np.float32)
        if lt < 1.32:
            k = suave(progreso(lt, 1.02, 1.32))
            c = cajon * (1 - k) + vientre * k
        else:
            k = suave(progreso(lt, 1.32, 1.64))
            c = vientre * (1 - k) + delante * k + np.array([0, -18, 0], np.float32) * math.sin(math.pi * k)
        inclina = suave(progreso(lt, 1.92, 2.32))
        rot = J.rot_x(-0.72 * inclina)
        abierta = suave(progreso(lt, 1.64, 2.0))
        atras = suave(progreso(lt, 0.0, 0.3)) * (1 - suave(progreso(lt, 1.32, 1.64)))  # da un paso atrás para abrir el cajón
        return c, rot, abierta, atras

    def manos_lata(self, c, rot):
        hx_ = U.LATA_H[0]
        izq = c + rot @ np.array([-hx_ - 3, 6, 0], np.float32)
        der = c + rot @ np.array([hx_ + 3, 6, 0], np.float32)
        return ({"palma": tuple(izq), "dir": tuple(rot @ np.array([0.2, 0.3, 1.0], np.float32)), "nrm": tuple(rot @ np.array([1.0, 0, 0], np.float32)), "forma": "relajada"},
                {"palma": tuple(der), "dir": tuple(rot @ np.array([-0.2, 0.3, 1.0], np.float32)), "nrm": tuple(rot @ np.array([-1.0, 0, 0], np.float32)), "forma": "relajada"})

    # 16 · Abres el cajón buscando hilo dental y encuentras la cajita: tus 20 dientes de leche (acercamiento)
    def escena_16(self, t, e):
        lt = t - e["inicio"]
        c, rot, abierta, atras = self.lata_en_manos(lt)
        pos = (HIJO_SOLO[0], HIJO_SOLO[1], HIJO_SOLO[2] - 115 * atras)
        prims_lata, borde_tapa = U.lata(abierta, centro=c, rot=rot)
        prims = list(prims_lata)
        self.sostenido(prims_lata, "hijo", "izq")
        self.sostenido(prims_lata, "hijo", "der")
        if lt < 0.85:  # las manos buscan en el cajón abierto (por debajo de la encimera, delante de él)
            hurga = 8 * math.sin(lt * 9)
            mano_izq = {"centro": (HIJO_SOLO[0] - 62, S.ENCIMERA_Y + 84 + hurga, -22), "dir": (0.1, 0.95, 0.3), "forma": "relajada", "polo": (-0.8, 0.3, -0.5)}
            mano_der = {"centro": (HIJO_SOLO[0] + 88, S.ENCIMERA_Y + 80 - hurga, -22), "dir": (-0.1, 0.95, 0.3), "forma": "relajada", "polo": (0.8, 0.3, -0.5)}
            if lt < 0.46:  # tras el paso atrás se inclina y las manos bajan al cajón
                k_c = suave(progreso(lt, 0.16, 0.46))
                pose_c = {"inclinacion": 0.08 + 0.42 * suave(progreso(lt, 0.18, 0.45))}
                mano_izq = self.transicion(t, hijo(32), pose_c, pos, "izq", None, mano_izq, k_c, desfase=2.2)
                mano_der = self.transicion(t, hijo(32), pose_c, pos, "der", None, mano_der, k_c, desfase=2.2)
            prims = []
        else:
            mano_izq, mano_der = self.manos_lata(c, rot)
            if 1.56 < lt < 2.02:  # levanta la tapa con las yemas (la suelta a medio camino: cae abierta) y vuelve a sujetar la cajita
                yemas = {"yemas": tuple(np.asarray(borde_tapa, np.float32) + np.array([0, -4, 6], np.float32)), "dir": (0.0, 0.55, -0.83), "nrm": (0, 0.6, 0.8), "forma": "abierta", "polo": (-0.8, 0.6, -0.2)}
                if lt < 1.82:
                    mano_izq = self.transicion(t, hijo(32), {}, pos, "izq", mano_izq, yemas, suave(progreso(lt, 1.56, 1.66)), desfase=2.2)
                else:
                    mano_izq = self.transicion(t, hijo(32), {}, pos, "izq", yemas, mano_izq, suave(progreso(lt, 1.82, 2.0)), desfase=2.2)
        cara = "neutral" if lt < 2.05 else ("sorpresa" if lt < 3.2 else "sonrisa")
        mira = tuple(c) if lt > 0.85 else (HIJO_SOLO[0], S.ENCIMERA_Y + 150, -20)
        agacha = suave(progreso(lt, 0.18, 0.45)) * (1 - suave(progreso(lt, 0.85, 1.05)))  # se inclina para buscar en el cajón de arriba (y se endereza antes de subirla)
        ph, _ = self.persona(t, hijo(32), {"cara": cara, "mirar": mira, "mano_izq": mano_izq, "mano_der": mano_der, "inclinacion": 0.08 + 0.42 * agacha}, pos, desfase=2.2, evitar=prims)
        pm, _ = self.mama_puerta(t, 58, {"cara": "neutral" if lt < 2.2 else "sonrisa", "mirar": tuple(c) if lt > 1.0 else None})
        prims += ph + pm + self.encimera_objetos(t, ("amarillo", None))
        return self.componer(t, "noche", prims, marco=tarjeta_en_ranura("mama_vieja", 1))

    # 17 · Mamá se ríe desde la puerta tapándose la boca; tú la miras por el espejo con la cajita abierta
    def escena_17(self, t, e):
        lt = t - e["inicio"]
        c, rot, _, _ = self.lata_en_manos(9.0)
        prims = list(U.lata(1.0, centro=c, rot=rot)[0])
        self.sostenido(prims, "hijo", "izq")
        self.sostenido(prims, "hijo", "der")
        mano_izq, mano_der = self.manos_lata(c, rot)
        ph, _ = self.persona(t, hijo(32), {"cara": "sonrisa", "inclinacion": 0.08, "mirar": tuple(c) if lt < 0.8 else None, "mano_izq": mano_izq, "mano_der": mano_der}, HIJO_SOLO, desfase=2.2)
        x, z = PUERTA_XZ
        risa = suave(progreso(lt, 0.0, 0.4)) * (1 - suave(progreso(lt, 3.4, 3.9)))
        sacude = 3.0 * math.sin(lt * 2 * math.pi * 4.5) * risa
        im0 = self.info(t, mama(58), {"cara": "tapada"}, (x, YM + sacude, z), desfase=0.4, vaiven=False)
        hx, hy, hz = im0["cabeza"]
        R = im0["R"]
        boca = np.array([hx, hy + 0.38 * R, hz + 1.0 * R], np.float32)
        abajo = np.array([hx - 1.3 * R, hy + 4.4 * R, hz + 0.2 * R], np.float32)
        mano = {"palma": tuple(abajo * (1 - risa) + boca * risa), "dir": (0.25, -0.9, 0.3), "nrm": (0.0, 0.1, -1.0), "mezcla": ("relajada", "abierta", risa)} if risa > 0.02 else None
        pm, _ = self.persona(t, mama(58), {"cara": "tapada" if risa > 0.3 else "risa", "mano_izq": mano, "piernas": "de_pie"}, (x, YM + sacude, z), desfase=0.4, vaiven=False)
        prims += ph + pm + self.encimera_objetos(t, ("amarillo", None))
        return self.componer(t, "noche", prims, marco=tarjeta_en_ranura("mama_vieja", 1))

    # 18 · Metes en el marco una tarjeta nueva; ella se acerca, la lee y sonríe enseñando los dientes por primera vez
    def escena_18(self, t, e):
        lt = t - e["inicio"]
        lata_c = np.array([298.0, S.ENCIMERA_Y - U.LATA_H[1], 200.0], np.float32)  # detrás del tubo de pasta, sin tocarlo  # la cajita abierta, a un lado (lejos de la mano que se apoya)
        base_h = (426.0, HIJO_SOLO[1], HIJO_SOLO[2])  # un poco a la izquierda: su brazo sube por el lado de la cara, sin taparla
        prims = list(U.lata(1.0, centro=lata_c)[0])
        pinza_ini = (base_h[0] + 108, base_h[1] + 330 + 3 * math.sin(lt * 2), base_h[2] + 160)
        prims_t, mano, incl, mirar, marco_n, f_sombra = self.meter_en_marco(lt, "nueva", 2, 0.2, 0.4, 1.3, 1.5, pinza_ini, "hijo", "der", esquina="izq", inclinacion=0.5, polo=(0.3, 0.9, -0.35), reposo=pinza_ini)
        if mano is None:
            mano = {"pinza": pinza_ini, "dir": (0.3, -0.85, 0.4), "nrm": (0.2, 0.3, -1.0), "forma": "relajada", "polo": (0.3, 0.9, -0.35)}
        puntas = 30.0 * incl / 0.5  # de puntitas para llegar al marco
        pos_h = (base_h[0], base_h[1] - puntas, base_h[2])
        ph, _ = self.persona(t, hijo(32), {"cara": "sonrisa", "inclinacion": incl, "mano_der": mano, "mano_izq": self.mano_en_encimera(base_h[0] - 50, "izq", 98), "mirar": mirar}, pos_h, desfase=2.2, evitar=prims + list(prims_t or []))
        cara = "neutral" if lt < 2.95 else "dientes"
        mira_tarjeta = tuple(float(v) for v in ranura(2, *medidas_tarjeta("nueva"))[0])
        if 1.6 <= lt < 2.95:
            pm, _ = self.mama_caminando(t, 58, 1.6, 2.7, PUERTA_XZ, (1010.0, -470.0), {"cara": cara, "mirar": mira_tarjeta}, giro_cuerpo=-0.15)
        else:
            xz = PUERTA_XZ if lt < 1.6 else (1010.0, -470.0)
            pm, _ = self.persona(t, mama(58), {"cara": cara, "piernas": "de_pie", "mirar": mira_tarjeta if lt > 1.2 else None}, (xz[0], YM, xz[1]), desfase=0.4, vaiven=False)
        prims += ph + pm + (prims_t or []) + self.encimera_objetos(t, ("amarillo", None))
        marco = list(tarjeta_en_ranura("mama_vieja", 1)) + list(marco_n or [])
        return self.componer(t, "noche", prims, marco=marco, marco_sombra=1.0, sombras=[(298, 200, 62, 42, 0.55), (base_h[0] - 50, 98, 36, 20, 0.4)])

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
        return img
