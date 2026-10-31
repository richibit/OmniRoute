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
    return PJ.Personaje("adulta", SUETER, (92, 74, 64), pelo, "mono", arrugas=edad >= 55)


@functools.lru_cache(maxsize=None)
def hijo(edad, pegatina=False):
    if edad < 1:
        return PJ.Personaje("bebe", (176, 208, 238), (176, 208, 238), (70, 48, 34), "bebe")
    if edad < 3:
        return PJ.Personaje("pequeno", (128, 174, 230), (128, 174, 230), (64, 42, 30), "corto")
    if edad < 6.5:
        return PJ.Personaje("nino4", (86, 128, 212), (86, 128, 212), (60, 40, 28), "corto", pegatina=(250, 200, 70) if pegatina else None)
    if edad < 10:
        return PJ.Personaje("nino7", (86, 128, 212), (86, 128, 212), (60, 40, 28), "corto")
    if edad < 25:
        return PJ.Personaje("adolescente", (150, 153, 160), (64, 70, 92), (54, 38, 30), "capucha")
    return PJ.Personaje("adulto", (40, 58, 98), (62, 62, 70), (54, 38, 30), "corto")


@functools.lru_cache(maxsize=None)
def nieta():
    return PJ.Personaje("pequeno", (240, 142, 152), (240, 142, 152), (72, 46, 34), "coletas")


YM = y_cabeza("adulta")
# El niño (o quien se cepilla) al centro, frente al lavabo; mamá a su derecha (del lado de la puerta,
# el vaso, el grifo y el marco), así nunca tiene que cruzar por detrás de él.
MAMA = (800.0, YM, -5.0)
MAMA_SOLA = (545.0, YM, -5.0)
NINO4 = (548.0, y_cabeza("nino4", 215), 22.0)
PEQ = (548.0, y_cabeza("pequeno", 330), 22.0)
NINO7 = (548.0, y_cabeza("nino7", 150), 22.0)
TEEN = (548.0, y_cabeza("adolescente"), -5.0)
HIJO32 = (485.0, y_cabeza("adulto"), -5.0)
HIJO_SOLO = (505.0, y_cabeza("adulto"), -5.0)
NIETA = (712.0, y_cabeza("pequeno", 340), 22.0)
BEBE = (548.0, S.ENCIMERA_Y - PJ.TIPOS["bebe"]["cadera_y"] * PJ.TIPOS["bebe"]["R"] - 6, 108.0)
MAMA_BEBE = (805.0, YM, -25.0)
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


NOTA_LINEAS = (("Del Ratón", 0.10, 0.36), ("de los Dientes", 0.10, 0.74))  # texto, u inicial, v de la línea
NOTA_PX = (360, 240)


@functools.lru_cache(maxsize=64)
def _metricas_nota():
    f = fuente("manuscrita", 74)
    return [f.getlength(t) / NOTA_PX[0] for t, _, _ in NOTA_LINEAS]


@functools.lru_cache(maxsize=128)
def tex_nota(k):
    """Nota del Ratón de los Dientes con el texto escrito hasta `k` (0–1, por líneas)."""
    w, h = NOTA_PX
    img = Image.new("RGBA", (w, h), (255, 250, 222, 255))
    d = ImageDraw.Draw(img)
    f = fuente("manuscrita", 74)
    largos = _metricas_nota()
    total = sum(largos)
    hecho = k * total
    for (texto, u0, v), largo in zip(NOTA_LINEAS, largos):
        if hecho <= 0:
            break
        capa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(capa).text((u0 * w, v * h - 52), texto, font=f, fill=(80, 60, 120, 255))
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
RANURAS = {0: ((885.0, 480.0), -6.0), 1: ((205.0, 520.0), 4.0), 2: ((800.0, 365.0), -3.0)}  # la 1 no tapa la puerta
Z_MARCO = S.ESPEJO_Z - 3.0


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
    for x, z, rx, rz, f in ((vx + 6, vz - 6, 52, 40, 0.75), (275, 158, 80, 24, 0.55), (S.PALANCA[0], S.PALANCA[1], 26, 20, 0.5)):
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


@functools.lru_cache(maxsize=16)
def capa_utileria(nombre, vaso_cepillos, pasta, con_palanca):
    """Vaso con cepillos, pasta, caño (y mando del agua) renderizados una vez con el encuadre base, con su t por subpíxel."""
    prims = U.vaso()
    colores = {"amarillo": U.AMARILLO, "azul": U.AZUL, "rosa": U.ROSA}
    for k, c in enumerate(vaso_cepillos):
        if c:
            prims += U.cepillo_en_vaso(U.VASO, k, colores[c])
    if pasta:
        prims += U.tubo_pasta()[0]
    prims += S.grifo(0.0, partes="todo" if con_palanca else "caño")
    luz = J.NOCHE if nombre == "noche" else J.MANANA
    img, x0, y0, tm = J.render(prims, luz, S.CAM, oclusor=S.oclusor, devolver_t="completo")
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
        cam = self.cam
        luz = self.luz(luz_nombre, k_manana)
        fondo, enc = self.capas_base(luz_nombre, k_manana)
        vaso, pasta = getattr(self, "utileria", (("amarillo", "azul"), True))
        palanca_movil = self.palanca > 0.0
        nombre = luz_nombre if k_manana is None else ("manana" if k_manana > 0.5 else "noche")
        fija, fx0, fy0, ftm = capa_utileria(nombre, vaso, pasta, not palanca_movil)
        zoom2 = (self.z2, self.f2[0], self.f2[1]) if self.z2 > 1.0001 else None
        prims = list(prims)
        if palanca_movil:
            prims += S.grifo(self.palanca, partes="palanca")
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
    def persona(self, t, quien, pose, pos, escala=1.0, desfase=0.0, respira=True):
        """Personaje con respiración, parpadeo y un leve vaivén de cabeza; registra si alguna mano no llega."""
        x, y, z = pos
        if respira:
            y += respiracion(t, desfase)
        pose = dict(pose)
        pose["cara"] = parpadeo(t, pose.get("cara", "neutral"), desfase)
        pose["giro"] = pose.get("giro", 0.0) + 0.035 * math.sin(t * 0.9 + desfase * 2.1)
        prims, info = quien.construir(x, y, pose, escala=escala, z0=z, camara=S.CAM)
        for lado, err in info["error"].items():
            if err > 6 and pose.get(f"mano_{lado}") is not None:
                self.avisos.append((round(t, 2), quien.tipo, lado, round(err, 1)))
        return prims, info

    def info(self, t, quien, pose, pos, escala=1.0, desfase=0.0):
        """Solo la cabeza, la boca y los hombros (para preparar agarres antes de construir)."""
        x, y, z = pos
        pose = dict(pose, solo_hombros=True)
        pose["giro"] = pose.get("giro", 0.0) + 0.035 * math.sin(t * 0.9 + desfase * 2.1)
        return quien.construir(x, y + respiracion(t, desfase), pose, escala=escala, z0=z, camara=S.CAM)[1]

    @staticmethod
    def cepillo_en_boca(t, info, eje, color, amp=8.0, f=2.8, fase=0.0, pasta=0.0, hondo=7.0, desplaza=(0.0, 0.0, 0.0)):
        """
        Cepillo con la cabeza dentro de la boca de `info` y el mango hacia `eje`: vaivén corto a lo
        largo de los dientes. Devuelve (primitivas del cepillo, dict de agarre para la mano).
        """
        hx, hy, hz = info["cabeza"]
        m = np.asarray(info["boca"], np.float32)
        e = N(eje)
        horiz = N((e[0], 0.0, e[2] * 0.4))
        v = math.sin(2 * math.pi * f * t + fase)
        off = horiz * amp * v + np.array([0, 2.2 * math.sin(4 * math.pi * f * t + fase), 0], np.float32)
        hacia = N(np.array([hx, hy, hz], np.float32) - m)
        cab = m + hacia * hondo + off + np.asarray(desplaza, np.float32)
        prims, agarre = U.cepillo(cab, e, hacia, color, pasta)
        return prims, {"agarre": tuple(agarre), "eje": tuple(-e)}

    @staticmethod
    def contar(info, lado, n, pulsos=(), alto=1.15):
        """Mano levantada junto a la cara mostrando `n` dedos al espejo (con un pequeño golpe en cada número)."""
        hx, hy, hz = info["cabeza"]
        R = info["R"]
        sx = -1 if lado == "izq" else 1
        golpe = 9.0 * sum(pulsos)
        centro = (hx + sx * 1.02 * R, hy + alto * R + golpe, hz + 1.35 * R)
        return {"centro": centro, "forma": "contar", "dir": (sx * 0.12, -1.0, 0.12), "nrm": (-sx * 0.2, 0.05, 1.0), "dedos": PJ.cuenta_dedos(n), "polo": (sx * 0.5, 1.0, 0.35)}

    def mama_puerta(self, t, edad, pose, desfase=0.4):
        """Mamá de pie en la puerta, mirando al lavabo."""
        x, z = PUERTA_XZ
        pose = dict(pose)
        pose.setdefault("piernas", "de_pie")
        return self.persona(t, mama(edad), pose, (x, YM, z), desfase=desfase)

    def mama_caminando(self, t, edad, t0, t1, a, b, pose, giro_cuerpo=0.0):
        """Mamá caminando de a=(x, z) a b=(x, z) entre t0 y t1 (tiempos de la escena)."""
        w = caminata(self.t_escena, t0, t1, a, b, "adulta")
        pose = dict(pose)
        pose["piernas"] = {"pies": w["pies"]}
        pose["balanceo"] = w["balanceo"]
        pose.setdefault("giro_cuerpo", giro_cuerpo)
        y = cabeza_desde_cadera("adulta", w["y_cadera"])
        return self.persona(t, mama(edad), pose, (w["x"], y, w["z"]), respira=False)

    # utilería fija de la encimera (se renderiza una vez y tapa lo que pase por detrás)
    def encimera_objetos(self, t, vaso_cepillos=("amarillo", "azul"), pasta=True):
        self.utileria = (tuple(vaso_cepillos), pasta)
        return []

    # ================================================================ ESCENAS
    # ------------------------------------------------------------ apoyos
    @staticmethod
    def mano_en_hombro(info_otro, lado_otro="izq"):
        """Mano apoyada encima del hombro de otra persona."""
        hx, hy, hz = info_otro["hombros"][lado_otro]
        R = info_otro["R"]
        return {"palma": (hx + 6, hy - 0.42 * R, hz + 6), "dir": (0.75, 0.15, 0.65), "nrm": (0.0, 1.0, 0.1), "forma": "abierta"}

    def tarjeta_mano(self, nombre, pinza, rot, tex=None, polo=(0.0, 0.9, 0.45)):
        """Tarjeta sujeta en pinza por su esquina inferior izquierda (la mano y la tarjeta van juntas)."""
        w, h = medidas_tarjeta(nombre)
        _, aw, ah, _ = ranura(0, w, h)
        R = np.asarray(rot, np.float32)
        local = np.array([-aw / 2 + 13, ah / 2 - 15, -1.6], np.float32)
        centro = np.asarray(pinza, np.float32) - R @ local
        tex = tex if tex is not None else tex_tarjeta(*TARJETAS[nombre]["args"], **TARJETAS[nombre]["kw"])
        mano = {"pinza": tuple(pinza), "dir": tuple(R @ np.array([0.5, -0.86, 0.1], np.float32)), "nrm": tuple(R @ np.array([0.1, 0.0, -1.0], np.float32)), "polo": polo}
        return U.tarjeta(centro, aw, ah, tex, R), mano

    def meter_en_marco(self, lt, nombre, indice, t_sube, t_llega, t_suelta, pinza_ini, rot_ini=(-0.35, 0.08), via=None, reposo=None, inclinacion=0.5, polo=(0.0, 0.9, 0.45)):
        """
        Lleva una tarjeta de la mano a su ranura del marco (subiendo por `via`, un punto intermedio que
        evita las caras), la suelta y la mano vuelve a `reposo` por el mismo camino. Devuelve (prims de la
        tarjeta en mano o None, spec de la mano o None, inclinación del cuerpo, tarjeta del marco o None,
        fuerza de su sombra).
        """
        w, h = medidas_tarjeta(nombre)
        c_r, aw, ah, R_r = ranura(indice, w, h)
        local = np.array([-aw / 2 + 13, ah / 2 - 15, -1.6], np.float32)
        pinza_r = c_r + R_r @ local
        ang_r = math.atan2(float(R_r[1, 0]), float(R_r[0, 0]))
        p_ini = np.asarray(pinza_ini, np.float32)
        p_via = np.asarray(via, np.float32) if via is not None else p_ini * 0.45 + pinza_r * 0.55 + np.array([0, -40, -30], np.float32)
        p_rep = np.asarray(reposo, np.float32) if reposo is not None else p_ini + np.array([0, 120, -40], np.float32)
        k = suave(progreso(lt, t_sube, t_llega))
        incl = inclinacion * suave(progreso(lt, t_sube + 0.15, t_llega - 0.1)) * (1 - suave(progreso(lt, t_suelta + 0.1, t_suelta + 0.75)))
        if lt < t_suelta:
            p = np.array(curva(lt, [(t_sube, tuple(p_ini)), (t_sube + (t_llega - t_sube) * 0.45, tuple(p_via)), (t_llega, tuple(pinza_r)), (t_suelta, tuple(pinza_r + np.array([0, -3, 0], np.float32)))]), np.float32)
            ax = rot_ini[0] * (1 - k)
            az = rot_ini[1] * (1 - k) + ang_r * k
            R = J.rot_z(az) @ J.rot_x(ax)
            prims, mano = self.tarjeta_mano(nombre, p, R, polo=polo)
            return prims, mano, incl, None, 0.0
        marco = tarjeta_en_ranura(nombre, indice)
        mano = None
        t_fin = t_suelta + 0.85
        if lt < t_fin:  # suelta y la mano vuelve por el mismo camino, sin cruzar caras
            p = np.array(curva(lt, [(t_suelta, tuple(pinza_r + np.array([0, -3, 0], np.float32))), (t_suelta + 0.12, tuple(pinza_r + np.array([0, 25, -45], np.float32))), (t_suelta + 0.45, tuple(p_via)), (t_fin, tuple(p_rep))]), np.float32)
            d = R_r @ np.array([0.5, -0.86, 0.1], np.float32)
            mano = {"palma": tuple(p), "dir": tuple(d), "nrm": tuple(R_r @ np.array([0.1, 0, -1.0], np.float32)), "forma": "abierta", "polo": polo}
        return None, mano, incl, marco, suave(progreso(lt, t_suelta + 0.15, t_suelta + 0.45))

    # Mano derecha de mamá (del lado del vaso) o izquierda (del lado del niño)
    EJE_MAMA = (0.86, 0.28, 0.45)  # mango hacia mamá, que está a la derecha del niño

    def mano_en_encimera(self, x, lado="der", z=64.0):
        sx = -1 if lado == "izq" else 1
        return {"palma": (x, S.ENCIMERA_Y - 8, z), "dir": (-sx * 0.1, 0.25, 1.0), "nrm": (0, 1, 0), "forma": "abierta"}

    def bebe_y_mama(self, t, lt, escala=1.0, fase=0.0, trae=None):
        """Bebé sentado en la encimera; mamá (a su derecha) lo sostiene y le cepilla el diente."""
        ib = self.info(t, hijo(0.7), {"cara": "abierta"}, BEBE, escala=escala, desfase=1.1)
        mb = np.asarray(ib["boca"], np.float32)
        eje = N((0.78, 0.3, 0.55))
        if trae is not None and lt < trae:  # trae el cepillo con su grano de pasta desde la encimera, en arco
            cab_final = mb + N(np.asarray(ib["cabeza"]) - mb) * 7
            cab = np.array(curva(lt, [(0.0, (700, 1110, 150)), (trae * 0.5, (672, 1070, 205)), (trae, tuple(cab_final))]), np.float32)
            cep, agarre = U.cepillo(cab, eje, (-0.3, 0, -1), U.AMARILLO, pasta=0.35)
            mano_izq = {"agarre": tuple(agarre), "eje": tuple(-eje)}
        else:
            cep, mano_izq = self.cepillo_en_boca(t, ib, eje, U.AMARILLO, amp=5.0, f=2.4, fase=fase, pasta=0.35)
        xb, yb, zb = BEBE
        R = ib["R"]
        abre = trae is None or lt > trae - 0.05
        mira_b = None
        if trae is not None and lt < trae + 0.3:  # sigue el cepillo con la mirada, sin agachar del todo la cabeza
            cc = np.asarray(cep[0].c, np.float32)
            mira_b = (float(cc[0]), float(min(cc[1], BEBE[1] + 60)), float(cc[2]) + 300)
        pb, _ = self.persona(t, hijo(0.7), {"cara": "abierta" if abre else "sonrisa", "piernas": "sentado",
                                            "mano_izq": (xb - 34, yb + 2.75 * R, zb + 40), "mano_der": (xb + 34, yb + 2.75 * R, zb + 40),
                                            "mirar": mira_b}, BEBE, escala=escala, desfase=1.1)
        sosten = {"palma": (xb + R * 0.98, yb + 1.95 * R, zb - 26), "dir": (-0.15, 1.0, 0.2), "nrm": (-1.0, 0.0, 0.1), "forma": "abierta"}
        pm, _ = self.persona(t, mama(26), {"cara": "sonrisa", "giro_cuerpo": -0.28, "inclinacion": 0.1, "mano_izq": mano_izq, "mano_der": sosten,
                                           "mirar": tuple(ib["cabeza"]) if lt < 2.2 else None}, MAMA_BEBE)
        sombras = [(xb, zb + 40, 70, 45, 0.5), (xb - 30, zb + 150, 50, 30, 0.25)]
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
            cep, mano_izq = self.cepillo_en_boca(t, ib, self.EJE_MAMA, U.AMARILLO, amp=7.0, f=2.7, fase=i, pasta=0.5)
            pb, _ = self.persona(t, hijo(2.0), {"cara": "abierta"}, pos, escala=esc, desfase=1.7)
            pm, _ = self.persona(t, mama(26), {"cara": "sonrisa", "giro_cuerpo": -0.18, "mano_izq": mano_izq, "mano_der": self.mano_en_encimera(MAMA[0] + 150, "der", 60), "mirar": tuple(ib["cabeza"]) if i % 2 else None}, MAMA)
            prims, sombras = pm + pb + cep, [(MAMA[0] + 150, 60, 40, 22, 0.4)]
        return self.componer(t, luz, prims + self.encimera_objetos(t), sombras=sombras)

    # 3 · El niño aprieta la boca; mamá hace caras al espejo y cuenta con los dedos
    def escena_3(self, t, e):
        lt = t - e["inicio"]
        abre = lt > 2.25
        cara_n = "apretada" if lt < 2.0 else ("feliz" if lt < 2.25 else "abierta")
        ib = self.info(t, hijo(2.0), {"cara": cara_n}, PEQ, desfase=1.7)
        if not abre:  # espera junto a los labios cerrados, moviéndolo un poco
            cep, mano_izq = self.cepillo_en_boca(t, ib, self.EJE_MAMA, U.AMARILLO, amp=2.5, f=1.1, pasta=0.5, hondo=-16, desplaza=(14, 4, 0))
        else:
            entra = suave(progreso(lt, 2.25, 2.55))
            cep, mano_izq = self.cepillo_en_boca(t, ib, self.EJE_MAMA, U.AMARILLO, amp=7.0 * entra, f=2.7, pasta=0.5, hondo=-16 + 23 * entra, desplaza=(14 * (1 - entra), 4 * (1 - entra), 0))
        pb, _ = self.persona(t, hijo(2.0), {"cara": cara_n}, PEQ, desfase=1.7)
        cuentas = [1.0, 1.6, 2.2, 2.8, 3.4]
        n = escalon(lt, cuentas)
        im0 = self.info(t, mama(26), {"giro_cuerpo": -0.15}, MAMA)
        sube = suave(progreso(lt, 0.25, 0.75))
        mano_der = None
        if sube > 0.01:
            mano_der = self.contar(im0, "der", n, [pulso(lt, [c + 0.06], 0.07) for c in cuentas])
            if sube < 0.99:  # la mano sube desde la cadera hasta junto a la cara
                hx, hy, hz = im0["hombros"]["der"]
                ini = (hx + 10, hy + 380, hz + 40)
                mano_der["centro"] = tuple(a + (b - a) * sube for a, b in zip(ini, mano_der["centro"]))
        cara_m = "mueca" if 0.3 < lt < 2.1 else "sonrisa"
        pm, _ = self.persona(t, mama(26), {"cara": cara_m, "giro_cuerpo": -0.15, "mano_izq": mano_izq, "mano_der": mano_der}, MAMA)
        return self.componer(t, "noche", pm + pb + cep + self.encimera_objetos(t))

    # 4 · El niño (4) solo: abre el agua, moja el cepillo seco, mira a la puerta, lo devuelve y cierra
    def escena_4(self, t, e):
        lt = t - e["inicio"]
        pos = NINO4
        palanca = kf(lt, [(0.0, 0.0), (0.3, 0.0), (0.52, 1.0), (3.25, 1.0), (3.47, 0.0)])
        self.palanca = palanca
        agua = limitar((palanca - 0.4) / 0.4)
        mira_puerta = suave(progreso(lt, 1.75, 2.05)) * (1 - suave(progreso(lt, 2.45, 2.75)))
        incl = 0.32 * suave(progreso(lt, 0.85, 1.2)) * (1 - suave(progreso(lt, 2.05, 2.4)))
        al_vaso = suave(progreso(lt, 0.45, 0.62)) * (1 - suave(progreso(lt, 0.75, 0.95))) + suave(progreso(lt, 2.6, 2.85)) * (1 - suave(progreso(lt, 3.0, 3.2)))
        pose = {"cara": "neutral", "inclinacion": incl + 0.2 * al_vaso, "giro_cuerpo": -0.35 * al_vaso}
        if mira_puerta > 0:
            pose["mirar"] = tuple(np.array(curva(mira_puerta, [(0, (S.CAM.cx, S.CAM.cy, S.CAM.D)), (1, (PUERTA_XZ[0], YM, PUERTA_XZ[1]))])))
        cab_vaso, e_vaso, _ = U.pose_en_vaso(U.VASO, 1)
        sx_, sy_, sz_ = S.SALIDA_AGUA
        bajo_chorro = np.array([sx_ + 6, sy_ + 50, sz_ - 4], np.float32)
        e_chorro = N((-0.3, -0.18, -0.92))
        en_mano = 0.62 < lt < 3.0
        prims_cep = []
        if en_mano:
            arriba_vaso = cab_vaso + np.array([-12, -70, -10], np.float32)
            cab = np.array(curva(lt, [(0.62, tuple(cab_vaso)), (0.8, tuple(arriba_vaso)), (1.12, tuple(bajo_chorro + np.array([30, -50, -60], np.float32))), (1.32, tuple(bajo_chorro)), (2.05, tuple(bajo_chorro)),
                                      (2.35, tuple(bajo_chorro + np.array([40, -60, -70], np.float32))), (2.8, tuple(arriba_vaso)), (3.0, tuple(cab_vaso))]), np.float32)
            k_e = suave(progreso(lt, 0.7, 1.3)) * (1 - suave(progreso(lt, 2.1, 2.9)))
            eje = N(e_vaso * (1 - k_e) + e_chorro * k_e)
            moja = suave(progreso(lt, 1.32, 1.45)) * (1 - suave(progreso(lt, 1.9, 2.05)))
            cab = cab + np.array([3.0 * math.sin(lt * 2 * math.pi * 3.0) * moja, 0, 0], np.float32)
            prims_cep, agarre = U.cepillo(cab, eje, (0, -0.3, 1), U.AZUL, agarre=62.0)
            mano_der = {"agarre": tuple(agarre), "eje": tuple(-eje)}
        else:
            p0 = S.punta_palanca(0.0) + np.array([0, -12, 6], np.float32)
            p1 = S.punta_palanca(1.0) + np.array([0, -12, 6], np.float32)
            if lt <= 0.62:  # sube el mando del agua con los dedos y va al vaso
                dedo = np.array(curva(lt, [(0.0, (690, 1300, 40)), (0.3, tuple(p0)), (0.52, tuple(p1)), (0.62, tuple(p1 + np.array([20, -40, 0], np.float32)))]), np.float32)
            else:  # baja el mando y la mano vuelve a la encimera
                dedo = np.array(curva(lt, [(3.0, tuple(cab_vaso + np.array([-30, 50, -30], np.float32))), (3.25, tuple(p1)), (3.47, tuple(p0)), (4.0, (690, 1300, 40))]), np.float32)
            mano_der = {"palma": tuple(dedo), "dir": (0.15, 0.55, 0.8), "nrm": (0, 1, 0.2), "forma": "abierta"}
        pose["mano_der"] = mano_der
        pose["mano_izq"] = self.mano_en_encimera(pos[0] - 70, "izq", 62)
        pose["polo_der"] = (1.0, 0.4, -0.4)
        pb, _ = self.persona(t, hijo(4), pose, pos, desfase=1.7)
        prims = pb + prims_cep + self.encimera_objetos(t, ("amarillo", None if en_mano else "azul"))
        return self.componer(t, "noche", prims, agua=agua, sombras=[(pos[0] - 70, 70, 40, 22, 0.45)])

    def handover(self, lt, t0, t1, cab, eje, ag_der=55.0, ag_izq=112.0):
        """Pasa el cepillo de la mano derecha a la izquierda: la izquierda agarra más abajo y la derecha suelta."""
        cep, agarre_d = U.cepillo(cab, eje, (-0.2, -0.2, 1), U.AZUL, agarre=ag_der)
        agarre_i = np.asarray(cab, np.float32) + N(eje) * ag_izq
        k = progreso(lt, t0, t1)
        izq = {"agarre": tuple(agarre_i), "eje": tuple(-N(eje)), "polo": (-0.6, 0.8, -0.2)} if k > 0.35 else None
        der = {"agarre": tuple(agarre_d), "eje": tuple(-N(eje))} if k < 0.7 else {"palma": tuple(np.asarray(agarre_d) + np.array([60, 90, -40], np.float32) * suave(progreso(lt, t0 + 0.7 * (t1 - t0), t1 + 0.2))), "dir": (-0.3, 0.6, 0.7), "nrm": (-0.8, 0.2, -0.3), "forma": "abierta"}
        return cep, izq, der

    # 5 · Mamá entra, toca las cerdas secas, lo mira por el espejo, sonríe y se los cepilla otra vez
    def escena_5(self, t, e):
        lt = t - e["inicio"]
        pos_n = NINO4
        cara_n = "neutral" if lt < 2.55 else ("sonrisa" if lt < 3.15 else "abierta")
        ib = self.info(t, hijo(4), {"cara": cara_n}, pos_n, desfase=1.7)
        cab_v, e_v, _ = U.pose_en_vaso(U.VASO, 1)
        en_vaso = lt < 1.7
        cep = []
        if lt < 1.25:  # entra caminando desde la puerta
            pm, _ = self.mama_caminando(t, 26, 0.0, 1.25, PUERTA_XZ, (MAMA[0], MAMA[2]), {"cara": "neutral", "mirar": tuple(ib["cabeza"])}, giro_cuerpo=-0.2)
        else:
            delante = np.array([735, 1005, 150], np.float32)  # cepillo delante del pecho, cabeza arriba
            e_del = N((0.08, 1.0, -0.12))
            mano_der, mano_izq = None, None
            if lt < 1.7:  # va al vaso
                agarre_v = cab_v + e_v * 55
                mano_der = {"centro": tuple(curva(lt, [(1.25, (890, 1250, 30)), (1.62, tuple(agarre_v))])), "forma": "relajada"} if lt < 1.55 else {"agarre": tuple(agarre_v), "eje": tuple(-e_v)}
            elif lt < 2.85:  # lo saca, lo pone delante y toca las cerdas con el pulgar
                k = suave(progreso(lt, 1.7, 2.05))
                cab = np.array(curva(lt, [(1.7, tuple(cab_v)), (1.84, tuple(cab_v + np.array([0, -75, -10], np.float32))), (2.05, tuple(delante))]), np.float32) if lt < 2.05 else delante
                eje = N(e_v * (1 - k) + e_del * k)
                cep, agarre = U.cepillo(cab, eje, (-0.2, -0.2, 1), U.AZUL, agarre=55.0)
                mano_der = {"agarre": tuple(agarre), "eje": tuple(-eje)}
                toca = suave(progreso(lt, 2.05, 2.25)) * (1 - suave(progreso(lt, 2.45, 2.65)))
                if toca > 0.01:
                    p_cerdas = cab + np.array([-2, -4 + 3 * math.sin(lt * 26), 12], np.float32)
                    mano_izq = {"pinza": tuple(p_cerdas + np.array([-90, 80, 30], np.float32) * (1 - toca)), "dir": (0.6, -0.55, 0.5), "nrm": (0.2, 0.2, -1.0)}
            elif lt < 3.1:  # se lo pasa a la mano izquierda
                cep, mano_izq, mano_der = self.handover(lt, 2.85, 3.1, delante, e_del)
            else:  # se los cepilla
                lleva = suave(progreso(lt, 3.1, 3.42))
                if lt < 3.42:
                    m = np.asarray(ib["boca"], np.float32)
                    destino = m + N(np.asarray(ib["cabeza"]) - m) * 7
                    cab = delante * (1 - lleva) + destino * lleva
                    eje = N(e_del * (1 - lleva) + N(self.EJE_MAMA) * lleva)
                    cep, agarre = U.cepillo(cab, eje, (-0.2, -0.2, 1), U.AZUL, agarre=112.0)
                    mano_izq = {"agarre": tuple(agarre), "eje": tuple(-eje), "polo": (-0.6, 0.8, -0.2)}
                else:
                    cep, mano_izq = self.cepillo_en_boca(t, ib, self.EJE_MAMA, U.AZUL, amp=7.0, f=2.7)
                mano_der = self.mano_en_encimera(MAMA[0] + 150, "der", 60) if lt > 3.35 else None
            mira = None if 2.4 < lt < 2.95 else tuple(ib["cabeza"])
            cara_m = "neutral" if lt < 2.5 else "sonrisa"
            pm, _ = self.persona(t, mama(26), {"cara": cara_m, "giro_cuerpo": -0.1 if lt < 3.0 else -0.2, "mano_der": mano_der, "mano_izq": mano_izq, "mirar": mira}, MAMA)
        pb, _ = self.persona(t, hijo(4), {"cara": cara_n, "mano_izq": self.mano_en_encimera(pos_n[0] - 70, "izq", 62)}, pos_n, desfase=1.7)
        prims = pm + pb + cep + self.encimera_objetos(t, ("amarillo", "azul" if en_vaso else None))
        return self.componer(t, "noche", prims, sombras=[(pos_n[0] - 70, 70, 40, 22, 0.45)])

    # 6 · De regreso del dentista: estampa «paciente valiente», sonríe con los dientes; mamá mete la tarjeta en el marco
    def escena_6(self, t, e):
        lt = t - e["inicio"]
        pos_n = (NINO4[0], y_cabeza("nino4", 215, 1.03), NINO4[2])
        ib = self.info(t, hijo(5, True), {"cara": "dientes"}, pos_n, escala=1.03, desfase=1.7)
        manos_n = {"mano_izq": self.mano_en_encimera(pos_n[0] - 72, "izq", 64), "mano_der": self.mano_en_encimera(pos_n[0] + 72, "der", 64)}
        pb, _ = self.persona(t, hijo(5, True), dict(manos_n, cara="dientes"), pos_n, escala=1.03, desfase=1.7)
        pinza_ini = (MAMA[0] - 95, MAMA[1] + 345 + 3 * math.sin(lt * 2), MAMA[2] + 105)
        via = (MAMA[0] - 120, MAMA[1] - 10, MAMA[2] + 210)  # sube entre las dos cabezas
        reposo = (MAMA[0] - 95, MAMA[1] + 420, MAMA[2] + 60)
        prims_t, mano, incl, marco, f_sombra = self.meter_en_marco(lt, "hijo", 0, 1.3, 2.35, 2.62, pinza_ini, via=via, reposo=reposo, polo=(0.45, 0.8, 0.4), inclinacion=0.6)
        if lt > 3.47:  # luego apoya la mano en el hombro del niño
            k = suave(progreso(lt, 3.47, 3.8))
            hombro = self.mano_en_hombro(ib, "der")
            mano = dict(hombro, palma=tuple(np.asarray(reposo, np.float32) * (1 - k) + np.asarray(hombro["palma"], np.float32) * k))
        mira = tuple(ib["cabeza"]) if lt < 1.25 or lt > 3.2 else None
        puntillas = 46.0 * incl / 0.6  # se pone de puntillas para llegar arriba del espejo
        pm, _ = self.persona(t, mama(26), {"cara": "sonrisa", "inclinacion": incl, "mano_izq": mano, "mano_der": self.mano_en_encimera(MAMA[0] + 150, "der", 60), "mirar": mira}, (MAMA[0], MAMA[1] - puntillas, MAMA[2]))
        prims = pm + pb + (prims_t or []) + self.encimera_objetos(t)
        sombras = [(pos_n[0] - 72, 72, 40, 22, 0.4), (pos_n[0] + 72, 72, 40, 22, 0.4), (MAMA[0] + 150, 60, 40, 22, 0.4)]
        return self.componer(t, "manana", prims, marco=marco or (), marco_sombra=f_sombra, sombras=sombras)

    # 7 · Se le cae el primer diente: lo deja en un pañuelito en la mano de mamá; ella lo envuelve y lo guarda en la cajita
    def escena_7(self, t, e):
        lt = t - e["inicio"]
        pos_n = (NINO4[0], y_cabeza("nino4", 215, 1.06), NINO4[2])
        ib = self.info(t, hijo(6), {"cara": "dientes"}, pos_n, escala=1.06, desfase=1.7)
        hx, hy, hz = ib["cabeza"]
        R = ib["R"]
        pinza_n = np.array([hx + 0.95 * R, hy + 1.35 * R, hz + 1.1 * R], np.float32)
        palma_m = pinza_n + np.array([8, 46, 4], np.float32)
        lata_c = np.array([680.0, S.ENCIMERA_Y - U.LATA_H[1], 100.0], np.float32)
        tapa = 1.0 - suave(progreso(lt, 2.72, 3.02))
        prims_lata, borde_tapa = U.lata(tapa, centro=lata_c, llena=False)
        prims = list(prims_lata)
        reposo_n = self.mano_en_encimera(pos_n[0] + 72, "der", 64)
        if lt < 1.45:
            mano_n = {"pinza": tuple(pinza_n + np.array([0, 3 * math.sin(lt * 3), 0], np.float32)), "dir": (0.35, -0.75, 0.5), "nrm": (-0.6, 0.1, 0.8)}
        else:
            baja = suave(progreso(lt, 1.6, 2.2))
            destino = np.array(reposo_n["palma"], np.float32)
            mano_n = {"pinza": tuple(pinza_n * (1 - baja) + destino * baja), "dir": (0.35, -0.75, 0.5), "nrm": (-0.6, 0.1, 0.8)} if baja < 0.98 else reposo_n
        cae = suave(progreso(lt, 1.45, 1.58))
        if lt < 1.8:
            prims += U.diente(pinza_n * (1 - cae) + (palma_m + np.array([0, -10, 0], np.float32)) * cae)
        pb, _ = self.persona(t, hijo(6), {"cara": "dientes", "hueco": True, "mano_der": mano_n, "mano_izq": self.mano_en_encimera(pos_n[0] - 72, "izq", 64), "cabeceo": 0.12 if 0.8 < lt < 1.8 else 0.0, "giro": 0.25 if 0.8 < lt < 1.8 else 0.0}, pos_n, escala=1.06, desfase=1.7)
        llega = np.array(curva(lt, [(0.6, (MAMA[0] - 120, S.ENCIMERA_Y - 20, 70)), (1.25, tuple(palma_m))]), np.float32)
        boca_lata = lata_c + np.array([0, -U.LATA_H[1] - 40, 0], np.float32)
        if lt < 1.65:
            palma = llega if lt < 1.25 else palma_m
            mano_m = {"palma": tuple(palma), "dir": (-0.9, 0.0, 0.45), "nrm": (0.0, -1.0, 0.0), "forma": "abierta"}
            prims += U.papelito(np.asarray(palma) + np.array([0, -14, 0], np.float32), 0.0, J.rot_y(-0.3))
        elif lt < 2.62:
            k_bola = suave(progreso(lt, 1.65, 2.0))
            p = np.array(curva(lt, [(1.65, tuple(palma_m)), (2.05, tuple(palma_m + np.array([30, 10, 0], np.float32))), (2.5, tuple(boca_lata))]), np.float32)
            mano_m = {"pinza": tuple(p), "dir": (-0.6, 0.5, 0.6), "nrm": (-0.1, 0.4, -0.9)}
            prims += U.papelito(p + np.array([0, 4, 0], np.float32), k_bola, J.rot_y(-0.3))
        else:
            if lt < 2.75:  # la bolita cae dentro
                k = progreso(lt, 2.62, 2.72)
                prims += U.papelito(boca_lata + np.array([0, 40 * k * k, 0], np.float32), 1.0)
            if lt < 3.05:
                yemas = np.asarray(borde_tapa, np.float32)
                mano_m = {"palma": tuple(yemas + np.array([0, -6, -30], np.float32)), "dir": (0.0, 0.35, 1.0), "nrm": (0, 1, -0.2), "forma": "abierta"}
            else:  # la mano descansa sobre la tapa y da dos palmaditas
                pal = 6 * pulso(lt, [3.3, 3.62], 0.07)
                mano_m = {"palma": tuple(lata_c + np.array([0, -U.LATA_H[1] - 10 - pal, 0], np.float32)), "dir": (-0.15, 0.2, 1.0), "nrm": (0, 1, 0), "forma": "abierta"}
        pm, _ = self.persona(t, mama(26), {"cara": "sonrisa", "giro_cuerpo": -0.15, "mano_izq": mano_m, "mirar": tuple(pinza_n) if lt < 1.7 else tuple(lata_c)}, MAMA)
        prims += pm + pb + self.encimera_objetos(t)
        sombras = [(680, 100, 80, 52, 0.6), (pos_n[0] - 72, 72, 40, 22, 0.35), (pos_n[0] + 72, 72, 40, 22, 0.35 if lt > 2.2 else 0.0)]
        return self.componer(t, "manana", prims, marco=tarjeta_en_ranura("hijo", 0), sombras=sombras)

    # 8 · De noche, mamá sola deja unas monedas y escribe la nota del Ratón de los Dientes (acercamiento lento)
    def escena_8(self, t, e):
        lt = t - e["inicio"]
        nota_c = np.array([588.0, S.ENCIMERA_Y - 0.9, 102.0], np.float32)
        ancho, fondo = 132.0, 88.0
        largos = _metricas_nota()
        f1 = largos[0] / sum(largos)
        if lt < 1.45:
            k_txt, viaje = 0.0, 0.0
        elif lt < 2.55:
            k_txt, viaje = f1 * progreso(lt, 1.45, 2.55), 0.0
        elif lt < 2.75:
            k_txt, viaje = f1, progreso(lt, 2.55, 2.75)
        else:
            k_txt, viaje = f1 + (1 - f1) * progreso(lt, 2.75, 3.7), 0.0
        tex = tex_nota(round(k_txt * 60) / 60)
        prims = U.nota_en_encimera(nota_c, ancho, fondo, tex)

        def en_nota(u, v, alto=0.0):
            return nota_c + np.array([(u - 0.5) * ancho, -1.0 - alto, (v - 0.5) * fondo], np.float32)

        if viaje > 0:
            u0, v0 = NOTA_LINEAS[0][1] + largos[0], NOTA_LINEAS[0][2]
            u1, v1 = NOTA_LINEAS[1][1], NOTA_LINEAS[1][2]
            punta = en_nota(u0 + (u1 - u0) * suave(viaje), v0 + (v1 - v0) * suave(viaje), 14 * math.sin(math.pi * viaje))
        elif 1.45 <= lt < 3.7:
            u, v = punta_nota(k_txt, t)
            punta = en_nota(u, v, 1.5 * max(0.0, math.sin(lt * 2 * math.pi * 3.2)) ** 6)
        elif lt < 1.45:
            ini = en_nota(NOTA_LINEAS[0][1], NOTA_LINEAS[0][2], 0)
            punta = np.array(curva(lt, [(0.0, tuple(ini + np.array([90, -70, 30], np.float32))), (1.0, tuple(ini + np.array([60, -45, 15], np.float32))), (1.45, tuple(ini))]), np.float32)
        else:
            fin = en_nota(*punta_nota(1.0, 3.7))
            punta = fin + np.array([18, -40, 10], np.float32) * suave(progreso(lt, 3.7, 4.0))
        eje_l = N((0.42, -0.78, -0.45))
        prims += U.lapiz(punta, eje_l)
        mano_der = {"pinza": tuple(punta + eje_l * 34), "dir": (-0.15, 0.55, 0.82), "nrm": (-0.75, 0.1, -0.2), "polo": (0.35, 0.6, -0.8)}
        sitio = np.array([455.0, S.ENCIMERA_Y, 100.0], np.float32)
        p_mon = np.array(curva(lt, [(0.0, (420, 1090, 120)), (0.5, tuple(sitio + np.array([0, -16, 0], np.float32)))]), np.float32)
        if lt < 0.55:
            mano_izq = {"pinza": tuple(p_mon), "dir": (0.2, 0.75, 0.6), "nrm": (0.6, 0.3, -0.6), "polo": (-0.35, 0.6, -0.8)}
            for k in range(3):
                prims += U.moneda(p_mon + np.array([0, k * 3.6 - 4, 0], np.float32))
        else:
            for k, (dx_, dz_) in enumerate(((0, 0), (14, 9), (-9, 13))):
                prims += U.moneda(sitio + np.array([dx_, -2 - 3.6 * (k == 0), dz_], np.float32))
            sujeta = en_nota(0.03, 0.62, 0) + np.array([-22, -10, 0], np.float32)
            p = np.array(curva(lt, [(0.55, tuple(sitio + np.array([0, -30, 0], np.float32))), (1.05, tuple(sujeta))]), np.float32)
            mano_izq = {"palma": tuple(p), "dir": (0.45, 0.3, 0.85), "nrm": (0, 1, 0), "forma": "abierta", "polo": (-0.35, 0.6, -0.8)}
        cara = "concentrada" if lt < 3.75 else "sonrisa"
        pm, _ = self.persona(t, mama(26), {"cara": cara, "mirar": tuple(punta), "inclinacion": 0.22, "mano_der": mano_der, "mano_izq": mano_izq}, MAMA_SOLA)
        prims += pm + self.encimera_objetos(t)
        sombras = [(588, 102, 70, 46, 0.25), (455, 104, 26, 18, 0.4 if lt > 0.5 else 0.0)]
        return self.componer(t, "noche", prims, marco=tarjeta_en_ranura("hijo", 0), sombras=sombras)

    def nino_cepillandose(self, t, edad, pos, lado="der", cara="abierta", amp=9.0, f=2.6, agarre=70.0, extra=None, escala=1.0):
        """El niño se cepilla solo (por defecto con la mano derecha, mango hacia la derecha)."""
        quien = hijo(edad)
        base = {"cara": cara}
        base.update(extra or {})
        ib = self.info(t, quien, base, pos, escala=escala, desfase=1.7)
        sx = -1 if lado == "izq" else 1
        e = N((sx * 0.84, 0.3, 0.48))
        hx, hy, hz = ib["cabeza"]
        m = np.asarray(ib["boca"], np.float32)
        hacia = N(np.array([hx, hy, hz], np.float32) - m)
        horiz = N((e[0], 0.0, e[2] * 0.4))
        v = math.sin(2 * math.pi * f * t)
        cab = m + hacia * 7 + horiz * amp * v + np.array([0, 2.2 * math.sin(4 * math.pi * f * t), 0], np.float32)
        cep, agarre_p = U.cepillo(cab, e, hacia, U.AZUL, agarre=agarre)
        pose = dict(base)
        pose[f"mano_{lado}"] = {"agarre": tuple(agarre_p), "eje": tuple(-e), "polo": (sx * 0.55, 1.0, 0.3)}
        pose.setdefault(f"mano_{'izq' if lado == 'der' else 'der'}", self.mano_en_encimera(pos[0] - sx * 72, "izq" if lado == "der" else "der", 62))
        pb, ib = self.persona(t, quien, pose, pos, escala=escala, desfase=1.7)
        return pb + cep, ib

    # 9 · El niño (7) agarra el cepillo («yo solito»): ella lo suelta y da pasos atrás hasta la puerta
    def escena_9(self, t, e):
        lt = t - e["inicio"]
        pos_n = NINO7
        ib = self.info(t, hijo(7), {"cara": "abierta"}, pos_n, desfase=1.7)
        para = suave(progreso(lt, 0.5, 0.8))
        cep, mano_m = self.cepillo_en_boca(t, ib, self.EJE_MAMA, U.AZUL, amp=7.0 * (1 - para), f=2.7)
        e_m = N(self.EJE_MAMA)
        cab = np.asarray(mano_m["agarre"], np.float32) - e_m * U.AGARRE_CEPILLO
        if lt < 1.15:
            agarre_n = cab + e_m * 68
            sube = suave(progreso(lt, 0.45, 0.9))
            mano_n = None
            if sube > 0.01:
                lejos = agarre_n + np.array([30, 170, -40], np.float32)
                mano_n = {"centro": tuple(lejos * (1 - sube) + agarre_n * sube), "forma": "relajada"} if sube < 0.97 else {"agarre": tuple(agarre_n), "eje": tuple(-e_m), "polo": (1.0, 0.7, -0.1)}
            pb, _ = self.persona(t, hijo(7), {"cara": "neutral" if lt < 0.9 else "abierta", "mano_der": mano_n, "mano_izq": self.mano_en_encimera(pos_n[0] - 72, "izq", 62)}, pos_n, desfase=1.7)
            prims = pb + cep
        else:
            prims, _ = self.nino_cepillandose(t, 7, pos_n, amp=8.0 * suave(progreso(lt, 1.15, 1.6)), f=2.5, agarre=68.0)
        if lt < 1.0:  # mamá todavía lo sostiene
            pm, _ = self.persona(t, mama(26), {"cara": "sonrisa", "giro_cuerpo": -0.15, "mano_izq": mano_m, "mirar": tuple(ib["cabeza"])}, MAMA)
        elif lt < 1.45:  # lo suelta: la mano se abre y se retira
            k = suave(progreso(lt, 1.0, 1.45))
            ag = np.asarray(mano_m["agarre"], np.float32)
            pm, _ = self.persona(t, mama(26), {"cara": "sonrisa", "giro_cuerpo": -0.15 * (1 - k), "mano_izq": {"palma": tuple(ag + np.array([40, 60, -50], np.float32) * k), "dir": (-0.7, 0.2, 0.7), "nrm": (0, 1, -0.3), "forma": "abierta"} if k < 0.9 else None, "mirar": tuple(ib["cabeza"])}, MAMA)
        else:
            pm, _ = self.mama_caminando(t, 26, 1.45, 3.65, (MAMA[0], MAMA[2]), PUERTA_XZ, {"cara": "sonrisa", "mirar": tuple(ib["cabeza"])}, giro_cuerpo=-0.1)
        return self.componer(t, "noche", prims + pm + self.encimera_objetos(t, ("amarillo", None)), marco=tarjeta_en_ranura("hijo", 0))

    # 10 · Mamá lo mira desde la puerta; levanta la mano para ayudarle y la baja (acercamiento lento)
    def escena_10(self, t, e):
        lt = t - e["inicio"]
        prims, ib = self.nino_cepillandose(t, 7, NINO7, amp=9.0 + 3 * math.sin(lt * 1.3), f=2.3, agarre=68.0)
        x, z = PUERTA_XZ
        sube = suave(progreso(lt, 1.0, 1.6)) * (1 - suave(progreso(lt, 2.4, 3.0)))
        im0 = self.info(t, mama(26), {}, (x, YM, z))
        hxs, hys, hzs = im0["hombros"]["izq"]
        R = im0["R"]
        abajo = np.array([hxs - 0.15 * R, hys + 4.0 * R, hzs + 0.3 * R], np.float32)
        arriba = np.array([hxs - 1.4 * R, hys + 1.9 * R, hzs + 2.4 * R], np.float32)
        mano = None
        if sube > 0.01:
            p = abajo * (1 - sube) + arriba * sube
            mano = {"palma": tuple(p), "dir": (-0.45, 0.15, 0.9), "nrm": (0.2, 0.8, 0.4), "forma": "abierta"}
        cara = "sonrisa" if lt < 2.5 else "neutral"
        pm, _ = self.mama_puerta(t, 26, {"cara": cara, "mano_izq": mano, "inclinacion": 0.06 * sube, "mirar": tuple(ib["cabeza"])})
        return self.componer(t, "noche", prims + pm + self.encimera_objetos(t, ("amarillo", None)), marco=tarjeta_en_ranura("hijo", 0))

    # 11 · Solo cambia el contador
    def escena_11(self, t, e):
        prims, _ = self.nino_cepillandose(t, 7, NINO7, f=2.4, agarre=68.0)
        return self.componer(t, "noche", prims + self.encimera_objetos(t, ("amarillo", None)), marco=tarjeta_en_ranura("hijo", 0))

    # 12 · Adolescente (15) se cepilla con desgana mirando el celular; llega la notificación
    def escena_12(self, t, e):
        lt = t - e["inicio"]
        quien = hijo(15)
        tel_c = np.array([TEEN[0] - 60, TEEN[1] + 262, 150.0], np.float32)
        ib0 = self.info(t, quien, {}, TEEN)
        hacia_ojos = N(np.asarray(ib0["cabeza"], np.float32) - tel_c)
        arriba = np.array([0.0, -1.0, 0.0], np.float32)
        rot = J.base_desde(-(arriba - hacia_ojos * float(arriba @ hacia_ojos)), -hacia_ojos)
        tel_c = tel_c + np.array([0, 2 * math.sin(lt * 1.7), 0], np.float32)
        lee = suave(progreso(lt, 1.7, 1.9)) * (1 - suave(progreso(lt, 2.4, 2.7)))
        pose = {"cara": "abierta", "mirar": tuple(tel_c)}
        ib = self.info(t, quien, pose, TEEN)
        cep, mano_der = self.cepillo_en_boca(t, ib, (0.82, 0.32, 0.48), U.AZUL, amp=6.0 * (1 - 0.9 * lee), f=2.0)
        mano_der["polo"] = (0.5, 1.0, 0.3)
        pulgar = 4 * pulso(lt, [2.05, 2.25], 0.05)
        mano_izq = {"centro": tuple(tel_c + rot @ np.array([0, 40, -14], np.float32) + np.array([0, 0, pulgar], np.float32)), "dir": tuple(rot @ np.array([0, -1, 0], np.float32)), "nrm": tuple(rot @ np.array([0, 0, -1], np.float32))}
        pose.update({"mano_der": mano_der, "mano_izq": mano_izq})
        pt, _ = self.persona(t, quien, pose, TEEN)
        prims = pt + cep + U.telefono(tel_c, rot) + self.encimera_objetos(t, ("amarillo", None))
        return self.componer(t, "manana", prims, marco=tarjeta_en_ranura("hijo", 0))

    # 13 · De noche, mamá (45) se toca la mejilla con dolor; toma su tarjeta, tacha la fecha y escribe otra
    def escena_13(self, t, e):
        lt = t - e["inicio"]
        quien = mama(45)
        im0 = self.info(t, quien, {"giro": -0.15}, MAMA_SOLA)
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
        en_encimera = np.array([598.0, S.ENCIMERA_Y - 1.4, 98.0], np.float32)
        en_mano = np.array([578.0, 1058.0, 150.0], np.float32)
        sube = suave(progreso(lt, 1.5, 2.05))
        c = np.array(curva(lt, [(1.5, tuple(en_encimera)), (1.78, tuple(en_encimera * 0.5 + en_mano * 0.5 + np.array([0, -40, 0], np.float32))), (2.05, tuple(en_mano))]), np.float32) if lt < 2.05 else en_mano
        ang = math.radians(90) * (1 - sube) + math.radians(-38) * sube
        Rc = J.rot_x(ang) @ J.rot_z(math.radians(-4))
        prims = U.tarjeta(c, aw, ah, tex, Rc)

        def en_tarjeta(px, py, alto=0.0):
            return c + Rc @ np.array([(px / w - 0.5) * aw, (py / h - 0.5) * ah, 1.4 + alto], np.float32)

        if lt < 1.35:
            pega = suave(progreso(lt, 0.0, 0.3))
            mejilla = np.array([hx - 0.76 * R, hy + 0.36 * R, hz + 0.66 * R], np.float32)
            lejos = np.array([hx - 1.6 * R, hy + 3.0 * R, hz + 0.6 * R], np.float32)
            mano_izq = {"palma": tuple(mejilla * pega + lejos * (1 - pega)), "dir": (0.2, -1.0, 0.15), "nrm": (0.75, 0.0, -0.65), "forma": "abierta"}
        else:
            borde = en_tarjeta(10, h * 0.62)
            k = suave(progreso(lt, 1.35, 1.5))
            p = np.array([hx - 1.3 * R, hy + 2.5 * R, hz + 0.8 * R], np.float32) * (1 - k) + borde * k if lt < 1.5 else borde
            mano_izq = {"pinza": tuple(p), "dir": tuple(Rc @ np.array([0.85, -0.4, 0.2], np.float32)), "nrm": tuple(Rc @ np.array([0, 0, -1.0], np.float32))}
        f_t = fuente("manuscrita", 40)
        y_ult = 76 + 46 * (len(tach) - 1)
        w_ult = f_t.getlength(tach[-1])
        y_nueva = 76 + 46 * len(tach)
        w_nueva = f_t.getlength(datos["nueva"])
        if lt < 2.15:
            reposo = np.array([hx + 1.5 * R, S.ENCIMERA_Y - 30, 70], np.float32)
            punta = np.array(curva(lt, [(0.0, tuple(reposo)), (1.7, tuple(reposo + np.array([-10, -20, 20], np.float32))), (2.15, tuple(en_tarjeta(18, y_ult + 23, 6)))]), np.float32)
        elif lt < 2.55:
            k = progreso(lt, 2.2, 2.55)
            punta = en_tarjeta(18 + (w_ult + 8) * k, y_ult + 26 - 6 * k)
        elif lt < 2.7:
            k = suave(progreso(lt, 2.55, 2.7))
            punta = en_tarjeta(26 + w_ult - (w_ult + 4) * k, y_ult + 22 + (y_nueva - y_ult) * k, 10 * math.sin(math.pi * k))
        elif lt < 3.65:
            k = progreso(lt, 2.7, 3.65)
            punta = en_tarjeta(22 + w_nueva * k, y_nueva + 20 + 7 * math.sin(lt * 2 * math.pi * 6.5))
        else:
            punta = en_tarjeta(22 + w_nueva, y_nueva + 18, 22 * suave(progreso(lt, 3.65, 3.95)))
        eje_l = N((0.45, -0.75, -0.4))
        prims += U.lapiz(punta, eje_l, color=(150, 60, 60))
        mano_der = {"pinza": tuple(punta + eje_l * 34), "dir": (-0.2, 0.55, 0.8), "nrm": (-0.75, 0.1, -0.2), "polo": (0.35, 0.6, -0.8)}
        cara = "dolor" if lt < 1.4 else ("concentrada" if lt < 3.7 else "neutral")
        pm, _ = self.persona(t, quien, {"cara": cara, "giro": -0.15 if lt < 1.4 else 0.0, "mano_izq": mano_izq, "mano_der": mano_der, "mirar": None if lt < 1.4 else tuple(c), "inclinacion": 0.06}, MAMA_SOLA)
        prims += pm + self.encimera_objetos(t, ("amarillo", None))
        return self.componer(t, "noche", prims, marco=tarjeta_en_ranura("hijo", 0), sombras=[(598, 98, 70, 50, 0.35 * (1 - sube))])

    # 14 · «25 AÑOS DESPUÉS» (el baño vacío; la tarjeta vieja de mamá en el marco)
    def escena_14(self, t, e):
        return self.componer(t, "manana", self.encimera_objetos(t, ("amarillo", "rosa")), marco=tarjeta_en_ranura("mama_vieja", 1))

    # 15 · Tú (32) le cepillas los dientes a tu hija (3) y cuentas con los dedos; mamá (58) mira desde la puerta
    def escena_15(self, t, e):
        lt = t - e["inicio"]
        inf = self.info(t, nieta(), {"cara": "abierta"}, NIETA, desfase=0.6)
        cep, mano_der = self.cepillo_en_boca(t, inf, (-0.86, 0.28, 0.45), U.ROSA, amp=6.5, f=2.6, pasta=0.4)
        pn, _ = self.persona(t, nieta(), {"cara": "abierta", "mano_izq": self.mano_en_encimera(NIETA[0] - 68, "izq", 62), "mano_der": self.mano_en_encimera(NIETA[0] + 68, "der", 62)}, NIETA, desfase=0.6)
        cuentas = [0.6, 1.2, 1.8, 2.4, 3.0]
        ih0 = self.info(t, hijo(32), {"giro_cuerpo": 0.15}, HIJO32, desfase=2.2)
        sube = suave(progreso(lt, 0.0, 0.4))
        mano_izq = self.contar(ih0, "izq", escalon(lt, cuentas), [pulso(lt, [c + 0.06], 0.07) for c in cuentas])
        cx_, cy_, cz_ = mano_izq["centro"]
        mano_izq["centro"] = (cx_ + 0.3 * ih0["R"], cy_ + 0.25 * ih0["R"], cz_ + 0.15 * ih0["R"])  # delante del pecho, lejos del contador
        if sube < 0.99:
            hx, hy, hz = ih0["hombros"]["izq"]
            ini = (hx - 10, hy + 380, hz + 40)
            mano_izq["centro"] = tuple(a + (b - a) * sube for a, b in zip(ini, mano_izq["centro"]))
        ph, _ = self.persona(t, hijo(32), {"cara": "sonrisa", "giro_cuerpo": 0.15, "mano_der": mano_der, "mano_izq": mano_izq}, HIJO32, desfase=2.2)
        pm, _ = self.mama_puerta(t, 58, {"cara": "sonrisa", "mirar": tuple(inf["cabeza"])})
        prims = ph + pn + cep + pm + self.encimera_objetos(t, ("amarillo", None))
        return self.componer(t, "noche", prims, marco=tarjeta_en_ranura("mama_vieja", 1), sombras=[(NIETA[0] - 68, 62, 36, 20, 0.35), (NIETA[0] + 68, 62, 36, 20, 0.35)])

    def lata_en_manos(self, lt, t_sube=0.8, t_listo=1.4, t_abre=(1.45, 1.9)):
        """Pose de la cajita que el hijo saca del cajón, abre y sostiene inclinada hacia el espejo."""
        abajo = np.array([HIJO_SOLO[0] + 20, S.ENCIMERA_Y + 110, 60.0], np.float32)
        arriba = np.array([HIJO_SOLO[0] + 25, 1085.0, 135.0], np.float32)
        k = suave(progreso(lt, t_sube, t_listo))
        c = np.array(curva(lt, [(t_sube, tuple(abajo)), ((t_sube + t_listo) / 2, tuple(abajo * 0.4 + arriba * 0.6 + np.array([0, -30, 0], np.float32))), (t_listo, tuple(arriba))]), np.float32)
        inclina = suave(progreso(lt, t_abre[1], t_abre[1] + 0.4))
        rot = J.rot_x(-0.72 * inclina)
        abierta = suave(progreso(lt, *t_abre))
        return c, rot, abierta, k

    def manos_lata(self, c, rot):
        hx_ = U.LATA_H[0]
        izq = c + rot @ np.array([-hx_ - 4, 6, 0], np.float32)
        der = c + rot @ np.array([hx_ + 4, 6, 0], np.float32)
        return ({"palma": tuple(izq), "dir": tuple(rot @ np.array([0.2, 0.3, 1.0], np.float32)), "nrm": tuple(rot @ np.array([1.0, 0, 0], np.float32)), "forma": "abierta"},
                {"palma": tuple(der), "dir": tuple(rot @ np.array([-0.2, 0.3, 1.0], np.float32)), "nrm": tuple(rot @ np.array([-1.0, 0, 0], np.float32)), "forma": "abierta"})

    # 16 · Abres el cajón buscando hilo dental y encuentras la cajita: tus 20 dientes de leche (acercamiento)
    def escena_16(self, t, e):
        lt = t - e["inicio"]
        c, rot, abierta, k = self.lata_en_manos(lt)
        prims_lata, borde_tapa = U.lata(abierta, centro=c, rot=rot)
        prims = list(prims_lata)
        if lt < 0.8:  # las manos buscan en el cajón (debajo de la encimera)
            hurga = 8 * math.sin(lt * 9)
            mano_izq = {"centro": (HIJO_SOLO[0] - 90, S.ENCIMERA_Y + 120 + hurga, 60), "forma": "relajada"}
            mano_der = {"centro": (HIJO_SOLO[0] + 110, S.ENCIMERA_Y + 115 - hurga, 60), "forma": "relajada"}
            prims = []
        else:
            mano_izq, mano_der = self.manos_lata(c, rot)
            if 1.4 < lt < 2.0:  # abre la tapa con los dedos
                mano_der = {"palma": tuple(np.asarray(borde_tapa, np.float32) + np.array([0, -4, -26], np.float32)), "dir": (0.0, 0.25, 1.0), "nrm": (0, 1, -0.3), "forma": "abierta"}
        cara = "neutral" if lt < 1.95 else ("sorpresa" if lt < 3.2 else "sonrisa")
        mira = tuple(c) if lt > 0.8 else (HIJO_SOLO[0], S.ENCIMERA_Y + 150, 80)
        ph, _ = self.persona(t, hijo(32), {"cara": cara, "mirar": mira, "mano_izq": mano_izq, "mano_der": mano_der, "inclinacion": 0.08}, HIJO_SOLO, desfase=2.2)
        pm, _ = self.mama_puerta(t, 58, {"cara": "neutral" if lt < 2.2 else "sonrisa", "mirar": tuple(c) if lt > 1.0 else None})
        prims += ph + pm + self.encimera_objetos(t, ("amarillo", None))
        return self.componer(t, "noche", prims, marco=tarjeta_en_ranura("mama_vieja", 1))

    # 17 · Mamá se ríe desde la puerta tapándose la boca; tú la miras por el espejo con la cajita abierta
    def escena_17(self, t, e):
        lt = t - e["inicio"]
        c, rot, _, _ = self.lata_en_manos(9.0)
        prims = list(U.lata(1.0, centro=c, rot=rot)[0])
        mano_izq, mano_der = self.manos_lata(c, rot)
        ph, _ = self.persona(t, hijo(32), {"cara": "sonrisa", "inclinacion": 0.08, "mirar": tuple(c) if lt < 0.8 else None, "mano_izq": mano_izq, "mano_der": mano_der}, HIJO_SOLO, desfase=2.2)
        x, z = PUERTA_XZ
        risa = suave(progreso(lt, 0.0, 0.4)) * (1 - suave(progreso(lt, 3.4, 3.9)))
        sacude = 3.0 * math.sin(lt * 2 * math.pi * 4.5) * risa
        im0 = self.info(t, mama(58), {"cara": "tapada"}, (x, YM + sacude, z), desfase=0.4)
        hx, hy, hz = im0["cabeza"]
        R = im0["R"]
        boca = np.array([hx, hy + 0.38 * R, hz + 0.95 * R], np.float32)
        abajo = np.array([hx - 1.3 * R, hy + 4.4 * R, hz + 0.2 * R], np.float32)
        mano = {"palma": tuple(abajo * (1 - risa) + boca * risa), "dir": (0.25, -0.9, 0.3), "nrm": (0.0, 0.1, -1.0), "forma": "abierta"} if risa > 0.02 else None
        pm, _ = self.persona(t, mama(58), {"cara": "tapada" if risa > 0.3 else "risa", "mano_izq": mano, "piernas": "de_pie"}, (x, YM + sacude, z), desfase=0.4)
        prims += ph + pm + self.encimera_objetos(t, ("amarillo", None))
        return self.componer(t, "noche", prims, marco=tarjeta_en_ranura("mama_vieja", 1))

    # 18 · Metes en el marco una tarjeta nueva; ella se acerca, la lee y sonríe enseñando los dientes por primera vez
    def escena_18(self, t, e):
        lt = t - e["inicio"]
        lata_c = np.array([300.0, S.ENCIMERA_Y - U.LATA_H[1], 112.0], np.float32)
        prims = list(U.lata(1.0, centro=lata_c)[0])
        pinza_ini = (HIJO_SOLO[0] + 115, HIJO_SOLO[1] + 360 + 3 * math.sin(lt * 2), HIJO_SOLO[2] + 125)
        via = (HIJO_SOLO[0] + 185, HIJO_SOLO[1] + 60, HIJO_SOLO[2] + 200)  # sube por su derecha, sin taparse la cara
        reposo = (HIJO_SOLO[0] + 120, HIJO_SOLO[1] + 470, HIJO_SOLO[2] + 60)
        prims_t, mano, incl, marco_n, f_sombra = self.meter_en_marco(lt, "nueva", 2, 0.35, 1.25, 1.45, pinza_ini, via=via, reposo=reposo, inclinacion=0.62, polo=(0.6, 0.8, 0.3))
        puntillas = 40.0 * incl / 0.62
        ph, _ = self.persona(t, hijo(32), {"cara": "sonrisa", "inclinacion": incl, "mano_der": mano, "mano_izq": self.mano_en_encimera(HIJO_SOLO[0] - 150, "izq", 60)}, (HIJO_SOLO[0], HIJO_SOLO[1] - puntillas, HIJO_SOLO[2]), desfase=2.2)
        cara = "neutral" if lt < 2.95 else "dientes"
        mira_tarjeta = tuple(float(v) for v in ranura(2, *medidas_tarjeta("nueva"))[0])
        if 1.6 <= lt < 2.95:
            pm, _ = self.mama_caminando(t, 58, 1.6, 2.7, PUERTA_XZ, (1010.0, -470.0), {"cara": cara, "mirar": mira_tarjeta}, giro_cuerpo=-0.15)
        else:
            xz = PUERTA_XZ if lt < 1.6 else (1010.0, -470.0)
            pm, _ = self.persona(t, mama(58), {"cara": cara, "piernas": "de_pie", "mirar": mira_tarjeta if lt > 1.2 else None}, (xz[0], YM, xz[1]), desfase=0.4)
        prims += ph + pm + (prims_t or []) + self.encimera_objetos(t, ("amarillo", None))
        marco = list(tarjeta_en_ranura("mama_vieja", 1)) + list(marco_n or [])
        return self.componer(t, "noche", prims, marco=marco, sombras=[(300, 112, 80, 52, 0.55), (HIJO_SOLO[0] - 150, 60, 40, 22, 0.4)])

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
