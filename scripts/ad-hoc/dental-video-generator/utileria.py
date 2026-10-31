"""
Utilería 3D de «La última vez que tu mamá te cepilló los dientes»: cepillos
rígidos (mango, cuello, cabezal y cerdas), vaso, tubo de pasta, cajita de lata
con tapa de bisagra, tarjetas y nota con texto (textura), lápiz, monedas,
celular, diente y papelito. Todo en coordenadas del mundo (ver `juguete3d`).

Los objetos se colocan por su punto natural (la cabeza del cepillo, la punta
del lápiz, el centro de la tarjeta…) y devuelven también el punto por donde
los agarra la mano, para que mano y objeto vayan siempre juntos.
"""

import functools
import math

import numpy as np
from PIL import Image, ImageDraw

import guion_espejo as GE
import juguete3d as J
from escenario import ENCIMERA_Y
from utilidades import fuente

BLANCO_CERDAS = (238, 242, 246)
AMARILLO = (238, 196, 70)  # cepillo de mamá
AZUL = (90, 140, 230)  # cepillo del niño
ROSA = (240, 120, 150)  # cepillo de la nieta
LARGO_CEPILLO = 172.0
AGARRE_CEPILLO = 118.0  # distancia de la cabeza al centro del puño


# ---------------------------------------------------------------------------
# Cepillo de dientes
# ---------------------------------------------------------------------------
def cepillo(cabeza, eje, cerdas, color, pasta=0.0, agarre=AGARRE_CEPILLO):
    """
    Cepillo rígido: `cabeza` es el centro del cabezal, `eje` apunta de la cabeza hacia el
    final del mango y `cerdas` hacia donde miran las cerdas. `pasta` 0–1 = tamaño del grano.
    Devuelve (primitivas, punto de agarre).
    """
    c = np.asarray(cabeza, np.float32)
    e = J._norm(eje)
    s = np.asarray(cerdas, np.float32)
    s = J._norm(s - e * float(s @ e))
    rot = np.stack([J._norm(np.cross(e, s)), s, e], 1)  # x lateral, y cerdas, z a lo largo del mango
    prims = [
        J.Caja(tuple(c + s * 5), (6.5, 5.5, 14), BLANCO_CERDAS, rot=rot, bisel=0.5, brillo=0.05),  # cerdas
        J.Caja(tuple(c - s * 2.5), (6.5, 3.0, 16), color, rot=rot, bisel=0.6, brillo=0.35),  # base del cabezal
        J.Capsula(tuple(c + e * 14 - s * 2.5), tuple(c + e * 48 - s * 1), 3.6, color, brillo=0.35),  # cuello
        J.Capsula(tuple(c + e * 48 - s * 1), tuple(c + e * LARGO_CEPILLO), 6.8, color, escala=(1.0, 1.0, 1.0), brillo=0.35),  # mango
        J.Capsula(tuple(c + e * 92), tuple(c + e * 132), 7.4, tuple(int(v * 0.8) for v in color), brillo=0.2),  # goma del agarre
    ]
    if pasta > 0.01:
        r = 3 + 7 * pasta
        prims.append(J.Elipsoide(tuple(c + s * (11 + r * 0.4) + e * 2), (r * 0.8, r * 0.75, r * 1.3), (250, 252, 252), rot=rot, brillo=0.3))
    return prims, c + e * agarre


def pose_en_vaso(vaso, k):
    """(cabeza, eje, cerdas) del cepillo `k` metido en el vaso (cabeza arriba, mango apoyado en el borde)."""
    x, y_borde, z = vaso
    inclinacion = (-0.36, 0.30, 0.05)[k % 3]
    fondo = np.array([x + (k - 1) * 9, y_borde + 80, z + (k - 1) * 6], np.float32)
    e = J._norm((-inclinacion, 1.0, -0.12 + 0.1 * k))
    cabeza = fondo - e * (LARGO_CEPILLO - 4)
    cerdas = J._norm((0.2 * (1 if inclinacion > 0 else -1), 0, 1))
    return cabeza, e, cerdas


def cepillo_en_vaso(vaso, k, color):
    """Cepillo `k` metido en el vaso."""
    cabeza, e, cerdas = pose_en_vaso(vaso, k)
    return cepillo(cabeza, e, cerdas, color)[0]


# ---------------------------------------------------------------------------
# Encimera: vaso, pasta, cajita de lata
# ---------------------------------------------------------------------------
VASO = (868.0, ENCIMERA_Y - 112.0, 178.0)  # x, y del borde, z


def vaso():
    x, yb, z = VASO
    hy = (ENCIMERA_Y - yb) / 2
    return [J.Cilindro((x, yb + hy, z), 40, hy, (150, 210, 196), brillo=0.35, abierto=True, color_interior=(70, 104, 98))]


def tubo_pasta(pos=(215.0, 160.0), giro=0.12, en_mano=None, eje_mano=None):
    """Tubo de pasta tumbado en la encimera (x, z) o en la mano (`en_mano` = centro, `eje_mano` hacia la boquilla)."""
    if en_mano is None:
        x, z = pos
        c = np.array([x, ENCIMERA_Y - 13, z], np.float32)
        e = np.array([math.cos(giro), 0, -math.sin(giro)], np.float32)
    else:
        c = np.asarray(en_mano, np.float32)
        e = J._norm(eje_mano)
    a, b = c - e * 62, c + e * 50
    prims = [
        J.Capsula(tuple(a), tuple(b), 20, (244, 246, 246), brillo=0.3, escala=(1, 0.66, 1) if en_mano is None else None),
        J.Capsula(tuple(c - e * 8), tuple(c + e * 18), 20.6, (66, 140, 210), brillo=0.3, escala=(1, 0.67, 1) if en_mano is None else None),
        J.Caja(tuple(a - e * 8), (21, 3.0, 7), (226, 228, 230), rot=_rot_eje(e), bisel=0.4),  # pliegue del final
        J.Cilindro(tuple(b + e * 13), 11, 12, (210, 60, 66), rot=_rot_eje_y(e), brillo=0.4),
    ]
    return prims, b + e * 25  # boquilla


def _rot_eje(e):
    """Base con la z local a lo largo de `e`."""
    e = J._norm(e)
    ref = np.array([0, 1, 0], np.float32) if abs(e[1]) < 0.9 else np.array([1, 0, 0], np.float32)
    x = J._norm(np.cross(ref, e))
    y = np.cross(e, x)
    return np.stack([x, y, e], 1)


def _rot_eje_y(e):
    """Base con la y local a lo largo de `e` (para cilindros)."""
    e = J._norm(e)
    ref = np.array([0, 0, 1], np.float32) if abs(e[2]) < 0.9 else np.array([1, 0, 0], np.float32)
    x = J._norm(np.cross(e, ref))
    z = np.cross(x, e)
    return np.stack([x, e, z], 1)


LATA = (300.0, 118.0)  # x, z sobre la encimera
LATA_H = (72.0, 27.0, 46.0)


@functools.lru_cache(maxsize=2)
def _tex_etiqueta():
    ss = 4
    w, h = 144, 54
    img = Image.new("RGBA", (w * ss, h * ss), (176, 62, 58, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([6 * ss, 6 * ss, (w - 6) * ss, (h - 6) * ss], radius=8 * ss, outline=(214, 120, 100, 255), width=2 * ss)
    d.rounded_rectangle([34 * ss, 15 * ss, (w - 34) * ss, (h - 13) * ss], radius=4 * ss, fill=(250, 244, 228, 255))
    f = fuente("manuscrita", 22 * ss)
    tw = f.getlength(GE.NOMBRE_HIJO)
    d.text(((w * ss - tw) / 2, 13 * ss), GE.NOMBRE_HIJO, font=f, fill=(150, 50, 46, 255))
    return np.asarray(img.resize((w * 2, h * 2), Image.LANCZOS), np.float32) / 255


@functools.lru_cache(maxsize=2)
def _tex_vacia():
    """Interior de la cajita todavía vacía (solo el forro)."""
    ss = 2
    w, h = 470, 300
    img = Image.new("RGBA", (w * ss, h * ss), (150, 52, 48, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([16 * ss, 16 * ss, (w - 16) * ss, (h - 16) * ss], radius=18 * ss, fill=(108, 36, 34, 255))
    d.rounded_rectangle([30 * ss, 30 * ss, (w - 30) * ss, (h - 26) * ss], radius=14 * ss, fill=(240, 226, 206, 255))
    return np.asarray(img, np.float32) / 255


@functools.lru_cache(maxsize=2)
def _tex_interior():
    """Interior de la cajita: los 20 dientes de leche, cada uno con su fecha (visto desde arriba)."""
    ss = 2
    w, h = 470, 300
    img = Image.new("RGBA", (w * ss, h * ss), (150, 52, 48, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([16 * ss, 16 * ss, (w - 16) * ss, (h - 16) * ss], radius=18 * ss, fill=(108, 36, 34, 255))  # pared interior
    d.rounded_rectangle([30 * ss, 30 * ss, (w - 30) * ss, (h - 26) * ss], radius=14 * ss, fill=(240, 226, 206, 255))  # forro
    f = fuente("manuscrita", 15 * ss)
    meses = ["mar", "may", "jun", "ago", "oct", "nov", "ene", "feb", "abr", "jun"]
    for i in range(20):
        fila, col = divmod(i, 5)
        x = (78 + col * 79) * ss
        y = (60 + fila * 58) * ss
        d.ellipse([x - 13 * ss, y - 15 * ss, x + 13 * ss, y + 11 * ss], fill=(252, 250, 242, 255), outline=(196, 186, 166, 255), width=2 * ss)
        d.ellipse([x - 6 * ss, y - 10 * ss, x + 1 * ss, y - 4 * ss], fill=(255, 255, 255, 255))
        d.text((x - 24 * ss, y + 12 * ss), f"{(i * 7) % 27 + 1} {meses[i % 10]}", font=f, fill=(90, 70, 60, 255))
    return np.asarray(img, np.float32) / 255


def lata(abierta=0.0, centro=None, rot=None, llena=True):
    """
    Cajita de lata roja con el nombre del niño. `abierta` 0–1 = ángulo de la tapa (bisagra atrás).
    Por defecto está sobre la encimera; con `centro` y `rot` se puede llevar en las manos.
    """
    hx, hy, hz = LATA_H
    if centro is None:
        centro = (LATA[0], ENCIMERA_Y - hy, LATA[1])
    c = np.asarray(centro, np.float32)
    R = np.asarray(rot, np.float32) if rot is not None else np.eye(3, dtype=np.float32)
    texturas = {"+z": _tex_etiqueta()}
    if abierta > 0.3:
        texturas["-y"] = _tex_interior() if llena else _tex_vacia()
    prims = [J.Caja(tuple(c), (hx, hy, hz), (176, 62, 58), rot=R, bisel=0.28, brillo=0.45, texturas=texturas)]
    # tapa con bisagra en el borde de atrás (−z local)
    ang = abierta * math.radians(105)
    bis = c + R @ np.array([0, -hy, -hz], np.float32)
    Rt = R @ J.rot_x(ang)
    centro_tapa = bis + Rt @ np.array([0, -5, hz + 2], np.float32)
    prims.append(J.Caja(tuple(centro_tapa), (hx + 3, 6, hz + 2), (204, 88, 78), rot=Rt, bisel=0.35, brillo=0.5))
    return prims, bis + Rt @ np.array([0, -10, 2 * hz], np.float32)  # borde delantero de la tapa (para empujarla)


# ---------------------------------------------------------------------------
# Papel: tarjetas, nota, papelito
# ---------------------------------------------------------------------------
def textura(sprite):
    """Sprite PIL RGBA → textura (array 0–1)."""
    return np.asarray(sprite.convert("RGBA"), np.float32) / 255


def tarjeta(centro, ancho, alto, tex, rot=None, color=(250, 246, 236)):
    """Tarjeta fina con su texto en la cara +z (de frente a la cámara si `rot` es la identidad)."""
    return [J.Caja(tuple(centro), (ancho / 2, alto / 2, 1.3), color, rot=rot, bisel=0.06, brillo=0.04, texturas={"+z": tex})]


def nota_en_encimera(centro, ancho, fondo, tex, giro=0.0):
    """Nota tumbada sobre la encimera; el texto se lee desde el espejo."""
    R = J.rot_y(giro)
    return [J.Caja(tuple(centro), (ancho / 2, 0.9, fondo / 2), (255, 250, 222), rot=R, bisel=0.05, brillo=0.03, texturas={"-y": tex})]


def papelito(centro, k_bola=0.0, rot=None):
    """Pañuelito de papel: plano (0) → bolita arrugada (1)."""
    c = np.asarray(centro, np.float32)
    if k_bola < 0.5:
        lado = 34 * (1 - k_bola * 0.9)
        return [J.Caja(tuple(c), (lado, 1.2 + 10 * k_bola, lado * 0.9), (246, 244, 236), rot=rot, bisel=0.3, brillo=0.02)]
    r = 13 + 8 * (1 - k_bola)
    return [J.Elipsoide(tuple(c), (r, r * 0.92, r * 0.96), (246, 244, 236), giro=20, brillo=0.02)]


# ---------------------------------------------------------------------------
# Otros
# ---------------------------------------------------------------------------
def lapiz(punta, eje, color=(70, 90, 170)):
    """Bolígrafo: `punta` es la punta que toca el papel y `eje` va hacia el otro extremo."""
    p = np.asarray(punta, np.float32)
    e = J._norm(eje)
    return [
        J.Capsula(tuple(p + e * 3), tuple(p + e * 12), 2.4, (60, 60, 66), brillo=0.4),
        J.Capsula(tuple(p + e * 12), tuple(p + e * 130), 4.6, color, brillo=0.35),
        J.Capsula(tuple(p + e * 112), tuple(p + e * 132), 4.9, (232, 232, 236), brillo=0.4),
    ]


def moneda(centro, inclinacion=0.0):
    R = J.rot_x(inclinacion)
    return [J.Cilindro(tuple(centro), 12.5, 1.8, (226, 184, 76), rot=R, brillo=0.9)]


def diente(centro):
    return [J.Elipsoide(tuple(centro), (6.5, 8, 6), (252, 250, 242), brillo=0.5)]


def telefono(centro, rot):
    """Celular en la mano: la pantalla mira hacia −z local (hacia quien lo usa); vemos la parte de atrás."""
    c = np.asarray(centro, np.float32)
    R = np.asarray(rot, np.float32)
    prims = [J.Caja(tuple(c), (34, 70, 4.5), (40, 42, 50), rot=R, bisel=0.45, brillo=0.5)]
    prims.append(J.Caja(tuple(c + R @ np.array([-17, -50, 4.5], np.float32)), (9, 12, 1.6), (24, 24, 28), rot=R, bisel=0.5, brillo=0.7))
    return prims
