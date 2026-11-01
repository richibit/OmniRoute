"""
El baño visto desde el espejo para «La última vez que tu mamá te cepilló los dientes».

Una sola cámara en perspectiva (`CAM`) para todo: la pared del fondo con la
puerta y el pasillo (pintada), la encimera con el lavabo (calculada rayo a
rayo con la misma cámara, así que lo que queda detrás o dentro del lavabo se
oculta de verdad), el grifo cromado y la utilería que está encima.
Medidas en píxeles del mundo (1 px = 1 px de pantalla en el plano z = 0).
"""

import functools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import juguete3d as J
from utilidades import gradiente_radial

W, H = 1080, 1920
# Encuadre base: un poco más cerrado que el campo de la cámara (los personajes llenan el plano).
CAM = J.Camara(D=800.0, cx=540.0, cy=640.0, zoom=1.12, foco=(540.0, 820.0))

SUELO = 1985.0  # y del suelo
ENCIMERA_Y = 1225.0  # superficie de la encimera (a la altura de la cadera de un adulto)
ENCIMERA_Z0 = 40.0  # borde delantero de la encimera (lado de quien se cepilla)
ESPEJO_Z = 480.0  # pared del espejo (fondo de la encimera)
PARED_Z = -800.0  # pared del fondo, con la puerta
LAVABO = {"x": 548.0, "z": 262.0, "rx": 190.0, "rz": 102.0, "hondo": 100.0}
GRIFO = {"x": 548.0, "z": 400.0, "alto": 126.0, "r": 23.0}  # monomando justo detrás del lavabo, al centro
SALIDA_AGUA = (606.0, ENCIMERA_Y - 104.0, 288.0)  # boca del caño (girado un poco a la derecha): el agua cae dentro del lavabo y se ve junto al grifo

# Pintura de la pared del fondo (en píxeles de pantalla)
PISO = CAM.cy + (SUELO - CAM.cy) * CAM.D / (CAM.D - PARED_Z)  # unión suelo-pared al fondo (plano de imagen sin zoom)
PUERTA = (752, 468, 978)  # x0, y0 (dintel), x1 del vano de la puerta


def a_pantalla(p, cam=CAM):
    X, Y = cam.proyectar(p[0], p[1], p[2])
    return float(X), float(Y)


def y_borde_encimera(cam=CAM):
    return a_pantalla((cam.cx, ENCIMERA_Y, ENCIMERA_Z0), cam)[1]


# ---------------------------------------------------------------------------
# Oclusión: encimera y lavabo
# ---------------------------------------------------------------------------
def _t_lavabo_interior(O, dx, dy, dz):
    """Distancia a la cara interior del lavabo (media elipsoide bajo la encimera), por rayo."""
    L = LAVABO
    c = np.array([L["x"], ENCIMERA_Y, L["z"]], np.float32)
    r = np.array([L["rx"], L["hondo"], L["rz"]], np.float32)
    q = (O - c) / r
    ex, ey, ez = dx / r[0], dy / r[1], dz / r[2]
    a = ex * ex + ey * ey + ez * ez
    b = q[0] * ex + q[1] * ey + q[2] * ez
    cc = float(q @ q) - 1.0
    disc = b * b - a * cc
    return np.where(disc > 0, (-b + np.sqrt(np.clip(disc, 0, None))) / a, np.inf)


def oclusor(O, dx, dy, dz):
    """t de la encimera (o del fondo del lavabo) para cada rayo; inf si el rayo no la toca."""
    t = np.full(np.shape(dx), np.inf, np.float32)
    baja = dy > 1e-6
    tp = np.where(baja, (ENCIMERA_Y - O[1]) / np.where(baja, dy, 1.0), np.inf)
    x = O[0] + tp * dx
    z = O[2] + tp * dz
    sobre = baja & (z >= ENCIMERA_Z0) & (z <= ESPEJO_Z + 400)
    L = LAVABO
    dentro = ((x - L["x"]) / L["rx"]) ** 2 + ((z - L["z"]) / L["rz"]) ** 2 < 1.0
    t = np.where(sobre & ~dentro, tp, t)
    if (sobre & dentro).any():
        tl = _t_lavabo_interior(O, dx, dy, dz)
        t = np.where(sobre & dentro, np.maximum(tl, tp), t)
    return t


# ---------------------------------------------------------------------------
# Pared del fondo (pintada)
# ---------------------------------------------------------------------------
def _luz_mapa(noche):
    """Mapa de iluminación del decorado (multiplicador por píxel)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if noche:  # lámpara sobre el espejo (arriba, hacia la cámara) + resplandor del pasillo
        base = 0.40 + 0.78 * np.exp(-(((xx - 520) / 700) ** 2 + ((yy - 560) / 900) ** 2))
        base += 0.5 * np.exp(-(((xx - 870) / 130) ** 2 + ((yy - 900) / 320) ** 2))
    else:  # luz de la ventana (izquierda)
        base = 0.62 + 0.5 * np.exp(-(((xx - 280) / 560) ** 2 + ((yy - 560) / 760) ** 2))
    return base[..., None]


@functools.lru_cache(maxsize=2)
def fondo_bano(nombre):
    """Pared del fondo con ventana, toalla, aplique y la puerta abierta al pasillo (pintada en el plano de imagen y encuadrada con CAM)."""
    noche = nombre == "noche"
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    piso = int(round(PISO))
    bx0, bx1, by0, zocalo = 120, 1000, 120, 930
    pared = (190, 142, 104) if noche else (204, 208, 196)
    azulejo = (230, 220, 202) if noche else (232, 238, 236)
    junta = (188, 172, 150) if noche else (190, 200, 198)
    d.polygon([(0, 0), (W, 0), (bx1, by0), (bx0, by0)], fill=(140, 104, 78) if noche else (182, 186, 180))  # techo
    d.polygon([(0, 0), (bx0, by0), (bx0, piso), (0, H)], fill=tuple(int(c * 0.8) for c in pared))
    d.polygon([(W, 0), (bx1, by0), (bx1, piso), (W, H)], fill=tuple(int(c * 0.74) for c in pared))
    d.polygon([(bx0, piso), (bx1, piso), (W, H), (0, H)], fill=(118, 94, 78) if noche else (160, 160, 152))
    d.rectangle([bx0, by0, bx1, zocalo], fill=pared)
    d.rectangle([bx0, zocalo, bx1, piso], fill=azulejo)
    for y in range(zocalo, piso, 52):
        d.line([(bx0, y), (bx1, y)], fill=junta, width=3)
    for x in range(bx0, bx1, 52):
        d.line([(x, zocalo), (x, piso)], fill=junta, width=3)
    for lado, x_borde in ((-1, bx0), (1, bx1)):
        ext = 0 if lado < 0 else W
        d.polygon([(x_borde, zocalo), (x_borde, piso), (ext, H), (ext, zocalo + 40)], fill=tuple(int(c * (0.84 if lado < 0 else 0.78)) for c in azulejo))
    d.rectangle([bx0, zocalo - 8, bx1, zocalo + 6], fill=tuple(int(c * 0.9) for c in azulejo))
    # ventana (izquierda)
    wx0, wy0, wx1, wy1 = 170, 330, 380, 640
    marco = (236, 230, 220) if noche else (244, 246, 246)
    d.rectangle([wx0 - 14, wy0 - 14, wx1 + 14, wy1 + 14], fill=marco)
    d.rectangle([wx0, wy0, wx1, wy1], fill=(58, 66, 92) if noche else (226, 240, 252))
    d.line([((wx0 + wx1) / 2, wy0), ((wx0 + wx1) / 2, wy1)], fill=marco, width=10)
    d.line([(wx0, (wy0 + wy1) / 2), (wx1, (wy0 + wy1) / 2)], fill=marco, width=10)
    # aplique de luz en la pared
    d.rounded_rectangle([615, 470, 685, 550], radius=26, fill=(255, 236, 196) if noche else (236, 232, 222))
    d.rectangle([642, 550, 658, 580], fill=(150, 130, 110))
    # toallero con toalla a rayas
    d.rounded_rectangle([545, 760, 695, 772], radius=6, fill=(188, 188, 194))
    d.rounded_rectangle([560, 768, 680, 1010], radius=20, fill=(222, 110, 96) if noche else (212, 122, 112))
    for y in (840, 880):
        d.rectangle([560, y, 680, y + 14], fill=(244, 230, 214))
    # puerta (derecha) con el pasillo iluminado
    px0, py0, px1 = PUERTA
    d.rectangle([px0 - 18, py0 - 18, px1 + 18, piso], fill=(244, 236, 224) if noche else (238, 240, 238))
    alto = piso - py0
    pasillo = gradiente_radial(px1 - px0, alto, ((px1 - px0) / 2, 200), 600, (255, 212, 150) if noche else (220, 216, 206), (140, 90, 56) if noche else (150, 150, 144))
    dp = ImageDraw.Draw(pasillo)
    dp.rectangle([46, 150, 150, 250], outline=(120, 80, 50) if noche else (130, 126, 120), width=6)  # cuadro del pasillo
    dp.polygon([(0, alto - 120), (px1 - px0, alto - 120), (px1 - px0, alto), (0, alto)], fill=(166, 106, 68) if noche else (156, 150, 140))
    img.paste(pasillo.convert("RGB").filter(ImageFilter.GaussianBlur(3)), (px0, py0))
    arr = np.asarray(img, np.float32) * _luz_mapa(noche)
    if noche:
        arr += np.asarray(gradiente_radial(W, H, (870, 960), 320, (60, 34, 12), (0, 0, 0)).convert("RGB"), np.float32)
        arr += np.asarray(gradiente_radial(W, H, (650, 510), 220, (90, 60, 26), (0, 0, 0)).convert("RGB"), np.float32)
    else:
        rayos = Image.new("L", (W, H), 0)
        ImageDraw.Draw(rayos).polygon([(wx0, wy0), (wx1, wy0 + 40), (920, 1500), (320, 1500)], fill=52)
        arr += np.asarray(rayos.filter(ImageFilter.GaussianBlur(40)), np.float32)[..., None] * np.array([1.0, 0.97, 0.9])
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(1.2))
    return acercar(out.convert("RGBA"), CAM.zoom, CAM.foco)


def acercar(img, zoom, foco):
    """Acercamiento de una capa de pantalla (zoom alrededor de `foco`)."""
    if abs(zoom - 1.0) < 1e-4:
        return img.copy()
    fx, fy = foco
    return img.transform(img.size, Image.AFFINE, (1 / zoom, 0, fx - fx / zoom, 0, 1 / zoom, fy - fy / zoom), resample=Image.BICUBIC)


# ---------------------------------------------------------------------------
# Encimera y lavabo (rayo a rayo, con la cámara de la escena)
# ---------------------------------------------------------------------------
def _sombrear(n, vista, albedo, luz, brillo, ao):
    dif_p = np.clip(n @ luz.principal * 0.5 + 0.5, 0, 1) ** 1.8
    dif_r = np.clip(n @ luz.relleno * 0.5 + 0.5, 0, 1) ** 2.0
    medio = luz.principal[None, :] + vista
    medio /= np.linalg.norm(medio, axis=-1, keepdims=True) + 1e-9
    esp = np.clip((n * medio).sum(-1), 0, 1) ** 40 * brillo
    luzc = luz.ambiente[None, :] + dif_p[:, None] * luz.color_principal + dif_r[:, None] * luz.color_relleno
    rgb = albedo * luzc * (1 - ao[:, None]) + esp[:, None] * 0.5
    rgb = rgb / (1 + rgb * 0.25)
    return np.clip(rgb ** (1 / 1.12), 0, 1)


@functools.lru_cache(maxsize=4)
def capa_encimera(nombre, ss=2):
    """Encimera con el lavabo en primer plano (RGBA de pantalla completa, transparente arriba del borde)."""
    noche = nombre == "noche"
    luz = J.NOCHE if noche else J.MANANA
    cam = CAM
    y0 = int(y_borde_encimera(cam)) - 4
    hh = H - y0
    Yg, Xg = np.mgrid[0 : hh * ss, 0 : W * ss].astype(np.float32)
    X = Xg / ss + 0.5 / ss
    Y = Yg / ss + y0 + 0.5 / ss
    O = cam.origen
    dx, dy, dz = cam.direcciones(X, Y)
    dx, dy, dz = dx.ravel(), dy.ravel(), dz.ravel()
    tp = np.where(dy > 1e-6, (ENCIMERA_Y - O[1]) / np.maximum(dy, 1e-6), np.inf)
    px, pz = O[0] + tp * dx, O[2] + tp * dz
    sobre = np.isfinite(tp) & (pz >= ENCIMERA_Z0)
    L = LAVABO
    e = np.sqrt(((px - L["x"]) / L["rx"]) ** 2 + ((pz - L["z"]) / L["rz"]) ** 2)
    dentro = sobre & (e < 1.0)
    vista = -np.stack([dx, dy, dz], -1)
    vista /= np.linalg.norm(vista, axis=-1, keepdims=True)
    rgb = np.zeros((dx.size, 3), np.float32)
    # superficie de la encimera
    sup = np.array((236, 228, 216) if noche else (226, 228, 224), np.float32) / 255
    m = sobre & ~dentro
    n = np.tile(np.array([0, -1, 0], np.float32), (int(m.sum()), 1))
    borde_lav = np.clip(1 - (e[m] - 1.0) / 0.08, 0, 1)  # reborde de cerámica alrededor del lavabo
    alb = np.where(borde_lav[:, None] > 0, sup * (1 - borde_lav[:, None]) + np.array([0.97, 0.97, 0.96]) * borde_lav[:, None], sup)
    ao = 0.10 * np.clip(1 - (pz[m] - ENCIMERA_Z0) / 30, 0, 1)  # junto al borde delantero
    ao += 0.18 * np.clip(1 - (e[m] - 1.06) / 0.25, 0, 1) * (borde_lav <= 0)  # sombra suave alrededor del reborde
    if noche:  # la lámpara sobre el espejo ilumina más el fondo de la encimera
        caida = 0.82 + 0.25 * np.clip((pz[m] - ENCIMERA_Z0) / (ESPEJO_Z - ENCIMERA_Z0), 0, 1)
    else:
        caida = 0.9 + 0.18 * np.clip(1 - (px[m] - 100) / 900, 0, 1)
    rgb[m] = _sombrear(n, vista[m], alb, luz, 0.25 + 0.5 * borde_lav, ao) * caida[:, None]
    # interior del lavabo (cerámica blanca) con desagüe
    if dentro.any():
        tl = _t_lavabo_interior(O, dx[dentro], dy[dentro], dz[dentro])
        qx = O[0] + tl * dx[dentro] - L["x"]
        qy = O[1] + tl * dy[dentro] - ENCIMERA_Y
        qz = O[2] + tl * dz[dentro] - L["z"]
        n = -np.stack([qx / L["rx"] ** 2, qy / L["hondo"] ** 2, qz / L["rz"] ** 2], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        cer = np.array([0.95, 0.95, 0.95], np.float32)
        r_des = np.sqrt((qx / 34) ** 2 + (qz / 22) ** 2)
        alb = np.where((r_des < 1.0)[:, None], np.array([0.42, 0.44, 0.48]), cer)
        alb = np.where(((r_des >= 1.0) & (r_des < 1.25))[:, None], np.array([0.78, 0.80, 0.84]), alb)
        prof = np.clip(qy / L["hondo"], 0, 1)
        ao = 0.28 * prof + 0.12 * np.clip(1 - e[dentro] / 0.2, 0, 1)
        rgb[dentro] = _sombrear(n, vista[dentro], alb, luz, 0.6, ao) * 0.98
    # brillo del canto delantero de la encimera
    canto = sobre & (pz < ENCIMERA_Z0 + 7)
    rgb[canto] = np.clip(rgb[canto] * 1.12 + 0.03, 0, 1)
    arr = np.zeros((hh * ss, W * ss, 4), np.uint8)
    arr[..., :3] = (rgb.reshape(hh * ss, W * ss, 3) * 255).astype(np.uint8)
    arr[..., 3] = (sobre.reshape(hh * ss, W * ss) * 255).astype(np.uint8)
    capa = Image.fromarray(arr, "RGBA").resize((W, hh), Image.LANCZOS)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(capa, (0, y0))
    return out


# ---------------------------------------------------------------------------
# Grifo (cromado, 3D: puede tapar una mano que pasa por detrás)
# ---------------------------------------------------------------------------
def grifo(abierto=0.0):
    """Grifo monomando cromado detrás del lavabo: cuerpo, caño hacia el lavabo y palanca arriba (`abierto` 0–1 la levanta)."""
    g = GRIFO
    x, z, h, r = g["x"], g["z"], g["alto"], g["r"]
    yb = ENCIMERA_Y
    sx, sy, sz = SALIDA_AGUA
    cromo = (200, 206, 216)
    piv = np.array([x, yb - h - 8, z + 4], np.float32)
    punta = punta_palanca(abierto)
    largo = math.hypot(sx - x, sz - z)
    ux, uz = (sx - x) / largo, (sz - z) / largo  # dirección del caño en planta
    prims = [
        J.Cilindro((x, yb - 4, z), r + 12, 4, cromo, brillo=1.0),  # embellecedor sobre la encimera
        J.Cilindro((x, yb - h / 2, z), r, h / 2, cromo, brillo=1.0),  # cuerpo
        J.Elipsoide((x, yb - h, z), (r, 8, r), cromo, brillo=1.0),
        J.Capsula((x + ux * r * 0.6, yb - h + 30, z + uz * r * 0.6), (sx - ux * 16, yb - h + 27, sz - uz * 16), 13, cromo, brillo=1.0),  # caño
        J.Capsula((sx - ux * 16, yb - h + 27, sz - uz * 16), (sx, sy - 10, sz), 12, cromo, brillo=1.0),
        J.Cilindro((sx, sy - 5, sz), 13.5, 5, cromo, brillo=1.0),  # aireador
        J.Cilindro((sx, sy - 0.5, sz), 9, 1.0, (96, 100, 108), brillo=0.6),
        J.Elipsoide(tuple(piv), (15, 10, 15), cromo, brillo=1.0),  # rótula de la palanca
        J.Capsula(tuple(piv), tuple(punta), 8, cromo, brillo=1.0),  # palanca (apunta hacia quien la usa)
    ]
    for i, p in enumerate(prims):
        p.metal = True
        p.dueno, p.grupo, p.es_objeto = "grifo", ("palanca" if i >= len(prims) - 2 else "objeto"), True
    return prims


def punta_palanca(abierto=0.0):
    """Extremo de la palanca del grifo (donde la empuja un dedo)."""
    g = GRIFO
    ang = 0.32 + 0.5 * abierto  # cerrada apenas levantada (deja sitio a los dedos sobre el caño); abierta, arriba
    piv = np.array([g["x"], ENCIMERA_Y - g["alto"] - 8, g["z"] + 4], np.float32)
    return piv + np.array([0, -math.sin(ang) * 62, -math.cos(ang) * 62], np.float32)


def chorro(t, intensidad=1.0, halo=False):
    """
    Chorro de agua (cápsulas que ondulan), para renderizar aparte con transparencia. Con `halo`,
    el contorno azulado que lo separa del lavabo claro y la salpicadura donde cae.
    """
    if intensidad <= 0.01:
        return []
    x, y, z = SALIDA_AGUA
    prims = []
    fondo = ENCIMERA_Y + 86
    for k in range(3):
        off = math.sin(t * 31 + k * 2.1) * 1.6
        r = 7.0 - k * 1.5 + (2.4 if halo else 0.0)
        color = (118, 158, 206) if halo else (228, 241, 255)
        prims.append(J.Capsula((x + off + (k - 1) * 3, y + 6, z), (x + off * 2 + (k - 1) * 4, fondo, z - 6), r, color, brillo=1.0))
    if halo:  # la salpicadura: un anillo claro que tiembla sobre el fondo del lavabo
        w = 1.0 + 0.12 * math.sin(t * 23)
        prims.append(J.Elipsoide((x + 4, fondo + 5, z - 6), (26 * w, 3.0, 17 * w), (204, 226, 250), brillo=0.8))
    return prims
