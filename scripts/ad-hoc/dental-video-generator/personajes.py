"""
Personajes tipo juguete (cabeza grande, cuerpo de cápsulas, cara pintada) para
«La última vez que tu mamá te cepilló los dientes»: mamá (26 → 58 años),
hijo (8 meses → 32 años) y nieta (3 años).

Cada personaje se construye con primitivas de `juguete3d` a partir de la
posición de su cabeza y de una pose: manos en un punto (cinemática inversa de
dos huesos), dedos para contar, giro de cabeza y expresión de la cara.
"""

import functools
import math

import numpy as np
from PIL import Image, ImageDraw

from juguete3d import Capsula, Elipsoide

PIEL = (238, 190, 152)

# Proporciones relativas al radio de la cabeza R (en píxeles a escala 1).
TIPOS = {
    "adulto": {"R": 74, "torso": 2.9, "ancho": 1.06, "brazo": (1.32, 1.22), "grosor": 0.30, "mano": 0.30, "pierna": 3.2},
    "adolescente": {"R": 72, "torso": 2.75, "ancho": 1.0, "brazo": (1.28, 1.18), "grosor": 0.29, "mano": 0.29, "pierna": 3.0},
    "nino": {"R": 64, "torso": 1.95, "ancho": 0.92, "brazo": (1.02, 0.94), "grosor": 0.30, "mano": 0.31, "pierna": 2.2},
    "pequeno": {"R": 60, "torso": 1.55, "ancho": 0.88, "brazo": (0.86, 0.78), "grosor": 0.32, "mano": 0.32, "pierna": 1.7},
    "bebe": {"R": 56, "torso": 1.25, "ancho": 0.92, "brazo": (0.70, 0.64), "grosor": 0.34, "mano": 0.34, "pierna": 1.2},
}


# ---------------------------------------------------------------------------
# Caras (texturas pintadas que se proyectan sobre la cabeza)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def textura_cara(expresion="neutral", hueco=False, rubor=0.5, arrugas=False):
    """
    Cara frontal en un lienzo cuadrado (centro = frente de la cabeza).
    Expresiones: neutral, sonrisa, dientes, apretada, mueca, dolor, feliz (ojos cerrados),
    abierta (boca abierta para cepillarse), sorpresa, parpadeo, tapada.
    """
    S = 512
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tinta = (34, 22, 18, 255)
    ojo_y = 236
    for sx in (-1, 1):
        ex = 256 + sx * 70
        if expresion in ("feliz", "tapada"):
            d.arc([ex - 26, ojo_y - 16, ex + 26, ojo_y + 26], 200, 340, fill=tinta, width=9)
        elif expresion == "parpadeo":
            d.line([(ex - 22, ojo_y + 4), (ex + 22, ojo_y + 4)], fill=tinta, width=8)
        elif expresion == "dolor":
            d.line([(ex - 20, ojo_y - 12), (ex + 18, ojo_y + 2)] if sx < 0 else [(ex + 20, ojo_y - 12), (ex - 18, ojo_y + 2)], fill=tinta, width=8)
            d.line([(ex - 20, ojo_y + 14), (ex + 18, ojo_y + 2)] if sx < 0 else [(ex + 20, ojo_y + 14), (ex - 18, ojo_y + 2)], fill=tinta, width=8)
        elif expresion == "mueca" and sx > 0:
            d.arc([ex - 24, ojo_y - 10, ex + 24, ojo_y + 20], 200, 340, fill=tinta, width=9)  # guiño
        else:
            alto = 34 if expresion == "sorpresa" else 30
            d.ellipse([ex - 15, ojo_y - alto / 2, ex + 15, ojo_y + alto / 2], fill=tinta)
            d.ellipse([ex - 7, ojo_y - alto / 2 + 5, ex + 1, ojo_y - alto / 2 + 13], fill=(255, 255, 255, 230))
    if expresion in ("mueca", "sorpresa", "dolor"):  # cejas expresivas
        for sx in (-1, 1):
            ex = 256 + sx * 70
            d.arc([ex - 30, ojo_y - 62, ex + 30, ojo_y - 22], 200, 340, fill=tinta, width=7)
    if rubor:
        for sx in (-1, 1):
            capa = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            ImageDraw.Draw(capa).ellipse([256 + sx * 112 - 34, 300 - 20, 256 + sx * 112 + 34, 300 + 20], fill=(236, 120, 110, int(110 * rubor)))
            img.alpha_composite(capa)
    boca_y = 330
    labio = (150, 60, 58, 255)
    if expresion in ("sonrisa", "feliz"):
        d.arc([256 - 46, boca_y - 34, 256 + 46, boca_y + 18], 20, 160, fill=labio, width=9)
    elif expresion in ("dientes", "tapada"):
        d.chord([256 - 62, boca_y - 30, 256 + 62, boca_y + 46], 0, 180, fill=(110, 34, 40, 255))
        d.rectangle([256 - 54, boca_y + 8, 256 + 54, boca_y + 24], fill=(252, 250, 244, 255))
        if hueco:
            d.rectangle([256 - 8, boca_y + 8, 256 + 10, boca_y + 24], fill=(70, 20, 26, 255))
    elif expresion == "apretada":
        d.line([(256 - 40, boca_y + 4), (256 - 14, boca_y - 4), (256 + 14, boca_y + 4), (256 + 40, boca_y - 4)], fill=labio, width=9, joint="curve")
    elif expresion == "mueca":
        d.ellipse([256 - 30, boca_y - 20, 256 + 30, boca_y + 34], fill=(110, 34, 40, 255))
        d.ellipse([256 - 20, boca_y + 6, 256 + 20, boca_y + 40], fill=(236, 110, 120, 255))  # lengua
    elif expresion == "abierta":
        d.ellipse([256 - 34, boca_y - 16, 256 + 34, boca_y + 36], fill=(110, 34, 40, 255))
        d.rectangle([256 - 26, boca_y - 12, 256 + 26, boca_y], fill=(252, 250, 244, 255))
    elif expresion == "sorpresa":
        d.ellipse([256 - 18, boca_y - 10, 256 + 18, boca_y + 26], fill=(110, 34, 40, 255))
    elif expresion == "dolor":
        d.line([(256 - 36, boca_y + 10), (256 - 12, boca_y), (256 + 12, boca_y + 10), (256 + 36, boca_y)], fill=labio, width=8, joint="curve")
    else:
        d.line([(256 - 24, boca_y), (256 + 24, boca_y)], fill=labio, width=8)
    if arrugas:
        for sx in (-1, 1):
            ex = 256 + sx * 70
            d.arc([ex + sx * 34 - 14, ojo_y - 8, ex + sx * 34 + 14, ojo_y + 20], 300 if sx > 0 else 120, 60 if sx > 0 else 240, fill=(120, 80, 64, 140), width=4)
        d.arc([256 - 80, boca_y - 70, 256 - 40, boca_y + 10], 100, 170, fill=(120, 80, 64, 120), width=4)
        d.arc([256 + 40, boca_y - 70, 256 + 80, boca_y + 10], 10, 80, fill=(120, 80, 64, 120), width=4)
    img = img.resize((256, 256), Image.LANCZOS)
    return np.asarray(img, np.float32) / 255.0


# ---------------------------------------------------------------------------
# Cinemática inversa de dos huesos
# ---------------------------------------------------------------------------
def ik(hombro, objetivo, l1, l2, polo):
    """Codo y muñeca para llevar la mano a `objetivo` (si no llega, estira el brazo hacia él)."""
    S = np.asarray(hombro, np.float32)
    T = np.asarray(objetivo, np.float32)
    v = T - S
    d = float(np.linalg.norm(v))
    d = min(max(d, abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    u = v / (np.linalg.norm(v) + 1e-9)
    T = S + u * d
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    p = np.asarray(polo, np.float32)
    p = p - u * float(p @ u)
    p = p / (np.linalg.norm(p) + 1e-9)
    codo = S + u * a + p * h
    return codo, T


# ---------------------------------------------------------------------------
# Personaje
# ---------------------------------------------------------------------------
class Personaje:
    """
    tipo: adulto | adolescente | nino | pequeno | bebe
    ropa: color de la prenda superior · pantalon: color de abajo
    pelo: color · estilo: mono | corto | bebe | coletas | capucha
    """

    def __init__(self, tipo, ropa, pantalon, pelo, estilo, piel=PIEL, arrugas=False, pegatina=None):
        self.p = TIPOS[tipo]
        self.tipo = tipo
        self.ropa = ropa
        self.pantalon = pantalon
        self.pelo = pelo
        self.estilo = estilo
        self.piel = piel
        self.arrugas = arrugas
        self.pegatina = pegatina

    def construir(self, cx, cy, pose=None, escala=1.0, z0=0.0):
        """
        Primitivas del personaje con la cabeza centrada en (cx, cy) y profundidad z0.
        Devuelve (primitivas, info) — info tiene cabeza, manos y hombros en pantalla.
        """
        pose = pose or {}
        P = self.p
        R = P["R"] * escala
        prims = []
        giro = pose.get("giro", 0.0)  # -1 mira a la izquierda de la pantalla, +1 a la derecha
        cabeceo = pose.get("cabeceo", 0.0)
        cuerpo_dx = pose.get("cuerpo_dx", 0.0) * R
        # ---- cabeza
        cabeza = Elipsoide((cx, cy, z0), (R, R * 1.02, R * 0.96), self.piel, brillo=0.06)
        cabeza.textura = textura_cara(pose.get("cara", "neutral"), pose.get("hueco", False), 0.5, self.arrugas)
        cabeza.giro_textura = giro * 0.6
        cabeza.cabeceo_textura = cabeceo * 0.5
        prims.append(cabeza)
        for sx in (-1, 1):  # orejas
            prims.append(Elipsoide((cx + sx * R * 0.97 - giro * R * 0.25, cy + R * 0.05, z0 - R * 0.12), (R * 0.15, R * 0.22, R * 0.12), self.piel))
        dz = -giro * R * 0.1
        if self.estilo in ("mono", "corto", "coletas"):
            prims.append(Elipsoide((cx - giro * R * 0.12, cy - R * 0.2, z0 - R * 0.14), (R * 1.05, R * 0.93, R * 0.96), self.pelo, brillo=0.12))
            prims.append(Elipsoide((cx + R * 0.18 - giro * R * 0.35, cy - R * 0.62, z0 + R * 0.42 + dz), (R * 0.72, R * 0.26, R * 0.46), self.pelo, giro=-12, brillo=0.12))
        if self.estilo == "mono":
            prims.append(Elipsoide((cx - giro * R * 0.3, cy - R * 1.02, z0 - R * 0.35), (R * 0.42, R * 0.36, R * 0.38), self.pelo, brillo=0.12))
            for sx in (-1, 1):  # mechones laterales hasta la mandíbula
                prims.append(Elipsoide((cx + sx * R * 0.9 - giro * R * 0.2, cy + R * 0.12, z0 - R * 0.28), (R * 0.26, R * 0.62, R * 0.4), self.pelo, brillo=0.12))
        elif self.estilo == "coletas":
            for sx in (-1, 1):
                prims.append(Elipsoide((cx + sx * R * 1.08, cy - R * 0.18, z0 - R * 0.3), (R * 0.32, R * 0.42, R * 0.32), self.pelo, giro=sx * 25, brillo=0.12))
        elif self.estilo == "bebe":
            prims.append(Elipsoide((cx + R * 0.1, cy - R * 0.94, z0 + R * 0.08), (R * 0.26, R * 0.14, R * 0.2), self.pelo, giro=-20))
        elif self.estilo == "capucha":
            prims.append(Elipsoide((cx - giro * R * 0.12, cy - R * 0.2, z0 - R * 0.14), (R * 1.04, R * 0.92, R * 0.96), self.pelo, brillo=0.12))
            prims.append(Elipsoide((cx + R * 0.15 - giro * R * 0.35, cy - R * 0.6, z0 + R * 0.4 + dz), (R * 0.7, R * 0.25, R * 0.45), self.pelo, giro=-10, brillo=0.12))
        # ---- cuello, torso y cadera
        xb = cx + cuerpo_dx
        prims.append(Capsula((cx, cy + R * 0.8, z0 - R * 0.3), (xb, cy + R * 1.25, z0 - R * 0.3), R * 0.3, self.piel))
        top = cy + R * 1.55
        bot = top + P["torso"] * R
        ancho = P["ancho"] * R
        prims.append(Capsula((xb, top, z0 - R * 0.42), (xb, bot, z0 - R * 0.42), ancho, self.ropa, aplanado=0.62, brillo=0.04))
        if self.estilo == "capucha":  # capucha caída sobre los hombros
            prims.append(Capsula((xb - ancho * 0.6, top - R * 0.15, z0 - R * 0.75), (xb + ancho * 0.6, top - R * 0.15, z0 - R * 0.75), R * 0.42, self.ropa))
        if pose.get("piernas", "ocultas") != "ninguna":
            fondo_cadera = max(bot + R * 1.1, pose.get("hasta_y", 0))
            prims.append(Capsula((xb, bot - R * 0.2, z0 - R * 0.45), (xb, fondo_cadera, z0 - R * 0.45), ancho * 0.95, self.pantalon, aplanado=0.62, brillo=0.03))
        piernas = pose.get("piernas", "ocultas")
        if piernas in ("de_pie", "caminando"):
            fase = pose.get("paso", 0.0)
            for sx in (-1, 1):
                swing = math.sin(fase) * sx * R * 0.45 if piernas == "caminando" else 0.0
                cadera = (xb + sx * ancho * 0.42, bot + R * 0.6, z0 - R * 0.45)
                pie = (xb + sx * ancho * 0.45, bot + R * (0.6 + P["pierna"]), z0 - R * 0.45 + swing)
                prims.append(Capsula(cadera, pie, R * 0.36, self.pantalon, brillo=0.03))
                prims.append(Elipsoide((pie[0], pie[1] + R * 0.18, pie[2] + R * 0.22), (R * 0.38, R * 0.2, R * 0.5), (60, 46, 40)))
        elif piernas == "sentado":  # bebé sentado en la encimera: piernas estiradas hacia la cámara, en V
            for sx in (-1, 1):
                cadera = (xb + sx * ancho * 0.4, bot + R * 0.35, z0 - R * 0.1)
                rodilla = (xb + sx * ancho * 0.95, bot + R * 0.75, z0 + R * 1.3)
                pie = (xb + sx * ancho * 1.15, bot + R * 1.1, z0 + R * 2.3)
                prims.append(Capsula(cadera, rodilla, R * 0.46, self.pantalon))
                prims.append(Capsula(rodilla, pie, R * 0.4, self.pantalon))
                prims.append(Elipsoide((pie[0], pie[1] + R * 0.1, pie[2] + R * 0.2), (R * 0.34, R * 0.3, R * 0.4), self.piel))
        if self.pegatina:  # pegatina «paciente valiente» en el pecho
            prims.append(Elipsoide((xb + ancho * 0.35, top + R * 0.75, z0 - R * 0.42 + ancho * 0.62 * 0.98), (R * 0.26, R * 0.26, R * 0.05), self.pegatina, brillo=0.25))
        # ---- brazos (cinemática inversa hacia la mano objetivo)
        l1, l2 = P["brazo"][0] * R, P["brazo"][1] * R
        info = {"cabeza": (cx, cy), "R": R, "giro": giro, "manos": {}, "hombros": {}}
        for q in prims:
            q.parte = "cuerpo"
        n_antes = len(prims)
        for lado, sx in (("izq", -1), ("der", 1)):
            hombro = (xb + sx * ancho * 0.88, top + R * 0.28, z0 - R * 0.35)
            objetivo = pose.get(f"mano_{lado}")
            if objetivo is None:
                objetivo = (hombro[0] + sx * R * 0.18, hombro[1] + (l1 + l2) * 0.96, hombro[2] + R * 0.1)
            elif len(objetivo) == 2:
                objetivo = (objetivo[0], objetivo[1], z0 + R * 0.6)
            polo = pose.get(f"polo_{lado}", (sx * 1.0, 0.8, -0.6))
            codo, muneca = ik(hombro, objetivo, l1, l2, polo)
            prims.append(Capsula(hombro, codo, P["grosor"] * R, self.ropa, brillo=0.04))
            prims.append(Capsula(codo, muneca, P["grosor"] * R * 0.92, self.ropa, brillo=0.04))
            dirm = (muneca - codo) / (np.linalg.norm(muneca - codo) + 1e-6)
            mano = muneca + dirm * P["mano"] * R * 0.6
            prims.append(Elipsoide(tuple(mano), (P["mano"] * R, P["mano"] * R * 1.08, P["mano"] * R * 0.85), self.piel))
            dedos = pose.get(f"dedos_{lado}")
            if dedos:  # contar con los dedos: dedos levantados sobre la mano
                for k in range(int(dedos)):
                    fx = mano[0] + (k - (dedos - 1) / 2) * P["mano"] * R * 0.55
                    prims.append(Capsula((fx, mano[1] - P["mano"] * R * 0.5, mano[2] + R * 0.05), (fx, mano[1] - P["mano"] * R * 1.6, mano[2] + R * 0.08), P["mano"] * R * 0.22, self.piel))
            info["manos"][lado] = tuple(float(v) for v in mano)
            info["hombros"][lado] = hombro
            for q in prims[n_antes:]:
                q.parte = "brazo"
            n_antes = len(prims)
        info["torso"] = (xb, top, bot, ancho)
        return prims, info
