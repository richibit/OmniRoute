"""
Personajes tipo juguete para «La última vez que tu mamá te cepilló los dientes»:
mamá (26 → 58 años), hijo (8 meses → 32 años) y nieta (3 años).

Proporciones cercanas a las de los videos de referencia (cabeza grande pero
brazos largos: la mano cuelga a medio muslo), así que llegan de verdad a la
boca, a la encimera, al vaso o al marco del espejo. Cada personaje se
construye en coordenadas del mundo (ver `juguete3d`): cabeza en (cx, cy, z0),
cinemática inversa de dos huesos para brazos y piernas, manos con palma,
dedos y pulgar (puño que agarra, pinza, mano abierta, contar con los dedos),
inclinación del torso hacia el espejo, giro del cuerpo y cara pintada.
"""

import functools
import math

import numpy as np
from PIL import Image, ImageDraw

from juguete3d import Capsula, Elipsoide, _norm

PIEL = (238, 190, 152)

# Medidas en múltiplos del radio de la cabeza R (y hacia abajo desde el centro de la cabeza).
TIPOS = {
    "adulta": {"R": 104, "hombro_y": 1.78, "hombro_x": 1.36, "pecho": 1.1, "cintura_y": 3.65, "cadera": 1.04, "cadera_y": 4.7,
               "prof": 0.58, "brazo": (2.35, 2.0), "grosor": (0.26, 0.22), "mano": 1.0, "muslo": 3.45, "pierna": 3.3, "pie": 0.85},
    "adulto": {"R": 103, "hombro_y": 1.8, "hombro_x": 1.48, "pecho": 1.24, "cintura_y": 3.75, "cadera": 1.06, "cadera_y": 4.8,
               "prof": 0.6, "brazo": (2.45, 2.08), "grosor": (0.29, 0.25), "mano": 1.04, "muslo": 3.55, "pierna": 3.4, "pie": 0.9},
    "adolescente": {"R": 100, "hombro_y": 1.78, "hombro_x": 1.3, "pecho": 1.1, "cintura_y": 3.7, "cadera": 1.0, "cadera_y": 4.7,
                    "prof": 0.6, "brazo": (2.4, 2.05), "grosor": (0.27, 0.23), "mano": 1.02, "muslo": 3.5, "pierna": 3.35, "pie": 0.9},
    "nino7": {"R": 89, "hombro_y": 1.62, "hombro_x": 1.12, "pecho": 0.98, "cintura_y": 3.15, "cadera": 0.95, "cadera_y": 3.95,
              "prof": 0.62, "brazo": (1.95, 1.66), "grosor": (0.27, 0.23), "mano": 0.9, "muslo": 2.75, "pierna": 2.6, "pie": 0.8},
    "nino4": {"R": 87, "hombro_y": 1.55, "hombro_x": 1.06, "pecho": 0.96, "cintura_y": 2.9, "cadera": 0.94, "cadera_y": 3.55,
              "prof": 0.64, "brazo": (1.72, 1.46), "grosor": (0.28, 0.24), "mano": 0.86, "muslo": 2.3, "pierna": 2.15, "pie": 0.75},
    "pequeno": {"R": 82, "hombro_y": 1.5, "hombro_x": 1.0, "pecho": 0.95, "cintura_y": 2.6, "cadera": 0.95, "cadera_y": 3.2,
                "prof": 0.66, "brazo": (1.52, 1.3), "grosor": (0.29, 0.25), "mano": 0.8, "muslo": 2.0, "pierna": 1.85, "pie": 0.7},
    "bebe": {"R": 74, "hombro_y": 1.42, "hombro_x": 0.95, "pecho": 0.98, "cintura_y": 2.3, "cadera": 1.0, "cadera_y": 2.8,
             "prof": 0.7, "brazo": (1.22, 1.06), "grosor": (0.32, 0.28), "mano": 0.72, "muslo": 1.0, "pierna": 0.85, "pie": 0.6},
}


def altura_pies(tipo, escala=1.0):
    """Distancia (px del mundo) del centro de la cabeza a la planta de los pies."""
    P = TIPOS[tipo]
    return (P["cadera_y"] + 0.25 + P["muslo"] + P["pierna"] + 0.42) * P["R"] * escala


# ---------------------------------------------------------------------------
# Caras (texturas pintadas que se proyectan sobre la cabeza)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=96)
def textura_cara(expresion="neutral", hueco=False, rubor=0.5, arrugas=False):
    """
    Cara frontal en un lienzo cuadrado (centro = frente de la cabeza).
    Expresiones: neutral, sonrisa, dientes, apretada, mueca, dolor, feliz (ojos cerrados),
    abierta (boca abierta para cepillarse), sorpresa, parpadeo, tapada, risa, concentrada.
    """
    S = 1024
    k = S / 512
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tinta = (34, 22, 18, 255)
    ojo_y = 236 * k

    def B(*v):
        return [x * k for x in v]

    for sx in (-1, 1):
        ex = (256 + sx * 70) * k
        if expresion in ("feliz", "tapada", "risa"):
            d.arc([ex - 26 * k, ojo_y - 16 * k, ex + 26 * k, ojo_y + 26 * k], 200, 340, fill=tinta, width=int(9 * k))
        elif expresion == "parpadeo":
            d.line([(ex - 22 * k, ojo_y + 4 * k), (ex + 22 * k, ojo_y + 4 * k)], fill=tinta, width=int(8 * k))
        elif expresion == "dolor":
            if sx < 0:
                d.line([(ex - 20 * k, ojo_y - 12 * k), (ex + 18 * k, ojo_y + 2 * k)], fill=tinta, width=int(8 * k))
                d.line([(ex - 20 * k, ojo_y + 14 * k), (ex + 18 * k, ojo_y + 2 * k)], fill=tinta, width=int(8 * k))
            else:
                d.line([(ex + 20 * k, ojo_y - 12 * k), (ex - 18 * k, ojo_y + 2 * k)], fill=tinta, width=int(8 * k))
                d.line([(ex + 20 * k, ojo_y + 14 * k), (ex - 18 * k, ojo_y + 2 * k)], fill=tinta, width=int(8 * k))
        elif expresion == "mueca" and sx > 0:
            d.arc([ex - 24 * k, ojo_y - 10 * k, ex + 24 * k, ojo_y + 20 * k], 200, 340, fill=tinta, width=int(9 * k))  # guiño
        else:
            alto = (34 if expresion == "sorpresa" else 30) * k
            dy = 8 * k if expresion == "concentrada" else 0  # mira hacia abajo
            d.ellipse([ex - 15 * k, ojo_y - alto / 2 + dy, ex + 15 * k, ojo_y + alto / 2 + dy], fill=tinta)
            d.ellipse([ex - 7 * k, ojo_y - alto / 2 + 5 * k + dy, ex + 1 * k, ojo_y - alto / 2 + 13 * k + dy], fill=(255, 255, 255, 230))
            if expresion == "concentrada":  # párpado superior
                d.line([(ex - 18 * k, ojo_y - 8 * k), (ex + 18 * k, ojo_y - 8 * k)], fill=tinta, width=int(6 * k))
    if expresion in ("mueca", "sorpresa", "dolor"):  # cejas expresivas
        for sx in (-1, 1):
            ex = (256 + sx * 70) * k
            d.arc([ex - 30 * k, ojo_y - 62 * k, ex + 30 * k, ojo_y - 22 * k], 200, 340, fill=tinta, width=int(7 * k))
    if rubor:
        for sx in (-1, 1):
            capa = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            cx = (256 + sx * 112) * k
            ImageDraw.Draw(capa).ellipse([cx - 34 * k, 280 * k, cx + 34 * k, 320 * k], fill=(236, 120, 110, int(110 * rubor)))
            img.alpha_composite(capa)
    boca_y = 330
    labio = (150, 60, 58, 255)
    oscuro = (110, 34, 40, 255)
    blanco = (252, 250, 244, 255)
    if expresion in ("sonrisa", "feliz"):
        d.arc(B(256 - 46, boca_y - 34, 256 + 46, boca_y + 18), 20, 160, fill=labio, width=int(9 * k))
    elif expresion in ("dientes", "tapada", "risa"):
        d.chord(B(256 - 62, boca_y - 30, 256 + 62, boca_y + 46), 0, 180, fill=oscuro)
        d.rectangle(B(256 - 54, boca_y + 8, 256 + 54, boca_y + 24), fill=blanco)
        if expresion == "risa":
            d.chord(B(256 - 30, boca_y + 22, 256 + 30, boca_y + 44), 180, 360, fill=(236, 110, 120, 255))
        if hueco:
            d.rectangle(B(256 - 8, boca_y + 8, 256 + 10, boca_y + 24), fill=(70, 20, 26, 255))
    elif expresion == "apretada":
        d.line([tuple(B(256 - 40, boca_y + 4)), tuple(B(256 - 14, boca_y - 4)), tuple(B(256 + 14, boca_y + 4)), tuple(B(256 + 40, boca_y - 4))], fill=labio, width=int(9 * k), joint="curve")
    elif expresion == "mueca":
        d.ellipse(B(256 - 30, boca_y - 20, 256 + 30, boca_y + 34), fill=oscuro)
        d.ellipse(B(256 - 20, boca_y + 6, 256 + 20, boca_y + 40), fill=(236, 110, 120, 255))  # lengua
    elif expresion == "abierta":
        d.ellipse(B(256 - 40, boca_y - 18, 256 + 40, boca_y + 38), fill=oscuro)
        d.rectangle(B(256 - 30, boca_y - 13, 256 + 30, boca_y - 1), fill=blanco)
        d.rectangle(B(256 - 26, boca_y + 24, 256 + 26, boca_y + 32), fill=blanco)
    elif expresion == "sorpresa":
        d.ellipse(B(256 - 18, boca_y - 10, 256 + 18, boca_y + 26), fill=oscuro)
    elif expresion == "dolor":
        d.line([tuple(B(256 - 36, boca_y + 10)), tuple(B(256 - 12, boca_y)), tuple(B(256 + 12, boca_y + 10)), tuple(B(256 + 36, boca_y))], fill=labio, width=int(8 * k), joint="curve")
    else:
        d.line([tuple(B(256 - 24, boca_y)), tuple(B(256 + 24, boca_y))], fill=labio, width=int(8 * k))
    if arrugas:
        for sx in (-1, 1):
            ex = (256 + sx * 70) * k
            d.arc([ex + sx * 34 * k - 14 * k, ojo_y - 8 * k, ex + sx * 34 * k + 14 * k, ojo_y + 20 * k], 300 if sx > 0 else 120, 60 if sx > 0 else 240, fill=(120, 80, 64, 140), width=int(4 * k))
        d.arc(B(256 - 80, boca_y - 70, 256 - 40, boca_y + 10), 100, 170, fill=(120, 80, 64, 120), width=int(4 * k))
        d.arc(B(256 + 40, boca_y - 70, 256 + 80, boca_y + 10), 10, 80, fill=(120, 80, 64, 120), width=int(4 * k))
    img = img.resize((512, 512), Image.LANCZOS)
    return np.asarray(img, np.float32) / 255.0


# ---------------------------------------------------------------------------
# Cinemática inversa de dos huesos
# ---------------------------------------------------------------------------
def ik(hombro, objetivo, l1, l2, polo):
    """Codo y muñeca para llevar la muñeca a `objetivo` (si no llega, estira el brazo hacia él)."""
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
    if np.linalg.norm(p) < 1e-6:
        p = np.array([0, 1, 0], np.float32) - u * float(u[1])
    p = p / (np.linalg.norm(p) + 1e-9)
    codo = S + u * a + p * h
    return codo, T


# ---------------------------------------------------------------------------
# Manos
# ---------------------------------------------------------------------------
DEDOS_OFFS = (0.15, 0.05, -0.06, -0.16)  # índice, medio, anular, meñique (a lo ancho de la palma)
DEDOS_LARGO = (0.42, 0.46, 0.42, 0.34)


def marco_mano(d, nrm, sx):
    """Base de la mano: d (muñeca → dedos), nrm (sale de la palma) y lat (hacia el pulgar)."""
    d = _norm(d)
    nrm = np.asarray(nrm, np.float32)
    nrm = _norm(nrm - d * float(nrm @ d))
    lat = _norm(sx * np.cross(d, nrm))
    return d, nrm, lat


def construir_mano(prims, muneca, d, nrm, sx, L, piel, forma="relajada", dedos=None):
    """
    Añade la mano a `prims` y devuelve sus puntos útiles (centro, agarre, pinza, palma).
    forma: relajada | puño | pinza | abierta | contar (dedos = extensión 0–1 de índice,
    medio, anular, meñique y pulgar).
    """
    d, nrm, lat = marco_mano(d, nrm, sx)
    W = np.asarray(muneca, np.float32)

    def P(a, b, c):
        return W + d * (a * L) + lat * (b * L) + nrm * (c * L)

    rot = np.stack([lat, d, nrm], 1)
    puntos = {"centro": P(0.4, 0.0, 0.05), "marco": (d, nrm, lat)}
    if forma == "puño":
        prims.append(Elipsoide(P(0.30, 0.0, 0.05), (0.25 * L, 0.27 * L, 0.17 * L), piel, rot=rot))
        prims.append(Capsula(P(0.47, -0.16, 0.11), P(0.47, 0.12, 0.11), 0.115 * L, piel))  # dedos doblados
        prims.append(Capsula(P(0.14, 0.21, 0.03), P(0.40, 0.15, 0.21), 0.078 * L, piel))  # pulgar sobre los dedos
        puntos["agarre"] = P(0.38, 0.0, 0.11)
    elif forma == "pinza":
        prims.append(Elipsoide(P(0.30, 0.0, 0.04), (0.24 * L, 0.27 * L, 0.15 * L), piel, rot=rot))
        prims.append(Capsula(P(0.47, -0.17, 0.10), P(0.47, 0.01, 0.10), 0.105 * L, piel))  # medio, anular y meñique doblados
        prims.append(Capsula(P(0.52, 0.12, 0.02), P(0.70, 0.15, 0.11), 0.07 * L, piel))  # índice
        prims.append(Capsula(P(0.14, 0.21, 0.03), P(0.62, 0.21, 0.13), 0.078 * L, piel))  # pulgar que se junta con el índice
        puntos["pinza"] = P(0.70, 0.19, 0.15)
    elif forma == "abierta":
        prims.append(Elipsoide(P(0.30, 0.0, 0.0), (0.25 * L, 0.28 * L, 0.11 * L), piel, rot=rot))
        prims.append(Elipsoide(P(0.70, -0.02, 0.0), (0.22 * L, 0.27 * L, 0.085 * L), piel, rot=rot))  # dedos juntos
        prims.append(Capsula(P(0.16, 0.20, 0.03), P(0.40, 0.40, 0.05), 0.075 * L, piel))  # pulgar abierto
        puntos["palma"] = P(0.42, 0.0, 0.12)
        puntos["yemas"] = P(0.92, -0.02, 0.0)
    elif forma == "contar":
        dedos = list(dedos or (0, 0, 0, 0, 0)) + [0] * 5
        prims.append(Elipsoide(P(0.30, 0.0, 0.0), (0.25 * L, 0.28 * L, 0.12 * L), piel, rot=rot))
        for k in range(4):
            e = float(np.clip(dedos[k], 0, 1))
            base = P(0.53, DEDOS_OFFS[k], 0.0)
            abanico = DEDOS_OFFS[k] * 0.45 * e
            a1 = (1 - e) * math.radians(88)
            a2 = (1 - e) * math.radians(105)
            l1, l2 = DEDOS_LARGO[k] * 0.55 * L, DEDOS_LARGO[k] * 0.45 * L
            u1 = _norm(d * math.cos(a1) + nrm * math.sin(a1) + lat * abanico)
            p1 = base + u1 * l1
            u2 = _norm(d * math.cos(a1 + a2) + nrm * math.sin(a1 + a2) + lat * abanico)
            p2 = p1 + u2 * l2
            prims.append(Capsula(base, p1, 0.068 * L, piel))
            prims.append(Capsula(p1, p2, 0.063 * L, piel))
        e = float(np.clip(dedos[4], 0, 1))
        base = P(0.13, 0.21, 0.02)
        abierto = base + (d * 0.24 + lat * 0.30) * L
        doblado = base + (d * 0.34 - lat * 0.14 + nrm * 0.13) * L
        prims.append(Capsula(base, doblado + (abierto - doblado) * e, 0.076 * L, piel))
    else:  # relajada: dedos juntos un poco curvados
        prims.append(Elipsoide(P(0.30, 0.0, 0.02), (0.25 * L, 0.28 * L, 0.12 * L), piel, rot=rot))
        prims.append(Elipsoide(P(0.64, -0.01, 0.07), (0.22 * L, 0.23 * L, 0.10 * L), piel, rot=rot))
        prims.append(Capsula(P(0.16, 0.20, 0.04), P(0.40, 0.19, 0.14), 0.075 * L, piel))
    return puntos


def cuenta_dedos(n):
    """Extensión de cada dedo (índice, medio, anular, meñique, pulgar) para mostrar `n` (fraccionario: se estiran poco a poco)."""
    return tuple(float(np.clip(n - k, 0, 1)) for k in range(5))


# ---------------------------------------------------------------------------
# Personaje
# ---------------------------------------------------------------------------
class Personaje:
    """
    tipo: adulta | adulto | adolescente | nino7 | nino4 | pequeno | bebe
    ropa: color de la prenda superior · pantalon: color de abajo
    pelo: color · estilo: mono | corto | bebe | coletas | capucha
    """

    def __init__(self, tipo, ropa, pantalon, pelo, estilo, piel=PIEL, arrugas=False, pegatina=None, zapatos=(70, 54, 46)):
        self.p = TIPOS[tipo]
        self.tipo = tipo
        self.ropa = ropa
        self.pantalon = pantalon
        self.pelo = pelo
        self.estilo = estilo
        self.piel = piel
        self.arrugas = arrugas
        self.pegatina = pegatina
        self.zapatos = zapatos

    def hombro(self, cx, cy, z0, lado, pose=None, escala=1.0):
        """Posición del hombro (para preparar agarres antes de construir)."""
        _, info = self.construir(cx, cy, dict(pose or {}, solo_hombros=True), escala, z0)
        return info["hombros"][lado]

    def construir(self, cx, cy, pose=None, escala=1.0, z0=0.0, camara=None):
        """
        Primitivas del personaje con la cabeza en (cx, cy, z0). Devuelve (primitivas, info):
        info tiene cabeza, R, boca, manos (centro y puntos de agarre), hombros, error de alcance.

        pose: cara, hueco, giro (cabeza, ±1 ≈ ±35°), cabeceo (+ mira abajo), mirar=(x, y, z)
        o mirar_camara, inclinacion (rad, hacia el espejo), giro_cuerpo (rad), cuerpo_dx,
        mano_izq / mano_der (None, punto del centro de la mano o dict con muneca/centro/agarre+eje/
        pinza, dir, nrm, forma, dedos, polo), piernas (ocultas | de_pie | sentado | dict con pies
        {"izq": p, "der": p}), balanceo (rad).
        """
        pose = pose or {}
        P = self.p
        R = P["R"] * escala
        prims = []
        cabeceo = pose.get("cabeceo", 0.0)
        giro = pose.get("giro", 0.0)
        incl = pose.get("inclinacion", 0.0)
        yaw = pose.get("giro_cuerpo", 0.0)
        xb = cx + pose.get("cuerpo_dx", 0.0) * R
        zb = z0 - R * 0.36
        y_cadera = cy + P["cadera_y"] * R
        ci, si = math.cos(incl), math.sin(incl)
        cyw, syw = math.cos(yaw), math.sin(yaw)

        def tf(p, inclinar=True):
            """Cuerpo neutro → mundo: giro del cuerpo (eje vertical) y luego inclinación sobre la cadera."""
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            dx_, dz_ = x - xb, z - zb
            x, z = xb + dx_ * cyw + dz_ * syw, zb - dx_ * syw + dz_ * cyw
            if inclinar and incl:
                dy_, dz_ = y - y_cadera, z - zb
                y, z = y_cadera + dy_ * ci + dz_ * si, zb - dy_ * si + dz_ * ci  # la cabeza va hacia +z (el espejo)
            return np.array([x, y, z], np.float32)

        # ---- cabeza (el centro sigue a la inclinación)
        hc = tf((cx, cy, z0))
        hx, hy, hz = float(hc[0]), float(hc[1]), float(hc[2])
        mirar = pose.get("mirar")
        if mirar is None and pose.get("mirar_camara", True) and camara is not None:
            mirar = (camara.cx, camara.cy, camara.D)
        if mirar is not None:  # giro y cabeceo base para mirar a un punto (el espejo, otra cara…): los ojos hacen parte del giro
            mx, my, mz = mirar
            giro += float(np.clip(math.atan2(mx - hx, mz - hz) / 0.6 * 0.7, -0.95, 0.95))
            cabeceo += float(np.clip(math.atan2(my - hy, math.hypot(mx - hx, mz - hz)) * 0.8, -0.42, 0.55))
        giro = float(np.clip(giro, -1.1, 1.1))
        cabeceo = float(np.clip(cabeceo, -0.6, 0.85))
        if not pose.get("solo_hombros"):
            cabeza = Elipsoide((hx, hy, hz), (R, R * 1.02, R * 0.96), self.piel, brillo=0.06)
            cabeza.textura = textura_cara(pose.get("cara", "neutral"), pose.get("hueco", False), 0.5, self.arrugas)
            cabeza.giro_textura = giro * 0.6
            cabeza.cabeceo_textura = -cabeceo * 0.85
            prims.append(cabeza)
            self._pelo_y_orejas(prims, hx, hy, hz, R, giro, cabeceo)
        # ---- cuello, torso y cadera
        top = cy + R * 1.45
        cin = cy + P["cintura_y"] * R
        if not pose.get("solo_hombros"):
            prims.append(Capsula(tf((cx, cy + R * 0.7, z0 - R * 0.3)), tf((xb, top + R * 0.25, zb + R * 0.05)), R * 0.3, self.piel))
            pecho_r = P["pecho"] * R
            ys = cy + P["hombro_y"] * R
            hx_ = (P["hombro_x"] - 0.18) * R
            prims.append(Capsula(tf((xb - hx_, ys, zb)), tf((xb + hx_, ys, zb)), R * 0.36, self.ropa, aplanado=0.85, brillo=0.04))  # línea de hombros
            prims.append(Capsula(tf((xb, top + pecho_r * 0.95, zb)), tf((xb, cin - R * 0.1, zb)), pecho_r, self.ropa, aplanado=P["prof"], brillo=0.04))
            if self.estilo == "capucha":  # capucha caída sobre los hombros
                prims.append(Capsula(tf((xb - pecho_r * 0.6, top + R * 0.12, zb - R * 0.32)), tf((xb + pecho_r * 0.6, top + R * 0.12, zb - R * 0.32)), R * 0.42, self.ropa))
            if pose.get("piernas", "ocultas") != "ninguna":
                cad_r = P["cadera"] * R
                prims.append(Capsula(tf((xb, cin - R * 0.25, zb)), tf((xb, y_cadera + R * 0.15, zb), False), cad_r, self.pantalon, aplanado=P["prof"] * 1.04, brillo=0.03))
                prims.append(Capsula(tf((xb, cin - R * 0.32, zb)), tf((xb, cin + R * 0.05, zb)), pecho_r * 0.97, self.ropa, aplanado=P["prof"] * 1.06, brillo=0.04))  # bajo del suéter
            self._piernas(prims, pose, xb, zb, y_cadera, R, tf)
            if self.pegatina:  # pegatina «paciente valiente» en el pecho
                pz = zb + pecho_r * P["prof"] * 0.97
                prims.append(Elipsoide(tf((xb + pecho_r * 0.38, top + R * 0.95, pz)), (R * 0.27, R * 0.27, R * 0.05), self.pegatina, brillo=0.25))
        info = {"cabeza": (hx, hy, hz), "R": R, "giro": giro, "cabeceo": cabeceo, "manos": {}, "hombros": {}, "puntos": {}, "error": {}}
        info["boca"] = self.boca(hx, hy, hz, R, giro, cabeceo)
        info["torso"] = (xb, top, cin, zb)
        for q in prims:
            q.parte = "cuerpo"
        # ---- brazos (cinemática inversa hacia la muñeca)
        l1, l2 = P["brazo"][0] * R, P["brazo"][1] * R
        L = P["mano"] * R
        for lado, sx in (("izq", -1), ("der", 1)):
            hombro = tf((xb + sx * P["hombro_x"] * R, cy + P["hombro_y"] * R, zb + R * 0.02))
            info["hombros"][lado] = tuple(float(v) for v in hombro)
            if pose.get("solo_hombros"):
                continue
            n0 = len(prims)
            muneca, d, nrm, forma, dedos, polo = self._mano_objetivo(pose, lado, sx, hombro, l1, l2, L, R)
            codo, mu = ik(hombro, muneca, l1, l2, polo)
            info["error"][lado] = float(np.linalg.norm(mu - muneca))
            if forma is None:  # mano colgando: sigue al antebrazo
                d = _norm(mu - codo)
                nrm = np.array([-sx, 0.0, 0.25], np.float32)
                forma = "relajada"
            g1, g2 = P["grosor"][0] * R, P["grosor"][1] * R
            prims.append(Elipsoide(hombro, (g1 * 1.15, g1 * 1.15, g1 * 1.1), self.ropa, brillo=0.04))
            prims.append(Capsula(hombro, codo, g1, self.ropa, brillo=0.04))
            prims.append(Capsula(codo, mu, g2, self.ropa, brillo=0.04))
            prims.append(Capsula(mu - _norm(mu - codo) * g2 * 0.6, mu + _norm(d) * L * 0.08, g2 * 0.62, self.piel))  # muñeca
            pts = construir_mano(prims, mu, d, nrm, sx, L, self.piel, forma, dedos)
            info["manos"][lado] = tuple(float(v) for v in pts["centro"])
            info["puntos"][lado] = pts
            info.setdefault("codos", {})[lado] = tuple(float(v) for v in codo)
            for q in prims[n0:]:
                q.parte = "brazo"
        return prims, info

    # ------------------------------------------------------------ partes
    @staticmethod
    def boca(hx, hy, hz, R, giro, cabeceo):
        """Punto de la boca sobre la superficie de la cabeza (sigue al giro y al cabeceo)."""
        g = giro * 0.6
        p = -cabeceo * 0.85
        # dirección de la boca en la textura: un poco por debajo del centro de la cara
        ny0 = 0.36
        nz0 = math.sqrt(1 - ny0 * ny0)
        ny = ny0 * math.cos(p) - nz0 * math.sin(p)
        nz = ny0 * math.sin(p) + nz0 * math.cos(p)
        nx = nz * math.sin(g)
        nz = nz * math.cos(g)
        return (hx + nx * R, hy + ny * R * 1.02, hz + nz * R * 0.96)

    def _pelo_y_orejas(self, prims, cx, cy, z0, R, giro, cabeceo):
        dz = -giro * R * 0.1
        dyc = cabeceo * R * 0.12
        zc = max(cabeceo, 0.0) * R * 0.32  # al mirar abajo el pelo se ve más desde arriba: avanza hacia la cara
        for sx in (-1, 1):  # orejas
            prims.append(Elipsoide((cx + sx * R * 0.97 - giro * R * 0.25, cy + R * 0.05 - dyc * 0.5, z0 - R * 0.12), (R * 0.15, R * 0.22, R * 0.12), self.piel))
        if self.estilo in ("mono", "corto", "coletas"):
            prims.append(Elipsoide((cx - giro * R * 0.12, cy - R * 0.2 + dyc * 0.6, z0 - R * 0.14 + zc), (R * 1.05, R * 0.93, R * 0.96), self.pelo, brillo=0.12))
            prims.append(Elipsoide((cx + R * 0.18 - giro * R * 0.35, cy - R * 0.62 + dyc * 1.6, z0 + R * 0.42 + dz + zc * 0.5), (R * 0.72, R * 0.26, R * 0.46), self.pelo, giro=-12, brillo=0.12))
        if self.estilo == "mono":
            prims.append(Elipsoide((cx - giro * R * 0.3, cy - R * 1.02 + dyc * 0.3, z0 - R * 0.35 - dyc), (R * 0.42, R * 0.36, R * 0.38), self.pelo, brillo=0.12))
            for sx in (-1, 1):  # mechones laterales hasta la mandíbula
                prims.append(Elipsoide((cx + sx * R * 0.9 - giro * R * 0.2, cy + R * 0.12, z0 - R * 0.28), (R * 0.26, R * 0.62, R * 0.4), self.pelo, brillo=0.12))
        elif self.estilo == "coletas":
            for sx in (-1, 1):
                prims.append(Elipsoide((cx + sx * R * 1.08, cy - R * 0.18, z0 - R * 0.3), (R * 0.32, R * 0.42, R * 0.32), self.pelo, giro=sx * 25, brillo=0.12))
        elif self.estilo == "bebe":
            prims.append(Elipsoide((cx + R * 0.1, cy - R * 0.94 + dyc * 0.3, z0 + R * 0.08), (R * 0.26, R * 0.14, R * 0.2), self.pelo, giro=-20))
        elif self.estilo == "capucha":
            prims.append(Elipsoide((cx - giro * R * 0.12, cy - R * 0.2 + dyc * 0.6, z0 - R * 0.14 + zc), (R * 1.04, R * 0.92, R * 0.96), self.pelo, brillo=0.12))
            prims.append(Elipsoide((cx + R * 0.15 - giro * R * 0.35, cy - R * 0.6 + dyc * 1.6, z0 + R * 0.4 + dz + zc * 0.5), (R * 0.7, R * 0.25, R * 0.45), self.pelo, giro=-10, brillo=0.12))

    def _piernas(self, prims, pose, xb, zb, y_cadera, R, tf):
        P = self.p
        piernas = pose.get("piernas", "ocultas")
        if piernas in ("ocultas", "ninguna"):
            return
        ancho = P["cadera"] * R
        l1, l2 = P["muslo"] * R, P["pierna"] * R
        g = R * 0.36
        if piernas == "sentado":  # sentado en la encimera: muslos hacia la cámara, piernas colgando un poco
            for sx in (-1, 1):
                cadera = tf((xb + sx * ancho * 0.45, y_cadera, zb), False)
                rodilla = cadera + np.array([sx * ancho * 0.45, R * 0.05, l1 * 0.92], np.float32)
                pie = rodilla + np.array([sx * ancho * 0.2, l2 * 0.42, l2 * 0.82], np.float32)
                prims.append(Capsula(cadera, rodilla, g * 1.25, self.pantalon))
                prims.append(Capsula(rodilla, pie, g * 1.1, self.pantalon))
                prims.append(Elipsoide(tuple(pie + np.array([0, R * 0.05, R * 0.15], np.float32)), (R * 0.32, R * 0.28, R * 0.42), self.piel))
            return
        pies = piernas.get("pies", {}) if isinstance(piernas, dict) else {}
        suelo = y_cadera + R * 0.25 + l1 + l2
        for lado, sx in (("izq", -1), ("der", 1)):
            cadera = tf((xb + sx * ancho * 0.45, y_cadera + R * 0.25, zb), False)
            tobillo = np.asarray(pies.get(lado, (cadera[0], suelo, cadera[2])), np.float32)
            rodilla, tob = ik(cadera, tobillo, l1, l2, (0, 0, 1))
            prims.append(Capsula(cadera, rodilla, g, self.pantalon, brillo=0.03))
            prims.append(Capsula(rodilla, tob, g * 0.86, self.pantalon, brillo=0.03))
            prims.append(Elipsoide(tuple(tob + np.array([0, R * 0.2, R * 0.28], np.float32)), (R * 0.36, R * 0.22, R * P["pie"] * 0.6), self.zapatos, brillo=0.1))

    def _mano_objetivo(self, pose, lado, sx, hombro, l1, l2, L, R):
        """(muñeca, dir, nrm, forma, dedos, polo) a partir de lo que pide la pose para esa mano."""
        spec = pose.get(f"mano_{lado}")
        polo = np.array([sx * 0.75, 0.55, -0.35], np.float32)
        H = np.asarray(hombro, np.float32)
        if spec is None:  # colgando, con balanceo al caminar
            ang = pose.get("balanceo", 0.0) * (-sx)
            larg = (l1 + l2) * 0.97
            muneca = H + np.array([sx * R * 0.2, larg * math.cos(ang), larg * math.sin(ang) + R * 0.12], np.float32)
            return muneca, None, None, None, None, polo
        if not isinstance(spec, dict):
            spec = {"centro": spec}
        polo = np.asarray(spec.get("polo", polo), np.float32)
        forma = spec.get("forma", "relajada")
        dedos = spec.get("dedos")
        if "agarre" in spec:  # puño alrededor de un mango: el mango sale por el lado del pulgar hacia `eje`
            g = np.asarray(spec["agarre"], np.float32)
            lat = _norm(spec["eje"])
            d0 = _norm(np.asarray(spec.get("dir", g - H + np.array([0, R * 0.8, 0], np.float32)), np.float32))
            d = d0 - lat * float(d0 @ lat)
            d = _norm(d if np.linalg.norm(d) > 1e-4 else np.array([0, 1, 0], np.float32))
            nrm = _norm(sx * np.cross(lat, d))
            muneca = g - (d * 0.38 + nrm * 0.11) * L
            return muneca, d, nrm, "puño", None, polo
        d = np.asarray(spec.get("dir", (0, 1, 0)), np.float32)
        nrm = np.asarray(spec.get("nrm", (-sx, 0, 0.3)), np.float32)
        dd, nn, ll = marco_mano(d, nrm, sx)
        if "pinza" in spec:
            q = np.asarray(spec["pinza"], np.float32)
            muneca = q - (dd * 0.70 + ll * 0.19 + nn * 0.15) * L
            forma = "pinza"
        elif "palma" in spec:  # punto donde apoya la palma (mano abierta)
            q = np.asarray(spec["palma"], np.float32)
            muneca = q - (dd * 0.42 + nn * 0.12) * L
            forma = spec.get("forma", "abierta")
        elif "muneca" in spec:
            muneca = np.asarray(spec["muneca"], np.float32)
        else:
            q = np.asarray(spec["centro"], np.float32)
            if "dir" not in spec:
                dd = _norm(q - H + np.array([0, R * 0.5, 0], np.float32))
                nrm = np.asarray(spec.get("nrm", (-sx, 0, 0.3)), np.float32)
                dd, nn, ll = marco_mano(dd, nrm, sx)
            muneca = q - (dd * 0.4 + nn * 0.05) * L
            d = dd
        return muneca, dd, nn, forma, dedos, polo
