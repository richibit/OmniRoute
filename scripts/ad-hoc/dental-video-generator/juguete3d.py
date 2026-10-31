"""
Motor 3D mínimo para personajes y objetos «tipo juguete» (acabado mate).

Cámara estenopeica con el eje óptico horizontal (las verticales siguen
verticales, como con un objetivo descentrable): el mundo usa x a la derecha,
y hacia abajo y z hacia la cámara, en píxeles del plano de referencia z = 0
(donde 1 px del mundo = 1 px de pantalla). Lo que se acerca a la cámara crece
y baja en el encuadre, así que las manos pueden llegar a lo que está sobre la
encimera, al grifo o al marco del espejo con distancias reales.

Cada primitiva (elipsoide orientable, cápsula, caja con texturas, cilindro)
se interseca de forma analítica con los rayos de la cámara; la visibilidad se
resuelve con un z-buffer y se ilumina con luz principal envolvente, relleno,
contorno, ambiente, especular, reflejos de metal y oclusión ambiental en
espacio de pantalla. Todo con numpy; se renderiza a 2× y se reduce (antialias).
"""

import math

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Vectores y rotaciones
# ---------------------------------------------------------------------------
def _norm(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def base_desde(eje_y, eje_z_aprox):
    """Base ortonormal (columnas x, y, z) con la y local en `eje_y` y la z lo más cerca posible de `eje_z_aprox`."""
    y = _norm(eje_y)
    z = np.asarray(eje_z_aprox, np.float32)
    z = z - y * float(z @ y)
    if np.linalg.norm(z) < 1e-5:
        z = np.array([0, 0, 1], np.float32) - y * float(y[2])
        if np.linalg.norm(z) < 1e-5:
            z = np.array([1, 0, 0], np.float32)
    z = _norm(z)
    x = np.cross(y, z)
    return np.stack([x, y, z], 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Cámara
# ---------------------------------------------------------------------------
class Camara:
    """
    Cámara en (cx, cy, D) mirando hacia −z. Un punto del plano z = 0 se ve en su
    misma (x, y) de pantalla; a profundidad z la escala es D / (D − z).
    `zoom` y `foco` acercan la imagen (recorte nítido, se renderiza a resolución completa).
    """

    def __init__(self, D=800.0, cx=540.0, cy=640.0, zoom=1.0, foco=(540.0, 960.0)):
        self.D, self.cx, self.cy = float(D), float(cx), float(cy)
        self.zoom = float(zoom)
        self.foco = (float(foco[0]), float(foco[1]))

    @property
    def origen(self):
        return np.array([self.cx, self.cy, self.D], np.float32)

    def escala(self, z):
        return self.D / (self.D - z) * self.zoom

    def proyectar(self, x, y, z):
        s = self.D / np.maximum(self.D - np.asarray(z, np.float32), 1.0)
        fx, fy = self.foco
        xv = self.cx + (np.asarray(x, np.float32) - self.cx) * s
        yv = self.cy + (np.asarray(y, np.float32) - self.cy) * s
        return fx + (xv - fx) * self.zoom, fy + (yv - fy) * self.zoom

    def a_virtual(self, X, Y):
        """Coordenadas de salida → plano de imagen sin zoom."""
        fx, fy = self.foco
        return fx + (X - fx) / self.zoom, fy + (Y - fy) / self.zoom

    def direcciones(self, X, Y):
        """Dirección (sin normalizar) del rayo de cada píxel; t = 1 cae en el plano z = 0."""
        xv, yv = self.a_virtual(X, Y)
        return xv - self.cx, yv - self.cy, np.full(np.shape(X), -self.D, np.float32)


# ---------------------------------------------------------------------------
# Iluminación
# ---------------------------------------------------------------------------
class Iluminacion:
    """Luz principal, relleno, contorno y ambiente (colores en 0–1, lineales)."""

    def __init__(self, principal, color_principal, relleno, color_relleno, ambiente, contorno=(0.0, 0.0, 0.0)):
        self.principal = _norm(principal)
        self.color_principal = np.asarray(color_principal, np.float32)
        self.relleno = _norm(relleno)
        self.color_relleno = np.asarray(color_relleno, np.float32)
        self.ambiente = np.asarray(ambiente, np.float32)
        self.contorno = np.asarray(contorno, np.float32)

    def mezclar(self, otra, k):
        """Interpolación entre dos ambientes de luz (fundidos mañana ↔ noche)."""
        f = lambda a, b: a * (1 - k) + b * k  # noqa: E731
        return Iluminacion(
            f(self.principal, otra.principal),
            f(self.color_principal, otra.color_principal),
            f(self.relleno, otra.relleno),
            f(self.color_relleno, otra.color_relleno),
            f(self.ambiente, otra.ambiente),
            f(self.contorno, otra.contorno),
        )


# Noche: lámpara cálida sobre el espejo (de frente y arriba) + luz del pasillo por detrás.
NOCHE = Iluminacion((-0.35, -0.6, 0.72), (1.05, 0.80, 0.55), (0.8, -0.1, 0.4), (0.28, 0.22, 0.30), (0.30, 0.22, 0.18), (0.55, 0.38, 0.20))
# Mañana: luz fría que entra por la ventana (izquierda) + rebote cálido suave.
MANANA = Iluminacion((-0.7, -0.5, 0.5), (1.02, 0.99, 0.94), (0.75, 0.05, 0.45), (0.26, 0.30, 0.38), (0.22, 0.26, 0.31), (0.36, 0.42, 0.52))


# ---------------------------------------------------------------------------
# Primitivas
# ---------------------------------------------------------------------------
class Primitiva:
    def __init__(self, color, brillo=0.08):
        self.color = np.asarray(color, np.float32) / 255.0
        self.brillo = brillo
        self.metal = False  # cromado: refleja un entorno sencillo (techo claro, encimera oscura)
        self.textura = None  # calcomanía proyectada por normales (caras): array H×W×4 en 0–1
        self.giro_textura = 0.0  # giro de la cabeza (radianes)
        self.cabeceo_textura = 0.0  # mirar arriba/abajo (radianes)
        self.parte = "cuerpo"

    def esquinas(self):
        mn, mx = self.aabb()
        return np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])], np.float32)


def _a_local(R, dx, dy, dz):
    """Direcciones de mundo → base local (R tiene los ejes locales en columnas)."""
    return (
        dx * R[0, 0] + dy * R[1, 0] + dz * R[2, 0],
        dx * R[0, 1] + dy * R[1, 1] + dz * R[2, 1],
        dx * R[0, 2] + dy * R[1, 2] + dz * R[2, 2],
    )


def _a_mundo(R, nx, ny, nz):
    return (
        R[0, 0] * nx + R[0, 1] * ny + R[0, 2] * nz,
        R[1, 0] * nx + R[1, 1] * ny + R[1, 2] * nz,
        R[2, 0] * nx + R[2, 1] * ny + R[2, 2] * nz,
    )


def _apilar_normal(nx, ny, nz):
    n = np.stack([nx, ny, nz], -1)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)


class Elipsoide(Primitiva):
    """Elipsoide de centro c y radios r=(rx, ry, rz) en su base local `rot` (o girado `giro` grados en pantalla)."""

    def __init__(self, c, r, color, giro=0.0, brillo=0.08, rot=None):
        super().__init__(color, brillo)
        self.c = np.asarray(c, np.float32)
        self.r = np.asarray(r if hasattr(r, "__len__") else (r, r, r), np.float32)
        self.rot = np.asarray(rot, np.float32) if rot is not None else rot_z(math.radians(giro))

    def aabb(self):
        e = np.sqrt(((self.rot * self.r[None, :]) ** 2).sum(1))
        return self.c - e, self.c + e

    def intersecar(self, O, dx, dy, dz):
        R, r = self.rot, self.r
        q = (O - self.c) @ R / r
        lx, ly, lz = _a_local(R, dx, dy, dz)
        lx, ly, lz = lx / r[0], ly / r[1], lz / r[2]
        a = lx * lx + ly * ly + lz * lz
        b = q[0] * lx + q[1] * ly + q[2] * lz
        c = float(q @ q) - 1.0
        disc = b * b - a * c
        ok = disc > 0
        t = (-b - np.sqrt(np.clip(disc, 0, None))) / a
        ok &= t > 1e-4
        px, py, pz = q[0] + t * lx, q[1] + t * ly, q[2] + t * lz
        n = _apilar_normal(*_a_mundo(R, px / r[0], py / r[1], pz / r[2]))
        return ok, t, n, None


class Capsula(Primitiva):
    """Cápsula entre a y b de radio r; `aplanado` < 1 la achata en profundidad y `escala` por ejes la deforma."""

    def __init__(self, a, b, r, color, aplanado=1.0, brillo=0.08, escala=None):
        super().__init__(color, brillo)
        self.a = np.asarray(a, np.float32)
        self.b = np.asarray(b, np.float32)
        self.r = float(r)
        self.k = np.asarray(escala if escala is not None else (1.0, 1.0, aplanado), np.float32)

    def aabb(self):
        return np.minimum(self.a, self.b) - self.r * self.k, np.maximum(self.a, self.b) + self.r * self.k

    def intersecar(self, O, dx, dy, dz):
        k, r = self.k, self.r
        o, a, b = O / k, self.a / k, self.b / k
        ex, ey, ez = dx / k[0], dy / k[1], dz / k[2]
        ba, oa = b - a, o - a
        baba, baoa, oaoa = float(ba @ ba), float(ba @ oa), float(oa @ oa)
        bard = ba[0] * ex + ba[1] * ey + ba[2] * ez
        rdoa = oa[0] * ex + oa[1] * ey + oa[2] * ez
        rdrd = ex * ex + ey * ey + ez * ez
        A = baba * rdrd - bard * bard
        B = baba * rdoa - baoa * bard
        C = baba * oaoa - baoa * baoa - r * r * baba
        h = B * B - A * C
        A_ok = np.abs(A) > 1e-9
        t_cil = np.where(A_ok, (-B - np.sqrt(np.clip(h, 0, None))) / np.where(A_ok, A, 1.0), np.inf)
        y = baoa + t_cil * bard
        ok_cil = A_ok & (h >= 0) & (y > 0) & (y < baba) & (t_cil > 1e-4)
        t = np.where(ok_cil, t_cil, np.inf)
        for centro in (a, b):  # tapas esféricas
            oc = o - centro
            Bc = oc[0] * ex + oc[1] * ey + oc[2] * ez
            hc = Bc * Bc - rdrd * (float(oc @ oc) - r * r)
            tc = (-Bc - np.sqrt(np.clip(hc, 0, None))) / rdrd
            mejor = (hc > 0) & (tc > 1e-4) & (tc < t)
            t = np.where(mejor, tc, t)
        ok = np.isfinite(t)
        t = np.where(ok, t, 0.0)
        px, py, pz = o[0] + t * ex - a[0], o[1] + t * ey - a[1], o[2] + t * ez - a[2]
        s = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / max(baba, 1e-9), 0, 1)
        nx, ny, nz = px - s * ba[0], py - s * ba[1], pz - s * ba[2]
        n = _apilar_normal(nx / k[0], ny / k[1], nz / k[2])
        return ok, t, n, None


def _muestrear(tex, u, v):
    """Muestreo bilineal de una textura H×W×4 (0–1) con u, v en 0–1."""
    th, tw = tex.shape[:2]
    x = np.clip(u, 0, 1) * (tw - 1)
    y = np.clip(v, 0, 1) * (th - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, tw - 1)
    y1 = np.minimum(y0 + 1, th - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    arriba = tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx
    abajo = tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx
    return arriba * (1 - fy) + abajo * fy


class Caja(Primitiva):
    """
    Caja de semilados h=(hx, hy, hz) en la base local `rot`, con aristas suavizadas
    (`bisel`) y texturas opcionales por cara: {"+z": array, "-y": array, ...}.
    En las caras ±z la textura se lee de izquierda a derecha con +x local y de arriba
    abajo con +y local; en la de arriba (−y), de arriba abajo con +z local (se lee
    desde la cámara, que la ve desde delante y desde arriba).
    """

    def __init__(self, c, h, color, rot=None, brillo=0.08, bisel=0.12, texturas=None):
        super().__init__(color, brillo)
        self.c = np.asarray(c, np.float32)
        self.h = np.asarray(h, np.float32)
        self.rot = np.asarray(rot, np.float32) if rot is not None else np.eye(3, dtype=np.float32)
        self.bisel = bisel
        self.texturas = texturas or {}

    def aabb(self):
        e = (np.abs(self.rot) * self.h[None, :]).sum(1)
        return self.c - e, self.c + e

    def intersecar(self, O, dx, dy, dz):
        R, h = self.rot, self.h
        q = (O - self.c) @ R
        ls = _a_local(R, dx, dy, dz)
        tnear = np.full(np.shape(dx), -np.inf, np.float32)
        tfar = np.full(np.shape(dx), np.inf, np.float32)
        for i in range(3):
            l = ls[i]
            seguro = np.where(np.abs(l) < 1e-9, 1e-9, l)
            t1 = (-h[i] - q[i]) / seguro
            t2 = (h[i] - q[i]) / seguro
            tnear = np.maximum(tnear, np.minimum(t1, t2))
            tfar = np.minimum(tfar, np.maximum(t1, t2))
        ok = (tnear <= tfar) & (tnear > 1e-4)
        t = np.where(ok, tnear, 0.0)
        u = [(q[i] + t * ls[i]) / h[i] for i in range(3)]
        # normal de caja redondeada: superelipsoide (caras planas, aristas suaves)
        p = 2.0 / max(self.bisel, 0.02)
        nl = [np.sign(ui) * np.abs(np.clip(ui, -1, 1)) ** p / h[i] for i, ui in enumerate(u)]
        n = _apilar_normal(*_a_mundo(R, *nl))
        albedo = None
        if self.texturas:
            albedo = np.broadcast_to(self.color, np.shape(dx) + (3,)).copy()
            au = np.abs(np.stack(u, -1))
            eje = np.argmax(au, -1)
            for clave, tex in self.texturas.items():
                i = "xyz".index(clave[1])
                signo = 1 if clave[0] == "+" else -1
                m = ok & (eje == i) & (np.sign(u[i]) == signo)
                if not m.any():
                    continue
                if i == 2:
                    uu, vv = (u[0][m] * signo + 1) / 2, (u[1][m] + 1) / 2
                elif i == 1:
                    uu, vv = (u[0][m] + 1) / 2, (u[2][m] * -signo + 1) / 2
                else:
                    uu, vv = (-u[2][m] * signo + 1) / 2, (u[1][m] + 1) / 2
                s = _muestrear(tex, uu, vv)
                albedo[m] = albedo[m] * (1 - s[:, 3:4]) + s[:, :3] * s[:, 3:4]
        return ok, t, n, albedo


class Cilindro(Primitiva):
    """Cilindro de radio r y semialtura hy sobre el eje y local (`rot`), con tapas planas; `abierto` oscurece la tapa de arriba (−y)."""

    def __init__(self, c, r, hy, color, rot=None, brillo=0.08, abierto=False, color_interior=None):
        super().__init__(color, brillo)
        self.c = np.asarray(c, np.float32)
        self.r, self.hy = float(r), float(hy)
        self.rot = np.asarray(rot, np.float32) if rot is not None else np.eye(3, dtype=np.float32)
        self.abierto = abierto
        self.color_interior = np.asarray(color_interior if color_interior is not None else (0, 0, 0), np.float32) / 255.0

    def aabb(self):
        h = np.array([self.r, self.hy, self.r], np.float32)
        e = (np.abs(self.rot) * h[None, :]).sum(1)
        return self.c - e, self.c + e

    def intersecar(self, O, dx, dy, dz):
        R = self.rot
        q = (O - self.c) @ R
        lx, ly, lz = _a_local(R, dx, dy, dz)
        a = lx * lx + lz * lz
        b = q[0] * lx + q[2] * lz
        c = q[0] ** 2 + q[2] ** 2 - self.r ** 2
        disc = b * b - a * c
        a_ok = a > 1e-9
        t_lado = np.where(a_ok & (disc > 0), (-b - np.sqrt(np.clip(disc, 0, None))) / np.where(a_ok, a, 1.0), np.inf)
        y_lado = q[1] + t_lado * ly
        t_lado = np.where((t_lado > 1e-4) & (np.abs(y_lado) <= self.hy), t_lado, np.inf)
        t = t_lado
        tapa = np.zeros(np.shape(dx), np.int8)
        for signo in (-1, 1):
            seguro = np.where(np.abs(ly) < 1e-9, 1e-9, ly)
            tt = (signo * self.hy - q[1]) / seguro
            x_, z_ = q[0] + tt * lx, q[2] + tt * lz
            dentro = (tt > 1e-4) & (x_ * x_ + z_ * z_ <= self.r ** 2) & (tt < t)
            t = np.where(dentro, tt, t)
            tapa = np.where(dentro, signo, tapa)
        ok = np.isfinite(t)
        t = np.where(ok, t, 0.0)
        px, pz = q[0] + t * lx, q[2] + t * lz
        nlx = np.where(tapa == 0, px, 0.0)
        nlz = np.where(tapa == 0, pz, 0.0)
        nly = np.where(tapa == 0, 0.0, tapa.astype(np.float32) * self.r)
        n = _apilar_normal(*_a_mundo(R, nlx, nly, nlz))
        albedo = None
        if self.abierto:
            albedo = np.broadcast_to(self.color, np.shape(dx) + (3,)).copy()
            m = ok & (tapa == -1)
            rr = np.sqrt(px ** 2 + pz ** 2) / self.r
            borde = np.clip((rr - 0.86) / 0.06, 0, 1)[..., None]
            interior = self.color_interior * (0.55 + 0.45 * np.clip(-pz / self.r * 0.5 + 0.5, 0, 1))[..., None]
            albedo[m] = (interior * (1 - borde) + self.color * borde)[m]
        return ok, t, n, albedo


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def _caja1d(a, r, eje):
    """Media móvil de ancho 2r+1 a lo largo de un eje (con bordes replicados)."""
    pad = [(0, 0)] * a.ndim
    pad[eje] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=eje, dtype=np.float64)
    n = a.shape[eje]
    alto = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=eje)
    bajo = np.take(c, np.arange(0, n), axis=eje)
    return ((alto - bajo) / (2 * r + 1)).astype(np.float32)


def desenfocar(a, radio):
    """Desenfoque casi gaussiano (tres pasadas de caja) en numpy, para mapas de profundidad."""
    r = max(1, int(radio / 1.7))
    for _ in range(3):
        a = _caja1d(_caja1d(a, r, 0), r, 1)
    return a


def caja_pantalla(p, cam):
    """Rectángulo de pantalla (x0, y0, x1, y1) que ocupa la primitiva."""
    esq = p.esquinas()
    X, Y = cam.proyectar(esq[:, 0], esq[:, 1], np.minimum(esq[:, 2], cam.D - 2.0))
    return float(X.min()), float(Y.min()), float(X.max()), float(Y.max())


def render(primitivas, luz, cam, ss=2, margen=4, oclusion=0.55, oclusor=None, limites=(0, 0, 1080, 1920), devolver_t=False, mapa_t=None):
    """
    Renderiza las primitivas con la cámara `cam` y devuelve (imagen RGBA, x0, y0) con la
    esquina superior izquierda en coordenadas de pantalla. `oclusor(O, dx, dy, dz)` da,
    por rayo, la distancia t de lo que tapa (encimera, lavabo): lo que quede detrás no se ve.
    `mapa_t` = (t por subpíxel, x0, y0, ss) de una capa ya renderizada (utilería fija) que
    también tapa. Con `devolver_t` devuelve además la t por píxel ("completo": por subpíxel).
    """
    nada = (None, 0, 0, None) if devolver_t else (None, 0, 0)
    if not primitivas:
        return nada
    L0, T0, L1, T1 = limites
    cajas = [caja_pantalla(p, cam) for p in primitivas]
    validas = [(p, c) for p, c in zip(primitivas, cajas) if c[2] > L0 and c[0] < L1 and c[3] > T0 and c[1] < T1]
    if not validas:
        return nada
    x0 = max(L0, math.floor(min(c[0] for _, c in validas)) - margen)
    y0 = max(T0, math.floor(min(c[1] for _, c in validas)) - margen)
    x1 = min(L1, math.ceil(max(c[2] for _, c in validas)) + margen)
    y1 = min(T1, math.ceil(max(c[3] for _, c in validas)) + margen)
    Wp, Hp = (x1 - x0) * ss, (y1 - y0) * ss
    if Wp <= 0 or Hp <= 0:
        return nada
    O = cam.origen
    tbuf = np.full((Hp, Wp), np.inf, np.float32)
    nbuf = np.zeros((Hp, Wp, 3), np.float32)
    cbuf = np.zeros((Hp, Wp, 3), np.float32)
    bbuf = np.zeros((Hp, Wp), np.float32)
    mbuf = np.zeros((Hp, Wp), bool)
    ibuf = np.full((Hp, Wp), -1, np.int32)
    indices = {id(p): i for i, p in enumerate(primitivas)}
    for p, (bx0, by0, bx1, by1) in validas:
        i0, i1 = max(0, int((by0 - y0) * ss) - 1), min(Hp, int((by1 - y0) * ss) + 2)
        j0, j1 = max(0, int((bx0 - x0) * ss) - 1), min(Wp, int((bx1 - x0) * ss) + 2)
        if i1 <= i0 or j1 <= j0:
            continue
        Yg, Xg = np.mgrid[i0:i1, j0:j1].astype(np.float32)
        X = Xg / ss + x0 + 0.5 / ss
        Y = Yg / ss + y0 + 0.5 / ss
        dx, dy, dz = cam.direcciones(X, Y)
        ok, t, n, albedo = p.intersecar(O, dx, dy, dz)
        zona = tbuf[i0:i1, j0:j1]
        gana = ok & (t < zona)
        if not gana.any():
            continue
        zona[gana] = t[gana]
        nbuf[i0:i1, j0:j1][gana] = n[gana]
        cbuf[i0:i1, j0:j1][gana] = albedo[gana] if albedo is not None else p.color
        bbuf[i0:i1, j0:j1][gana] = p.brillo
        mbuf[i0:i1, j0:j1][gana] = p.metal
        ibuf[i0:i1, j0:j1][gana] = indices[id(p)]
    sel = np.flatnonzero(np.isfinite(tbuf).ravel())
    if sel.size == 0:
        return nada
    ii, jj = np.divmod(sel, Wp)
    Xs = jj.astype(np.float32) / ss + x0 + 0.5 / ss
    Ys = ii.astype(np.float32) / ss + y0 + 0.5 / ss
    dx, dy, dz = cam.direcciones(Xs, Ys)
    tsel = tbuf.ravel()[sel]
    visible = np.ones(sel.size, bool)
    if oclusor is not None:  # lo que queda detrás de la encimera o dentro del lavabo no se ve
        visible &= tsel <= oclusor(O, dx, dy, dz)
    if mapa_t is not None:  # ni lo que queda detrás de la utilería fija
        tm, ox, oy, sm = mapa_t[:4]
        Xm, Ym = Xs, Ys
        if len(mapa_t) > 4 and mapa_t[4] is not None:  # el mapa se hizo sin el acercamiento de la escena
            z2, fx2, fy2 = mapa_t[4]
            Xm, Ym = fx2 + (Xs - fx2) / z2, fy2 + (Ys - fy2) / z2
        fi = (Ym - oy) * sm
        fj = (Xm - ox) * sm
        dentro = (fi >= 0) & (fi < tm.shape[0]) & (fj >= 0) & (fj < tm.shape[1])
        mi = np.clip(fi.astype(np.int32), 0, tm.shape[0] - 1)
        mj = np.clip(fj.astype(np.int32), 0, tm.shape[1] - 1)
        visible &= ~(dentro & (tsel > tm[mi, mj]))
    if not visible.all():
        sel, Xs, Ys, dx, dy, dz, tsel = sel[visible], Xs[visible], Ys[visible], dx[visible], dy[visible], dz[visible], tsel[visible]
    if sel.size == 0:
        return nada
    n = nbuf.reshape(-1, 3)[sel]
    c = cbuf.reshape(-1, 3)[sel]
    b = bbuf.ravel()[sel]
    met = mbuf.ravel()[sel]
    isel = ibuf.ravel()[sel]
    for idx, p in enumerate(primitivas):  # calcomanías (caras) sobre el albedo, antes de iluminar
        if p.textura is None:
            continue
        m = isel == idx
        if not m.any():
            continue
        nn = n[m]
        cy_, sy_ = math.cos(p.giro_textura), math.sin(p.giro_textura)
        cp_, sp_ = math.cos(p.cabeceo_textura), math.sin(p.cabeceo_textura)
        nx = nn[:, 0] * cy_ - nn[:, 2] * sy_
        nz = nn[:, 0] * sy_ + nn[:, 2] * cy_
        ny = nn[:, 1] * cp_ + nz * sp_
        nz = -nn[:, 1] * sp_ + nz * cp_
        tex = _muestrear(p.textura, nx * 0.5 + 0.5, ny * 0.5 + 0.5)
        a_ = tex[:, 3:4] * np.clip((nz[:, None] - 0.02) / 0.08, 0, 1)
        c[m] = c[m] * (1 - a_) + tex[:, :3] * a_
    vista = -np.stack([dx, dy, dz], -1)
    vista /= np.linalg.norm(vista, axis=-1, keepdims=True) + 1e-9
    nv = np.clip((n * vista).sum(-1), 0, 1)
    # luz envolvente (half-lambert) para el aspecto suave de juguete
    dif_p = np.clip(n @ luz.principal * 0.5 + 0.5, 0, 1) ** 1.8
    dif_r = np.clip(n @ luz.relleno * 0.5 + 0.5, 0, 1) ** 2.0
    rim = np.clip(1 - nv, 0, 1) ** 3 * np.clip(-n[:, 1] * 0.5 + 0.6, 0, 1)
    medio = luz.principal[None, :] + vista
    medio /= np.linalg.norm(medio, axis=-1, keepdims=True) + 1e-9
    nm = np.clip((n * medio).sum(-1), 0, 1)
    esp = nm**24 * b
    # oclusión ambiental en espacio de pantalla (a 1/4 de resolución): lo que está detrás de sus vecinos se oscurece
    f = 4
    prof = np.full(Hp * Wp, np.nan, np.float32)
    prof[sel] = -tsel * cam.D
    pmin = float(np.nanmin(prof[sel]))
    prof = np.where(np.isnan(prof), pmin - 40, prof).reshape(Hp, Wp)
    hp4, wp4 = -(-Hp // f), -(-Wp // f)
    rell = np.pad(prof, ((0, hp4 * f - Hp), (0, wp4 * f - Wp)), mode="edge")
    peq = rell.reshape(hp4, f, wp4, f).mean(axis=(1, 3))
    pbl = desenfocar(peq, max(2, 10 * ss // f))
    pbl = np.repeat(np.repeat(pbl, f, 0), f, 1)[:Hp, :Wp].ravel()[sel]
    ao = np.clip((pbl - prof.ravel()[sel]) / 18.0, 0, 1) * oclusion
    luzc = luz.ambiente[None, :] + dif_p[:, None] * luz.color_principal + dif_r[:, None] * luz.color_relleno
    rgb = c * luzc * (1 - ao[:, None]) + rim[:, None] * luz.contorno * 0.9 + esp[:, None] * 0.6
    if met.any():  # cromado: reflejo de un entorno vertical (techo claro, franja del horizonte, encimera)
        r = 2 * nv[:, None] * n - vista
        ry = -r[:, 1]
        entorno = 0.34 + 1.05 * np.clip((ry - 0.02) / 0.22, 0, 1) ** 0.7 + 0.35 * np.exp(-(((ry + 0.25) / 0.14) ** 2))
        entorno = entorno[:, None] * (luz.color_principal * 0.62 + luz.ambiente * 1.4)
        cromo = c * entorno * (1 - 0.5 * ao[:, None]) + (nm**60 * 1.4)[:, None]
        rgb = np.where(met[:, None], cromo, rgb)
    rgb = rgb / (1 + rgb * 0.25)  # compresión suave de altas luces
    rgb = np.clip(rgb ** (1 / 1.12), 0, 1)
    arr = np.zeros((Hp * Wp, 4), np.uint8)
    arr[sel, :3] = (rgb * 255).astype(np.uint8)
    arr[sel, 3] = 255
    img = Image.fromarray(arr.reshape(Hp, Wp, 4), "RGBA")
    if ss > 1:
        img = img.resize((Wp // ss, Hp // ss), Image.LANCZOS)
    if devolver_t:
        tfull = np.full(Hp * Wp, np.inf, np.float32)
        tfull[sel] = tsel
        tfull = tfull.reshape(Hp, Wp)
        if devolver_t == "completo":
            return img, x0, y0, tfull
        return img, x0, y0, tfull.reshape(Hp // ss, ss, Wp // ss, ss).min(axis=(1, 3))
    return img, x0, y0


def sombra_de(sprite, desplazamiento=(18, 14), radio=22, fuerza=0.45):
    """Sombra suave proyectada (silueta difuminada) para apoyar el objeto en la escena."""
    pad = radio * 3
    a = Image.new("L", (sprite.width + pad * 2, sprite.height + pad * 2), 0)
    a.paste(sprite.getchannel("A"), (pad + desplazamiento[0], pad + desplazamiento[1]))
    if radio >= 16:  # desenfoque grande a 1/4 de resolución (mucho más rápido, mismo aspecto)
        w, h = a.size
        a = a.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radio / 4)).resize((w, h), Image.BILINEAR)
    else:
        a = a.filter(ImageFilter.GaussianBlur(radio))
    a = a.point(lambda v: int(v * fuerza))
    s = Image.new("RGBA", a.size, (20, 10, 6, 0))
    s.putalpha(a)
    return s, pad
