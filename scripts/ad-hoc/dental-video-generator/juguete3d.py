"""
Motor 3D mínimo para personajes y objetos «tipo juguete» (acabado mate).

Proyección ortográfica sobre el plano de la pantalla: x a la derecha, y hacia
abajo (en píxeles) y z hacia la cámara. Cada primitiva (esfera, elipsoide,
cápsula aplanable) se interseca de forma analítica con los rayos de cámara,
se resuelve la visibilidad con un z-buffer y se ilumina con luz principal
envolvente, luz de relleno, luz de contorno, ambiente y oclusión ambiental en
espacio de pantalla. Todo con numpy; se renderiza a 2× y se reduce (antialias).
"""

import math

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Iluminación
# ---------------------------------------------------------------------------
def _norm(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


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


# Noche: luz cálida sobre el espejo (de frente y arriba) + luz del pasillo por detrás.
NOCHE = Iluminacion((-0.35, -0.55, 0.75), (1.05, 0.80, 0.55), (0.8, -0.1, 0.4), (0.28, 0.22, 0.30), (0.30, 0.22, 0.18), (0.55, 0.38, 0.20))
# Mañana: luz fría que entra por la ventana (izquierda) + rebote cálido suave.
MANANA = Iluminacion((-0.7, -0.45, 0.55), (1.02, 0.99, 0.94), (0.75, 0.05, 0.45), (0.26, 0.30, 0.38), (0.22, 0.26, 0.31), (0.36, 0.42, 0.52))


# ---------------------------------------------------------------------------
# Primitivas
# ---------------------------------------------------------------------------
class Primitiva:
    def __init__(self, color, brillo=0.08):
        self.color = np.asarray(color, np.float32) / 255.0
        self.brillo = brillo
        self.limite_y = None  # coordenada y de pantalla a partir de la cual no se ve (detrás de la encimera)
        self.textura = None  # array H×W×4 (0–1) proyectado sobre el hemisferio frontal (caras)
        self.giro_textura = 0.0  # giro de la cabeza (radianes): la cara se desplaza con él
        self.cabeceo_textura = 0.0  # mirar arriba/abajo (radianes)


class Elipsoide(Primitiva):
    """Elipsoide (o esfera) con centro c=(x, y, z), radios r=(rx, ry, rz) y giro en el plano de pantalla."""

    def __init__(self, c, r, color, giro=0.0, brillo=0.08):
        super().__init__(color, brillo)
        self.c = np.asarray(c, np.float32)
        self.r = np.asarray(r if hasattr(r, "__len__") else (r, r, r), np.float32)
        self.giro = math.radians(giro)

    def caja(self):
        m = max(self.r[0], self.r[1])
        return self.c[0] - m, self.c[1] - m, self.c[0] + m, self.c[1] + m

    def intersecar(self, X, Y):
        dx, dy = X - self.c[0], Y - self.c[1]
        if self.giro:
            cg, sg = math.cos(self.giro), math.sin(self.giro)
            dx, dy = dx * cg + dy * sg, -dx * sg + dy * cg
        u, v = dx / self.r[0], dy / self.r[1]
        t = 1 - u * u - v * v
        dentro = t > 0
        s = np.sqrt(np.clip(t, 0, None))
        z = self.c[2] + self.r[2] * s
        nx, ny, nz = u / self.r[0], v / self.r[1], s / self.r[2]
        if self.giro:
            nx, ny = nx * cg - ny * sg, nx * sg + ny * cg
        n = np.stack([nx, ny, nz], -1)
        return dentro, z, n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)


class Capsula(Primitiva):
    """Cápsula entre a y b (3D) de radio r; `aplanado` < 1 la achata en profundidad (torsos)."""

    def __init__(self, a, b, r, color, aplanado=1.0, brillo=0.08):
        super().__init__(color, brillo)
        self.a = np.asarray(a, np.float32)
        self.b = np.asarray(b, np.float32)
        self.r = float(r)
        self.k = float(aplanado)

    def caja(self):
        return (
            min(self.a[0], self.b[0]) - self.r,
            min(self.a[1], self.b[1]) - self.r,
            max(self.a[0], self.b[0]) + self.r,
            max(self.a[1], self.b[1]) + self.r,
        )

    def intersecar(self, X, Y):
        k = self.k
        a = self.a / np.array([1, 1, k], np.float32)
        b = self.b / np.array([1, 1, k], np.float32)
        r = self.r
        d = b - a
        L = float(np.linalg.norm(d)) + 1e-6
        u = d / L
        w0x, w0y, w0z = X - a[0], Y - a[1], -a[2]
        wu = w0x * u[0] + w0y * u[1] + w0z * u[2]
        A = 1 - u[2] ** 2
        mejor_z = np.full(X.shape, -1e9, np.float32)
        normal = np.zeros(X.shape + (3,), np.float32)
        if A > 1e-4:
            B = 2 * (w0z - wu * u[2])
            C = w0x**2 + w0y**2 + w0z**2 - wu**2 - r * r
            disc = B * B - 4 * A * C
            ok = disc >= 0
            z = (-B + np.sqrt(np.clip(disc, 0, None))) / (2 * A)
            s = wu + z * u[2]
            ok &= (s >= 0) & (s <= L)
            px, py, pz = w0x - s * u[0], w0y - s * u[1], w0z + z - s * u[2]
            mejor_z = np.where(ok, z, mejor_z)
            for i, comp in enumerate((px, py, pz)):
                normal[..., i] = np.where(ok, comp / r, normal[..., i])
        for centro in (a, b):  # tapas esféricas
            dx, dy = X - centro[0], Y - centro[1]
            t = r * r - dx * dx - dy * dy
            ok = t > 0
            z = centro[2] + np.sqrt(np.clip(t, 0, None))
            mejor = ok & (z > mejor_z)
            mejor_z = np.where(mejor, z, mejor_z)
            for i, comp in enumerate((dx, dy, z - centro[2])):
                normal[..., i] = np.where(mejor, comp / r, normal[..., i])
        dentro = mejor_z > -1e8
        normal[..., 2] /= k
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-9
        return dentro, mejor_z * k, normal


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
def render(primitivas, luz, ss=2, margen=24, oclusion=0.55, sombra_propia=True):
    """
    Renderiza las primitivas y devuelve (imagen RGBA, x0, y0) con la esquina superior
    izquierda de la imagen en coordenadas de pantalla.
    """
    if not primitivas:
        return None, 0, 0
    cajas = np.array([p.caja() for p in primitivas], np.float32)
    x0, y0 = math.floor(cajas[:, 0].min()) - margen, math.floor(cajas[:, 1].min()) - margen
    x1, y1 = math.ceil(cajas[:, 2].max()) + margen, math.ceil(cajas[:, 3].max()) + margen
    W, H = (x1 - x0) * ss, (y1 - y0) * ss
    zbuf = np.full((H, W), -1e9, np.float32)
    nbuf = np.zeros((H, W, 3), np.float32)
    cbuf = np.zeros((H, W, 3), np.float32)
    bbuf = np.zeros((H, W), np.float32)
    ibuf = np.full((H, W), -1, np.int32)
    for idx, p in enumerate(primitivas):
        bx0, by0, bx1, by1 = p.caja()
        i0, i1 = max(0, int((by0 - y0) * ss) - 1), min(H, int((by1 - y0) * ss) + 2)
        j0, j1 = max(0, int((bx0 - x0) * ss) - 1), min(W, int((bx1 - x0) * ss) + 2)
        if i1 <= i0 or j1 <= j0:
            continue
        Y, X = np.mgrid[i0:i1, j0:j1].astype(np.float32)
        X = X / ss + x0 + 0.5 / ss
        Y = Y / ss + y0 + 0.5 / ss
        dentro, z, n = p.intersecar(X, Y)
        if p.limite_y is not None:
            dentro &= Y < p.limite_y
        zona = zbuf[i0:i1, j0:j1]
        gana = dentro & (z > zona)
        zona[gana] = z[gana]
        nbuf[i0:i1, j0:j1][gana] = n[gana]
        cbuf[i0:i1, j0:j1][gana] = p.color
        bbuf[i0:i1, j0:j1][gana] = p.brillo
        ibuf[i0:i1, j0:j1][gana] = idx
    for idx, p in enumerate(primitivas):  # calcomanías (caras) sobre el albedo, antes de iluminar
        if p.textura is None:
            continue
        m = ibuf == idx
        if not m.any():
            continue
        n = nbuf[m]
        cy_, sy_ = math.cos(p.giro_textura), math.sin(p.giro_textura)
        cp_, sp_ = math.cos(p.cabeceo_textura), math.sin(p.cabeceo_textura)
        nx = n[:, 0] * cy_ - n[:, 2] * sy_
        nz = n[:, 0] * sy_ + n[:, 2] * cy_
        ny = n[:, 1] * cp_ + nz * sp_
        nz = -n[:, 1] * sp_ + nz * cp_
        th, tw = p.textura.shape[:2]
        u = np.clip(((nx * 0.5 + 0.5) * (tw - 1)).astype(int), 0, tw - 1)
        v = np.clip(((ny * 0.5 + 0.5) * (th - 1)).astype(int), 0, th - 1)
        tex = p.textura[v, u]
        a_ = tex[:, 3:4] * (nz[:, None] > 0.05)
        cbuf[m] = cbuf[m] * (1 - a_) + tex[:, :3] * a_
    alfa = zbuf > -1e8
    if not alfa.any():
        return None, 0, 0
    n = nbuf
    # luz envolvente (half-lambert) para el aspecto suave de juguete
    dif_p = np.clip(n @ luz.principal * 0.5 + 0.5, 0, 1) ** 1.8
    dif_r = np.clip(n @ luz.relleno * 0.5 + 0.5, 0, 1) ** 2.0
    rim = np.clip(1 - n[..., 2], 0, 1) ** 3 * np.clip(-n[..., 1] * 0.5 + 0.6, 0, 1)
    medio = _norm(luz.principal + np.array([0, 0, 1], np.float32))
    esp = np.clip(n @ medio, 0, 1) ** 24 * bbuf
    # oclusión ambiental en espacio de pantalla: lo que está detrás de sus vecinos se oscurece
    zmin = zbuf[alfa].min()
    zrell = np.where(alfa, zbuf, zmin - 40)
    zbl = desenfocar(zrell, 10 * ss)
    ao = np.clip((zbl - zrell) / 18.0, 0, 1) * oclusion
    luzc = luz.ambiente[None, None, :] + dif_p[..., None] * luz.color_principal + dif_r[..., None] * luz.color_relleno
    rgb = cbuf * luzc * (1 - ao[..., None]) + rim[..., None] * luz.contorno * 0.9 + esp[..., None] * 0.6
    rgb = rgb / (1 + rgb * 0.25)  # compresión suave de altas luces
    rgb = np.clip(rgb ** (1 / 1.12), 0, 1)
    arr = np.zeros((H, W, 4), np.uint8)
    arr[..., :3] = (rgb * 255).astype(np.uint8)
    arr[..., 3] = alfa.astype(np.uint8) * 255
    img = Image.fromarray(arr, "RGBA")
    if ss > 1:
        img = img.resize((W // ss, H // ss), Image.LANCZOS)
    return img, x0, y0


def sombra_de(sprite, desplazamiento=(18, 14), radio=22, fuerza=0.45):
    """Sombra suave proyectada (silueta difuminada) para apoyar el objeto en la escena."""
    pad = radio * 3
    a = Image.new("L", (sprite.width + pad * 2, sprite.height + pad * 2), 0)
    a.paste(sprite.getchannel("A"), (pad + desplazamiento[0], pad + desplazamiento[1]))
    a = a.filter(ImageFilter.GaussianBlur(radio)).point(lambda v: int(v * fuerza))
    s = Image.new("RGBA", a.size, (20, 10, 6, 0))
    s.putalpha(a)
    return s, pad
