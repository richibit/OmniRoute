#!/usr/bin/env python3
"""
Revisión automática de choques del video «La última vez que tu mamá…».

Para cada fotograma pide a las escenas sus primitivas (sin renderizar) y mide
cuánto se mete cada pieza dentro de otra con funciones de distancia con signo
(SDF) de elipsoides, cápsulas, cajas y cilindros (el vaso es hueco). Revisa:

- partes del cuerpo de personas distintas (el brazo de mamá y el cuello del niño…),
- partes no contiguas de la misma persona (la mano dentro de la propia cabeza…),
- objetos contra personas y contra otros objetos,
- la encimera (nada atraviesa la superficie salvo dentro del hueco del lavabo)
  y la pared del espejo.

Se permiten los contactos que tienen sentido (el cepillo dentro del puño que lo
agarra, la cabeza del cepillo dentro de la boca, el lápiz entre los dedos…):
cada objeto declara con quién puede tocarse en `ignorar`.

Uso:
    python revision_choques.py                    # todo el video, cada 2 fotogramas
    python revision_choques.py --desde 15 --hasta 19 --paso 1
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import escenario as S  # noqa: E402
import juguete3d as J  # noqa: E402

TOL = 2.5  # px que se toleran de contacto entre piezas
TOL_MANO = 5.0  # una mano que se apoya en algo puede hundirse un poco (piel blanda)

TRONCO = ("cabeza", "cuello", "torso", "cadera")  # un solo cuerpo blando: sus piezas se funden entre sí
CONTIGUAS = {frozenset((a, b)) for a in TRONCO for b in TRONCO if a != b} | {
    frozenset(("cadera", "pierna_izq")),
    frozenset(("cadera", "pierna_der")),
    frozenset(("torso", "pierna_izq")),
    frozenset(("torso", "pierna_der")),
    frozenset(("pierna_izq", "pierna_der")),
}
for _l in ("izq", "der"):
    CONTIGUAS |= {frozenset(("torso", f"brazo_{_l}_1")), frozenset((f"brazo_{_l}_1", f"brazo_{_l}_2")), frozenset((f"brazo_{_l}_2", f"mano_{_l}"))}


# ---------------------------------------------------------------------------
# Muestras de superficie y distancias con signo
# ---------------------------------------------------------------------------
def _fibonacci(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1).astype(np.float32)


ESFERA = _fibonacci(72)


def _base_perp(u):
    ref = np.array([0, 0, 1], np.float32) if abs(u[2]) < 0.9 else np.array([1, 0, 0], np.float32)
    e1 = J._norm(np.cross(u, ref))
    return e1, np.cross(u, e1)


def muestras(p):
    if isinstance(p, J.Elipsoide):
        return p.c + (ESFERA * p.r) @ p.rot.T
    if isinstance(p, J.Capsula):
        k = p.k
        a, b = p.a / k, p.b / k
        u = b - a
        L = float(np.linalg.norm(u))
        u = u / L if L > 1e-6 else np.array([0, 1, 0], np.float32)
        e1, e2 = _base_perp(u)
        ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        anillo = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
        ts = np.linspace(0, 1, max(3, min(12, int(L / max(p.r, 1.0)) + 3)))
        pts = [(a + u * L * t) + anillo * p.r for t in ts]
        for extremo, signo in ((a, -1), (b, 1)):
            pts.append(extremo + (anillo * 0.7071 + u * signo * 0.7071) * p.r)
            pts.append((extremo + u * signo * p.r)[None, :])
        return np.concatenate(pts, 0) * k
    if isinstance(p, J.Caja):
        g = np.linspace(-1, 1, 5)
        A, B = np.meshgrid(g, g)
        A, B = A.ravel(), B.ravel()
        caras = []
        for eje in range(3):
            o1, o2 = [i for i in range(3) if i != eje]
            for signo in (-1, 1):
                q = np.zeros((A.size, 3), np.float32)
                q[:, eje] = signo
                q[:, o1], q[:, o2] = A, B
                caras.append(q)
        q = np.concatenate(caras, 0) * p.h
        return p.c + q @ p.rot.T
    if isinstance(p, J.Cilindro):
        ang = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        pts = []
        for y in (-p.hy, 0.0, p.hy):
            pts.append(np.stack([np.cos(ang) * p.r, np.full_like(ang, y), np.sin(ang) * p.r], 1))
        for y in (-p.hy, p.hy):
            for f in (0.0, 0.5):
                pts.append(np.stack([np.cos(ang) * p.r * f, np.full_like(ang, y), np.sin(ang) * p.r * f], 1))
        q = np.concatenate(pts, 0).astype(np.float32)
        return p.c + q @ p.rot.T
    return np.zeros((0, 3), np.float32)


def _sdf_cilindro(q, r, hy):
    dr = np.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2) - r
    dy = np.abs(q[:, 1]) - hy
    fuera = np.sqrt(np.maximum(dr, 0) ** 2 + np.maximum(dy, 0) ** 2)
    return fuera + np.minimum(np.maximum(dr, dy), 0)


def sdf(p, pts):
    """Distancia con signo (negativa dentro) de los puntos a la primitiva."""
    if isinstance(p, J.Elipsoide):
        q = (pts - p.c) @ p.rot
        k0 = np.linalg.norm(q / p.r, axis=1)
        k1 = np.linalg.norm(q / (p.r * p.r), axis=1)
        return k0 * (k0 - 1) / np.maximum(k1, 1e-9)
    if isinstance(p, J.Capsula):
        k = p.k
        q = pts / k
        a, b = p.a / k, p.b / k
        ba = b - a
        t = np.clip(((q - a) @ ba) / max(float(ba @ ba), 1e-9), 0, 1)
        d = np.linalg.norm(q - (a + t[:, None] * ba), axis=1) - p.r
        return d * float(k.min())
    if isinstance(p, J.Caja):
        q = np.abs((pts - p.c) @ p.rot) - p.h
        fuera = np.linalg.norm(np.maximum(q, 0), axis=1)
        return fuera + np.minimum(q.max(axis=1), 0)
    if isinstance(p, J.Cilindro):
        q = (pts - p.c) @ p.rot
        exterior = _sdf_cilindro(q, p.r, p.hy)
        if p.abierto:  # vaso: hueco por dentro (pared de 5 px y fondo de 6 px), abierto por arriba (−y local)
            pared, fondo = 5.0, 6.0
            y0, y1 = -p.hy - 80.0, p.hy - fondo
            qc = q.copy()
            qc[:, 1] -= (y0 + y1) / 2
            hueco = _sdf_cilindro(qc, p.r - pared, (y1 - y0) / 2)
            return np.maximum(exterior, -hueco)
        return exterior
    return np.full(len(pts), np.inf, np.float32)


def caja3d(p):
    mn, mx = p.aabb()
    return np.asarray(mn, np.float32), np.asarray(mx, np.float32)


# ---------------------------------------------------------------------------
# Reglas
# ---------------------------------------------------------------------------
def info(p):
    return getattr(p, "dueno", "escena"), getattr(p, "grupo", "objeto")


def permitido(pa, pb):
    da, ga = info(pa)
    db, gb = info(pb)
    if da == db:
        if ga == gb or frozenset((ga, gb)) in CONTIGUAS:
            return True
        if getattr(pa, "es_objeto", False) and getattr(pb, "es_objeto", False):
            return True  # piezas del mismo objeto
    for x, y in ((pa, pb), (pb, pa)):
        for dueno, prefijo in getattr(x, "ignorar", ()):
            yd, yg = info(y)
            if yd == dueno and yg.startswith(prefijo):
                return True
    return False


def tolerancia(pa, pb):
    ga, gb = info(pa)[1], info(pb)[1]
    return TOL_MANO if ga.startswith("mano") or gb.startswith("mano") else TOL


def dentro_lavabo(pts):
    L = S.LAVABO
    e = ((pts[:, 0] - L["x"]) / L["rx"]) ** 2 + ((pts[:, 2] - L["z"]) / L["rz"]) ** 2
    prof = ((pts[:, 1] - S.ENCIMERA_Y) / L["hondo"]) ** 2
    return (e + prof) < 1.0


def revisar(prims):
    """Lista de choques [(profundidad, a, b)] de un fotograma."""
    prims = [p for p in prims if p is not None]
    datos = [(p, muestras(p), *caja3d(p)) for p in prims]
    choques = {}
    for i, (pb, _, mnb, mxb) in enumerate(datos):
        for j, (pa, sa, mna, mxa) in enumerate(datos):
            if i == j or len(sa) == 0:
                continue
            if np.any(mna > mxb + 1) or np.any(mxa < mnb - 1):
                continue
            if permitido(pa, pb):
                continue
            d = sdf(pb, sa)
            prof = float(-d.min())
            tol = tolerancia(pa, pb)
            if prof > tol:
                clave = tuple(sorted((info(pa), info(pb))))
                if prof > choques.get(clave, (0,))[0]:
                    choques[clave] = (prof, info(pa), info(pb))
    # la encimera (y la pared del espejo) para objetos y brazos/manos
    for p, sp, _, _ in datos:
        dueno, grupo = info(p)
        es_bebe_sentado = grupo.startswith("pierna") and getattr(p, "sentado", False)
        if not (getattr(p, "es_objeto", False) or grupo.startswith(("brazo", "mano")) or es_bebe_sentado):
            continue  # las piernas de quien está de pie quedan detrás del mueble y no se ven
        if getattr(p, "sobre_encimera_ok", False):
            continue
        sobre = sp[:, 2] > S.ENCIMERA_Z0 + 1
        bajo = sp[:, 1] > S.ENCIMERA_Y + TOL
        mal = sobre & bajo & ~dentro_lavabo(sp)
        if mal.any():
            prof = float((sp[mal, 1] - S.ENCIMERA_Y).max())
            clave = (info(p), ("escena", "encimera"))
            if prof > choques.get(clave, (0,))[0]:
                choques[clave] = (prof, info(p), ("escena", "encimera"))
        espejo = sp[:, 2] > S.ESPEJO_Z + TOL
        if espejo.any() and not getattr(p, "en_espejo", False):
            prof = float((sp[espejo, 2] - S.ESPEJO_Z).max())
            clave = (info(p), ("escena", "espejo"))
            if prof > choques.get(clave, (0,))[0]:
                choques[clave] = (prof, info(p), ("escena", "espejo"))
    return sorted(choques.values(), key=lambda c: -c[0])


def main():
    ap = argparse.ArgumentParser(description="Revisa choques (piezas que se atraviesan) en el video del espejo.")
    ap.add_argument("--desde", type=float, default=0.0)
    ap.add_argument("--hasta", type=float, default=66.0)
    ap.add_argument("--paso", type=int, default=2, help="cada cuántos fotogramas revisar")
    ap.add_argument("--max", type=int, default=60, help="máximo de líneas de detalle")
    args = ap.parse_args()
    import escenas_espejo as EE

    r = EE.Renderizador()
    r.modo_revision = True
    fps = 30
    por_escena = defaultdict(lambda: [0, 0.0])
    detalle = []
    f0, f1 = int(round(args.desde * fps)), int(round(args.hasta * fps))
    for f in range(f0, f1, args.paso):
        t = f / fps
        e = r.escena_en(t)
        prims = r.cuadro(t, e)
        for prof, a, b in revisar(prims):
            por_escena[e["id"]][0] += 1
            por_escena[e["id"]][1] = max(por_escena[e["id"]][1], prof)
            detalle.append((t, e["id"], prof, a, b))
    alcance = defaultdict(float)
    for t, quien, lado, err in r.avisos:
        alcance[(r.escena_en(t)["id"], quien, lado)] = max(alcance[(r.escena_en(t)["id"], quien, lado)], err)
    print(f"Fotogramas revisados: {len(range(f0, f1, args.paso))}")
    grupos = {}
    for t, eid, prof, a, b in detalle:  # agrupa por escena y pareja: profundidad máxima y tramo de tiempo
        clave = (eid, a, b)
        g = grupos.setdefault(clave, [prof, t, t, t])
        if prof > g[0]:
            g[0], g[3] = prof, t
        g[1], g[2] = min(g[1], t), max(g[2], t)
    for (eid, a, b), (prof, t0, t1, tm) in sorted(grupos.items(), key=lambda x: -x[1][0])[: args.max]:
        print(f"  escena {eid:2d}  {prof:6.1f} px (t={tm:5.2f}, {t0:5.2f}–{t1:5.2f})  {a[0]}/{a[1]}  ↔  {b[0]}/{b[1]}")
    print("Resumen por escena (choques, máx px):", {k: (v[0], round(v[1], 1)) for k, v in sorted(por_escena.items())})
    print("Manos que no alcanzan (máx px):", {k: round(v, 1) for k, v in sorted(alcance.items())})
    return 1 if detalle or alcance else 0


if __name__ == "__main__":
    sys.exit(main())
