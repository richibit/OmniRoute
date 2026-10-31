"""
Subtítulos karaoke de la fórmula: aparecen palabra a palabra en la parte
superior; la palabra que se está diciendo va en blanco y negrita, las ya dichas
en gris, y la palabra clave de cada frase lleva un recuadro azul de «selección»
con dos tiradores, como en un editor de diseño. También exporta SRT y VTT.
"""

import functools
import json

from PIL import Image, ImageDraw, ImageFilter

import config
from utilidades import fuente, limitar, normalizar_palabra, progreso, salida_cubica

CFG = config.SUBTITULOS
P = config.PALETA
FIN_FRASE = (".", "?", "!", "…")
PAUSA_CORTA = (",", ":", ";")


def _ancho(texto):
    return fuente("sub_activa", CFG["tamano"]).getlength(texto)


def _maquetar(textos):
    """Reparte palabras en líneas centradas. Devuelve [(linea, x_izquierda, ancho)] por palabra o None si no cabe."""
    espacio = _ancho(" ")
    lineas, actual, ancho_actual = [], [], 0.0
    for i, t in enumerate(textos):
        w = _ancho(t)
        nuevo = w if not actual else ancho_actual + espacio + w
        if actual and nuevo > CFG["ancho_max"]:
            lineas.append((actual, ancho_actual))
            actual, ancho_actual = [i], w
        else:
            actual.append(i)
            ancho_actual = nuevo
    if actual:
        lineas.append((actual, ancho_actual))
    if len(lineas) > CFG["max_lineas"]:
        return None
    pos = [None] * len(textos)
    for n, (indices, ancho_linea) in enumerate(lineas):
        x = (config.ANCHO - ancho_linea) / 2
        for i in indices:
            w = _ancho(textos[i])
            pos[i] = (n, x, w)
            x += w + espacio
    return pos


def paginar(palabras_por_bloque, bloques, duracion_total):
    """Agrupa las palabras en «páginas» de subtítulo (1–2 líneas) que cortan en la puntuación."""
    paginas = []
    for b in bloques:
        claves = {normalizar_palabra(k) for k in b.get("resaltar", [])}
        actual = []
        palabras = palabras_por_bloque[b["id"]]
        for i, p in enumerate(palabras):
            prueba = actual + [p]
            if actual and _maquetar([x["texto"] for x in prueba]) is None:
                paginas.append({"bloque": b["id"], "palabras": actual})
                actual = [p]
            else:
                actual = prueba
            texto = p["texto"].rstrip("»\"')")
            cierra = texto.endswith(FIN_FRASE) or (texto.endswith(PAUSA_CORTA) and len(actual) >= 3)
            if cierra or i == len(palabras) - 1:
                paginas.append({"bloque": b["id"], "palabras": actual})
                actual = []
        for pg in paginas:
            if pg["bloque"] == b["id"]:
                for p in pg["palabras"]:
                    p["resaltar"] = normalizar_palabra(p["texto"]) in claves
    for i, pg in enumerate(paginas):
        pg["posiciones"] = _maquetar([p["texto"] for p in pg["palabras"]])
        pg["inicio"] = round(pg["palabras"][0]["inicio"] - CFG["adelanto"], 3)
    for i, pg in enumerate(paginas):
        # la última página se queda en pantalla durante el cierre (como en la fórmula)
        pg["fin"] = paginas[i + 1]["inicio"] if i + 1 < len(paginas) else duracion_total + 1.0
    return paginas


def _formato_srt(t):
    t = max(0.0, t)
    h, resto = divmod(t, 3600)
    m, s = divmod(resto, 60)
    ms = min(999, int(round((s - int(s)) * 1000)))
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def exportar(paginas, palabras_por_bloque, carpeta, duracion_total):
    """Escribe subtitulos.srt, subtitulos.vtt y palabras.json (tiempos por palabra)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    srt, vtt = [], ["WEBVTT", ""]
    for i, pg in enumerate(paginas, 1):
        texto = " ".join(p["texto"] for p in pg["palabras"])
        ini, fin = _formato_srt(pg["inicio"]), _formato_srt(min(pg["fin"], duracion_total))
        srt += [str(i), f"{ini} --> {fin}", texto, ""]
        vtt += [f"{ini.replace(',', '.')} --> {fin.replace(',', '.')}", texto, ""]
    (carpeta / "subtitulos.srt").write_text("\n".join(srt), encoding="utf-8")
    (carpeta / "subtitulos.vtt").write_text("\n".join(vtt), encoding="utf-8")
    (carpeta / "palabras.json").write_text(
        json.dumps({str(k): v for k, v in palabras_por_bloque.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
ALTO_CAPA = 260
Y0 = CFG["y_primera_linea"] - 80  # coordenada vertical de la capa en el fotograma


def _baseline(linea):
    return 80 + linea * CFG["interlineado"]


def _caja(pos, linea_y, escala=1.0):
    """Recuadro de selección alrededor de una palabra (coordenadas de la capa)."""
    _, x, w = pos
    tam = CFG["tamano"]
    cx, cy = x + w / 2, linea_y - tam * 0.36
    medio_w, medio_h = (w / 2 + 10) * escala, (tam * 0.52 + 2) * escala
    return cx - medio_w, cy - medio_h, cx + medio_w, cy + medio_h


@functools.lru_cache(maxsize=512)
def _capa(indice_pagina, n_visibles, actual_idx, cajas, textos, posiciones):
    """Capa con las palabras visibles (sin la más nueva, que se anima aparte)."""
    capa = Image.new("RGBA", (config.ANCHO, ALTO_CAPA), (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    for i in cajas:
        n, _, _ = posiciones[i]
        x0, y0, x1, y1 = _caja(posiciones[i], _baseline(n))
        d.rounded_rectangle([x0, y0, x1, y1], radius=6, fill=(30, 64, 175, 120), outline=P["seleccion"] + (255,), width=3)
    texto_capa = Image.new("RGBA", capa.size, (0, 0, 0, 0))
    dt = ImageDraw.Draw(texto_capa)
    for i in range(n_visibles):
        if i == actual_idx:
            continue
        n, x, _ = posiciones[i]
        dt.text((x, _baseline(n)), textos[i], font=fuente("sub_pasada", CFG["tamano"]), fill=P["gris_texto"] + (255,), anchor="ls")
    sombra = Image.new("RGBA", capa.size, (0, 0, 0, 0))
    sombra.putalpha(texto_capa.getchannel("A").filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.6)))
    capa.alpha_composite(sombra, (0, 3))
    capa.alpha_composite(texto_capa)
    for i in cajas:  # tiradores de la selección (encima del texto)
        n, _, _ = posiciones[i]
        x0, y0, x1, y1 = _caja(posiciones[i], _baseline(n))
        for hx, hy in ((x0, y0), (x1, y1)):
            d2 = ImageDraw.Draw(capa)
            d2.ellipse([hx - 7, hy - 7, hx + 7, hy + 7], fill=P["seleccion"] + (255,), outline=(235, 242, 255, 255), width=2)
    return capa


@functools.lru_cache(maxsize=256)
def _palabra_activa(texto):
    """Palabra que se está diciendo: blanca, negrita y con sombra suave."""
    f = fuente("sub_activa", CFG["tamano"])
    w = int(f.getlength(texto)) + 30
    img = Image.new("RGBA", (w, 130), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((15, 90), texto, font=f, fill=(255, 255, 255, 255), anchor="ls")
    sombra = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sombra.putalpha(img.getchannel("A").filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.65)))
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(sombra, (0, 3))
    out.alpha_composite(img)
    return out


def pagina_en(paginas, t):
    for i, pg in enumerate(paginas):
        if pg["inicio"] <= t < pg["fin"]:
            return i, pg
    return None, None


def dibujar(lienzo, paginas, t, atenuar=1.0):
    """Compone los subtítulos del instante t sobre el lienzo RGBA."""
    i, pg = pagina_en(paginas, t)
    if pg is None:
        return
    adelanto = CFG["adelanto"]
    visibles = [p for p in pg["palabras"] if p["inicio"] - adelanto <= t]
    if not visibles:
        return
    n = len(visibles)
    actual = n - 1
    cajas = tuple(k for k, p in enumerate(visibles) if p.get("resaltar"))
    textos = tuple(p["texto"] for p in pg["palabras"])
    posiciones = tuple(tuple(x) for x in pg["posiciones"])  # el plan llega de JSON con listas: la caché necesita tuplas
    capa = _capa(i, n, actual, cajas, textos, posiciones)
    if atenuar < 1.0:
        capa = capa.copy()
        capa.putalpha(capa.getchannel("A").point(lambda v: int(v * atenuar)))
    lienzo.alpha_composite(capa, (0, Y0))
    # palabra activa con una aparición rápida (sube 8 px y gana opacidad en 90 ms)
    p = visibles[actual]
    k = salida_cubica(progreso(t, p["inicio"] - adelanto, p["inicio"] - adelanto + 0.09))
    spr = _palabra_activa(p["texto"])
    if atenuar < 1.0 or k < 1:
        spr = spr.copy()
        spr.putalpha(spr.getchannel("A").point(lambda v: int(v * limitar(0.25 + 0.75 * k) * atenuar)))
    linea, x, _ = posiciones[actual]
    lienzo.alpha_composite(spr, (int(x - 15), int(Y0 + _baseline(linea) - 90 + (1 - k) * 8)))
