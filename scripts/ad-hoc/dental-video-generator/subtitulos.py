"""
Subtítulos dinámicos: agrupación en bloques de 2–3 palabras, exportación SRT/VTT
y render con la palabra activa en menta y las palabras clave resaltadas.
"""

import functools
import json

import config
from utilidades import normalizar_palabra, render_lineas, salida_rebote, progreso

CFG = config.SUBTITULOS


def agrupar(palabras_por_bloque, bloques):
    """
    Convierte las palabras alineadas en trozos de pantalla de 2–3 palabras.
    Corta siempre en signos de puntuación para que cada trozo sea legible solo.
    """
    trozos = []
    for b in bloques:
        palabras = palabras_por_bloque.get(b["id"], [])
        clave = {normalizar_palabra(k) for k in b.get("palabras_clave", [])}
        actual = []
        for i, p in enumerate(palabras):
            actual.append(dict(p, clave=normalizar_palabra(p["texto"]) in clave))
            texto = " ".join(x["texto"] for x in actual)
            fin_frase = p["texto"][-1:] in ".,:;?!…"
            siguiente = palabras[i + 1]["texto"] if i + 1 < len(palabras) else ""
            demasiado_largo = len(texto) + 1 + len(siguiente) > CFG["max_caracteres"]
            if fin_frase or len(actual) >= CFG["max_palabras"] or demasiado_largo or not siguiente:
                trozos.append({"bloque": b["id"], "palabras": actual})
                actual = []
    # Cada trozo dura desde su primera palabra hasta que empieza el siguiente (máx. 0,6 s de cola).
    for i, tr in enumerate(trozos):
        tr["inicio"] = tr["palabras"][0]["inicio"]
        fin_natural = tr["palabras"][-1]["fin"] + 0.6
        if i + 1 < len(trozos):
            fin_natural = min(fin_natural, trozos[i + 1]["palabras"][0]["inicio"])
        tr["fin"] = max(fin_natural, tr["palabras"][-1]["fin"])
    return trozos


def _formato_srt(t):
    h, resto = divmod(t, 3600)
    m, s = divmod(resto, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}".replace(",1000", ",999")


def exportar(trozos, palabras_por_bloque, carpeta):
    """Escribe subtitulos.srt, subtitulos.vtt y palabras.json (tiempos por palabra)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    srt, vtt = [], ["WEBVTT", ""]
    for i, tr in enumerate(trozos, 1):
        texto = " ".join(p["texto"] for p in tr["palabras"])
        srt += [str(i), f"{_formato_srt(tr['inicio'])} --> {_formato_srt(tr['fin'])}", texto, ""]
        vtt += [
            f"{_formato_srt(tr['inicio']).replace(',', '.')} --> {_formato_srt(tr['fin']).replace(',', '.')}",
            texto,
            "",
        ]
    (carpeta / "subtitulos.srt").write_text("\n".join(srt), encoding="utf-8")
    (carpeta / "subtitulos.vtt").write_text("\n".join(vtt), encoding="utf-8")
    (carpeta / "palabras.json").write_text(
        json.dumps({str(k): v for k, v in palabras_por_bloque.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


@functools.lru_cache(maxsize=256)
def _sprite_trozo(indice, activa, textos, claves):
    paleta = config.PALETA
    segmentos = []
    for j, (txt, es_clave) in enumerate(zip(textos, claves)):
        if CFG["mayusculas"]:
            txt = txt.upper()
        if j == activa:
            color = paleta["menta"]
        elif es_clave:
            color = paleta["ambar"]
        else:
            color = paleta["blanco"]
        segmentos.append((txt + (" " if j < len(textos) - 1 else ""), color))
    return render_lineas(
        [segmentos],
        "titulo",
        CFG["tamano"],
        contorno=CFG["contorno"],
        color_contorno=config.PALETA["tinta"],
        sombra=6,
    )


def sprite_en(trozos, t):
    """Devuelve (sprite, escala) del subtítulo visible en el instante t, o (None, 0)."""
    for i, tr in enumerate(trozos):
        if tr["inicio"] <= t < tr["fin"]:
            activa = 0
            for j, p in enumerate(tr["palabras"]):
                if t >= p["inicio"]:
                    activa = j
            textos = tuple(p["texto"] for p in tr["palabras"])
            claves = tuple(p.get("clave", False) for p in tr["palabras"])
            escala = 0.82 + 0.18 * salida_rebote(progreso(t, tr["inicio"], tr["inicio"] + 0.14))
            return _sprite_trozo(i, activa, textos, claves), escala
    return None, 0.0
