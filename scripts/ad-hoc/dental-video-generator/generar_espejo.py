#!/usr/bin/env python3
"""
Generador de «La última vez que tu mamá te cepilló los dientes» (contado por el
espejo del baño): 1080×1920, 30 FPS, H.264 + AAC, ≈75 s, sin voz en off y sin
marca de agua durante el video (el logo y el @usuario solo aparecen al final).

Uso:
    python generar_espejo.py                       # video final en export/
    python generar_espejo.py --calidad borrador    # render rápido para revisar
    python generar_espejo.py --previsualizar 1 21 63   # PNG de esos instantes en temp/previews_espejo/
    python generar_espejo.py --desde 48 --hasta 58     # solo un tramo

Flujo: comprobaciones → música y efectos (−14 LUFS) → fotogramas en paralelo
(motor 3D tipo juguete de juguete3d.py) → unión y multiplexado → verificación.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generar_video as base  # noqa: E402  (comprueba las dependencias al importarse)

import audio  # noqa: E402
import config  # noqa: E402
import guion_espejo as GE  # noqa: E402
from utilidades import ErrorProduccion, ruta_fuente  # noqa: E402

ARCHIVO = "la_ultima_vez_que_tu_mama.mp4"


def informar(msg=""):
    print(msg, flush=True)


def validar_guion():
    """Escenas contiguas de 0 al cierre, efectos existentes y subtítulos de una sola línea."""
    import musica

    previa = 0.0
    for e in GE.ESCENAS:
        if abs(e["inicio"] - previa) > 1e-6:
            raise ErrorProduccion(f"La escena {e['id']} empieza en {e['inicio']} s, pero la anterior acaba en {previa} s.")
        if e["fin"] <= e["inicio"]:
            raise ErrorProduccion(f"La escena {e['id']} termina antes de empezar.")
        for nombre, _, _ in e.get("sonidos", []):
            if nombre not in musica.SFX and not audio.buscar_archivo(config.DIR_SFX, nombre, audio.EXT_AUDIO):
                raise ErrorProduccion(f"Sonido desconocido «{nombre}» en la escena {e['id']}.")
        if e.get("subtitulo") and len(e["subtitulo"]) > 52:
            raise ErrorProduccion(f"El subtítulo de la escena {e['id']} es demasiado largo para una línea.")
        previa = e["fin"]
    if abs(previa - GE.CIERRE_INICIO) > 1e-6 or GE.LOGO_INICIO >= GE.DURACION:
        raise ErrorProduccion("Las escenas deben acabar en CIERRE_INICIO y el logo debe aparecer antes del final.")


def main():
    p = argparse.ArgumentParser(description="Genera «La última vez que tu mamá te cepilló los dientes» (1080×1920, 30 FPS).")
    p.add_argument("--calidad", choices=list(config.CALIDADES), default="final")
    p.add_argument("--hilos", type=int, default=max(1, os.cpu_count() or 2))
    p.add_argument("--desde", type=float, default=0.0)
    p.add_argument("--hasta", type=float, default=None)
    p.add_argument("--previsualizar", type=float, nargs="+", metavar="SEG")
    p.add_argument("--descargar-fuentes", action="store_true", help="descarga Newsreader, Inter, Caveat… (OFL)")
    p.add_argument("--salida", default=ARCHIVO)
    args = p.parse_args()
    inicio = time.time()
    informar("=" * 64)
    informar(" La última vez que tu mamá te cepilló los dientes · contado por el espejo")
    informar("=" * 64)
    informar("\n[1/4] Comprobaciones")
    base.crear_carpetas()
    if args.descargar_fuentes:
        base.descargar_fuentes()
    audio.ffmpeg_exe()
    validar_guion()
    for rol in ("serif", "serif_italica", "versalitas", "manuscrita"):
        ruta = ruta_fuente(rol)
        informar(f"  · {rol:<14} {Path(ruta).name if ruta else 'fuente integrada (ver README)'}")
    informar(f"  · Guion válido: {len(GE.ESCENAS)} escenas + cierre, {GE.DURACION:.0f} s, sin marca de agua")

    duracion = GE.DURACION
    plan = {"duracion": duracion}
    ruta_plan = config.DIR_TEMP / "plan_espejo.json"
    ruta_plan.write_text(json.dumps(plan), encoding="utf-8")

    if args.previsualizar:
        import escenas_espejo
        from PIL import Image

        carpeta = config.DIR_TEMP / "previews_espejo"
        carpeta.mkdir(parents=True, exist_ok=True)
        rend = escenas_espejo.Renderizador(plan)
        for seg in args.previsualizar:
            ruta = carpeta / f"frame_{seg:06.2f}.png"
            Image.fromarray(rend.fotograma(min(seg, duracion - 1 / config.FPS))).save(ruta)
            informar(f"  · {ruta.relative_to(config.RAIZ)}")
        return

    informar("\n[2/4] Música y efectos")
    import musica

    mezcla = musica.mezcla(duracion, informar)
    ruta_audio = config.DIR_TEMP / "mezcla_espejo.wav"
    audio.guardar_wav(ruta_audio, mezcla * (0.98 / (abs(mezcla).max() + 1e-9)))
    ganancia = config.LOUDNESS_OBJETIVO_LUFS - audio.medir_lufs(ruta_audio)
    informar(f"  · Ganancia para {config.LOUDNESS_OBJETIVO_LUFS:.0f} LUFS: {ganancia:+.1f} dB")

    t_fin = duracion if args.hasta is None else min(args.hasta, duracion)
    f_ini, f_fin = int(round(args.desde * config.FPS)), int(round(t_fin * config.FPS))
    if f_fin <= f_ini:
        raise ErrorProduccion("--hasta debe ser mayor que --desde")
    parcial = f_ini > 0 or f_fin < int(round(duracion * config.FPS))
    informar(f"\n[3/4] Render de video ({f_fin - f_ini} fotogramas, {args.hilos} procesos, calidad {args.calidad})")
    video = base.renderizar_video(ruta_plan, f_ini, f_fin, args.calidad, args.hilos, modulo="escenas_espejo")

    informar("\n[4/4] Audio + exportación")
    if parcial:
        salida = config.DIR_EXPORT / f"espejo_parcial_{args.desde:05.1f}-{t_fin:05.1f}.mp4"
        base.multiplexar(video, ruta_audio, salida, ganancia, f_ini / config.FPS, (f_fin - f_ini) / config.FPS)
        base.verificar_salida(salida, (f_fin - f_ini) / config.FPS)
    else:
        salida = config.DIR_EXPORT / args.salida
        base.multiplexar(video, ruta_audio, salida, ganancia)
        base.verificar_salida(salida, duracion)
    informar(f"\nListo en {time.time() - inicio:.0f} s → {salida}")


if __name__ == "__main__":
    try:
        main()
    except ErrorProduccion as exc:
        sys.exit(f"\n✖ {exc}")
    except KeyboardInterrupt:
        sys.exit("\n✖ Cancelado por el usuario.")
