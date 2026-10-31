#!/usr/bin/env python3
"""
Generador del video vertical «24 horas sin cepillarse» con la fórmula de los
explicadores virales (1080×1920, 30 FPS, H.264 + AAC).

Uso rápido:
    python generar_video.py                       # video final (+ versión sin voz) en export/
    python generar_video.py --calidad borrador    # render rápido para revisar
    python generar_video.py --previsualizar 1 12 33.5   # PNG de esos instantes en temp/previews/
    python generar_video.py --desde 20 --hasta 31        # solo un tramo (export/parcial_...)
    python generar_video.py --descargar-voz       # descarga la voz neuronal Kokoro (una sola vez)

Flujo:
  1. Comprueba dependencias, carpetas, guion y recursos.
  2. Genera la narración (grabación propia o Kokoro) y obtiene el instante de cada palabra.
  3. Construye la línea de tiempo: cada bloque dura lo que su narración.
  4. Genera los subtítulos karaoke (y SRT/VTT) y la mezcla de audio (con y sin voz).
  5. Renderiza los fotogramas en paralelo (MoviePy/FFmpeg) y los une.
  6. Multiplexa el audio (AAC, −14 LUFS) y verifica los MP4.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

FALTAN = []
for modulo, paquete in (("numpy", "numpy"), ("PIL", "pillow"), ("moviepy", "moviepy")):
    try:
        __import__(modulo)
    except ImportError:
        FALTAN.append(paquete)
if FALTAN:
    sys.exit(f"Faltan dependencias de Python: {', '.join(FALTAN)}\nInstálalas con:  pip install -r requirements.txt")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import audio  # noqa: E402
import config  # noqa: E402
import guion  # noqa: E402
import subtitulos  # noqa: E402
from utilidades import ErrorProduccion, normalizar_palabra, ruta_emoji, ruta_fuente  # noqa: E402


def informar(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Comprobaciones
# ---------------------------------------------------------------------------
def crear_carpetas():
    for carpeta in (
        config.DIR_IMAGENES,
        config.DIR_VIDEOS,
        config.DIR_NARRACION,
        config.DIR_MUSICA,
        config.DIR_SFX,
        config.DIR_FUENTES,
        config.DIR_MODELOS_VOZ,
        config.DIR_SUBTITULOS,
        config.DIR_TEMP,
        config.DIR_EXPORT,
    ):
        carpeta.mkdir(parents=True, exist_ok=True)


def descargar_fuentes():
    """Descarga Montserrat, Inter, JetBrains Mono y Oswald (licencia OFL) a assets/fuentes/."""
    for nombre, url in config.URLS_FUENTES.items():
        destino = config.DIR_FUENTES / nombre
        if destino.exists():
            informar(f"  · {nombre} ya existe")
            continue
        try:
            informar(f"  · Descargando {nombre} …")
            with urllib.request.urlopen(url, timeout=60) as r:
                destino.write_bytes(r.read())
        except Exception as exc:
            informar(f"  ! No se pudo descargar {nombre}: {exc}. Se usará una fuente del sistema.")


def validar_guion():
    """Comprueba ids, escenas, sonidos y que cada palabra resaltada esté en su narración."""
    import escenas

    ids = set()
    escenas_validas = {
        "gancho", "negacion", "pelicula", "pioneras", "colonias", "dato_clave",
        "noche", "revelacion", "historia", "costumbre", "compartir",
    }  # fmt: skip
    for b in guion.BLOQUES:
        for clave in ("id", "escena", "narracion"):
            if clave not in b:
                raise ErrorProduccion(f"Al bloque {b.get('id', '?')} le falta el campo «{clave}» en guion.py")
        if b["id"] in ids:
            raise ErrorProduccion(f"Id de bloque repetido: {b['id']}")
        ids.add(b["id"])
        if b["escena"] not in escenas_validas:
            raise ErrorProduccion(f"El bloque {b['id']} usa una escena desconocida: {b['escena']}")
        normalizadas = {normalizar_palabra(p) for p in b["narracion"].split()}
        for palabra in b.get("resaltar", []):
            if normalizar_palabra(palabra) not in normalizadas:
                raise ErrorProduccion(f"El bloque {b['id']} resalta «{palabra}», que no está en su narración.")
        for s in b.get("sonidos", []):
            if s["nombre"] not in audio.GENERADORES_SFX and not audio.buscar_archivo(config.DIR_SFX, s["nombre"], audio.EXT_AUDIO):
                raise ErrorProduccion(f"Sonido desconocido «{s['nombre']}» en el bloque {b['id']}.")
    del escenas


def informe_recursos():
    informar("  Tipografías:")
    for rol in ("titulo", "sub_activa", "dato", "condensada", "reloj"):
        ruta = ruta_fuente(rol)
        informar(f"    {rol:<11} {Path(ruta).name if ruta else 'fuente integrada de Pillow (ver README)'}")
    informar(f"    {'emoji':<11} {Path(ruta_emoji()).name if ruta_emoji() else 'no disponible → símbolos'}")
    estado = "lista" if audio.kokoro_instalado() else "no instalada (python generar_video.py --descargar-voz)"
    tiempos = audio.rutas_kokoro()[1].exists()
    informar(f"  Voz Kokoro: {estado}{' · tiempos exactos por fonema' if tiempos else ''}")


# ---------------------------------------------------------------------------
# 2–4. Narración, línea de tiempo, subtítulos y mezcla
# ---------------------------------------------------------------------------
def preparar_audio_y_plan():
    import escenas

    informar("\n[2/6] Narración")
    narr = audio.preparar_narracion(guion.BLOQUES, config.DIR_TEMP, informar)
    lt = audio.construir_linea_de_tiempo(guion.BLOQUES, narr, config.FPS)
    informar(f"  · Duración del video: {lt['duracion']:.2f} s (voz hasta {lt['cierre']['inicio']:.2f} s + cierre con logo)")
    if lt["duracion"] > config.DURACION_MAXIMA:
        informar(f"  ! El video supera {config.DURACION_MAXIMA:.0f} s: considera acortar el guion.")

    informar("\n[3/6] Subtítulos karaoke")
    paginas = subtitulos.paginar(lt["palabras"], guion.BLOQUES, lt["duracion"])
    subtitulos.exportar(paginas, lt["palabras"], config.DIR_SUBTITULOS, lt["duracion"])
    informar(f"  · {len(paginas)} páginas → subtitulos/subtitulos.srt, subtitulos.vtt, palabras.json")

    informar("\n[4/6] Efectos y mezcla")
    tramos = {tr["id"]: tr for tr in lt["tramos"]}
    eventos = []
    for b in guion.BLOQUES:
        for s in b.get("sonidos", []):
            t0 = escenas.resolver_tiempo(s["en"], lt["palabras"], tramos, b["id"])
            duracion = s.get("duracion")
            if "hasta" in s:
                duracion = max(0.1, escenas.resolver_tiempo(s["hasta"], lt["palabras"], tramos, b["id"]) - t0)
            eventos.append((t0, audio.obtener_sfx(s["nombre"], duracion), s.get("ganancia", 1.0)))
    informar(f"  · {len(eventos)} efectos discretos + golpe final a los {lt['cierre']['golpe']:.2f} s")
    con_voz = audio.mezclar(lt["pistas"], eventos, lt["cierre"]["golpe"], lt["duracion"], con_voz=True)
    sin_voz = audio.mezclar(lt["pistas"], eventos, lt["cierre"]["golpe"], lt["duracion"], con_voz=False)
    escala = 0.98 / max(np.max(np.abs(con_voz)), np.max(np.abs(sin_voz)), 1e-6)
    rutas = {"con_voz": config.DIR_TEMP / "mezcla.wav", "sin_voz": config.DIR_TEMP / "mezcla_sin_voz.wav"}
    audio.guardar_wav(rutas["con_voz"], con_voz * escala)
    audio.guardar_wav(rutas["sin_voz"], sin_voz * escala)
    if lt["pistas"]:
        ganancia = config.LOUDNESS_OBJETIVO_LUFS - audio.medir_lufs(rutas["con_voz"])
    else:
        ganancia = 0.0
    informar(f"  · Ganancia para {config.LOUDNESS_OBJETIVO_LUFS:.0f} LUFS: {ganancia:+.1f} dB (igual en ambas versiones)")

    plan = {
        "palabras": {str(k): v for k, v in lt["palabras"].items()},
        "tramos": lt["tramos"],
        "cierre": lt["cierre"],
        "duracion": lt["duracion"],
        "paginas": paginas,
        "motor_voz": narr["motor"],
        "ganancia_db": round(ganancia, 2),
    }
    escenas.Renderizador(plan)  # comprobación temprana de todas las palabras ancla
    ruta_plan = config.DIR_TEMP / "plan.json"
    ruta_plan.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return plan, ruta_plan, rutas


# ---------------------------------------------------------------------------
# 5. Render de video en paralelo
# ---------------------------------------------------------------------------
_RENDER = None


def _iniciar_trabajador(ruta_plan):
    global _RENDER
    import escenas

    _RENDER = escenas.Renderizador(json.loads(Path(ruta_plan).read_text(encoding="utf-8")))


def _renderizar_tramo(args):
    """Escribe los fotogramas [f0, f1) en un MP4 parcial con el escritor FFmpeg de MoviePy."""
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

    f0, f1, ruta, calidad = args
    perfil = config.CALIDADES[calidad]
    escritor = FFMPEG_VideoWriter(
        str(ruta),
        (config.ANCHO, config.ALTO),
        config.FPS,
        codec="libx264",
        preset=perfil["preset"],
        ffmpeg_params=["-crf", str(perfil["crf"]), "-pix_fmt", "yuv420p", "-profile:v", "high", "-g", str(config.FPS * 2)],
    )
    try:
        for f in range(f0, f1):
            escritor.write_frame(_RENDER.fotograma(f / config.FPS))
    finally:
        escritor.close()
    return f1 - f0


def renderizar_video(ruta_plan, f_ini, f_fin, calidad, hilos):
    carpeta = config.DIR_TEMP / "tramos"
    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True)
    paso = config.FPS
    tareas = [(f0, min(f_fin, f0 + paso), carpeta / f"tramo_{i:04d}.mp4", calidad) for i, f0 in enumerate(range(f_ini, f_fin, paso))]
    total, hechos, inicio = f_fin - f_ini, 0, time.time()
    with mp.get_context("spawn").Pool(hilos, initializer=_iniciar_trabajador, initargs=(str(ruta_plan),)) as pool:
        for n in pool.imap_unordered(_renderizar_tramo, tareas):
            hechos += n
            restante = (time.time() - inicio) / hechos * (total - hechos)
            print(f"\r  · Fotogramas {hechos}/{total} ({100 * hechos / total:5.1f} %) · quedan ~{restante:4.0f} s ", end="", flush=True)
    print()
    lista = carpeta / "lista.txt"
    lista.write_text("".join(f"file '{t[2].name}'\n" for t in tareas), encoding="utf-8")
    salida = config.DIR_TEMP / "video_sin_audio.mp4"
    audio.ejecutar(
        [audio.ffmpeg_exe(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lista), "-c", "copy", str(salida)],
        "la unión de los tramos de video",
    )
    return salida


# ---------------------------------------------------------------------------
# 6. Multiplexado y verificación
# ---------------------------------------------------------------------------
def multiplexar(video, ruta_audio, salida, ganancia_db, t_ini=0.0, duracion=None):
    cmd = [audio.ffmpeg_exe(), "-v", "error", "-y", "-i", str(video)]
    if t_ini or duracion:
        cmd += ["-ss", f"{t_ini:.3f}"] + (["-t", f"{duracion:.3f}"] if duracion else [])
    filtro = f"volume={ganancia_db:.2f}dB,alimiter=limit=0.89:attack=3:release=60:level=0"
    cmd += [
        "-i", str(ruta_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", config.AUDIO_BITRATE, "-ar", str(config.AUDIO_SR),
        "-af", filtro,
        "-movflags", "+faststart",
        "-shortest",
        str(salida),
    ]  # fmt: skip
    audio.ejecutar(cmd, "el multiplexado de audio y video")


def verificar_salida(ruta, duracion_esperada):
    """Comprueba resolución, FPS, duración y códecs del MP4 final."""
    import subprocess

    from moviepy import VideoFileClip

    clip = VideoFileClip(str(ruta))
    try:
        (w, h), fps, dur = clip.size, clip.fps, clip.duration
    finally:
        clip.close()
    info = subprocess.run([audio.ffmpeg_exe(), "-hide_banner", "-i", str(ruta)], capture_output=True).stderr.decode("utf-8", "replace")
    problemas = []
    if (w, h) != (config.ANCHO, config.ALTO):
        problemas.append(f"resolución {w}×{h}")
    if abs(fps - config.FPS) > 0.01:
        problemas.append(f"{fps} FPS")
    if abs(dur - duracion_esperada) > 0.15:
        problemas.append(f"duración {dur:.2f} s")
    if "Video: h264" not in info:
        problemas.append("el video no es H.264")
    if "Audio: aac" not in info:
        problemas.append("el audio no es AAC")
    if problemas:
        raise ErrorProduccion(f"{ruta.name} no cumple el formato: " + ", ".join(problemas))
    lufs = audio.medir_lufs(ruta)
    informar(f"  ✔ {ruta.name}: {w}×{h} · {fps:.0f} FPS · {dur:.2f} s · H.264 + AAC · {lufs:.1f} LUFS · {ruta.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def argumentos():
    p = argparse.ArgumentParser(description="Genera el video «24 horas sin cepillarse» (1080×1920, 30 FPS).")
    p.add_argument("--calidad", choices=list(config.CALIDADES), default="final", help="final (lento, máxima calidad) o borrador")
    p.add_argument("--hilos", type=int, default=max(1, os.cpu_count() or 2), help="procesos de render en paralelo")
    p.add_argument("--desde", type=float, default=0.0, help="segundo inicial (render parcial)")
    p.add_argument("--hasta", type=float, default=None, help="segundo final (render parcial)")
    p.add_argument("--previsualizar", type=float, nargs="+", metavar="SEG", help="solo guarda PNG de esos instantes")
    p.add_argument("--motor-voz", choices=["auto", "kokoro", "edge", "espeak", "silencio"], help="sobrescribe config.MOTOR_TTS")
    p.add_argument("--voz", help="voz Kokoro: ef_dora (femenina), em_alex o em_santa (masculinas)")
    p.add_argument("--descargar-fuentes", action="store_true", help="descarga Montserrat, Inter, JetBrains Mono y Oswald (OFL)")
    p.add_argument("--descargar-voz", action="store_true", help="descarga el modelo de voz Kokoro (≈350 MB) a assets/modelos_voz/")
    p.add_argument("--solo-con-voz", action="store_true", help="no exportar la versión sin narración")
    p.add_argument("--limpiar", action="store_true", help="borra temp/ antes de empezar (rehace la voz)")
    p.add_argument("--salida", default=config.ARCHIVO_SALIDA, help="nombre del MP4 dentro de export/")
    return p.parse_args()


def main():
    args = argumentos()
    inicio = time.time()
    if args.motor_voz:
        config.MOTOR_TTS = args.motor_voz
    if args.voz:
        config.VOZ_KOKORO = args.voz
    informar("=" * 64)
    informar(" 24 horas sin cepillarse · fórmula de explicador vertical")
    informar("=" * 64)
    informar("\n[1/6] Comprobaciones")
    if args.limpiar and config.DIR_TEMP.exists():
        shutil.rmtree(config.DIR_TEMP)
    crear_carpetas()
    if args.descargar_fuentes:
        descargar_fuentes()
    if args.descargar_voz:
        audio.descargar_kokoro(informar)
    else:
        audio.preparar_modelo_tiempos(informar)
    audio.ffmpeg_exe()
    validar_guion()
    informar(f"  · Guion válido: {len(guion.BLOQUES)} bloques")
    informe_recursos()

    plan, ruta_plan, rutas_audio = preparar_audio_y_plan()
    informar(f"  · Voz: {plan['motor_voz']}")
    duracion = plan["duracion"]

    if args.previsualizar:
        import escenas
        from PIL import Image

        carpeta = config.DIR_TEMP / "previews"
        carpeta.mkdir(parents=True, exist_ok=True)
        rend = escenas.Renderizador(plan)
        for seg in args.previsualizar:
            ruta = carpeta / f"frame_{seg:06.2f}.png"
            Image.fromarray(rend.fotograma(min(seg, duracion - 1 / config.FPS))).save(ruta)
            informar(f"  · {ruta.relative_to(config.RAIZ)}")
        return

    t_fin = duracion if args.hasta is None else min(args.hasta, duracion)
    f_ini, f_fin = int(round(args.desde * config.FPS)), int(round(t_fin * config.FPS))
    if f_fin <= f_ini:
        raise ErrorProduccion("--hasta debe ser mayor que --desde")
    parcial = f_ini > 0 or f_fin < int(round(duracion * config.FPS))

    informar(f"\n[5/6] Render de video ({f_fin - f_ini} fotogramas, {args.hilos} procesos, calidad {args.calidad})")
    video = renderizar_video(ruta_plan, f_ini, f_fin, args.calidad, args.hilos)

    informar("\n[6/6] Audio + exportación")
    ganancia = plan["ganancia_db"]
    if parcial:
        salida = config.DIR_EXPORT / f"parcial_{args.desde:05.1f}-{t_fin:05.1f}.mp4"
        multiplexar(video, rutas_audio["con_voz"], salida, ganancia, f_ini / config.FPS, (f_fin - f_ini) / config.FPS)
        verificar_salida(salida, (f_fin - f_ini) / config.FPS)
        salidas = [salida]
    else:
        salida = config.DIR_EXPORT / args.salida
        multiplexar(video, rutas_audio["con_voz"], salida, ganancia)
        verificar_salida(salida, duracion)
        salidas = [salida]
        if not args.solo_con_voz:
            muda = salida.with_name(salida.stem + "_sin_voz" + salida.suffix)
            multiplexar(video, rutas_audio["sin_voz"], muda, ganancia)
            verificar_salida(muda, duracion)
            salidas.append(muda)
    informar(f"\nListo en {time.time() - inicio:.0f} s → " + ", ".join(str(s) for s in salidas))


if __name__ == "__main__":
    try:
        main()
    except ErrorProduccion as exc:
        sys.exit(f"\n✖ {exc}")
    except KeyboardInterrupt:
        sys.exit("\n✖ Cancelado por el usuario.")
