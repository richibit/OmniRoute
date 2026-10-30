#!/usr/bin/env python3
"""
Generador del video vertical «24 horas sin cepillarse» (1080×1920, 30 FPS, H.264 + AAC).

Uso rápido:
    python generar_video.py                      # video final en export/
    python generar_video.py --calidad borrador   # render rápido para revisar
    python generar_video.py --previsualizar 1 12 33.5   # PNG de esos instantes en temp/previews/
    python generar_video.py --desde 20 --hasta 31        # solo un tramo (export/parcial_...)

Flujo:
  1. Comprueba dependencias, carpetas, guion y recursos.
  2. Prepara la narración (grabación propia o texto a voz) y la alinea palabra a palabra.
  3. Genera subtítulos (SRT/VTT) sincronizados.
  4. Mezcla narración + música + efectos con ducking.
  5. Renderiza los fotogramas en paralelo (MoviePy/FFmpeg) y los une.
  6. Multiplexa el audio (AAC, −14 LUFS) y verifica el MP4 final.
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

# Las importaciones pesadas se comprueban primero para dar un mensaje claro.
FALTAN = []
for modulo, paquete in (("numpy", "numpy"), ("PIL", "pillow"), ("moviepy", "moviepy")):
    try:
        __import__(modulo)
    except ImportError:
        FALTAN.append(paquete)
if FALTAN:
    sys.exit(
        f"Faltan dependencias de Python: {', '.join(FALTAN)}\n"
        f"Instálalas con:  pip install -r requirements.txt"
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))


import audio  # noqa: E402
import config  # noqa: E402
import guion  # noqa: E402
import subtitulos  # noqa: E402
from utilidades import ErrorProduccion, ruta_emoji, ruta_fuente  # noqa: E402


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
        config.DIR_SUBTITULOS,
        config.DIR_TEMP,
        config.DIR_EXPORT,
    ):
        carpeta.mkdir(parents=True, exist_ok=True)


def descargar_fuentes():
    """Descarga Montserrat, Inter y JetBrains Mono (licencia OFL) a assets/fuentes/."""
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
    """Comprueba que los bloques sean contiguos, cubran toda la duración y usen nombres válidos."""
    avisos = []
    bloques = guion.BLOQUES
    if not bloques:
        raise ErrorProduccion("guion.py no tiene bloques.")
    if abs(bloques[0]["inicio"]) > 1e-6:
        raise ErrorProduccion("El primer bloque debe empezar en 0 s.")
    ids = set()
    for anterior, b in zip([None] + bloques[:-1], bloques):
        for clave in ("id", "inicio", "fin", "narracion"):
            if clave not in b:
                raise ErrorProduccion(f"Al bloque {b.get('id', '?')} le falta el campo «{clave}» en guion.py")
        if b["id"] in ids:
            raise ErrorProduccion(f"Id de bloque repetido: {b['id']}")
        ids.add(b["id"])
        if b["fin"] <= b["inicio"]:
            raise ErrorProduccion(f"El bloque {b['id']} termina antes de empezar.")
        if anterior and abs(anterior["fin"] - b["inicio"]) > 1e-6:
            raise ErrorProduccion(f"Hay un hueco o solape entre los bloques {anterior['id']} y {b['id']}.")
        if b["fin"] - b["inicio"] > 6.0 + 1e-6:
            avisos.append(f"El bloque {b['id']} dura más de 6 s (el documento recomienda ≤ 6 s).")
        for s in b.get("sonidos", []):
            if s["nombre"] not in audio.GENERADORES_SFX and not audio.buscar_archivo(config.DIR_SFX, s["nombre"], audio.EXT_AUDIO):
                raise ErrorProduccion(f"Sonido desconocido «{s['nombre']}» en el bloque {b['id']}.")
    if abs(bloques[-1]["fin"] - config.DURACION_TOTAL) > 1e-6:
        raise ErrorProduccion(
            f"El último bloque termina en {bloques[-1]['fin']} s pero DURACION_TOTAL es {config.DURACION_TOTAL} s (config.py)."
        )
    if set(b["id"] for b in bloques) != set(range(1, 11)):
        raise ErrorProduccion("Las escenas visuales están programadas para los bloques 1–10 (ids 1 a 10).")
    return avisos


def informe_recursos():
    """Muestra qué recursos opcionales se usarán y cuáles se sustituyen por animación."""
    import escenas

    informar("  Recursos visuales opcionales (assets/):")
    for nombre, descripcion in guion.RECURSOS.items():
        ruta = escenas.buscar_recurso(nombre)
        estado = f"✔ {ruta.relative_to(config.RAIZ)}" if ruta else "– no encontrado → se usa la ilustración animada"
        informar(f"    {nombre:<18} {estado}")
    informar("  Tipografías:")
    for rol in config.FUENTES:
        ruta = ruta_fuente(rol)
        informar(f"    {rol:<13} {Path(ruta).name if ruta else 'fuente integrada de Pillow (instala fuentes, ver README)'}")
    informar(f"    {'emoji':<13} {Path(ruta_emoji()).name if ruta_emoji() else 'no disponible → se sustituyen por símbolos'}")


# ---------------------------------------------------------------------------
# 2–4. Narración, subtítulos y mezcla
# ---------------------------------------------------------------------------
def preparar_audio_y_plan():
    import escenas

    informar("\n[2/6] Narración")
    narr = audio.preparar_narracion(guion.BLOQUES, config.DIR_TEMP, informar)
    palabras = narr["palabras"]

    informar("\n[3/6] Subtítulos")
    trozos = subtitulos.agrupar(palabras, guion.BLOQUES)
    subtitulos.exportar(trozos, palabras, config.DIR_SUBTITULOS)
    informar(f"  · {len(trozos)} subtítulos → subtitulos/subtitulos.srt, subtitulos.vtt, palabras.json")

    informar("\n[4/6] Música, efectos y mezcla")
    resolver = lambda ref: escenas.resolver_tiempo(ref, palabras)  # noqa: E731
    secciones = audio.resolver_secciones(guion.MUSICA_SECCIONES, resolver)
    musica = audio.preparar_musica(secciones, config.DURACION_TOTAL, guion.BPM, informar)
    eventos = []
    for b in guion.BLOQUES:
        for s in b.get("sonidos", []):
            t0 = resolver(s["en"])
            duracion = s.get("duracion")
            if "hasta" in s:
                duracion = max(0.1, resolver(s["hasta"]) - t0)
            eventos.append((t0, audio.obtener_sfx(s["nombre"], duracion), s.get("ganancia", 1.0)))
    informar(f"  · {len(eventos)} efectos de sonido colocados")
    mezcla = audio.mezclar(narr["pistas"], musica, eventos, config.DURACION_TOTAL)
    ruta_mezcla = config.DIR_TEMP / "mezcla.wav"
    audio.guardar_wav(ruta_mezcla, mezcla)
    informar(f"  · Mezcla guardada en {ruta_mezcla.relative_to(config.RAIZ)}")

    plan = {
        "palabras": {str(k): v for k, v in palabras.items()},
        "trozos": trozos,
        "env_voz": audio.envolvente_voz_por_fotograma(narr["pistas"], config.DURACION_TOTAL, config.FPS),
        "motor_voz": narr["motor"],
    }
    # Comprobación temprana: todas las palabras ancla del guion deben existir.
    escenas.Renderizador(plan)
    ruta_plan = config.DIR_TEMP / "plan.json"
    ruta_plan.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return plan, ruta_plan, ruta_mezcla


# ---------------------------------------------------------------------------
# 5. Render de video en paralelo
# ---------------------------------------------------------------------------
_RENDER = None


def _iniciar_trabajador(ruta_plan):
    global _RENDER
    import escenas

    plan = json.loads(Path(ruta_plan).read_text(encoding="utf-8"))
    _RENDER = escenas.Renderizador(plan)


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
    paso = config.FPS  # tramos de 1 s: buen reparto entre procesos y progreso visible
    tareas = []
    for i, f0 in enumerate(range(f_ini, f_fin, paso)):
        tareas.append((f0, min(f_fin, f0 + paso), carpeta / f"tramo_{i:04d}.mp4", calidad))
    total = f_fin - f_ini
    hechos = 0
    inicio = time.time()
    contexto = mp.get_context("spawn")  # igual en Windows, macOS y Linux
    with contexto.Pool(hilos, initializer=_iniciar_trabajador, initargs=(str(ruta_plan),)) as pool:
        for n in pool.imap_unordered(_renderizar_tramo, tareas):
            hechos += n
            transcurrido = time.time() - inicio
            restante = transcurrido / hechos * (total - hechos)
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
def multiplexar(video, ruta_audio, salida, t_ini=0.0, duracion=None):
    cmd = [audio.ffmpeg_exe(), "-v", "error", "-y", "-i", str(video)]
    if t_ini or duracion:
        cmd += ["-ss", f"{t_ini:.3f}"]
        if duracion:
            cmd += ["-t", f"{duracion:.3f}"]
    cmd += [
        "-i", str(ruta_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", config.AUDIO_BITRATE, "-ar", str(config.AUDIO_SR),
        "-af", f"loudnorm=I={config.LOUDNESS_OBJETIVO_LUFS}:TP=-1.5:LRA=11",
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
        w, h = clip.size
        fps, dur = clip.fps, clip.duration
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
        raise ErrorProduccion("El archivo final no cumple el formato: " + ", ".join(problemas))
    peso = ruta.stat().st_size / 1e6
    informar(f"  ✔ {w}×{h} · {fps:.0f} FPS · {dur:.2f} s · H.264 + AAC · {peso:.1f} MB")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def argumentos():
    p = argparse.ArgumentParser(description="Genera el video «24 horas sin cepillarse» (1080×1920, 30 FPS).")
    p.add_argument("--calidad", choices=list(config.CALIDADES), default="final", help="final (lento, máxima calidad) o borrador")
    p.add_argument("--hilos", type=int, default=max(1, (os.cpu_count() or 2) - 0), help="procesos de render en paralelo")
    p.add_argument("--desde", type=float, default=0.0, help="segundo inicial (render parcial)")
    p.add_argument("--hasta", type=float, default=None, help="segundo final (render parcial)")
    p.add_argument("--previsualizar", type=float, nargs="+", metavar="SEG", help="solo guarda PNG de esos instantes")
    p.add_argument("--motor-voz", choices=["auto", "edge", "espeak", "silencio"], help="sobrescribe config.MOTOR_TTS")
    p.add_argument("--descargar-fuentes", action="store_true", help="descarga Montserrat, Inter y JetBrains Mono (OFL)")
    p.add_argument("--limpiar", action="store_true", help="borra temp/ antes de empezar (rehace la voz)")
    p.add_argument("--salida", default=config.ARCHIVO_SALIDA, help="nombre del MP4 dentro de export/")
    return p.parse_args()


def main():
    args = argumentos()
    inicio = time.time()
    if args.motor_voz:
        config.MOTOR_TTS = args.motor_voz
    informar("=" * 64)
    informar(" 24 horas sin cepillarse · generador de video vertical")
    informar("=" * 64)
    informar("\n[1/6] Comprobaciones")
    if args.limpiar and config.DIR_TEMP.exists():
        shutil.rmtree(config.DIR_TEMP)
    crear_carpetas()
    if args.descargar_fuentes:
        descargar_fuentes()
    audio.ffmpeg_exe()
    for aviso in validar_guion():
        informar(f"  ! {aviso}")
    informar(f"  · Guion válido: {len(guion.BLOQUES)} bloques, {config.DURACION_TOTAL:.0f} s")
    informe_recursos()

    plan, ruta_plan, ruta_mezcla = preparar_audio_y_plan()
    informar(f"  · Voz: {plan['motor_voz']}")

    if args.previsualizar:
        import escenas

        carpeta = config.DIR_TEMP / "previews"
        carpeta.mkdir(parents=True, exist_ok=True)
        rend = escenas.Renderizador(plan)
        from PIL import Image

        for seg in args.previsualizar:
            ruta = carpeta / f"frame_{seg:06.2f}.png"
            Image.fromarray(rend.fotograma(seg)).save(ruta)
            informar(f"  · {ruta.relative_to(config.RAIZ)}")
        return

    t_fin = config.DURACION_TOTAL if args.hasta is None else min(args.hasta, config.DURACION_TOTAL)
    f_ini, f_fin = int(round(args.desde * config.FPS)), int(round(t_fin * config.FPS))
    if f_fin <= f_ini:
        raise ErrorProduccion("--hasta debe ser mayor que --desde")
    parcial = f_ini > 0 or f_fin < int(round(config.DURACION_TOTAL * config.FPS))

    informar(f"\n[5/6] Render de video ({f_fin - f_ini} fotogramas, {args.hilos} procesos, calidad {args.calidad})")
    video = renderizar_video(ruta_plan, f_ini, f_fin, args.calidad, args.hilos)

    informar("\n[6/6] Audio + exportación")
    if parcial:
        nombre = f"parcial_{args.desde:05.1f}-{t_fin:05.1f}.mp4"
        salida = config.DIR_EXPORT / nombre
        multiplexar(video, ruta_mezcla, salida, f_ini / config.FPS, (f_fin - f_ini) / config.FPS)
        verificar_salida(salida, (f_fin - f_ini) / config.FPS)
    else:
        salida = config.DIR_EXPORT / args.salida
        multiplexar(video, ruta_mezcla, salida)
        verificar_salida(salida, config.DURACION_TOTAL)
    informar(f"\nListo en {time.time() - inicio:.0f} s → {salida}")


if __name__ == "__main__":
    try:
        main()
    except ErrorProduccion as exc:
        sys.exit(f"\n✖ {exc}")
    except KeyboardInterrupt:
        sys.exit("\n✖ Cancelado por el usuario.")
