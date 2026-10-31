"""
Audio: narración (grabada o texto a voz neuronal), tiempos palabra a palabra,
línea de tiempo dinámica, efectos de sonido generados y mezcla final.

La fórmula de referencia es voz sola: sin música bajo la narración, efectos
casi inaudibles y, al final, 1,2 s de silencio y un golpe grave con el logo.
Todo el audio se procesa como arrays de numpy a 44,1 kHz; FFmpeg se usa para
decodificar archivos, cambiar el tempo y medir/normalizar la sonoridad.
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

import config
from utilidades import ErrorProduccion, aleatorio

SR = config.AUDIO_SR
EXT_AUDIO = [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"]
PUNTUACION = set(".,;:!?…¿¡—-«»\"'()")


# ---------------------------------------------------------------------------
# FFmpeg y E/S de audio
# ---------------------------------------------------------------------------
def ffmpeg_exe():
    """Ruta a FFmpeg: el del sistema o el que instala imageio-ffmpeg (dependencia de MoviePy)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depende del sistema
        raise ErrorProduccion(
            "No se encontró FFmpeg. Instálalo (ver README, sección «Instalación») "
            "o ejecuta: pip install imageio-ffmpeg"
        ) from exc


def ejecutar(cmd, que_hace):
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        detalle = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise ErrorProduccion(f"Falló {que_hace}:\n  " + "\n  ".join(detalle))
    return proc


def cargar_audio(ruta, sr=SR):
    """Decodifica cualquier formato de audio a un array mono float32."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise ErrorProduccion(f"No existe el archivo de audio: {ruta}")
    proc = ejecutar(
        [ffmpeg_exe(), "-v", "error", "-i", str(ruta), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        f"la lectura de {ruta.name}",
    )
    datos = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if datos.size == 0:
        raise ErrorProduccion(f"El archivo de audio está vacío o dañado: {ruta}")
    return datos


def guardar_wav(ruta, datos, sr=SR):
    """Guarda un array mono o estéreo (n, 2) como WAV de 16 bits."""
    datos = np.asarray(datos, dtype=np.float32)
    if datos.ndim == 1:
        datos = np.stack([datos, datos], axis=1)
    pcm = (np.clip(datos, -1, 1) * 32767).astype("<i2")
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def buscar_archivo(carpeta, nombre, extensiones):
    for ext in extensiones:
        ruta = Path(carpeta) / f"{nombre}{ext}"
        if ruta.exists():
            return ruta
    return None


def db(valor):
    return 10 ** (valor / 20)


def recortar_silencios(x, umbral_rel=0.02, margen=0.015):
    if x.size == 0 or np.max(np.abs(x)) == 0:
        return x
    umbral = np.max(np.abs(x)) * umbral_rel
    idx = np.nonzero(np.abs(x) > umbral)[0]
    m = int(margen * SR)
    return x[max(0, idx[0] - m) : min(x.size, idx[-1] + m)]


def cambiar_tempo(x, factor, temp_dir):
    """Acelera/ralentiza sin cambiar el tono (filtro atempo de FFmpeg)."""
    if abs(factor - 1) < 0.01:
        return x
    entrada = Path(temp_dir) / "_tempo_in.wav"
    salida = Path(temp_dir) / "_tempo_out.wav"
    guardar_wav(entrada, x)
    ejecutar(
        [ffmpeg_exe(), "-v", "error", "-y", "-i", str(entrada), "-filter:a", f"atempo={factor:.4f}", "-ac", "1", str(salida)],
        "el ajuste de tempo de la narración",
    )
    y = cargar_audio(salida)
    entrada.unlink(missing_ok=True)
    salida.unlink(missing_ok=True)
    return y


def medir_lufs(ruta):
    """Sonoridad integrada (LUFS) medida con el filtro ebur128 de FFmpeg."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(ruta), "-af", "ebur128", "-f", "null", "-"],
        capture_output=True,
    )
    texto = proc.stderr.decode("utf-8", "replace")
    valores = re.findall(r"I:\s+(-?[\d.]+) LUFS", texto)
    if not valores:
        raise ErrorProduccion("No se pudo medir la sonoridad de la mezcla.")
    return float(valores[-1])


# ---------------------------------------------------------------------------
# Kokoro: voz neuronal offline con duración de cada fonema
# ---------------------------------------------------------------------------
_KOKORO = {}


def rutas_kokoro():
    d = config.DIR_MODELOS_VOZ
    a = config.ARCHIVOS_KOKORO
    return d / a["modelo"], d / a["modelo_tiempos"], d / a["voces"]


def kokoro_instalado():
    try:
        import kokoro_onnx  # noqa: F401
    except ImportError:
        return False
    modelo, tiempos, voces = rutas_kokoro()
    return voces.exists() and (modelo.exists() or tiempos.exists())


def preparar_modelo_tiempos(informar=print):
    """
    Crea una copia del modelo que expone la duración de cada fonema (salida «duration»).
    Así se sabe en qué milisegundo empieza cada palabra. Requiere el paquete `onnx`.
    """
    modelo, tiempos, _ = rutas_kokoro()
    if tiempos.exists() or not modelo.exists():
        return
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        informar("  ! Sin el paquete «onnx» los tiempos por palabra se estiman (pip install onnx para exactitud).")
        return
    m = onnx.load(str(modelo))
    salidas = {o.name for o in m.graph.output}
    if "duration" not in salidas:
        nodos = {o for n in m.graph.node for o in n.output}
        origen = "/encoder/Gather_output_0"  # duraciones enteras por fonema (antes del CumSum)
        if origen not in nodos:
            informar("  ! Este modelo de Kokoro no tiene el nodo de duraciones esperado; se estimarán los tiempos.")
            return
        m.graph.node.append(helper.make_node("Identity", [origen], ["duration"], name="expone_duracion"))
        m.graph.output.append(helper.make_tensor_value_info("duration", TensorProto.INT64, ["n_fonemas"]))
    onnx.save(m, str(tiempos))
    informar(f"  · Modelo con tiempos por fonema creado: {tiempos.name}")


def descargar_kokoro(informar=print):
    """Descarga el modelo Kokoro v1.0 (≈350 MB, licencia Apache-2.0) a assets/modelos_voz/."""
    config.DIR_MODELOS_VOZ.mkdir(parents=True, exist_ok=True)
    modelo, _, voces = rutas_kokoro()
    for ruta in (modelo, voces):
        if ruta.exists():
            informar(f"  · {ruta.name} ya existe")
            continue
        url = config.URL_KOKORO + ruta.name
        informar(f"  · Descargando {ruta.name} …")
        try:
            tmp = ruta.with_suffix(ruta.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.rename(ruta)
        except Exception as exc:
            raise ErrorProduccion(f"No se pudo descargar {url}: {exc}") from exc
    preparar_modelo_tiempos(informar)


def _cargar_kokoro():
    if "k" not in _KOKORO:
        from kokoro_onnx import Kokoro

        modelo, tiempos, voces = rutas_kokoro()
        _KOKORO["k"] = Kokoro(str(tiempos if tiempos.exists() else modelo), str(voces))
    return _KOKORO["k"]


def palabras_desde_fonemas(texto, tiempos, tokenizador, idioma):
    """
    Convierte los tiempos por fonema de Kokoro en tiempos por palabra del texto.
    Cada palabra se fonemiza sola para saber cuántos «grupos fonéticos» ocupa
    (p. ej. «1965» son cinco palabras habladas).
    """
    grupos = []
    inicio = fin = None
    for t in tiempos:
        if t.phoneme.isspace():
            if inicio is not None:
                grupos.append((inicio, fin))
            inicio = fin = None
            continue
        if inicio is None:
            inicio = t.start
        if t.phoneme not in PUNTUACION:
            fin = t.end
        elif fin is None:
            fin = t.start
    if inicio is not None:
        grupos.append((inicio, fin))
    palabras = texto.split()
    cuentas = [max(1, len(tokenizador.phonemize(p, idioma).split())) for p in palabras]
    if sum(cuentas) != len(grupos) or not grupos:
        return None  # no cuadra: el llamador usará la estimación
    resultado, i = [], 0
    for p, c in zip(palabras, cuentas):
        resultado.append({"texto": p, "inicio": round(grupos[i][0], 3), "fin": round(grupos[i + c - 1][1], 3)})
        i += c
    return resultado


def sintetizar_kokoro(texto, temp_dir):
    """Voz Kokoro de un bloque. Devuelve (array 44,1 kHz, palabras con tiempos relativos o None)."""
    clave = hashlib.sha1(
        f"kokoro|{config.VOZ_KOKORO}|{config.VELOCIDAD_KOKORO}|{config.IDIOMA_KOKORO}|{texto}".encode()
    ).hexdigest()[:12]
    base = Path(temp_dir) / "narracion" / f"kokoro_{clave}"
    wav, js = base.with_suffix(".wav"), base.with_suffix(".json")
    if not (wav.exists() and js.exists()):
        k = _cargar_kokoro()
        if config.VOZ_KOKORO not in k.get_voices():
            raise ErrorProduccion(f"La voz Kokoro «{config.VOZ_KOKORO}» no existe. Españolas: ef_dora, em_alex, em_santa.")
        x, sr, tiempos = k.create_timed(texto, voice=config.VOZ_KOKORO, speed=config.VELOCIDAD_KOKORO, lang=config.IDIOMA_KOKORO)
        palabras = palabras_desde_fonemas(texto, tiempos, k.tokenizer, config.IDIOMA_KOKORO) if tiempos else None
        guardar_wav(base.with_name(base.name + "_24k.wav"), x, sr)
        guardar_wav(wav, cargar_audio(base.with_name(base.name + "_24k.wav")))
        base.with_name(base.name + "_24k.wav").unlink(missing_ok=True)
        js.write_text(json.dumps(palabras, ensure_ascii=False), encoding="utf-8")
    return cargar_audio(wav), json.loads(js.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Otros motores de voz (alternativas)
# ---------------------------------------------------------------------------
def texto_para_voz(texto):
    for original, dicho in config.PRONUNCIACION.items():
        texto = texto.replace(original, dicho)
    return texto


def _espeak():
    return shutil.which("espeak-ng") or shutil.which("espeak")


_VOZ_ESPEAK = {}


def voz_espeak_disponible(temp_dir):
    if "voz" in _VOZ_ESPEAK:
        return _VOZ_ESPEAK["voz"]
    exe, voz_ok = _espeak(), None
    if exe:
        prueba = Path(temp_dir) / "_prueba_voz.wav"
        for voz in config.VOCES_ESPEAK:
            proc = subprocess.run([exe, "-v", voz, "-w", str(prueba), "hola"], capture_output=True)
            if proc.returncode == 0 and prueba.exists() and prueba.stat().st_size > 2000:
                voz_ok = voz
                break
        prueba.unlink(missing_ok=True)
    _VOZ_ESPEAK["voz"] = voz_ok
    return voz_ok


def sintetizar_espeak(texto, temp_dir):
    clave = hashlib.sha1(f"espeak|{texto}".encode()).hexdigest()[:12]
    wav = Path(temp_dir) / "narracion" / f"espeak_{clave}.wav"
    if not wav.exists():
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(texto_para_voz(texto))
            ruta_txt = f.name
        bruto = wav.with_name(wav.stem + "_raw.wav")
        try:
            ejecutar(
                [_espeak(), "-v", voz_espeak_disponible(temp_dir), "-s", str(config.VELOCIDAD_ESPEAK), "-w", str(bruto), "-f", ruta_txt],
                "la síntesis de voz con espeak-ng",
            )
        finally:
            os.unlink(ruta_txt)
        guardar_wav(wav, cargar_audio(bruto))
        bruto.unlink(missing_ok=True)
    return cargar_audio(wav)


def edge_disponible():
    try:
        import edge_tts  # noqa: F401

        return True
    except ImportError:
        return False


def sintetizar_edge(texto, temp_dir):
    import edge_tts

    clave = hashlib.sha1(f"edge|{config.VOZ_EDGE}|{texto}".encode()).hexdigest()[:12]
    wav = Path(temp_dir) / "narracion" / f"edge_{clave}.wav"
    if not wav.exists():
        mp3 = wav.with_suffix(".mp3")
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

        async def _tarea():
            await edge_tts.Communicate(texto, config.VOZ_EDGE, proxy=proxy).save(str(mp3))

        asyncio.run(asyncio.wait_for(_tarea(), timeout=45))
        guardar_wav(wav, cargar_audio(mp3))
        mp3.unlink(missing_ok=True)
    return cargar_audio(wav)


def elegir_motor(temp_dir, informar=print):
    """Decide el motor de voz según config.MOTOR_TTS y lo que haya instalado."""
    pedido = config.MOTOR_TTS
    if pedido in ("auto", "kokoro"):
        if kokoro_instalado():
            return "kokoro"
        if pedido == "kokoro":
            raise ErrorProduccion(
                "Kokoro no está listo. Ejecuta: pip install kokoro-onnx onnx  y luego  "
                "python generar_video.py --descargar-voz"
            )
        informar("  ! Kokoro no está instalado (mejor voz offline): python generar_video.py --descargar-voz")
    if pedido in ("auto", "edge") and edge_disponible():
        try:
            sintetizar_edge("Hola.", temp_dir)
            return "edge"
        except Exception as exc:
            if pedido == "edge":
                raise ErrorProduccion(f"edge-tts no está disponible (¿sin internet?): {exc}") from exc
    if pedido in ("auto", "espeak") and voz_espeak_disponible(temp_dir):
        return "espeak"
    if pedido == "espeak":
        raise ErrorProduccion("espeak-ng no está instalado. Ver README → «Voz».")
    return "silencio"


# ---------------------------------------------------------------------------
# Estimación de tiempos por palabra (grabaciones propias y voces sin tiempos)
# ---------------------------------------------------------------------------
_VOCALES = re.compile(r"[aeiouáéíóúü]+", re.IGNORECASE)


def _peso_palabra(p):
    digitos = sum(c.isdigit() for c in p)
    return 0.6 + max(1, len(_VOCALES.findall(p)) + digitos * 2)


def _pausa_tras(p):
    p = p.rstrip("»\"')")
    if p.endswith("…") or p.endswith("..."):
        return 2.6
    if p and p[-1] in ".?!":
        return 2.2
    if p and p[-1] in ",:;":
        return 1.3
    return 0.0


def detectar_pausas(x, sr=SR, duracion_min=0.09):
    salto = int(sr * 0.01)
    n = max(1, x.size // salto)
    rms = np.sqrt(np.mean(x[: n * salto].reshape(n, salto) ** 2, axis=1) + 1e-12)
    if rms.max() <= 1e-6:
        return 0.0, x.size / sr, []
    voz = rms > rms.max() * 0.07
    indices = np.nonzero(voz)[0]
    v0, v1 = indices[0], indices[-1] + 1
    pausas, i = [], v0
    while i < v1:
        if not voz[i]:
            j = i
            while j < v1 and not voz[j]:
                j += 1
            if (j - i) * 0.01 >= duracion_min:
                pausas.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1
    return v0 * 0.01, v1 * 0.01, pausas


def estimar_palabras(texto, x):
    """Reparte el tiempo de voz según las sílabas y ancla la puntuación a los silencios reales."""
    palabras = texto.split()
    pesos = [_peso_palabra(p) for p in palabras]
    pausas_t = [_pausa_tras(p) for p in palabras]
    inicios, fines, u = [], [], 0.0
    for w, pa in zip(pesos, pausas_t):
        inicios.append(u)
        fines.append(u + w)
        u += w + pa
    total = fines[-1]
    v0, v1, pausas = detectar_pausas(x)
    anclas_u, anclas_t, usadas = [0.0], [v0], set()
    for i, pa in enumerate(pausas_t[:-1]):
        if pa <= 0:
            continue
        estimado = v0 + (v1 - v0) * (fines[i] + pa / 2) / total
        mejor, dist = None, 0.7
        for k, (p0, p1) in enumerate(pausas):
            centro = (p0 + p1) / 2
            if k not in usadas and centro > anclas_t[-1] and abs(centro - estimado) < dist:
                mejor, dist = k, abs(centro - estimado)
        if mejor is not None:
            usadas.add(mejor)
            anclas_u += [fines[i], fines[i] + pa]
            anclas_t += list(pausas[mejor])
    anclas_u.append(total)
    anclas_t.append(v1)
    au, at = [anclas_u[0]], [anclas_t[0]]
    for a, b in zip(anclas_u[1:], anclas_t[1:]):
        if a > au[-1] and b > at[-1]:
            au.append(a)
            at.append(b)
    return [
        {"texto": p, "inicio": round(float(np.interp(s, au, at)), 3), "fin": round(float(np.interp(e, au, at)), 3)}
        for p, s, e in zip(palabras, inicios, fines)
    ]


# ---------------------------------------------------------------------------
# Narración completa y línea de tiempo
# ---------------------------------------------------------------------------
def preparar_narracion(bloques, temp_dir, informar=print):
    """
    Devuelve {"motor": str, "bloques": {id: {"audio": array|None, "duracion": s, "palabras": [...]}}}
    con tiempos relativos al inicio de la voz de cada bloque.
    """
    temp_dir = Path(temp_dir)
    (temp_dir / "narracion").mkdir(parents=True, exist_ok=True)
    grabados = {b["id"]: buscar_archivo(config.DIR_NARRACION, f"bloque_{b['id']:02d}", EXT_AUDIO) for b in bloques}
    motor = None if all(grabados.values()) else elegir_motor(temp_dir, informar)
    n_grab = sum(1 for v in grabados.values() if v)
    if n_grab:
        informar(f"  · Grabaciones propias: {n_grab} de {len(bloques)} bloques")
    if motor:
        detalle = f" ({config.VOZ_KOKORO}, {config.IDIOMA_KOKORO}, ×{config.VELOCIDAD_KOKORO})" if motor == "kokoro" else ""
        informar(f"  · Voz sintética para el resto: {motor}{detalle}")
    resultado = {"motor": motor if not n_grab else ("grabación" if not motor else f"grabación + {motor}"), "bloques": {}}
    exactos = 0
    for b in bloques:
        texto = b["narracion"]
        palabras = None
        if grabados[b["id"]]:
            x = recortar_silencios(cargar_audio(grabados[b["id"]]))
        elif motor == "kokoro":
            x, palabras = sintetizar_kokoro(texto, temp_dir)
            x = x.copy()
        elif motor == "edge":
            x = recortar_silencios(sintetizar_edge(texto, temp_dir))
        elif motor == "espeak":
            x = recortar_silencios(sintetizar_espeak(texto, temp_dir))
        else:
            dur = max(1.2, len(texto.split()) / 2.9)
            resultado["bloques"][b["id"]] = {
                "audio": None,
                "duracion": dur,
                "palabras": estimar_palabras(texto, np.ones(int(dur * SR), np.float32) * 0.3),
            }
            continue
        if palabras:
            exactos += 1
        else:
            palabras = estimar_palabras(texto, x)
        rms = np.sqrt(np.mean(x**2)) + 1e-9
        x = (x * (0.1 / rms)).astype(np.float32)  # todos los bloques igual de fuertes
        resultado["bloques"][b["id"]] = {"audio": x, "duracion": x.size / SR, "palabras": palabras}
    if motor == "kokoro":
        informar(f"  · Tiempos por palabra exactos (duración de fonemas) en {exactos} de {len(bloques)} bloques")
    return resultado


def construir_linea_de_tiempo(bloques, narracion, fps):
    """
    Coloca los bloques uno tras otro (casi sin pausas, como en la fórmula) y calcula
    los tramos visuales, el cierre y la duración total (redondeada a fotogramas).
    """
    t = config.INICIO_VOZ
    tramos, palabras, pistas = [], {}, []
    for b in bloques:
        nb = narracion["bloques"][b["id"]]
        palabras[b["id"]] = [dict(p, inicio=round(p["inicio"] + t, 3), fin=round(p["fin"] + t, 3)) for p in nb["palabras"]]
        if nb["audio"] is not None:
            pistas.append((t, nb["audio"]))
        tramos.append({"id": b["id"], "inicio_voz": round(t, 3), "fin_voz": round(t + nb["duracion"], 3)})
        t += nb["duracion"] + config.PAUSA_ENTRE_BLOQUES
    for i, tr in enumerate(tramos):
        tr["inicio"] = 0.0 if i == 0 else round(tr["inicio_voz"] - 0.08, 3)
    fin_voz = max(p["fin"] for p in palabras[bloques[-1]["id"]])
    for i, tr in enumerate(tramos):
        tr["fin"] = tramos[i + 1]["inicio"] if i + 1 < len(tramos) else round(fin_voz + 0.30, 3)
    golpe = fin_voz + config.CIERRE_SILENCIO
    duracion = round(round((golpe + config.CIERRE_COLA) * fps) / fps, 4)
    return {
        "tramos": tramos,
        "palabras": palabras,
        "pistas": pistas,
        "cierre": {"inicio": tramos[-1]["fin"], "golpe": round(golpe, 3)},
        "duracion": duracion,
    }


def envolvente_voz_por_fotograma(pistas, duracion, fps):
    """Intensidad de la voz (0–1) en cada fotograma (para pequeñas animaciones sincronizadas)."""
    n = int(round(duracion * SR))
    voz = np.zeros(n, dtype=np.float32)
    for t0, x in pistas:
        i0 = int(t0 * SR)
        i1 = min(n, i0 + x.size)
        if i1 > i0:
            voz[i0:i1] += x[: i1 - i0]
    nf = int(round(duracion * fps))
    por = SR / fps
    env = np.array([np.sqrt(np.mean(voz[int(i * por) : int((i + 1) * por)] ** 2) + 1e-12) for i in range(nf)])
    ref = np.percentile(env[env > 1e-4], 90) if np.any(env > 1e-4) else 1.0
    return [round(float(v), 3) for v in np.clip(env / (ref + 1e-9), 0, 1)]


# ---------------------------------------------------------------------------
# Herramientas de síntesis
# ---------------------------------------------------------------------------
def _t(dur):
    return np.arange(int(dur * SR), dtype=np.float32) / SR


def pasa_bajos(x, n):
    """Media móvil (filtro pasa bajos simple y vectorizado)."""
    n = max(1, int(n))
    if n == 1:
        return x
    c = np.cumsum(np.concatenate([[0.0], x]), dtype=np.float64)
    y = (c[n:] - c[:-n]) / n
    return np.concatenate([y, np.full(n - 1, y[-1] if y.size else 0.0)]).astype(np.float32)


def pasa_altos(x, n):
    return x - pasa_bajos(x, n)


def _ruido(n, semilla):
    return aleatorio(semilla).standard_normal(n).astype(np.float32)


def _barrido(f):
    """Seno con frecuencia variable f(t) (array)."""
    return np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32)


def _caida(t, tau):
    return np.exp(-t / tau).astype(np.float32)


def _ataque(t, a):
    return np.clip(t / max(a, 1e-4), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Efectos de sonido generados (se reemplazan por assets/audios/sfx/<nombre>.wav)
# ---------------------------------------------------------------------------
def sfx_whoosh(dur=0.45):
    t = _t(dur)
    n = _ruido(t.size, 11)
    k = t / dur
    y = pasa_bajos(n, 22) * (1 - k) * 2.2 + pasa_altos(pasa_bajos(n, 3), 12) * k * 0.9
    return y * np.sin(np.pi * k) ** 2


def sfx_clic(dur=0.08):
    t = _t(dur)
    y = _ruido(t.size, 3) * _caida(t, 0.0015) + 0.6 * np.sin(2 * np.pi * 1900 * t) * _caida(t, 0.006)
    t2 = np.clip(t - 0.012, 0, None)
    return y + (0.5 * np.sin(2 * np.pi * 2600 * t2) * _caida(t2, 0.004)) * (t > 0.012)


def sfx_burbujas(dur=1.0):
    t = _t(dur)
    y = np.zeros_like(t)
    rng = aleatorio(5)
    for inicio in np.sort(rng.uniform(0, dur - 0.1, 9)):
        i0 = int(inicio * SR)
        tt = _t(0.07)
        f = 320 + 900 * (tt / 0.07) ** 0.7 * rng.uniform(0.7, 1.3)
        y[i0 : i0 + tt.size] += _barrido(f) * _caida(tt, 0.025) * rng.uniform(0.4, 0.9)
    return y


def sfx_velcro(dur=0.5):
    t = _t(dur)
    rng = aleatorio(8)
    impulsos = (rng.random(t.size) < 140 / SR).astype(np.float32) * rng.uniform(0.3, 1, t.size).astype(np.float32)
    granos = np.convolve(impulsos, np.exp(-np.arange(int(0.004 * SR)) / (0.0012 * SR)), mode="same")
    return pasa_altos(_ruido(t.size, 9), 6) * (0.25 + granos * 2.5) * np.sin(np.pi * t / dur) ** 0.5


def sfx_plop(dur=0.12):
    t = _t(dur)
    return _barrido(180 + 600 * np.exp(-t / 0.025)) * _caida(t, 0.035) * _ataque(t, 0.002)


def sfx_crujido(dur=0.4):
    y = np.zeros(int(dur * SR), dtype=np.float32)
    rng = aleatorio(13)
    for inicio in rng.uniform(0, dur - 0.02, 26):
        largo = int(rng.uniform(0.004, 0.012) * SR)
        i0 = int(inicio * SR)
        y[i0 : i0 + largo] += pasa_altos(_ruido(largo, int(inicio * 1e5)), 4) * np.hanning(largo) * rng.uniform(0.4, 1)
    return y


def sfx_efervescencia(dur=1.3):
    t = _t(dur)
    rng = aleatorio(17)
    chispas = np.convolve((rng.random(t.size) < 900 / SR).astype(np.float32), np.exp(-np.arange(80) / 15.0), mode="same")
    return pasa_altos(_ruido(t.size, 18), 3) * (0.15 + chispas) * _ataque(t, 0.15) * np.clip((dur - t) / 0.5, 0, 1)


def sfx_riser(dur=1.5):
    t = _t(dur)
    k = t / dur
    n = _ruido(t.size, 21)
    ruido = pasa_altos(n, 22) * (1 - k) + pasa_altos(n, 3) * k
    return (ruido * 0.6 + _barrido(180 + 700 * k**2) * 0.5) * k**2.2  # termina en seco


def sfx_grillos(dur=3.0):
    y = np.zeros(int(dur * SR), dtype=np.float32)
    pulso_t = _t(0.025)
    pulso = np.sin(2 * np.pi * 4700 * pulso_t) * np.hanning(pulso_t.size)
    for desfase, periodo, gan in ((0.0, 0.85, 1.0), (0.37, 1.1, 0.6)):
        tiempo = desfase
        while tiempo < dur - 0.2:
            for k in range(3):
                i0 = int((tiempo + k * 0.045) * SR)
                y[i0 : i0 + pulso.size] += pulso * gan
            tiempo += periodo
    return y


def sfx_comico(dur=0.45):
    t = _t(dur)
    f = (1100 * np.exp(-t / 0.22) + 260) * (1 + 0.03 * np.sin(2 * np.pi * 14 * t))
    return _barrido(f) * np.sin(np.pi * t / dur) ** 0.5 * 0.8


def sfx_impacto(dur=0.9):
    t = _t(dur)
    sub = _barrido(32 + 60 * np.exp(-t / 0.08)) * _caida(t, 0.3)
    return sub * 1.2 + pasa_bajos(_ruido(t.size, 41), 12) * _caida(t, 0.05) * 2.5


def sfx_ding_error(dur=0.45):
    t = _t(dur)
    y = np.zeros_like(t)
    for inicio, f in ((0.0, 660), (0.13, 523)):
        tt = np.clip(t - inicio, 0, None)
        y += (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)) * _caida(tt, 0.11) * (t >= inicio)
    return y * 0.6


def sfx_cepillado(dur=1.7):
    t = _t(dur)
    n = pasa_altos(pasa_bajos(_ruido(t.size, 51), 3), 25)
    return n * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 22 * t))) * np.sin(np.pi * t / dur) ** 0.8 * 1.6


def sfx_brillo(dur=0.9):
    t = _t(dur)
    y = np.zeros_like(t)
    for k, f in enumerate((1568, 2093, 2637, 3136, 4186)):
        tt = np.clip(t - k * 0.06, 0, None)
        y += np.sin(2 * np.pi * f * tt) * _caida(tt, 0.25) * (t >= k * 0.06) * (0.5 - k * 0.05)
    return y


def sfx_pop(dur=0.1):
    t = _t(dur)
    return _barrido(280 + 900 * (t / dur)) * _caida(t, 0.03) + sfx_clic(dur) * 0.3


def sfx_golpe(dur=2.2):
    """«Boom» grave del cierre (≈60→35 Hz, centroide ≈0,3 kHz como el de las referencias)."""
    t = _t(dur)
    sub = _barrido(35 + 45 * np.exp(-t / 0.12)) * _caida(t, 0.55)
    cuerpo = _barrido(70 + 50 * np.exp(-t / 0.05)) * _caida(t, 0.25) * 0.6
    golpe = pasa_bajos(_ruido(t.size, 77), 18) * _caida(t, 0.06) * 3.0
    cola = pasa_bajos(_ruido(t.size, 78), 40) * _caida(t, 0.7) * 0.25  # reverberación grave
    y = (sub + cuerpo + golpe + cola) * _ataque(t, 0.004)
    return np.tanh(y * 1.6) / np.tanh(1.6)  # saturación suave: más «pegada»


GENERADORES_SFX = {
    "whoosh": sfx_whoosh,
    "clic": sfx_clic,
    "burbujas": sfx_burbujas,
    "velcro": sfx_velcro,
    "plop": sfx_plop,
    "crujido": sfx_crujido,
    "efervescencia": sfx_efervescencia,
    "riser": sfx_riser,
    "grillos": sfx_grillos,
    "comico": sfx_comico,
    "impacto": sfx_impacto,
    "ding_error": sfx_ding_error,
    "cepillado": sfx_cepillado,
    "brillo": sfx_brillo,
    "pop": sfx_pop,
    "golpe": sfx_golpe,
}


def obtener_sfx(nombre, duracion=None):
    """Usa assets/audios/sfx/<nombre>.<ext> si existe; si no, lo genera."""
    propio = buscar_archivo(config.DIR_SFX, nombre, EXT_AUDIO)
    if propio:
        x = cargar_audio(propio)
        return x[: int(duracion * SR)] if duracion else x
    if nombre not in GENERADORES_SFX:
        raise ErrorProduccion(
            f"Efecto de sonido desconocido «{nombre}» en guion.py. "
            f"Usa uno de: {', '.join(sorted(GENERADORES_SFX))} o añade assets/audios/sfx/{nombre}.wav"
        )
    gen = GENERADORES_SFX[nombre]
    x = gen(duracion) if duracion else gen()
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.7).astype(np.float32)


# ---------------------------------------------------------------------------
# Mezcla final
# ---------------------------------------------------------------------------
def mezclar(pistas_voz, eventos_sfx, t_golpe, duracion, con_voz=True):
    """
    Voz + efectos discretos + golpe final (+ música opcional). Devuelve un array mono.
    `con_voz=False` produce la versión sin narración con exactamente los mismos efectos.
    """
    n = int(round(duracion * SR))
    mezcla = np.zeros(n + SR * 4, dtype=np.float32)
    if con_voz:
        for t0, x in pistas_voz:
            i0 = int(t0 * SR)
            mezcla[i0 : i0 + x.size] += x * db(config.NIVELES_DB["narracion"])
    for t0, x, gan in eventos_sfx:
        i0 = max(0, int(t0 * SR))
        mezcla[i0 : i0 + x.size] += x * gan * db(config.NIVELES_DB["sfx"])
    golpe = obtener_sfx("golpe")
    i0 = int(t_golpe * SR)
    # El golpe suena ~7 dB por encima de la voz (como en las referencias); el limitador final lo contiene.
    mezcla[i0 : i0 + golpe.size] += golpe / 0.7 * db(config.NIVELES_DB["golpe_final"])
    if config.MUSICA_DE_FONDO:
        propia = buscar_archivo(config.DIR_MUSICA, "musica_fondo", EXT_AUDIO)
        if propia:
            m = cargar_audio(propia)
            m = np.tile(m, int(np.ceil(n / m.size)))[:n]
            m = m / (np.sqrt(np.mean(m**2)) + 1e-9) * 0.1
            corte = int(min(n, (t_golpe - 0.6) * SR))
            m[corte:] *= np.linspace(1, 0, n - corte) ** 2 if n > corte else 1
            mezcla[:n] += m * db(config.NIVELES_DB["musica"])
    mezcla = mezcla[:n]
    f = int(0.008 * SR)
    mezcla[:f] *= np.linspace(0, 1, f)
    mezcla[-f:] *= np.linspace(1, 0, f)
    return mezcla
