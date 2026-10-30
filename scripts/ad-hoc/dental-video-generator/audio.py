"""
Audio: narración (grabada o texto a voz), alineación palabra a palabra,
música generada, efectos de sonido generados y mezcla con ducking.

Todo el audio se procesa como arrays de numpy a 44,1 kHz. FFmpeg se usa solo
para decodificar archivos del usuario y para cambiar el tempo de la voz.
"""

import asyncio
import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

import config
from utilidades import ErrorProduccion, aleatorio

SR = config.AUDIO_SR
EXT_AUDIO = [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"]


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


def recortar_silencios(x, umbral_rel=0.03, margen=0.02):
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
    filtros = []
    f = factor
    while f > 2.0:
        filtros.append("atempo=2.0")
        f /= 2.0
    filtros.append(f"atempo={f:.4f}")
    entrada = Path(temp_dir) / "_tempo_in.wav"
    salida = Path(temp_dir) / "_tempo_out.wav"
    guardar_wav(entrada, x)
    ejecutar(
        [ffmpeg_exe(), "-v", "error", "-y", "-i", str(entrada), "-filter:a", ",".join(filtros), "-ac", "1", str(salida)],
        "el ajuste de tempo de la narración",
    )
    y = cargar_audio(salida)
    entrada.unlink(missing_ok=True)
    salida.unlink(missing_ok=True)
    return y


# ---------------------------------------------------------------------------
# Texto a voz
# ---------------------------------------------------------------------------
def texto_para_voz(texto):
    for original, dicho in config.PRONUNCIACION.items():
        texto = texto.replace(original, dicho)
    return texto


def _espeak():
    return shutil.which("espeak-ng") or shutil.which("espeak")


_VOZ_ESPEAK = {}


def voz_espeak_disponible(temp_dir):
    """Devuelve la primera voz de config.VOCES_ESPEAK que funciona (p. ej. MBROLA femenina)."""
    if "voz" in _VOZ_ESPEAK:
        return _VOZ_ESPEAK["voz"]
    exe = _espeak()
    voz_ok = None
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


def sintetizar_espeak(texto, salida, voz, wpm):
    exe = _espeak()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(texto)
        ruta_txt = f.name
    try:
        ejecutar(
            [exe, "-v", voz, "-s", str(int(wpm)), "-p", "54", "-a", "170", "-g", "2", "-w", str(salida), "-f", ruta_txt],
            "la síntesis de voz con espeak-ng",
        )
    finally:
        os.unlink(ruta_txt)


def edge_disponible():
    try:
        import edge_tts  # noqa: F401

        return True
    except ImportError:
        return False


def sintetizar_edge(texto, salida, velocidad_pct=0):
    import edge_tts

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    async def _tarea():
        com = edge_tts.Communicate(texto, config.VOZ_EDGE, rate=f"{velocidad_pct:+d}%", proxy=proxy)
        await com.save(str(salida))

    asyncio.run(asyncio.wait_for(_tarea(), timeout=45))
    if not Path(salida).exists() or Path(salida).stat().st_size < 1000:
        raise ErrorProduccion("edge-tts no devolvió audio")


def elegir_motor(temp_dir):
    """Decide el motor de voz según config.MOTOR_TTS y lo que haya instalado."""
    pedido = config.MOTOR_TTS
    if pedido in ("auto", "edge") and edge_disponible():
        try:
            prueba = Path(temp_dir) / "_prueba_edge.mp3"
            sintetizar_edge("Hola.", prueba)
            prueba.unlink(missing_ok=True)
            return "edge"
        except Exception as exc:
            if pedido == "edge":
                raise ErrorProduccion(f"edge-tts no está disponible (¿sin internet?): {exc}") from exc
    if pedido in ("auto", "espeak", "edge") and voz_espeak_disponible(temp_dir):
        return "espeak"
    if pedido == "espeak":
        raise ErrorProduccion("espeak-ng no está instalado. Ver README → «Voz sin internet».")
    return "silencio"


def sintetizar_bloque(texto, motor, temp_dir, factor_velocidad=1.0):
    """Genera la voz de un bloque y la devuelve como array (con caché en temp/)."""
    texto_voz = texto_para_voz(texto)
    clave = hashlib.sha1(f"{motor}|{texto_voz}|{factor_velocidad:.3f}|{config.VOZ_EDGE}".encode()).hexdigest()[:12]
    cache = Path(temp_dir) / "narracion" / f"tts_{clave}.wav"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        if motor == "edge":
            tmp = cache.with_suffix(".mp3")
            sintetizar_edge(texto_voz, tmp, int(round((factor_velocidad - 1) * 100)))
            guardar_wav(cache, cargar_audio(tmp))
            tmp.unlink(missing_ok=True)
        else:
            voz = voz_espeak_disponible(temp_dir)
            bruto = cache.with_name(cache.stem + "_raw.wav")
            sintetizar_espeak(texto_voz, bruto, voz, config.VELOCIDAD_ESPEAK * factor_velocidad)
            guardar_wav(cache, cargar_audio(bruto))
            bruto.unlink(missing_ok=True)
    return cargar_audio(cache)


# ---------------------------------------------------------------------------
# Alineación palabra a palabra (para subtítulos y sincronización de imágenes)
# ---------------------------------------------------------------------------
_VOCALES = re.compile(r"[aeiouáéíóúü]+", re.IGNORECASE)


def _peso_palabra(p):
    digitos = sum(c.isdigit() for c in p)
    silabas = len(_VOCALES.findall(p)) + digitos * 2
    return 0.6 + max(1, silabas)


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
    """Devuelve (inicio_voz, fin_voz, [pausas]) en segundos relativos al array."""
    salto = int(sr * 0.01)
    n = max(1, x.size // salto)
    rms = np.sqrt(np.mean(x[: n * salto].reshape(n, salto) ** 2, axis=1) + 1e-12)
    if rms.max() <= 1e-6:
        return 0.0, x.size / sr, []
    umbral = rms.max() * 0.07
    voz = rms > umbral
    indices = np.nonzero(voz)[0]
    v0, v1 = indices[0], indices[-1] + 1
    pausas = []
    i = v0
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


def alinear_palabras(texto, x, desplazamiento):
    """
    Estima cuándo se dice cada palabra. Reparte el tiempo de voz según las
    sílabas y ancla los signos de puntuación a los silencios reales del audio.
    """
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
    anclas_u, anclas_t = [0.0], [v0]
    usadas = set()
    for i, pa in enumerate(pausas_t[:-1]):
        if pa <= 0:
            continue
        estimado = v0 + (v1 - v0) * (fines[i] + pa / 2) / total
        mejor, dist = None, 0.7
        for k, (p0, p1) in enumerate(pausas):
            centro = (p0 + p1) / 2
            if k in usadas or centro <= anclas_t[-1]:
                continue
            if abs(centro - estimado) < dist:
                mejor, dist = k, abs(centro - estimado)
        if mejor is not None:
            usadas.add(mejor)
            p0, p1 = pausas[mejor]
            anclas_u += [fines[i], fines[i] + pa]
            anclas_t += [p0, p1]
    anclas_u.append(total)
    anclas_t.append(v1)
    au, at = [anclas_u[0]], [anclas_t[0]]
    for a, b in zip(anclas_u[1:], anclas_t[1:]):
        if a > au[-1] and b > at[-1]:
            au.append(a)
            at.append(b)
    resultado = []
    for p, s, e in zip(palabras, inicios, fines):
        resultado.append(
            {
                "texto": p,
                "inicio": round(float(np.interp(s, au, at)) + desplazamiento, 3),
                "fin": round(float(np.interp(e, au, at)) + desplazamiento, 3),
            }
        )
    return resultado


def alinear_por_proporcion(texto, inicio, fin):
    """Sin audio (motor «silencio»): reparte las palabras en el tiempo del bloque."""
    x = np.ones(int((fin - inicio - 0.2) * SR), dtype=np.float32) * 0.5
    return alinear_palabras(texto, x, inicio + 0.1)


# ---------------------------------------------------------------------------
# Preparación de la narración completa
# ---------------------------------------------------------------------------
def preparar_narracion(bloques, temp_dir, informar=print):
    """
    Devuelve un dict con:
      pistas: lista de (instante_inicio, array) para mezclar
      palabras: {id_bloque: [{texto, inicio, fin}, ...]}
      motor: origen de la voz
    """
    temp_dir = Path(temp_dir)
    (temp_dir / "narracion").mkdir(parents=True, exist_ok=True)
    resultado = {"pistas": [], "palabras": {}, "motor": None}

    completa = buscar_archivo(config.DIR_NARRACION, "narracion_completa", EXT_AUDIO)
    if completa:
        informar(f"  · Narración grabada completa: {completa.name}")
        x = cargar_audio(completa)
        resultado["motor"] = f"grabación ({completa.name})"
        resultado["pistas"].append((0.0, x))
        for b in bloques:
            seg = x[int(b["inicio"] * SR) : int(b["fin"] * SR)]
            if seg.size < SR * 0.2 or np.max(np.abs(seg)) < 1e-4:
                informar(f"  ! El bloque {b['id']} no tiene voz en {b['inicio']}–{b['fin']} s; se reparte por proporción")
                resultado["palabras"][b["id"]] = alinear_por_proporcion(b["narracion"], b["inicio"], b["fin"])
            else:
                resultado["palabras"][b["id"]] = alinear_palabras(b["narracion"], seg, b["inicio"])
        return resultado

    por_bloque = {b["id"]: buscar_archivo(config.DIR_NARRACION, f"bloque_{b['id']:02d}", EXT_AUDIO) for b in bloques}
    hay_grabaciones = any(por_bloque.values())
    motor = None
    if not all(por_bloque.values()):
        motor = elegir_motor(temp_dir)
    if hay_grabaciones:
        informar(f"  · Grabaciones por bloque encontradas: {sum(1 for v in por_bloque.values() if v)} de {len(bloques)}")
    if motor:
        informar(f"  · Motor de voz para los bloques sin grabación: {motor}")
    resultado["motor"] = "grabación por bloques" if hay_grabaciones and not motor else motor

    for b in bloques:
        hueco = b["fin"] - b["inicio"] - config.MARGEN_INICIO_BLOQUE - config.MARGEN_FIN_BLOQUE
        inicio_voz = b["inicio"] + config.MARGEN_INICIO_BLOQUE
        if por_bloque[b["id"]]:
            x = recortar_silencios(cargar_audio(por_bloque[b["id"]]))
        elif motor == "silencio":
            resultado["palabras"][b["id"]] = alinear_por_proporcion(b["narracion"], b["inicio"], b["fin"])
            continue
        else:
            x = recortar_silencios(sintetizar_bloque(b["narracion"], motor, temp_dir))
            factor = (x.size / SR) / hueco
            if factor > 1.0:
                # Primero se regenera más rápido (suena más natural) y luego se ajusta fino.
                x = recortar_silencios(sintetizar_bloque(b["narracion"], motor, temp_dir, min(factor * 1.03, 1.6)))
        dur = x.size / SR
        if dur > hueco:
            factor = dur / hueco
            if factor > config.TEMPO_MAXIMO:
                informar(
                    f"  ! Bloque {b['id']}: la voz dura {dur:.2f} s y el hueco es {hueco:.2f} s "
                    f"(se acelera ×{factor:.2f}). Acorta el texto o alarga el bloque en guion.py."
                )
            x = cambiar_tempo(x, factor, temp_dir)
        # Normalización por bloque para que todos suenen igual de fuerte.
        rms = np.sqrt(np.mean(x**2)) + 1e-9
        x = x * (0.1 / rms)
        resultado["pistas"].append((inicio_voz, x))
        resultado["palabras"][b["id"]] = alinear_palabras(b["narracion"], x, inicio_voz)
    return resultado


def envolvente_voz_por_fotograma(pistas, duracion, fps):
    """Intensidad de la voz (0–1) en cada fotograma: anima la boca de la dentista ilustrada."""
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


def _barrido(f, dur):
    """Seno con frecuencia variable f(t) (array)."""
    fase = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(fase).astype(np.float32)


def _caida(t, tau):
    return np.exp(-t / tau).astype(np.float32)


def _ataque(t, a):
    return np.clip(t / max(a, 1e-4), 0, 1).astype(np.float32)


def midi_a_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------------------------------------------------------------------------
# Efectos de sonido generados (se reemplazan por assets/audios/sfx/<nombre>.wav)
# ---------------------------------------------------------------------------
def sfx_squeak(dur=0.3):
    t = _t(dur)
    f = 2300 + 1000 * np.sin(np.pi * t / dur) + 140 * np.sin(2 * np.pi * 26 * t)
    y = _barrido(f, dur) * 0.7 + _barrido(2 * f, dur) * 0.15
    return y * _ataque(t, 0.02) * np.sin(np.pi * t / dur) ** 0.6


def sfx_whoosh(dur=0.45):
    t = _t(dur)
    n = _ruido(t.size, 11)
    grave, agudo = pasa_bajos(n, 22), pasa_altos(pasa_bajos(n, 3), 12)
    k = t / dur
    y = grave * (1 - k) * 2.2 + agudo * k * 0.9
    return y * np.sin(np.pi * k) ** 2


def sfx_clic(dur=0.08):
    t = _t(dur)
    y = _ruido(t.size, 3) * _caida(t, 0.0015) + 0.6 * np.sin(2 * np.pi * 1900 * t) * _caida(t, 0.006)
    t2 = np.clip(t - 0.012, 0, None)
    y += (0.5 * np.sin(2 * np.pi * 2600 * t2) * _caida(t2, 0.004)) * (t > 0.012)
    return y


def sfx_burbujas(dur=1.3):
    t = _t(dur)
    y = np.zeros_like(t)
    rng = aleatorio(5)
    for inicio in np.sort(rng.uniform(0, dur - 0.1, 11)):
        i0 = int(inicio * SR)
        tt = _t(0.07)
        f = 320 + 900 * (tt / 0.07) ** 0.7 * rng.uniform(0.7, 1.3)
        y[i0 : i0 + tt.size] += _barrido(f, 0.07) * _caida(tt, 0.025) * rng.uniform(0.4, 0.9)
    return y


def sfx_velcro(dur=0.5):
    t = _t(dur)
    rng = aleatorio(8)
    impulsos = (rng.random(t.size) < 140 / SR).astype(np.float32) * rng.uniform(0.3, 1, t.size).astype(np.float32)
    granos = np.convolve(impulsos, np.exp(-np.arange(int(0.004 * SR)) / (0.0012 * SR)), mode="same")
    y = pasa_altos(_ruido(t.size, 9), 6) * (0.25 + granos * 2.5)
    return y * np.sin(np.pi * t / dur) ** 0.5


def sfx_ticks_rapidos(dur=0.75):
    y = np.zeros(int(dur * SR), dtype=np.float32)
    clic = sfx_clic(0.03)
    tiempo, intervalo, alto = 0.0, 0.12, True
    while tiempo < dur - 0.03:
        i0 = int(tiempo * SR)
        y[i0 : i0 + clic.size] += clic * (0.8 if alto else 0.55)
        tiempo += intervalo
        intervalo = max(0.03, intervalo * 0.86)
        alto = not alto
    return y


def sfx_plop(dur=0.12):
    t = _t(dur)
    f = 180 + 600 * np.exp(-t / 0.025)
    return _barrido(f, dur) * _caida(t, 0.035) * _ataque(t, 0.002)


def sfx_crujido(dur=0.4):
    y = np.zeros(int(dur * SR), dtype=np.float32)
    rng = aleatorio(13)
    for inicio in rng.uniform(0, dur - 0.02, 26):
        largo = int(rng.uniform(0.004, 0.012) * SR)
        i0 = int(inicio * SR)
        g = pasa_altos(_ruido(largo, int(inicio * 1e5)), 4) * np.hanning(largo) * rng.uniform(0.4, 1)
        y[i0 : i0 + largo] += g
    return y


def sfx_efervescencia(dur=1.4):
    t = _t(dur)
    rng = aleatorio(17)
    chispas = (rng.random(t.size) < 900 / SR).astype(np.float32)
    chispas = np.convolve(chispas, np.exp(-np.arange(80) / 15.0), mode="same")
    y = pasa_altos(_ruido(t.size, 18), 3) * (0.15 + chispas)
    return y * _ataque(t, 0.15) * np.clip((dur - t) / 0.5, 0, 1)


def sfx_riser(dur=1.5):
    t = _t(dur)
    k = t / dur
    n = _ruido(t.size, 21)
    ruido = pasa_altos(n, 22) * (1 - k) + pasa_altos(n, 3) * k  # el ruido se vuelve más agudo
    tono = _barrido(180 + 700 * k**2, dur)
    return (ruido * 0.6 + tono * 0.5) * k**2.2  # termina en seco, sin fundido


def sfx_grillos(dur=4.0):
    y = np.zeros(int(dur * SR), dtype=np.float32)
    pulso_t = _t(0.025)
    pulso = np.sin(2 * np.pi * 4700 * pulso_t) * np.hanning(pulso_t.size)
    for desfase, freq_chirrido, gan in ((0.0, 0.85, 1.0), (0.37, 1.1, 0.6)):
        tiempo = desfase
        while tiempo < dur - 0.2:
            for k in range(3):
                i0 = int((tiempo + k * 0.045) * SR)
                y[i0 : i0 + pulso.size] += pulso * gan
            tiempo += freq_chirrido
    return y


def sfx_rio(dur=3.2):
    t = _t(dur)
    y = pasa_bajos(_ruido(t.size, 31), 30) * 3 + pasa_altos(pasa_bajos(_ruido(t.size, 32), 4), 20) * 0.4
    mod = 0.7 + 0.3 * np.sin(2 * np.pi * 1.7 * t) * np.sin(2 * np.pi * 0.6 * t)
    apagado = np.clip(1 - (t / dur) ** 1.4, 0, 1)  # el «río» de saliva se apaga
    return y * mod * apagado * _ataque(t, 0.3)


def sfx_comico(dur=0.45):
    t = _t(dur)
    f = 1100 * np.exp(-t / 0.22) + 260  # silbato descendente
    vib = 1 + 0.03 * np.sin(2 * np.pi * 14 * t)
    return _barrido(f * vib, dur) * np.sin(np.pi * t / dur) ** 0.5 * 0.8


def sfx_impacto(dur=1.0):
    t = _t(dur)
    f = 32 + 60 * np.exp(-t / 0.08)
    sub = _barrido(f, dur) * _caida(t, 0.35)
    golpe = pasa_bajos(_ruido(t.size, 41), 12) * _caida(t, 0.05) * 2.5
    return sub * 1.2 + golpe


def sfx_ding_error(dur=0.45):
    t = _t(dur)
    y = np.zeros_like(t)
    for inicio, f in ((0.0, 660), (0.13, 523)):
        tt = np.clip(t - inicio, 0, None)
        tono = np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)
        y += tono * _caida(tt, 0.11) * (t >= inicio)
    return y * 0.6


def sfx_cepillado(dur=1.7):
    t = _t(dur)
    n = pasa_altos(pasa_bajos(_ruido(t.size, 51), 3), 25)
    cerdas = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 22 * t))
    return n * cerdas * np.sin(np.pi * t / dur) ** 0.8 * 1.6


def sfx_brillo(dur=0.9):
    t = _t(dur)
    y = np.zeros_like(t)
    for k, f in enumerate((1568, 2093, 2637, 3136, 4186)):
        inicio = k * 0.06
        tt = np.clip(t - inicio, 0, None)
        y += np.sin(2 * np.pi * f * tt) * _caida(tt, 0.25) * (t >= inicio) * (0.5 - k * 0.05)
    return y


def sfx_pop(dur=0.1):
    t = _t(dur)
    f = 280 + 900 * (t / dur)
    return _barrido(f, dur) * _caida(t, 0.03) + sfx_clic(dur) * 0.3


GENERADORES_SFX = {
    "squeak": sfx_squeak,
    "whoosh": sfx_whoosh,
    "clic": sfx_clic,
    "burbujas": sfx_burbujas,
    "velcro": sfx_velcro,
    "ticks_rapidos": sfx_ticks_rapidos,
    "plop": sfx_plop,
    "crujido": sfx_crujido,
    "efervescencia": sfx_efervescencia,
    "riser": sfx_riser,
    "grillos": sfx_grillos,
    "rio": sfx_rio,
    "comico": sfx_comico,
    "impacto": sfx_impacto,
    "ding_error": sfx_ding_error,
    "cepillado": sfx_cepillado,
    "brillo": sfx_brillo,
    "pop": sfx_pop,
}


def obtener_sfx(nombre, duracion=None):
    """Usa assets/audios/sfx/<nombre>.<ext> si existe; si no, lo genera."""
    propio = buscar_archivo(config.DIR_SFX, nombre, EXT_AUDIO)
    if propio:
        x = cargar_audio(propio)
        if duracion:
            x = x[: int(duracion * SR)]
        return x
    if nombre not in GENERADORES_SFX:
        raise ErrorProduccion(
            f"Efecto de sonido desconocido «{nombre}» en guion.py. "
            f"Usa uno de: {', '.join(sorted(GENERADORES_SFX))} o añade assets/audios/sfx/{nombre}.wav"
        )
    gen = GENERADORES_SFX[nombre]
    x = gen(duracion) if duracion else gen()
    pico = np.max(np.abs(x)) + 1e-9
    return (x / pico * 0.7).astype(np.float32)


# ---------------------------------------------------------------------------
# Música generada: electrónica minimalista a 105 BPM con pulso de reloj
# ---------------------------------------------------------------------------
PROGRESIONES = {
    "base": [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [48, 52, 55]), (55, [55, 59, 62])],
    "oscura": [(57, [57, 60, 64]), (50, [50, 53, 57]), (46, [46, 50, 53]), (52, [52, 56, 59])],
    "mayor": [(48, [48, 52, 55]), (53, [53, 57, 60]), (55, [55, 59, 62]), (48, [48, 52, 55])],
}


def _nota_tic(agudo):
    t = _t(0.03)
    f = 3200 if agudo else 2350
    return (np.sin(2 * np.pi * f * t) * _caida(t, 0.004) + _ruido(t.size, 61) * _caida(t, 0.0015) * 0.5)


def _nota_bombo():
    t = _t(0.32)
    f = 45 + 85 * np.exp(-t / 0.035)
    return _barrido(f, 0.32) * _caida(t, 0.16) * _ataque(t, 0.002)


def _nota_bajo(midi, dur):
    t = _t(dur)
    f = midi_a_hz(midi)
    y = sum(np.sin(2 * np.pi * f * k * t) / k ** 1.3 for k in range(1, 7))
    return y * _ataque(t, 0.006) * _caida(t, dur * 0.55)


def _nota_pad(midis, dur):
    t = _t(dur)
    y = np.zeros_like(t)
    for m in midis:
        f = midi_a_hz(m)
        for desafine in (-0.004, 0.004):
            y += np.sin(2 * np.pi * f * (1 + desafine) * t)
        y += 0.3 * np.sin(2 * np.pi * 2 * f * t)
    env = np.clip(t / 0.35, 0, 1) * np.clip((dur - t) / 0.35, 0, 1)
    return y * env / len(midis)


def _nota_hat():
    t = _t(0.05)
    return pasa_altos(_ruido(t.size, 71), 2) * _caida(t, 0.012)


def resolver_secciones(secciones, resolver):
    return [dict(s, inicio=resolver(s["inicio"]), fin=resolver(s["fin"])) for s in secciones]


def _seccion_en(secciones, t):
    for s in secciones:
        if s["inicio"] <= t < s["fin"]:
            return s
    return secciones[-1]


def envolvente_secciones(secciones, n, rampa=0.02):
    """Ganancia por muestra según las secciones, con rampas cortas para evitar chasquidos."""
    env = np.zeros(n, dtype=np.float32)
    noche = np.zeros(n, dtype=np.float32)
    for s in secciones:
        i0, i1 = int(s["inicio"] * SR), min(n, int(s["fin"] * SR))
        env[i0:i1] = s.get("ganancia", 1.0)
        noche[i0:i1] = 1.0 if s.get("noche") else 0.0
    return pasa_bajos(env, rampa * SR), pasa_bajos(noche, 0.3 * SR)


def sintetizar_musica(secciones, duracion, bpm):
    n = int(round(duracion * SR))
    pulso = 60.0 / bpm
    compas = 4 * pulso
    capas = {k: np.zeros(n + SR * 3, dtype=np.float32) for k in ("tic", "bombo", "bajo", "pad", "hats")}

    def poner(capa, t0, x, gan=1.0):
        i0 = int(t0 * SR)
        capas[capa][i0 : i0 + x.size] += x * gan

    num_compases = int(math.ceil(duracion / compas))
    for b in range(num_compases):
        t_compas = b * compas
        sec = _seccion_en(secciones, t_compas + 0.01)
        raiz, acorde = PROGRESIONES[sec.get("armonia", "base")][b % 4]
        bajo = raiz
        while bajo > 47:
            bajo -= 12
        if "pad" in sec["capas"]:
            poner("pad", t_compas, _nota_pad([m + 12 for m in acorde] + [raiz], compas + 0.3), 0.22)
        for corchea in range(8):
            t = t_compas + corchea * pulso / 2
            if t >= duracion:
                break
            capas_aqui = _seccion_en(secciones, t + 0.001)["capas"]
            if "tic" in capas_aqui:
                poner("tic", t, _nota_tic(corchea % 2 == 0), 0.32 if corchea % 2 == 0 else 0.2)
            if "hats" in capas_aqui and corchea % 2 == 1:
                poner("hats", t, _nota_hat(), 0.16)
            if corchea % 2 == 0:
                if "bombo" in capas_aqui and corchea in (0, 4):
                    poner("bombo", t, _nota_bombo(), 0.9)
                if "bajo" in capas_aqui:
                    poner("bajo", t + pulso / 2 * 0.98, _nota_bajo(bajo, pulso * 0.48), 0.34)
    musica = sum(c[:n] for c in capas.values())
    return musica.astype(np.float32)


def preparar_musica(secciones, duracion, bpm, informar=print):
    n = int(round(duracion * SR))
    propia = buscar_archivo(config.DIR_MUSICA, "musica_fondo", EXT_AUDIO)
    if propia:
        informar(f"  · Música propia: {propia.name}")
        x = cargar_audio(propia)
        if x.size < n:
            x = np.tile(x, int(math.ceil(n / x.size)))
        x = x[:n]
    else:
        informar("  · Música: pista electrónica generada a 105 BPM (sin derechos de autor)")
        x = sintetizar_musica(secciones, duracion, bpm)
    env, noche = envolvente_secciones(secciones, n)
    filtrada = pasa_bajos(pasa_bajos(x, 28), 28)  # versión nocturna «filtrada»
    x = x * (1 - noche) + filtrada * noche * 1.3
    rms = np.sqrt(np.mean(x[x != 0] ** 2)) if np.any(x != 0) else 1.0
    return (x / (rms + 1e-9) * 0.1 * env).astype(np.float32)


# ---------------------------------------------------------------------------
# Mezcla final con ducking
# ---------------------------------------------------------------------------
def mezclar(pistas_voz, musica, eventos_sfx, duracion):
    """
    pistas_voz: [(t0, array)], musica: array, eventos_sfx: [(t0, array, ganancia)].
    La música baja automáticamente cuando habla la narradora.
    """
    n = int(round(duracion * SR))
    voz = np.zeros(n, dtype=np.float32)
    for t0, x in pistas_voz:
        i0 = int(t0 * SR)
        i1 = min(n, i0 + x.size)
        if i1 > i0:
            voz[i0:i1] += x[: i1 - i0]
    voz *= db(config.NIVELES_DB["narracion"])

    # Detección de voz → ganancia de la música (ataque rápido, liberación suave).
    ventana = int(0.03 * SR)
    energia = pasa_bajos(np.abs(voz), ventana)
    activa = (energia > 0.012).astype(np.float32)
    activa = np.clip(pasa_bajos(activa, 0.25 * SR) * 3, 0, 1)
    activa = pasa_bajos(activa, 0.12 * SR)
    g_musica = db(config.NIVELES_DB["musica_libre"]) * (1 - activa) + db(config.NIVELES_DB["musica_bajo_voz"]) * activa
    # La música llega normalizada a RMS 0,1 (igual que la voz): se escala a su nivel relativo.
    mezcla = voz + musica[:n] * g_musica * 1.0

    sfx = np.zeros(n + SR * 5, dtype=np.float32)
    for t0, x, gan in eventos_sfx:
        i0 = max(0, int(t0 * SR))
        sfx[i0 : i0 + x.size] += x * gan
    mezcla += sfx[:n] * db(config.NIVELES_DB["sfx"])

    # Fundido de 10 ms en los extremos y normalización de pico a −1 dBFS.
    f = int(0.01 * SR)
    mezcla[:f] *= np.linspace(0, 1, f)
    mezcla[-f:] *= np.linspace(1, 0, f)
    pico = np.max(np.abs(mezcla)) + 1e-9
    return (mezcla / pico * db(-1.0)).astype(np.float32)
