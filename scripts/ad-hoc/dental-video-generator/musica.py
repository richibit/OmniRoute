"""
Banda sonora de «La última vez que tu mamá te cepilló los dientes» (sin voz):
caja de música y piano suaves sobre un fondo cálido, generados por síntesis
aditiva y con reverberación por convolución, más los efectos del guion
(agua, cepillado, notificación, cajita de lata, monedas, lápiz, papel, cajón).

Se reemplaza por música propia poniendo assets/audios/musica/musica_espejo.(wav|mp3)
y cualquier efecto con assets/audios/sfx/<nombre>.wav.
"""

import math

import numpy as np

import audio
import guion_espejo as GE
from audio import SR, pasa_altos, pasa_bajos
from utilidades import aleatorio


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _t(dur):
    return np.arange(int(dur * SR), dtype=np.float32) / SR


# ---------------------------------------------------------------------------
# Timbres
# ---------------------------------------------------------------------------
def caja_musica(midi, dur=2.2, vel=0.6):
    """Púa metálica: parciales ligeramente inarmónicos con caída rápida."""
    t = _t(dur)
    f = midi_hz(midi)
    y = np.zeros_like(t)
    for k, (mult, amp, tau) in enumerate(((1.0, 1.0, 0.9), (2.01, 0.35, 0.45), (3.02, 0.14, 0.3), (4.17, 0.1, 0.18), (5.4, 0.05, 0.1))):
        y += amp * np.sin(2 * np.pi * f * mult * t + k) * np.exp(-t / tau)
    y *= np.clip(t / 0.002, 0, 1)
    return (y * vel * 0.5).astype(np.float32)


def piano(midi, dur=3.0, vel=0.5):
    """Piano suave: armónicos con caída por frecuencia, leve inarmonicidad y golpe de martillo."""
    t = _t(dur)
    f = midi_hz(midi)
    y = np.zeros_like(t)
    for k in range(1, 9):
        fk = f * k * math.sqrt(1 + 0.00025 * k * k)
        if fk > SR / 2.2:
            break
        amp = (1 / k**1.25) * (0.6 + 0.4 * vel)
        y += amp * np.sin(2 * np.pi * fk * t) * np.exp(-t * (0.7 + 0.45 * k) / (1 + 0.004 * (108 - midi)))
    martillo = pasa_bajos(aleatorio(int(midi * 7)).standard_normal(t.size).astype(np.float32), 12) * np.exp(-t / 0.006) * 0.4
    y = (y + martillo) * np.clip(t / 0.004, 0, 1) * np.clip((dur - t) / 0.25, 0, 1)
    return (y * vel * 0.35).astype(np.float32)


def pad(midis, dur, vel=0.4):
    """Fondo cálido: senos desafinados con ataque y salida lentos."""
    t = _t(dur)
    y = np.zeros_like(t)
    for m in midis:
        f = midi_hz(m)
        for d in (-0.0035, 0.0, 0.0035):
            y += np.sin(2 * np.pi * f * (1 + d) * t + d * 900)
        y += 0.25 * np.sin(4 * np.pi * f * t)
    env = np.clip(t / 1.4, 0, 1) * np.clip((dur - t) / 1.6, 0, 1)
    return (y * env * vel * 0.06 / max(1, len(midis))).astype(np.float32)


def bajo(midi, dur=3.0, vel=0.5):
    t = _t(dur)
    f = midi_hz(midi)
    y = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
    return (y * np.clip(t / 0.02, 0, 1) * np.exp(-t / 1.8) * np.clip((dur - t) / 0.3, 0, 1) * vel * 0.3).astype(np.float32)


def reverb(x, segundos=2.6, mezcla=0.3, semilla=3):
    """Reverberación por convolución con una respuesta de ruido que decae (RT60 = segundos)."""
    n = int(segundos * SR)
    t = np.arange(n, dtype=np.float32) / SR
    ir = aleatorio(semilla).standard_normal(n).astype(np.float32) * np.exp(-t * 6.9 / segundos)
    ir = pasa_bajos(ir, 6)  # sala cálida (menos agudos)
    ir[: int(0.012 * SR)] = 0  # predelay
    ir /= np.sqrt(np.sum(ir**2)) + 1e-9
    tam = 1 << int(np.ceil(np.log2(x.size + n)))
    y = np.fft.irfft(np.fft.rfft(x, tam) * np.fft.rfft(ir, tam), tam)[: x.size].astype(np.float32)
    return x * (1 - mezcla) + y * mezcla * 2.2


# ---------------------------------------------------------------------------
# Composición (Do mayor, 72 BPM)
# ---------------------------------------------------------------------------
PULSO = 60 / 72
COMPAS = 4 * PULSO
ACORDES = {
    "C": (48, [60, 64, 67]), "G/B": (47, [59, 62, 67]), "Am": (45, [57, 60, 64]), "F": (41, [57, 60, 65]),
    "G": (43, [59, 62, 67]), "Em": (40, [59, 64, 67]), "Dm": (38, [57, 62, 65]),
}  # fmt: skip
MELODIA = [  # (pulso, nota, duración en pulsos) por compás; dos frases de 4 compases
    [(0, 76, 1), (1, 79, 1), (2, 84, 1.5), (3.5, 83, 0.5)],
    [(0, 81, 1), (1, 79, 1), (2, 74, 2)],
    [(0, 72, 1), (1, 76, 1), (2, 81, 1.5), (3.5, 79, 0.5)],
    [(0, 77, 1), (1, 76, 1), (2, 72, 2)],
    [(0, 79, 1), (1, 84, 1), (2, 88, 1.5), (3.5, 86, 0.5)],
    [(0, 86, 1), (1, 83, 1), (2, 79, 2)],
    [(0, 81, 1), (1, 84, 1), (2, 88, 1), (3, 86, 1)],
    [(0, 84, 1.5), (1.5, 81, 0.5), (2, 77, 2)],
]
ARPEGIO = [0, 2, 3, 1, 3, 2, 3, 1]  # índices sobre [raíz, quinta, octava, tercera aguda]

# Secciones: (inicio, fin, progresión, capas, intensidad)
SECCIONES = [
    (0.25, 3.3, ["C"], {"caja"}, 0.55),
    (3.3, 31.0, ["G/B", "Am", "F", "C"], {"caja", "piano"}, 0.55),
    (31.0, 39.0, ["Am", "F", "C", "G"], {"caja_lenta", "piano", "pad"}, 0.6),
    (39.0, 40.0, ["C"], {"pad"}, 0.5),
    (40.0, 47.0, ["Am", "F", "C"], {"piano_lento", "pad"}, 0.42),
    (48.0, 66.0, ["C", "G/B", "Am", "F"], {"caja_alta", "piano", "pad", "bajo"}, 0.85),
    (66.0, 72.0, ["F", "G", "C"], {"caja_lenta", "pad", "piano_lento"}, 0.5),
]


def componer(duracion):
    n = int(duracion * SR) + SR * 4
    pista = np.zeros(n, dtype=np.float32)

    def poner(t0, x):
        i0 = int(t0 * SR)
        if 0 <= i0 < n:
            fin = min(n, i0 + x.size)
            pista[i0:fin] += x[: fin - i0]

    frase = 0
    for inicio, fin, prog, capas, inten in SECCIONES:
        compas = COMPAS * (1.5 if "piano_lento" in capas and "caja_alta" not in capas else 1.0)
        t = inicio
        i = 0
        while t < fin - 0.2:
            raiz, tonos = ACORDES[prog[i % len(prog)]]
            dur_c = min(compas, fin - t)
            crec = 1.0 + (0.35 * (t - inicio) / max(1e-6, fin - inicio) if inicio == 48.0 else 0.0)
            if "pad" in capas:
                poner(t, pad(tonos + [raiz + 12], dur_c + 1.2, 0.5 * inten * crec))
            if "bajo" in capas:
                poner(t, bajo(raiz - 12 if raiz > 45 else raiz, dur_c, 0.55 * inten * crec))
            if "piano" in capas or "piano_lento" in capas:
                notas = [raiz, raiz + 7, raiz + 12, tonos[1] + 12]
                paso = PULSO / 2 if "piano" in capas else PULSO
                k = 0
                while k * paso < dur_c - 0.05:
                    poner(t + k * paso, piano(notas[ARPEGIO[k % 8]], 2.6, 0.45 * inten * crec))
                    k += 1
            if capas & {"caja", "caja_lenta", "caja_alta"}:
                for p, nota, d in MELODIA[frase % len(MELODIA)]:
                    if "caja_lenta" in capas and p % 2:
                        continue
                    if p * PULSO < dur_c:
                        poner(t + p * PULSO, caja_musica(nota + (12 if "caja_alta" in capas and nota < 80 else 0), 2.4, 0.6 * inten * crec))
                frase += 1
            t += compas
            i += 1
    # cortes de silencio que pide el guion (1 s antes de «25 AÑOS DESPUÉS»)
    s0, s1 = GE.SILENCIO_ANTES_DEL_SALTO
    i0, i1 = int(s0 * SR), int(s1 * SR)
    pista[i0 - int(0.25 * SR) : i0] *= np.linspace(1, 0, int(0.25 * SR)) ** 2
    pista[i0:i1] = 0
    pista = reverb(pista, 2.8, 0.32)
    pista[i0:i1] *= 0.0  # la cola de la reverberación también se corta: silencio real
    return pista[: int(duracion * SR)]


# ---------------------------------------------------------------------------
# Efectos de sonido del guion
# ---------------------------------------------------------------------------
def _ruido(n, s):
    return aleatorio(s).standard_normal(n).astype(np.float32)


def sfx_agua(dur=1.0):
    t = _t(dur)
    n = pasa_altos(pasa_bajos(_ruido(t.size, 101), 3), 18)
    rng = aleatorio(102)
    burbujas = np.zeros_like(t)
    for t0 in rng.uniform(0, dur - 0.05, 18):
        tt = _t(0.04)
        i0 = int(t0 * SR)
        burbujas[i0 : i0 + tt.size] += np.sin(2 * np.pi * (600 + 1800 * tt / 0.04) * tt) * np.exp(-tt / 0.012) * 0.3
    return (n * 0.5 + burbujas) * np.clip(t / 0.15, 0, 1) * np.clip((dur - t) / 0.2, 0, 1)


def sfx_cepillado_suave(dur=1.4):
    t = _t(dur)
    n = pasa_altos(pasa_bajos(_ruido(t.size, 111), 2), 10)
    trazo = 0.35 + 0.65 * np.abs(np.sin(2 * np.pi * 4.2 * t)) ** 2
    return n * trazo * np.clip(t / 0.1, 0, 1) * np.clip((dur - t) / 0.2, 0, 1) * 0.8


def sfx_vaso(dur=0.5):
    t = _t(dur)
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau) for f, a, tau in ((2480, 1.0, 0.18), (3910, 0.6, 0.12), (6120, 0.3, 0.07)))
    return y + pasa_bajos(_ruido(t.size, 121), 6) * np.exp(-t / 0.01) * 0.5


def sfx_pasos(dur=1.3):
    y = np.zeros(int(dur * SR), dtype=np.float32)
    for k, t0 in enumerate((0.05, 0.47, 0.89)):
        tt = _t(0.25)
        golpe = np.sin(2 * np.pi * (55 + 60 * np.exp(-tt / 0.03)) * tt) * np.exp(-tt / 0.06)
        golpe += pasa_bajos(_ruido(tt.size, 131 + k), 10) * np.exp(-tt / 0.02) * 0.6
        i0 = int(t0 * SR)
        y[i0 : i0 + tt.size] += golpe * (0.9 - k * 0.15)
    return y


def sfx_papel(dur=0.6):
    t = _t(dur)
    rng = aleatorio(141)
    env = np.convolve((rng.random(t.size) < 60 / SR).astype(np.float32), np.exp(-np.arange(int(0.03 * SR)) / (0.008 * SR)), "same")
    return pasa_altos(_ruido(t.size, 142), 3) * (0.15 + env * 1.5) * np.clip((dur - t) / 0.1, 0, 1)


def sfx_lata(dur=0.8):
    t = _t(dur)
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau) for f, a, tau in ((1180, 1.0, 0.25), (2710, 0.6, 0.16), (4430, 0.35, 0.09), (6060, 0.2, 0.05)))
    for t0 in (0.18, 0.27, 0.33):  # traqueteo de los dientitos dentro
        tt = np.clip(t - t0, 0, None)
        y += np.sin(2 * np.pi * 3300 * tt) * np.exp(-tt / 0.015) * (t >= t0) * 0.35
    return y


def sfx_monedas(dur=0.9):
    y = np.zeros(int(dur * SR), dtype=np.float32)
    rng = aleatorio(151)
    for t0 in sorted(rng.uniform(0, dur - 0.3, 6)):
        tt = _t(0.3)
        f = rng.uniform(4800, 7200)
        ping = (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 1.48 * tt)) * np.exp(-tt / 0.07)
        i0 = int(t0 * SR)
        y[i0 : i0 + tt.size] += ping * rng.uniform(0.4, 1.0)
    return y


def sfx_lapiz(dur=1.2):
    t = _t(dur)
    rng = aleatorio(161)
    trazo = np.clip(np.sin(2 * np.pi * rng.uniform(7, 10) * t + np.cumsum(rng.standard_normal(t.size)) * 0.002), 0, 1) ** 1.5
    return pasa_altos(pasa_bajos(_ruido(t.size, 162), 2), 6) * trazo * 0.9


def sfx_notificacion(dur=0.7):
    t = _t(dur)
    y = np.zeros_like(t)
    for t0, f in ((0.0, 1318.5), (0.12, 1760.0)):
        tt = np.clip(t - t0, 0, None)
        y += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt / 0.18) * (t >= t0)
    return y * 0.7


def sfx_cajon(dur=0.8):
    t = _t(dur)
    k = t / dur
    n = _ruido(t.size, 171)
    roce = (pasa_bajos(n, 8) * (1 - k) + pasa_bajos(n, 28) * k) * np.sin(np.pi * np.clip(k / 0.8, 0, 1)) * 1.5
    tt = np.clip(t - 0.65, 0, None)
    tope = np.sin(2 * np.pi * 90 * tt) * np.exp(-tt / 0.05) * (t >= 0.65)
    return roce + tope


def sfx_golpe_logo(dur=2.4):
    """Golpe grave y suave con un brillo de caja de música (aparición del logo)."""
    t = _t(dur)
    sub = np.sin(2 * np.pi * np.cumsum(38 + 50 * np.exp(-t / 0.1)) / SR) * np.exp(-t / 0.5)
    golpe = pasa_bajos(_ruido(t.size, 181), 16) * np.exp(-t / 0.05) * 2.0
    y = np.tanh((sub + golpe) * 1.5) / np.tanh(1.5)
    return y + caja_musica(96, dur, 0.6) * 0.8


SFX = {
    "agua": sfx_agua,
    "cepillado_suave": sfx_cepillado_suave,
    "vaso": sfx_vaso,
    "pasos": sfx_pasos,
    "papel": sfx_papel,
    "lata": sfx_lata,
    "monedas": sfx_monedas,
    "lapiz": sfx_lapiz,
    "notificacion": sfx_notificacion,
    "cajon": sfx_cajon,
    "golpe_logo": sfx_golpe_logo,
}


def obtener(nombre):
    propio = audio.buscar_archivo(audio.config.DIR_SFX, nombre, audio.EXT_AUDIO)
    if propio:
        return audio.cargar_audio(propio)
    x = SFX[nombre]()
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.7).astype(np.float32)


def mezcla(duracion, informar=print):
    """Música (propia o generada) + efectos del guion + golpe del logo. Devuelve un array mono."""
    n = int(round(duracion * SR))
    propia = audio.buscar_archivo(audio.config.DIR_MUSICA, "musica_espejo", audio.EXT_AUDIO)
    if propia:
        informar(f"  · Música propia: {propia.name}")
        m = audio.cargar_audio(propia)
        m = np.tile(m, int(np.ceil(n / m.size)))[:n]
        s0, s1 = GE.SILENCIO_ANTES_DEL_SALTO
        m[int(s0 * SR) : int(s1 * SR)] = 0
    else:
        informar("  · Música: caja de música + piano + fondo cálido (generada, sin derechos de terceros)")
        m = componer(duracion)
    m = m / (np.sqrt(np.mean(m[m != 0] ** 2)) + 1e-9) * 0.1
    fin_musica = int((GE.LOGO_INICIO - 0.1) * SR)
    m[fin_musica - int(0.6 * SR) : fin_musica] *= np.linspace(1, 0.15, int(0.6 * SR))
    m[fin_musica:] *= 0.15
    total = np.zeros(n + SR * 4, dtype=np.float32)
    total[:n] += m[:n]
    eventos = 0
    for e in GE.ESCENAS:
        for nombre, en, gan in e.get("sonidos", []):
            x = obtener(nombre)
            i0 = int((e["inicio"] + en) * SR)
            total[i0 : i0 + x.size] += x * gan * 0.32
            eventos += 1
    g = obtener("golpe_logo")
    i0 = int(GE.LOGO_INICIO * SR)
    total[i0 : i0 + g.size] += g * 0.55
    informar(f"  · {eventos} efectos de sonido + golpe con el logo a los {GE.LOGO_INICIO:.0f} s")
    total = total[:n]
    s0, s1 = GE.SILENCIO_ANTES_DEL_SALTO  # el segundo de silencio es total (también sin efectos)
    i0, i1 = int(s0 * SR), int(s1 * SR)
    total[i0 - int(0.15 * SR) : i0] *= np.linspace(1, 0, int(0.15 * SR))
    total[i0:i1] = 0
    cola = int(1.0 * SR)  # silencio digital al final, como en las referencias
    total[-cola:] *= np.linspace(1, 0, cola) ** 3
    return total
