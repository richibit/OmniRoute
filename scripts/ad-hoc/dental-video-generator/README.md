# 24 horas sin cepillarse — generador de video vertical

Programa en Python que produce, de principio a fin, el video de 48 s descrito en el documento de producción
«24 horas sin cepillarse»: 1080×1920 (9:16), 30 FPS, H.264 + AAC, listo para TikTok, Instagram Reels y YouTube Shorts.

Sin recursos externos el video sale completo: todas las imágenes (dientes, película salival, estreptococos, colonias,
cristales de azúcar, medidor de pH, lengua, dentista ilustrada, calendario, cepillo, reloj de 24 h), la música y los
efectos de sonido se generan por código. Cuando tengas grabaciones reales (dentista, voz, logotipo, música), cópialas
en `assets/` con el nombre indicado y sustituyen automáticamente a la versión generada.

## Viabilidad técnica (comprobada)

| Requisito | Cómo se cumple |
| --- | --- |
| 1080×1920, 9:16, 30 FPS, MP4 H.264/AAC | Escritor FFmpeg de MoviePy (`libx264`, `yuv420p`, CRF 17) + AAC 192 kbps; el programa verifica el archivo final |
| Duración del guion | 10 bloques contiguos de 0 a 48 s (`guion.py`), validados al arrancar |
| Escenas, imágenes, videos, cámara, transiciones | `escenas.py`: push-in, pull-back, tilt, órbita, dolly, match cut, whip pan, barrido vertical, fundido a negro, freeze frame, wipe con el cepillo, corte de bucle |
| Sincronía imagen–voz | Los cortes, textos y sonidos se anclan a palabras de la narración (`("cubriendo", 1)`), cuyo instante se mide en el audio real |
| Subtítulos sincronizados y resaltados | 2–3 palabras, palabra activa en menta, palabras clave en ámbar, contorno de 4 px, al 62,5 % de la altura; además se exportan SRT y VTT |
| Música y efectos con niveles controlados | Música a 105 BPM con pulso de reloj (21 compases = bucle limpio), 18 efectos, ducking automático bajo la voz, normalización a −14 LUFS |
| Narración grabada o texto a voz | Grabación completa o por bloques > edge-tts (en línea) > espeak-ng/MBROLA (offline) > silencio |
| Rigor científico | Rangos aproximados, texto legal (6–37 s), aviso «Ilustración esquemática · no a escala» en las vistas microscópicas, fuente Svatun y Attramadal (1978) en pantalla |

## B. Dependencias e instalación

Requisitos: **Python 3.9 o superior** y **FFmpeg** (MoviePy instala uno portátil mediante `imageio-ffmpeg`, pero se
recomienda el del sistema).

```bash
cd scripts/ad-hoc/dental-video-generator
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generar_video.py --descargar-fuentes --previsualizar 1   # descarga fuentes y prueba un fotograma
```

FFmpeg del sistema (opcional pero recomendado):

- Ubuntu/Debian: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: `winget install ffmpeg`

### Voz sin internet (espeak-ng + voz femenina MBROLA)

- Ubuntu/Debian: `sudo apt install espeak-ng mbrola mbrola-es3`
- macOS: `brew install espeak-ng` (MBROLA no está en Homebrew: se usa la voz `es-419`)
- Windows: instalador de <https://github.com/espeak-ng/espeak-ng/releases> (y añade `espeak-ng` al PATH)

La voz sintética sirve para revisar el montaje; para publicar, el documento pide la voz de la dentista grabada en
estudio (ver sección D).

## C. Estructura de carpetas

```
dental-video-generator/
├── generar_video.py      ← programa principal (ejecuta todo el flujo)
├── guion.py              ← textos, tiempos, sonidos, reloj, música y recursos (lo que más editarás)
├── config.py             ← formato, rutas, paleta, fuentes, marca, niveles de audio, motor de voz
├── escenas.py            ← las 10 escenas, transiciones y capas (textos, reloj, subtítulos)
├── graficos.py           ← dibujos procedurales (dientes, bacterias, pH, reloj, dentista…)
├── audio.py              ← voz, alineación por palabra, música, efectos y mezcla
├── subtitulos.py         ← agrupación, exportación SRT/VTT y render de subtítulos
├── utilidades.py         ← curvas de animación, tipografías, texto con emoji, composición
├── requirements.txt
├── assets/
│   ├── imagenes/         ← logo.png, velcro.jpg, fotos de la dentista (opcional)
│   ├── videos/           ← dentista_espejo.mp4, dentista_no.mp4, dentista_cierre.mp4… (opcional)
│   ├── audios/
│   │   ├── narracion/    ← narracion_completa.wav o bloque_01.wav … bloque_10.wav
│   │   ├── musica/       ← musica_fondo.mp3
│   │   └── sfx/          ← impacto.wav, whoosh.wav… (sustituyen a los generados)
│   └── fuentes/          ← Montserrat, Inter, JetBrains Mono (.ttf)
├── subtitulos/           ← subtitulos.srt, subtitulos.vtt, palabras.json (generados)
├── temp/                 ← caché de voz, mezcla.wav, tramos de video, previsualizaciones
└── export/               ← 24_horas_sin_cepillarse.mp4
```

Cada carpeta de `assets/` tiene un `LEEME.md` con los nombres exactos que reconoce el programa.

## D. Cómo agregar imágenes, videos y audios

No hay rutas inventadas que tengas que editar: el programa busca **nombres fijos** dentro de `assets/` y, si no
encuentra el archivo, usa la alternativa animada. Al arrancar imprime qué recursos encontró.

| Recurso | Dónde copiarlo | Si no existe |
| --- | --- | --- |
| Dentista frente al espejo (escena 1) | `assets/videos/dentista_espejo.mp4` (o `.jpg` en imagenes) | Dentista ilustrada que se cepilla y sonríe |
| Velcro real (escena 3, 0,5 s) | `assets/imagenes/velcro.jpg` o `assets/videos/velcro.mp4` | Textura de ganchos dibujada |
| Persona que despierta (escena 7) | `assets/videos/persona_despierta.mp4` | Ilustración cómica en la cama |
| Dentista responde «No» (escena 8) | `assets/videos/dentista_no.mp4` | Dentista ilustrada con boca sincronizada a la voz |
| Dentista cierre (escena 10) | `assets/videos/dentista_cierre.mp4` | Dentista ilustrada sonriente |
| Logotipo | `assets/imagenes/logo.png` | Círculo menta con un diente |
| Voz completa | `assets/audios/narracion/narracion_completa.wav` | Por bloques o texto a voz |
| Voz por bloques | `assets/audios/narracion/bloque_01.wav` … `bloque_10.wav` | Texto a voz para los que falten |
| Música | `assets/audios/musica/musica_fondo.mp3` | Pista electrónica generada a 105 BPM |
| Efecto concreto | `assets/audios/sfx/<nombre>.wav` | Efecto generado |

Consejos para la voz grabada: un archivo por bloque es lo más cómodo (cada bloque dura 3–6 s; si una toma es más
larga que su hueco, el programa la acelera hasta ×1,45 y avisa). Los subtítulos y las palabras ancla se recalculan a
partir de tu grabación, así que las imágenes siguen sincronizadas.

Nombre del consultorio y @usuario: edita `MARCA` en `config.py`.

### Cambiar textos, tiempos o escenas

- **Narración, textos en pantalla, sonidos:** `guion.py` → `BLOQUES`. Las referencias `("palabra", bloque, desplazamiento)`
  se mueven solas si cambia la voz.
- **Duración de un bloque:** cambia `inicio`/`fin` en `guion.py` (los bloques deben quedar contiguos) y
  `DURACION_TOTAL` en `config.py` si cambia el total.
- **Reloj:** `RELOJ_CLAVES` (instante, horas). **Música:** `MUSICA_SECCIONES` y `BPM`.
- **Colores, fuentes, subtítulos, niveles de audio:** `config.py`.
- **Aspecto de una escena:** el método `escena_*` correspondiente en `escenas.py`.

## E. Comando exacto para ejecutar

```bash
cd scripts/ad-hoc/dental-video-generator
python generar_video.py
```

Resultado: `export/24_horas_sin_cepillarse.mp4` (≈ 1,5 min con 4 núcleos).

Otras opciones útiles:

```bash
python generar_video.py --calidad borrador            # render rápido para revisar
python generar_video.py --previsualizar 1 12 33.5     # solo PNG en temp/previews/
python generar_video.py --desde 20 --hasta 31         # solo un tramo → export/parcial_020.0-031.0.mp4
python generar_video.py --motor-voz espeak            # forzar el motor de voz (auto|edge|espeak|silencio)
python generar_video.py --limpiar                     # borra temp/ y rehace la voz
python generar_video.py --hilos 2                     # menos procesos (equipos con poca RAM)
```

## F. Solución de errores comunes

| Mensaje / síntoma | Causa y solución |
| --- | --- |
| `Faltan dependencias de Python: moviepy…` | Activa el entorno virtual y ejecuta `pip install -r requirements.txt`. |
| `No se encontró FFmpeg` | Instala FFmpeg (sección B) o `pip install imageio-ffmpeg`. |
| Voz: `silencio` (video sin narración) | No hay internet para edge-tts ni espeak-ng instalado. Instala espeak-ng o copia tu grabación en `assets/audios/narracion/`. El video se genera igual, con subtítulos. |
| `edge-tts no está disponible (¿sin internet?)` | Usaste `--motor-voz edge` sin conexión o detrás de un proxy que la bloquea. Usa `auto` o `espeak`. |
| `! Bloque N: la voz dura X s y el hueco es Y s` | El texto de ese bloque es demasiado largo para su tiempo. Acórtalo o amplía el bloque en `guion.py`. |
| `guion.py usa la palabra ancla «…»` | Cambiaste la narración y una referencia `("palabra", bloque)` ya no existe. Actualízala. |
| `Hay un hueco o solape entre los bloques…` | Los `fin`/`inicio` de bloques consecutivos deben coincidir. |
| `Sonido desconocido «…»` | Usa un nombre de la lista de `assets/audios/sfx/LEEME.md` o añade ese archivo. |
| Letras con otra tipografía | Faltan las fuentes: `python generar_video.py --descargar-fuentes` o copia los .ttf en `assets/fuentes/`. |
| Emoji como flecha «↓» | No hay fuente de emoji a color. En Linux: `sudo apt install fonts-noto-color-emoji`. |
| `No se pudo abrir el video …` | Archivo dañado o códec raro: conviértelo con `ffmpeg -i entrada.mov -c:v libx264 -pix_fmt yuv420p salida.mp4`. |
| Se queda sin memoria o va muy lento | `--hilos 2` o `--calidad borrador`. |
| La voz cambió pero no se nota | La voz se guarda en caché en `temp/`; usa `--limpiar`. |
| `El archivo final no cumple el formato…` | Algún paso de FFmpeg falló a medias: borra `temp/` y repite; revisa que haya espacio en disco. |

## Antes de publicar (puntos de control del documento)

- Validar el guion con la dentista que narra y sustituir la voz sintética por su grabación.
- Revisar la animación con un odontólogo (forma de bacterias, escala, pH).
- Comprobar la legibilidad de los subtítulos en un teléfono real con la interfaz de cada app.
- Confirmar la licencia de la música si usas una propia (la generada no tiene derechos de terceros).
