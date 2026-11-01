# Generador de videos verticales de salud dental

Este proyecto produce dos videos verticales (1080×1920, 9:16, 30 FPS, H.264 + AAC) listos para
TikTok, Instagram Reels y YouTube Shorts, cada uno con la fórmula de sus videos de referencia:

| Video | Programa | Fórmula |
| --- | --- | --- |
| **La última vez que tu mamá te cepilló los dientes** (≈75 s) | `python generar_espejo.py` | Historia contada por el espejo del baño: personajes 3D en perspectiva, contador, salto de 25 años, sin voz |
| **24 horas sin cepillarse** (≈66 s) | `python generar_video.py` | Explicador con voz, subtítulos karaoke, tarjetas de datos y llamada a compartir |

## Video 2 · «La última vez que tu mamá te cepilló los dientes»

Sigue el PDF «Guion de video: salud dental» escena por escena (tiempos, contador, subtítulos,
tarjetas y cierre) con la fórmula de las referencias: cámara fija (el narrador es el espejo), contador
a la izquierda a media altura, subtítulos cortos en itálica minúscula, salto de tiempo, giro final y
cierre oscuro con un dato. **Sin voz en off y sin marca de agua durante el video**: el logo y el
@usuario solo aparecen al final, al centro.

| Elemento de la fórmula (medido en las referencias) | Cómo se aplica |
| --- | --- |
| Cámara fija con perspectiva (el espejo mira al baño), mundo 3D «de juguete», luz cálida de noche y fría de mañana | `juguete3d.py`: cámara estenopeica con eje horizontal (lo que se acerca al espejo crece y baja), elipsoides orientables, cápsulas, cajas con texturas (tarjetas, nota, cajita), cilindros y cromo; z-buffer, luz envolvente, contorno, especular y oclusión ambiental |
| Acciones creíbles: lo que se toca se toca de verdad | La encimera y el lavabo se calculan rayo a rayo (`escenario.py`) y tapan lo que queda detrás; brazos de largo real (`personajes.py`, cinemática inversa de dos huesos) con manos de palma, dedos y pulgar; cada objeto (`utileria.py`) se coloca por su punto natural (cabeza del cepillo, punta del lápiz, esquina de la tarjeta) y la mano se calcula a partir de él, así que van siempre juntos; el programa avisa si una mano no alcanza su objetivo |
| Contador a la izquierda: rótulo en versalitas, número grande en serif, línea de edad, lo especial en dorado | `VECES QUE TE CEPILLÓ LOS DIENTES · 4,626 · TIENES 7 AÑOS · LA ÚLTIMA` |
| Título serif crema con la palabra clave en dorado + etiqueta «contado por…» (primeros 3 s) | «La última vez que tu mamá / te **cepilló** los dientes» · CONTADO POR EL ESPEJO DEL BAÑO |
| Subtítulos en itálica minúscula, una línea, ~4 s con medio segundo sin texto, fundido de 0,3 s | Igual, centrados un poco abajo del centro |
| Salto de tiempo con texto grande a la izquierda | «25 AÑOS / DESPUÉS» |
| Cierre: último cuadro desenfocado y oscuro + título + dato con fuente + pregunta en dorado; logo al final | Igual (SEOP y AEP) y logo de Instagram con el @ de `config.py` |
| Sonido: sin voz, música suave, efectos, golpe con el logo y silencio final, −14 LUFS | Caja de música + piano + fondo cálido (`musica.py`), 1 s de silencio antes del salto y la música sube ahí |
| Nada atraviesa nada | `revision_choques.py` revisa cada fotograma con distancias con signo: brazos contra cuellos, cabezas y cuerpos ajenos, manos dentro del propio cuerpo, objetos contra personas y contra otros objetos, la encimera (fuera del lavabo) y la pared del espejo. Además cada codo se coloca solo (`personajes.Personaje._mejor_polo`): prueba girar el codo alrededor del eje hombro→muñeca y se queda con la postura que no atraviesa a la otra persona, la encimera, los objetos ni el propio cuerpo |
| Movimientos continuos | `Renderizador.transicion()` pasa de un agarre a otro (palma, yemas, pinza, agarre de mango) sin saltos, con un arco opcional para que la mano suba antes de avanzar |
| Grifo realista | Monomando detrás del lavabo con la palanca encima apuntando a quien lo usa; el caño gira un poco a un lado para que el chorro caiga dentro del lavabo y se vea desde el espejo |

**Revisar choques** (la salida enumera escena, profundidad en px, tramo y las dos piezas; termina con 0
cuando no hay ninguno):

```bash
python revision_choques.py                         # todo el video, cada 2 fotogramas
python revision_choques.py --desde 44 --hasta 48 --paso 1
```

**Ejecutar:**

```bash
cd scripts/ad-hoc/dental-video-generator
python generar_espejo.py --descargar-fuentes   # la primera vez (Newsreader, Caveat…)
python generar_espejo.py                       # → export/la_ultima_vez_que_tu_mama.mp4 (≈25 min con 4 núcleos)
python generar_espejo.py --previsualizar 1 21 63   # PNG en temp/previews_espejo/
python generar_espejo.py --desde 48 --hasta 58     # solo un tramo
```

**Qué editar:**

- Textos, tiempos, contador, subtítulos, efectos, tarjetas, nombres y cierre: `guion_espejo.py`
  (los nombres «Mateo» y «Ana» de la cajita y de la tarjeta son provisionales; si el protagonista es
  niña, cambia «yo solito» por «yo solita» y «el que cuenta» por «la que cuenta»).
- Coreografía de cada escena (posiciones, gestos, expresiones): métodos `escena_N` de `escenas_espejo.py`
  (trayectorias suaves con `curva()`, pasos con los pies apoyados con `caminata()`, cepillado con
  `cepillo_en_boca()`, dedos con `contar()`, tarjetas al marco con `meter_en_marco()`).
- Baño, encimera, lavabo, grifo y cámara: `escenario.py`; objetos (cepillos, vaso, pasta, cajita, tarjetas,
  nota, lápiz, monedas, celular): `utileria.py`.
- Aspecto de los personajes (colores de ropa, pelo, proporciones): `personajes.py` y `mama()`/`hijo()` en `escenas_espejo.py`.
- Música propia: `assets/audios/musica/musica_espejo.mp3` (el segundo de silencio antes del salto se respeta).
- Efectos propios: `assets/audios/sfx/<nombre>.wav` con los nombres de `musica.SFX` (agua, cepillado_suave,
  vaso, pasos, papel, lata, monedas, lapiz, notificacion, cajon, golpe_logo).

---

## Video 1 · «24 horas sin cepillarse»

Programa en Python que produce, de principio a fin, el video vertical «24 horas sin cepillarse»
(1080×1920, 9:16, 30 FPS, H.264 + AAC), listo para TikTok, Instagram Reels y YouTube Shorts.
Los datos y escenas salen del documento de producción (y su anexo de verificación científica); el
estilo, el ritmo, el sonido y la estructura copian la fórmula de cuatro explicadores verticales que
funcionaron bien.

Sin recursos externos el video sale completo: todas las imágenes y efectos se generan por código y
la narración se sintetiza con una voz neuronal offline (Kokoro). Se exportan dos archivos: con voz
y sin voz (mismo video y mismos efectos, para montar después la voz de la dentista).

## La fórmula (medida en los videos de referencia)

| Elemento | Qué hacen los videos de referencia | Cómo lo aplica este proyecto |
| --- | --- | --- |
| Duración y ritmo | 55–72 s; voz continua a ≈2,8–3 palabras/s | La duración sale de la narración (≈66 s); Kokoro a ×1,1 |
| Gancho | Primera frase dirigida a «tú» con imagen literal | «Te acabas de cepillar… y tus dientes ya se están cubriendo otra vez.» + el cepillo sale de cuadro |
| Negación | «No es por tu cara», «no es un sueño», «no es tu imaginación» | «Y no es suciedad: es tu propia saliva.» |
| Mecanismo | Una idea por frase, corte de plano en cada frase, cámara en movimiento | 11 bloques = 11 planos con *punch-in* en cada corte, zooms y barridos |
| Datos | Tarjetas con número grande, barras comparativas, etiquetas con línea guía, fuente en letra pequeña | «60–90 %», barras de pH sin/con placa, «S. mitis / S. oralis», «Svatun y Attramadal, 1978» |
| Prueba / giro | Escáner, laboratorio, anécdota histórica | ¿Caries? No. ¿Gingivitis? Tampoco + experimento de Löe (1965) |
| Reencuadre | Frase final que replantea el gancho | «Un día no te arruina la boca. La costumbre, sí.» |
| Llamada a la acción | «Mándaselo a quien…» con destinatario concreto | «Y mándaselo a quien se duerme sin cepillarse.» + teléfono con «Enviar a» |
| Subtítulos | Arriba (≈18 % de la altura), palabra a palabra: la actual en blanco y negrita, las dichas en gris, una palabra clave por frase en recuadro azul de selección | Igual, sincronizados al milisegundo con la duración real de cada fonema |
| Marca | Icono de Instagram + @usuario a la derecha bajo los subtítulos | Igual (con el @ de `config.py`) |
| Estética | Fondo oscuro con foco de luz, partículas, viñeta, brillos | Igual, con la paleta del documento (menta, ámbar, coral) |
| Sonido | Voz sola (sin música bajo la voz, piso entre palabras ≈ −52 dB), −14 LUFS | Igual: voz −10,5 dB p90, piso −52,6 dB, −14,5 LUFS |
| Cierre | Última palabra → 1,2 s de silencio → golpe grave fuerte con el logo → silencio; el último subtítulo queda en pantalla | Igual, con el logo de Instagram sobre el mismo diente del inicio (bucle) |

## Viabilidad técnica (comprobada)

| Requisito | Cómo se cumple |
| --- | --- |
| 1080×1920, 9:16, 30 FPS, MP4 H.264/AAC | Escritor FFmpeg de MoviePy (`libx264`, `yuv420p`, CRF 17) + AAC 192 kbps; el programa verifica cada archivo |
| Combinar escenas, imágenes y videos | 11 escenas animadas; cualquier bloque se puede sustituir por un video o una foto propios (`bloque_XX`) |
| Animaciones, cámara, transiciones | *Punch-in* en los cortes, push-in, pull-back, inclinación, órbita, zoom a través de la superficie, barrido con desenfoque, congelado con destello, fundido a negro |
| Sincronía imagen–voz | Etiquetas, tarjetas, cortes y sonidos se anclan a palabras de la narración (`("velcro", 0.0)`) |
| Subtítulos y palabras resaltadas | Karaoke arriba + recuadro azul en la palabra clave; también SRT y VTT |
| Niveles de audio | Voz normalizada por bloque, efectos a −20 dB, golpe final, ganancia común a −14 LUFS con limitador |
| Rigor científico | Rangos aproximados, «Tiempos aproximados: varían entre personas», «Ilustración esquemática · no a escala», fuentes en pantalla |

## B. Dependencias e instalación

Requisitos: **Python 3.9 o superior** y **FFmpeg** (MoviePy instala uno portátil mediante `imageio-ffmpeg`,
pero se recomienda el del sistema).

```bash
cd scripts/ad-hoc/dental-video-generator
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generar_video.py --descargar-fuentes --descargar-voz --previsualizar 1
```

La última línea descarga las fuentes (≈2 MB) y la voz Kokoro (≈350 MB, una sola vez) y guarda un
fotograma de prueba en `temp/previews/`.

FFmpeg del sistema: `sudo apt install ffmpeg` (Ubuntu/Debian), `brew install ffmpeg` (macOS),
`winget install ffmpeg` (Windows).

### Voz

- **Kokoro (recomendada, offline):** `ef_dora` es femenina (la dentista del documento); `em_alex` es
  masculina, con un tono parecido al de los narradores de referencia. Cámbiala con `--voz em_alex` o en
  `config.VOZ_KOKORO`. La pronunciación es latinoamericana (`IDIOMA_KOKORO = "es-419"`).
- **Grabación propia (para publicar):** `assets/audios/narracion/bloque_01.wav` … `bloque_11.wav`.
- Alternativas automáticas si no hay Kokoro: edge-tts (requiere internet), espeak-ng (robótica) o
  silencio (el video sale igual con subtítulos).

## C. Estructura de carpetas

```
dental-video-generator/
├── generar_espejo.py     ← video 2: «La última vez que tu mamá…» (programa principal)
├── guion_espejo.py       ← video 2: escenas, contador, subtítulos, tarjetas y cierre
├── escenas_espejo.py     ← video 2: coreografía de las 18 escenas y textos en pantalla
├── escenario.py          ← video 2: cámara, baño, encimera y lavabo (con oclusión), grifo y agua
├── utileria.py           ← video 2: cepillos, vaso, pasta, cajita, tarjetas, nota, lápiz, monedas…
├── juguete3d.py          ← motor 3D mínimo con perspectiva (personajes y objetos tipo juguete)
├── personajes.py         ← mamá, hijo y nieta (brazos y piernas con cinemática inversa, manos, caras)
├── musica.py             ← video 2: caja de música, piano y efectos
├── generar_video.py      ← video 1: «24 horas sin cepillarse» (programa principal)
├── guion.py              ← narración, palabras resaltadas, etiquetas, sonidos y reloj (lo que más editarás)
├── config.py             ← formato, rutas, paleta, fuentes, marca, voz y niveles de audio
├── escenas.py            ← las 11 escenas, el cierre y las capas (subtítulos, marca de agua, reloj)
├── graficos.py           ← dibujos procedurales y elementos de la fórmula (tarjetas, etiquetas, teléfono…)
├── audio.py              ← voz Kokoro con tiempos por palabra, línea de tiempo, efectos y mezcla
├── subtitulos.py         ← subtítulos karaoke y exportación SRT/VTT
├── utilidades.py         ← curvas de animación, tipografías, texto con emoji, composición
├── requirements.txt
├── assets/
│   ├── imagenes/         ← bloque_XX.jpg (sustituye un plano), logo.png
│   ├── videos/           ← bloque_XX.mp4 (sustituye un plano)
│   ├── audios/
│   │   ├── narracion/    ← bloque_01.wav … bloque_11.wav (tu voz)
│   │   ├── musica/       ← musica_fondo.mp3 (opcional, desactivada en la fórmula)
│   │   └── sfx/          ← golpe.wav, whoosh.wav… (sustituyen a los generados)
│   ├── fuentes/          ← Montserrat, Inter, JetBrains Mono, Oswald (.ttf)
│   └── modelos_voz/      ← modelo Kokoro (se descarga, no se sube a git)
├── subtitulos/           ← subtitulos.srt, subtitulos.vtt, palabras.json (generados)
├── temp/                 ← caché de voz, mezclas, plan.json, tramos de video, previsualizaciones
└── export/               ← la_ultima_vez_que_tu_mama.mp4, 24_horas_sin_cepillarse.mp4 (+ _sin_voz)
```

Cada carpeta de `assets/` tiene un `LEEME.md` con los nombres exactos que reconoce el programa.

## D. Cómo agregar imágenes, videos y audios

No hay rutas que editar: el programa busca **nombres fijos** dentro de `assets/`; si no encuentra el
archivo, usa la versión generada.

| Recurso | Dónde copiarlo | Efecto |
| --- | --- | --- |
| Tu voz | `assets/audios/narracion/bloque_01.wav` … `bloque_11.wav` | Sustituye la voz sintética; **todo el video se resincroniza** con la duración de tus tomas |
| Un plano propio (video) | `assets/videos/bloque_XX.mp4` | Sustituye el plano animado del bloque XX (sin su audio, en bucle si es corto) |
| Un plano propio (foto) | `assets/imagenes/bloque_XX.jpg` | Igual, con zoom lento |
| Logotipo | `assets/imagenes/logo.png` | Aparece en el cierre bajo el @usuario |
| Efecto de sonido | `assets/audios/sfx/<nombre>.wav` | Sustituye el efecto generado |
| Música (opcional) | `assets/audios/musica/musica_fondo.mp3` + `MUSICA_DE_FONDO = True` | Música a −24 dB bajo la voz |

Los textos de cada bloque (y su número) están en `guion.py`. El @ y el nombre del consultorio se
cambian en `MARCA` (`config.py`).

### Cambiar textos, palabras resaltadas o tiempos

- **Narración, palabra resaltada, etiquetas y sonidos:** `guion.py` → `BLOQUES`. Las referencias
  `("palabra", desplazamiento)` siguen a la voz aunque cambie el texto o la grabación.
- **Duración:** no se fija a mano; sale de la narración. Para acortar el video, acorta el texto o sube
  `VELOCIDAD_KOKORO` (`config.py`); las pausas están en `PAUSA_ENTRE_BLOQUES` y `CIERRE_*`.
- **Reloj de 24 h:** `RELOJ_CLAVES` (palabra en la que cambia, horas).
- **Colores, fuentes, subtítulos, marca de agua y niveles de audio:** `config.py`.
- **Aspecto de una escena:** el método `escena_*` correspondiente en `escenas.py`.

## E. Comando exacto para ejecutar

```bash
cd scripts/ad-hoc/dental-video-generator
python generar_video.py
```

Resultado: `export/24_horas_sin_cepillarse.mp4` y `export/24_horas_sin_cepillarse_sin_voz.mp4`
(≈3 min con 4 núcleos).

Otras opciones útiles:

```bash
python generar_video.py --calidad borrador            # render rápido para revisar
python generar_video.py --previsualizar 1 12 33.5     # solo PNG en temp/previews/
python generar_video.py --desde 20 --hasta 31         # solo un tramo → export/parcial_020.0-031.0.mp4
python generar_video.py --voz em_alex                 # voz masculina
python generar_video.py --motor-voz espeak            # forzar motor (auto|kokoro|edge|espeak|silencio)
python generar_video.py --solo-con-voz                # no exportar la versión sin voz
python generar_video.py --limpiar                     # borra temp/ y rehace la voz
python generar_video.py --hilos 2                     # menos procesos (equipos con poca RAM)
```

## F. Solución de errores comunes

| Mensaje / síntoma | Causa y solución |
| --- | --- |
| `Faltan dependencias de Python: …` | Activa el entorno virtual y ejecuta `pip install -r requirements.txt`. |
| `No se encontró FFmpeg` | Instala FFmpeg (sección B) o `pip install imageio-ffmpeg`. |
| `Kokoro no está instalado` / voz robótica | Ejecuta `pip install kokoro-onnx onnx` y `python generar_video.py --descargar-voz`. |
| `No se pudo descargar …kokoro…` | Sin acceso a GitHub: descarga `kokoro-v1.0.onnx` y `voices-v1.0.bin` desde otra red y cópialos en `assets/modelos_voz/`. |
| `Sin el paquete «onnx» los tiempos por palabra se estiman` | Funciona, pero la sincronía de subtítulos es aproximada: `pip install onnx` y vuelve a ejecutar. |
| `La voz Kokoro «…» no existe` | Voces en español: `ef_dora`, `em_alex`, `em_santa`. |
| `guion.py usa la palabra ancla «…»` | Cambiaste la narración y una referencia ya no existe: actualízala en `guion.py`. |
| `El bloque N resalta «…», que no está en su narración` | La palabra de `resaltar` debe aparecer en el texto del bloque. |
| `Sonido desconocido «…»` | Usa un nombre de `assets/audios/sfx/LEEME.md` o añade ese archivo. |
| `! El video supera 75 s` | Acorta el guion o sube `VELOCIDAD_KOKORO`. |
| Letras con otra tipografía | `python generar_video.py --descargar-fuentes` o copia los .ttf en `assets/fuentes/`. |
| Emoji como símbolo | Instala una fuente de emoji a color (`sudo apt install fonts-noto-color-emoji`). |
| `No se pudo abrir …` (video propio) | Conviértelo: `ffmpeg -i entrada.mov -c:v libx264 -pix_fmt yuv420p salida.mp4`. |
| La voz cambió pero no se nota | La voz se guarda en caché en `temp/`: usa `--limpiar`. |
| Lento o sin memoria | `--hilos 2` o `--calidad borrador`. |

## Antes de publicar (puntos de control del documento)

- Validar el guion con la dentista y sustituir la voz sintética por su grabación (`bloque_XX.wav`).
- Revisar la animación con un odontólogo (forma de bacterias, escala, pH) y la mención del
  experimento de Löe (1965).
- Comprobar la legibilidad de los subtítulos en un teléfono real con la interfaz de cada app.
- Fijar en los comentarios el dato del pH con su fuente, como propone el documento.
