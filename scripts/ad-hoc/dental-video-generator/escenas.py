"""
Composición de fotogramas con la fórmula de los explicadores verticales:
un plano por frase (corte en cada bloque), cámara siempre en movimiento,
etiquetas y tarjetas que aparecen en la palabra exacta, subtítulos karaoke
arriba, marca de agua, reloj de 24 h y cierre con el logo.

Cada fotograma es una función pura del tiempo `t`, así que el video se puede
renderizar en paralelo por tramos sin que cambie el resultado.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

import config
import graficos as G
import guion
import subtitulos
from utilidades import (
    ErrorProduccion,
    aleatorio,
    con_alfa,
    desenfoque_movimiento,
    disco_suave,
    entrada_cubica,
    lerp,
    limitar,
    mezclar_color,
    normalizar_palabra,
    pegar,
    progreso,
    render_lineas,
    salida_cubica,
    salida_rebote,
    suave,
)

W, H, FPS = config.ANCHO, config.ALTO, config.FPS
P = config.PALETA


# ---------------------------------------------------------------------------
# Tiempos
# ---------------------------------------------------------------------------
def buscar_palabra(palabras, bloque, palabra):
    objetivo = normalizar_palabra(palabra)
    lista = palabras.get(bloque) or palabras.get(str(bloque)) or []
    for p in lista:
        if normalizar_palabra(p["texto"]) == objetivo:
            return p["inicio"]
    for p in lista:
        if normalizar_palabra(p["texto"]).startswith(objetivo):
            return p["inicio"]
    raise ErrorProduccion(
        f"guion.py usa la palabra ancla «{palabra}» en el bloque {bloque}, pero no está en su narración. Corrígela."
    )


def resolver_tiempo(ref, palabras, tramos, bloque=None):
    """
    Convierte una referencia del guion en segundos absolutos:
    número (relativo al inicio de la voz del bloque), ("palabra", desplazamiento)
    o ("palabra", bloque, desplazamiento).
    """
    if ref is None:
        return None
    if isinstance(ref, (int, float)):
        return tramos[bloque]["inicio_voz"] + float(ref)
    if len(ref) == 2:
        palabra, desplazamiento = ref
    else:
        palabra, bloque, desplazamiento = ref
    return buscar_palabra(palabras, bloque, palabra) + desplazamiento


def vista_zoom(sprite, escala, foco_sprite, foco_pantalla=(W / 2, H / 2), tam=(W, H)):
    """Encuadre de cámara: muestra `sprite` ampliado `escala` veces con foco_sprite en foco_pantalla."""
    a = 1.0 / escala
    return sprite.transform(
        tam,
        Image.AFFINE,
        (a, 0, foco_sprite[0] - foco_pantalla[0] * a, 0, a, foco_sprite[1] - foco_pantalla[1] * a),
        resample=Image.BILINEAR,
    )


def aparicion(t, t0, dur=0.28):
    """(alfa, escala) de un elemento que aparece con rebote en t0."""
    k = progreso(t, t0, t0 + dur)
    return limitar(k * 2.2), 0.6 + 0.4 * salida_rebote(k)


# ---------------------------------------------------------------------------
# Recursos del usuario (imágenes o videos que sustituyen a un plano animado)
# ---------------------------------------------------------------------------
EXTENSIONES_VIDEO = [".mp4", ".mov", ".m4v", ".webm", ".mkv"]
EXTENSIONES_IMAGEN = [".png", ".jpg", ".jpeg", ".webp"]


def buscar_recurso(nombre, solo_imagen=False):
    """Busca assets/videos/<nombre>.<ext> o assets/imagenes/<nombre>.<ext>."""
    carpetas = [(config.DIR_IMAGENES, EXTENSIONES_IMAGEN)]
    if not solo_imagen:
        carpetas.insert(0, (config.DIR_VIDEOS, EXTENSIONES_VIDEO))
    for carpeta, extensiones in carpetas:
        for ext in extensiones:
            ruta = carpeta / f"{nombre}{ext}"
            if ruta.exists():
                return ruta
    return None


class Medio:
    """Imagen o video del usuario ajustado a 9:16 (recorte centrado tipo «cover»)."""

    def __init__(self, ruta):
        self.ruta = ruta
        self.es_video = ruta.suffix.lower() in EXTENSIONES_VIDEO
        try:
            if self.es_video:
                from moviepy import VideoFileClip

                self.clip = VideoFileClip(str(ruta), audio=False)
                self.duracion = max(0.1, self.clip.duration)
            else:
                self.imagen = self._cubrir(Image.open(ruta).convert("RGB"), 1.12)
        except Exception as exc:
            raise ErrorProduccion(f"No se pudo abrir {ruta.name}: {exc}") from exc

    @staticmethod
    def _cubrir(img, margen=1.0):
        escala = max(W / img.width, H / img.height) * margen
        img = img.resize((int(img.width * escala + 0.5), int(img.height * escala + 0.5)), Image.LANCZOS)
        x0, y0 = (img.width - int(W * margen)) // 2, (img.height - int(H * margen)) // 2
        return img.crop((x0, y0, x0 + int(W * margen), y0 + int(H * margen)))

    def fotograma(self, t_local, duracion_uso):
        if self.es_video:  # se usa sin su audio y se repite si es más corto que el bloque
            return self._cubrir(Image.fromarray(self.clip.get_frame(t_local % self.duracion))).convert("RGBA")
        z = 1.0 + 0.08 * limitar(t_local / max(duracion_uso, 0.1))  # imagen fija con zoom lento
        foco = (self.imagen.width / 2, self.imagen.height / 2)
        return vista_zoom(self.imagen, z * W / self.imagen.width * 1.12, foco).convert("RGBA")


# ---------------------------------------------------------------------------
# Renderizador
# ---------------------------------------------------------------------------
class Renderizador:
    def __init__(self, plan):
        self.palabras = {int(k): v for k, v in plan["palabras"].items()}
        self.tramos = {tr["id"]: tr for tr in plan["tramos"]}
        self.orden = [tr["id"] for tr in plan["tramos"]]
        self.paginas = plan["paginas"]
        self.cierre = plan["cierre"]
        self.duracion = plan["duracion"]
        self.bloques = {b["id"]: b for b in guion.BLOQUES}
        R = self.T
        self.reloj = [(R(ref), horas) for ref, horas in guion.RELOJ_CLAVES]
        self.reloj_aparece = R(guion.RELOJ_APARECE)
        self.reloj_noche = tuple(R(x) for x in guion.RELOJ_NOCHE)
        self.reloj_oculto = [(R(a), R(b)) for a, b in guion.RELOJ_OCULTO]
        self.legales = [dict(x, desde=R(x["desde"]), hasta=R(x["hasta"])) for x in guion.TEXTOS_LEGALES]
        self.escenas = {
            "gancho": self.escena_gancho,
            "negacion": self.escena_negacion,
            "pelicula": self.escena_pelicula,
            "pioneras": self.escena_pioneras,
            "colonias": self.escena_colonias,
            "dato_clave": self.escena_dato_clave,
            "noche": self.escena_noche,
            "revelacion": self.escena_revelacion,
            "historia": self.escena_historia,
            "costumbre": self.escena_costumbre,
            "compartir": self.escena_compartir,
        }
        for b in guion.BLOQUES:
            if b["escena"] not in self.escenas:
                raise ErrorProduccion(f"El bloque {b['id']} usa la escena «{b['escena']}», que no existe en escenas.py.")
        # Imágenes o videos propios: assets/videos/bloque_XX.mp4 o assets/imagenes/bloque_XX.jpg
        # sustituyen el plano animado de ese bloque (subtítulos, marca de agua y reloj siguen encima).
        self.medios = {}
        for bid in self.orden:
            ruta = buscar_recurso(f"bloque_{bid:02d}")
            if ruta:
                self.medios[bid] = Medio(ruta)
        ruta_logo = buscar_recurso("logo", solo_imagen=True)
        self.logo = None
        if ruta_logo:
            logo = Image.open(ruta_logo).convert("RGBA")
            self.logo = logo.resize((max(1, int(logo.width * 170 / logo.height)), 170), Image.LANCZOS)
        self._preparar()

    # ----------------------------------------------------------------- tiempo
    def T(self, ref, bloque=None):
        return resolver_tiempo(ref, self.palabras, self.tramos, bloque)

    def tp(self, palabra, b, desplazamiento=0.0):
        """Instante en que se dice `palabra` en el bloque b."""
        return buscar_palabra(self.palabras, b["id"], palabra) + desplazamiento

    def textos(self, b):
        return b.get("textos", {})

    # -------------------------------------------------------------- fotograma
    def fotograma(self, t):
        """Devuelve el fotograma RGB (numpy H×W×3) del instante t."""
        lienzo = self.escena(t)
        lienzo.alpha_composite(G.vineta())
        en_cierre = t >= self.cierre["inicio"]
        self.capa_legal(lienzo, t)
        if not en_cierre:
            self.capa_reloj(lienzo, t)
            self.capa_marca_agua(lienzo, t)
        atenuar = 1.0 - 0.5 * progreso(t, self.cierre["golpe"] - 0.15, self.cierre["golpe"] + 0.35)
        subtitulos.dibujar(lienzo, self.paginas, t, atenuar)
        return np.asarray(lienzo.convert("RGB"))

    def tramo_en(self, t):
        for bid in self.orden:
            tr = self.tramos[bid]
            if tr["inicio"] <= t < tr["fin"]:
                return self.bloques[bid], tr
        return self.bloques[self.orden[-1]], self.tramos[self.orden[-1]]

    def escena(self, t):
        if t >= self.cierre["inicio"]:
            return self.escena_cierre(t)
        b, tr = self.tramo_en(t)
        if b["id"] in self.medios:
            img = self.medios[b["id"]].fotograma(t - tr["inicio"], tr["fin"] - tr["inicio"])
        else:
            img = self.escenas[b["escena"]](t, b, tr)
        # «Punch-in» en cada corte: el plano entra un poco ampliado y se asienta (energía de la fórmula).
        if b["id"] != self.orden[0]:
            k = progreso(t, tr["inicio"], tr["inicio"] + 0.22)
            if k < 1:
                img = vista_zoom(img, 1.0 + 0.07 * (1 - salida_cubica(k)), (W / 2, H / 2))
        return img

    # ------------------------------------------------------------------ capas
    def capa_marca_agua(self, lienzo, t):
        m = config.MARCA_AGUA
        # el icono se ve a color un instante al principio y luego queda en blanco (como en la fórmula)
        a_color = 0.9 <= t < 2.6
        pegar(lienzo, G.icono_instagram(m["tam_icono"], a_color), m["x_icono"], m["y_icono"], alfa=0.95)
        usuario = sprite_usuario(config.MARCA["usuario"].upper(), 26)
        pegar(lienzo, usuario, m["x_derecha"] - usuario.width / 2, m["y_usuario"], alfa=0.9)

    def capa_reloj(self, lienzo, t):
        if t < self.reloj_aparece or any(a <= t < b for a, b in self.reloj_oculto):
            return
        # Último fotograma clave alcanzado; durante RELOJ_TRANSICION se interpola desde el valor anterior.
        horas = self.reloj[0][1]
        for i, (tk, hk) in enumerate(self.reloj):
            if t < tk:
                break
            previo = self.reloj[i - 1][1] if i else hk
            k = progreso(t, tk, tk + guion.RELOJ_TRANSICION)
            horas = lerp(previo, hk, k * k * (3 - 2 * k))
        noche = self.reloj_noche[0] <= t < self.reloj_noche[1]
        t24 = next((tk for tk, hk in self.reloj if hk >= 24), self.duracion)
        brillo = int(round(4 * (1 - progreso(t, t24 + 0.8, t24 + 2.2)))) if t >= t24 + guion.RELOJ_TRANSICION else 0
        alfa, escala = aparicion(t, self.reloj_aparece, 0.35)
        spr = G.pildora_reloj(int(round(horas * 60)), noche, brillo)
        pegar(lienzo, spr, 30 + spr.width / 2 * escala, config.MARCA_AGUA["y_icono"], escala=escala, alfa=alfa)

    def capa_legal(self, lienzo, t):
        y = 1505
        for x in self.legales:
            if x["desde"] <= t < x["hasta"]:
                spr = sprite_legal(x["texto"])
                pegar(lienzo, spr, 50 + spr.width / 2, y, alfa=0.85)
                y -= 38

    # -------------------------------------------------------- piezas comunes
    def fondo(self, t, c_centro, c_borde, color_polvo=(190, 215, 255), semilla=1, intensidad=1.0, cy=980):
        img = G.fondo_foco(c_centro, c_borde, 540, cy).copy()
        G.polvo(img, t, color_polvo, 50, semilla, intensidad)
        return img

    def diente_macro(self, img, t, escala, centro=(540, 1080), placa=0.0, pelicula=None, brillos=True, tam=(600, 840)):
        """Incisivo en primer plano con halo de luz, película o placa y destellos."""
        sprite, _ = G.diente(*tam)
        capa = sprite.copy()
        if pelicula:
            capa.alpha_composite(G.pelicula(tam[0], tam[1], pelicula))
        if placa > 0:
            capa.alpha_composite(con_alfa(G.placa(*tam), placa))
        if brillos:
            for k, (bx, by) in enumerate(((0.33, 0.42), (0.62, 0.55), (0.40, 0.78))):
                fase = (t * 0.9 + k * 0.37) % 1.0
                pegar(capa, G.destello(70), bx * tam[0], by * tam[1], alfa=math.sin(fase * math.pi) ** 3)
        halo = G.resplandor(("diente", tam), sprite, 40, (90, 200, 230), 0.9)
        pegar(img, halo, centro[0], centro[1], escala=escala, alfa=0.55)
        pegar(img, capa, centro[0], centro[1], escala=escala)

    # ---------------------------------------------------------- preparación
    def _preparar(self):
        rng = aleatorio(301)
        self.proteinas = []
        for i in range(60):
            frac = 0.05 + 0.7 * (i / 60) + rng.uniform(-0.02, 0.02)
            self.proteinas.append((rng.uniform(160, 1240), rng.uniform(700, 1560), frac, int(rng.integers(0, 3))))
        self.enlaces = []
        for i, (x1, y1, f1, _) in enumerate(self.proteinas):
            for j, (x2, y2, f2, _) in enumerate(self.proteinas[:i]):
                if math.hypot(x1 - x2, y1 - y2) < 180:
                    self.enlaces.append((i, j, max(f1, f2)))
        rng = aleatorio(401)
        xs = [200, 650, 420, 880, 290, 760, 540, 130]
        self.cadenas = [
            {"x": x, "frac": 0.12 + 0.11 * i + rng.uniform(-0.02, 0.02), "n": int(rng.integers(3, 6)), "semilla": 400 + i, "fase": rng.uniform(0, 6)}
            for i, x in enumerate(xs)
        ]
        self.superficie = 2160
        rng = aleatorio(501)
        self.colonias = []
        base_y = self.superficie - 22
        for cx, ancho, alto, z in ((170, 230, 300, 0.75), (420, 300, 430, 1.0), (650, 240, 330, 0.8), (870, 280, 400, 1.0), (1010, 180, 240, 0.7)):
            r = int(15 * z + 3)
            pts, paso, fila, y = [], r * 1.75, 0, base_y - r
            while y > base_y - alto:
                frac = (base_y - y) / alto
                semi = ancho / 2 * math.sqrt(max(0.0, 1 - frac**1.6))
                x = cx - semi + (paso / 2 if fila % 2 else 0)
                while x <= cx + semi:
                    pts.append((x + rng.uniform(-3, 3), y + rng.uniform(-3, 3)))
                    x += paso
                y -= paso * 0.86
                fila += 1
            pts.sort(key=lambda p: (-p[1], abs(p[0] - cx)))
            tonos = [mezclar_color(P["turquesa"], (120, 230, 210), rng.uniform(0, 0.6)) for _ in pts]
            self.colonias.append({"cx": cx, "z": z, "r": r, "pts": pts, "tonos": tonos, "alto": alto})
        self.cubos = [
            {"col": i % 5, "dx": rng.uniform(-60, 60), "rel": i * 0.09 + rng.uniform(0, 0.04), "ang": rng.uniform(0, 360), "giro": rng.uniform(-200, 200)}
            for i in range(10)
        ]

    # ===================================================== 1 · GANCHO
    def escena_gancho(self, t, b, tr):
        img = self.fondo(t, (18, 44, 74), (3, 6, 14), semilla=1)
        t_cub = self.tp("cubriendo", b, -0.05)
        escala = 1.0 + 0.05 * (t - tr["inicio"])
        radio = 0.95 * salida_cubica(progreso(t, t_cub, t_cub + 1.1))
        self.diente_macro(img, t, escala, pelicula=radio if radio > 0.01 else None)
        # «Te acabas de cepillar…»: el cepillo sale de cuadro dejando el diente brillante
        k = progreso(t, 0.0, 0.9)
        if k < 1:
            cep = G.cepillo()
            pegar(img, cep, lerp(420, 1650, entrada_cubica(k)) - cep.width / 2 + 120, 1050, escala=1.0)
        return img

    # ===================================================== 2 · NEGACIÓN + PROMESA
    def escena_negacion(self, t, b, tr):
        t_esto = self.tp("Esto", b, -0.06)
        tx = self.textos(b)
        if t < t_esto:
            img = self.fondo(t, (16, 46, 80), (3, 6, 14), (150, 210, 255), semilla=2)
            escala = 1.12 + 0.04 * (t - tr["inicio"])
            self.diente_macro(img, t, escala, pelicula=1.0)
            rng = aleatorio(202)
            t_sal = self.tp("saliva", b)
            destino = None
            for i in range(9):
                llegada = tr["inicio"] + 0.25 + i * 0.18
                x, y = rng.uniform(380, 700), rng.uniform(820, 1350)
                if i == 4:
                    destino = (x, y)
                if t < llegada - 0.45:
                    continue
                if t < llegada:
                    k = progreso(t, llegada - 0.45, llegada)
                    pegar(img, G.esfera(18, (120, 190, 255), 200), x, lerp(560, y, entrada_cubica(k)), escala=1.0)
                else:
                    k = progreso(t, llegada, llegada + 0.35)
                    if k < 1:
                        anillo = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
                        r = 14 + 60 * k
                        ImageDraw.Draw(anillo).ellipse([90 - r, 90 - r, 90 + r, 90 + r], outline=(170, 220, 255, int(220 * (1 - k))), width=4)
                        pegar(img, anillo, x, y)
            alfa, _ = aparicion(t, t_sal - 0.05)
            G.etiqueta_con_guia(img, tx.get("etiqueta_saliva", "saliva"), "blanca", (250, 760), destino, alfa, 36)
            return img
        # «Esto pasa si no te cepillas en 24 horas»: la cámara sale a la boca y un anillo marca las 24 h
        img = self.fondo(t, (24, 40, 72), (3, 5, 12), semilla=3)
        spr = G.boca()
        k = salida_cubica(progreso(t, t_esto, t_esto + 1.3))
        zoom = lerp(2.6, 1.0, k)
        foco = (lerp(spr.width * 0.56, spr.width / 2, k), lerp(spr.height * 0.40, spr.height / 2, k))
        img.alpha_composite(vista_zoom(spr, 900 / spr.width * zoom, foco, (W / 2, 1070)))
        t24 = self.tp("24", b)
        if t >= t24 - 0.05:
            anillo = anillo_24h(progreso(t, t24 - 0.05, t24 + 0.75))
            pegar(img, anillo, W / 2, 1070)
        return img

    # ===================================================== 3 · PELÍCULA (MINUTO 1)
    def escena_pelicula(self, t, b, tr):
        tx = self.textos(b)
        t_vel = self.tp("Funciona", b, -0.08)
        if t >= t_vel:
            img = self.fondo(t, (30, 30, 50), (4, 5, 12), (230, 200, 140), semilla=4, intensidad=0.7)
            dib = progreso(t, t_vel + 0.05, t_vel + 0.75)
            union = suave(progreso(t, self.tp("velcro", b, -0.05), self.tp("velcro", b, 0.3)))
            alfa, escala = aparicion(t, t_vel, 0.3)
            pegar(img, G.velcro_lineal(dib, union, tx), W / 2, 1010, escala=escala, alfa=alfa)
            return img
        # Superficie del esmalte al microscopio: las proteínas de la saliva aterrizan y forman una malla
        dur = max(0.5, t_vel - tr["inicio_voz"])
        cam_x, cam_y = 70 + 30 * (t - tr["inicio"]), 100 + 8 * (t - tr["inicio"])
        img = G.textura_esmalte_superior().crop((int(cam_x), int(cam_y), int(cam_x) + W, int(cam_y) + H)).copy()
        capa = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(capa)
        for i, j, f in self.enlaces:
            a = progreso(t, tr["inicio_voz"] + f * dur + 0.15, tr["inicio_voz"] + f * dur + 0.5)
            if a > 0:
                d.line(
                    [(self.proteinas[i][0] - cam_x, self.proteinas[i][1] - cam_y), (self.proteinas[j][0] - cam_x, self.proteinas[j][1] - cam_y)],
                    fill=(200, 230, 255, int(150 * a)),
                    width=4,
                )
        img.alpha_composite(capa)
        colores = [(196, 170, 255), (120, 230, 205), (255, 190, 230)]
        for x, y, f, c in self.proteinas:
            llegada = tr["inicio_voz"] + f * dur
            sx, sy = x - cam_x, y - cam_y
            if t < llegada - 0.35:
                continue
            if t < llegada:
                k = progreso(t, llegada - 0.35, llegada)
                pegar(img, G.esfera(30, (120, 190, 255), 150), sx, sy, escala=lerp(2.2, 0.6, k), alfa=0.35 + 0.5 * k)
                continue
            k = progreso(t, llegada, llegada + 0.25)
            pegar(img, disco_suave(30, colores[c], 0.0, 90), sx, sy, escala=salida_rebote(k))
            pegar(img, G.esfera(17, colores[c]), sx, sy, escala=salida_rebote(k))
        # zoom de entrada (efecto «atravesar» la superficie del diente)
        k = progreso(t, tr["inicio"], tr["inicio"] + 0.4)
        if k < 1:
            img = vista_zoom(img, 1.0 + 0.45 * (1 - salida_cubica(k)), (W / 2, H / 2))
        t_capa = self.tp("capa", b)
        alfa, _ = aparicion(t, t_capa)
        G.etiqueta_con_guia(img, tx.get("etiqueta", "película salival"), "ambar", (540, 700), (560, 1000), alfa, 36)
        return img

    # ===================================================== 4 · PIONERAS (HORA 4)
    def escena_pioneras(self, t, b, tr):
        tx = self.textos(b)
        cam_y = lerp(260, 700, suave(progreso(t, tr["inicio"], tr["inicio"] + 1.5)))
        img = G.mundo_lateral().crop((0, int(cam_y), W, int(cam_y) + H)).copy()
        G.polvo(img, t, (150, 230, 235), 30, 5, 0.6)
        t_est = self.tp("estreptococos", b)
        ventana = max(0.8, t_est + 0.2 - tr["inicio_voz"])
        posiciones = {}
        for c in self.cadenas:
            spr = G.cadena_cocos(c["n"], 22, c["semilla"])
            llegada = tr["inicio_voz"] + c["frac"] * ventana
            inicio = llegada - 1.3
            if t < inicio:
                continue
            k = salida_cubica(progreso(t, inicio, llegada))
            y_fin = self.superficie - 24 - spr.height / 2
            y = lerp(cam_y - 200, y_fin, k)
            x = c["x"] + math.sin(t * 2.2 + c["fase"]) * 26 * (1 - k)
            if k < 1:
                pegar(img, G.paracaidas(spr.width + 30), x, y - cam_y - spr.height / 2 - 70, alfa=1 - progreso(t, llegada - 0.15, llegada))
            pegar(img, spr, x, y - cam_y)
            posiciones[c["semilla"]] = (x, y - cam_y)
        alfa, escala = aparicion(t, t_est - 0.05)
        pegar(img, G.tarjeta_dato(tx["dato"], "", tx["dato_sub"], P["menta"]), W / 2, 760, escala=escala, alfa=alfa)
        especies = tx.get("especies", [])
        for i, (nombre, semilla, pos) in enumerate(zip(especies, (402, 403), ((250, 1150), (820, 1150)))):
            if semilla in posiciones:
                a, _ = aparicion(t, t_est + 0.15 + i * 0.12)
                G.etiqueta_con_guia(img, nombre, "blanca", pos, posiciones[semilla], a, 32)
        t_ino = self.tp("inofensivas", b)
        a, _ = aparicion(t, t_ino - 0.05)
        if posiciones:
            destino = posiciones.get(406) or next(iter(posiciones.values()))  # la cadena central si ya aterrizó
            G.etiqueta_con_guia(img, tx.get("comensales", "comensales ✓"), "menta", (540, 1010), destino, a, 36)
        return img

    # ===================================================== 5 · COLONIAS Y ÁCIDO (HORA 8)
    def vista_colonias(self, t, b, tr):
        cam_y = 700
        img = G.mundo_lateral().crop((0, cam_y, W, cam_y + H)).copy()
        G.polvo(img, t, (150, 230, 235), 30, 6, 0.6)
        orbita = math.sin(progreso(t, tr["inicio"], tr["fin"]) * math.pi / 2) * 70
        crecimiento = 0.25 + 0.75 * suave(progreso(t, tr["inicio"], tr["inicio"] + 0.45 * (tr["fin"] - tr["inicio"])))
        for col in sorted(self.colonias, key=lambda c: c["z"]):
            desplaz = orbita * (1.0 if col["z"] > 0.9 else -0.6)
            n = int(len(col["pts"]) * crecimiento)
            for (x, y), tono in zip(col["pts"][:n], col["tonos"][:n]):
                pegar(img, G.esfera(col["r"], tono), x + desplaz, y - cam_y)
        t_az = self.tp("azúcar", b, -0.35)
        for cubo in self.cubos:
            col = self.colonias[cubo["col"]]
            desplaz = orbita * (1.0 if col["z"] > 0.9 else -0.6)
            t0 = t_az + cubo["rel"]
            llegada = t0 + 0.6
            if t < t0:
                continue
            x = col["cx"] + cubo["dx"] + desplaz
            y_cima = self.superficie - 22 - col["alto"] * crecimiento * 0.9 - cam_y
            if t < llegada:
                ang = int((cubo["ang"] + cubo["giro"] * (t - t0)) // 15 * 15) % 360
                pegar(img, G.cubo_azucar(64, ang), x, lerp(-80, y_cima, entrada_cubica(progreso(t, t0, llegada))))
                continue
            k = progreso(t, llegada, llegada + 0.35)
            pegar(img, G.cubo_azucar(64, int(cubo["ang"] // 15 * 15) % 360), x, y_cima, escala=1 - 0.6 * k, alfa=1 - k)
            for j in range(3):  # niebla ámbar (ácido) que sale de la colonia
                edad = t - llegada - j * 0.18
                if edad > 0:
                    ke = limitar(edad / 1.6)
                    pegar(img, disco_suave(int(40 + 70 * ke), P["ambar"], 0.1, int(170 * (1 - ke))), x + math.sin(edad * 3 + j) * 30, y_cima - 40 - 260 * ke)
        return img

    def vista_ph(self, t, b, t_inicio):
        img = self.fondo(t, (60, 40, 12), (6, 4, 2), (255, 200, 120), semilla=7, intensidad=0.6)
        valor = lerp(7.0, 5.5, suave(progreso(t, t_inicio + 0.1, t_inicio + 1.3)))
        pegar(img, G.medidor_ph(valor, 640, 150, self.textos(b).get("umbral")), 470, 1040)
        lectura = render_lineas([[("pH ", P["blanco"]), (f"{valor:.1f}".replace(".", ","), G.color_ph(valor))]], "titulo", 112, contorno=6, color_contorno=P["tinta"])
        pegar(img, lectura, 760, 770)
        return img

    def escena_colonias(self, t, b, tr):
        t_whip = self.tp("ácido", b, -0.22)
        dur = 0.24
        if t < t_whip:
            return self.vista_colonias(t, b, tr)
        if t >= t_whip + dur:
            return self.vista_ph(t, b, t_whip + dur)
        k = suave(progreso(t, t_whip, t_whip + dur))  # whip pan con desenfoque
        a = self.vista_colonias(t_whip, b, tr)
        c = self.vista_ph(t, b, t_whip + dur)
        lienzo = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        dx = int(k * W)
        if dx < W:
            lienzo.alpha_composite(a.crop((dx, 0, W, H)), (0, 0))
        if dx > 0:
            lienzo.alpha_composite(c.crop((0, 0, dx, H)), (W - dx, 0))
        return desenfoque_movimiento(lienzo, 170 * math.sin(k * math.pi) + 1)

    # ===================================================== 6 · DATO CLAVE
    def escena_dato_clave(self, t, b, tr):
        tx = self.textos(b)
        t_ellas = self.tp("fabrican", b, -0.08)
        if t >= t_ellas:
            # Primer plano de las bacterias de la placa fabricando ácido
            img = self.fondo(t, (70, 44, 10), (6, 4, 2), (255, 190, 90), semilla=8)
            k = t - t_ellas
            esc = 1.0 + 0.08 * k
            rng = aleatorio(606)
            for i in range(9):
                x, y = rng.uniform(220, 860), rng.uniform(900, 1380)
                spr = G.cadena_cocos(int(rng.integers(3, 6)), 30, 600 + i, P["ambar"])
                pegar(img, disco_suave(140, P["ambar"], 0.0, 70), x, y, escala=esc)
                pegar(img, spr, W / 2 + (x - W / 2) * esc, 1140 + (y - 1140) * esc, escala=esc)
            alfa, _ = aparicion(t, t_ellas + 0.05)
            G.etiqueta_con_guia(img, tx["etiqueta_ellas"], "ambar", (540, 760), (540, 1000), alfa, 36)
            return img
        img = self.fondo(t, (52, 38, 16), (5, 4, 3), (255, 214, 150), semilla=9, intensidad=0.7)
        t_sin = self.tp("sin", b)
        t_baja = self.tp("baja", b)
        t_acido = self.tp("ácido", b)
        v1 = lerp(7.0, 6.9, suave(progreso(t, t_baja, t_baja + 0.8)))
        v2 = lerp(7.0, 5.5, suave(progreso(t, t_acido - 0.2, t_acido + 0.9)))
        filas = [
            (tx["fila_1"], f"pH {v1:.1f}".replace(".", ","), (v1 - 4) / 4, P["menta"], aparicion(t, t_sin)[0]),
            (tx["fila_2"], f"pH {v2:.1f}".replace(".", ","), (v2 - 4) / 4, G.color_ph(v2), aparicion(t, t_acido - 0.25)[0]),
        ]
        alfa, escala = aparicion(t, tr["inicio"] + 0.05, 0.3)
        tarjeta = G.tarjeta_barras(tx["titulo"], filas)
        pegar(img, tarjeta, W / 2, 980, escala=escala, alfa=alfa)
        fuente_txt = sprite_legal(tx["fuente"])
        pegar(img, fuente_txt, W / 2, 980 + tarjeta.height / 2 + 40, alfa=alfa * 0.9)
        return img

    # ===================================================== 7 · LA NOCHE
    def escena_noche(self, t, b, tr):
        tx = self.textos(b)
        t_buenos = self.tp("Buenos", b, -0.05)
        if t >= t_buenos:
            img = G.dormitorio().copy()
            k = t - t_buenos
            ke = salida_rebote(progreso(t, t_buenos + 0.1, t_buenos + 0.45))  # nube verde del aliento
            if ke > 0:
                pegar(img, G.nube_aliento(), 820 + 30 * k, 860 - 40 * k, escala=0.35 + 0.5 * ke, alfa=limitar(ke * 1.5))
            sacudida = math.sin(k * 60) * 10 * (1 - progreso(t, t_buenos, t_buenos + 0.35))
            return vista_zoom(img, 1.0 + 0.03 * k, (W / 2 + sacudida, H / 2))
        img = self.fondo(t, (14, 24, 60), (2, 3, 10), (160, 180, 255), semilla=10, cy=700)
        pegar(img, disco_suave(380, (90, 120, 220), 0.0, 90), 200, 600)
        k = progreso(t, tr["inicio"], t_buenos)
        pegar(img, G.textura_lengua(), W / 2 + 30 * k, 1300 - 60 * k, escala=1.0 + 0.08 * k)
        # «río» de saliva que se reduce a un hilo
        sub = 2
        capa = Image.new("RGBA", (W // sub, H // sub), (0, 0, 0, 0))
        d = ImageDraw.Draw(capa)
        t_det = self.tp("detiene", b)
        ancho = lerp(64, 5, suave(progreso(t, tr["inicio"] + 0.2, t_det + 0.3)))
        pts = [(x / sub, (1230 - 60 * k + 55 * math.sin(x / 170 + 0.6)) / sub) for x in range(-40, W + 60, 30)]
        d.line(pts, fill=(80, 150, 255, 60), width=max(2, int(ancho * 2.4 / sub)), joint="curve")  # halo
        d.line(pts, fill=(110, 180, 255, 210), width=max(1, int(ancho / sub)), joint="curve")
        desfase = (t * 240 * lerp(1.0, 0.15, progreso(t, tr["inicio"], t_det + 0.3))) % 90
        for i in range(len(pts) - 1):
            if ((i * 30 + desfase) % 90) < 30 and ancho > 8:
                d.line([pts[i], pts[i + 1]], fill=(225, 245, 255, 220), width=max(1, int(ancho / sub / 4)))
        img.alpha_composite(capa.resize((W, H), Image.BILINEAR))
        alfa, _ = aparicion(t, t_det - 0.1)
        G.etiqueta_con_guia(img, tx["rio"], "oscura", (300, 900), (360, 1205 - 60 * k), alfa, 34)
        # volutas verdes de compuestos sulfurados volátiles que suben de la lengua
        t_bac = self.tp("bacterias", b, -0.2)
        t_azu = self.tp("azufre", b, -0.2)
        if t > t_bac:
            sub = 3
            capa = Image.new("RGBA", (W // sub, H // sub), (0, 0, 0, 0))
            d = ImageDraw.Draw(capa)
            conv = suave(progreso(t, t_azu - 0.3, t_azu + 0.2))
            for i in range(8):
                x0 = 120 + i * 120
                subida = progreso(t, t_bac + i * 0.08, t_bac + i * 0.08 + 1.3)
                puntos = []
                for s in range(14):
                    f = s / 13
                    y = 1200 - 600 * f * subida
                    x = x0 + 50 * math.sin(f * 5 + t * 2.5 + i)
                    puntos.append((lerp(x, W / 2 + (x0 - W / 2) * 0.15, conv * f) / sub, lerp(y, 850 + (y - 850) * 0.3, conv * f) / sub))
                d.line(puntos, fill=(150, 230, 120, int(140 * subida)), width=10 // sub + 3, joint="curve")
            img.alpha_composite(capa.resize((W, H), Image.BILINEAR))
            a = progreso(t, t_azu, t_azu + 0.3)
            if a > 0:
                pegar(img, sprite_h2s(), W / 2, 850, alfa=a, escala=0.8 + 0.2 * salida_cubica(a))
                al, _ = aparicion(t, t_azu + 0.25)
                G.etiqueta_con_guia(img, tx["gases"], "ambar", (540, 640), (540, 760), al, 32)
        return img

    # ===================================================== 8 · HORA 24: LA REVELACIÓN
    def escena_revelacion(self, t, b, tr):
        tx = self.textos(b)
        t_car = self.tp("Caries", b)
        t_no = self.tp("No", b)
        t_gin = self.tp("Gingivitis", b)
        t_tam = self.tp("Tampoco", b)
        t_tarda = self.tp("tarda", b)
        congelado = t_car <= t < t_car + 0.15
        tt = t_car if congelado else t
        img = self.fondo(tt, (40, 34, 20), (4, 4, 4), (255, 230, 170), semilla=11)
        escala = 1.0 + 0.16 * suave(progreso(tt, tr["inicio"], t_car)) + 0.02 * max(0.0, tt - t_car)
        self.diente_macro(img, tt, escala, placa=0.9)
        t_peg = self.tp("pegajosa", b)
        alfa, _ = aparicion(t, t_peg)
        if t < t_car:
            G.etiqueta_con_guia(img, tx["placa"], "blanca", (300, 760), (470, 950), alfa, 36)
        oscuro = 0.6 * progreso(t, t_car, t_car + 0.3)
        if oscuro > 0:
            img.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * oscuro))))
        for texto, t_q, t_x, y in ((tx["caries"], t_car, t_no, 760), (tx["gingivitis"], t_gin, t_tam, 930)):
            a, e = aparicion(t, t_q - 0.03)
            if a > 0:
                pil = G.pildora(f"¿{texto}?", "blanca", 46)
                pegar(img, pil, W / 2 - 50, y, escala=e, alfa=a)
                ax, ex = aparicion(t, t_x - 0.02, 0.3)
                pegar(img, G.insignia("x", 110), W / 2 - 50 + pil.width / 2 + 75, y, escala=ex, alfa=ax)
        if t >= t_tarda - 0.1:
            a, e = aparicion(t, t_tarda - 0.1)
            pegar(img, G.linea_tiempo(progreso(t, t_tarda - 0.1, t_tarda + 1.2)), W / 2, 1200, escala=e, alfa=a)
        if congelado or t_car <= t < t_car + 0.14:
            flash = 1 - progreso(t, t_car, t_car + 0.14)
            img.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(170 * flash))))
        return img

    # ===================================================== 9 · PRUEBA HISTÓRICA (1965)
    def escena_historia(self, t, b, tr):
        tx = self.textos(b)
        img = self.fondo(t, (64, 48, 30), (8, 6, 4), (255, 220, 170), semilla=12, intensidad=0.8)
        t_vol = self.tp("voluntarios", b)
        t_dej = self.tp("dejaron", b)
        t_gin = self.tp("gingivitis", b)
        t_sem = self.tp("semana", b)
        t_des = self.tp("desapareció", b)
        a, e = aparicion(t, tr["inicio"] + 0.02, 0.3)
        pegar(img, G.tarjeta_dato(tx["anio"], "", tx["anio_sub"], P["oro"]), W / 2, 740, escala=e, alfa=a)
        inflamacion = progreso(t, t_dej, t_gin + 0.2) * (1 - progreso(t, t_sem, t_sem + 0.9))
        q = int(round(10 * inflamacion))
        for i in range(5):
            ai, ei = aparicion(t, t_vol - 0.1 + i * 0.07)
            if ai > 0:
                x = 140 + i * 200
                y = 1150 + (20 if i % 2 else 0) + math.sin(t * 2 + i) * 4
                pegar(img, G.persona_busto(i, q), x, y, escala=0.85 * ei, alfa=ai)
        if t >= t_dej:
            dia = 1 + int(20 * progreso(t, t_dej, t_gin + 0.2))
            dia = dia if t < t_sem else max(1, 21 - int(7 * progreso(t, t_sem, t_sem + 0.9)))
            pegar(img, G.pildora(f"{tx['dias']} {dia}", "ambar", 38), W / 2, 1400)
        ag, eg = aparicion(t, t_gin - 0.05)
        if ag > 0 and t < t_sem:
            pegar(img, G.pildora(tx["gingivitis"], "coral", 38), W / 2, 960, escala=eg, alfa=ag)
        asn, esn = aparicion(t, self.tp("cepillado", b, 0.1))
        if asn > 0:
            pegar(img, G.pildora(tx["sanos"], "menta", 38), W / 2, 960, escala=esn, alfa=asn)
            for j in range(5):
                fase = progreso(t, t_des + j * 0.1, t_des + j * 0.1 + 0.7)
                pegar(img, G.destello(80), 140 + j * 200, 1110, alfa=math.sin(fase * math.pi))
        pegar(img, sprite_legal(tx["fuente"]), W / 2, 1460, alfa=0.9 * a)
        return img

    # ===================================================== 10 · LA COSTUMBRE
    def escena_costumbre(self, t, b, tr):
        t_cos = self.tp("costumbre", b, -0.1)
        t_todo = self.tp("todo", b)
        cambio = t_todo - 0.2
        if t < cambio:
            return self.vista_calendario(t, b, tr, t_cos)
        cep = self.vista_cepillado(t, b, tr, t_todo)
        if t < cambio + 0.2:
            return Image.blend(self.vista_calendario(t, b, tr, t_cos), cep, progreso(t, cambio, cambio + 0.2))
        return cep

    def vista_calendario(self, t, b, tr, t_cos):
        img = self.fondo(t, (24, 50, 80), (3, 6, 14), semilla=13)
        dia_f = max(0.0, (t - t_cos) / 0.09)
        dia = int(limitar(dia_f, 0, 29)) + 1
        pegar(img, G.hoja_calendario(dia), 300, 1060)
        if 0 < dia_f < 29 and dia_f % 1 < 0.5:
            ant = G.hoja_calendario(max(1, dia - 1))
            k = (dia_f % 1) / 0.5
            alto = max(1, int(ant.height * (1 - k)))
            pegar(img, ant.resize((ant.width, alto)), 300, 1060 - ant.height / 2 + alto / 2 - 80 * k, alfa=1 - k)
        self.diente_macro(img, t, 0.62, centro=(770, 1080), placa=0.25 + 0.7 * limitar(dia / 14), brillos=False)
        return img

    def vista_cepillado(self, t, b, tr, t_todo):
        img = self.fondo(t, (20, 70, 80), (3, 8, 12), semilla=14)
        tam = (600, 840)
        cx, cy = W / 2, 1080
        izq = cx - tam[0] / 2
        sprite, _ = G.diente(*tam)
        pegar(img, G.resplandor(("diente", tam), sprite, 40, (90, 200, 230), 0.9), cx, cy, alfa=0.55)
        pegar(img, sprite, cx, cy)
        fin = max(t_todo + 1.3, self.tp("cepillado", b) + 0.2)
        k = suave(progreso(t, t_todo, fin))  # barrido en cámara lenta
        cabeza_x = lerp(-150, W + 250, k)
        corte = int(limitar(cabeza_x - 40 - izq, 0, tam[0]))
        if corte < tam[0]:
            pegar(img, G.placa(*tam).crop((corte, 0, tam[0], tam[1])), izq + corte, cy - tam[1] / 2, ancla="esquina")
        t_cep = self.tp("cepillado", b, 0.2)
        if t > t_cep:
            for j, (bx, by) in enumerate(((0.32, 0.4), (0.6, 0.52), (0.45, 0.75), (0.7, 0.3))):
                fase = progreso(t, t_cep + j * 0.12, t_cep + j * 0.12 + 0.7)
                pegar(img, G.destello(110), izq + bx * tam[0], cy - tam[1] / 2 + by * tam[1], alfa=math.sin(fase * math.pi))
        cep = G.cepillo()
        pegar(img, cep, cabeza_x - cep.width / 2 + 120, cy - 40)
        rot = progreso(t, t_todo + 0.15, t_todo + 0.4) * (1 - progreso(t, fin - 0.3, fin))
        if rot > 0:
            pegar(img, G.pildora(self.textos(b).get("pasta", "pasta con flúor"), "menta", 34), cabeza_x - 40, cy - 250, alfa=rot)
        return img

    # ===================================================== 11 · COMPARTIR
    def miniatura(self, t):
        img = Image.new("RGBA", (420, 560), (0, 0, 0, 0))
        sprite, _ = G.diente(300, 420)
        pegar(img, disco_suave(170, (90, 200, 230), 0.0, 70), 210, 290)
        pegar(img, sprite, 210, 290)
        fase = (t * 0.8) % 1.0
        pegar(img, G.destello(60), 160, 250, alfa=math.sin(fase * math.pi) ** 2)
        return img

    def estado_telefono(self, t, b, tr):
        t_quien = self.tp("quien", b, 0.05)
        t_env = self.tp("cepillarse", b, 0.15)
        hoja = salida_cubica(progreso(t, tr["inicio"], tr["inicio"] + 0.35))
        seleccion = salida_rebote(progreso(t, t_quien + 0.18, t_quien + 0.4))
        enviado = progreso(t, t_env + 0.12, t_env + 0.3)
        return hoja, seleccion, enviado, t_quien, t_env

    def escena_compartir(self, t, b, tr):
        img = self.fondo(t, (30, 30, 60), (3, 4, 10), semilla=15)
        hoja, sel, env, t_quien, t_env = self.estado_telefono(t, b, tr)
        tel = G.telefono(self.miniatura(t), hoja, sel, env, self.textos(b))
        esc = 0.9
        cx, cy = W / 2, 1010
        pegar(img, tel, cx, cy, escala=esc)
        # posiciones (en el fotograma) del primer contacto y del botón «Enviar»
        x0, y0 = cx - tel.width * esc / 2 + 18 * esc, cy - tel.height * esc / 2 + 18 * esc
        y_hoja = y0 + (tel.height - 36 - 430 * hoja) * esc
        contacto = (x0 + 70 * esc, y_hoja + 166 * esc)
        boton = (x0 + (tel.width - 36) / 2 * esc, y_hoja + 335 * esc)
        inicio_mano = (980, 1750)
        if t < t_quien:
            pos = (lerp(inicio_mano[0], contacto[0], salida_cubica(progreso(t, tr["inicio"] + 0.2, t_quien))),
                   lerp(inicio_mano[1], contacto[1], salida_cubica(progreso(t, tr["inicio"] + 0.2, t_quien))))
            pulsando = 0.0
        elif t < t_env:
            k = salida_cubica(progreso(t, t_quien + 0.3, t_env))
            pos = (lerp(contacto[0], boton[0], k), lerp(contacto[1], boton[1], k))
            pulsando = 1 - abs(progreso(t, t_quien, t_quien + 0.3) * 2 - 1)
        else:
            pos = boton
            pulsando = 1 - abs(progreso(t, t_env, t_env + 0.3) * 2 - 1)
        mano = G.mano()
        pegar(img, mano, pos[0] + mano.width * 0.08, pos[1] + mano.height * 0.42, escala=1.0 - 0.08 * pulsando)
        return img

    # ===================================================== CIERRE
    def escena_cierre(self, t, b=None, tr=None):
        golpe = self.cierre["golpe"]
        t0 = self.cierre["inicio"]
        ultimo = self.bloques[self.orden[-1]]
        img = self.escena_compartir(min(t, t0), ultimo, self.tramos[self.orden[-1]])
        oscuro = progreso(t, t0, t0 + 0.6)
        img.alpha_composite(Image.new("RGBA", (W, H), (2, 3, 8, int(255 * oscuro))))
        # el mismo diente del gancho, en penumbra (bucle visual) con el logo dentro
        if oscuro > 0.3:
            fondo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            self.diente_macro(fondo, t, 1.0, brillos=False)
            img.alpha_composite(con_alfa(fondo, 0.22 * progreso(t, t0 + 0.2, t0 + 0.8)))
            pegar(img, disco_suave(130, (0, 0, 0), 0.0, 200), W / 2, 1040)
        a, e = aparicion(t, t0 + 0.35, 0.35)
        golpeado = t >= golpe
        rebote = 1.0 + 0.22 * (1 - salida_cubica(progreso(t, golpe, golpe + 0.35))) if golpeado else 1.0
        if golpeado:
            pegar(img, disco_suave(260, (255, 120, 170), 0.0, int(120 * (1 - progreso(t, golpe, golpe + 0.8)))), W / 2, 1040)
        pegar(img, G.icono_instagram(150, golpeado), W / 2, 1040, escala=e * rebote, alfa=a)
        usuario = sprite_usuario(config.MARCA["usuario"].upper(), 44)
        pegar(img, usuario, W / 2, 1160, alfa=a)
        if self.logo is not None:  # logotipo discreto del consultorio (assets/imagenes/logo.png)
            pegar(img, self.logo, W / 2, 1290, escala=0.7, alfa=a * 0.9)
        if golpeado:
            k = progreso(t, golpe, golpe + 0.25)
            if k < 1:
                img = vista_zoom(img, 1.0 + 0.02 * (1 - k), (W / 2 + math.sin(t * 90) * 6 * (1 - k), H / 2))
        return img


# ---------------------------------------------------------------------------
# Sprites auxiliares con caché
# ---------------------------------------------------------------------------
_CACHE = {}


def _cache(clave, fabrica):
    if clave not in _CACHE:
        _CACHE[clave] = fabrica()
    return _CACHE[clave]


def sprite_usuario(texto, tam):
    return _cache(("usuario", texto, tam), lambda: G.texto_espaciado(texto, "condensada", tam, (255, 255, 255), 0.04))


def sprite_legal(texto):
    return _cache(("legal", texto), lambda: render_lineas([texto], "dato", 26, color=(200, 206, 216), contorno=2, color_contorno=P["tinta"]))


def sprite_h2s():
    def fab():
        t = render_lineas(["H₂S"], "titulo", 170, color=(175, 240, 140), contorno=6, color_contorno=(20, 60, 30))
        glow = disco_suave(int(t.width * 0.55), (150, 225, 120), 0.0, 90)
        lienzo = Image.new("RGBA", (glow.width, glow.width), (0, 0, 0, 0))
        lienzo.alpha_composite(glow)
        pegar(lienzo, t, lienzo.width / 2, lienzo.height / 2)
        return lienzo

    return _cache(("h2s",), fab)


def anillo_24h(k):
    """Anillo menta que se dibuja alrededor de la boca: «24 horas»."""
    q = int(limitar(k) * 40)

    def fab():
        tam = 1060
        img = Image.new("RGBA", (tam, tam), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        r = tam / 2 - 30
        d.arc([30, 30, tam - 30, tam - 30], -90, -90 + 360 * q / 40, fill=P["menta"] + (255,), width=12)
        for i in range(24):
            if i / 24 <= q / 40:
                a, b = i * 15, i * 15
                x0, y0 = (tam / 2 + (r - 30) * math.cos(math.radians(a - 90)), tam / 2 + (r - 30) * math.sin(math.radians(a - 90)))
                x1, y1 = (tam / 2 + (r - 10) * math.cos(math.radians(b - 90)), tam / 2 + (r - 10) * math.sin(math.radians(b - 90)))
                d.line([(x0, y0), (x1, y1)], fill=(255, 255, 255, 200), width=5 if i % 6 == 0 else 3)
        glow = G.resplandor(("anillo", q), img, 16, P["menta"], 1.4)
        out = Image.new("RGBA", glow.size, (0, 0, 0, 0))
        out.alpha_composite(glow)
        out.alpha_composite(img, ((glow.width - tam) // 2, (glow.height - tam) // 2))
        return out

    return _cache(("anillo24", q), fab)
