"""
Composición de fotogramas: las 10 escenas del documento de producción, las
transiciones y las capas globales (textos, reloj de 24 h, subtítulos).

Cada fotograma es una función pura del tiempo `t` (en segundos). Por eso el
video se puede renderizar en paralelo por tramos sin que cambie el resultado.
"""

import functools
import math
from pathlib import Path

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
    gradiente_radial,
    gradiente_vertical,
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
# Resolución de referencias de tiempo («palabras ancla»)
# ---------------------------------------------------------------------------
def resolver_tiempo(ref, palabras):
    """Convierte 12.5 o ("palabra", bloque[, desplazamiento]) en segundos absolutos."""
    if ref is None:
        return None
    if isinstance(ref, (int, float)):
        return float(ref)
    palabra, bloque, *resto = ref
    desplazamiento = resto[0] if resto else 0.0
    objetivo = normalizar_palabra(palabra)
    lista = palabras.get(bloque) or palabras.get(str(bloque)) or []
    for p in lista:
        if normalizar_palabra(p["texto"]) == objetivo:
            return p["inicio"] + desplazamiento
    for p in lista:
        if normalizar_palabra(p["texto"]).startswith(objetivo):
            return p["inicio"] + desplazamiento
    raise ErrorProduccion(
        f"guion.py usa la palabra ancla «{palabra}» en el bloque {bloque}, "
        f"pero esa palabra no está en la narración de ese bloque. Corrige la referencia."
    )


def bloque_en(t):
    for b in guion.BLOQUES:
        if b["inicio"] <= t < b["fin"]:
            return b
    return guion.BLOQUES[-1]


def interpolar_claves(claves, t):
    """Interpolación con aceleración/frenado (time-lapse) entre fotogramas clave (t, valor)."""
    if t <= claves[0][0]:
        return claves[0][1]
    for (t0, v0), (t1, v1) in zip(claves, claves[1:]):
        if t0 <= t <= t1:
            k = progreso(t, t0, t1)
            k = k * k * k * (k * (6 * k - 15) + 10)  # smootherstep
            return lerp(v0, v1, k)
    return claves[-1][1]


def vista_zoom(sprite, escala, foco_sprite, foco_pantalla=(W / 2, H / 2), tam=(W, H)):
    """Encuadre de cámara: muestra `sprite` ampliado `escala` veces con foco_sprite en foco_pantalla."""
    a = 1.0 / escala
    c = foco_sprite[0] - foco_pantalla[0] * a
    f = foco_sprite[1] - foco_pantalla[1] * a
    return sprite.transform(tam, Image.AFFINE, (a, 0, c, 0, a, f), resample=Image.BILINEAR)


# ---------------------------------------------------------------------------
# Recursos del usuario (imágenes o videos que sustituyen a las ilustraciones)
# ---------------------------------------------------------------------------
class Medio:
    """Imagen o video del usuario ajustado a 9:16 (recorte centrado tipo «cover»)."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.es_video = self.ruta.suffix.lower() in guion.EXTENSIONES_VIDEO
        if self.es_video:
            from moviepy import VideoFileClip

            try:
                self.clip = VideoFileClip(str(self.ruta), audio=False)
            except Exception as exc:
                raise ErrorProduccion(f"No se pudo abrir el video {self.ruta.name}: {exc}") from exc
            self.duracion = max(0.1, self.clip.duration)
        else:
            try:
                self.imagen = self._cubrir(Image.open(self.ruta).convert("RGB"), margen=1.12)
            except Exception as exc:
                raise ErrorProduccion(f"No se pudo abrir la imagen {self.ruta.name}: {exc}") from exc

    @staticmethod
    def _cubrir(img, margen=1.0):
        escala = max(W / img.width, H / img.height) * margen
        img = img.resize((int(img.width * escala + 0.5), int(img.height * escala + 0.5)), Image.LANCZOS)
        x0, y0 = (img.width - int(W * margen)) // 2, (img.height - int(H * margen)) // 2
        return img.crop((x0, y0, x0 + int(W * margen), y0 + int(H * margen)))

    def fotograma(self, t_local, duracion_uso=3.0):
        if self.es_video:
            frame = self.clip.get_frame(t_local % self.duracion)
            return self._cubrir(Image.fromarray(frame)).convert("RGBA")
        # Imagen fija con zoom lento (efecto Ken Burns)
        k = limitar(t_local / max(duracion_uso, 0.1))
        z = 1.0 + 0.1 * k
        foco = (self.imagen.width / 2, self.imagen.height / 2)
        return vista_zoom(self.imagen, z * W / self.imagen.width * 1.12, foco).convert("RGBA")


def buscar_recurso(nombre):
    for carpeta, exts in (
        (config.DIR_VIDEOS, guion.EXTENSIONES_VIDEO),
        (config.DIR_IMAGENES, guion.EXTENSIONES_IMAGEN),
    ):
        for ext in exts:
            ruta = carpeta / f"{nombre}{ext}"
            if ruta.exists():
                return ruta
    return None


# ---------------------------------------------------------------------------
# Estilos de texto en pantalla
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=128)
def sprite_texto(texto, estilo):
    if estilo == "titulo":
        return render_lineas(texto.split("\n"), "titulo", 92, contorno=7, color_contorno=P["tinta"], sombra=8)
    if estilo == "destacado":
        partes = texto.split("\n")
        colores = [P["blanco"], P["menta"]]
        lineas = [[(l, colores[i % 2])] for i, l in enumerate(partes)]
        return render_lineas(lineas, "titulo", 104, contorno=8, color_contorno=P["tinta"], sombra=10)
    if estilo == "dato":
        # «04:00 · resto»: la hora en menta, el resto en blanco, sobre una píldora azul.
        if " · " in texto:
            cabeza, cola = texto.split(" · ", 1)
            segs = [(cabeza + " · ", P["menta"]), (cola, P["blanco"])]
        else:
            segs = [(texto, P["blanco"])]
        t = render_lineas([segs], "dato_negrita", 46)
        return _pildora(t, P["azul"] + (215,))
    if estilo.startswith("badge_"):
        color = P[estilo.split("_", 1)[1]]
        tinta = P["blanco"] if estilo == "badge_coral" else P["azul"]
        t = render_lineas(texto.split("\n"), "titulo", 50, color=tinta)
        return _pildora(t, color + (245,))
    if estilo == "fuente":
        return render_lineas([texto], "dato", 32, color=(235, 240, 245), contorno=2, color_contorno=P["tinta"])
    if estilo in ("legal", "legal2"):
        return render_lineas([texto], "dato", 27, color=(230, 236, 240), contorno=2, color_contorno=P["tinta"])
    raise ErrorProduccion(f"Estilo de texto desconocido en guion.py: {estilo}")


def _pildora(texto_img, color):
    pad_x, pad_y = 34, 16
    img = Image.new("RGBA", (texto_img.width + pad_x * 2, texto_img.height + pad_y * 2), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=img.height // 2, fill=color)
    img.alpha_composite(texto_img, (pad_x, pad_y))
    return img


POSICIONES = {
    "titulo": ("centro", 520),
    "destacado": ("centro", 500),
    "dato": ("izquierda", 455),
    "fuente": ("centro", 1440),
    "legal": ("izquierda", 1508),
    "legal2": ("izquierda", 1468),
}


# ---------------------------------------------------------------------------
# Renderizador
# ---------------------------------------------------------------------------
class Renderizador:
    def __init__(self, plan):
        self.palabras = {int(k): v for k, v in plan["palabras"].items()}
        self.env_voz = plan.get("env_voz", [])
        self.trozos = plan["trozos"]
        self.t = lambda ref: resolver_tiempo(ref, self.palabras)
        self.textos = []
        for b in guion.BLOQUES:
            for tx in b.get("textos", []):
                hasta = self.t(tx.get("hasta")) if tx.get("hasta") is not None else b["fin"]
                self.textos.append(dict(tx, desde=self.t(tx["desde"]), hasta=hasta, fin_bloque=b["fin"]))
        for tx in guion.TEXTOS_GLOBALES:
            self.textos.append(dict(tx, desde=self.t(tx["desde"]), hasta=self.t(tx["hasta"]), fin_bloque=None))
        self.reloj_claves = [(self.t(a), v) for a, v in guion.RELOJ_CLAVES]
        self.noche = tuple(self.t(x) for x in guion.RELOJ_NOCHE)
        self.medios = {}
        for nombre in guion.RECURSOS:
            ruta = buscar_recurso(nombre)
            self.medios[nombre] = Medio(ruta) if ruta and nombre != "logo" else None
        ruta_logo = buscar_recurso("logo")
        self.logo = None
        if ruta_logo and ruta_logo.suffix.lower() in guion.EXTENSIONES_IMAGEN:
            logo = Image.open(ruta_logo).convert("RGBA")
            self.logo = logo.resize((int(logo.width * 150 / logo.height), 150), Image.LANCZOS)
        self._preparar_escena3()
        self._preparar_escena4()
        self._preparar_escena5()
        self.escenas = {
            1: self.escena_gancho,
            2: self.escena_promesa,
            3: self.escena_pelicula,
            4: self.escena_pioneros,
            5: self.escena_colonias,
            6: self.escena_dato_clave,
            7: self.escena_noche,
            8: self.escena_revelacion,
            9: self.escena_conclusion,
            10: self.escena_cierre,
        }

    # ------------------------------------------------------------------ util
    def voz(self, t):
        i = int(t * FPS)
        return self.env_voz[i] if 0 <= i < len(self.env_voz) else 0.0

    def fotograma(self, t):
        """Devuelve el fotograma RGB (numpy H×W×3) del instante t."""
        lienzo = self.escena(t)
        self.capa_textos(lienzo, t)
        self.capa_reloj(lienzo, t)
        self.capa_subtitulos(lienzo, t)
        return np.asarray(lienzo.convert("RGB"))

    def escena(self, t):
        b = bloque_en(t)
        return self.escenas[b["id"]](t, b)

    # --------------------------------------------------------------- capas
    def capa_textos(self, lienzo, t):
        for tx in self.textos:
            if not (tx["desde"] <= t < tx["hasta"]):
                continue
            spr = sprite_texto(tx["texto"], tx["estilo"])
            entrada = progreso(t, tx["desde"], tx["desde"] + 0.22)
            salida = 1.0
            if tx.get("fin_bloque") is None or tx["hasta"] < tx["fin_bloque"] - 0.01:
                salida = 1 - progreso(t, tx["hasta"] - 0.15, tx["hasta"])
            alfa = min(limitar(entrada * 1.8), salida)
            escala = 1.0
            if tx["estilo"] in ("titulo", "destacado") or tx["estilo"].startswith("badge_"):
                escala = 0.6 + 0.4 * salida_rebote(entrada)
            alineacion, y = POSICIONES.get(tx["estilo"], ("centro", 600))
            y = tx.get("y", y)
            if alineacion == "izquierda":
                pegar(lienzo, spr, 50 + spr.width * escala / 2, y, escala=escala, alfa=alfa)
            else:
                ancho_max = 1000
                if spr.width * escala > ancho_max:
                    escala *= ancho_max / (spr.width * escala)
                pegar(lienzo, spr, W / 2, y, escala=escala, alfa=alfa)

    def capa_reloj(self, lienzo, t):
        if t < guion.RELOJ_APARECE:
            return
        horas = interpolar_claves(self.reloj_claves, t)
        noche = self.noche[0] <= t < self.noche[1]
        brillo = 0
        if 31.8 <= t < 33.6:
            brillo = int(round(4 * (1 - progreso(t, 32.6, 33.6))))
        spr = G.reloj(int(round(horas * 60)), noche, brillo)
        escala = 0.25 + 0.75 * salida_rebote(progreso(t, guion.RELOJ_APARECE, guion.RELOJ_APARECE + 0.35))
        pegar(lienzo, spr, W - 160, 290, escala=escala)

    def capa_subtitulos(self, lienzo, t):
        spr, escala = subtitulos.sprite_en(self.trozos, t)
        if spr is None:
            return
        ancho_max = 860
        if spr.width * escala > ancho_max:
            escala *= ancho_max / (spr.width * escala)
        cfg = config.SUBTITULOS
        pegar(lienzo, spr, cfg["centro_x"], cfg["centro_y"] * H, escala=escala)

    # ------------------------------------------------- piezas reutilizables
    def macro_diente(self, escala=1.0, densidad_placa=0.0, radio_pelicula=None, brillos=True, t=0.0):
        """Plano macro de un incisivo (limpio, con película o con placa)."""
        img = G.fondo_macro().copy()
        sprite, _ = G.diente(700, 980)
        capa = sprite.copy()
        if radio_pelicula:
            capa.alpha_composite(G.pelicula(700, 980, radio_pelicula))
        if densidad_placa > 0:
            capa.alpha_composite(con_alfa(G.placa(700, 980), densidad_placa))
        if brillos:
            for k, (bx, by) in enumerate(((0.33, 0.42), (0.62, 0.55), (0.4, 0.78))):
                fase = (t * 0.9 + k * 0.37) % 1.0
                a = math.sin(fase * math.pi) ** 3
                pegar(capa, G.destello(70), bx * 700, by * 980, alfa=a)
        pegar(img, capa, W / 2, 960, escala=escala, remuestreo=Image.BILINEAR)
        return img

    def dentista(self, t, nombre_medio, t0, sonrisa=0.5, cepillo=None, hablar=True, duracion_uso=4.0):
        medio = self.medios.get(nombre_medio)
        if medio is not None:
            return medio.fotograma(t - t0, duracion_uso)
        img = G.fondo_consultorio().copy()
        balanceo = math.sin(t * 1.6) * 6
        img.alpha_composite(G.cuerpo_dentista(), (0, int(balanceo)))
        parpadeo = (t % 3.1) < 0.12
        apertura = self.voz(t) if hablar else 0.0
        cara = G.cara_dentista(apertura=apertura, parpadeo=parpadeo, sonrisa=sonrisa, cepillo=cepillo)
        img.alpha_composite(cara, (0, int(balanceo)))
        return img

    # ======================================================== ESCENA 1
    def escena_gancho(self, t, b):
        corte = self.t(("cubriendo", 1, -0.05))
        if t < corte:
            # Espejo del consultorio: termina de cepillarse y sonríe.
            fin_cepillado = min(1.05, corte - 0.4)
            if t < fin_cepillado:
                cepillo = math.sin(t * 2 * math.pi * 3.2) * 28
            else:
                cepillo = 28 + 900 * entrada_cubica(progreso(t, fin_cepillado, fin_cepillado + 0.35))
            sonrisa = 0.2 + 0.8 * suave(progreso(t, fin_cepillado, fin_cepillado + 0.3))
            img = self.dentista(t, "dentista_espejo", b["inicio"], sonrisa=sonrisa, cepillo=cepillo, hablar=False, duracion_uso=corte)
            if self.medios.get("dentista_espejo") is None:
                img = vista_zoom(img, 1.18 + 0.04 * t, (540, 860), (540, 900))
                if t > fin_cepillado + 0.2:
                    pegar(img, G.destello(90), 600, 1010, alfa=math.sin(progreso(t, fin_cepillado + 0.2, corte) * math.pi))
            img.alpha_composite(marco_espejo())
            return img
        # Corte seco en «cubriendo» → macro con push-in rápido y la película que se abre como una gota.
        lt = t - corte
        escala = 1.0 + 0.28 * salida_cubica(progreso(lt, 0, 0.45)) + 0.03 * lt
        radio = 0.9 * salida_cubica(progreso(lt, 0.15, 1.35))
        return self.macro_diente(escala, radio_pelicula=radio, t=t)

    # ======================================================== ESCENA 2
    def escena_promesa(self, t, b):
        lt = t - b["inicio"]
        img = gradiente_radial(W, H, (W / 2, 1050), 1100, (30, 80, 110), P["azul"])
        spr = G.boca()
        k = salida_cubica(progreso(lt, 0, 1.9))
        zoom = lerp(4.2, 1.0, k)
        escala_base = 1000 / spr.width
        foco_incisivo = (spr.width / 2 + spr.width * 0.06, spr.height * 0.40)
        foco_centro = (spr.width / 2, spr.height / 2)
        foco = (lerp(foco_incisivo[0], foco_centro[0], k), lerp(foco_incisivo[1], foco_centro[1], k))
        vista = vista_zoom(spr, escala_base * zoom, foco, (W / 2, 1060))
        img.alpha_composite(vista)
        return img

    # ======================================================== ESCENA 3
    def _preparar_escena3(self):
        rng = aleatorio(301)
        self.prot = []
        for i in range(64):
            llegada = 6.25 + 2.9 * (i / 64) + rng.uniform(-0.08, 0.08)
            x, y = rng.uniform(140, 1260), rng.uniform(640, 1500)
            self.prot.append((x, y, llegada, int(rng.integers(0, 3))))
        self.enlaces = []
        for i, (x1, y1, a1, _) in enumerate(self.prot):
            for j, (x2, y2, a2, _) in enumerate(self.prot[:i]):
                if math.hypot(x1 - x2, y1 - y2) < 175:
                    self.enlaces.append((i, j, max(a1, a2)))

    def vista_pelicula(self, t):
        """Superficie del esmalte al microscopio con la malla de proteínas formándose."""
        cam_x, cam_y = 60 + 32 * (t - 6), 80 + 6 * (t - 6)
        tex = G.textura_esmalte_superior()
        img = tex.crop((int(cam_x), int(cam_y), int(cam_x) + W, int(cam_y) + H))
        capa = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(capa)
        pulso = 0.5 + 0.5 * math.sin(t * 7) if t > 9.0 else 0.0
        for i, j, llegada in self.enlaces:
            a = progreso(t, llegada + 0.15, llegada + 0.5)
            if a > 0:
                x1, y1 = self.prot[i][0] - cam_x, self.prot[i][1] - cam_y
                x2, y2 = self.prot[j][0] - cam_x, self.prot[j][1] - cam_y
                d.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, int(150 * a + 70 * pulso * a)), width=5)
        img.alpha_composite(capa)
        colores = [(196, 170, 255), (140, 220, 200), (255, 200, 230)]
        for x, y, llegada, c in self.prot:
            sx, sy = x - cam_x, y - cam_y
            if t < llegada - 0.35:
                continue
            if t < llegada:
                # gota de saliva que se acerca a la superficie (vista cenital: se encoge)
                k = progreso(t, llegada - 0.35, llegada)
                pegar(img, G.esfera(34, (150, 200, 255), 150), sx, sy, escala=lerp(2.4, 0.6, k), alfa=0.4 + 0.5 * k)
                continue
            k = progreso(t, llegada, llegada + 0.25)
            if k < 1:
                anillo = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
                r = 20 + 55 * k
                ImageDraw.Draw(anillo).ellipse([80 - r, 80 - r, 80 + r, 80 + r], outline=(200, 230, 255, int(200 * (1 - k))), width=4)
                pegar(img, anillo, sx, sy)
            brillo = 1 + 0.15 * pulso
            pegar(img, G.esfera(19, colores[c]), sx, sy, escala=salida_rebote(k) * brillo)
        return img

    def escena_pelicula(self, t, b):
        t_v = self.t(("velcro", 3, -0.05))
        if t_v <= t < t_v + 0.5:
            medio = self.medios.get("velcro")
            img = medio.fotograma(t - t_v, 0.5) if medio else G.textura_velcro().copy()
            return img
        return self.vista_pelicula(t)

    # ======================================================== ESCENA 4
    def _preparar_escena4(self):
        rng = aleatorio(401)
        aterrizajes = [11.35, 11.9, 12.5, 13.15, 13.8, 12.2, 12.85, 13.5, 14.2]
        xs = [190, 640, 400, 860, 270, 760, 520, 120, 950]
        self.cadenas = []
        for i, (ta, x) in enumerate(zip(aterrizajes, xs)):
            n = int(rng.integers(3, 6))
            self.cadenas.append({"x": x, "llegada": ta, "n": n, "semilla": 400 + i, "fase": rng.uniform(0, 6)})
        self.superficie = 2160

    def camara_lateral(self, t):
        return lerp(250, 700, suave(progreso(t, 10.0, 12.6)))

    def vista_lateral(self, t, cam_y):
        mundo = G.mundo_lateral()
        return mundo.crop((0, int(cam_y), W, int(cam_y) + H))

    def dibujar_cadenas(self, img, t, cam_y, alfa_paracaidas=True):
        aterrizadas = 0
        r = 22
        for c in self.cadenas:
            spr = G.cadena_cocos(c["n"], r, c["semilla"])
            y_fin = self.superficie - 22 - spr.height / 2 + 6
            inicio = c["llegada"] - 1.4
            if t < inicio:
                continue
            k = salida_cubica(progreso(t, inicio, c["llegada"]))
            y = lerp(cam_y - 260, y_fin, k)
            x = c["x"] + math.sin(t * 2.2 + c["fase"]) * 26 * (1 - k)
            if alfa_paracaidas and k < 1:
                alfa = 1 - progreso(t, c["llegada"] - 0.15, c["llegada"])
                pegar(img, G.paracaidas(spr.width + 30), x, y - cam_y - spr.height / 2 - 70, alfa=alfa)
            pegar(img, spr, x, y - cam_y)
            if t >= c["llegada"]:
                aterrizadas += c["n"]
        return aterrizadas

    def escena_pioneros(self, t, b):
        cam_y = self.camara_lateral(t)
        img = self.vista_lateral(t, cam_y).copy()
        n = self.dibujar_cadenas(img, t, cam_y)
        # contador visual de bacterias adheridas
        contador = render_lineas([[("Pioneras adheridas: ", P["blanco"]), (str(n), P["menta"])]], "dato_negrita", 38, contorno=3, color_contorno=P["tinta"])
        pegar(img, contador, 50 + contador.width / 2, 545, alfa=progreso(t, 10.6, 10.9))
        # mini etiqueta S. mitis / S. oralis apuntando a una cadena aterrizada
        a = progreso(t, 12.0, 12.3)
        if a > 0:
            c = self.cadenas[2]
            ex, ey = c["x"], self.superficie - cam_y - 60
            etiqueta = render_lineas(["S. mitis / S. oralis"], "dato_negrita", 34, color=P["azul"])
            etiqueta = _pildora(etiqueta, P["menta"] + (240,))
            capa = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(capa).line([(ex, ey), (ex + 60, ey - 200)], fill=(255, 255, 255, 230), width=4)
            img.alpha_composite(con_alfa(capa, a))
            pegar(img, etiqueta, ex + 60 + etiqueta.width / 2 - 30, ey - 230, alfa=a)
        return img

    # ======================================================== ESCENA 5
    def _preparar_escena5(self):
        rng = aleatorio(501)
        self.colonias = []
        base_y = self.superficie - 22
        defs = [(170, 230, 300, 0.75), (420, 300, 420, 1.0), (650, 240, 330, 0.8), (870, 280, 390, 1.0), (1010, 180, 240, 0.7)]
        for cx, ancho, alto, z in defs:
            r = int(15 * z + 3)
            pts = []
            paso = r * 1.75
            fila = 0
            y = base_y - r
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
            tonos = [mezclar_color(P["turquesa"], (120, 200, 190), rng.uniform(0, 0.6)) for _ in pts]
            self.colonias.append({"cx": cx, "z": z, "r": r, "pts": pts, "tonos": tonos, "alto": alto})
        self.cubos = []
        for i in range(10):
            self.cubos.append({"col": i % 5, "dx": rng.uniform(-60, 60), "rel": i * 0.11 + rng.uniform(0, 0.05), "ang": rng.uniform(0, 360), "giro": rng.uniform(-200, 200)})

    def vista_colonias(self, t):
        cam_y = 700
        img = self.vista_lateral(t, cam_y).copy()
        orbita = math.sin(progreso(t, 15.0, 20.0) * math.pi / 2) * 70
        crecimiento = 0.3 + 0.7 * suave(progreso(t, 15.0, 17.2))
        for col in sorted(self.colonias, key=lambda c: c["z"]):
            desplaz = orbita * (1.0 if col["z"] > 0.9 else -0.6)
            n = int(len(col["pts"]) * crecimiento)
            for (x, y), tono in zip(col["pts"][:n], col["tonos"][:n]):
                pegar(img, G.esfera(col["r"], tono), x + desplaz, y - cam_y)
        t_az = self.t(("azúcar", 5, -0.45))
        for cubo in self.cubos:
            col = self.colonias[cubo["col"]]
            desplaz = orbita * (1.0 if col["z"] > 0.9 else -0.6)
            t0 = t_az + cubo["rel"]
            llegada = t0 + 0.7
            if t < t0:
                continue
            x = col["cx"] + cubo["dx"] + desplaz
            y_cima = self.superficie - 22 - col["alto"] * crecimiento * 0.9 - cam_y
            if t < llegada:
                y = lerp(-80, y_cima, entrada_cubica(progreso(t, t0, llegada)))
                ang = int((cubo["ang"] + cubo["giro"] * (t - t0)) // 15 * 15) % 360
                pegar(img, G.cubo_azucar(64, ang), x, y)
            else:
                k = progreso(t, llegada, llegada + 0.35)
                pegar(img, G.cubo_azucar(64, int(cubo["ang"] // 15 * 15) % 360), x, y_cima, escala=1 - 0.6 * k, alfa=1 - k)
                # niebla ámbar (ácido) que sale de la colonia
                for j in range(3):
                    edad = t - llegada - j * 0.18
                    if edad <= 0:
                        continue
                    ke = limitar(edad / 1.6)
                    pegar(
                        img,
                        disco_suave(int(40 + 70 * ke), P["ambar"], 0.15, int(150 * (1 - ke))),
                        x + math.sin(edad * 3 + j) * 30,
                        y_cima - 40 - 260 * ke,
                    )
        return img

    def vista_ph(self, t, t_inicio):
        img = gradiente_radial(W, H, (W / 2, 860), 1100, (26, 66, 96), P["azul"])
        valor = lerp(7.0, 5.5, suave(progreso(t, t_inicio + 0.15, t_inicio + 1.35)))
        pegar(img, G.medidor_ph(valor, alto=620, ancho=150), 470, 870)
        etiqueta = render_lineas([[("pH ", P["blanco"]), (f"{valor:.1f}".replace(".", ","), G.color_ph(valor))]], "titulo", 110, contorno=6, color_contorno=P["tinta"])
        pegar(img, etiqueta, 760, 650)
        return img

    def escena_colonias(self, t, b):
        t_whip = self.t(("convierten", 5, -0.2))
        dur = 0.3
        if t < t_whip:
            return self.vista_colonias(t)
        if t >= t_whip + dur:
            return self.vista_ph(t, t_whip + dur)
        # Whip pan: la colonia sale hacia la izquierda y entra el medidor de pH, con desenfoque.
        k = suave(progreso(t, t_whip, t_whip + dur))
        a = self.vista_colonias(t_whip)
        bimg = self.vista_ph(t, t_whip + dur)
        lienzo = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        dx = int(k * W)
        if dx < W:
            lienzo.alpha_composite(a.crop((dx, 0, W, H)), (0, 0))
        if dx > 0:
            lienzo.alpha_composite(bimg.crop((0, 0, dx, H)), (W - dx, 0))
        return desenfoque_movimiento(lienzo, 160 * math.sin(k * math.pi) + 1)

    # ======================================================== ESCENA 6
    def escena_dato_clave(self, t, b):
        lt = t - b["inicio"]
        img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        izq = gradiente_vertical(W // 2, H, (18, 70, 86), (10, 36, 58))
        der = gradiente_vertical(W // 2, H, (70, 50, 40), (34, 26, 40))
        img.alpha_composite(izq, (0, 0))
        img.alpha_composite(der, (W // 2, 0))
        sprite, _ = G.diente(250, 350)
        placa = G.placa(250, 350)
        # mitad izquierda: diente limpio + azúcar, pH estable
        pegar(img, sprite, 190, 830)
        # mitad derecha: diente con placa + azúcar, pH que cae
        mitad_der = Image.new("RGBA", (W // 2, H), (0, 0, 0, 0))
        mitad_der.alpha_composite(sprite, (190 - 125, 830 - 175))
        mitad_der.alpha_composite(con_alfa(placa, 0.95), (190 - 125, 830 - 175))
        for j in range(3):
            edad = (lt - 0.6 - j * 0.5) % 1.8
            ke = edad / 1.8
            pegar(mitad_der, disco_suave(int(30 + 40 * ke), P["ambar"], 0.15, int(130 * (1 - ke))), 190 + math.sin(edad * 3 + j) * 25, 700 - 200 * ke)
        for k, (dx, dy) in enumerate(((-60, -150), (10, -160), (70, -140))):
            ang = (k * 35) % 360 // 15 * 15
            bob = math.sin(t * 2 + k) * 4
            pegar(img, G.cubo_azucar(46, ang), 190 + dx, 830 + dy + bob)
            pegar(mitad_der, G.cubo_azucar(46, ang), 190 + dx, 830 + dy + bob)
        v_izq = lerp(7.0, 6.9, suave(progreso(lt, 0.6, 4.0)))
        v_der = lerp(7.0, 5.5, suave(progreso(lt, 0.6, 3.8)))
        pegar(img, G.medidor_ph(v_izq, alto=380, ancho=90, etiqueta_umbral=False), 420, 830)
        pegar(mitad_der, G.medidor_ph(v_der, alto=380, ancho=90, etiqueta_umbral=False), 420, 830)
        for capa, x, v in ((img, 270, v_izq), (mitad_der, 270, v_der)):
            lectura = render_lineas([[("pH ", P["blanco"]), (f"{v:.1f}".replace(".", ","), G.color_ph(v))]], "titulo", 64, contorno=5, color_contorno=P["tinta"])
            pegar(capa, lectura, x, 1075)
        t_ellas = self.t(("ellas", 6, -0.1))
        z = 1 + 0.14 * suave(progreso(t, t_ellas, t_ellas + 0.9))
        if z > 1.001:
            mitad_der = vista_zoom(mitad_der, z, (270, 850), (270, 850), (W // 2, H))
        img.alpha_composite(mitad_der, (W // 2, 0))
        tit_i = sprite_etiqueta("SIN PLACA:\npH 7 ✓", P["menta"])
        tit_d = sprite_etiqueta("CON PLACA:\npH ↓", P["ambar"])
        a = progreso(lt, 0.3, 0.55)
        pegar(img, tit_i, 270, 560, alfa=a, escala=0.7 + 0.3 * salida_rebote(a))
        pegar(img, tit_d, 810, 560, alfa=a, escala=0.7 + 0.3 * salida_rebote(a))
        # barrido vertical que divide la pantalla
        corte_y = H * salida_cubica(progreso(lt, 0, 0.35))
        d = ImageDraw.Draw(img)
        d.rectangle([W / 2 - 4, 0, W / 2 + 4, corte_y], fill=(255, 255, 255, 255))
        if lt < 0.35:
            previo = self.vista_ph(b["inicio"] - 1 / FPS, self.t(("convierten", 5, -0.2)) + 0.3)
            mascara = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mascara).rectangle([0, corte_y, W, H], fill=255)
            img.paste(previo, (0, 0), mascara)
        # fundido rápido a oscuro para entrar en la noche
        oscuro = progreso(t, b["fin"] - 0.15, b["fin"])
        if oscuro > 0:
            img.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * oscuro))))
        return img

    # ======================================================== ESCENA 7
    def escena_noche(self, t, b):
        t_buenos = self.t(("Buenos", 7, 0.0))
        if t >= t_buenos:
            medio = self.medios.get("persona_despierta")
            if medio:
                return medio.fotograma(t - t_buenos, b["fin"] - t_buenos)
            img = G.dormitorio().copy()
            sacudida = math.sin((t - t_buenos) * 60) * 10 * (1 - progreso(t, t_buenos, t_buenos + 0.35))
            return vista_zoom(img, 1.0, (W / 2 + sacudida, H / 2))
        lt = t - b["inicio"]
        img = gradiente_vertical(W, H, (4, 10, 28), P["azul"])
        pegar(img, disco_suave(420, (90, 130, 220), 0.0, 90), 180, 260)
        lengua = G.textura_lengua()
        k = progreso(lt, 0, 6)
        pegar(img, lengua, W / 2 + 30 * k, 1250 - 60 * k, escala=1.0 + 0.1 * k)
        # «río» de saliva que se reduce a un hilo
        sub = 2
        capa = Image.new("RGBA", (W // sub, H // sub), (0, 0, 0, 0))
        d = ImageDraw.Draw(capa)
        ancho = lerp(70, 5, suave(progreso(t, 25.4, 28.2)))
        pts = [((x) / sub, (1180 - 60 * k + 55 * math.sin(x / 170 + 0.6)) / sub) for x in range(-40, W + 60, 30)]
        d.line(pts, fill=(110, 170, 255, 175), width=max(1, int(ancho / sub)), joint="curve")
        velocidad = lerp(1.0, 0.15, progreso(t, 25.4, 28.5))
        desfase = (t * 240 * velocidad) % 90
        for i in range(len(pts) - 1):
            if ((i * 30 + desfase) % 90) < 30 and ancho > 8:
                d.line([pts[i], pts[i + 1]], fill=(220, 240, 255, 200), width=max(1, int(ancho / sub / 4)))
        img.alpha_composite(capa.resize((W, H), Image.BILINEAR))
        # volutas verdes de compuestos sulfurados volátiles que suben de la lengua
        t_vol = self.t(("bacterias", 7, -0.3))
        t_h2s = self.t(("azufre", 7, -0.25))
        if t > t_vol:
            sub = 3
            capa = Image.new("RGBA", (W // sub, H // sub), (0, 0, 0, 0))
            d = ImageDraw.Draw(capa)
            convergencia = suave(progreso(t, t_h2s - 0.3, t_h2s + 0.2)) * (1 - progreso(t, t_h2s + 1.0, t_h2s + 1.4))
            for i in range(8):
                x0 = 120 + i * 120
                subida = progreso(t, t_vol + i * 0.08, t_vol + i * 0.08 + 1.4)
                puntos = []
                for s in range(14):
                    f = s / 13
                    y = 1150 - 620 * f * subida
                    x = x0 + 50 * math.sin(f * 5 + t * 2.5 + i)
                    x = lerp(x, W / 2 + (x0 - W / 2) * 0.15, convergencia * f)
                    y = lerp(y, 780 + (y - 780) * 0.3, convergencia * f)
                    puntos.append((x / sub, y / sub))
                d.line(puntos, fill=(150, 225, 120, int(130 * subida)), width=10 // sub + 3, joint="curve")
            img.alpha_composite(capa.resize((W, H), Image.BILINEAR))
            a_txt = progreso(t, t_h2s, t_h2s + 0.3) * (1 - progreso(t, t_h2s + 1.1, t_h2s + 1.4))
            if a_txt > 0:
                pegar(img, sprite_h2s(), W / 2, 780, alfa=a_txt, escala=0.8 + 0.2 * salida_cubica(a_txt))
        # entrada desde negro (segunda mitad del fundido de 0,3 s)
        claro = progreso(lt, 0, 0.15)
        if claro < 1:
            img.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (1 - claro)))))
        return img

    # ======================================================== ESCENA 8
    def escena_revelacion(self, t, b):
        t_caries = self.t(("caries", 8, 0.0))
        t_no = self.t(("No", 8, -0.08))
        if t < t_no:
            congelado = t_caries <= t < t_caries + 0.2
            tt = t_caries if congelado or t > t_caries else t
            escala = 1.0 + 0.25 * suave(progreso(tt, 31.0, t_caries))
            densidad = 0.75 + 0.2 * progreso(tt, 31.0, 32.5)
            img = self.macro_diente(escala * (1.03 if congelado else 1.0), densidad_placa=densidad, t=tt)
            if t >= t_caries:
                flash = 1 - progreso(t, t_caries, t_caries + 0.14)
                if flash > 0:
                    img.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(200 * flash))))
            return img
        img = self.dentista(t, "dentista_no", t_no, sonrisa=0.35, duracion_uso=b["fin"] - t_no)
        t_tampoco = self.t(("Tampoco", 8, 0.0))
        if t >= t_tampoco:
            img.alpha_composite(linea_de_tiempo(progreso(t, t_tampoco, t_tampoco + 1.3)), (0, 0))
        return img

    # ======================================================== ESCENA 9
    def escena_conclusion(self, t, b):
        t_todo = self.t(("todo", 9, 0.0))
        cambio = t_todo - 0.25
        if t >= 42.6:
            return self.transicion_cepillo(t)
        if t < cambio:
            return self.vista_calendario(t, b)
        brush = self.vista_cepillado(t, t_todo)
        if t < cambio + 0.25:
            previo = self.vista_calendario(t, b)
            return Image.blend(previo, brush, progreso(t, cambio, cambio + 0.25))
        return brush

    def vista_calendario(self, t, b):
        img = gradiente_radial(W, H, (W / 2, 900), 1200, (30, 76, 104), P["azul"])
        paso = 0.2
        dia_f = (t - 37.2) / paso
        dia = int(limitar(dia_f, 0, 28)) + 1
        hoja = G.hoja_calendario(dia)
        pegar(img, hoja, 290, 930)
        if 0 < dia_f < 28:
            f = dia_f % 1
            if f < 0.5:
                anterior = G.hoja_calendario(max(1, dia - 1))
                k = f / 0.5
                alto = max(1, int(anterior.height * (1 - k)))
                pegar(img, anterior.resize((anterior.width, alto)), 290, 930 - anterior.height / 2 + alto / 2 - 80 * k, alfa=1 - k)
        sprite, _ = G.diente(360, 504)
        capa = sprite.copy()
        capa.alpha_composite(con_alfa(G.placa(360, 504), limitar(dia / 14)))
        pegar(img, capa, 780, 960)
        return img

    def vista_cepillado(self, t, t_todo):
        img = gradiente_radial(W, H, (W / 2, 900), 1200, (40, 110, 120), P["azul"])
        sprite, _ = G.diente(640, 896)
        placa = G.placa(640, 896)
        cx, cy = W / 2, 930
        izq = cx - 320
        pegar(img, sprite, cx, cy)
        k = suave(progreso(t, t_todo, t_todo + 1.9))  # barrido en cámara lenta
        cabeza_x = lerp(-150, W + 250, k)
        corte = int(limitar(cabeza_x - 40 - izq, 0, 640))
        if corte < 640:
            pegar(img, placa.crop((corte, 0, 640, 896)), izq + corte, cy - 448, ancla="esquina")
        t_cep = self.t(("cepillado", 9, 0.35))
        if t > t_cep:
            for j, (bx, by) in enumerate(((0.32, 0.4), (0.6, 0.52), (0.45, 0.75), (0.7, 0.3))):
                fase = progreso(t, t_cep + j * 0.12, t_cep + j * 0.12 + 0.7)
                pegar(img, G.destello(110), izq + bx * 640, cy - 448 + by * 896, alfa=math.sin(fase * math.pi))
        cep = G.cepillo()
        pegar(img, cep, cabeza_x - cep.width / 2 + 120, cy - 30)
        rotulo_a = progreso(t, t_todo + 0.2, t_todo + 0.45) * (1 - progreso(t, t_todo + 1.6, t_todo + 1.9))
        if rotulo_a > 0:
            r = _pildora(render_lineas(["pasta con flúor"], "dato_negrita", 36, color=P["azul"]), P["menta"] + (240,))
            pegar(img, r, cabeza_x - 60, cy - 260, alfa=rotulo_a)
        return img

    def transicion_cepillo(self, t):
        """El barrido del cepillo funciona como wipe hacia la escena final."""
        t_todo = self.t(("todo", 9, 0.0))
        k = progreso(t, 42.6, 43.0)
        fondo = self.vista_cepillado(t, t_todo)
        final = self.escena_cierre(43.0, guion.BLOQUES[-1])
        frontera = int(lerp(-200, W + 200, suave(k)))
        if frontera > 0:
            fondo.paste(final.crop((0, 0, min(W, frontera), H)), (0, 0))
        cep = G.cepillo()
        pegar(fondo, cep, frontera - cep.width * 1.1 + 280, H / 2, escala=2.2)
        return fondo

    # ======================================================== ESCENA 10
    def escena_cierre(self, t, b):
        if t >= guion.CORTE_BUCLE:
            # Bucle: mismo diente brillante del inicio, reloj en 00:00.
            return self.macro_diente(1.0, t=0.0)
        img = self.dentista(t, "dentista_cierre", b["inicio"], sonrisa=0.85, duracion_uso=guion.CORTE_BUCLE - b["inicio"])
        colores = [(255, 176, 32), (63, 209, 181)]
        posiciones = [(310, 1000), (760, 1430)]
        for (texto, t0), color, (x, y) in zip(guion.COMENTARIOS_EJEMPLO, colores, posiciones):
            a = progreso(t, t0, t0 + 0.25)
            if a > 0:
                pegar(img, sprite_comentario(texto, color), x, y, escala=0.5 + 0.5 * salida_rebote(a), alfa=limitar(a * 2))
        logo = self.logo or G.logo_alternativo()
        a = progreso(t, 43.3, 43.7)
        pegar(img, logo, 60 + logo.width * 0.4, 1480, escala=0.8, alfa=0.9 * a)
        marca = render_lineas([config.MARCA["usuario"]], "dato_negrita", 34, contorno=3, color_contorno=P["tinta"])
        pegar(img, marca, 60 + logo.width * 0.8 + 20 + marca.width / 2, 1480, alfa=0.9 * a)
        return img


# ---------------------------------------------------------------------------
# Sprites auxiliares con caché
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def marco_espejo():
    """Marco del espejo del consultorio con reflejos diagonales."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(40):
        c = int(lerp(225, 150, i / 40))
        d.rounded_rectangle([i, i, W - 1 - i, H - 1 - i], radius=90 - i, outline=(c, c + 5, c + 10, 255), width=1)
    reflejo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(reflejo)
    dr.polygon([(80, 0), (260, 0), (-100, 700), (-280, 700)], fill=(255, 255, 255, 34))
    dr.polygon([(330, 0), (390, 0), (30, 700), (-30, 700)], fill=(255, 255, 255, 26))
    img.alpha_composite(reflejo)
    return img


@functools.lru_cache(maxsize=4)
def sprite_etiqueta(texto, color):
    return render_lineas(texto.split("\n"), "titulo", 50, color=color, contorno=5, color_contorno=P["tinta"])


@functools.lru_cache(maxsize=1)
def sprite_h2s():
    t = render_lineas(["H₂S"], "titulo", 170, color=(175, 240, 140), contorno=6, color_contorno=(20, 60, 30))
    glow = disco_suave(int(t.width * 0.55), (150, 225, 120), 0.0, 90)
    lienzo = Image.new("RGBA", (glow.width, glow.width), (0, 0, 0, 0))
    lienzo.alpha_composite(glow)
    pegar(lienzo, t, lienzo.width / 2, lienzo.height / 2)
    return lienzo


@functools.lru_cache(maxsize=4)
def sprite_comentario(texto, color):
    return G.burbuja_comentario(texto, color)


def linea_de_tiempo(k):
    """Escala de días: hoy (24 h) frente a la gingivitis (día 10–21)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = 1010
    d.rounded_rectangle([50, y - 75, W - 50, y + 85], radius=40, fill=(14, 42, 71, int(215 * min(1, k * 3))))
    x0, x1 = 110, 970

    def x_dia(dd):
        return x0 + (x1 - x0) * dd / 21

    largo = x0 + (x1 - x0) * salida_cubica(k)
    d.line([(x0, y), (largo, y)], fill=(255, 255, 255, 255), width=8)
    if largo > x_dia(10):
        d.line([(x_dia(10), y), (min(largo, x_dia(21)), y)], fill=P["coral"] + (255,), width=16)
    d.ellipse([x_dia(1) - 14, y - 14, x_dia(1) + 14, y + 14], fill=P["menta"] + (255,))
    etiquetas = [(1, "hoy · 24 h", P["menta"]), (10, "día 10", P["coral"]), (21, "día 21", P["coral"])]
    for dd, txt, col in etiquetas:
        if largo >= x_dia(dd) - 1:
            s = render_lineas([txt], "dato_negrita", 30, color=col)
            img.alpha_composite(s, (int(x_dia(dd) - s.width / 2), y + 18))
            d.line([(x_dia(dd), y - 18), (x_dia(dd), y + 18)], fill=col + (255,), width=5)
    if k > 0.7:
        s = render_lineas(["gingivitis"], "dato_negrita", 32, color=P["coral"])
        img.alpha_composite(con_alfa(s, progreso(k, 0.7, 1.0)), (int((x_dia(10) + x_dia(21)) / 2 - s.width / 2), y - 66))
    return img
