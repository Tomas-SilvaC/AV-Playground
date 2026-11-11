"""
Zomboid Asset Warper + Normal Map  (aplicacion de escritorio) - v2

Basado en ZomboidNormalMaper.cpp:
  1) Carga un tile de pared y le aplica el warp de perspectiva.
  2) A partir de la imagen warpeada genera un mapa de normales.
  3) Guarda el resultado donde el usuario elija (NOMBRE.png y NOMBRE_n.png).

Novedades v2: puntos arrastrables sobre la imagen, zoom con rueda, vista con
transparencia, comparacion (divisor), luz interactiva, presets, atajos de
teclado, deshacer/rehacer y tema oscuro.

Requisitos:  pip install PySide6 opencv-python numpy
Ejecutar:    python ZomboidNormalMapper.py

=============================================================================
CONCEPTOS DE VISION ARTIFICIAL QUE USA ESTE PROGRAMA
=============================================================================
* Transformacion de perspectiva (homografia): 4 puntos de origen -> 4 esquinas de
  un rectangulo. cv2.getPerspectiveTransform calcula la matriz 3x3 y
  cv2.warpPerspective la aplica (coordenadas homogeneas; la division por w es lo
  que produce el efecto de perspectiva). Endereza una pared dibujada en isometrico.
* Mapa de normales: cada pixel guarda un vector normal (x, y, z) codificado como
  color (v * 127.5 + 127.5). Aqui el relieve se simula usando la luminosidad como
  mapa de alturas y midiendo su pendiente con filtros espaciales:
    - GaussianBlur : filtro paso bajo para quitar ruido del pixel art.
    - Sobel        : convolucion 3x3 que estima el gradiente en X e Y (bordes).
    - CLAHE        : ecualizacion adaptativa de histograma con limite de contraste.
* Umbralizacion: el canal alfa se binariza (>= 128) para separar objeto y fondo.
* Modelo de iluminacion difusa (Lambert): brillo ~ max(0, N . L), usado en la
  vista "Luz" para comprobar el relieve.

Estructura del archivo:
  1. Constantes y utilidades de recursos.
  2. Logica de imagen pura (sin interfaz).
  3. Utilidades de vista (conversion a QImage, tema).
  4. ImageView: widget de imagen con zoom, pan, puntos arrastrables y comparacion.
  5. SliderConValor: control auxiliar.
  6. Ventana: ventana principal (UI, estado, deshacer, presets, guardado).
"""

import json
import os
import sys

import cv2
import numpy as np
from PySide6.QtCore import (QPointF, QRectF, QSettings, QStandardPaths, Qt,
                            QTimer, Signal)
from PySide6.QtGui import (QAction, QBrush, QColor, QIcon, QImage,
                           QKeySequence, QPainter, QPalette, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QVBoxLayout,
    QWidget,
)

# ----------------------------------------------------------------------------
# Valores por defecto (los mismos del programa en C++)
# ----------------------------------------------------------------------------
DEFAULT_SRC = [(60, 2), (123, 33), (60, 194), (123, 225)]  # TL, TR, BL, BR
DEFAULT_SIZE = (128, 256)  # ancho, alto de salida
POINT_NAMES = ["Arriba-izquierda", "Arriba-derecha", "Abajo-izquierda", "Abajo-derecha"]

# Estado completo que se guarda/restaura (deshacer, presets, sesion anterior).
DEFAULT_ESTADO = {
    "warp": True,
    "puntos": [list(p) for p in DEFAULT_SRC],
    "ancho": DEFAULT_SIZE[0],
    "alto": DEFAULT_SIZE[1],
    "tam_auto": False,   # True = tamano de salida calculado desde los 4 puntos
    "strength": 30,      # valor / 10
    "blur": 1,
    "contraste": 0,      # valor / 10
    "invertir": False,
    "directx": False,
    "fondo_plano": False,
}
# Claves que pertenecen al grupo "normales" (para restaurar o aplicar presets parciales).
CLAVES_NORMALES = ["strength", "blur", "contraste", "invertir", "directx", "fondo_plano"]


def ruta_recurso(nombre):
    """Devuelve la ruta de un archivo junto al script (o dentro del .exe de PyInstaller).

    PyInstaller descomprime los recursos en sys._MEIPASS; si no existe, se usa la
    carpeta del propio script.
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, nombre)


# ----------------------------------------------------------------------------
# Logica de imagen (sin interfaz, reutilizable)
# ----------------------------------------------------------------------------
def leer_imagen(ruta):
    """Lee una imagen (soporta rutas unicode en Windows) y la devuelve en BGRA.

    Se usa np.fromfile + imdecode porque cv2.imread falla con rutas no ASCII en
    Windows. IMREAD_UNCHANGED conserva el canal alfa. Si la imagen no es de 8 bits
    (p. ej. 16 bits) se normaliza a [0, 255]; las imagenes de 1 o 3 canales se
    convierten a 4 para trabajar siempre con el mismo formato.

    Raises:
        ValueError: si el archivo no se puede decodificar como imagen.
    """
    datos = np.fromfile(ruta, dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("No se pudo abrir la imagen: " + ruta)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def guardar_imagen(ruta, img):
    """Guarda una imagen PNG (soporta rutas unicode). Lanza excepcion si falla.

    Usa imencode + tofile por la misma razon que leer_imagen (rutas unicode).
    """
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError("No se pudo codificar la imagen")
    buf.tofile(ruta)


def tamano_auto(puntos):
    """Tamano en pixeles del area marcada por los 4 puntos (TL, TR, BL, BR).

    Ancho = promedio de los bordes superior e inferior; alto = promedio de los
    bordes izquierdo y derecho (distancias euclidianas). Asi la imagen enderezada
    conserva aproximadamente la resolucion original, sin estirarse ni encogerse.

    Returns:
        (ancho, alto) en pixeles, minimo 1.
    """
    (tl, tr, bl, br) = [np.array(p, dtype=np.float64) for p in puntos]
    ancho = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    alto = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return max(1, int(round(ancho))), max(1, int(round(alto)))


def aplicar_warp(img_bgra, src_pts, size):
    """Warp de perspectiva: los 4 puntos de origen pasan a las esquinas de un rectangulo.

    Args:
        img_bgra: imagen de entrada (BGRA).
        src_pts:  4 puntos (x, y) en orden TL, TR, BL, BR.
        size:     (ancho, alto) del rectangulo de salida.

    Calcula la homografia 3x3 con getPerspectiveTransform y la aplica con
    warpPerspective (interpolacion bilineal por defecto).
    """
    w, h = size
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgra, m, (int(w), int(h)))


def crear_mapa_normales(bgra, strength, blur_radius, invert, flip_green,
                        contrast_clip, fondo_plano):
    """Devuelve un mapa de normales BGRA a partir de una imagen BGRA.

    El canal alfa de la entrada se conserva tal cual (incluidos los semitransparentes).

    Pipeline:
        gris -> (inversion) -> relleno del fondo con la media -> CLAHE (opcional)
        -> alturas en [0,1] -> Gaussiano (opcional) -> Sobel dx, dy
        -> normal (nx, ny, nz) normalizada -> codificacion a color.

    Args:
        bgra:          imagen ya warpeada (BGRA).
        strength:      intensidad del relieve (multiplica el gradiente).
        blur_radius:   radio del Gaussiano previo (kernel = 2*radio + 1); 0 = sin blur.
        invert:        invierte claros/oscuros (oscuro = alto).
        flip_green:    True = formato DirectX (canal verde invertido); False = OpenGL.
        contrast_clip: limite de CLAHE; 0 = apagado.
        fondo_plano:   True = el fondo transparente conserva normal plana;
                       False = el fondo totalmente transparente queda negro.
    """
    bgr = bgra[:, :, :3]
    alpha = bgra[:, :, 3]

    # La luminosidad hace de mapa de alturas.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)

    # Umbralizacion del alfa: >= 128 es objeto. El fondo se rellena con la altura
    # media del objeto para que el contorno del sprite no genere un relieve falso.
    mask = (alpha >= 128).astype(np.uint8) * 255
    mask_inv = cv2.bitwise_not(mask)
    media = cv2.mean(gray, mask=mask)[0] if cv2.countNonZero(mask) > 0 else 128
    gray[mask_inv > 0] = int(round(media))

    # CLAHE: ecualizacion adaptativa de histograma por regiones (tiles de 4x4).
    if contrast_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

    height = gray.astype(np.float32) / 255.0

    # Suavizado paso bajo: reduce el ruido del pixel art.
    if blur_radius > 0:
        k = blur_radius * 2 + 1
        height = cv2.GaussianBlur(height, (k, k), 0)

    # Gradientes con Sobel 3x3. scale = 1/8 normaliza el kernel (pesos que suman 8)
    # para obtener la pendiente real; BORDER_REPLICATE evita falsos bordes.
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0,
                   borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0,
                   borderType=cv2.BORDER_REPLICATE)

    # Vector normal n = (-s*dx, +-s*dy, 1) / |n|. En imagen Y crece hacia abajo,
    # por eso el signo de ny depende del formato (OpenGL vs DirectX).
    nx = dx * (-strength)
    ny = dy * (-strength if flip_green else strength)
    nz = np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + 1.0)
    nx, ny, nz = nx / length, ny / length, nz / length

    def a8(v):
        """Codifica [-1, 1] -> [0, 255] (v * 127.5 + 127.5)."""
        return np.clip(v * 127.5 + 127.5, 0, 255).astype(np.uint8)

    normal = np.dstack([a8(nz), a8(ny), a8(nx)])  # BGR: B = z, G = y, R = x
    normal[mask_inv > 0] = (255, 128, 128)        # normal plana fuera del objeto
    if not fondo_plano:
        normal[alpha == 0] = (0, 0, 0)

    return np.dstack([normal, alpha])


def iluminar(warp, normal, flip_green, luz_xy, luz_z):
    """Previsualiza el relieve con una luz puntual en luz_xy (coordenadas de imagen).

    Modelo de iluminacion difusa (Lambert): para cada pixel se calcula el vector
    hacia la luz L y se usa el producto punto N . L (recortado a [0, 1]) como
    factor de brillo, con un minimo ambiental de 0.30.

    Args:
        warp:       imagen warpeada (BGRA), da el color base.
        normal:     mapa de normales (BGRA).
        flip_green: si el mapa esta en formato DirectX, se invierte ny al leerlo.
        luz_xy:     posicion (x, y) de la luz en coordenadas de imagen.
        luz_z:      altura de la luz sobre la imagen (mas alta = luz mas uniforme).
    """
    h, w = warp.shape[:2]
    # Decodificar el color a vector normal: BGR -> (x, y, z) en [-1, 1].
    n = normal[:, :, 2::-1].astype(np.float32) / 127.5 - 1.0
    nx, ny, nz = n[:, :, 0], n[:, :, 1], n[:, :, 2]
    if flip_green:
        ny = -ny
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    escala = max(w, h) / 2.0
    lx = (luz_xy[0] - xx) / escala
    ly = (yy - luz_xy[1]) / escala          # +Y hacia arriba
    lz = np.full_like(lx, luz_z)
    ln = np.sqrt(lx * lx + ly * ly + lz * lz)
    dif = np.clip((nx * lx + ny * ly + nz * lz) / ln, 0, 1)   # N . L normalizado
    brillo = (0.30 + 0.90 * dif)[:, :, None]
    bgr = np.clip(warp[:, :, :3].astype(np.float32) * brillo, 0, 255).astype(np.uint8)
    return np.dstack([bgr, warp[:, :, 3]])


# ----------------------------------------------------------------------------
# Utilidades de vista
# ----------------------------------------------------------------------------
def bgra_a_qimage(img, solo_rgb=False):
    """numpy BGR/BGRA -> QImage RGBA (con transparencia). solo_rgb ignora el alfa.

    OpenCV usa orden BGR(A) y Qt espera RGBA, por eso se convierte. El .copy()
    final desliga la QImage del buffer de numpy (que podria liberarse).
    """
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    if solo_rgb:
        rgba[:, :, 3] = 255
    rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()


def aplicar_tema(app, oscuro):
    """Aplica el tema oscuro (paleta Fusion personalizada) o el estandar claro."""
    app.setStyle("Fusion")
    if not oscuro:
        app.setPalette(app.style().standardPalette())
        return
    p = QPalette()
    gris = QColor(220, 220, 220)
    apagado = QColor(120, 120, 120)
    p.setColor(QPalette.Window, QColor(45, 45, 48))
    p.setColor(QPalette.WindowText, gris)
    p.setColor(QPalette.Base, QColor(30, 30, 30))
    p.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
    p.setColor(QPalette.ToolTipBase, QColor(60, 60, 60))
    p.setColor(QPalette.ToolTipText, gris)
    p.setColor(QPalette.Text, gris)
    p.setColor(QPalette.Button, QColor(58, 58, 62))
    p.setColor(QPalette.ButtonText, gris)
    p.setColor(QPalette.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.Link, QColor(80, 160, 255))
    p.setColor(QPalette.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    for rol in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, rol, apagado)
    app.setPalette(p)


# ----------------------------------------------------------------------------
# Vista de imagen: zoom con rueda, mover, puntos arrastrables, comparacion
# ----------------------------------------------------------------------------
class ImageView(QWidget):
    """Widget que muestra una imagen con zoom, desplazamiento y herramientas de edicion.

    Funciones:
      - Zoom con la rueda (centrado en el cursor) y desplazamiento arrastrando.
      - Fondo configurable (cuadros de transparencia o color solido).
      - Cuadrilatero con 4 puntos arrastrables (los puntos del warp), si 'puntos' esta definido.
      - Modo comparacion: dos imagenes separadas por un divisor movible.

    Maneja dos sistemas de coordenadas: imagen (pixeles de la imagen) y vista
    (pixeles del widget); _i2v y _v2i convierten entre ambos usando zoom y offset.

    Signals:
        puntoMovido(indice, x, y): se arrastro un punto (coordenadas de imagen).
        hoverImagen(x, y): posicion del mouse en coordenadas de imagen.
    """
    puntoMovido = Signal(int, float, float)   # indice, x, y (coordenadas de imagen)
    hoverImagen = Signal(float, float)        # posicion del mouse en coordenadas de imagen

    FONDOS = {"Negro": QColor(0, 0, 0), "Gris": QColor(128, 128, 128),
              "Blanco": QColor(255, 255, 255)}

    def __init__(self, mensaje=""):
        super().__init__()
        self.setMinimumSize(240, 240)
        self.setMouseTracking(True)
        self.mensaje = mensaje    # texto cuando no hay imagen
        self.img = None
        self.img2 = None          # segunda imagen (modo comparacion)
        self.zoom = 1.0
        self.off = QPointF(0, 0)  # desplazamiento de la imagen en la vista
        self.auto_fit = True      # True = ajustar al redimensionar la ventana
        self.fondo = "Cuadros"
        self.puntos = None        # lista de 4 (x, y) o None
        self.comparar = False
        self.split = 0.5          # posicion del divisor (0..1)
        self._drag = None         # estado del arrastre actual
        self._tile = None         # cache del patron de cuadros

    # --- API ---
    def set_images(self, img, img2=None):
        """Establece la(s) imagen(es) a mostrar. Reajusta la vista si cambia el tamano."""
        cambio = self.img is None or self.img.size() != img.size()
        self.img, self.img2 = img, img2
        if cambio:
            self.ajustar()
        else:
            self.update()

    def set_fondo(self, nombre):
        """Cambia el fondo detras de la imagen (Cuadros, Negro, Gris, Blanco)."""
        self.fondo = nombre
        self.update()

    def ajustar(self):
        """Ajusta zoom y posicion para que la imagen quepa centrada en la vista.

        Si se puede ampliar (zoom >= 1) usa zoom entero para que el pixel art se
        vea nitido, sin remuestreo irregular.
        """
        if self.img is None:
            return
        iw, ih = self.img.width(), self.img.height()
        z = min((self.width() - 24) / iw, (self.height() - 24) / ih)
        z = float(int(z)) if z >= 1 else max(z, 0.05)
        self.zoom = max(z, 0.05)
        self.off = QPointF((self.width() - iw * self.zoom) / 2,
                           (self.height() - ih * self.zoom) / 2)
        self.auto_fit = True
        self.update()

    # --- coordenadas ---
    def _i2v(self, x, y):
        """Imagen -> vista."""
        return QPointF(self.off.x() + x * self.zoom, self.off.y() + y * self.zoom)

    def _v2i(self, pos):
        """Vista -> imagen."""
        return QPointF((pos.x() - self.off.x()) / self.zoom,
                       (pos.y() - self.off.y()) / self.zoom)

    def _rect_img(self):
        """Rectangulo que ocupa la imagen en coordenadas de vista."""
        return QRectF(self.off.x(), self.off.y(),
                      self.img.width() * self.zoom, self.img.height() * self.zoom)

    def _hit_punto(self, pos):
        """Indice del punto del warp bajo 'pos' (radio ~14 px de la vista), o None."""
        if not self.puntos:
            return None
        mejor, dmin = None, 14
        for i, (x, y) in enumerate(self.puntos):
            d = (self._i2v(x, y) - pos).manhattanLength()
            if d <= dmin:
                mejor, dmin = i, d
        return mejor

    # --- dibujo ---
    def _pincel_cuadros(self):
        """Pincel con el patron de cuadros de transparencia (se crea una sola vez)."""
        if self._tile is None:
            pm = QPixmap(16, 16)
            pm.fill(QColor(92, 92, 92))
            q = QPainter(pm)
            q.fillRect(0, 0, 8, 8, QColor(132, 132, 132))
            q.fillRect(8, 8, 8, 8, QColor(132, 132, 132))
            q.end()
            self._tile = pm
        return QBrush(self._tile)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().color(QPalette.Base).darker(120))
        if self.img is None:
            p.setPen(self.palette().color(QPalette.Mid))
            p.drawText(self.rect(), Qt.AlignCenter, self.mensaje)
            return
        r = self._rect_img()
        vis = r.intersected(QRectF(self.rect()))   # solo se pinta la parte visible
        if vis.isEmpty():
            return
        if self.fondo == "Cuadros":
            p.fillRect(vis, self._pincel_cuadros())
        else:
            p.fillRect(vis, self.FONDOS.get(self.fondo, QColor(0, 0, 0)))
        # Sin suavizado: al ampliar, los pixeles se ven como bloques (util en pixel art).
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)

        if self.comparar and self.img2 is not None:
            # Izquierda del divisor: img; derecha: img2 (se recorta con setClipRect).
            sx = r.left() + r.width() * self.split
            p.save()
            p.setClipRect(QRectF(r.left(), r.top(), max(sx - r.left(), 0), r.height()))
            p.drawImage(r, self.img)
            p.restore()
            p.save()
            p.setClipRect(QRectF(sx, r.top(), max(r.right() - sx, 0) + 1, r.height()))
            p.drawImage(r, self.img2)
            p.restore()
            # Linea del divisor (contorno oscuro + linea blanca) y su manija circular.
            p.setPen(QPen(QColor(0, 0, 0, 200), 3))
            p.drawLine(QPointF(sx, r.top()), QPointF(sx, r.bottom()))
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawLine(QPointF(sx, r.top()), QPointF(sx, r.bottom()))
            cy = r.center().y()
            p.setBrush(QColor(255, 255, 255))
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.drawEllipse(QPointF(sx, cy), 6, 6)
        else:
            p.drawImage(r, self.img)

        # borde de la imagen
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 60), 1))
        p.drawRect(r)

        # cuadrilatero y puntos del warp
        if self.puntos:
            p.setRenderHint(QPainter.Antialiasing, True)
            orden = [0, 1, 3, 2]   # TL, TR, BR, BL: orden perimetral del poligono
            poly = QPolygonF([self._i2v(*self.puntos[i]) for i in orden])
            p.setPen(QPen(QColor(255, 220, 0, 220), 1.5))
            p.setBrush(QColor(255, 220, 0, 25))
            p.drawPolygon(poly)
            for i, (x, y) in enumerate(self.puntos):
                c = self._i2v(x, y)
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.setBrush(QColor(230, 40, 40))
                p.drawEllipse(c, 7, 7)
                p.setPen(QColor(255, 255, 255))
                p.drawText(QPointF(c.x() + 10, c.y() - 8), str(i + 1))

    # --- interaccion ---
    def resizeEvent(self, _):
        if self.auto_fit:
            self.ajustar()

    def wheelEvent(self, e):
        """Zoom hacia el cursor: el punto de la imagen bajo el mouse permanece fijo."""
        if self.img is None:
            return
        factor = 1.25 if e.angleDelta().y() > 0 else 1 / 1.25
        pos = e.position()
        ip = self._v2i(pos)
        self.zoom = min(64.0, max(0.05, self.zoom * factor))
        self.off = QPointF(pos.x() - ip.x() * self.zoom, pos.y() - ip.y() * self.zoom)
        self.auto_fit = False
        self.update()

    def _mover_split(self, pos):
        """Mueve el divisor de comparacion a la posicion horizontal del mouse."""
        r = self._rect_img()
        if r.width() > 0:
            self.split = min(1.0, max(0.0, (pos.x() - r.left()) / r.width()))
            self.update()

    def mousePressEvent(self, e):
        """Clic izquierdo: punto > divisor > pan. Clic derecho/central: pan."""
        if self.img is None:
            return
        pos = e.position()
        if e.button() == Qt.LeftButton:
            i = self._hit_punto(pos)
            if i is not None:
                self._drag = ("punto", i)
            elif self.comparar:
                self._drag = ("split",)
                self._mover_split(pos)
            else:
                self._drag = ("pan", pos)
        elif e.button() in (Qt.MiddleButton, Qt.RightButton):
            self._drag = ("pan", pos)
        if self._drag and self._drag[0] == "pan":
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self.img is None:
            return
        pos = e.position()
        ip = self._v2i(pos)
        if self._drag:
            tipo = self._drag[0]
            if tipo == "punto":
                # Los puntos se limitan a los bordes de la imagen.
                x = min(max(ip.x(), 0), self.img.width() - 1)
                y = min(max(ip.y(), 0), self.img.height() - 1)
                self.puntoMovido.emit(self._drag[1], x, y)
            elif tipo == "split":
                self._mover_split(pos)
            elif tipo == "pan":
                self.off = self.off + (pos - self._drag[1])
                self._drag = ("pan", pos)
                self.auto_fit = False
                self.update()
        else:
            # Cursor contextual segun lo que haya bajo el mouse.
            if self._hit_punto(pos) is not None:
                self.setCursor(Qt.SizeAllCursor)
            elif self.comparar:
                self.setCursor(Qt.SplitHCursor)
            else:
                self.unsetCursor()
        self.hoverImagen.emit(ip.x(), ip.y())

    def mouseReleaseEvent(self, _):
        self._drag = None
        self.unsetCursor()


# ----------------------------------------------------------------------------
# Controles auxiliares
# ----------------------------------------------------------------------------
class SliderConValor(QWidget):
    """Slider horizontal con una etiqueta que muestra su valor formateado."""

    def __init__(self, minimo, maximo, valor, formato=lambda v: str(v)):
        """
        Args:
            minimo, maximo, valor: rango y valor inicial (enteros).
            formato: funcion que convierte el valor entero al texto mostrado
                     (p. ej. dividir entre 10 para mostrar decimales).
        """
        super().__init__()
        self.formato = formato
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimo, maximo)
        self.slider.setValue(valor)
        self.etiqueta = QLabel(formato(valor))
        self.etiqueta.setMinimumWidth(34)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider)
        lay.addWidget(self.etiqueta)
        self.slider.valueChanged.connect(lambda v: self.etiqueta.setText(self.formato(v)))

    def value(self):
        return self.slider.value()

    def setValue(self, v):
        self.slider.setValue(int(v))


# ----------------------------------------------------------------------------
# Ventana principal
# ----------------------------------------------------------------------------
class Ventana(QMainWindow):
    """Ventana principal: panel de controles, vista original y vista de resultado.

    Flujo de datos:
        original (BGRA) --aplicar_warp--> warp --crear_mapa_normales--> normal
    Cada cambio de un control llama a procesar(), que recalcula warp y normal y
    refresca las vistas. El estado completo de los controles se serializa en un
    diccionario (estado()/aplicar_estado()) que alimenta deshacer/rehacer, los
    presets y la restauracion de la sesion anterior.
    """
    MODOS = ["Imagen", "Normales", "Comparar", "Luz"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zomboid Asset Warper + Normal Map")
        self.resize(1400, 800)
        self.setAcceptDrops(True)   # permite arrastrar una imagen a la ventana
        for n in ("logo.ico", "logo.png"):
            if os.path.exists(ruta_recurso(n)):
                self.setWindowIcon(QIcon(ruta_recurso(n)))
                break

        self.cfg = QSettings()     # ajustes persistentes (geometria, tema, ultima sesion)
        self.ruta = None
        self.original = None   # BGRA
        self.warp = None       # BGRA
        self.normal = None     # BGRA
        self.modo = 0
        self._luz = None       # posicion de la luz en el modo "Luz"
        self._bloq = False     # True = ignorar cambios de controles (carga programatica)

        # deshacer / rehacer: pilas de estados; el "commit" se retrasa 450 ms para
        # agrupar cambios continuos (p. ej. arrastrar un slider) en un solo paso.
        self._undo, self._redo = [], []
        self._timer_commit = QTimer(self)
        self._timer_commit.setSingleShot(True)
        self._timer_commit.timeout.connect(self._commit)

        self.presets = self._cargar_presets()
        self._construir_ui()
        self._construir_menus()
        self._cargar_ajustes()
        self._ultimo = self.estado()
        self._estado_msg("Abre una imagen (Ctrl+O) o arrastrala a la ventana.")

    # ======================= construccion =======================
    def _construir_ui(self):
        """Crea todos los widgets: panel lateral, vistas y conexiones de senales."""
        # --- panel de controles (con scroll) ---
        panel = QWidget()
        pl = QVBoxLayout(panel)

        self.btn_abrir = QPushButton("Abrir imagen...")
        self.btn_abrir.clicked.connect(self.abrir)
        pl.addWidget(self.btn_abrir)
        self.lbl_archivo = QLabel("(ningun archivo)")
        self.lbl_archivo.setWordWrap(True)
        pl.addWidget(self.lbl_archivo)

        # Presets
        g_pre = QGroupBox("Presets")
        lp = QVBoxLayout(g_pre)
        self.cmb_preset = QComboBox()
        self.cmb_preset.setPlaceholderText("Elegir preset...")
        self.cmb_preset.activated.connect(self.aplicar_preset)
        self.chk_preset_warp = QCheckBox("Incluir el warp al aplicar")
        self.chk_preset_warp.setChecked(True)
        fila = QHBoxLayout()
        b_gp = QPushButton("Guardar actual...")
        b_gp.clicked.connect(self.guardar_preset)
        b_dp = QPushButton("Eliminar")
        b_dp.clicked.connect(self.eliminar_preset)
        fila.addWidget(b_gp)
        fila.addWidget(b_dp)
        lp.addWidget(self.cmb_preset)
        lp.addWidget(self.chk_preset_warp)
        lp.addLayout(fila)
        pl.addWidget(g_pre)
        self._refrescar_combo_presets()

        # Warp
        g_warp = QGroupBox("Warp de perspectiva")
        f = QFormLayout(g_warp)
        self.chk_warp = QCheckBox("Aplicar warp")
        self.chk_warp.setChecked(True)
        f.addRow(self.chk_warp)
        self.sp_src = []   # [(spin_x, spin_y)] para cada uno de los 4 puntos
        for nombre, (x, y) in zip(POINT_NAMES, DEFAULT_SRC):
            sx, sy = QSpinBox(), QSpinBox()
            for s in (sx, sy):
                s.setRange(0, 99999)
            sx.setValue(x)
            sy.setValue(y)
            fila = QHBoxLayout()
            fila.addWidget(QLabel("x")); fila.addWidget(sx)
            fila.addWidget(QLabel("y")); fila.addWidget(sy)
            f.addRow(nombre, fila)
            self.sp_src.append((sx, sy))
        self.sp_w, self.sp_h = QSpinBox(), QSpinBox()
        for s, v in ((self.sp_w, DEFAULT_SIZE[0]), (self.sp_h, DEFAULT_SIZE[1])):
            s.setRange(1, 8192)
            s.setValue(v)
        self.chk_tam_auto = QCheckBox("Tamano automatico (pixeles dentro del area)")
        self.chk_tam_auto.setToolTip(
            "Calcula el ancho y alto de salida midiendo el area marcada por los 4 puntos,\n"
            "para no estirar ni encoger la imagen. Desactivalo para escribir el tamano a mano.")
        self.lbl_tam_auto = QLabel("")
        fila = QHBoxLayout()
        fila.addWidget(QLabel("ancho")); fila.addWidget(self.sp_w)
        fila.addWidget(QLabel("alto")); fila.addWidget(self.sp_h)
        f.addRow(self.chk_tam_auto)
        f.addRow("Tamano de salida", fila)
        f.addRow(self.lbl_tam_auto)
        b_rw = QPushButton("Restaurar warp")
        b_rw.clicked.connect(self.restaurar_warp)
        f.addRow(b_rw)
        pl.addWidget(g_warp)

        # Normales
        g_norm = QGroupBox("Mapa de normales")
        f2 = QFormLayout(g_norm)
        self.s_strength = SliderConValor(0, 100, 30, lambda v: "%.1f" % (v / 10))
        self.s_blur = SliderConValor(0, 10, 1)
        self.s_contrast = SliderConValor(0, 40, 0, lambda v: "%.1f" % (v / 10))
        self.c_invert = QCheckBox("Invertir claros/oscuros")
        self.c_flip = QCheckBox("Formato DirectX (invertir verde)")
        self.c_flat = QCheckBox("Fondo con normal plana")
        f2.addRow("Strength", self.s_strength)
        f2.addRow("Blur", self.s_blur)
        f2.addRow("Contraste", self.s_contrast)
        f2.addRow(self.c_invert)
        f2.addRow(self.c_flip)
        f2.addRow(self.c_flat)
        b_rn = QPushButton("Restaurar normales")
        b_rn.clicked.connect(self.restaurar_normales)
        f2.addRow(b_rn)
        pl.addWidget(g_norm)

        # Luz
        g_luz = QGroupBox("Vista de luz")
        f3 = QFormLayout(g_luz)
        self.s_luz = SliderConValor(2, 40, 6, lambda v: "%.1f" % (v / 10))
        f3.addRow("Altura de la luz", self.s_luz)
        f3.addRow(QLabel("(modo Luz: mueve el mouse sobre la imagen)"))
        pl.addWidget(g_luz)

        # Guardar (deshabilitados hasta que haya un resultado)
        self.btn_guardar = QPushButton("Guardar ambos (imagen + normales)...")
        self.btn_guardar.clicked.connect(self.guardar_ambos)
        self.btn_guardar_n = QPushButton("Guardar solo normales...")
        self.btn_guardar_n.clicked.connect(self.guardar_normales)
        self.btn_guardar_w = QPushButton("Guardar solo imagen warpeada...")
        self.btn_guardar_w.clicked.connect(self.guardar_warp)
        for b in (self.btn_guardar, self.btn_guardar_n, self.btn_guardar_w):
            b.setEnabled(False)
            pl.addWidget(b)
        pl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(370)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # --- vista original ---
        col_o = QWidget()
        lo = QVBoxLayout(col_o)
        lo.setContentsMargins(0, 0, 0, 0)
        t1 = QLabel("Original: arrastra los puntos rojos")
        t1.setAlignment(Qt.AlignHCenter)
        self.v_orig = ImageView("Abre una imagen o arrastrala aqui")
        self.v_orig.puntoMovido.connect(self._punto_arrastrado)
        lo.addWidget(t1)
        lo.addWidget(self.v_orig, 1)

        # --- vista resultado ---
        col_r = QWidget()
        lr = QVBoxLayout(col_r)
        lr.setContentsMargins(0, 0, 0, 0)
        barra = QHBoxLayout()
        self.grupo_modo = QButtonGroup(self)
        self.btn_modo = []
        for i, nombre in enumerate(self.MODOS):
            b = QPushButton(nombre)
            b.setCheckable(True)
            b.setChecked(i == 0)
            self.grupo_modo.addButton(b, i)
            b.toggled.connect(lambda ok, i=i: ok and self.set_modo(i))
            barra.addWidget(b)
            self.btn_modo.append(b)
        barra.addSpacing(12)
        barra.addWidget(QLabel("Fondo"))
        self.cmb_fondo = QComboBox()
        self.cmb_fondo.addItems(["Cuadros", "Negro", "Gris", "Blanco"])
        self.cmb_fondo.currentTextChanged.connect(self._cambiar_fondo)
        barra.addWidget(self.cmb_fondo)
        self.chk_rgb = QCheckBox("Normales sin transparencia")
        self.chk_rgb.toggled.connect(lambda _: self.refrescar_resultado())
        barra.addWidget(self.chk_rgb)
        barra.addStretch(1)
        self.v_res = ImageView("Aqui aparece el resultado")
        self.v_res.hoverImagen.connect(self._hover_resultado)
        lr.addLayout(barra)
        lr.addWidget(self.v_res, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(col_o)
        split.addWidget(col_r)
        split.setSizes([500, 600])

        central = QWidget()
        raiz = QHBoxLayout(central)
        raiz.addWidget(scroll)
        raiz.addWidget(split, 1)
        self.setCentralWidget(central)

        # cualquier cambio de ajuste -> _cambio (reprocesa y programa el commit de deshacer)
        for sp in [self.sp_w, self.sp_h] + [s for par in self.sp_src for s in par]:
            sp.valueChanged.connect(self._cambio)
        self.chk_warp.toggled.connect(self._cambio)
        self.chk_tam_auto.toggled.connect(self._actualizar_tam_ui)
        self.chk_tam_auto.toggled.connect(self._cambio)
        for s in (self.s_strength, self.s_blur, self.s_contrast):
            s.slider.valueChanged.connect(self._cambio)
        for c in (self.c_invert, self.c_flip, self.c_flat):
            c.toggled.connect(self._cambio)
        # La altura de la luz solo afecta a la vista, no al resultado guardado.
        self.s_luz.slider.valueChanged.connect(lambda _: self.refrescar_resultado())

    def _construir_menus(self):
        """Crea la barra de menus (Archivo, Edicion, Vista, Ayuda) con sus atajos."""
        mb = self.menuBar()
        m_arch = mb.addMenu("&Archivo")
        self._accion(m_arch, "Abrir imagen...", "Ctrl+O", self.abrir)
        m_arch.addSeparator()
        self.act_guardar = self._accion(m_arch, "Guardar ambos...", "Ctrl+S", self.guardar_ambos)
        self.act_guardar_n = self._accion(m_arch, "Guardar solo normales...", "Ctrl+Shift+S", self.guardar_normales)
        self.act_guardar_w = self._accion(m_arch, "Guardar solo imagen warpeada...", "Ctrl+Alt+S", self.guardar_warp)
        for a in (self.act_guardar, self.act_guardar_n, self.act_guardar_w):
            a.setEnabled(False)
        m_arch.addSeparator()
        self._accion(m_arch, "Salir", "Ctrl+Q", self.close)

        m_ed = mb.addMenu("&Edicion")
        self._accion(m_ed, "Deshacer", "Ctrl+Z", self.deshacer)
        self._accion(m_ed, "Rehacer", "Ctrl+Y", self.rehacer)
        self._accion(m_ed, "Rehacer (alternativo)", "Ctrl+Shift+Z", self.rehacer)
        m_ed.addSeparator()
        self._accion(m_ed, "Restaurar warp", "Ctrl+R", self.restaurar_warp)
        self._accion(m_ed, "Restaurar normales", "Ctrl+Shift+R", self.restaurar_normales)
        m_ed.addSeparator()
        self._accion(m_ed, "Guardar preset actual...", "Ctrl+P", self.guardar_preset)

        m_vista = mb.addMenu("&Vista")
        for i, nombre in enumerate(self.MODOS):
            self._accion(m_vista, "Modo: " + nombre, "Ctrl+%d" % (i + 1),
                         lambda i=i: self.btn_modo[i].setChecked(True))
        m_vista.addSeparator()
        self._accion(m_vista, "Ajustar vistas a la ventana", "Ctrl+0", self.ajustar_vistas)
        a_rgb = self._accion(m_vista, "Normales sin transparencia", "Ctrl+T",
                             lambda: self.chk_rgb.toggle())
        self.act_tema = QAction("Tema oscuro", self)
        self.act_tema.setCheckable(True)
        self.act_tema.toggled.connect(self._cambiar_tema)
        m_vista.addAction(self.act_tema)

        m_ayuda = mb.addMenu("A&yuda")
        self._accion(m_ayuda, "Atajos y controles", "F1", self.mostrar_ayuda)

    def _accion(self, menu, texto, atajo, slot):
        """Crea una QAction con atajo opcional, la agrega al menu y la devuelve."""
        a = QAction(texto, self)
        if atajo:
            a.setShortcut(QKeySequence(atajo))
        a.triggered.connect(lambda _=False: slot())
        menu.addAction(a)
        return a

    def _estado_msg(self, texto):
        """Muestra un mensaje en la barra de estado."""
        self.statusBar().showMessage(texto)

    # ======================= ajustes persistentes =======================
    def _cargar_ajustes(self):
        """Restaura geometria, tema, fondo y los ultimos ajustes usados (QSettings)."""
        geo = self.cfg.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        oscuro = str(self.cfg.value("oscuro", "true")).lower() == "true"
        self.act_tema.setChecked(oscuro)
        aplicar_tema(QApplication.instance(), oscuro)
        fondo = self.cfg.value("fondo", "Cuadros")
        if fondo in ("Cuadros", "Negro", "Gris", "Blanco"):
            self.cmb_fondo.setCurrentText(fondo)
        try:
            e = json.loads(self.cfg.value("estado", ""))
            self.aplicar_estado(e, push=False)
        except Exception:
            pass   # primera ejecucion o ajustes corruptos: se usan los valores por defecto

    def closeEvent(self, ev):
        """Guarda la configuracion al cerrar la ventana."""
        self.cfg.setValue("geometry", self.saveGeometry())
        self.cfg.setValue("estado", json.dumps(self.estado()))
        self.cfg.setValue("oscuro", self.act_tema.isChecked())
        self.cfg.setValue("fondo", self.cmb_fondo.currentText())
        super().closeEvent(ev)

    def _cambiar_tema(self, oscuro):
        aplicar_tema(QApplication.instance(), oscuro)
        self.cfg.setValue("oscuro", oscuro)
        self.v_orig.update(); self.v_res.update()

    def _cambiar_fondo(self, nombre):
        self.v_orig.set_fondo(nombre)
        self.v_res.set_fondo(nombre)

    # ======================= estado, deshacer, presets =======================
    def estado(self):
        """Devuelve un diccionario con el valor actual de todos los controles."""
        return {
            "warp": self.chk_warp.isChecked(),
            "puntos": [[sx.value(), sy.value()] for sx, sy in self.sp_src],
            "ancho": self.sp_w.value(),
            "alto": self.sp_h.value(),
            "tam_auto": self.chk_tam_auto.isChecked(),
            "strength": self.s_strength.value(),
            "blur": self.s_blur.value(),
            "contraste": self.s_contrast.value(),
            "invertir": self.c_invert.isChecked(),
            "directx": self.c_flip.isChecked(),
            "fondo_plano": self.c_flat.isChecked(),
        }

    def aplicar_estado(self, e, push=True):
        """Pone los controles segun el diccionario 'e' y reprocesa la imagen.

        Las claves que falten se completan con DEFAULT_ESTADO. Mientras se
        escriben los controles, _bloq evita que cada cambio dispare un reproceso.

        Args:
            push: True = registrar el cambio en el historial de deshacer.
        """
        d = dict(DEFAULT_ESTADO)
        d.update(e)
        self._bloq = True
        try:
            self.chk_warp.setChecked(bool(d["warp"]))
            for (sx, sy), (x, y) in zip(self.sp_src, d["puntos"]):
                sx.setValue(int(x)); sy.setValue(int(y))
            self.sp_w.setValue(int(d["ancho"])); self.sp_h.setValue(int(d["alto"]))
            self.chk_tam_auto.setChecked(bool(d["tam_auto"]))
            self.s_strength.setValue(d["strength"])
            self.s_blur.setValue(d["blur"])
            self.s_contrast.setValue(d["contraste"])
            self.c_invert.setChecked(bool(d["invertir"]))
            self.c_flip.setChecked(bool(d["directx"]))
            self.c_flat.setChecked(bool(d["fondo_plano"]))
        finally:
            self._bloq = False
        self.procesar()
        if push:
            self._commit()
        else:
            self._ultimo = self.estado()

    def _cambio(self, *_):
        """Reacciona a un cambio de control: reprocesa y programa el commit al historial."""
        if self._bloq:
            return
        self.procesar()
        self._timer_commit.start(450)

    def _commit(self):
        """Guarda el estado anterior en la pila de deshacer si algo cambio (max. 100)."""
        self._timer_commit.stop()
        e = self.estado()
        if e != self._ultimo:
            self._undo.append(self._ultimo)
            if len(self._undo) > 100:
                self._undo.pop(0)
            self._redo.clear()
            self._ultimo = e

    def deshacer(self):
        self._commit()
        if not self._undo:
            self._estado_msg("Nada que deshacer")
            return
        self._redo.append(self._ultimo)
        e = self._undo.pop()
        self.aplicar_estado(e, push=False)
        self._estado_msg("Deshacer")

    def rehacer(self):
        self._commit()
        if not self._redo:
            self._estado_msg("Nada que rehacer")
            return
        self._undo.append(self._ultimo)
        e = self._redo.pop()
        self.aplicar_estado(e, push=False)
        self._estado_msg("Rehacer")

    def restaurar_warp(self):
        """Vuelve los ajustes del warp a sus valores por defecto."""
        e = self.estado()
        for k in ("warp", "puntos", "ancho", "alto", "tam_auto"):
            e[k] = DEFAULT_ESTADO[k]
        self.aplicar_estado(e)

    def restaurar_normales(self):
        """Vuelve los ajustes del mapa de normales a sus valores por defecto."""
        e = self.estado()
        for k in CLAVES_NORMALES:
            e[k] = DEFAULT_ESTADO[k]
        self.aplicar_estado(e)

    # --- presets ---
    def _dir_config(self):
        """Carpeta de configuracion de la aplicacion (se crea si no existe)."""
        d = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
        os.makedirs(d, exist_ok=True)
        return d

    def _ruta_presets(self):
        return os.path.join(self._dir_config(), "presets.json")

    def _cargar_presets(self):
        """Lee presets.json; si no existe o es invalido, devuelve un preset por defecto."""
        try:
            with open(self._ruta_presets(), "r", encoding="utf-8") as f:
                datos = json.load(f)
            if isinstance(datos, dict) and datos:
                return datos
        except Exception:
            pass
        return {"Pared (por defecto)": json.loads(json.dumps(DEFAULT_ESTADO))}

    def _guardar_presets(self):
        try:
            with open(self._ruta_presets(), "w", encoding="utf-8") as f:
                json.dump(self.presets, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.warning(self, "Presets", "No se pudo guardar el archivo de presets:\n%s" % e)

    def _refrescar_combo_presets(self, seleccionar=None):
        """Rellena el combo de presets (orden alfabetico) sin disparar sus senales."""
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.clear()
        for n in sorted(self.presets, key=str.lower):
            self.cmb_preset.addItem(n)
        if seleccionar and seleccionar in self.presets:
            self.cmb_preset.setCurrentText(seleccionar)
        else:
            self.cmb_preset.setCurrentIndex(-1)
        self.cmb_preset.blockSignals(False)

    def aplicar_preset(self, idx):
        """Aplica el preset elegido. Si 'Incluir el warp' esta apagado, solo se
        aplican los ajustes de normales y se conserva el warp actual."""
        nombre = self.cmb_preset.itemText(idx)
        p = self.presets.get(nombre)
        if not p:
            return
        if self.chk_preset_warp.isChecked():
            e = dict(DEFAULT_ESTADO); e.update(p)
        else:
            e = self.estado()
            for k in CLAVES_NORMALES:
                if k in p:
                    e[k] = p[k]
        self.aplicar_estado(e)
        self._estado_msg("Preset aplicado: " + nombre)

    def guardar_preset(self):
        """Pide un nombre y guarda el estado actual como preset (confirma si reemplaza)."""
        nombre, ok = QInputDialog.getText(self, "Guardar preset", "Nombre del preset:",
                                          text=self.cmb_preset.currentText())
        nombre = nombre.strip()
        if not ok or not nombre:
            return
        if nombre in self.presets:
            r = QMessageBox.question(self, "Guardar preset",
                                     "Ya existe '%s'. Quieres reemplazarlo?" % nombre)
            if r != QMessageBox.Yes:
                return
        self.presets[nombre] = self.estado()
        self._guardar_presets()
        self._refrescar_combo_presets(nombre)
        self._estado_msg("Preset guardado: " + nombre)

    def eliminar_preset(self):
        """Elimina el preset seleccionado tras confirmar."""
        nombre = self.cmb_preset.currentText()
        if not nombre or nombre not in self.presets:
            return
        r = QMessageBox.question(self, "Eliminar preset", "Eliminar el preset '%s'?" % nombre)
        if r != QMessageBox.Yes:
            return
        del self.presets[nombre]
        self._guardar_presets()
        self._refrescar_combo_presets()
        self._estado_msg("Preset eliminado: " + nombre)

    # ======================= abrir =======================
    def abrir(self):
        """Abre un dialogo para elegir una imagen (recuerda la ultima carpeta)."""
        ini = self.cfg.value("dir_abrir", "")
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen", ini if isinstance(ini, str) else "",
            "Imagenes (*.png *.jpg *.jpeg *.bmp *.tga *.tif *.tiff);;Todos los archivos (*)")
        if ruta:
            self.cargar(ruta)

    def cargar(self, ruta):
        """Carga la imagen, la muestra en la vista original y genera el resultado."""
        try:
            img = leer_imagen(ruta)
        except Exception as e:
            QMessageBox.critical(self, "Error al abrir", str(e))
            return
        self.original = img
        self.ruta = ruta
        self.cfg.setValue("dir_abrir", os.path.dirname(ruta))
        self.lbl_archivo.setText(os.path.basename(ruta))
        h, w = img.shape[:2]
        self.v_orig.set_images(bgra_a_qimage(img))
        self.v_orig.ajustar()
        self._estado_msg("Cargada: %s  (%dx%d)" % (ruta, w, h))
        self.procesar()
        self.v_res.ajustar()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        """Carga la primera imagen soltada sobre la ventana."""
        rutas = [u.toLocalFile() for u in e.mimeData().urls() if os.path.isfile(u.toLocalFile())]
        if rutas:
            self.cargar(rutas[0])
            if len(rutas) > 1:
                self._estado_msg("Se cargo la primera imagen (%d soltadas)" % len(rutas))

    # ======================= procesar y vistas =======================
    def puntos_src(self):
        """Los 4 puntos del warp actuales, leidos de los spinboxes."""
        return [(sx.value(), sy.value()) for sx, sy in self.sp_src]

    def _punto_arrastrado(self, i, x, y):
        """Actualiza los spinboxes del punto i al arrastrarlo en la vista.

        Se bloquean sus senales para reprocesar una sola vez (con _cambio) en lugar
        de una por cada spinbox modificado.
        """
        sx, sy = self.sp_src[i]
        sx.blockSignals(True); sy.blockSignals(True)
        sx.setValue(int(round(x))); sy.setValue(int(round(y)))
        sx.blockSignals(False); sy.blockSignals(False)
        self._cambio()

    def tamano_salida(self):
        """Tamano real de salida: automatico (desde los puntos) o el escrito a mano."""
        if self.chk_tam_auto.isChecked():
            return tamano_auto(self.puntos_src())
        return self.sp_w.value(), self.sp_h.value()

    def _actualizar_tam_ui(self, *_):
        """Habilita/deshabilita los spinboxes de tamano y muestra el tamano calculado."""
        auto = self.chk_tam_auto.isChecked()
        self.sp_w.setEnabled(not auto)
        self.sp_h.setEnabled(not auto)
        if auto:
            w, h = tamano_auto(self.puntos_src())
            self.lbl_tam_auto.setText("Calculado: %d x %d px" % (w, h))
        else:
            self.lbl_tam_auto.setText("")

    def procesar(self, *_):
        """Recalcula warp y mapa de normales con los ajustes actuales y refresca las vistas."""
        self.v_orig.puntos = self.puntos_src() if self.chk_warp.isChecked() else None
        self.v_orig.update()
        self._actualizar_tam_ui()
        if self.original is None:
            return
        try:
            if self.chk_warp.isChecked():
                self.warp = aplicar_warp(self.original, self.puntos_src(),
                                         self.tamano_salida())
            else:
                self.warp = self.original.copy()
            self.normal = crear_mapa_normales(
                self.warp,
                strength=self.s_strength.value() / 10.0,
                blur_radius=self.s_blur.value(),
                invert=self.c_invert.isChecked(),
                flip_green=self.c_flip.isChecked(),
                contrast_clip=self.s_contrast.value() / 10.0,
                fondo_plano=self.c_flat.isChecked())
        except Exception as e:
            self._estado_msg("Error: %s" % e)
            return
        # Con un resultado disponible se habilitan los botones y acciones de guardado.
        for w in (self.btn_guardar, self.btn_guardar_n, self.btn_guardar_w,
                  self.act_guardar, self.act_guardar_n, self.act_guardar_w):
            w.setEnabled(True)
        self.refrescar_resultado()

    def set_modo(self, i):
        """Cambia el modo de la vista de resultado (Imagen, Normales, Comparar, Luz)."""
        self.modo = i
        self.v_res.comparar = (i == 2)
        self.refrescar_resultado()

    def refrescar_resultado(self):
        """Actualiza la vista de resultado segun el modo activo."""
        if self.warp is None or self.normal is None:
            return
        m = self.modo
        solo_rgb = self.chk_rgb.isChecked()
        if m == 0:
            self.v_res.set_images(bgra_a_qimage(self.warp))
        elif m == 1:
            self.v_res.set_images(bgra_a_qimage(self.normal, solo_rgb))
        elif m == 2:
            self.v_res.set_images(bgra_a_qimage(self.warp),
                                  bgra_a_qimage(self.normal, solo_rgb))
        else:
            h, w = self.warp.shape[:2]
            if self._luz is None:
                self._luz = (w * 0.25, h * 0.25)   # posicion inicial de la luz
            lit = iluminar(self.warp, self.normal, self.c_flip.isChecked(),
                           self._luz, self.s_luz.value() / 10.0)
            self.v_res.set_images(bgra_a_qimage(lit))
        self.v_res.update()

    def _hover_resultado(self, x, y):
        """En modo Luz, la luz sigue al mouse sobre la imagen de resultado."""
        if self.modo == 3 and self.warp is not None:
            self._luz = (x, y)
            self.refrescar_resultado()

    def ajustar_vistas(self):
        self.v_orig.ajustar()
        self.v_res.ajustar()

    # ======================= guardar =======================
    def _nombre_sugerido(self):
        """Nombre base sugerido: <nombre de la imagen>_Wall."""
        base = os.path.splitext(os.path.basename(self.ruta or "imagen"))[0]
        return base + "_Wall"

    def _carpeta_guardar(self):
        """Carpeta inicial del dialogo de guardado: la ultima usada, o la de la imagen."""
        d = self.cfg.value("dir_guardar", "")
        if isinstance(d, str) and d and os.path.isdir(d):
            return d
        return os.path.dirname(self.ruta) if self.ruta else ""

    def _pedir_ruta(self, titulo, nombre):
        """Dialogo 'Guardar como' para PNG. Devuelve la ruta (con .png) o None si se cancela."""
        ruta, _ = QFileDialog.getSaveFileName(
            self, titulo, os.path.join(self._carpeta_guardar(), nombre), "PNG (*.png)")
        if not ruta:
            return None
        if not ruta.lower().endswith(".png"):
            ruta += ".png"
        self.cfg.setValue("dir_guardar", os.path.dirname(ruta))
        return ruta

    def _escribir(self, ruta, img):
        """Guarda la imagen mostrando un error si falla. Devuelve True si tuvo exito."""
        try:
            guardar_imagen(ruta, img)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", "%s\n\n%s" % (ruta, e))
            return False

    def guardar_ambos(self):
        """Guarda NOMBRE.png (warp) y NOMBRE_n.png (normales) en la misma carpeta."""
        if self.warp is None:
            return
        ruta = self._pedir_ruta("Guardar imagen warpeada (las normales se guardan al lado como _n)",
                                self._nombre_sugerido() + ".png")
        if not ruta:
            return
        ruta_n = ruta[:-4] + "_n.png"
        if self._escribir(ruta, self.warp) and self._escribir(ruta_n, self.normal):
            self._estado_msg("Guardado: %s  y  %s" % (ruta, ruta_n))

    def guardar_normales(self):
        """Guarda solo el mapa de normales (sugiere el sufijo _n)."""
        if self.normal is None:
            return
        ruta = self._pedir_ruta("Guardar mapa de normales", self._nombre_sugerido() + "_n.png")
        if ruta and self._escribir(ruta, self.normal):
            self._estado_msg("Guardado: " + ruta)

    def guardar_warp(self):
        """Guarda solo la imagen warpeada."""
        if self.warp is None:
            return
        ruta = self._pedir_ruta("Guardar imagen warpeada", self._nombre_sugerido() + ".png")
        if ruta and self._escribir(ruta, self.warp):
            self._estado_msg("Guardado: " + ruta)

    # ======================= ayuda =======================
    def mostrar_ayuda(self):
        """Muestra el cuadro con atajos de teclado y controles del mouse."""
        QMessageBox.information(self, "Atajos y controles", (
            "ATAJOS\n"
            "  Ctrl+O            Abrir imagen\n"
            "  Ctrl+S            Guardar ambos (imagen + normales)\n"
            "  Ctrl+Shift+S      Guardar solo normales\n"
            "  Ctrl+Alt+S        Guardar solo imagen warpeada\n"
            "  Ctrl+Z / Ctrl+Y   Deshacer / Rehacer\n"
            "  Ctrl+R            Restaurar warp\n"
            "  Ctrl+Shift+R      Restaurar normales\n"
            "  Ctrl+P            Guardar preset\n"
            "  Ctrl+1..4         Modo: Imagen, Normales, Comparar, Luz\n"
            "  Ctrl+0            Ajustar vistas\n"
            "  Ctrl+T            Normales con/sin transparencia\n"
            "  F1                Esta ayuda\n\n"
            "MOUSE (en las vistas)\n"
            "  Rueda             Zoom hacia el cursor\n"
            "  Arrastrar         Mover la imagen (clic derecho/central tambien)\n"
            "  Puntos rojos      Arrastrarlos para ajustar el warp\n"
            "  Modo Comparar     Arrastrar para mover el divisor\n"
            "  Modo Luz          Mueve el mouse para mover la luz"))


def main():
    """Punto de entrada: crea la aplicacion, abre la ventana y, si se paso una imagen
    como argumento de linea de comandos, la carga."""
    app = QApplication(sys.argv)
    app.setOrganizationName("ZomboidTools")
    app.setApplicationName("ZomboidNormalMapper")
    v = Ventana()
    v.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        v.cargar(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()