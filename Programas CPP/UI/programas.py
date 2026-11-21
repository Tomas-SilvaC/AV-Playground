"""
programas.py
Registro de los programas que la interfaz (launcher.py) sabe compilar y ejecutar.

## Idea general
Cada programa del repositorio recibe sus datos de una forma distinta: unos piden la
ruta de la imagen por consola (stdin), otros la reciben como argumento, y la
calibración lee una carpeta fija ./images. En vez de modificar esos programas, este
archivo describe PARA CADA UNO:

  - dónde está su código fuente y a qué tema pertenece,
  - qué campos debe llenar el usuario (imagen, tamaño de kernel, etc.),
  - cómo convertir esos campos en argumentos y/o texto para stdin (`construir`),
  - si necesita preparar algo antes de correr (`preparar`).

Para añadir un programa nuevo (p. ej. Caracterización) basta con agregar un objeto
`Programa` a la lista PROGRAMAS al final de este archivo; la interfaz lo muestra sola.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Raíz del repositorio (esta carpeta, UI/, vive dentro de ella).
RAIZ = Path(__file__).resolve().parent.parent
CPP = RAIZ / "Programas CPP"
IMAGENES_EJEMPLO = CPP / "images"


@dataclass
class Campo:
    """Un dato que el usuario llena en el formulario de un programa."""
    nombre: str                    # clave en el diccionario de valores
    etiqueta: str                  # texto mostrado junto al control
    tipo: str                      # "imagen" | "carpeta" | "int" | "bool" | "opcion"
    default: object = None
    minimo: int = 0
    maximo: int = 999
    opciones: tuple = ()           # solo para tipo "opcion"
    opcional: bool = False         # solo "imagen"/"carpeta": se puede dejar vacío
    ayuda: str = ""


@dataclass
class Programa:
    id: str
    grupo: str                     # tema visto en clase (agrupa en el árbol)
    titulo: str
    fuente: Path                   # archivo .cpp (o .py) del programa
    descripcion: str
    campos: list = field(default_factory=list)
    # valores -> (lista de argumentos, texto para stdin)
    construir: Callable[[dict], tuple] = lambda v: ([], "")
    # (carpeta de trabajo, valores) -> None; prepara archivos antes de ejecutar
    preparar: Optional[Callable[[Path, dict], None]] = None
    tipo: str = "cpp"              # "cpp" (se compila) | "python" (se ejecuta directo)
    teclas: str = ""               # ayuda de teclas de las ventanas de OpenCV
    aviso: str = ""                # advertencia visible en la interfaz
    muestra_resultados: bool = True


def imagen_ejemplo() -> str:
    """Primera imagen del repositorio, para que todo funcione sin elegir nada."""
    for nombre in ("image_1.jpg",):
        p = IMAGENES_EJEMPLO / nombre
        if p.exists():
            return str(p)
    fotos = sorted(IMAGENES_EJEMPLO.glob("*.jpg"))
    return str(fotos[0]) if fotos else ""


CAMPO_IMAGEN = Campo("imagen", "Imagen", "imagen", default=imagen_ejemplo())


# --------------------------------------------------------------------------
# Constructores de argumentos / stdin (uno por forma de entrada)
# --------------------------------------------------------------------------
def _stdin_imagen_kernel(v):
    # BoxFilter.cpp pregunta primero la ruta y luego el tamaño del kernel.
    return [], f"{v['imagen']}\n{v['kernel']}\n"


def _stdin_imagen_o_webcam(v):
    # erosion_dilatacion.cpp: Enter vacío = usar la webcam.
    return [], ("\n" if v["webcam"] else f"{v['imagen']}\n")


def _arg_imagen(v):
    # Fourier, Histogramas, Umbralizacion: la ruta va como primer argumento.
    return [v["imagen"]], ""


def _args_caracterizacion(v):
    # Caracterizacion.cpp <imagen> [modo inicial 0-3]
    return [v["imagen"], v["modo"]], ""


def _args_segmentacion(v):
    # Segmentacion2.cpp <imagen> <parche> <k> --show
    return [v["imagen"], str(v["parche"]), str(v["k"]), "--show"], ""


def _preparar_calibracion(carpeta_trabajo: Path, v: dict) -> None:
    """CameraCalib.cpp lee ./images/*.jpg relativo a su carpeta de trabajo.
    Se copian allí las fotos elegidas para no ensuciar el repositorio."""
    destino = carpeta_trabajo / "images"
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    origen = Path(v["carpeta"])
    fotos = [p for p in origen.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
    if not fotos:
        raise ValueError(f"No hay archivos .jpg en: {origen}")
    for foto in fotos:
        # CameraCalib usa el patrón *.jpg: se normaliza la extensión.
        shutil.copy(foto, destino / (foto.stem + ".jpg"))


# --------------------------------------------------------------------------
# Lista de programas (el orden aquí es el orden del árbol)
# --------------------------------------------------------------------------
PROGRAMAS = [
    Programa(
        id="calibracion", grupo="Modelos de cámara y calibración",
        titulo="Calibración de cámara",
        fuente=CPP / "Calibracion de camara" / "CameraCalib.cpp",
        descripcion=(
            "Estima la matriz intrínseca K y los coeficientes de distorsión de una lente a "
            "partir de fotos de un tablero de ajedrez (esquinas interiores 5x8), calcula el "
            "error RMS de reproyección y guarda calibration.yml. Después intenta abrir la "
            "cámara para mostrar el video original y el corregido lado a lado."
        ),
        campos=[Campo("carpeta", "Carpeta de fotos (.jpg)", "carpeta",
                      default=str(IMAGENES_EJEMPLO),
                      ayuda="Carpeta con fotos del tablero; se copian a la carpeta de salida.")],
        preparar=_preparar_calibracion,
        teclas="Cualquier tecla = siguiente foto. En el video en vivo: q o ESC = salir.",
        aviso="La parte en vivo usa la cámara de índice 1; si no existe, el programa lo "
              "informa en la consola tras calibrar (la calibración y calibration.yml sí se generan).",
    ),
    Programa(
        id="boxfilter", grupo="Filtros", titulo="Filtro de caja (box filter)",
        fuente=CPP / "Filtros" / "BoxFilter.cpp",
        descripcion=(
            "Convolución con un kernel de pesos iguales 1/(k·k): promedia el vecindario "
            "(filtro paso bajo). Reduce el ruido a costa de desenfocar bordes. Muestra la "
            "imagen original, en grises y filtrada, y guarda gris.jpg y filtro_caja.jpg."
        ),
        campos=[CAMPO_IMAGEN, Campo("kernel", "Tamaño del kernel (impar)", "int", 5, 1, 99)],
        construir=_stdin_imagen_kernel,
        teclas="Con una ventana seleccionada, cualquier tecla cierra el programa.",
    ),
    Programa(
        id="histogramas", grupo="Histogramas", titulo="Histogramas y ecualización",
        fuente=CPP / "Histogramas" / "Histogramas.cpp",
        descripcion=(
            "Histograma de grises y por canal B/G/R, versión normalizada y escala log, "
            "tabla de frecuencias (regla de Sturges) en la consola, y mejora de contraste "
            "con equalizeHist y CLAHE. Barras: Bins, Normalizar, Log."
        ),
        campos=[CAMPO_IMAGEN], construir=_arg_imagen,
        teclas="q o ESC = salir, s = guardar imagenes.png e histogramas.png.",
    ),
    Programa(
        id="umbralizacion", grupo="Umbralización y binarización",
        titulo="Umbralización y binarización",
        fuente=CPP / "Umbralizacion" / "Umbralizacion.cpp",
        descripcion=(
            "Umbral global manual y de Otsu, los cinco tipos (BINARY, BINARY_INV, TRUNC, "
            "TOZERO, TOZERO_INV) y umbral adaptativo (media y gaussiano). Incluye el "
            "histograma con el umbral manual y el de Otsu. Barras: Umbral, Tipo, Bloque, C."
        ),
        campos=[CAMPO_IMAGEN], construir=_arg_imagen,
        teclas="q o ESC = salir, s = guardar umbralizacion.png.",
    ),
    Programa(
        id="morfologia", grupo="Morfología matemática",
        titulo="Erosión y dilatación",
        fuente=CPP / "Filtros" / "erosion_dilatacion.cpp",
        descripcion=(
            "Erosión (mínimo local / AND) y dilatación (máximo local / OR) con elemento "
            "estructurante cuadrado, disco o cruz, sobre grises o binaria. Funciona con una "
            "imagen o con la webcam. Barras: Tamano, Forma, Binaria, Umbral."
        ),
        campos=[CAMPO_IMAGEN, Campo("webcam", "Usar la webcam en vez de la imagen", "bool", False)],
        construir=_stdin_imagen_o_webcam,
        teclas="q o ESC = salir, s = guardar resultado.png.",
    ),
    Programa(
        id="fourier", grupo="Transformadas", titulo="Transformada de Fourier",
        fuente=CPP / "Transformadas" / "Fourier.cpp",
        descripcion=(
            "Descompone la imagen en frecuencias con la DFT y muestra el espectro de "
            "magnitud (escala log, componente DC al centro): frecuencias bajas = zonas "
            "suaves, altas = bordes y texturas."
        ),
        campos=[CAMPO_IMAGEN], construir=_arg_imagen,
        teclas="Cualquier tecla cierra las ventanas.",
    ),
    Programa(
        id="segmentacion", grupo="Segmentación", titulo="CNN y Vision Transformer",
        fuente=CPP / "Segmentacion" / "Segmentacion2.cpp",
        descripcion=(
            "Reproduce las etapas de una CNN (convolución, ReLU, pooling) y de un ViT "
            "(parches, embedding, posición, autoatención) sin pesos entrenados, y segmenta "
            "con k-means sobre los tokens. Guarda out_panel_vit.png, out_segmentacion.png, "
            "out_atencion.png y out_cnn_features.png."
        ),
        campos=[CAMPO_IMAGEN,
                Campo("parche", "Tamaño de parche", "opcion", "16",
                      opciones=("8", "14", "16", "28", "32", "56"),
                      ayuda="Debe dividir 224 (tamaño de entrada del ViT)."),
                Campo("k", "Número de grupos k", "int", 4, 2, 12)],
        construir=_args_segmentacion,
        teclas="Cualquier tecla cierra las ventanas.",
    ),
    Programa(
        id="caracterizacion", grupo="Caracterización", titulo="Caracterización de imágenes",
        fuente=CPP / "Caracterizacion" / "Caracterizacion.cpp",
        descripcion=(
            "Cuatro modos (barra 'Modo' en la ventana): 0 intensidad (histograma + calidad "
            "perceptual), 1 objetos (área, perímetro, circularidad, forma, color y relaciones "
            "RCC-8), 2 textura (GLCM) y 3 puntos clave sin IA (SIFT, ORB y BRIEF a mano sobre "
            "una copia rotada y escalada)."
        ),
        campos=[CAMPO_IMAGEN,
                Campo("modo", "Modo con el que empezar", "opcion", "0",
                      opciones=("0", "1", "2", "3"),
                      ayuda="0 Intensidad · 1 Objetos · 2 Textura · 3 Puntos clave "
                            "(se puede cambiar después con la barra 'Modo').")],
        construir=_args_caracterizacion,
        teclas="q o ESC = salir, p = reporte completo en la consola, s = guardar caracterizacion.png.",
    ),
]
