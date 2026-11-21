# AV-Playground
# Tomas Alejandro Silva Correal
Playground general para los conceptos trabajados en la clase de **Visión Artificial 2026-3**.

Cada programa es una implementación práctica y autocontenida de un tema visto en clase, escrita con **OpenCV** (mayormente en archivos CPP) y una **interfaz en Python** (`UI/`) que permite elegir y ejecutar los programas de los temas vistos en clase desde una sola ventana.
---

## Estructura del repositorio

### Programas CPP
Se presentan programas respecto a los siguientes topicos:
 - Calibración de cámara (`Calibracion de camara/`)
 - Filtros: caja, erosión y dilatación (`Filtros/`)
 - Histogramas (`Histogramas/`)
 - Umbralización y binarización (`Umbralizacion/`)
 - Segmentación (`Segmentacion/`)
 - Transformadas (`Transformadas/`)
 - Caracterización (`Caracterizacion/`)

### UI
Interfaz de escritorio (Python + PySide6) para elegir un tema, llenar sus parámetros, compilarlo y ejecutarlo, viendo la consola, los archivos que genera y el código fuente. Ver la sección [Interfaz de interacción](#interfaz-de-interacción-ui).

---------------------------------------------------------------------------------------------------
Tras esto, se encuentra un pequeño proyecto personal, usando algunos conceptos vistos en la materia
### **ZomboidNormalMapper**
El objetivo de este proyecto es el poder utilizar texturas de assets del videojuego "Project Zomboid" para renderizaciones en blender, animaciones, entre otros.

Inicialmente, las texturas del juego estan en un angulo isometrico, requiriendo una limpieza manual textura a textura a la hora de utilizarlos en cubos u otras formas en la plataforma blender. Por igual, se deseaba tener la oportunidad de general mapas de normales para estas texturas de forma directa, sin tener que acceder a sitios externos, y que funcionara directamente con las texturas tras hacer warping.

Este proyecto tiene dos secciones: 
 - CPP: Se encuentran dos programas en formato .CPP que funcionan como las fases iniciales del proyecto
     - AssetWrapper.cpp -> La idea inicial del proyecto, encargada de "enderezar" la textura para quitarla de la vista isometrica
     - NormalMapper.cpp -> La adición de generación de mapas de normal utilizando filtros
 - Python: Se encuentran tres documentos, el programa en python como aplicación, un listado de requerimientos para compilar, y la linea de comando en terminar para compilar y correr el programa.

| Programa | Tema | Conceptos clave |
|---|---|---|
| `CameraCalib.cpp` | Modelos de cámara y calibración | Matriz intrínseca K, distorsión radial/tangencial, tablero de ajedrez, error de reproyección |
| `BoxFilter.cpp` | Convolución y filtros espaciales | Kernel, convolución, filtro paso bajo, suavizado |
| `erosion_dilatacion.cpp` | Morfología matemática | Elemento estructurante, erosión (mín/AND), dilatación (máx/OR), binarización |
| `Histogramas.cpp` | Histogramas | Histograma gris y por canal, normalizado, tabla de frecuencias (Sturges), ecualización y CLAHE |
| `Umbralizacion.cpp` | Umbralización y binarización | Umbral global, Otsu, tipos (BINARY, TRUNC, TOZERO...), umbral adaptativo (media y gaussiano) |
| `Fourier.cpp` | Transformada de Fourier | Dominio espacial vs. frecuencia, DFT, espectro de magnitud |
| `Segmentacion2.cpp` | CNN y Vision Transformer | Convolución + ReLU + pooling, parches, atención, k-means |
| `Caracterizacion.cpp` | Caracterización | Estadísticas del histograma, área/perímetro/circularidad, RCC-8, GLCM, SIFT / ORB / BRIEF |
| `ZomboidAssetWrapper.cpp` | Transformaciones de perspectiva | `getPerspectiveTransform`, `warpPerspective` |
| `ZomboidNormalMaper.cpp` | Perspectiva + filtros | Warp, Sobel, Gauss, CLAHE, umbral del alfa, mapa de normales |
| `ZomboidNormalMapper.py` | Versión de escritorio de lo anterior | Todo lo anterior + iluminación Lambert e interfaz PySide6 |
| `UI/launcher.py` | Interfaz de interacción con los temas | Compila y ejecuta los programas C++ desde una ventana (PySide6) |

---

## Requisitos

- **OpenCV 4.x** para los programas C++ (`Segmentacion2.cpp` usa structured bindings, por lo que requiere C++17). `Caracterizacion.cpp` usa SIFT, disponible desde OpenCV 4.4. `CameraCalib.cpp` contiene soporte condicional para OpenCV 5.
- **Python 3** con `PySide6`, `opencv-python` y `numpy` para la herramienta en Python (Zomboid). La interfaz `UI/` solo necesita `PySide6` y, para compilar desde ella, `g++` y `pkg-config` con OpenCV.
- Varios programas C++ usan `system("pause")` o rutas de Windows; funcionan igual en Linux/macOS salvo esas líneas.

Compilación genérica en Linux/macOS:

```bash
g++ -std=c++17 archivo.cpp -o programa `pkg-config --cflags --libs opencv4`
```

En Visual Studio basta con enlazar OpenCV al proyecto y compilar el archivo.

---

## Programas

### 1. Calibración de cámara — `CameraCalib.cpp`

Toda lente deforma la imagen. Calibrar es estimar matemáticamente esa deformación para corregirla. El programa obtiene:

- **Matriz intrínseca K (3×3):** distancia focal `(fx, fy)` y punto principal `(cx, cy)`.
- **Coeficientes de distorsión** `(k1, k2, p1, p2, k3)`: radial (barril/cojín) y tangencial.
- **Extrínsecos** (rotación y traslación del tablero) por cada imagen.

**Flujo:**

1. Define las coordenadas 3D del tablero (plano, `Z = 0`).
2. Para cada foto en `./images/*.jpg` detecta las esquinas interiores (`findChessboardCorners`) y las refina a precisión subpíxel (`cornerSubPix`).
3. `calibrateCamera` calcula K, D y el error RMS de reproyección.
4. Exporta todo a `calibration.yml` (`FileStorage`).
5. Abre la cámara y muestra, lado a lado, el video original y el corregido. Los mapas de corrección se calculan una sola vez (`getOptimalNewCameraMatrix` + `initUndistortRectifyMap`) y se aplican con `remap` en cada fotograma.

### 2. Filtro de caja — `BoxFilter.cpp`

Filtrar una imagen es **convolucionarla** con un kernel que se desliza sobre ella: se multiplican los pesos por los píxeles cubiertos, se suman, y el resultado reemplaza al píxel central. El filtro de caja usa pesos iguales `1/(k·k)`, es decir, promedia el vecindario: es un filtro **paso bajo** que reduce el ruido a costa de desenfocar bordes.

Pide la ruta de la imagen y el tamaño del kernel (impar; se corrige si es par), convierte a grises, aplica `boxFilter` y guarda `gris.jpg` y `filtro_caja.jpg`.

### 3. Erosión y dilatación — `erosion_dilatacion.cpp`

La morfología matemática sondea la imagen con un **elemento estructurante**:

- **Erosión:** el píxel sobrevive solo si todo su vecindario es objeto (AND / mínimo local). Reduce objetos, elimina ruido de "sal" y separa objetos que se tocan.
- **Dilatación:** el píxel pasa a objeto si al menos un vecino lo es (OR / máximo local). Engrosa objetos y rellena huecos pequeños.

Funciona con una imagen o con la webcam (Enter sin ruta). Barras de control: `Tamano` (lado = 2·valor+1), `Forma` (cuadrado / disco / cruz), `Binaria` y `Umbral`. Teclas: `q`/`ESC` salir, `s` guarda `resultado.png`.

### 4. Transformada de Fourier — `Fourier.cpp`

Descompone la imagen en ondas seno/coseno 2D ortogonales: las **frecuencias bajas** representan zonas suaves y las **altas** bordes y texturas. Implementa los 8 pasos: cargar en grises → padding óptimo (`getOptimalDFTSize`) → imagen compleja → `dft` → magnitud → escala logarítmica → reorganizar cuadrantes (DC al centro) → normalizar y mostrar.

```bash
./fourier                 # pide la ruta por consola
./fourier imagen.jpg      # o como argumento
```

### 5. CNN y Vision Transformer — `Segmentacion2.cpp`

Reproduce las etapas de ambas arquitecturas **sin pesos entrenados**:

- **CNN:** convolución (Sobel X/Y, Laplaciano, Gauss) → ReLU → max-pooling 2×2.
- **ViT:** división en parches → aplanado → embedding lineal (proyección aleatoria fija) → codificación posicional sinusoidal → token CLS → autoatención `softmax(QKᵀ/T)·V`.
- **Segmentación:** k-means sobre los tokens contextualizados, pintando una máscara por parche.

Genera `out_panel_vit.png`, `out_segmentacion.png`, `out_atencion.png` y `out_cnn_features.png`. El tamaño de parche debe dividir a 224.

### 6. Histogramas — `Histogramas.cpp`

Un **histograma** cuenta cuántos píxeles tiene cada nivel de intensidad (eje X: intervalos o *bins*; eje Y: frecuencia). Una imagen oscura concentra los picos a la izquierda, una brillante a la derecha y una bien contrastada reparte los valores por todo el rango.

Muestra el histograma de la imagen en grises, el de cada canal B/G/R, y la versión **normalizada** (probabilidades). En la consola imprime la **tabla de frecuencias** con la regla de Sturges (`k = 1 + 3.3·log10(n)`): límites, marca de clase, frecuencia absoluta, relativa y acumulada, la moda y un diagnóstico (oscura / brillante / contraste). Compara además la imagen original con la **ecualización** global (`equalizeHist`) y con `CLAHE`. Barras: `Bins`, `Normalizar`, `Log`. Teclas: `q`/`ESC` salir, `s` guarda `imagenes.png` e `histogramas.png`.

```bash
./histogramas                 # pide la ruta por consola
./histogramas imagen.jpg      # o como argumento
```

### 7. Umbralización y binarización — `Umbralizacion.cpp`

**Umbralizar** es comparar cada píxel con un umbral `T` para separar el objeto del fondo; si la salida solo tiene 0 y 255 se habla de **binarización**. El programa muestra:

- **Global manual** (barra `Umbral`) y **global con Otsu** (el umbral se calcula solo minimizando la varianza dentro de las dos clases).
- Los cinco tipos de regla: `BINARY`, `BINARY_INV`, `TRUNC`, `TOZERO`, `TOZERO_INV` (barra `Tipo`).
- **Adaptativa** por media y gaussiana (`adaptiveThreshold`), con las barras `Bloque` y `C`; resiste sombras e iluminación irregular donde un umbral global falla.
- El histograma con el umbral manual y el de Otsu marcados.

Al iniciar comprueba en consola que la regla escrita a mano, píxel a píxel, da el mismo resultado que `cv::threshold`. Teclas: `q`/`ESC` salir, `s` guarda `umbralizacion.png`.

### 8. Caracterización — `Caracterizacion.cpp`

**Caracterizar** es resumir una imagen con pocas medidas o descripciones. El programa tiene cuatro modos (barra `Modo`):

| Modo | Qué hace |
|---|---|
| 0. Intensidad | Media, varianza, asimetría y entropía del histograma (**cuantitativo**) y calidad perceptual: exposición, nitidez (varianza del Laplaciano), contraste y color medio (**cualitativo**) |
| 1. Objetos | Segmenta con Otsu y mide área, perímetro y circularidad `C = 4πA/P²` por objeto; etiqueta forma y color; calcula las relaciones topológicas **RCC-8** (DC, EC, PO, TPP, NTPP, EQ...) |
| 2. Textura | Matriz de co-ocurrencia **GLCM** y sus propiedades: contraste, homogeneidad, uniformidad (ASM), entropía, correlación |
| 3. Puntos clave | **SIFT**, **ORB** y **BRIEF** (implementado a mano) sobre una copia rotada y escalada de la imagen; cuenta las coincidencias geométricamente correctas |

El modo 3 reproduce la teoría: SIFT y ORB siguen emparejando bien con rotación y escala, mientras que BRIEF, que no es invariante a la rotación, se desploma. SURF no se incluye porque está patentado y vive en el módulo "no libre" de `opencv_contrib`. Teclas: `q`/`ESC` salir, `p` imprime el reporte completo en consola, `s` guarda `caracterizacion.png`.

```bash
./caracterizacion imagen.jpg      # opcionalmente: ./caracterizacion imagen.jpg 3  (modo inicial)
```

### 9. Zomboid Asset Warper — `ZomboidAssetWrapper.cpp`

Los tiles de pared de Project Zomboid están dibujados en perspectiva isométrica. Se enderezan con una **homografía**: cuatro puntos de origen se llevan a las esquinas de un rectángulo (`getPerspectiveTransform` + `warpPerspective`). Es la versión inicial del siguiente programa.

### 10. Zomboid Normal Mapper (C++) — `ZomboidNormalMaper.cpp`

Añade al warp la generación de un **mapa de normales**. La luminosidad se interpreta como altura y su pendiente se mide con derivadas:

```
gris → (invertir) → relleno del fondo → CLAHE → Gauss → Sobel dx, dy
     → n = (-s·dx, ±s·dy, 1)/|n| → color = n·127.5 + 127.5
```

Controles: `Strength`, `Blur`, `Invert`, `FlipGreen` (OpenGL/DirectX), `Contrast` (CLAHE), `FlatBG`, `Save`. Guarda `NNN_Wall.png` y `NNN_Wall_n.png`. Las carpetas `INPUT_DIR` y `OUTPUT_DIR` deben ajustarse a tu equipo.

### 11. Zomboid Normal Mapper (Python) — `ZomboidNormalMapper.py`

Versión de escritorio (PySide6) del programa anterior, con: puntos del warp arrastrables, zoom con rueda, vista de **comparación** con divisor, vista de **luz** interactiva (iluminación difusa de Lambert, `N·L`), presets, deshacer/rehacer, tema oscuro y atajos de teclado (`F1` los muestra).

```bash
pip install -r requirements.txt
python ZomboidNormalMapper.py [imagen.png]
```

---

## Interfaz de interacción (UI)

La carpeta `UI/` contiene un lanzador en **Python + PySide6** para usar los temas vistos sin escribir comandos. No reescribe los programas: los compila y los ejecuta tal como están.

```bash
pip install -r UI/requirements.txt
python UI/launcher.py
```

**Qué hace al elegir un tema en el árbol de la izquierda:**

1. Muestra su descripción, las teclas de sus ventanas y un formulario con los datos que necesita (imagen con vista previa, tamaño de kernel, parche, modo, etc.).
2. Lo **compila** con `g++` + `pkg-config` solo si el ejecutable no existe o el código cambió (los binarios quedan en `UI/build/`).
3. Lo **ejecuta** entregándole los datos como argumentos o por `stdin`. Sus ventanas de OpenCV se abren aparte y se cierran con `q` / `ESC`.
4. Muestra la **consola**, los archivos que generó (pestaña *Resultados*; se guardan en `UI/salidas/<programa>/`) y el **código fuente** con colores.

**Archivos:**

| Archivo | Función |
|---|---|
| `UI/launcher.py` | La ventana: árbol de temas, formulario, compilación, ejecución, consola, resultados y código |
| `UI/programas.py` | Registro de programas: ruta, descripción, campos del formulario y cómo se convierten en argumentos / `stdin` |
| `UI/requirements.txt` | Dependencias de la interfaz (`PySide6`) |

Para añadir un programa nuevo basta con agregar una entrada `Programa(...)` en `UI/programas.py`.

**Si no usas `g++` / `pkg-config`** (por ejemplo, Visual Studio en Windows): en *Configuración del compilador…* se puede escribir un comando propio (`{src}` = archivo `.cpp`, `{out}` = ejecutable), o con *Elegir…* usar un `.exe` que ya compilaste. Sin `stdbuf` (Windows), la salida de consola de un programa puede aparecer solo al cerrarlo.

---

## Mapa de conceptos

| Concepto | Dónde se ve |
|---|---|
| Convolución y kernels | `BoxFilter.cpp`, `Segmentacion2.cpp` (capa CNN), Sobel/Gauss en el Zomboid Mapper |
| Filtros paso bajo / paso alto | Caja y Gauss (suavizado), Sobel y Laplaciano (bordes) |
| Binarización / umbralización | `Umbralizacion.cpp` (global, Otsu, adaptativa), `erosion_dilatacion.cpp` (`threshold`), máscara del alfa en el Zomboid Mapper |
| Histograma y ecualización | `Histogramas.cpp` (`equalizeHist`, CLAHE), `Caracterizacion.cpp` (estadísticas), CLAHE en el Zomboid Mapper |
| Morfología | `erosion_dilatacion.cpp` |
| Espacio afín / perspectiva | `ZomboidAssetWrapper.cpp` y `ZomboidNormalMaper.*` |
| Dominio de frecuencia | `Fourier.cpp` |
| Modelo de cámara y distorsión | `CameraCalib.cpp` |
| Segmentación y Transformers | `Segmentacion2.cpp` |
| Caracterización cuantitativa (área, perímetro, circularidad, GLCM) | `Caracterizacion.cpp` (modos 1 y 2) |
| Caracterización cualitativa (forma, color, calidad, RCC-8) | `Caracterizacion.cpp` (modos 0 y 1) |
| Detección y descripción de puntos clave sin IA (SIFT, ORB, BRIEF) | `Caracterizacion.cpp` (modo 3) |
| Interfaz para interactuar con todos los temas | `UI/launcher.py` |
