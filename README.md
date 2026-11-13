# AV-Playground
# Tomas Alejandro Silva Correal
Playground general para los conceptos trabajados en la clase de **Visión Artificial 2026-3**.

Cada programa es una implementación práctica y autocontenida de un tema visto en clase, escrita con **OpenCV** (mayormente en archivos CPP junto a un programa extra realizado en Python).
---

## Estructura del repositorio

### Programas CPP
Se presentan programas respecto a los siguientes topicos:
 - Calibración de camara
 - Filtros (Caja, erosión y dilatación
 - Segmentación
 - Transformadas
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
| `Fourier.cpp` | Transformada de Fourier | Dominio espacial vs. frecuencia, DFT, espectro de magnitud |
| `Segmentacion2.cpp` | CNN y Vision Transformer | Convolución + ReLU + pooling, parches, atención, k-means |
| `ZomboidAssetWrapper.cpp` | Transformaciones de perspectiva | `getPerspectiveTransform`, `warpPerspective` |
| `ZomboidNormalMaper.cpp` | Perspectiva + filtros | Warp, Sobel, Gauss, CLAHE, umbral del alfa, mapa de normales |
| `ZomboidNormalMapper.py` | Versión de escritorio de lo anterior | Todo lo anterior + iluminación Lambert e interfaz PySide6 |

---

## Requisitos

- **OpenCV 4.x** para los programas C++ (`Segmentacion2.cpp` usa structured bindings, por lo que requiere C++17). `CameraCalib.cpp` contiene soporte condicional para OpenCV 5.
- **Python 3** con `PySide6`, `opencv-python` y `numpy` para la herramienta en Python.
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

### 6. Zomboid Asset Warper — `ZomboidAssetWrapper.cpp`

Los tiles de pared de Project Zomboid están dibujados en perspectiva isométrica. Se enderezan con una **homografía**: cuatro puntos de origen se llevan a las esquinas de un rectángulo (`getPerspectiveTransform` + `warpPerspective`). Es la versión inicial del siguiente programa.

### 7. Zomboid Normal Mapper (C++) — `ZomboidNormalMaper.cpp`

Añade al warp la generación de un **mapa de normales**. La luminosidad se interpreta como altura y su pendiente se mide con derivadas:

```
gris → (invertir) → relleno del fondo → CLAHE → Gauss → Sobel dx, dy
     → n = (-s·dx, ±s·dy, 1)/|n| → color = n·127.5 + 127.5
```

Controles: `Strength`, `Blur`, `Invert`, `FlipGreen` (OpenGL/DirectX), `Contrast` (CLAHE), `FlatBG`, `Save`. Guarda `NNN_Wall.png` y `NNN_Wall_n.png`. Las carpetas `INPUT_DIR` y `OUTPUT_DIR` deben ajustarse a tu equipo.

### 8. Zomboid Normal Mapper (Python) — `ZomboidNormalMapper.py`

Versión de escritorio (PySide6) del programa anterior, con: puntos del warp arrastrables, zoom con rueda, vista de **comparación** con divisor, vista de **luz** interactiva (iluminación difusa de Lambert, `N·L`), presets, deshacer/rehacer, tema oscuro y atajos de teclado (`F1` los muestra).

```bash
pip install -r requirements.txt
python ZomboidNormalMapper.py [imagen.png]
```

---

## Mapa de conceptos

| Concepto | Dónde se ve |
|---|---|
| Convolución y kernels | `BoxFilter.cpp`, `Segmentacion2.cpp` (capa CNN), Sobel/Gauss en el Zomboid Mapper |
| Filtros paso bajo / paso alto | Caja y Gauss (suavizado), Sobel y Laplaciano (bordes) |
| Binarización / umbralización | `erosion_dilatacion.cpp` (`threshold`), máscara del alfa en el Zomboid Mapper |
| Histograma y ecualización | CLAHE en el Zomboid Mapper |
| Morfología | `erosion_dilatacion.cpp` |
| Espacio afín / perspectiva | `ZomboidAssetWrapper.cpp` y `ZomboidNormalMaper.*` |
| Dominio de frecuencia | `Fourier.cpp` |
| Modelo de cámara y distorsión | `CameraCalib.cpp` |
| Segmentación y Transformers | `Segmentacion2.cpp` |
