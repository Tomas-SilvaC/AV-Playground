/**
 * Histogramas.cpp
 * Histogramas de intensidad de una imagen: gris, por canal BGR, normalizado,
 * tabla de frecuencias y mejora de contraste por ecualización.
 *
 * ## Idea general
 * Un histograma resume la distribución de las intensidades de una imagen sin
 * importar DÓNDE está cada píxel: es un gráfico de barras donde el eje X agrupa
 * los valores posibles en intervalos (contenedores o "bins") y el eje Y cuenta
 * cuántos píxeles caen en cada uno.
 *
 *   - Imagen oscura          -> picos cerca de 0   (a la izquierda).
 *   - Imagen brillante       -> picos cerca de 255 (a la derecha).
 *   - Imagen bien contrastada-> valores repartidos por todo el rango.
 *
 * Variantes que muestra este programa:
 *   - Escala de grises: 1 histograma (0 = negro, 255 = blanco).
 *   - Color: un histograma independiente por canal (B, G, R).
 *   - Normalizado: cada cuenta se divide entre el total de píxeles (probabilidad),
 *     lo que permite comparar imágenes de distinto tamaño.
 *   - Ecualización: redistribuye las intensidades usando la función de
 *     distribución acumulada (CDF) para aprovechar todo el rango y ganar contraste.
 *       * equalizeHist -> global (una sola transformación para toda la imagen).
 *       * CLAHE        -> adaptativa por regiones, con límite de contraste.
 *
 * ## Tabla de frecuencias (consola)
 * Se construye con los pasos clásicos de estadística descriptiva:
 *   1. Número de intervalos:  k = 1 + 3.3 * log10(n)     (regla de Sturges, n = píxeles)
 *   2. Amplitud de cada uno:  (máximo - mínimo + 1) / k, redondeada hacia arriba.
 *   3. Límite inferior del primer intervalo = mínimo observado.
 *   4. Intervalos contiguos, sin huecos ni solapamientos y de igual amplitud.
 *   5. Contar los píxeles de cada intervalo.
 * Por intervalo se reporta: límites, marca de clase (punto medio), frecuencia
 * absoluta, relativa (%) y acumulada.
 * (Se usa +1 en la amplitud porque las intensidades son enteros: así el máximo
 * también queda dentro del último intervalo.)
 *
 * ## Compilar
 *   g++ -std=c++17 Histogramas.cpp -o histogramas `pkg-config --cflags --libs opencv4`
 *
 * ## Ejecutar
 *   ./histogramas                -> el programa pide la ruta (copiar y pegar)
 *   ./histogramas imagen.jpg     -> también acepta la ruta como argumento
 *
 * Ventanas:
 *   "Imagenes"    : Gris original | Ecualizada (equalizeHist) | CLAHE
 *   "Histogramas" : gris, ecualizada, CLAHE y canales B/G/R de la imagen original.
 *
 * Barras de "Histogramas":
 *   Bins      : cantidad de contenedores (2..256). 256 = un bin por intensidad.
 *   Normalizar: 0 = conteo de píxeles, 1 = probabilidad (suma 1)
 *   Log       : 0 = eje Y lineal, 1 = eje Y logarítmico (log(1+f)). Útil cuando un
 *               pico enorme (p. ej. fondo blanco en 255) aplasta al resto.
 *
 * Teclas: q o ESC = salir, s = guardar imagenes.png e histogramas.png
 */
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

static const string VENT_IMG  = "Imagenes";
static const string VENT_HIST = "Histogramas";

// Tamaño de cada gráfico de histograma (en píxeles).
static const int HIST_W = 400;
static const int HIST_H = 260;


// Quita espacios y comillas del inicio/final (Windows pone comillas
// al usar "Copiar como ruta de acceso").
static string limpiarRuta(string s) {
    const string basura = " \t\r\n\"'";
    size_t ini = s.find_first_not_of(basura);
    if (ini == string::npos) return "";
    size_t fin = s.find_last_not_of(basura);
    return s.substr(ini, fin - ini + 1);
}

// Convierte a BGR (si es de un canal) y dibuja una franja negra con un título arriba.
// Sirve para concatenar paneles de distinto tipo y rotularlos.
static Mat etiquetar(const Mat& img, const string& texto) {
    Mat out;
    if (img.channels() == 1) cvtColor(img, out, COLOR_GRAY2BGR);
    else out = img.clone();
    rectangle(out, Point(0, 0), Point(out.cols, 28), Scalar(0, 0, 0), FILLED);
    putText(out, texto, Point(8, 20), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(255, 255, 255), 1, LINE_AA);
    return out;
}

// Redimensiona a un ancho fijo conservando la proporción.
static Mat redimensionar(const Mat& img, int ancho) {
    int alto = max(1, img.rows * ancho / img.cols);
    Mat out;
    resize(img, out, Size(ancho, alto));
    return out;
}


// Calcula el histograma de UN canal (imagen CV_8UC1) con 'bins' contenedores sobre
// el rango [0, 256). Devuelve un vector columna CV_32F de 'bins' filas.
// Si 'normalizar' es true, divide entre el total de píxeles (probabilidades).
static Mat calcularHistograma(const Mat& canal, int bins, bool normalizar) {
    float rango[] = {0.f, 256.f};
    const float* rangos[] = {rango};
    int canales[] = {0};
    int tam[] = {bins};
    Mat hist;
    calcHist(&canal, 1, canales, noArray(), hist, 1, tam, rangos, true, false);
    if (normalizar) hist /= (double)canal.total();
    return hist;
}

// Dibuja uno o varios histogramas (un vector columna por serie) en un mismo gráfico.
// Una sola serie se dibuja como barras; varias, como curvas superpuestas (una por
// color) para poder comparar canales. El eje Y se escala al máximo de todas las
// series, así que las alturas son comparables entre ellas.
static Mat dibujarHistograma(const vector<Mat>& series, const vector<Scalar>& colores,
                             const string& titulo, bool normalizar, bool logY) {
    const int margen = 28;           // espacio superior para el título
    const int base   = HIST_H - 18;  // línea del eje X (deja sitio para las marcas)
    Mat lienzo(HIST_H, HIST_W, CV_8UC3, Scalar(30, 30, 30));

    // Escala vertical: lineal o logarítmica. En log se usa log(1 + f) para que f = 0
    // dé 0 y las cuentas pequeñas sigan siendo visibles junto a los picos grandes.
    auto escala = [logY](double v) { return logY ? log1p(v) : v; };

    double maxVal = 0;
    for (const Mat& s : series) {
        double mx; minMaxLoc(s, nullptr, &mx);
        maxVal = max(maxVal, mx);
    }
    if (maxVal <= 0) maxVal = 1;
    double maxEsc = escala(maxVal);
    // En modo normalizado f < 1 y log(1+f) casi no se distingue; se reescala por
    // el máximo, así que la forma sigue siendo legible.

    int bins = series[0].rows;
    double anchoBin = (double)HIST_W / bins;
    int altoUtil = base - margen - 4;

    for (size_t k = 0; k < series.size(); ++k) {
        vector<Point> puntos;
        for (int i = 0; i < bins; ++i) {
            double v = series[k].at<float>(i);
            int x0 = cvRound(i * anchoBin);
            // Con barras anchas se deja 1 px de separación entre ellas.
            int x1 = max(x0 + 1, cvRound((i + 1) * anchoBin)) - (anchoBin > 3 ? 1 : 0);
            int y  = base - cvRound(escala(v) / maxEsc * altoUtil);
            if (series.size() == 1) {
                rectangle(lienzo, Point(x0, y), Point(x1, base), colores[k], FILLED);
            } else {
                puntos.push_back(Point((x0 + x1) / 2, y));
            }
        }
        if (series.size() > 1) polylines(lienzo, puntos, false, colores[k], 1, LINE_AA);
    }

    // Eje X con marcas en 0, 64, 128, 192 y 255 (intensidades).
    line(lienzo, Point(0, base), Point(HIST_W, base), Scalar(200, 200, 200), 1);
    const int marcas[] = {0, 64, 128, 192, 255};
    for (int m : marcas) {
        int x = min(HIST_W - 1, cvRound((m + 0.5) / 256.0 * HIST_W));
        line(lienzo, Point(x, base), Point(x, base + 4), Scalar(200, 200, 200), 1);
        putText(lienzo, to_string(m), Point(max(0, min(x - 10, HIST_W - 24)), HIST_H - 4),
                FONT_HERSHEY_SIMPLEX, 0.35, Scalar(200, 200, 200), 1, LINE_AA);
    }

    // Título y valor máximo del eje Y.
    ostringstream pie;
    if (normalizar) pie << fixed << setprecision(4) << "max=" << maxVal;
    else            pie << "max=" << cvRound(maxVal);
    if (logY) pie << " (log)";
    putText(lienzo, titulo, Point(8, 18), FONT_HERSHEY_SIMPLEX, 0.5,
            Scalar(255, 255, 255), 1, LINE_AA);
    putText(lienzo, pie.str(), Point(HIST_W - 120, 18), FONT_HERSHEY_SIMPLEX, 0.4,
            Scalar(180, 180, 180), 1, LINE_AA);
    return lienzo;
}

// Diagnóstico textual a partir del brillo medio y la desviación estándar:
// resume lo que el histograma "dice" de la imagen (ver ejemplo de la clase).
static string diagnostico(const Mat& gris) {
    Scalar media, desv;
    meanStdDev(gris, media, desv);
    double mu = media[0], sigma = desv[0];

    string brillo = (mu < 85) ? "oscura (histograma hacia la izquierda)"
                  : (mu > 170) ? "brillante (histograma hacia la derecha)"
                               : "de brillo medio";
    string contraste = (sigma < 40) ? "bajo contraste (histograma estrecho)"
                     : (sigma > 70) ? "alto contraste (histograma muy extendido)"
                                    : "contraste moderado";
    ostringstream os;
    os << fixed << setprecision(1)
       << "media=" << mu << ", desv.est.=" << sigma << " -> " << brillo << "; " << contraste;
    return os.str();
}

// Imprime la tabla de frecuencias de la imagen en gris siguiendo la regla de Sturges
// (ver "Tabla de frecuencias" en la cabecera). Parte del histograma de 256 bins, que
// ya contiene el conteo exacto de cada intensidad, y lo agrupa en k intervalos.
static void imprimirTablaFrecuencias(const Mat& gris) {
    const double n = (double)gris.total();
    double mn, mx;
    minMaxLoc(gris, &mn, &mx);
    int minimo = (int)mn, maximo = (int)mx;

    // Paso 1: número de intervalos (se redondea al entero más cercano, mínimo 1).
    int k = max(1, cvRound(1.0 + 3.3 * log10(n)));
    // Paso 2: amplitud (entera, hacia arriba para cubrir todo el rango).
    int rangoValores = maximo - minimo + 1;
    int amplitud = max(1, (int)ceil((double)rangoValores / k));
    // Si la amplitud redondeada hace que sobren intervalos, se reduce k.
    k = (int)ceil((double)rangoValores / amplitud);

    Mat hist = calcularHistograma(gris, 256, false);

    cout << "\n=== Tabla de frecuencias (imagen en grises) ===\n";
    cout << "Pixeles n = " << (long long)n << " | minimo = " << minimo
         << " | maximo = " << maximo << "\n";
    cout << "Intervalos k = 1 + 3.3*log10(n) = " << k
         << " | amplitud = " << amplitud << "\n\n";
    cout << left << setw(7) << "Clase" << setw(14) << "Intervalo"
         << right << setw(10) << "Marca" << setw(12) << "f"
         << setw(10) << "f (%)" << setw(12) << "F acum." << "\n";
    cout << string(65, '-') << "\n";

    long long acumulada = 0;
    for (int c = 0; c < k; ++c) {
        // Intervalos contiguos [inf, sup]: sin huecos ni solapamientos.
        int inf = minimo + c * amplitud;
        int sup = min(maximo, inf + amplitud - 1);
        long long f = 0;
        for (int v = inf; v <= sup; ++v) f += cvRound(hist.at<float>(v));
        acumulada += f;
        double marca = (inf + sup) / 2.0;

        ostringstream rango;
        rango << "[" << inf << ", " << sup << "]";
        cout << left << setw(7) << (c + 1) << setw(14) << rango.str()
             << right << setw(10) << fixed << setprecision(1) << marca
             << setw(12) << f
             << setw(10) << setprecision(2) << (100.0 * f / n)
             << setw(12) << acumulada << "\n";
    }

    // Respuesta tipo "¿cuál fue el valor más frecuente?" de la clase.
    Point pico;
    minMaxLoc(hist, nullptr, nullptr, nullptr, &pico);
    cout << "\nIntensidad mas frecuente (moda): " << pico.y
         << " (" << cvRound(hist.at<float>(pico.y)) << " pixeles)\n";
    cout << "Diagnostico: " << diagnostico(gris) << "\n\n";
}


int main(int argc, char** argv) {
    // ---- Pedir la ruta (argumento o consola) ----
    string ruta;
    if (argc >= 2) {
        ruta = limpiarRuta(argv[1]);
    } else {
        cout << "=== Histogramas ===\n";
        cout << "Ingrese la ruta de la imagen: ";
        getline(cin, ruta);
        ruta = limpiarRuta(ruta);
    }

    Mat color = imread(ruta);
    if (color.empty()) {
        cerr << "\nError: no se pudo cargar la imagen en la ruta:\n" << ruta << "\n";
        cerr << "Verifica que la ruta sea correcta y que el archivo exista.\n";
        return 1;
    }
    color = redimensionar(color, 420);

    // ---- Imágenes de trabajo ----
    // El histograma de intensidad se calcula sobre un solo canal (luminosidad).
    Mat gris;
    cvtColor(color, gris, COLOR_BGR2GRAY);

    // Ecualización global: mapea cada nivel con la CDF normalizada a [0, 255].
    Mat ecualizada;
    equalizeHist(gris, ecualizada);

    // CLAHE: ecualiza por baldosas (8x8) recortando el histograma de cada una
    // (clipLimit) para no amplificar el ruido en zonas uniformes.
    Mat clahe;
    createCLAHE(3.0, Size(8, 8))->apply(gris, clahe);

    // Canales B, G, R de la imagen original.
    vector<Mat> bgr;
    split(color, bgr);

    imprimirTablaFrecuencias(gris);

    // ---- Panel de imágenes (no cambia con las barras, se arma una vez) ----
    Mat panelImg;
    hconcat(vector<Mat>{etiquetar(gris, "Gris original"),
                        etiquetar(ecualizada, "Ecualizada (equalizeHist)"),
                        etiquetar(clahe, "CLAHE")}, panelImg);

    // ---- Ventanas y barras ----
    namedWindow(VENT_IMG, WINDOW_AUTOSIZE);
    namedWindow(VENT_HIST, WINDOW_AUTOSIZE);
    createTrackbar("Bins", VENT_HIST, nullptr, 256);
    createTrackbar("Normalizar", VENT_HIST, nullptr, 1);
    createTrackbar("Log", VENT_HIST, nullptr, 1);
    setTrackbarPos("Bins", VENT_HIST, 256);
    setTrackbarPos("Normalizar", VENT_HIST, 0);
    setTrackbarPos("Log", VENT_HIST, 0);

    imshow(VENT_IMG, panelImg);

    const Scalar GRIS_C(220, 220, 220);
    Mat panelHist;
    while (true) {
        // ---- Leer parámetros ----
        // Un histograma necesita al menos 2 contenedores para tener forma.
        int bins = max(2, getTrackbarPos("Bins", VENT_HIST));
        bool norm = getTrackbarPos("Normalizar", VENT_HIST) == 1;
        bool logY = getTrackbarPos("Log", VENT_HIST) == 1;

        // ---- Calcular histogramas ----
        Mat hGris = calcularHistograma(gris, bins, norm);
        Mat hEq   = calcularHistograma(ecualizada, bins, norm);
        Mat hCl   = calcularHistograma(clahe, bins, norm);
        vector<Mat> hCanales = {calcularHistograma(bgr[0], bins, norm),
                                calcularHistograma(bgr[1], bins, norm),
                                calcularHistograma(bgr[2], bins, norm)};

        // ---- Dibujar: 2 x 2 gráficos ----
        string sufijo = " (" + to_string(bins) + " bins" + (norm ? ", norm.)" : ")");
        Mat g1 = dibujarHistograma({hGris}, {GRIS_C}, "Gris original" + sufijo, norm, logY);
        Mat g2 = dibujarHistograma({hEq},   {GRIS_C}, "Ecualizada" + sufijo, norm, logY);
        Mat g3 = dibujarHistograma({hCl},   {GRIS_C}, "CLAHE" + sufijo, norm, logY);
        // En OpenCV el orden de color es BGR: azul, verde, rojo.
        Mat g4 = dibujarHistograma(hCanales,
                                   {Scalar(255, 80, 80), Scalar(80, 255, 80), Scalar(80, 80, 255)},
                                   "Canales B/G/R" + sufijo, norm, logY);
        Mat fila1, fila2;
        hconcat(g1, g2, fila1);
        hconcat(g3, g4, fila2);
        vconcat(fila1, fila2, panelHist);
        imshow(VENT_HIST, panelHist);

        int tecla = waitKey(30) & 0xFF;
        if (tecla == 'q' || tecla == 27) break;
        if (tecla == 's') {
            imwrite("imagenes.png", panelImg);
            imwrite("histogramas.png", panelHist);
            cout << "Guardado: imagenes.png e histogramas.png" << endl;
        }
    }

    return 0;
}
