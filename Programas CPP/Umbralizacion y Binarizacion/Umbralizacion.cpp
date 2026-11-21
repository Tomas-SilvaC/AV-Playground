/**
 * Umbralizacion.cpp
 * Umbralización (thresholding) y binarización de una imagen en escala de grises:
 * global (fija y Otsu) y adaptativa (media y gaussiana).
 *
 * ## Idea general
 * Umbralizar es comparar la intensidad de cada píxel con un valor llamado umbral T
 * y decidir qué valor de salida le toca. Es la técnica de segmentación más simple:
 * separa el objeto de interés del fondo. Cuando la salida solo tiene dos valores
 * (0 y 255) se habla de binarización y el resultado es una imagen binaria / máscara.
 *
 * Regla básica (THRESH_BINARY):
 *
 *               | 255   si  I(x, y) >  T
 *   B(x, y)  =  |
 *               |   0   en otro caso
 *
 * ## Variantes de la regla (parámetro "Tipo")
 *   - THRESH_BINARY     : > T -> 255; el resto -> 0.
 *   - THRESH_BINARY_INV : > T -> 0;   el resto -> 255 (objeto oscuro sobre fondo claro).
 *   - THRESH_TRUNC      : > T -> T;   el resto no cambia (limita los brillos).
 *   - THRESH_TOZERO     : <= T -> 0;  el resto no cambia (conserva solo lo brillante).
 *   - THRESH_TOZERO_INV : > T -> 0;   el resto no cambia (conserva solo lo oscuro).
 * Las tres últimas NO producen una imagen binaria: dejan tonos de gris.
 *
 * ## Estrategias
 *   1. Umbralización global: un único T para toda la imagen. Funciona bien con
 *      iluminación uniforme y buen contraste objeto/fondo.
 *        - Manual: T lo elige el usuario con la barra "Umbral".
 *        - Otsu  : T se calcula solo a partir del histograma, eligiendo el valor que
 *                  minimiza la varianza DENTRO de los dos grupos (fondo y objeto)
 *                  o, de forma equivalente, maximiza la varianza ENTRE ellos.
 *                  Funciona mejor con histogramas bimodales (dos "montañas").
 *   2. Umbralización adaptativa (local): T se calcula para cada píxel a partir de su
 *      vecindario bloque x bloque, restando una constante C:
 *        - ADAPTIVE_THRESH_MEAN_C     : T(x,y) = media del vecindario - C
 *        - ADAPTIVE_THRESH_GAUSSIAN_C : T(x,y) = media ponderada gaussiana - C
 *      Resiste iluminación irregular, sombras y gradientes, donde un T global fallaría.
 *      Solo admite los tipos BINARY y BINARY_INV (aquí: BINARY_INV si "Tipo" = 1).
 *
 * ## Compilar
 *   g++ -std=c++17 Umbralizacion.cpp -o umbralizacion `pkg-config --cflags --libs opencv4`
 *
 * ## Ejecutar
 *   ./umbralizacion              -> el programa pide la ruta (copiar y pegar)
 *   ./umbralizacion imagen.jpg   -> también acepta la ruta como argumento
 *
 * Al iniciar, comprueba en consola que la regla escrita a mano (bucle píxel a píxel)
 * da exactamente el mismo resultado que cv::threshold.
 *
 * Barras de la ventana:
 *   Umbral : T para la umbralización global manual (0..255)
 *   Tipo   : 0 = BINARY, 1 = BINARY_INV, 2 = TRUNC, 3 = TOZERO, 4 = TOZERO_INV
 *   Bloque : tamaño del vecindario adaptativo (lado = 2*valor + 3, siempre impar)
 *   C      : constante restada al umbral local (0..30)
 *
 * Paneles: Gris | Global manual | Otsu | Adaptativa media | Adaptativa gaussiana |
 *          Histograma con T manual (rojo) y T de Otsu (verde).
 *
 * Teclas: q o ESC = salir, s = guardar umbralizacion.png
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

static const string VENTANA = "Umbralizacion y binarizacion";

// Tipos de umbralización global disponibles (se indexan con la barra "Tipo").
static const int TIPOS[5] = {THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC,
                             THRESH_TOZERO, THRESH_TOZERO_INV};
static const char* NOMBRES_TIPOS[5] = {"BINARY", "BINARY_INV", "TRUNC",
                                       "TOZERO", "TOZERO_INV"};

// Ancho de cada panel (3 columnas x 2 filas).
static const int PANEL_W = 360;


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
static Mat etiquetar(const Mat& img, const string& texto) {
    Mat out;
    if (img.channels() == 1) cvtColor(img, out, COLOR_GRAY2BGR);
    else out = img.clone();
    rectangle(out, Point(0, 0), Point(out.cols, 28), Scalar(0, 0, 0), FILLED);
    putText(out, texto, Point(8, 20), FONT_HERSHEY_SIMPLEX, 0.5,
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


// Umbralización global escrita a mano, píxel a píxel, para los 5 tipos.
// Es la regla de la cabecera sin ayuda de OpenCV: sirve para ver que no hay magia.
static Mat umbralManual(const Mat& gris, int T, int tipo) {
    Mat out(gris.size(), CV_8UC1);
    for (int y = 0; y < gris.rows; ++y) {
        for (int x = 0; x < gris.cols; ++x) {
            int v = gris.at<uchar>(y, x);
            int r = 0;
            switch (tipo) {
                case THRESH_BINARY:     r = (v > T) ? 255 : 0;  break;
                case THRESH_BINARY_INV: r = (v > T) ? 0 : 255;  break;
                case THRESH_TRUNC:      r = (v > T) ? T : v;    break;
                case THRESH_TOZERO:     r = (v > T) ? v : 0;    break;
                case THRESH_TOZERO_INV: r = (v > T) ? 0 : v;    break;
            }
            out.at<uchar>(y, x) = (uchar)r;
        }
    }
    return out;
}

// Comprueba la regla manual contra cv::threshold con un umbral de ejemplo (127)
// y todos los tipos; imprime el resultado en consola.
static void verificarContraOpenCV(const Mat& gris) {
    cout << "Verificacion: regla manual vs cv::threshold (T = 127)\n";
    for (int i = 0; i < 5; ++i) {
        Mat esperado;
        threshold(gris, esperado, 127, 255, TIPOS[i]);
        Mat manual = umbralManual(gris, 127, TIPOS[i]);
        int difs = countNonZero(manual != esperado);
        cout << "  " << NOMBRES_TIPOS[i] << ": "
             << (difs == 0 ? "identico" : to_string(difs) + " pixeles distintos") << "\n";
    }
    cout << endl;
}

// Dibuja el histograma de intensidades (256 bins) con dos líneas verticales:
// el umbral manual (rojo) y el de Otsu (verde). Permite ver DÓNDE corta cada umbral
// y por qué Otsu cae entre las dos "montañas" de un histograma bimodal.
static Mat histogramaConUmbrales(const Mat& gris, int Tmanual, double Totsu, int alto) {
    float rango[] = {0.f, 256.f};
    const float* rangos[] = {rango};
    int canales[] = {0}, tam[] = {256};
    Mat hist;
    calcHist(&gris, 1, canales, noArray(), hist, 1, tam, rangos);

    // Se usa escala logarítmica para que un pico enorme no aplaste al resto.
    Mat logHist;
    log(hist + 1, logHist);
    double mx;
    minMaxLoc(logHist, nullptr, &mx);
    if (mx <= 0) mx = 1;

    const int base = alto - 18;
    Mat lienzo(alto, PANEL_W, CV_8UC3, Scalar(30, 30, 30));
    for (int i = 0; i < 256; ++i) {
        int x0 = cvRound(i * (double)PANEL_W / 256);
        int x1 = max(x0 + 1, cvRound((i + 1) * (double)PANEL_W / 256));
        int h = cvRound(logHist.at<float>(i) / mx * (base - 34));
        rectangle(lienzo, Point(x0, base - h), Point(x1, base), Scalar(200, 200, 200), FILLED);
    }
    line(lienzo, Point(0, base), Point(PANEL_W, base), Scalar(200, 200, 200), 1);

    int xm = cvRound((Tmanual + 0.5) / 256 * PANEL_W);
    int xo = cvRound((Totsu + 0.5) / 256 * PANEL_W);
    line(lienzo, Point(xm, 28), Point(xm, base), Scalar(60, 60, 255), 2);   // rojo (BGR)
    line(lienzo, Point(xo, 28), Point(xo, base), Scalar(60, 220, 60), 2);   // verde (BGR)
    putText(lienzo, "0", Point(0, alto - 4), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(200, 200, 200), 1, LINE_AA);
    putText(lienzo, "255", Point(PANEL_W - 26, alto - 4), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(200, 200, 200), 1, LINE_AA);

    return etiquetar(lienzo, "Histograma (log)  T=" + to_string(Tmanual) +
                             "  Otsu=" + to_string(cvRound(Totsu)));
}


int main(int argc, char** argv) {
    // ---- Pedir la ruta (argumento o consola) ----
    string ruta;
    if (argc >= 2) {
        ruta = limpiarRuta(argv[1]);
    } else {
        cout << "=== Umbralizacion y binarizacion ===\n";
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

    // La umbralización trabaja sobre un solo canal de intensidad.
    Mat gris;
    cvtColor(redimensionar(color, PANEL_W), gris, COLOR_BGR2GRAY);

    verificarContraOpenCV(gris);

    // ---- Ventana y barras ----
    namedWindow(VENTANA, WINDOW_AUTOSIZE);
    createTrackbar("Umbral", VENTANA, nullptr, 255);
    createTrackbar("Tipo", VENTANA, nullptr, 4);
    createTrackbar("Bloque", VENTANA, nullptr, 25);
    createTrackbar("C", VENTANA, nullptr, 30);
    setTrackbarPos("Umbral", VENTANA, 127);
    setTrackbarPos("Tipo", VENTANA, 0);
    setTrackbarPos("Bloque", VENTANA, 5);
    setTrackbarPos("C", VENTANA, 10);

    Mat panel;
    while (true) {
        // ---- Leer parámetros ----
        int T    = getTrackbarPos("Umbral", VENTANA);
        int tipo = getTrackbarPos("Tipo", VENTANA);
        int bloque = 2 * getTrackbarPos("Bloque", VENTANA) + 3;  // impar y >= 3
        int C    = getTrackbarPos("C", VENTANA);

        // ---- Global manual ----
        Mat global;
        threshold(gris, global, T, 255, TIPOS[tipo]);

        // ---- Global con Otsu ----
        // Con THRESH_OTSU el T pasado se ignora; la función devuelve el T calculado.
        Mat otsu;
        double Totsu = threshold(gris, otsu, 0, 255, TIPOS[tipo] | THRESH_OTSU);

        // ---- Adaptativa (solo BINARY / BINARY_INV) ----
        int tipoAdapt = (tipo == 1) ? THRESH_BINARY_INV : THRESH_BINARY;
        Mat adaMedia, adaGauss;
        adaptiveThreshold(gris, adaMedia, 255, ADAPTIVE_THRESH_MEAN_C, tipoAdapt, bloque, C);
        adaptiveThreshold(gris, adaGauss, 255, ADAPTIVE_THRESH_GAUSSIAN_C, tipoAdapt, bloque, C);

        // ---- Mostrar: 2 filas x 3 columnas ----
        string t = NOMBRES_TIPOS[tipo];
        string ta = (tipo == 1) ? "BINARY_INV" : "BINARY";
        string info = to_string(bloque) + "x" + to_string(bloque) + ", C=" + to_string(C);
        Mat fila1, fila2;
        hconcat(vector<Mat>{etiquetar(gris, "Gris original"),
                            etiquetar(global, "Global T=" + to_string(T) + " " + t),
                            etiquetar(otsu, "Otsu T=" + to_string(cvRound(Totsu)) + " " + t)},
                fila1);
        Mat h = histogramaConUmbrales(gris, T, Totsu, gris.rows);
        hconcat(vector<Mat>{etiquetar(adaMedia, "Adapt. media " + ta + " " + info),
                            etiquetar(adaGauss, "Adapt. gauss. " + ta + " " + info), h},
                fila2);
        vconcat(fila1, fila2, panel);
        imshow(VENTANA, panel);

        int tecla = waitKey(30) & 0xFF;
        if (tecla == 'q' || tecla == 27) break;
        if (tecla == 's') {
            imwrite("umbralizacion.png", panel);
            cout << "Guardado: umbralizacion.png" << endl;
        }
    }

    return 0;
}
