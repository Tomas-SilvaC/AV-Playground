/**
 * Caracterizacion.cpp
 * Caracterización de una imagen: descripción cuantitativa, cualitativa y extracción
 * de puntos clave sin aprendizaje automático (SIFT, ORB y BRIEF).
 *
 * ## Idea general
 * Caracterizar es resumir una imagen con pocas medidas o descripciones que representen
 * su contenido (formas, texturas, objetos) en lugar de trabajar con la matriz de
 * píxeles en bruto.
 *
 *   - Cuantitativa : números medibles y reproducibles (área = 1250 px, circularidad =
 *                    0.87, intensidad media = 142, contraste de textura = 3.6...).
 *   - Cualitativa  : etiquetas comprensibles para una persona ("borrosa", "circular",
 *                    "rojo oscuro", "A está dentro de B"). Aquí se obtienen a partir de
 *                    umbrales sobre las medidas, y se muestran junto a ellas para ver
 *                    cómo un número se traduce en una descripción.
 *   - Sin IA       : algoritmos deterministas (matemática pura, sin datos de
 *                    entrenamiento) que hallan puntos de referencia reconocibles.
 *
 * ## Modos (barra "Modo")
 *   0  INTENSIDAD     Histograma y sus estadísticas: media, varianza (contraste),
 *                     asimetría (skewness) y entropía. Calidad perceptual (IQA):
 *                     exposición, nitidez (varianza del Laplaciano), contraste y color
 *                     cualitativo medio.
 *   1  OBJETOS        Segmenta (Otsu), extrae contornos y mide por objeto:
 *                       Área A       = número de píxeles del objeto
 *                       Perímetro P  = longitud del contorno
 *                       Circularidad C = 4*pi*A / P^2   (1 = círculo perfecto)
 *                       Aspecto y solidez (A / área de la envolvente convexa)
 *                     Etiqueta cualitativamente la forma (circular, alargado, irregular,
 *                     convexo), nombra el color y calcula las relaciones topológicas
 *                     RCC-8 entre objetos: DC (desconectados), EC (conectados por el
 *                     borde), PO (solapamiento parcial), TPP / NTPP (A dentro de B,
 *                     tocando / sin tocar el borde), TPPi / NTPPi (a la inversa), EQ.
 *                     Ojo: la segmentación binaria funde en uno los objetos que se tocan
 *                     o solapan, así que EC/PO/TPP aparecen solo en casos límite (objetos
 *                     separados por 1-2 px, p. ej.). NTPP es el caso típico: un objeto
 *                     dentro del hueco de otro.
 *   2  TEXTURA        Matriz de co-ocurrencia de niveles de gris (GLCM, 16 niveles)
 *                     para una distancia y un ángulo; de ella se calculan contraste,
 *                     homogeneidad, uniformidad (segundo momento angular, ASM),
 *                     entropía y correlación.
 *   3  PUNTOS CLAVE   Compara la imagen con una copia rotada y escalada y empareja sus
 *                     puntos clave (prueba del cociente de Lowe, 0.75):
 *                       SIFT : espacio de escala + Diferencia de Gaussianas; descriptor
 *                              de 128 números (histogramas de gradientes 4x4x8).
 *                       ORB  : FAST + Harris + pirámide + orientación por centroide;
 *                              descriptor BRIEF rotado de 256 bits (distancia Hamming).
 *                       BRIEF: implementado a mano: FAST + 256 comparaciones binarias
 *                              de pares de píxeles aleatorios sobre la imagen suavizada.
 *                              NO es invariante a la rotación: al girar la imagen las
 *                              coincidencias correctas se desploman (ese es el punto).
 *                     Como se conoce la transformación aplicada, se cuenta cuántas
 *                     coincidencias son geométricamente correctas (error < 4 px).
 *                     SURF no se incluye: está patentado y vive en el módulo opencv_contrib
 *                     "no libre", que muchas instalaciones de OpenCV no traen.
 *
 * ## Compilar
 *   g++ -std=c++17 Caracterizacion.cpp -o caracterizacion `pkg-config --cflags --libs opencv4`
 *   (SIFT forma parte de OpenCV desde la versión 4.4.)
 *
 * ## Ejecutar
 *   ./caracterizacion                 -> el programa pide la ruta (copiar y pegar)
 *   ./caracterizacion imagen.jpg      -> también acepta la ruta como argumento
 *   ./caracterizacion imagen.jpg 1    -> y, opcionalmente, el modo con el que empezar (0-3)
 *
 * Barras de la ventana (cada una indica a qué modo pertenece):
 *   Modo           : 0 Intensidad, 1 Objetos, 2 Textura, 3 Puntos clave
 *   M1 AreaMin     : área mínima de un objeto, en milésimas de la imagen (5 = 0.5 %)
 *   M1 Polaridad   : 0 automática, 1 objetos claros sobre fondo oscuro, 2 al revés
 *   M2 Distancia   : separación (en píxeles) entre los dos píxeles de cada pareja
 *   M2 Angulo      : 0 = 0 grados, 1 = 45, 2 = 90, 3 = 135
 *   M3 Detector    : 0 SIFT, 1 ORB, 2 BRIEF (a mano)
 *   M3 Rotacion    : giro de la segunda imagen, en grados
 *   M3 Escala      : escala de la segunda imagen, en % (50..150)
 *   M3 Puntos      : máximo de puntos clave por imagen
 *
 * Teclas: q o ESC = salir, p = imprimir el reporte completo en la consola,
 *         s = guardar caracterizacion.png
 * (El reporte también se imprime solo al cambiar de modo.)
 */
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

static const string VENTANA = "Caracterizacion";
static const int ANCHO_IMG   = 480;   // ancho de la imagen analizada
static const int ALTO_MAX    = 560;   // alto máximo de la imagen analizada
static const int ANCHO_PANEL = 560;   // ancho del panel de texto

static const Scalar C_ACENTO(255, 160, 80);   // azul claro (BGR)
static const Scalar C_TEXTO(235, 235, 235);
static const Scalar C_FONDO(30, 30, 30);

// Resultado de un modo: lo que se dibuja y las líneas de reporte (panel y consola).
struct Resultado {
    Mat lienzo;
    vector<string> reporte;
    vector<string> detalle;   // si no está vacío, la consola imprime esto en vez del reporte
};


// ---------------------------------------------------------------------------
// Utilidades generales
// ---------------------------------------------------------------------------

// Quita espacios y comillas del inicio/final (Windows pone comillas
// al usar "Copiar como ruta de acceso").
static string limpiarRuta(string s) {
    const string basura = " \t\r\n\"'";
    size_t ini = s.find_first_not_of(basura);
    if (ini == string::npos) return "";
    size_t fin = s.find_last_not_of(basura);
    return s.substr(ini, fin - ini + 1);
}

// Número a texto con 'dec' decimales.
static string num(double v, int dec = 2) {
    ostringstream os;
    os << fixed << setprecision(dec) << v;
    return os.str();
}

// Reduce la imagen para que quepa en ANCHO_IMG x ALTO_MAX conservando la proporción.
static Mat ajustarTamano(const Mat& img) {
    double f = min((double)ANCHO_IMG / img.cols, (double)ALTO_MAX / img.rows);
    Mat out;
    resize(img, out, Size(), f, f, f < 1 ? INTER_AREA : INTER_LINEAR);
    return out;
}

// Franja negra con un título arriba de la imagen (se convierte a BGR si hace falta).
static Mat rotular(const Mat& img, const string& texto) {
    Mat out;
    if (img.channels() == 1) cvtColor(img, out, COLOR_GRAY2BGR);
    else out = img.clone();
    rectangle(out, Point(0, 0), Point(out.cols, 24), Scalar(0, 0, 0), FILLED);
    putText(out, texto, Point(8, 17), FONT_HERSHEY_SIMPLEX, 0.5, C_TEXTO, 1, LINE_AA);
    return out;
}

// Arma el lienzo final: imagen a la izquierda y, a la derecha, un panel con el título
// del modo, las líneas del reporte (las que empiezan con "# " son encabezados) y,
// opcionalmente, una imagen extra (histograma, GLCM, máscara...) bajo el texto.
static Mat componer(const Mat& izq, const string& titulo, const vector<string>& lineas,
                    Mat extra) {
    const int LH = 19, margen = 14;
    int anchoUtil = ANCHO_PANEL - 2 * margen;
    if (!extra.empty() && extra.cols > anchoUtil) {
        double f = (double)anchoUtil / extra.cols;
        resize(extra, extra, Size(), f, f, INTER_AREA);
    }
    int yTexto = 26 + (int)lineas.size() * LH + 24;
    int alto = max(izq.rows, yTexto + (extra.empty() ? 0 : extra.rows + 12));

    Mat lienzo(alto, izq.cols + ANCHO_PANEL, CV_8UC3, C_FONDO);
    izq.copyTo(lienzo(Rect(0, 0, izq.cols, izq.rows)));

    int x = izq.cols + margen, y = 24;
    putText(lienzo, titulo, Point(x, y), FONT_HERSHEY_SIMPLEX, 0.55, C_ACENTO, 1, LINE_AA);
    y += 10;
    for (const string& l : lineas) {
        y += LH;
        bool encabezado = l.rfind("# ", 0) == 0;
        putText(lienzo, encabezado ? l.substr(2) : l, Point(x, y), FONT_HERSHEY_SIMPLEX,
                encabezado ? 0.5 : 0.45, encabezado ? C_ACENTO : C_TEXTO, 1, LINE_AA);
    }
    if (!extra.empty()) {
        Mat e = extra.channels() == 1 ? Mat() : extra;
        if (extra.channels() == 1) cvtColor(extra, e, COLOR_GRAY2BGR);
        e.copyTo(lienzo(Rect(x, yTexto, e.cols, e.rows)));
    }
    return lienzo;
}

// Nombre cualitativo de un color medio (BGR): pasa de un espacio continuo a una etiqueta
// discreta mirando el matiz (H), la saturación (S) y el valor/brillo (V) en HSV.
static string nombreColor(const Scalar& bgr) {
    Mat px(1, 1, CV_8UC3, Scalar(cvRound(bgr[0]), cvRound(bgr[1]), cvRound(bgr[2])));
    Mat hsv;
    cvtColor(px, hsv, COLOR_BGR2HSV);
    Vec3b c = hsv.at<Vec3b>(0, 0);
    int h = c[0], s = c[1], v = c[2];       // en OpenCV: H 0..179, S y V 0..255
    if (v < 50) return "negro";
    if (s < 40) return v > 200 ? "blanco" : (v > 110 ? "gris" : "gris oscuro");
    string base;
    if (h < 10 || h >= 170) base = "rojo";
    else if (h < 22)  base = "naranja";
    else if (h < 35)  base = "amarillo";
    else if (h < 85)  base = "verde";
    else if (h < 100) base = "cian";
    else if (h < 130) base = "azul";
    else if (h < 160) base = "violeta";
    else              base = "magenta";
    if (v < 110) base += " oscuro";
    else if (s < 90) base += " desvaido";
    return base;
}


// ---------------------------------------------------------------------------
// Modo 0: INTENSIDAD (histograma + calidad perceptual)
// ---------------------------------------------------------------------------
static Resultado modoIntensidad(const Mat& color, const Mat& gris) {
    // Histograma normalizado p(i) = proporción de píxeles con intensidad i.
    float rango[] = {0.f, 256.f};
    const float* rangos[] = {rango};
    int canales[] = {0}, tam[] = {256};
    Mat hist;
    calcHist(&gris, 1, canales, noArray(), hist, 1, tam, rangos);
    vector<double> p(256);
    for (int i = 0; i < 256; ++i) p[i] = hist.at<float>(i) / (double)gris.total();

    // Momentos: media, varianza, asimetría (3er momento estandarizado) y entropía.
    double media = 0;
    for (int i = 0; i < 256; ++i) media += i * p[i];
    double varianza = 0, m3 = 0, entropia = 0, negros = 0, blancos = 0;
    for (int i = 0; i < 256; ++i) {
        double d = i - media;
        varianza += d * d * p[i];
        m3 += d * d * d * p[i];
        if (p[i] > 0) entropia -= p[i] * log2(p[i]);
        if (i <= 5)   negros += p[i];
        if (i >= 250) blancos += p[i];
    }
    double sigma = sqrt(varianza);
    double asimetria = sigma > 1e-9 ? m3 / (sigma * sigma * sigma) : 0;

    // Nitidez: un enfoque bueno tiene bordes marcados, y el Laplaciano (2a derivada)
    // responde fuerte en ellos; una imagen borrosa da una varianza baja.
    Mat lap;
    Laplacian(gris, lap, CV_64F);
    Scalar mLap, sLap;
    meanStdDev(lap, mLap, sLap);
    double nitidez = sLap[0] * sLap[0];

    // ---- De número a descripción (umbrales orientativos) ----
    vector<string> expo;
    if (media < 85) expo.push_back("subexpuesta");
    else if (media > 170) expo.push_back("sobreexpuesta");
    if (blancos > 0.05) expo.push_back("altas luces quemadas");
    if (negros > 0.05) expo.push_back("sombras bloqueadas");
    if (expo.empty()) expo.push_back("correcta");
    string exposicion;
    for (size_t i = 0; i < expo.size(); ++i) exposicion += (i ? ", " : "") + expo[i];

    string enfoque   = nitidez < 100 ? "borrosa" : (nitidez < 500 ? "aceptable" : "nitida");
    string contraste = sigma < 40 ? "bajo" : (sigma > 70 ? "alto" : "medio");
    string sesgo     = asimetria > 0.5 ? "predominan tonos oscuros"
                     : (asimetria < -0.5 ? "predominan tonos claros" : "distribucion simetrica");
    string colorMedio = nombreColor(mean(color));

    vector<string> rep = {
        "# CUANTITATIVA (histograma)",
        "Intensidad media     = " + num(media, 1) + "   (0 negro .. 255 blanco)",
        "Varianza (contraste) = " + num(varianza, 1) + "   (desv. est. = " + num(sigma, 1) + ")",
        "Asimetria (skewness) = " + num(asimetria, 2),
        "Entropia             = " + num(entropia, 2) + " bits  (max. 8.00)",
        "Pixeles 0-5 / 250-255 = " + num(100 * negros, 1) + " % / " + num(100 * blancos, 1) + " %",
        "Nitidez (var. Laplaciano) = " + num(nitidez, 1),
        "# CUALITATIVA (calidad perceptual)",
        "Exposicion : " + exposicion,
        "Enfoque    : " + enfoque,
        "Contraste  : " + contraste,
        "Tonos      : " + sesgo,
        "Color medio: " + colorMedio,
    };

    // Histograma dibujado (escala logarítmica para que un pico no aplaste al resto).
    const int W = 512, H = 150;
    Mat dib(H, W, CV_8UC3, Scalar(22, 22, 22));
    double mx = 0;
    for (int i = 0; i < 256; ++i) mx = max(mx, log1p((double)hist.at<float>(i)));
    for (int i = 0; i < 256; ++i) {
        int h = mx > 0 ? cvRound(log1p((double)hist.at<float>(i)) / mx * (H - 22)) : 0;
        rectangle(dib, Point(i * 2, H - 14 - h), Point(i * 2 + 1, H - 14), Scalar(210, 210, 210), FILLED);
    }
    int xm = cvRound(media * 2);
    line(dib, Point(xm, 0), Point(xm, H - 14), Scalar(60, 60, 255), 1);   // media (rojo)
    putText(dib, "histograma (log)  | media", Point(6, 14), FONT_HERSHEY_SIMPLEX, 0.4,
            Scalar(190, 190, 190), 1, LINE_AA);
    putText(dib, "0", Point(2, H - 2), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(190, 190, 190), 1, LINE_AA);
    putText(dib, "255", Point(W - 26, H - 2), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(190, 190, 190), 1, LINE_AA);

    Resultado r;
    r.reporte = rep;
    r.lienzo = componer(rotular(color, "Imagen analizada"), "MODO 0: INTENSIDAD", rep, dib);
    return r;
}


// ---------------------------------------------------------------------------
// Modo 1: OBJETOS (forma, color y relaciones RCC-8)
// ---------------------------------------------------------------------------

// Relación topológica RCC-8 entre dos regiones (máscaras 0/255 del mismo tamaño).
// Se decide con tres cantidades: el solape (A y B), si A cabe dentro de B (o al revés),
// y si las regiones se "tocan" (se prueba dilatando 2 píxeles: tolerancia a la
// discretización y al ruido del borde).
static string rcc8(const Mat& A, const Mat& B) {
    int aA = countNonZero(A), aB = countNonZero(B);
    int inter = countNonZero(A & B);
    Mat k = getStructuringElement(MORPH_RECT, Size(5, 5));   // 2 px por lado

    if (inter == 0) {
        Mat dA;
        dilate(A, dA, k);
        return countNonZero(dA & B) > 0 ? "EC" : "DC";   // tocan por el borde / separados
    }
    if (inter == aA && inter == aB) return "EQ";
    if (inter == aA) {                                   // A está contenido en B
        Mat eB;
        erode(B, eB, k);                                 // B sin su capa de borde (2 px)
        return countNonZero(A & eB) == aA ? "NTPP" : "TPP";
    }
    if (inter == aB) {                                   // B está contenido en A
        Mat eA;
        erode(A, eA, k);
        return countNonZero(B & eA) == aB ? "NTPPi" : "TPPi";
    }
    return "PO";                                         // solapan parcialmente
}

// Etiqueta cualitativa de la forma a partir de medidas numéricas.
static string etiquetaForma(double C, double aspecto, double solidez) {
    if (C >= 0.80 && aspecto < 1.3) return "circular";
    if (aspecto >= 2.0)             return "alargado";
    if (solidez < 0.85)             return "irregular/dentado";
    if (solidez >= 0.95)            return "convexo/regular";
    return "poco convexo";
}

static Resultado modoObjetos(const Mat& color, const Mat& gris, int areaMinMil, int polaridad) {
    // 1) Segmentación: suavizado + Otsu. Se busca que el objeto quede en BLANCO.
    Mat suave, bin;
    GaussianBlur(gris, suave, Size(5, 5), 0);
    threshold(suave, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    bool invertir;
    if (polaridad == 1)      invertir = false;   // objetos claros
    else if (polaridad == 2) invertir = true;    // objetos oscuros
    else {
        // Automática: el fondo es lo que domina en el borde de la imagen.
        int blancosBorde = countNonZero(bin.row(0)) + countNonZero(bin.row(bin.rows - 1)) +
                           countNonZero(bin.col(0)) + countNonZero(bin.col(bin.cols - 1));
        int totalBorde = 2 * bin.cols + 2 * bin.rows;
        invertir = blancosBorde > totalBorde / 2;
    }
    if (invertir) bin = 255 - bin;
    // Apertura: elimina motas pequeñas de ruido sin deformar los objetos grandes.
    morphologyEx(bin, bin, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

    // 2) Contornos. RETR_TREE da la jerarquía: profundidad par = borde exterior de un
    //    objeto, impar = borde de un agujero (que no es un objeto).
    vector<vector<Point>> cont;
    vector<Vec4i> jer;
    findContours(bin.clone(), cont, jer, RETR_TREE, CHAIN_APPROX_NONE);

    struct Obj {
        int idx; vector<Point> c; Mat mascara;
        double area, perimetro, circ, aspecto, solidez;
        string forma, colorNombre;
        Point centro;
    };
    vector<Obj> objs;
    double areaMin = areaMinMil / 1000.0 * gris.total();
    for (size_t i = 0; i < cont.size(); ++i) {
        int prof = 0;
        for (int p = jer[i][3]; p != -1; p = jer[p][3]) ++prof;
        if (prof % 2 != 0) continue;
        Obj o;
        o.idx = (int)i;
        o.c = cont[i];
        o.mascara = Mat::zeros(gris.size(), CV_8UC1);
        drawContours(o.mascara, cont, (int)i, Scalar(255), FILLED);
        o.area = countNonZero(o.mascara);          // área = conteo de píxeles
        if (o.area < max(areaMin, 20.0)) continue;
        objs.push_back(o);
    }
    sort(objs.begin(), objs.end(), [](const Obj& a, const Obj& b) { return a.area > b.area; });
    if (objs.size() > 12) objs.resize(12);          // se limita para que el reporte quepa

    // 3) Medidas y etiquetas por objeto.
    Mat vista = color.clone();
    for (size_t k = 0; k < objs.size(); ++k) {
        Obj& o = objs[k];
        o.perimetro = arcLength(o.c, true);
        // Circularidad C = 4*pi*A/P^2 (se recorta a 1: la discretización puede pasarse).
        o.circ = min(1.0, 4 * CV_PI * o.area / (o.perimetro * o.perimetro));
        RotatedRect rr = minAreaRect(o.c);
        double lado1 = max(rr.size.width, rr.size.height), lado2 = min(rr.size.width, rr.size.height);
        o.aspecto = lado2 > 0 ? lado1 / lado2 : 1;
        vector<Point> casco;
        convexHull(o.c, casco);
        double areaCasco = contourArea(casco);
        o.solidez = areaCasco > 0 ? min(1.0, o.area / areaCasco) : 1;
        o.forma = etiquetaForma(o.circ, o.aspecto, o.solidez);
        // El color se mide solo sobre los píxeles del objeto (no sobre sus agujeros).
        o.colorNombre = nombreColor(mean(color, o.mascara & bin));
        Moments m = moments(o.c);
        o.centro = m.m00 > 0 ? Point(cvRound(m.m10 / m.m00), cvRound(m.m01 / m.m00)) : o.c[0];

        Scalar col = o.forma == "circular" ? Scalar(80, 220, 80)
                   : o.forma == "alargado" ? Scalar(0, 165, 255)
                   : o.forma == "irregular/dentado" ? Scalar(70, 70, 255)
                   : Scalar(255, 170, 60);
        drawContours(vista, vector<vector<Point>>{o.c}, 0, col, 2, LINE_AA);
        putText(vista, to_string(k + 1), o.centro - Point(6, -5), FONT_HERSHEY_SIMPLEX, 0.6,
                Scalar(0, 0, 0), 4, LINE_AA);
        putText(vista, to_string(k + 1), o.centro - Point(6, -5), FONT_HERSHEY_SIMPLEX, 0.6,
                Scalar(255, 255, 255), 1, LINE_AA);
    }

    // 4) Relaciones RCC-8 entre cada pareja de objetos.
    vector<string> relaciones, relacionesNoDC;
    int pares = 0, desconectados = 0;
    for (size_t i = 0; i < objs.size(); ++i)
        for (size_t j = i + 1; j < objs.size(); ++j) {
            string r = rcc8(objs[i].mascara, objs[j].mascara);
            ++pares;
            if (r == "DC") { ++desconectados; continue; }
            relaciones.push_back("Obj" + to_string(i + 1) + " " + r + " Obj" + to_string(j + 1));
        }

    // 5) Reporte: panel (resumido) y consola (completo).
    vector<string> rep, det;
    rep.push_back("# CUANTITATIVA (por objeto)");
    rep.push_back("Objetos detectados: " + to_string(objs.size()) + "   (modo: " +
                  string(invertir ? "oscuros sobre claro" : "claros sobre oscuro") + ")");
    rep.push_back("N   Area   Perim.  Circ.  Aspec.  Solidez");
    for (size_t k = 0; k < objs.size(); ++k) {
        const Obj& o = objs[k];
        ostringstream os;
        os << left << setw(4) << (k + 1) << setw(7) << (long long)o.area << setw(8) << (int)o.perimetro
           << setw(7) << num(o.circ) << setw(8) << num(o.aspecto) << num(o.solidez);
        rep.push_back(os.str());
    }
    rep.push_back("# CUALITATIVA (forma y color)");
    for (size_t k = 0; k < objs.size(); ++k)
        rep.push_back("Obj" + to_string(k + 1) + ": " + objs[k].forma + ", " + objs[k].colorNombre);
    rep.push_back("# RELACIONES TOPOLOGICAS (RCC-8)");
    if (relaciones.empty()) rep.push_back("Todos los pares estan desconectados (DC).");
    for (size_t i = 0; i < relaciones.size() && i < 8; ++i) rep.push_back(relaciones[i]);
    if (relaciones.size() > 8) rep.push_back("... y " + to_string(relaciones.size() - 8) + " mas (ver consola)");
    rep.push_back(to_string(desconectados) + " de " + to_string(pares) + " pares son DC");

    det = rep;
    det.push_back("");
    det.push_back("Todas las relaciones no desconectadas:");
    for (const string& r : relaciones) det.push_back("  " + r);
    det.push_back("Lectura: 'A TPP B' = A es parte propia tangencial de B (dentro, tocando su borde);");
    det.push_back("NTPP = dentro sin tocar el borde; EC = se tocan por el borde; PO = solape parcial.");
    det.push_back("Estas relaciones no cambian al rotar o escalar la imagen (invariancia topologica).");

    Mat mascaraTotal = Mat::zeros(gris.size(), CV_8UC1);
    for (const Obj& o : objs) mascaraTotal |= o.mascara;
    Mat extra;
    resize(mascaraTotal, extra, Size(), 0.5, 0.5, INTER_NEAREST);
    cvtColor(extra, extra, COLOR_GRAY2BGR);
    putText(extra, "mascara de objetos", Point(6, 16), FONT_HERSHEY_SIMPLEX, 0.45,
            Scalar(0, 200, 255), 1, LINE_AA);

    Resultado res;
    res.reporte = rep;
    res.detalle = det;
    res.lienzo = componer(rotular(vista, "Objetos (verde=circular, naranja=alargado, rojo=irregular)"),
                          "MODO 1: OBJETOS", rep, extra);
    return res;
}


// ---------------------------------------------------------------------------
// Modo 2: TEXTURA (GLCM)
// ---------------------------------------------------------------------------

// Matriz de co-ocurrencia de niveles de gris: G[i][j] = proporción de parejas de
// píxeles, separadas por el desplazamiento (dx, dy), con niveles i y j. Se hace
// simétrica (cuenta (i,j) y (j,i)) y se normaliza para que sume 1.
static Mat calcularGLCM(const Mat& gris, int niveles, int dx, int dy) {
    Mat G = Mat::zeros(niveles, niveles, CV_64F);
    for (int y = 0; y < gris.rows; ++y)
        for (int x = 0; x < gris.cols; ++x) {
            int x2 = x + dx, y2 = y + dy;
            if (x2 < 0 || y2 < 0 || x2 >= gris.cols || y2 >= gris.rows) continue;
            int i = gris.at<uchar>(y, x) * niveles / 256;
            int j = gris.at<uchar>(y2, x2) * niveles / 256;
            G.at<double>(i, j) += 1;
            G.at<double>(j, i) += 1;
        }
    G /= max(1.0, sum(G)[0]);
    return G;
}

static Resultado modoTextura(const Mat& gris, int distancia, int idxAngulo) {
    const int L = 16;
    distancia = max(1, distancia);
    static const int ANG[4] = {0, 45, 90, 135};
    // Desplazamiento (dx, dy) para cada ángulo (el eje Y de la imagen crece hacia abajo).
    int dx = 0, dy = 0;
    switch (idxAngulo) {
        case 0: dx = distancia;  dy = 0;          break;
        case 1: dx = distancia;  dy = -distancia; break;
        case 2: dx = 0;          dy = -distancia; break;
        default: dx = -distancia; dy = -distancia; break;
    }
    Mat G = calcularGLCM(gris, L, dx, dy);

    // Propiedades de la GLCM.
    double contraste = 0, homog = 0, asm_ = 0, entropia = 0, mi = 0, mj = 0;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j) {
            double g = G.at<double>(i, j);
            contraste += g * (i - j) * (i - j);
            homog += g / (1.0 + (i - j) * (i - j));
            asm_ += g * g;
            if (g > 0) entropia -= g * log2(g);
            mi += i * g;
            mj += j * g;
        }
    double si = 0, sj = 0, cov = 0;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j) {
            double g = G.at<double>(i, j);
            si += g * (i - mi) * (i - mi);
            sj += g * (j - mj) * (j - mj);
            cov += g * (i - mi) * (j - mj);
        }
    double correl = (si > 1e-12 && sj > 1e-12) ? cov / sqrt(si * sj) : 1.0;

    string tHomog = homog >= 0.65 ? "homogenea (superficie lisa)"
                  : (homog >= 0.40 ? "intermedia" : "rugosa / heterogenea");
    string tContr = contraste < 3 ? "bajo" : (contraste < 12 ? "medio" : "alto");

    vector<string> rep = {
        "# GLCM (" + to_string(L) + " niveles)",
        "Desplazamiento: distancia " + to_string(distancia) + " px, angulo " + to_string(ANG[idxAngulo]) + " grados",
        "# CUANTITATIVA (propiedades de la GLCM)",
        "Contraste                = " + num(contraste, 3),
        "Homogeneidad             = " + num(homog, 3),
        "Uniformidad (ASM)        = " + num(asm_, 4),
        "Energia (raiz de ASM)    = " + num(sqrt(asm_), 3),
        "Entropia                 = " + num(entropia, 3) + " bits",
        "Correlacion              = " + num(correl, 3),
        "# CUALITATIVA",
        "Textura   : " + tHomog,
        "Contraste : " + tContr,
        "(Umbrales orientativos; dependen de la imagen y de los niveles.)",
        "Una diagonal intensa = vecinos parecidos = textura suave;",
        "un mapa disperso = vecinos muy distintos = textura rugosa.",
    };

    // GLCM como mapa de calor (raíz para que se vean las celdas poco pobladas).
    Mat vis, vis8;
    sqrt(G, vis);
    normalize(vis, vis, 0, 255, NORM_MINMAX);
    vis.convertTo(vis8, CV_8U);
    resize(vis8, vis8, Size(256, 256), 0, 0, INTER_NEAREST);
    Mat mapa;
    applyColorMap(vis8, mapa, COLORMAP_INFERNO);
    rectangle(mapa, Point(0, 0), Point(mapa.cols, 18), Scalar(0, 0, 0), FILLED);
    putText(mapa, "GLCM (i = fila, j = columna)", Point(6, 13), FONT_HERSHEY_SIMPLEX, 0.4,
            C_TEXTO, 1, LINE_AA);

    Resultado r;
    r.reporte = rep;
    r.lienzo = componer(rotular(gris, "Imagen en grises (textura global)"), "MODO 2: TEXTURA", rep, mapa);
    return r;
}


// ---------------------------------------------------------------------------
// Modo 3: PUNTOS CLAVE (SIFT, ORB, BRIEF a mano)
// ---------------------------------------------------------------------------

// Pares de píxeles de BRIEF: 256 parejas (p1, p2) generadas UNA vez con semilla fija,
// con distribución gaussiana alrededor del punto clave y dentro de una ventana 31x31.
// Es importante usar siempre los mismos pares en las dos imágenes.
static const int BRIEF_R = 15;       // semilla de la ventana: radio (ventana 31x31)
static const int BRIEF_BITS = 256;

static const vector<array<Point, 2>>& paresBrief() {
    static const vector<array<Point, 2>> pares = [] {
        RNG rng(12345);
        auto punto = [&rng] {
            int x = cvRound(rng.gaussian((2.0 * BRIEF_R + 1) / 5.0));
            int y = cvRound(rng.gaussian((2.0 * BRIEF_R + 1) / 5.0));
            return Point(max(-BRIEF_R, min(BRIEF_R, x)), max(-BRIEF_R, min(BRIEF_R, y)));
        };
        vector<array<Point, 2>> v;
        for (int i = 0; i < BRIEF_BITS; ++i) v.push_back({punto(), punto()});
        return v;
    }();
    return pares;
}

// Descriptor BRIEF: para cada pareja de píxeles se hace la prueba binaria
//     bit = 1 si I(p1) < I(p2), si no 0
// sobre la imagen suavizada (el suavizado reduce la sensibilidad al ruido). El resultado
// es una cadena de 256 bits (32 bytes) por punto clave. Se descartan los puntos cuya
// ventana se sale de la imagen.
static void calcularBrief(const Mat& gris, vector<KeyPoint>& kps, Mat& desc) {
    Mat suave;
    GaussianBlur(gris, suave, Size(9, 9), 2, 2);
    const auto& pares = paresBrief();
    vector<KeyPoint> validos;
    desc = Mat(0, BRIEF_BITS / 8, CV_8U);
    for (const KeyPoint& kp : kps) {
        int x = cvRound(kp.pt.x), y = cvRound(kp.pt.y);
        if (x < BRIEF_R || y < BRIEF_R || x >= gris.cols - BRIEF_R || y >= gris.rows - BRIEF_R) continue;
        Mat fila = Mat::zeros(1, BRIEF_BITS / 8, CV_8U);
        for (int k = 0; k < BRIEF_BITS; ++k) {
            uchar a = suave.at<uchar>(y + pares[k][0].y, x + pares[k][0].x);
            uchar b = suave.at<uchar>(y + pares[k][1].y, x + pares[k][1].x);
            if (a < b) fila.at<uchar>(0, k / 8) |= (uchar)(1 << (k % 8));
        }
        desc.push_back(fila);
        validos.push_back(kp);
    }
    kps = validos;
}

// Detecta puntos y calcula descriptores según el detector elegido.
// Devuelve la norma con la que se deben comparar (L2 para SIFT, Hamming para binarios).
static int extraer(int detector, const Mat& gris, int maxPuntos, vector<KeyPoint>& kps, Mat& desc) {
    if (detector == 0) {
        SIFT::create(maxPuntos)->detectAndCompute(gris, noArray(), kps, desc);
        return NORM_L2;
    }
    if (detector == 1) {
        ORB::create(maxPuntos)->detectAndCompute(gris, noArray(), kps, desc);
        return NORM_HAMMING;
    }
    // BRIEF a mano: FAST (esquinas) + descriptor binario. Se queda con los más fuertes.
    FastFeatureDetector::create(20, true)->detect(gris, kps);
    sort(kps.begin(), kps.end(), [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; });
    if ((int)kps.size() > maxPuntos) kps.resize(maxPuntos);
    calcularBrief(gris, kps, desc);
    return NORM_HAMMING;
}

static Resultado modoPuntos(const Mat& gris, int detector, int rotacion, int escalaPct, int maxPuntos) {
    static const char* NOMBRES[3] = {"SIFT", "ORB", "BRIEF (a mano)"};
    static const char* DESCRIPTORES[3] = {"128 numeros (float), distancia L2",
                                          "256 bits (rBRIEF orientado), distancia Hamming",
                                          "256 bits (sin orientacion), distancia Hamming"};
    maxPuntos = max(50, maxPuntos);
    escalaPct = max(50, escalaPct);

    // Segunda imagen: la misma, rotada y escalada alrededor del centro. M es conocida,
    // así que se puede comprobar si cada coincidencia es geométricamente correcta.
    Mat M = getRotationMatrix2D(Point2f(gris.cols / 2.f, gris.rows / 2.f), rotacion, escalaPct / 100.0);
    Mat gris2;
    warpAffine(gris, gris2, M, gris.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

    vector<KeyPoint> k1, k2;
    Mat d1, d2;
    int norma = extraer(detector, gris, maxPuntos, k1, d1);
    extraer(detector, gris2, maxPuntos, k2, d2);

    // Emparejamiento por fuerza bruta + prueba del cociente de Lowe: se acepta la mejor
    // coincidencia solo si es claramente mejor que la segunda (distancia < 0.75 x).
    vector<DMatch> buenas, correctas, erroneas;
    if (!d1.empty() && d2.rows >= 2) {
        vector<vector<DMatch>> knn;
        BFMatcher(norma).knnMatch(d1, d2, knn, 2);
        for (const auto& m : knn)
            if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) buenas.push_back(m[0]);
    }
    for (const DMatch& m : buenas) {
        Point2f p = k1[m.queryIdx].pt;
        Point2f esperado(M.at<double>(0, 0) * p.x + M.at<double>(0, 1) * p.y + M.at<double>(0, 2),
                         M.at<double>(1, 0) * p.x + M.at<double>(1, 1) * p.y + M.at<double>(1, 2));
        Point2f real = k2[m.trainIdx].pt;
        double err = hypot(esperado.x - real.x, esperado.y - real.y);
        (err < 4.0 ? correctas : erroneas).push_back(m);
    }
    double precision = buenas.empty() ? 0.0 : 100.0 * correctas.size() / buenas.size();

    // Dibujo: correctas en verde y erróneas en rojo sobre el mismo lienzo.
    Mat salida;
    drawMatches(gris, k1, gris2, k2, correctas, salida, Scalar(80, 220, 80), Scalar(0, 200, 255),
                vector<char>(), DrawMatchesFlags::DEFAULT);
    drawMatches(gris, k1, gris2, k2, erroneas, salida, Scalar(60, 60, 255), Scalar(0, 200, 255),
                vector<char>(), DrawMatchesFlags::DRAW_OVER_OUTIMG | DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    salida = rotular(salida, string("Original (izq.) y rotada/escalada (der.) - ") + NOMBRES[detector]);

    vector<string> rep = {
        "# " + string("Detector: ") + NOMBRES[detector],
        string("Descriptor: ") + DESCRIPTORES[detector],
        "Transformacion: rotacion " + to_string(rotacion) + " grados, escala " + to_string(escalaPct) + " %",
        "Puntos clave: " + to_string(k1.size()) + " (original) / " + to_string(k2.size()) + " (transformada)",
        "Coincidencias (Lowe 0.75): " + to_string(buenas.size()),
        "Correctas (error < 4 px): " + to_string(correctas.size()) + "  ->  " + num(precision, 1) + " %",
        "Verde = correctas, rojo = erroneas.",
    };
    if (detector == 2 && rotacion % 360 != 0)
        rep.push_back("BRIEF no es invariante a la rotacion: observa el desplome.");
    if (detector == 0)
        rep.push_back("SIFT: invariante a escala y rotacion (espacio de escala + orientacion).");
    if (detector == 1)
        rep.push_back("ORB: invariante a rotacion (orienta el parche) y a escala (piramide).");

    // Pie con el reporte debajo de las dos imágenes.
    const int LH = 19;
    Mat pie(14 + (int)rep.size() * LH, salida.cols, CV_8UC3, C_FONDO);
    for (size_t i = 0; i < rep.size(); ++i) {
        bool enc = rep[i].rfind("# ", 0) == 0;
        putText(pie, enc ? rep[i].substr(2) : rep[i], Point(10, 18 + (int)i * LH), FONT_HERSHEY_SIMPLEX,
                enc ? 0.5 : 0.45, enc ? C_ACENTO : C_TEXTO, 1, LINE_AA);
    }
    Resultado r;
    r.reporte = rep;
    vconcat(salida, pie, r.lienzo);
    return r;
}


// ---------------------------------------------------------------------------
// Programa principal
// ---------------------------------------------------------------------------
static void imprimirReporte(int modo, const Resultado& r) {
    static const char* TITULOS[4] = {"INTENSIDAD", "OBJETOS", "TEXTURA (GLCM)", "PUNTOS CLAVE"};
    cout << "\n=== Modo " << modo << ": " << TITULOS[modo] << " ===\n";
    const vector<string>& lineas = r.detalle.empty() ? r.reporte : r.detalle;
    for (const string& l : lineas)
        cout << (l.rfind("# ", 0) == 0 ? "-- " + l.substr(2) + " --" : l) << "\n";
    cout << flush;
}

int main(int argc, char** argv) {
    // ---- Pedir la ruta (argumento o consola) ----
    string ruta;
    if (argc >= 2) {
        ruta = limpiarRuta(argv[1]);
    } else {
        cout << "=== Caracterizacion ===\n";
        cout << "Ingrese la ruta de la imagen: ";
        getline(cin, ruta);
        ruta = limpiarRuta(ruta);
    }
    int modoInicial = 0;
    if (argc >= 3) {
        try { modoInicial = max(0, min(3, stoi(argv[2]))); } catch (...) { modoInicial = 0; }
    }

    Mat original = imread(ruta);
    if (original.empty()) {
        cerr << "\nError: no se pudo cargar la imagen en la ruta:\n" << ruta << "\n";
        cerr << "Verifica que la ruta sea correcta y que el archivo exista.\n";
        return 1;
    }
    Mat color = ajustarTamano(original);
    Mat gris;
    cvtColor(color, gris, COLOR_BGR2GRAY);

    // ---- Ventana y barras ----
    namedWindow(VENTANA, WINDOW_AUTOSIZE);
    createTrackbar("Modo", VENTANA, nullptr, 3);
    createTrackbar("M1 AreaMin", VENTANA, nullptr, 100);
    createTrackbar("M1 Polaridad", VENTANA, nullptr, 2);
    createTrackbar("M2 Distancia", VENTANA, nullptr, 10);
    createTrackbar("M2 Angulo", VENTANA, nullptr, 3);
    createTrackbar("M3 Detector", VENTANA, nullptr, 2);
    createTrackbar("M3 Rotacion", VENTANA, nullptr, 360);
    createTrackbar("M3 Escala", VENTANA, nullptr, 150);
    createTrackbar("M3 Puntos", VENTANA, nullptr, 2000);
    setTrackbarPos("Modo", VENTANA, modoInicial);
    setTrackbarPos("M1 AreaMin", VENTANA, 5);
    setTrackbarPos("M1 Polaridad", VENTANA, 0);
    setTrackbarPos("M2 Distancia", VENTANA, 1);
    setTrackbarPos("M2 Angulo", VENTANA, 0);
    setTrackbarPos("M3 Detector", VENTANA, 0);
    setTrackbarPos("M3 Rotacion", VENTANA, 30);
    setTrackbarPos("M3 Escala", VENTANA, 100);
    setTrackbarPos("M3 Puntos", VENTANA, 500);

    Resultado actual;
    vector<int> clavePrevia;
    int modoPrevio = -1;
    while (true) {
        // ---- Leer parámetros ----
        int modo       = getTrackbarPos("Modo", VENTANA);
        int areaMin    = getTrackbarPos("M1 AreaMin", VENTANA);
        int polaridad  = getTrackbarPos("M1 Polaridad", VENTANA);
        int distancia  = getTrackbarPos("M2 Distancia", VENTANA);
        int angulo     = getTrackbarPos("M2 Angulo", VENTANA);
        int detector   = getTrackbarPos("M3 Detector", VENTANA);
        int rotacion   = getTrackbarPos("M3 Rotacion", VENTANA);
        int escala     = getTrackbarPos("M3 Escala", VENTANA);
        int puntos     = getTrackbarPos("M3 Puntos", VENTANA);

        // Solo se recalcula si cambió algo que afecta al modo actual (SIFT, GLCM y las
        // relaciones RCC-8 son costosos y la imagen no cambia entre ciclos).
        vector<int> clave = {modo};
        if (modo == 1) clave.insert(clave.end(), {areaMin, polaridad});
        if (modo == 2) clave.insert(clave.end(), {distancia, angulo});
        if (modo == 3) clave.insert(clave.end(), {detector, rotacion, escala, puntos});

        if (clave != clavePrevia) {
            switch (modo) {
                case 0:  actual = modoIntensidad(color, gris); break;
                case 1:  actual = modoObjetos(color, gris, areaMin, polaridad); break;
                case 2:  actual = modoTextura(gris, distancia, angulo); break;
                default: actual = modoPuntos(gris, detector, rotacion, escala, puntos); break;
            }
            imshow(VENTANA, actual.lienzo);
            if (modo != modoPrevio) imprimirReporte(modo, actual);
            modoPrevio = modo;
            clavePrevia = clave;
        }

        int tecla = waitKey(30) & 0xFF;
        if (tecla == 'q' || tecla == 27) break;
        if (tecla == 'p') imprimirReporte(modo, actual);
        if (tecla == 's') {
            imwrite("caracterizacion.png", actual.lienzo);
            cout << "Guardado: caracterizacion.png" << endl;
        }
    }

    return 0;
}
