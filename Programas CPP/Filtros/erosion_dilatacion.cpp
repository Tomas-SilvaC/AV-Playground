/**
 * erosion_dilatacion.cpp
 * Erosión y dilatación morfológicas interactivas (imagen o webcam).
 *
 * ## Idea general
 * La morfología matemática "sondea" una imagen con una forma pequeña llamada
 * elemento estructurante (EE, o kernel). El EE se desliza por cada píxel y, según
 * cómo encaje con su vecindario, se decide el valor del píxel de salida:
 *
 *   - Erosión:    el píxel sobrevive solo si TODO el vecindario bajo el EE es objeto.
 *                 Binaria  -> AND local.   Escala de grises -> mínimo local.
 *                 Efecto: reduce objetos, elimina ruido tipo "sal" y puentes finos,
 *                 separa objetos que se tocan.
 *   - Dilatación: el píxel es objeto si AL MENOS UN vecino bajo el EE lo es.
 *                 Binaria  -> OR local.    Escala de grises -> máximo local.
 *                 Efecto: engrosa objetos, rellena huecos pequeños, puede fusionar
 *                 objetos cercanos.
 *
 * La forma del EE condiciona qué geometría se preserva o se elimina:
 *   - Cuadrado: bajo costo, bloques alineados con los ejes.
 *   - Disco (elipse): isotrópico, adecuado para manchas y redondeo.
 *   - Cruz: preserva estructura horizontal/vertical.
 *
 * Combinadas, dan las operaciones derivadas: apertura (erosión -> dilatación) y
 * cierre (dilatación -> erosión). Aquí se muestran solo las dos básicas.
 *
 * ## Uso
 * Al ejecutar, el programa pide la ruta de la imagen por consola.
 * Si se deja vacía (solo Enter), usa la webcam.
 *
 * Barras de la ventana:
 *   Tamano : tamaño del elemento estructurante (lado del kernel = 2*valor + 1)
 *   Forma  : 0 = Cuadrado, 1 = Disco (elipse), 2 = Cruz
 *   Binaria: 0 = escala de grises, 1 = binaria (con umbral)
 *   Umbral : umbral de binarización (solo si Binaria = 1)
 *
 * Teclas: q o ESC = salir, s = guardar resultado.png
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

static const string VENTANA = "Morfologia: Original | Erosion | Dilatacion";
// Formas de elemento estructurante disponibles (se indexan con la barra "Forma").
static const int FORMAS[3] = {MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS};
static const char* NOMBRES_FORMAS[3] = {"Cuadrado", "Disco", "Cruz"};


// Convierte la imagen a BGR (si es de un canal) y dibuja una franja negra con un
// título arriba. Sirve para poder concatenar paneles de distinto tipo y rotularlos.
 
static Mat etiquetar(const Mat& img, const string& texto) {
    Mat out;
    if (img.channels() == 1) cvtColor(img, out, COLOR_GRAY2BGR);
    else out = img.clone();
    rectangle(out, Point(0, 0), Point(out.cols, 28), Scalar(0, 0, 0), FILLED);
    putText(out, texto, Point(8, 20), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(255, 255, 255), 1, LINE_AA);
    return out;
}

// Redimensiona a un ancho fijo conservando la proporción. Mantiene los paneles
// del mismo tamaño y la ventana manejable sin importar la resolución de entrada.

static Mat redimensionar(const Mat& img, int ancho = 420) {
    int alto = img.rows * ancho / img.cols;
    Mat out;
    resize(img, out, Size(ancho, alto));
    return out;
}

// Quita espacios y comillas del inicio/final (Windows pone comillas
// al usar "Copiar como ruta de acceso").
static string limpiarRuta(string s) {
    const string basura = " \t\r\n\"'";
    size_t ini = s.find_first_not_of(basura);
    if (ini == string::npos) return "";
    size_t fin = s.find_last_not_of(basura);
    return s.substr(ini, fin - ini + 1);
}

int main() {
    // ---- Pedir la ruta al usuario ----
    cout << "Ingrese la ruta de la imagen (Enter para usar la webcam): ";
    string ruta;
    getline(cin, ruta);
    ruta = limpiarRuta(ruta);
    // Fuente de datos: una imagen fija (se reprocesa en cada ciclo para reflejar
    // los cambios de las barras) o la webcam (un fotograma nuevo por ciclo).
    Mat imagenFija;
    VideoCapture cap;
    bool usaImagen = false;

    if (!ruta.empty()) {
        imagenFija = imread(ruta);
        if (imagenFija.empty()) {
            cerr << "No se pudo abrir la imagen: " << ruta << endl;
            system("pause");
            return 1;
        }
        usaImagen = true;
    } else {
        cap.open(0);
        if (!cap.isOpened()) {
            cerr << "No se pudo abrir la webcam." << endl;
            system("pause");
            return 1;
        }
    }

    // ---- Ventana y barras ----
    namedWindow(VENTANA, WINDOW_AUTOSIZE);
    createTrackbar("Tamano", VENTANA, nullptr, 15);
    createTrackbar("Forma", VENTANA, nullptr, 2);
    createTrackbar("Binaria", VENTANA, nullptr, 1);
    createTrackbar("Umbral", VENTANA, nullptr, 255);
    setTrackbarPos("Tamano", VENTANA, 2);
    setTrackbarPos("Forma", VENTANA, 0);
    setTrackbarPos("Binaria", VENTANA, 0);
    setTrackbarPos("Umbral", VENTANA, 127);

    while (true) {
        // ---- Obtener frame ----
        Mat frame;
        if (usaImagen) {
            frame = imagenFija.clone();
        } else if (!cap.read(frame) || frame.empty()) {
            break;
        }

        // Las operaciones morfológicas se aplican sobre intensidades (un canal).
        frame = redimensionar(frame);
        Mat gris;
        cvtColor(frame, gris, COLOR_BGR2GRAY);

        // ---- Leer parámetros ----
        int k       = getTrackbarPos("Tamano", VENTANA);
        int forma   = getTrackbarPos("Forma", VENTANA);
        int binaria = getTrackbarPos("Binaria", VENTANA);
        int umbral  = getTrackbarPos("Umbral", VENTANA);

        // El lado del kernel debe ser impar para que tenga un píxel central (origen).
        int lado = 2 * k + 1; 
        Mat kernel = getStructuringElement(FORMAS[forma], Size(lado, lado));

        // ---- Imagen de entrada: gris o binaria ----
        // En modo binario se umbraliza (global y fijo): píxel > umbral -> 255, si no -> 0.
        // Así se ve la morfología "clásica" sobre conjuntos; en grises actúan min/max.
        Mat base;
        string modo;
        if (binaria) {
            threshold(gris, base, umbral, 255, THRESH_BINARY);
            modo = "Binaria";
        } else {
            base = gris;
            modo = "Grises";
        }

        // ---- Operaciones morfológicas ----
        Mat erosion, dilatacion;
        erode(base, erosion, kernel);     // mínimo local / AND
        dilate(base, dilatacion, kernel); // máximo local / OR

        // ---- Mostrar ----
        // Se arman tres paneles rotulados y se concatenan horizontalmente.
        string info = modo + " | " + NOMBRES_FORMAS[forma] + " " +
                      to_string(lado) + "x" + to_string(lado);
        vector<Mat> paneles = {
            etiquetar(base, "Original (" + modo + ")"),
            etiquetar(erosion, "Erosion | " + info),
            etiquetar(dilatacion, "Dilatacion | " + info)
        };
        Mat panel;
        hconcat(paneles, panel);
        imshow(VENTANA, panel);

        int tecla = waitKey(30) & 0xFF;
        if (tecla == 'q' || tecla == 27) break;
        if (tecla == 's') {
            imwrite("resultado.png", panel);
            cout << "Guardado: resultado.png" << endl;
        }
    }

    return 0;
}
