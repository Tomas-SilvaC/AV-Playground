// Erosión y Dilatación con OpenCV (C++)
// -------------------------------------
// Al ejecutar, el programa pide la ruta de la imagen por consola.
// Si se deja vacía (solo Enter), usa la webcam.
//
// Barras de la ventana:
//   Tamano : tamaño del elemento estructurante (kernel = 2*valor + 1)
//   Forma  : 0 = Cuadrado, 1 = Disco (elipse), 2 = Cruz
//   Binaria: 0 = escala de grises, 1 = binaria (con umbral)
//   Umbral : umbral de binarización (solo si Binaria = 1)
//
// Teclas: q o ESC = salir, s = guardar resultado.png

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

static const string VENTANA = "Morfologia: Original | Erosion | Dilatacion";
static const int FORMAS[3] = {MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS};
static const char* NOMBRES_FORMAS[3] = {"Cuadrado", "Disco", "Cruz"};

// Convierte a BGR (si hace falta) y escribe un título arriba.
static Mat etiquetar(const Mat& img, const string& texto) {
    Mat out;
    if (img.channels() == 1) cvtColor(img, out, COLOR_GRAY2BGR);
    else out = img.clone();
    rectangle(out, Point(0, 0), Point(out.cols, 28), Scalar(0, 0, 0), FILLED);
    putText(out, texto, Point(8, 20), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(255, 255, 255), 1, LINE_AA);
    return out;
}

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

        frame = redimensionar(frame);
        Mat gris;
        cvtColor(frame, gris, COLOR_BGR2GRAY);

        // ---- Leer parámetros ----
        int k       = getTrackbarPos("Tamano", VENTANA);
        int forma   = getTrackbarPos("Forma", VENTANA);
        int binaria = getTrackbarPos("Binaria", VENTANA);
        int umbral  = getTrackbarPos("Umbral", VENTANA);

        int lado = 2 * k + 1;  // siempre impar: 1, 3, 5, ...
        Mat kernel = getStructuringElement(FORMAS[forma], Size(lado, lado));

        // ---- Imagen de entrada: gris o binaria ----
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
