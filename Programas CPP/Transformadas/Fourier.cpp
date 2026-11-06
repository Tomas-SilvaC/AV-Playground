// Transformada de Fourier 2D con OpenCV (C++)
// Sigue los 8 pasos de la presentación "Transformada de Fourier aplicada".
//
// Compilar:
//   g++ fourier.cpp -o fourier `pkg-config --cflags --libs opencv4`
// Ejecutar:
//   ./fourier                -> el programa pide la ruta (copiar y pegar)
//   ./fourier imagen.jpg     -> también acepta la ruta como argumento

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// Paso 7: reorganizar cuadrantes para llevar la componente DC al centro
static void reorganizarCuadrantes(Mat& mag)
{
    // Recortar a dimensiones pares para poder dividir en 4 cuadrantes iguales
    mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    Mat q0(mag, Rect(0, 0, cx, cy));  // superior izquierdo
    Mat q1(mag, Rect(cx, 0, cx, cy));  // superior derecho
    Mat q2(mag, Rect(0, cy, cx, cy));  // inferior izquierdo
    Mat q3(mag, Rect(cx, cy, cx, cy));  // inferior derecho

    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);  // intercambio diagonal 0 <-> 3
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);  // intercambio diagonal 1 <-> 2
}

// Limpia la ruta pegada: quita espacios, saltos de línea y comillas
// (Windows añade comillas con "Copiar como ruta de acceso")
static string limpiarRuta(string r)
{
    const string espacios = " \t\r\n";
    size_t ini = r.find_first_not_of(espacios);
    if (ini == string::npos) return "";
    size_t fin = r.find_last_not_of(espacios);
    r = r.substr(ini, fin - ini + 1);

    if (r.size() >= 2 && (r.front() == '"' || r.front() == '\'') && r.back() == r.front())
        r = r.substr(1, r.size() - 2);
    return r;
}

int main(int argc, char** argv)
{
    // ---------------------------------------------------------------
    // Paso 1: Cargar la imagen en escala de grises (un solo canal)
    //         La ruta se puede pegar en la consola o pasar como argumento
    // ---------------------------------------------------------------
    Mat I;
    string ruta = (argc >= 2) ? limpiarRuta(argv[1]) : "";

    while (true) {
        if (ruta.empty()) {
            cout << "Pega la direccion de la imagen y presiona Enter"
                << " (o escribe 'salir'): ";
            string linea;
            if (!getline(cin, linea)) return -1;   // fin de entrada
            ruta = limpiarRuta(linea);
            if (ruta == "salir") return 0;
            if (ruta.empty()) continue;
        }

        I = imread(ruta, IMREAD_GRAYSCALE);
        if (!I.empty()) break;

        cerr << "No se pudo abrir la imagen: " << ruta << endl;
        ruta.clear();   // volver a pedir la ruta
    }

    // ---------------------------------------------------------------
    // Paso 2: Añadir relleno (padding) a un tamaño óptimo para la DFT
    // ---------------------------------------------------------------
    Mat padded;
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize(I.cols);
    copyMakeBorder(I, padded,
        0, m - I.rows,
        0, n - I.cols,
        BORDER_CONSTANT, Scalar::all(0));

    // ---------------------------------------------------------------
    // Paso 3: Crear la imagen compleja
    //         parte real = imagen, parte imaginaria = 0
    // ---------------------------------------------------------------
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    // ---------------------------------------------------------------
    // Paso 4: Aplicar la DFT (dominio espacial -> dominio de frecuencia)
    // ---------------------------------------------------------------
    dft(complexI, complexI);

    // ---------------------------------------------------------------
    // Paso 5: Espectro de magnitud = sqrt(Re^2 + Im^2)
    // ---------------------------------------------------------------
    split(complexI, planes);                 // planes[0] = Re, planes[1] = Im
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];

    // ---------------------------------------------------------------
    // Paso 6: Escala logarítmica  log(1 + magnitud)
    // ---------------------------------------------------------------
    magI += Scalar::all(1);
    log(magI, magI);

    // ---------------------------------------------------------------
    // Paso 7: Reorganizar cuadrantes (DC al centro)
    // ---------------------------------------------------------------
    reorganizarCuadrantes(magI);

    // ---------------------------------------------------------------
    // Paso 8: Normalizar a [0, 1] y mostrar
    // ---------------------------------------------------------------
    normalize(magI, magI, 0, 1, NORM_MINMAX);

    imshow("Imagen original", I);
    imshow("Espectro de magnitud (Fourier)", magI);
    waitKey(0);

    return 0;
}

