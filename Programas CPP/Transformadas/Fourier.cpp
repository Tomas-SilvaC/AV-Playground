/**
 * Fourier.cpp
 * Transformada de Fourier 2D de una imagen y visualización de su espectro de magnitud.
 *
 * ## Idea general
 * Una imagen en escala de grises es una señal 2D discreta f(x, y). La transformada
 * discreta de Fourier (DFT) la descompone en una suma de ondas seno/coseno 2D de
 * distintas frecuencias y orientaciones. Como esas ondas son ortogonales entre sí,
 * cada componente de frecuencia aporta su energía de forma independiente:
 *
 *   - Frecuencias bajas  -> zonas suaves, variaciones lentas (sombreado, cielo liso).
 *   - Frecuencias altas  -> bordes y texturas finas (cambios rápidos entre píxeles).
 *
 * El resultado de la DFT son números complejos (parte real + parte imaginaria). Para
 * poder "verlo" se calcula su magnitud sqrt(Re^2 + Im^2), que indica cuánta energía
 * hay en cada frecuencia, y se muestra como imagen (el "espectro").
 *
 * ## Pasos que implementa este programa
 *   1. Cargar la imagen en escala de grises (un solo canal de intensidad).
 *   2. Rellenar (padding) con ceros hasta un tamaño óptimo para la FFT.
 *   3. Construir la imagen compleja (real = imagen, imaginaria = 0).
 *   4. Aplicar la DFT (dominio espacial -> dominio de la frecuencia).
 *   5. Calcular el espectro de magnitud.
 *   6. Pasarlo a escala logarítmica (comprime el rango dinámico para poder verlo).
 *   7. Reorganizar cuadrantes para llevar la componente DC (frecuencia 0) al centro.
 *   8. Normalizar a [0, 1] y mostrar.
 *
 * ## Compilar
 *   g++ fourier.cpp -o fourier `pkg-config --cflags --libs opencv4`
 *
 * ## Ejecutar
 *   ./fourier                -> el programa pide la ruta (copiar y pegar)
 *   ./fourier imagen.jpg     -> también acepta la ruta como argumento
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;


// Paso 7: reorganiza los cuadrantes del espectro.
// La DFT entrega la componente DC (frecuencia 0, el brillo medio de la imagen) en la
// esquina superior izquierda, y las frecuencias "negativas" quedan en los otros
// cuadrantes. Intercambiando los cuadrantes en diagonal (0<->3 y 1<->2) la DC queda
// en el centro y las frecuencias altas hacia los bordes, que es la forma habitual
// de interpretar un espectro.
// @param mag  Espectro de magnitud; se modifica en el lugar (y se recorta a tamaño par).
 
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


// Limpia una ruta pegada desde la consola: quita espacios, saltos de línea y comillas.
// (Windows añade comillas con "Copiar como ruta de acceso").
// @param r  Texto crudo ingresado por el usuario.
// @return   Ruta lista para usar con imread, o cadena vacía si no había contenido.
//
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
    //         El análisis de Fourier opera sobre intensidades, así que el
    //         color no hace falta. La ruta se puede pegar en la consola o
    //         pasar como argumento; si no se puede abrir, se vuelve a pedir.
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
    //         La FFT es mucho más rápida cuando el tamaño es producto de
    //         potencias de 2, 3 y 5. getOptimalDFTSize devuelve el menor
    //         tamaño válido >= al dado; el relleno de ceros no altera el
    //         contenido frecuencial de forma relevante.
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
    //         La DFT trabaja con números complejos: se arma una matriz de
    //         2 canales (real, imaginaria). La real es la imagen (en float)
    //         y la imaginaria se inicializa a cero.
    // ---------------------------------------------------------------
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    // ---------------------------------------------------------------
    // Paso 4: Aplicar la DFT (dominio espacial -> dominio de frecuencia)
    //         Cada elemento del resultado es el coeficiente complejo de una
    //         frecuencia espacial (u, v). Se hace in-place.
    // ---------------------------------------------------------------
    dft(complexI, complexI);

    // ---------------------------------------------------------------
    // Paso 5: Espectro de magnitud = sqrt(Re^2 + Im^2)
    //         Indica la "intensidad" de cada frecuencia (se descarta la fase).
    // ---------------------------------------------------------------
    split(complexI, planes);                 // planes[0] = Re, planes[1] = Im
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];

    // ---------------------------------------------------------------
    // Paso 6: Escala logarítmica  log(1 + magnitud)
    //         Las magnitudes crudas varían en varios órdenes de magnitud
    //         (la DC domina). El logaritmo comprime el rango para que se
    //         aprecien las frecuencias débiles; el "+1" evita log(0).
    // ---------------------------------------------------------------
    magI += Scalar::all(1);
    log(magI, magI);

    // ---------------------------------------------------------------
    // Paso 7: Reorganizar cuadrantes (DC al centro)
    // ---------------------------------------------------------------
    reorganizarCuadrantes(magI);

    // ---------------------------------------------------------------
    // Paso 8: Normalizar a [0, 1] y mostrar
    //         imshow interpreta los float en [0, 1] como escala de grises.
    // ---------------------------------------------------------------
    normalize(magI, magI, 0, 1, NORM_MINMAX);

    imshow("Imagen original", I);
    imshow("Espectro de magnitud (Fourier)", magI);
    waitKey(0);

    return 0;
}

