/**
 * BoxFilter.cpp
 * Filtro de caja (box filter) sobre una imagen en escala de grises.
 *
 * ## Idea general
 * Filtrar una imagen es convolucionarla con un kernel: una matriz pequeña que se
 * desliza sobre la imagen; en cada posición se multiplican los valores del kernel
 * por los píxeles que cubre, se suman los resultados y el total reemplaza al píxel
 * central.
 *
 * El filtro de caja es el kernel lineal de suavizado más simple: todos los pesos
 * son iguales y suman 1, es decir cada elemento vale 1/(k*k). Equivale a reemplazar
 * cada píxel por el PROMEDIO de su vecindario k x k:
 *
 *        | 1 1 1 |
 *   1/9  | 1 1 1 |      (ejemplo con k = 3)
 *        | 1 1 1 |
 *
 * Es un filtro espacial lineal (actúa directamente sobre la cuadrícula de píxeles).
 *
 * ## Uso
 * Pide por consola la ruta de la imagen y el tamaño del kernel (impar, 5 por defecto).
 * Muestra original, gris y filtrada, y guarda gris.jpg y filtro_caja.jpg.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main() {
    std::string rutaImagen;
    int kernelSize = 5; // Tamaño del kernel predeterminado

    std::cout << "=== Filtro de Caja y Escala de Grises ===\n";
    std::cout << "Ingresa la ruta de la imagen: ";
    std::getline(std::cin, rutaImagen);

    // Por si el usuario pega la ruta con comillas (Shift+Click derecho -> "Copiar como ruta de acceso")
    if (!rutaImagen.empty() && rutaImagen.front() == '"' && rutaImagen.back() == '"') {
        rutaImagen = rutaImagen.substr(1, rutaImagen.size() - 2);
    }

    // El kernel debe tener lado impar para tener un píxel central.
    // Si el usuario ingresa un par, se suma 1; si es inválido, se usa el valor por defecto.
    std::cout << "Tamano del kernel (impar, Enter para usar 5 por defecto): ";
    std::string entradaKernel;
    std::getline(std::cin, entradaKernel);
    if (!entradaKernel.empty()) {
        try {
            kernelSize = std::stoi(entradaKernel);
            if (kernelSize <= 0) {
                std::cout << "Valor invalido, se usara 5.\n";
                kernelSize = 5;
            }
            else if (kernelSize % 2 == 0) {
                kernelSize += 1;
                std::cout << "El tamano debe ser impar, se ajusto a " << kernelSize << "\n";
            }
        }
        catch (...) {
            std::cout << "Valor invalido, se usara el tamano por defecto (5)\n";
            kernelSize = 5;
        }
    }

    cv::Mat imagenOriginal = cv::imread(rutaImagen);
    if (imagenOriginal.empty()) {
        std::cerr << "\nError: no se pudo cargar la imagen en la ruta:\n" << rutaImagen << "\n";
        std::cerr << "Verifica que la ruta sea correcta y que el archivo exista.\n";
        std::cout << "\nPresiona Enter para salir...";
        std::cin.get();
        return -1;
    }

    // 1) Convertir a escala de grises
    //    (OpenCV carga en orden BGR; se reduce a un solo canal de intensidad).
    cv::Mat imagenGris;
    cv::cvtColor(imagenOriginal, imagenGris, cv::COLOR_BGR2GRAY);

    // 2) Aplicar filtro de caja sobre la version en grises
    //    ddepth = -1 -> la salida conserva la profundidad de la entrada (8 bits).
    //    Por defecto boxFilter normaliza (divide por k*k), es decir, promedia.
    cv::Mat imagenFiltrada;
    cv::boxFilter(imagenGris, imagenFiltrada, -1, cv::Size(kernelSize, kernelSize));

    // Mostrar resultados en pantalla
    cv::imshow("Original", imagenOriginal);
    cv::imshow("Escala de grises", imagenGris);
    cv::imshow("Filtro de caja (kernel " + std::to_string(kernelSize) + ")", imagenFiltrada);

    // Guardar resultados junto al ejecutable
    cv::imwrite("gris.jpg", imagenGris);
    cv::imwrite("filtro_caja.jpg", imagenFiltrada);
    std::cout << "\nImagenes guardadas como 'gris.jpg' y 'filtro_caja.jpg' en la carpeta del ejecutable.\n";
    std::cout << "Con una ventana de imagen seleccionada, presiona cualquier tecla para cerrar...\n";

    cv::waitKey(0);
    return 0;
}