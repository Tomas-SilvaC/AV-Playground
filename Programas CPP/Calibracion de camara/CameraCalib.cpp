/**
 * CameraCalib.cpp
 * Calibración de cámara con tablero de ajedrez y corrección de distorsión en vivo.
 *
 * Toda lente deforma la imagen. Calibrar la cámara es estimar matemáticamente esa
 * deformación para poder corregirla por software. Se obtienen:
 *
 *   - Matriz intrínseca K (3x3): distancia focal (fx, fy, en píxeles) y punto
 *     principal (cx, cy). Modela una cámara estenopeica (pinhole) ideal y es única
 *     para cada combinación cámara + lente + resolución.
 *   - Coeficientes de distorsión D: radial (k1, k2, k3: barril o cojín) y
 *     tangencial (p1, p2: lente no paralela al sensor). El modelo estándar usa 5.
 *     (Para lentes ojo de pez extremas se debe usar el módulo cv::fisheye, que
 *     emplea otro modelo de proyección.)
 *   - Extrínsecos por imagen: vectores de rotación (rvecs) y traslación (tvecs) del
 *     tablero respecto a la cámara.
 *
 * ## Procedimiento
 *   1. Definir las coordenadas 3D del tablero (plano, Z = 0), en "unidades de casilla".
 *   2. Para cada foto: detectar las esquinas interiores y refinarlas a precisión subpíxel.
 *   3. cv::calibrateCamera relaciona los puntos 3D con los 2D detectados y devuelve
 *      K, D, rvecs, tvecs y el error RMS de reproyección (cuanto menor, mejor;
 *      valores < 1 px suelen considerarse buenos).
 *   4. Guardar los resultados en calibration.yml.
 *   5. Usarlos en vivo: construir los mapas de corrección una sola vez y aplicar
 *      cv::remap en cada fotograma de la cámara.
 *
 * Importante: CHECKERBOARD cuenta ESQUINAS INTERIORES, no casillas. Un tablero de
 * 6x9 casillas tiene 5x8 esquinas interiores. Para una buena calibración conviene
 * usar ~15-25 fotos con el tablero en distintas posiciones, ángulos y zonas del
 * encuadre (cobertura > cantidad).
 *
 * ## Entrada
 *   ./images/*.jpg   fotos del tablero, junto al ejecutable (o al .vcxproj).
 */

#include <opencv2/core/version.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/calib.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/geometry/3d.hpp>
#else
#include <opencv2/calib3d.hpp>
#endif

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{ 5, 8 };

int main()
{
    // Puntos 3D del mundo real, uno vector por imagen
    std::vector<std::vector<cv::Point3f>> objpoints;
    // Puntos 2D (píxeles) detectados, un vector por imagen
    std::vector<std::vector<cv::Point2f>> imgpoints;

    // Coordenadas 3D de las esquinas del tablero: (0,0,0), (1,0,0), ... con Z = 0
    // porque el tablero es plano. Las unidades son "casillas"; si se usara el
    // tamaño real de la casilla (p. ej. 25 mm), los tvecs saldrían en mm.
    std::vector<cv::Point3f> objp;
    for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
    {
        for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j, i, 0));
    }

    // Listar todas las imágenes de calibración de la carpeta
    std::vector<cv::String> images;
    std::string path = "./images/*.jpg";
    cv::glob(path, images);

    if (images.empty())
    {
        std::cerr << "No images found at " << path
            << " -- check that the 'images' folder sits next to the executable "
            << "(or next to the .vcxproj, depending on your working directory) "
            << "and that the files are .jpg." << std::endl;
        return -1;
    }

    cv::Mat frame, gray;
    // Coordenadas en píxeles de las esquinas detectadas en la imagen actual
    std::vector<cv::Point2f> corner_pts;
    bool success;

    // Recorrer todas las imágenes
    for (size_t i{ 0 }; i < images.size(); i++)
    {
        frame = cv::imread(images[i]);
        if (frame.empty())
        {
            std::cerr << "Could not read image: " << images[i] << std::endl;
            continue;
        }
        // La detección de esquinas trabaja sobre intensidades
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        //  Buscar las esquinas interiores del tablero.
        //  ADAPTIVE_THRESH : binariza con umbral adaptativo (robusto a iluminación irregular)
        //  FAST_CHECK      : descarte rápido si claramente no hay tablero
        //  NORMALIZE_IMAGE : ecualiza el contraste antes de binarizar
        success = cv::findChessboardCorners(
            gray,
            cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]),
            corner_pts,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (success)
        {   // Criterio de parada del refinamiento: 30 iteraciones o cambio < 0.001 px
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

            // Refinar a precisión subpíxel (ventana de búsqueda de 11x11)
            cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            // Dibujar las esquinas detectadas sobre la imagen para inspección visual
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
            // Solo las imágenes con detección exitosa aportan a la calibración
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }

        cv::imshow("Image", frame);
        cv::waitKey(0); // pasar a la siguiente imagen con cualquier tecla
    }

    cv::destroyAllWindows();

    if (objpoints.empty())
    {
        std::cerr << "No checkerboard was successfully detected in any image -- "
            << "calibration cannot proceed." << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix, distCoeffs;
    // rvecs/tvecs deben ser vectores: un vector de rotación y uno de traslación
    // por cada imagen (no una única cv::Mat)
    std::vector<cv::Mat> rvecs, tvecs;
    
    // Calibración: a partir de los puntos 3D conocidos (objpoints) y sus
    // correspondencias en píxeles (imgpoints), optimiza K, D y la pose del
    // tablero en cada foto minimizando el error de reproyección.
    // Devuelve el error RMS de reproyección en píxeles.
    double rms = cv::calibrateCamera(objpoints, imgpoints, gray.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

    std::cout << "RMS re-projection error : " << rms << std::endl;
    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << distCoeffs << std::endl;
    std::cout << "Rotation vectors : " << std::endl;
    for (const auto& r : rvecs) std::cout << r << std::endl;
    std::cout << "Translation vectors : " << std::endl;
    for (const auto& t : tvecs) std::cout << t << std::endl;

    // ------------------------------------------------------------------
    // Exportar los resultados de la calibración a un archivo .yml
    // ------------------------------------------------------------------
    std::string outputFile = "calibration.yml";
    cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);

    if (!fs.isOpened())
    {
        std::cerr << "No se pudo abrir " << outputFile << " para escritura." << std::endl;
        return -1;
    }

    // Marca de tiempo (opcional, pero util para saber cuando se calibro)
    std::time_t rawTime = std::time(nullptr);
    std::tm timeInfo{};
    char timeBuf[64];

#if defined(_WIN32)
    localtime_s(&timeInfo, &rawTime);   // version thread-safe de Windows
#else
    localtime_r(&rawTime, &timeInfo);   // version thread-safe de POSIX (Linux/Mac)
#endif

    std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &timeInfo);
    // Se guarda también la resolución: K y D solo son válidos para ella.
    fs << "calibration_date" << timeBuf;
    fs << "image_width" << gray.cols;
    fs << "image_height" << gray.rows;
    fs << "board_width" << CHECKERBOARD[0];
    fs << "board_height" << CHECKERBOARD[1];
    fs << "num_images_used" << static_cast<int>(objpoints.size());
    fs << "rms_reprojection_error" << rms;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "rotation_vectors" << rvecs;
    fs << "translation_vectors" << tvecs;

    fs.release();

    // Mostrar la ruta absoluta para saber exactamente donde quedo el archivo
    std::string absolutePath = cv::utils::fs::canonical(outputFile);
    std::cout << "Parametros de calibracion guardados en: " << absolutePath << std::endl;

    // ------------------------------------------------------------------
    // Usar la calibracion en tiempo real: abrir la camara y mostrar
    // el video corregido (sin distorsion) lado a lado con el original
    // ------------------------------------------------------------------
    int cameraIndex = 1; // cambiar si tenes mas de una camara conectada
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened())
    {
        std::cerr << "No se pudo abrir la camara (indice " << cameraIndex << ")." << std::endl;
        return -1;
    }

    // Intentar que la camara capture al mismo tamaño que las imagenes de
    // calibracion, porque cameraMatrix/distCoeffs son validos para esa
    // resolucion. Algunos drivers ignoran este set();
    cap.set(cv::CAP_PROP_FRAME_WIDTH, gray.cols);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, gray.rows);

    cv::Size imageSize(
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

    if (imageSize != gray.size())
    {
        std::cerr << "Aviso: la camara esta capturando a " << imageSize
            << " pero la calibracion se hizo a " << gray.size()
            << ". La correccion puede no ser exacta." << std::endl;
    }

    // Precalcular los mapas de undistorsion una sola vez (mas
    // eficiente que llamar a cv::undistort en cada frame).
    // alpha = 1 en getOptimalNewCameraMatrix conserva todos los píxeles de la
    // imagen original (puede dejar bordes negros tras corregir la distorsión).
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, imageSize, 1, imageSize);
    // Cada píxel destino guarda en map1/map2 de qué posición de la imagen
    // distorsionada debe leer su valor.
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(
        cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix,
        imageSize, CV_16SC2, map1, map2);

    std::cout << "Camara abierta. Presiona 'q' o ESC para salir." << std::endl;

    cv::Size boardSize(CHECKERBOARD[0], CHECKERBOARD[1]);
    cv::TermCriteria cornerCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

    cv::Mat liveFrame, undistorted, liveGray, undistortedGray;
    while (true)
    {
        cap >> liveFrame;
        if (liveFrame.empty())
        {
            std::cerr << "Frame vacio recibido de la camara, saliendo." << std::endl;
            break;
        }

        // Corregir la distorsión aplicando los mapas precalculados
        cv::remap(liveFrame, undistorted, map1, map2, cv::INTER_LINEAR);

        // Detectar y pintar las esquinas del tablero sobre la imagen original
        // NOTA: sin CALIB_CB_FAST_CHECK aca -- ese flag da falsos negativos
        // con video (blur de movimiento / ruido), a diferencia de fotos fijas.
        cv::cvtColor(liveFrame, liveGray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> liveCorners;
        bool foundLive = cv::findChessboardCorners(
            liveGray, boardSize, liveCorners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (foundLive)
        {
            cv::cornerSubPix(liveGray, liveCorners, cv::Size(11, 11), cv::Size(-1, -1), cornerCriteria);
            cv::drawChessboardCorners(liveFrame, boardSize, liveCorners, foundLive);
        }
        cv::putText(liveFrame, foundLive ? "Detectado" : "No detectado",
            cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
            foundLive ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        // Detectar y pintar las esquinas del tablero sobre la imagen ya corregida.
        // Comparar ambas vistas permite verificar visualmente que las líneas rectas
        // del tablero si quedan rectas tras la corrección.
        cv::cvtColor(undistorted, undistortedGray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> undistortedCorners;
        bool foundUndistorted = cv::findChessboardCorners(
            undistortedGray, boardSize, undistortedCorners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (foundUndistorted)
        {
            cv::cornerSubPix(undistortedGray, undistortedCorners, cv::Size(11, 11), cv::Size(-1, -1), cornerCriteria);
            cv::drawChessboardCorners(undistorted, boardSize, undistortedCorners, foundUndistorted);
        }
        cv::putText(undistorted, foundUndistorted ? "Detectado" : "No detectado",
            cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
            foundUndistorted ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        cv::imshow("Original", liveFrame);
        cv::imshow("Corregido (undistorted)", undistorted);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) // 27 = tecla ESC para parar el programa
            break;
    }
    // Liberar siempre la cámara para evitar bloquearla
    cap.release();
    cv::destroyAllWindows();

    return 0;
}