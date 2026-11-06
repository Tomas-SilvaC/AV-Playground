#include <opencv2/core/version.hpp>

// Always need these for basic image I/O, color conversion, and display
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/calib.hpp>
#include <opencv2/objdetect.hpp>
// NOTE: verify this path against your actual OpenCV 5 install
// (check <opencv_install_dir>/include/opencv2/ for the real file/folder name).
// Some sources list it as opencv2/geometry/3d.hpp, others as opencv2/3d.hpp.
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
    // Vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f>> objpoints;
    // Vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f>> imgpoints;

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
    {
        for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j, i, 0));
    }

    // Extracting path of individual image stored in a given directory
    std::vector<cv::String> images;
    // Path of the folder containing checkerboard images
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
    // Vector to store the pixel coordinates of detected checkerboard corners
    std::vector<cv::Point2f> corner_pts;
    bool success;

    // Looping over all the images in the directory
    for (size_t i{ 0 }; i < images.size(); i++)
    {
        frame = cv::imread(images[i]);
        if (frame.empty())
        {
            std::cerr << "Could not read image: " << images[i] << std::endl;
            continue;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Finding checkerboard corners
        success = cv::findChessboardCorners(
            gray,
            cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]),
            corner_pts,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (success)
        {
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

            // Refining pixel coordinates for given 2D points
            cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            // Displaying the detected corner points on the checkerboard
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }

        cv::imshow("Image", frame);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();

    if (objpoints.empty())
    {
        std::cerr << "No checkerboard was successfully detected in any image -- "
            << "calibration cannot proceed." << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix, distCoeffs;
    // rvecs/tvecs must be vectors -- one rotation & translation vector per image,
    // not a single cv::Mat
    std::vector<cv::Mat> rvecs, tvecs;

    /*
     * Performing camera calibration by passing the value of known 3D points
     * (objpoints) and corresponding pixel coordinates of the detected
     * corners (imgpoints)
     */
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

    // Intentar que la camara capture al mismo tamano que las imagenes de
    // calibracion, porque cameraMatrix/distCoeffs son validos para esa
    // resolucion. Algunos drivers ignoran este set(); si tu camara no
    // respeta este tamano, lo ideal es recalibrar con esa resolucion.
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

    // Precalcular los mapas de undistorsion una sola vez (mucho mas
    // eficiente que llamar a cv::undistort en cada frame)
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, imageSize, 1, imageSize);

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

        // Detectar y pintar las esquinas del tablero sobre la imagen ya corregida
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
        if (key == 'q' || key == 27) // 27 = tecla ESC
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}