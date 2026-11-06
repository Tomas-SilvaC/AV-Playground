#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace cv;
using namespace std;

///////////////  Zomboid Asset Warper + Normal Map  //////////////////////
//
// 1) Carga el tile de pared y le hace el warp de perspectiva (igual que antes).
// 2) A partir de la imagen YA warpeada genera un mapa de normales:
//      gris -> (invertir opcional) -> blur -> Sobel (dx, dy) -> normal (x,y,z)
//      -> color RGB  (R = x, G = y, B = z)
// 3) Guarda dos archivos: NNN_Wall.png (warp) y NNN_Wall_n.png (normales).
//
// Controles en la ventana "Normal Map":
//   Strength : intensidad del relieve (valor / 10)
//   Blur     : suavizado previo del mapa de alturas (0 = nada)
//   Invert   : 1 = invierte claros y oscuros (oscuro = alto)
//   FlipGreen: 1 = formato DirectX (Y invertido), 0 = OpenGL
//   Contrast : contraste local (CLAHE) del mapa de alturas (valor / 10, 0 = apagado)
//   FlatBG   : 1 = fondo transparente con normal plana, 0 = fondo transparente vacio
//   Save     : mueve el slider a 1 para guardar (tambien funciona la tecla S o Enter,
//              pero SOLO si una ventana de imagen tiene el foco, no la consola)
//   Tecla ESC: salir

// ---- Carpetas (ajustar a tu PC) ----
const string INPUT_DIR = "C:\\Users\\Tommy\\Documents\\ZomboidAssetWarper\\AssetTiles\\";
const string OUTPUT_DIR = "C:\\Users\\Tommy\\Documents\\ZomboidAssetWarper\\ConvertedAssets\\";

// Quita espacios y comillas (por si se pega la ruta con "Copiar como ruta de acceso")
static string limpiarTexto(string r)
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

// Una ruta es absoluta si trae "C:\..." o empieza con "/" o "\"
static bool esRutaAbsoluta(const string& r)
{
	return (r.size() > 1 && r[1] == ':') || (!r.empty() && (r[0] == '/' || r[0] == '\\'));
}

// Convierte la imagen warpeada en un mapa de normales (BGR, 3 canales).
// 'alpha' define que pixeles son parte del objeto; el resto queda como normal plana.
static Mat crearMapaNormales(const Mat& bgr, const Mat& alpha,
	float strength, int blurRadius, bool invert, bool flipGreen,
	double contrastClip, bool fondoPlano)
{
	// 1) Escala de grises (+ inversion opcional)
	Mat gray;
	cvtColor(bgr, gray, COLOR_BGR2GRAY);
	if (invert) bitwise_not(gray, gray);

	// 2) Los pixeles transparentes se rellenan con la altura media del objeto,
	//    para que el borde del sprite no genere un relieve falso.
	Mat mask = alpha >= 128, maskInv;
	bitwise_not(mask, maskInv);
	Scalar media = mean(gray, mask);
	gray.setTo(media[0], maskInv);

	// 2b) Contraste local (CLAHE): resalta detalles en zonas de poco contraste
	//     sin disparar las zonas que ya tienen mucho. 0 = apagado.
	if (contrastClip > 0.0) {
		Ptr<CLAHE> clahe = createCLAHE(contrastClip, Size(4, 4));
		clahe->apply(gray, gray);
	}

	// Mapa de alturas en [0,1]
	Mat height;
	gray.convertTo(height, CV_32F, 1.0 / 255.0);

	// 3) Suavizado opcional (reduce el ruido del pixel art)
	if (blurRadius > 0) {
		int k = blurRadius * 2 + 1;
		GaussianBlur(height, height, Size(k, k), 0);
	}

	// 4) Gradientes (Sobel normalizado: scale 1/8)
	Mat dx, dy;
	Sobel(height, dx, CV_32F, 1, 0, 3, 1.0 / 8.0, 0, BORDER_REPLICATE);
	Sobel(height, dy, CV_32F, 0, 1, 3, 1.0 / 8.0, 0, BORDER_REPLICATE);

	// 5) Vector normal (x, y, z) y normalizacion
	//    En imagen Y crece hacia abajo: OpenGL usa +Y arriba, DirectX lo invierte.
	Mat nx = dx * (-strength);
	Mat ny = dy * (flipGreen ? -strength : strength);
	Mat nz = Mat::ones(height.size(), CV_32F);

	Mat sum = nx.mul(nx) + ny.mul(ny);
	sum += 1.0f;
	Mat len;
	sqrt(sum, len);
	nx /= len;
	ny /= len;
	nz /= len;

	// 6) Codificar [-1,1] -> [0,255]  (v * 127.5 + 127.5)
	Mat r8, g8, b8;
	nx.convertTo(r8, CV_8U, 127.5, 127.5);
	ny.convertTo(g8, CV_8U, 127.5, 127.5);
	nz.convertTo(b8, CV_8U, 127.5, 127.5);

	vector<Mat> canales = { b8, g8, r8 };   // OpenCV usa orden BGR
	Mat normal;
	merge(canales, normal);

	// 7) Fuera del objeto: normal plana (128,128,255)
	normal.setTo(Scalar(255, 128, 128), maskInv);

	// 8) Fondo completamente transparente: sin color (negro) si no se quiere normal plana.
	//    (Los bordes semitransparentes conservan la normal plana para evitar halos oscuros.)
	if (!fondoPlano) {
		Mat vacio = alpha == 0;
		normal.setTo(Scalar(0, 0, 0), vacio);
	}
	return normal;
}

// Directorio de una ruta (con el separador final). Vacio si no tiene.
static string directorioDe(const string& ruta)
{
	size_t pos = ruta.find_last_of("\\/");
	return (pos == string::npos) ? "" : ruta.substr(0, pos + 1);
}

// Intenta escribir una imagen; devuelve false si falla (carpeta inexistente, sin permisos, etc.)
static bool escribirImagen(const string& archivo, const Mat& imagen)
{
	try {
		return imwrite(archivo, imagen);
	}
	catch (const cv::Exception& e) {
		cerr << "Error de OpenCV al guardar: " << e.what() << endl;
		return false;
	}
}

// Guarda warp + normales en 'carpeta'. Devuelve true si ambos archivos se escribieron.
static bool guardarPar(const string& carpeta, const string& nombreBase,
	const Mat& imgWarp, const Mat& normalConAlfa)
{
	if (!carpeta.empty()) {
		try { cv::utils::fs::createDirectories(carpeta); }
		catch (...) {}
	}
	string baseName = carpeta + nombreBase;
	bool ok1 = escribirImagen(baseName + ".png", imgWarp);
	bool ok2 = escribirImagen(baseName + "_n.png", normalConAlfa);
	if (ok1 && ok2)
		cout << "Guardado:\n  " << baseName << ".png\n  " << baseName << "_n.png\n";
	return ok1 && ok2;
}

int main()
{
	// ---------- Obtener el archivo ----------
	string entrada;
	cout << "Nombre del archivo (o ruta completa pegada) - incluye el formato (.png):\n";
	getline(cin, entrada);
	entrada = limpiarTexto(entrada);
	string path = esRutaAbsoluta(entrada) ? entrada : INPUT_DIR + entrada;

	Mat img = imread(path, IMREAD_UNCHANGED);
	if (img.empty()) {
		cerr << "No se pudo abrir la imagen: " << path << endl;
		return -1;
	}
	if (img.channels() < 4) {
		cout << "Warning: image has no alpha channel (" << img.channels() << " channels)" << endl;
	}

	// ---------- Warp (solo paredes) ----------
	// 60,2 - 123,33 - 60,194 - 123,225
	float w = 128, h = 256;
	Point2f src[4] = { {60,2},{123,33},{60,194},{123,225} };
	Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

	Mat matrix = getPerspectiveTransform(src, dst);
	Mat imgWarp;
	warpPerspective(img, imgWarp, matrix, Size((int)w, (int)h));

	// Separar color (BGR) y alfa de la imagen warpeada
	Mat warpBGR, warpAlpha;
	bool hayAlfa = imgWarp.channels() == 4;
	if (hayAlfa) {
		vector<Mat> ch;
		split(imgWarp, ch);
		warpAlpha = ch[3];
		merge(vector<Mat>{ ch[0], ch[1], ch[2] }, warpBGR);
	}
	else {
		warpBGR = imgWarp.clone();
		warpAlpha = Mat(imgWarp.size(), CV_8U, Scalar(255));
	}

	// Mostrar los puntos elegidos en la imagen original
	Mat imgPuntos = img.clone();
	for (int i = 0; i < 4; i++)
		circle(imgPuntos, src[i], 10, Scalar(0, 0, 255), FILLED);

	imshow("Selected Image", imgPuntos);
	imshow("Warped Image", imgWarp);

	// ---------- Mapa de normales con controles ----------
	const string win = "Normal Map";
	namedWindow(win, WINDOW_AUTOSIZE);
	int strengthI = 30;   // 3.0
	int blurI = 1;
	int invertI = 0;
	int flipGreenI = 0;
	int contrastI = 0;    // 0 = apagado, 10 = 1.0, ...
	int fondoPlanoI = 0;  // 0 = fondo vacio, 1 = normal plana
	createTrackbar("Strength x10", win, &strengthI, 100);
	createTrackbar("Blur", win, &blurI, 10);
	createTrackbar("Invert", win, &invertI, 1);
	createTrackbar("FlipGreen", win, &flipGreenI, 1);
	createTrackbar("Contrast x10", win, &contrastI, 40);
	createTrackbar("FlatBG", win, &fondoPlanoI, 1);

	// Numero aleatorio compartido por los dos archivos
	srand((unsigned)time(0));
	int randomNum = (rand() % 900) + 100;
	string nombreBase = to_string(randomNum) + "_Wall";

	int guardarI = 0;   // "boton" Save: mover a 1 para guardar
	createTrackbar("Save (1=guardar)", win, &guardarI, 1);

	Mat normal;
	int prev[6] = { -1, -1, -1, -1, -1, -1 };
	string mensaje;
	int mensajeFrames = 0;

	cout << "\nPara guardar: slider 'Save' a 1, o tecla S / Enter con una ventana de imagen activa."
		<< "\nESC = salir\n";

	while (true) {
		// Recalcular solo si cambio algun control
		int actual[6] = { strengthI, blurI, invertI, flipGreenI, contrastI, fondoPlanoI };
		bool cambio = false;
		for (int i = 0; i < 6; i++) if (actual[i] != prev[i]) { cambio = true; prev[i] = actual[i]; }

		if (cambio) {
			normal = crearMapaNormales(warpBGR, warpAlpha,
				strengthI / 10.0f, blurI, invertI == 1, flipGreenI == 1,
				contrastI / 10.0, fondoPlanoI == 1);
			imshow(win, normal);
		}

		int tecla = waitKey(30) & 0xFF;
		if (tecla == 27) break;                                   // ESC

		bool pedirGuardar = (tecla == 's' || tecla == 'S' || tecla == 13 || guardarI == 1);
		if (guardarI == 1) setTrackbarPos("Save (1=guardar)", win, 0);   // soltar el "boton"

		if (pedirGuardar && !normal.empty()) {
			// Mapa de normales con el mismo alfa que el warp
			Mat normalGuardar;
			if (hayAlfa) {
				vector<Mat> ch;
				split(normal, ch);
				ch.push_back(warpAlpha);
				merge(ch, normalGuardar);
			}
			else {
				normalGuardar = normal;
			}

			// 1) Carpeta de salida configurada
			bool ok = guardarPar(OUTPUT_DIR, nombreBase, imgWarp, normalGuardar);

			// 2) Si falla, guardar junto al archivo original
			if (!ok) {
				string alterna = directorioDe(path);
				cerr << "No se pudo guardar en: " << OUTPUT_DIR
					<< "\nProbando en la carpeta de la imagen: "
					<< (alterna.empty() ? "(carpeta actual)" : alterna) << endl;
				ok = guardarPar(alterna, nombreBase, imgWarp, normalGuardar);
			}

			mensaje = ok ? "Guardado!" : "ERROR al guardar";
			if (!ok) cerr << "No se pudo guardar el archivo." << endl;
			mensajeFrames = 70;   // ~2 segundos
		}

		// Aviso visual en la ventana del mapa de normales
		if (mensajeFrames > 0 && !normal.empty()) {
			Mat vista = normal.clone();
			putText(vista, mensaje, Point(4, 16), FONT_HERSHEY_SIMPLEX, 0.45,
				Scalar(0, 255, 0), 1, LINE_AA);
			imshow(win, vista);
			if (--mensajeFrames == 0) imshow(win, normal);
		}
	}
	return 0;
}