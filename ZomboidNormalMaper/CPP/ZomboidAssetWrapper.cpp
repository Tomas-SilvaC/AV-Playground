/**
 * ZomboidAssetWrapper.cpp
 * Versión inicial del "Zomboid Asset Warper": endereza (warp) un tile de pared.
 *
 * ## Idea general
 * Los tiles de pared de Project Zomboid se dibujan en perspectiva isométrica: el
 * rectángulo de la pared aparece como un paralelogramo/cuadrilátero inclinado.
 * Para reutilizarlo como textura plana se aplica una transformación de
 * PERSPECTIVA (homografía):
 *
 *   - Se eligen 4 puntos de la imagen original (las esquinas del cuadrilátero).
 *   - Se indica a qué 4 esquinas de un rectángulo de salida deben ir.
 *   - cv::getPerspectiveTransform calcula la matriz 3x3 H que lo logra.
 *   - cv::warpPerspective aplica H a cada píxel (coordenadas homogéneas, con la
 *     división por w que produce el efecto de perspectiva).
 *
 * A diferencia de una transformación afín (2x3, 3 pares de puntos, conserva el
 * paralelismo), la de perspectiva usa 4 pares de puntos y las rectas paralelas
 * pueden converger, por eso permite "enderezar" un plano visto en perspectiva.
 *
 * Esta versión es la base del programa ZomboidNormalMaper.cpp, que añade el mapa
 * de normales.
 *
 * Nota: usa `void main()` (extensión de MSVC); en otros compiladores debería ser
 * `int main()`. Las rutas son absolutas y específicas del equipo original.
 */

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <cstdlib>
#include <ctime> 

using namespace cv;
using namespace std;

///////////////  Zomboid Asset Warper  //////////////////////

void main() {
	// Obtener el archivo (por ahora el nombre escrito en consola; luego drag and drop)
	string fileName;
	string path = "C:\\Users\\Tommy\\Documents\\ZomboidAssetWarper\\AssetTiles\\";
	cout << "Type the name of the file from the path (Wall File) - include the format (.png):\n";
	cin >> fileName;
	path = path + fileName;

	// Puntos a deformar (solo paredes): TL, TR, BL, BR
	// 60,2 - 123,33 - 60,194 - 123,225
	// IMREAD_UNCHANGED conserva el canal alfa (transparencia) del sprite.
	Mat img = imread(path, IMREAD_UNCHANGED);
	Mat matrix, imgWarp;
	if (img.channels() < 4) {
		cout << "Warning: image has no alpha channel (" << img.channels() << " channels)" << endl;
	}

	// Tamaño del rectángulo de salida (en píxeles)
	float w = 128, h = 256;

	// Coordenadas específicas de las 4 esquinas de la pared en la imagen original (src)
	// y a dónde deben ir en la imagen enderezada (dst): las esquinas del rectángulo w x h.
	// El orden debe coincidir punto a punto entre src y dst.
	Point2f src[4] = { {60,2},{123,33},{60,194},{123,225} };
	Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };
	
	// Calcular la matriz de perspectiva 3x3 y aplicarla (warp)
	matrix = cv::getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(w, h));

	// Dibujar los puntos elegidos sobre la imagen original (círculos rojos en BGR)
	for (int i = 0; i < 4; i++)
	{
		circle(img, src[i], 10, Scalar(0, 0, 255), FILLED);
	}

	// Mostrar la imagen original con los puntos y la imagen ya deformada
	imshow("Selected Image", img);
	imshow("Warped Image", imgWarp);

	// Exportar el resultado con un número aleatorio de 3 cifras (100-999) en el nombre
	// para no sobrescribir exportaciones anteriores.
	srand(time(0));
	int randomNum = (rand() % 900) + 100;
	string outputPath = "C:\\Users\\Tommy\\Documents\\ZomboidAssetWarper\\ConvertedAssets\\" + to_string(randomNum) + "_Wall.png";
	
	
	imwrite(outputPath, imgWarp);

	waitKey(0);

}