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
	//Get the file (text of the path for now, later drag and drop)
	string fileName;
	string path = "C:\\Users\\Tommy\\Documents\\ZomboidAssetWarper\\AssetTiles\\";
	cout << "Type the name of the file from the path (Wall File) - include the format (.png):\n";
	cin >> fileName;
	path = path + fileName;
	//Points to warp (walls only)
	//60,2 - 123,33 - 60,194 - 123,225
	Mat img = imread(path, IMREAD_UNCHANGED);
	Mat matrix, imgWarp;
	if (img.channels() < 4) {
		cout << "Warning: image has no alpha channel (" << img.channels() << " channels)" << endl;
	}
	float w = 128, h = 256;
	//Specific coords for the getting of the 
	Point2f src[4] = { {60,2},{123,33},{60,194},{123,225} };
	Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };
	//Warping the image
	matrix = cv::getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(w, h));
	//Showing the selected coords in the original image
	for (int i = 0; i < 4; i++)
	{
		circle(img, src[i], 10, Scalar(0, 0, 255), FILLED);
	}

	//Show the image after warping and with circles
	imshow("Selected Image", img);
	imshow("Warped Image", imgWarp);

	//Exporting the image by adding the folder output and the name of the file
	srand(time(0));
	int randomNum = (rand() % 900) + 100;
	string outputPath = "C:\\Users\\Tommy\\Documents\\ZomboidAssetWarper\\ConvertedAssets\\" + to_string(randomNum) + "_Wall.png";
	
	
	imwrite(outputPath, imgWarp);

	waitKey(0);

}