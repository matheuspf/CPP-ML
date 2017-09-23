#include "Snake.h"




int main ()
{
	//auto rgb = cv::imread("img4.jpg");
	auto rgb = cv::imread("/home/matheus/Algoritmos/Vision/ISIC_1/ISIC_0000003.jpg");

	cv::resize(rgb, rgb, cv::Size(512, 512));


	Img img;

	cv::cvtColor(rgb, img, cv::COLOR_BGR2GRAY);


	//cv::GaussianBlur(img, img, cv::Size(3, 3), 0, 0);

	//cv::resize(img, img, cv::Size(min(img.rows, img.cols), min(img.rows, img.cols)));


	Snake snake(2, 1.0, 0.0, 1.0);


	auto points = snake(img);

	DB(points.size());

	for(int i = 0; i < points.size(); ++i)
	{
		cv::line(rgb, points[i], points[(i+1) % points.size()], cv::Scalar(0, 255, 0), 2, 8);

		cv::circle(rgb, points[i], 2, cv::Scalar(0, 0, 255), -1);
	}



	double maxVal = -1e20;

	Img gvf(snake.mag.rows(), snake.mag.cols());

	FOR(i, gvf.rows) FOR(j, gvf.cols)
		maxVal = max(maxVal, abs(snake.mag(i, j)));

	FOR(i, gvf.rows) FOR(j, gvf.cols)
		gvf(i, j) = llround(255 * (abs(snake.mag(i, j)) / max(maxVal, 1e-8)));




	// ofstream out("gfv.txt");

	// for(int i = 0; i < snake.mag.rows(); ++i)
	// {
	// 	for(int j = 0; j < snake.mag.cols(); ++j)
	// 	{
	// 		if(j) out << " ";

	// 		out << snake.mag(i, j);
	// 	}

	// 	out << "\n";
	// }



	cv::imshow("w", rgb);
	//cv::imshow("t", gvf);
	cv::waitKey(0);



	return 0;
}