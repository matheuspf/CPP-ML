#include "../Modelo.h"

#include <opencv2/opencv.hpp>

//#include "../Distributions/NormalPost/NormalPost.h"
//#include "../Distributions/GaussianMixture/GaussianMixture.h"
#include "../Distributions/Gaussian/Gaussian.h"
#include "../Distributions/Student/Student.h"

using Img = cv::Mat_<cv::Vec3b>;


template <class T>
using MatrixD = Matrix<T, -1, -1>;


ctx int Rows = 240, Cols = 320;

ctx double Alpha = 3.0, Beta = 1.0;



void ScreenShot ()
{
	cv::VideoCapture cap;

	if(!cap.open(0)) return;


	int p = 0;

	while(1)
	{
		Img img;

		cap >> img;
		
		if(img.empty()) return;

		cv::imshow("w", img);

		auto key = cv::waitKey(10);

		if(key == 27) return;

		else if(key == ' ')
		{
			//cv::resize(img, img, cv::Size(), 0.5, 0.5);
			//cv::GaussianBlur(img, img, cv::Size(7, 7), 0.0, 0.0);
			cv::imwrite((string("imgs/img") + to_string(++p) + string(".png")).c_str(), img);
		}
	}
}



vector<Img> readImgs (int M = 10)
{
	vector<Img> imgs(M);

	FOR(i, M)
		imgs[i] = cv::imread((string("imgs/img") + to_string(i+1) + string(".png")).c_str());

	return imgs;
}



template <class MatDist>
void train (MatDist& matDist, const vector<Img>& imgs)
{
	assert(imgs.size() && Rows == imgs[0].rows && Cols == imgs[0].cols && "Wrong Size");


	int M = imgs.size();


	Mat X(M, 3);

	FOR(i, Rows) FOR(j, Cols)
	{
		FOR(k, M) FOR(l, 3)
			X(k, l) = imgs[k](i, j)[l] / 255.0;

		matDist(i, j).fit(X, 1.0, 10);
		// matDist(i, j).sigma *= 1.2;
		// matDist(i, j).update();
	}
}


template <class MatDist>
cv::Mat_<uchar> predict (MatDist& matDist, const Img& img, double prior = 0.5)
{
	//double uProb = (1.0 / (255*255*255)) * (1.0 - prior);
	double uProb = (1.0 - prior);

	cv::Mat_<uchar> res(Rows, Cols);

	Vec x(3);

	FOR(i, Rows) FOR(j, Cols)
	{	
		FOR(k, 3) x(k) = img(i, j)[k] / 255.0;

		double prob = matDist(i, j)(x);

		res(i, j) = prob * prior < uProb ? 255 : 0;
	}


	return res;
}



template <class MatDist>
void test (MatDist& matDist, const string& path, double prior = 0.5)
{
	cv::VideoCapture cap;

	cap.open(path.c_str());


	int p = 0;

	while(1)
	{
		Img frame;

		cap >> frame;
		
		if(frame.empty()) return;


		//cv::resize(frame, frame, cv::Size(Cols, Rows));
		//cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0.0, 0.0);


		cv::Mat_<uchar> img = predict(matDist, frame, prior);


		//cv::resize(img, img, cv::Size(), 2.0, 2.0);

		cv::imshow("t", frame);
		cv::imshow("w", img);


		if(cv::waitKey(10) == 27) return;
	}
}



vector<Img> getFromVideo (const string& path, int nm)
{
	vector<Img> v(nm);

	cv::VideoCapture cap;

	cap.open(path.c_str());


	FOR(i, nm)
	{
		cap >> v[i];
		//cv::GaussianBlur(v[i], v[i], cv::Size(5, 5), 0.0, 0.0);
	}


	return v;
}




int main ()
{
	//ScreenShot(); exit(0);


	string path = "car.avi";


	//MatrixD<vector<NormalPost>> matDist = MatrixD<vector<NormalPost>>::Constant(Rows, Cols,
	//									  vector<NormalPost>(3, NormalPost(Alpha, Beta, 0.0, 0.0)));

	//MatrixD<GaussianMixture> matDist(Rows, Cols);

	//MatrixD<Gaussian> matDist(Rows, Cols);

	MatrixD<Student> matDist(Rows, Cols);

	vector<Img> imgs = getFromVideo(path, 100);


	train(matDist, imgs);

	test(matDist, path, 0.5);



	return 0;
}