#include "../Modelo.h"

#include "../Distributions/GaussianMixture/GaussianMixture.h"

#include <opencv2/opencv.hpp>

#include "/home/matheus/Algoritmos/ForEachMat/foreachmat.h"



Mat getData (const vector<string>& files)
{
	vector<cv::Vec3b> data;

	for(const auto& s : files)
	{
		cv::Mat_<cv::Vec3b> img(cv::imread(s.c_str()));

		FOR(i, img.rows) FOR(j, img.cols) data.pb(img(i, j));
	}


	Mat X(data.size(), 3);

	FOR(i, data.size())
		FOR(j, 3)
			//X(i, j) = data[i][j] / 255.0;
			X(i, j) = data[i][j];

	return X;
}



template <class C1>
void classify (C1& c1, cv::Mat_<cv::Vec3b>& img, int NP, double prior)
{
	double uProb = (1.0 / (255*255*255)) * (1.0 - prior);
	//double uProb = (1.0 - prior);


	itmat::forEach(img, [&](int i, int j, auto&& x)
	{
		Vec y(3);

		//FOR(k, 3) y(k) = x[k] / 255.0;
		FOR(k, 3) y(k) = x[k];


		int p = (i * img.cols + j) / NP;


		x = c1[p](y) * prior < uProb ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);

	}, 4);
}



int main ()
{
	vector<string> strSkin{"img.png"};

	Mat XSkin = getData(strSkin);



	int NC = 2, NP = XSkin.rows() / NC, KSkin = 2;

	//vector<GaussianMixture> gmSkin(NC);
	vector<Gaussian> gmSkin(NC);


	FOR(i, NC)
		gmSkin[i].fit(XSkin.block(i * NP, 0, min(NP, int(XSkin.rows()) - i * NP), 3));




	cv::VideoCapture cap;


	if(!cap.open(0))
        return 0;


    for(;;)
    {
          cv::Mat_<cv::Vec3b> frame;
          cap >> frame;

          if(frame.empty()) break;


          classify(gmSkin, frame, NP, 0.5);


          cv::imshow("w", frame);

          if(cv::waitKey(10) == 27) break;
    }




	return 0;
}