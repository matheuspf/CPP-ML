#include "../Modelo.h"

#include "../Distributions/GaussianMixture/GaussianMixture.h"

#include "../Distributions/Student/Student.h"

#include <opencv2/opencv.hpp>

#include "/home/matheus/Algoritmos/ForEachMat/foreachmat.h"





Mat getData (const vector<string>& files)
{
	vector<cv::Vec3b> data;

	for(const auto& s : files)
	{
		cv::Mat_<cv::Vec3b> img(cv::imread(s.c_str()));

		//cv::GaussianBlur(img, img, cv::Size(5, 5), 0, 0);

		//cvtColor(img, img, cv::COLOR_BGR2YCrCb);

		FOR(i, img.rows) FOR(j, img.cols) data.pb(img(i, j));

		//data.insert(data.end(), img.data, img.data + img.rows * img.cols);

	}


	Mat X(data.size(), 3);

	FOR(i, data.size())
		FOR(j, 3)
			//X(i, j) = data[i][j] / 255.0;
			X(i, j) = data[i][j];

	return X;
}



template <class C1, class C2>
void classify (C1& c1, C2& c2, cv::Mat_<cv::Vec3b>& img, double prior)
{
	double uProb = (1.0 / (255*255*255)) * (1.0 - prior);
	//double uProb = (1.0 - prior);


	itmat::forEach(img, [&](auto&& x)
	{
		Vec y(3);

		//FOR(k, 3) y(k) = x[k] / 255.0;
		FOR(k, 3) y(k) = x[k];

		x = c1(y) * prior > c2(y) * (1.0 - prior) ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);
		//x = c1(y) * prior > uProb ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);

	}, 8);
}



int main ()
{
	vector<string> strSkin{"img1.png", "img2.png", "img3.png"};
	vector<string> strBack{"back1.png", "back2.png", "back3.png", "back4.png"};


	Mat XSkin = getData(strSkin);
	Mat XBack = getData(strBack);



	int KSkin = 2, KBack = 3;

	//GaussianMixture gmSkin;
	Student gmSkin;
	GaussianMixture gmBack;
	//Student gmBack;

	//gmSkin.fit(XSkin, KSkin);
	gmSkin.fit(XSkin);
	gmBack.fit(XBack, KBack);


	//DB(gmSkin.mu.transpose());
	//DB(gmSkin.gaussians[0].mu.transpose() << "     " << gmSkin.gaussians[1].mu.transpose());



	//Gaussian ggmSkin;
	// Gaussian gmBack;

	//ggmSkin.fit(XSkin);
	// //gmBack.fit(XBack);



	double red = 0.5;


	cv::VideoCapture cap;


	if(!cap.open(0))
        return 0;


    for(;;)
    {
          cv::Mat_<cv::Vec3b> frame, frame2;
          cap >> frame;
          //frame2 = frame.clone();

          if(frame.empty()) break;
          //if(frame2.empty()) break;

          //cvtColor(frame, frame, cv::COLOR_BGR2YCrCb);

          //classify(gmSkin, gmBack, frame, double(XSkin.rows()) / (XSkin.rows() + XBack.rows()));
          cv::resize(frame, frame, cv::Size(), red, red);

          //cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0, 0);

          classify(gmSkin, gmBack, frame, 0.1);
          cv::resize(frame, frame, cv::Size(), 1.0 / red, 1.0 / red);
          //classify(ggmSkin, gmBack, frame2, 0.7);


          cv::imshow("w", frame);
          //cv::imshow("w2", frame2);
         
          if( cv::waitKey(10) == 27 ) break;
    }



	// cv::Mat_<cv::Vec3b> out = classify(gm);

	// imwrite("out.png", out);






	return 0;
}