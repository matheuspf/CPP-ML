#include "../Modelo.h"

#include <opencv2/opencv.hpp>


using VecU = Matrix<unsigned, Dynamic, 1>;
using MatU = Matrix<unsigned, Dynamic, Dynamic>;



int imgR = 640, imgC = 480, imgS = imgR * imgC;
int R = 5, C = 5, S = R * C, N = 3;



struct RAM
{
	RAM () {}

	template <class U>
	RAM (U&& u) : m(forward<U>(u)) {}

	RAM (unsigned x) { m.insert(x); }


	void write (unsigned x)
	{
		m.insert(x);
	}

	// void write (const vector<unsigned>& v)
	// {
	// 	for(auto x : v)
	// 		write(x);
	// }

	// void write (const VecU& v)
	// {
	// 	for_each(v.data(), v.data() + v.rows(), [&](auto x){ write(x); });
	// }


	auto read (unsigned x)
	{
		return m.find(x) == m.end() ? 0 : 1;
	}

	void clear ()
	{
		m.clear();
	}


	unset<unsigned> m;
};


struct WNN
{
	WNN (int M, int N) : M(M), N(N), R(ceil(M / N)), v(ceil(M / N)) {}



	void train (const MatU& data)
	{
		assert(data.cols() == R && "Wrong training data size");

		for(int i = 0; i < data.rows(); ++i)
			for(int j = 0; j < data.cols(); ++j)
				v[j].write(data(i, j));
	}


	bool predict (const VecU& x, double thresh = 0.5)
	{
		assert(x.rows() == R && "Wrong predict data size");

		double resp = 0.0;

		for(int i = 0; i < x.rows(); ++i)
			resp += v[i].read(x(i));

		return (resp / R) > thresh;
	}


	void clear ()
	{
		v.clear();
	}


	int M, N, R;

	vector<RAM> v;
};



template <class Img>
VecU mapImg (const Img& img, int N, const vi& perm)
{
	VecU r((img.rows * img.cols) / N);

	FOR(i, r.size())
	{
		unsigned x = 0;

		FOR(j, N)
		{
			int a = perm[i * N + j] / img.cols;
			int b = perm[i * N + j] % img.cols;

			x = (x << 1) | bool(img(a, b));
		}

		r(i) = x;
	}

	return r;
}


template <class Img>
MatU mapImg (const vector<Img>& img, int N, const vi& perm)
{
	MatU r(img.size(), (img[0].rows * img[0].cols) / N);

	FOR(i, img.size())
		r.row(i) = mapImg(img[i], N, perm);

	return r;
}





template <class Img>
void binaryze (Img&& img)
{
	// double mval = 0.0;

	// FOR(i, img.rows) FOR(j, img.cols) mval += img(i, j);

	// mval /= (img.rows * img.cols);

	// cv::threshold(img, img, mval, 255.0, CV_THRESH_BINARY);

	cv::adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 5, 2);
}


cv::Mat_<uchar> readImg (const string& s, int R = -1, int C = -1)
{
	cv::Mat_<uchar> img = cv::imread(s.c_str(), 0);

	if(R != -1)
		cv::resize(img, img, cv::Size(R, C));

	binaryze(img);

	return img;
}


vector<cv::Mat_<uchar>> readImg (const vector<string>& strImgs, int R, int C)
{
	vector<cv::Mat_<uchar>> imgs(strImgs.size());

	FOR(i, imgs.size())
		imgs[i] = readImg(strImgs[i], R, C);

	return imgs;
}




vi genPerm (int N, int rng = 0)
{
	vi v(N);

	iota(ALL(v), 0);


	mt19937 gen(rng);

	shuffle(ALL(v), gen);


	return v;
}



void run (WNN& wnn, const vi& perm)
{
	cv::VideoCapture cap;
	
	if(!cap.open(0))
        return;


    while(1)
    {
    	cv::Mat_<cv::Vec3b> frame;

    	cap >> frame;

    	if(frame.empty()) break;


    	cv::Mat_<uchar> testImg;

    	cv::cvtColor(frame, testImg, cv::COLOR_BGR2GRAY);

    	binaryze(testImg);


		vector<cv::Mat_<uchar>> testImgs(imgS / S);



		FOR(i, imgR / R) FOR(j, imgC / C)
		{
			//DB(i << "         " << j << "           " << i * R << "    " << j * C << "    " << R << "     " << C);

			testImgs[i * (imgC / C) + j] = testImg(cv::Rect(i * R, j * C, R, C)).clone();
		}


		auto testSet = mapImg(testImgs, N, perm);

		vi res(imgS / S);



		FOR(i, testSet.rows())
			res[i] = wnn.predict(testSet.row(i), 0.8);



		cv::Mat_<uchar> out(imgC, imgR, 255);


		//int k = 0, l = 0;

		FOR(i, res.size())
		{
			//DB(R * (i / (imgC / C)) << "      " << C * (i % (imgC / C)) << "        " << 255 * res[i]);
			//out(cv::Rect(R * (i / (imgC / C)), C * (i % (imgC / C)), R, C)) = cv::Mat_<uchar>(R, C, 255 * res[i]);]

			FOR(k, R) FOR(l, C)
				out(C * (i % (imgC / C)) + l, R * (i / (imgC / C)) + k) = 255 * res[i];
		}


		cv::imshow("w", out);
		//cv::imshow("w", testImg);

		if( cv::waitKey(10) == 27 ) break;
	}
}




int main ()
{
	vector<string> strImgs = {"img1.png", "img2.png", "img3.png", "img4.png", "img5.png", "img6.png", "img7.png"};


	auto perm = genPerm(S);


	auto data = mapImg(readImg(strImgs, R, C), N, perm);



	WNN wnn(S, N);

	wnn.train(data);

	run(wnn, perm);

	
	// cv::Mat_<uchar> testImg = cv::imread("img.png", 0);
	
	// binaryze(testImg);


	// vector<cv::Mat_<uchar>> testImgs(imgS / S);



	// FOR(i, imgR / R) FOR(j, imgC / C)
	// {
	// 	//DB(i << "         " << j << "           " << i * R << "    " << j * C << "    " << R << "     " << C);

	// 	testImgs[i * (imgC / C) + j] = testImg(cv::Rect(i * R, j * C, R, C)).clone();
	// }


	// auto testSet = mapImg(testImgs, N, perm);

	// vi res(imgS / S);


	
	// WNN wnn(S, N);

	// wnn.train(data);


	// FOR(i, testSet.rows())
	// 	res[i] = wnn.predict(testSet.row(i), 0.8);



	// cv::Mat_<uchar> out(imgR, imgC, 255);


	// //int k = 0, l = 0;

	// FOR(i, res.size())
	// {
	// 	//DB(R * (i / (imgC / C)) << "      " << C * (i % (imgC / C)) << "        " << 255 * res[i]);
	// 	//out(cv::Rect(R * (i / (imgC / C)), C * (i % (imgC / C)), R, C)) = cv::Mat_<uchar>(R, C, 255 * res[i]);]

	// 	FOR(k, R) FOR(l, C)
	// 		out(R * (i / (imgC / C)) + k, C * (i % (imgC / C)) + l) = 255 * res[i];
	// }

	// cv::imwrite("out.png", out);



	return 0;
}