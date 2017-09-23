#include "../Modelo.h"

#include "foreachmat.h"

#include <opencv2/opencv.hpp>


using VecU = Matrix<unsigned, Dynamic, 1>;
using MatU = Matrix<unsigned, Dynamic, Dynamic>;

template <typename T>
using MatD = Matrix<T, -1, -1>;

using Img = cv::Mat_<cv::Vec3b>;


ctx int imgR = 240, imgC = 320, imgS = imgR * imgC;
ctx int R = 128, C = 3, S = R * C, N = 16, M = S / N;

vi perm;



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

	void write (const vector<unsigned>& v)
	{
		for(auto x : v)
			write(x);
	}

	void write (const VecU& v)
	{
		for_each(v.data(), v.data() + v.rows(), [&](unsigned x){ this->write(x); });
	}


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
	WNN () {}

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




void mapImg (cv::Vec3b x, VecU& v)
{
	FOR(j, M)
	{
		unsigned val = 0;

		FOR(k, N)
		{
			int p = perm[j * N + k];
			int r = p / C, c = p % C;

			val = (val << 1) | (round(R * (x[c] / 255.0)) > p);
		}

		v(j) = val;
	}
}


VecU mapImg (cv::Vec3b x)
{
	VecU v(M);

	mapImg(x, v);

	return v;
}



void mapImg (const vector<cv::Vec3b>& v, MatU& X)
{
	FOR(i, v.size())
	{
		cv::Vec3b x = v[i];

		FOR(j, M)
		{
			unsigned val = 0;

			FOR(k, N)
			{
				int p = perm[j * N + k];
				int r = p / C, c = p % C;

				val = (val << 1) | (round(R * (x[c] / 255.0)) > p);
			}

			X(i, j) = val;
		}
	}
}


MatU mapImg (const vector<cv::Vec3b>& v)
{
	MatU X(v.size(), M);

	mapImg(v, X);

	return X;
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


Img readImg (const string& s, int R = -1, int C = -1)
{
	Img img = cv::imread(s.c_str());

	if(R != -1)
		cv::resize(img, img, cv::Size(imgC, imgR));


	return img;
}


vector<Img> readImg (const vector<string>& strImgs, int R = -1, int C = -1)
{
	vector<Img> imgs(strImgs.size());

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



void train (MatD<WNN>& wnns, const vector<Img>& imgs)
{
	vector<cv::Vec3b> v(imgs.size());

	MatU m(imgs.size(), M);

	FOR(i, imgR) FOR(j, imgC)
	{
		FOR(k, imgs.size()) v[k] = imgs[k](i, j);

		mapImg(v, m);

		wnns(i, j).train(m);
	}
}


cv::Mat_<uchar> predict (MatD<WNN>& wnns, const Img& img)
{
	cv::Mat_<uchar> res(imgR, imgC);


	// FOR(i, imgR) FOR(j, imgC)
	// {
	// 	mapImg(img(i, j), v);

	// 	res(i, j) = 255 * !wnns(i, j).predict(v);
	// }

	itmat::forEach(img, [&](int i, int j, auto&& x)
	{
		VecU v(M);

		mapImg(x, v);

		res(i, j) = 255 * !wnns(i, j).predict(v);

	}, 4);


	return res;
}





void run (MatD<WNN>& wnns, const string& path)
{
	cv::VideoCapture cap;

	cap.open(path.c_str());
	//cap.open(0);


	while(1)
    {
    	Img frame;

    	cap >> frame;

    	if(frame.empty()) break;

    	int orgR = frame.rows, orgC = frame.cols;


    	cv::resize(frame, frame, cv::Size(imgC, imgR));

    	cv::Mat_<uchar> img = predict(wnns, frame);

    	cv::resize(frame, frame, cv::Size(orgC, orgR));
    	cv::resize(img, img, cv::Size(orgC, orgR));

    	cv::imshow("w", img);
    	cv::imshow("t", frame);


    	if(cv::waitKey(10) == 27) break;
    }
}



vector<Img> getFromVideo (const string& path, int nm)
{
	vector<Img> v(nm);

	cv::VideoCapture cap;

	//cap.open(0);
	cap.open(path.c_str());


	FOR(i, nm)
	{
		cap >> v[i];

		cv::resize(v[i], v[i], cv::Size(imgC, imgR));
	}


	return v;
}


int main ()
{
	perm = genPerm(S);

	string videoStr = "car2.avi";


	vector<Img> imgs = getFromVideo(videoStr, 10);


	// vector<string> vPath(20);

	// FOR(i, vPath.size()) vPath[i] = string("imgs/img") + to_string(i+1) + string(".png");

	// vector<Img> imgs = readImg(vPath);


	MatD<WNN> wnns(imgR, imgC);

	FOR(i, imgR) FOR(j, imgC)
		wnns(i, j) = WNN(S, N);


	train(wnns, imgs);

	run(wnns, videoStr);



	return 0;
}