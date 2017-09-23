#ifndef PI_SNAKE_H
#define PI_SNAKE_H

#include "../../Modelo.h"

#include <opencv2/opencv.hpp>


using Img = cv::Mat_<uchar>;
using cv::Point;
using cv::Point2d;



struct Snake
{
	Snake (int M = 1, double alpha_ = 1.0, double beta_ = 1.0, double gamma_ = 1.2, double maxCurvature = 0.25,  
		   double maxMag = 0.5, int autoAdaptN = 10, double minAdapt = 8, double maxAdapt = 50, 
		   int maxIter = 1e5, double mu = 0.2, int iterGVF = 1e4) :
		   M(M), alpha_(alpha_), beta_(beta_), gamma_(gamma_), maxCurvature(maxCurvature), 
		   maxMag(maxMag), autoAdaptN(autoAdaptN), minAdapt(minAdapt), maxAdapt(maxAdapt), 
		   maxIter(maxIter), mu(mu), iterGVF(iterGVF) {}


	vector<Point> operator () (const Img& img, int N_ = -1, Point c_ = Point(-1, -1),
							   double r_ = -1, int mm_ = -1)
	{
		//N = N_ == -1 ? max(30, (img.rows * img.cols) / 10000) : N;
		N = 30;

		center = c_.x == -1 ? Point(img.cols / 2, img.rows / 2) : c_;

		radius = r_ == -1 ? min(img.rows / 2, img.cols / 2) - M: r_;

		//minMoved = mm_ == -1 ? max(N / 10, 2) : mm_;
		minMoved = 1;


		genPoints();

		//calcMag(img);

		gvf(img);

		return optimize();
	}


	// void calcMag (const Img& img)
	// {
	// 	cv::Mat gradX, gradY;

	// 	cv::Sobel(img, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	// 	cv::Sobel(img, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

	// 	cv::convertScaleAbs(gradX, gradX);
	// 	cv::convertScaleAbs(gradY, gradY);

	// 	cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, mag);
	// }


	void genPoints ()
	{
		points.resize(N);

		double h = (2 * pi()) / N;

		for(int i = 0; i < N; ++i)
			points[i] = center + Point(llround(radius * cos(i * h)), llround(radius * sin(i * h)));
	}




	vector<Point> optimize ()
	{
		vector<double> slope;
		vector<double> distances;
		vector<double> curvatures;
		vector<double> magnitudes;


		alpha.resize(N, alpha_);
		beta.resize(N, beta_);
		gamma.resize(N, gamma_);
		slope.resize(N, 0.0);


		vector<int> enumerate;
		vector<Point> neighbors;

		int moved, iter = 0;


		do
		{
			// if(iter % autoAdaptN == 0)
			// {
			// 	autoAdapt();

			// 	alpha.resize(N, alpha_);
			// 	beta.resize(N, beta_);
			// 	gamma.resize(N, gamma_);
			// 	slope.resize(N, 0.0);
			// }

			moved = 0;


			for(int i = 0; i < N; ++i)
			{
				double maxDistance = -1e20, meanDistance = 0.0;
				double maxCurvature = -1e20;
				double minMag = 1e20, maxMag = -1e20;

				neighbors.clear();


				for(int j = points[i].x - M; j <= points[i].x + M; ++j)
				for(int k = points[i].y - M; k <= points[i].y + M; ++k)
					neighbors.push_back(Point(j, k));


				int P = neighbors.size();

				distances.resize(P);
				curvatures.resize(P);
				magnitudes.resize(P);
				enumerate.resize(P);

				iota(ALL(enumerate), 0);



				Point2d a((double)points[mod(i-1, N)].x, (double)points[mod(i-1, N)].y);
				Point2d b((double)points[mod(i+1, N)].x, (double)points[mod(i+1, N)].y);


				for(int j = 0; j < P; ++j)
				{
					Point2d p((double)neighbors[j].x, (double)neighbors[j].y);

					distances[j] = pow(cv::norm(p - a), 2);

					curvatures[j] = pow(cv::norm(a - 2 * p + b), 2);

					// DB(a << "     " << p << "     " << b);
					// DB(curvatures[j] << "\n");

					magnitudes[j] = neighbors[j].inside(cv::Rect(0, 0, mag.cols(), mag.rows())) ? 
									mag(neighbors[j].y, neighbors[j].x) : 1e3;


					meanDistance += distances[j];

					
					maxCurvature = max(maxCurvature, curvatures[j]);
					//minMag = min(minMag, magnitudes[j]);
					maxMag = max(maxMag, abs(magnitudes[j]));
				}

				meanDistance /= P;

				for(int j = 0; j < P; ++j)
				{
					distances[j] = pow(distances[j] - meanDistance, 2);
					maxDistance = max(maxDistance, distances[j]);
				}

				for(int j = 0; j < P; ++j)
				{
					distances[j] /= maxDistance;
					curvatures[j] /= max(maxCurvature, 1e-8);

					magnitudes[j] = magnitudes[j] / max(maxMag, 1e-8);

					//magnitudes[j] = maxMag == 0.0 ? 0.0 : 1.0 - (magnitudes[j] / maxMag);
					//magnitudes[j] = (maxMag - minMag) == 0.0 ? 0.0 : (minMag - magnitudes[j]) / (maxMag - minMag);
				}


				Point best;
				double bestVal = 1e20;


				for(int j = 0; j < P; ++j)
				{
					double val = alpha[i] * distances[j] + beta[i] * curvatures[j] + gamma[i] * magnitudes[j];
					
					if(val < bestVal)
						bestVal = val, best = neighbors[j];
				}


				if(best != points[i])
					moved++;

				points[i] = best;
			}


			// for(int i = 0; i < N; ++i)
			// {
			// 	cv::Point2d v = cv::Point2d(points[i].x - points[mod(i-1, N)].x, points[i].y - points[mod(i-1, N)].y);
			// 	cv::Point2d u = cv::Point2d(points[mod(i+1, N)].x - points[i].x, points[mod(i+1, N)].y - points[i].y);

			// 	slope[i] = pow(cv::norm((v / (v.dot(v))) - (u / (u.dot(u)))), 2);
			// }


			// for(int i = 0; i < N; ++i)
			// {
			// 	if(slope[i] > slope[mod(i-1, N)] && slope[i] > slope[mod(i+1, N)] &&
			// 	   slope[i] > maxCurvature && points[i].inside(cv::Rect(0, 0, mag.cols(), mag.rows())) && 
			// 	   mag(points[i].y, points[i].x) > maxMag)
			// 		beta[i] = 0.0;
			// }

		} while(moved > minMoved && ++iter < maxIter);


		return points;
	}


	void autoAdapt ()
	{
		for(int i = 0; i < N; ++i)
		{
			Point x = points[i], y = points[mod(i+1, N)];

			double d = cv::norm(y - x);

			if(d < minAdapt)
			{
				points.erase(points.begin() + i);

				N--;
			}

			else if(d > maxAdapt)
			{
				Point a = points[mod(i-1, N)], b = points[mod(i+2, N)];

				points.insert(points.begin() + i+1, interpolate(a, x, y, b));

				N++;
			}
		}
	}



	Point interpolate (Point a, Point x, Point y, Point b)
	{
		return Point(llround((0.125 * (a.x + b.x) + 2.875 * (x.x + y.x)) / 6.0),
					 llround((0.125 * (a.y + b.y) + 2.875 * (x.y + y.y)) / 6.0));
	}




	void gvf (const Img& img)
	{
		int R = img.rows, C = img.cols;

		Mat mat(R, C);

		mag = Mat(R, C);


		ifstream in("gvf.txt");

		for(int i = 0; i < R; ++i)
			for(int j = 0; j < C; ++j)
				in >> mag(i, j);

		return;



		for(int i = 0; i < R; ++i)
			for(int j = 0; j < C; ++j)
				mat(i, j) = img(i, j) / 255.0;


		Mat gx = Mat::Constant(R, C, 0.0), gy = Mat::Constant(R, C, 0.0);
		Mat lu = Mat::Constant(R, C, 0.0), lv = Mat::Constant(R, C, 0.0);
		Mat u = Mat::Constant(R, C, 0.0), v = Mat::Constant(R, C, 0.0);


		double r = mu * 4.0;


		for(int i = 1; i < R-1; ++i)
		{
			for(int j = 1; j < C-1; ++j)
			{
				u(i, j) = gx(i, j) = (mat(i+1, j) - mat(i-1, j)) / 2.0;
				v(i, j) = gy(i, j) = (mat(i, j+1) - mat(i, j-1)) / 2.0;
			}
		}


		for(int iter = 0; iter < iterGVF; ++iter)
		{
			for (int i = 1; i < R-1; i++)
			{
				for (int j = 1; j < C-1; j++)
				{
					lu(i, j) = (u(i-1, j) + u(i+1, j) + u(i, j-1) + u(i, j+1) - 4 * u(i, j)) / 4.0; 
					lv(i, j) = (v(i-1, j) + v(i+1, j) + v(i, j-1) + v(i, j+1) - 4 * v(i, j)) / 4.0;
				}
			}			


			//plagio(u, v, lu, lv);


			// for(int i = 0; i < R; ++i)
			// {
			// 	for(int j = 0; j < C; ++j)
			// 	{
			// 		double prod = pow(gx(i, j), 2) + pow(gy(i, j), 2);

			// 		u(i, j) += 4.0 * mu * lu(i, j) - prod * (u(i, j) - gx(i, j));
			// 		v(i, j) += 4.0 * mu * lv(i, j) - prod * (v(i, j) - gy(i, j));

			// 		mag(i, j) = -sqrt(pow(u(i, j), 2) + pow(v(i, j), 2));
			// 	}
			// }


			plagio(u, v, lu, lv);


			for(int i = 0; i < R; ++i)
			{
				for(int j = 0; j < C; ++j)
				{
					double b = pow(gx(i, j), 2) + pow(gy(i, j), 2);
					double c1 = b * gx(i, j);
					double c2 = b * gy(i, j);

					u(i, j) = (1.0 - b) * u(i, j) + r * lu(i, j) + c1;
					v(i, j) = (1.0 - b) * v(i, j) + r * lv(i, j) + c2;

					double fac = 2.0;

					mag(i, j) = -pow((pow(abs(u(i, j)), fac) + pow(abs(v(i, j)), fac)), 1.0 / fac);
				}
			}
		}


		normMag = Mat(R, C);

		double maxElem = -1e20;

		for(int i = 0; i < R; ++i)
			for(int j = 0; j < C; ++j)
				maxElem = max(maxElem, abs(mag(i, j)));

		for(int i = 0; i < R; ++i)
			for(int j = 0; j < C; ++j)
				normMag(i, j) = abs(mag(i, j) / maxElem);
	}



	void plagio (Mat& u, Mat& v, Mat& Lu, Mat& Lv)
	{
		int w = u.rows(), h = u.cols();


		for (int x=0 ; x < w ; x++)  {
			for (int y=0 ; y < h ; y++) {

				if(x > 0 && y > 0 && x < w-1 && y < h-1){
				}else{
					if(x==0 && y ==0){
					}
					else if(y == 0 && x < w-1){
						Lu(x, y) = (-5 * u(x, y+1) + 4 * u(x, y+2) - u(x, y+3) + 2 * u(x, y) + u(x+1, y) + u(x-1, y) - 2 * u(x, y)) / 4;
						Lv(x, y) = (-5 * v(x, y+1) + 4 * v(x, y+2) - v(x, y+3) + 2 * v(x, y) + v(x+1, y) + v(x-1, y) - 2 * v(x, y)) / 4;
					}
					
					else if(x == 0  && y < h-1){
						Lu(x, y) = (-5 * u(x+1, y) + 4 * u(x+2, y) - u(x+3, y) + 2 * u(x, y) + u(x, y+1) + u(x, y-1) - 2 * u(x, y)) / 4;
						Lv(x, y) = (-5 * v(x+1, y) + 4 * v(x+2, y) - v(x+3, y) + 2 * v(x, y) + v(x, y+1) + v(x, y-1) - 2 * v(x, y)) / 4;
					}
					
					else if(y == h-1 && x > 0 && x < w-1){
						Lu(x, y) = (-5 * u(x, y-1) + 4 * u(x, y-2) - u(x, y-3) + 2 * u(x, y) + u(x+1, y) + u(x-1, y) - 2 * u(x, y)) / 4;
						Lv(x, y) = (-5 * v(x, y-1) + 4 * v(x, y-2) - v(x, y-3) + 2 * v(x, y) + v(x+1, y) + v(x-1, y) - 2 * v(x, y)) / 4;
					}
					
					else if(x == w-1 && y > 0 && y < h-1){
						Lu(x, y) = (-5 * u(x-1, y) + 4 * u(x-2, y) - u(x-3, y) + 2 * u(x, y) + u(x, y+1) + u(x, y-1) - 2 * u(x, y)) / 4;
						Lv(x, y) = (-5 * v(x-1, y) + 4 * v(x-2, y) - v(x-3, y) + 2 * v(x, y) + v(x, y+1) + v(x, y-1) - 2 * v(x, y)) / 4;
					}
				}
			}
		}
		
		//ul
		int x = 0;
		int y = 0;
		Lu(x, y) = (-5 * u(x, y+1) + 4 * u(x, y+2) - u(x, y+3) + 2 * u(x, y) - 5 * u(x+1, y) + 4 * u(x+2, y) - u(x+3, y) + 2 * u(x, y)) / 4;
		Lv(x, y) = (-5 * v(x, y+1) + 4 * v(x, y+2) - v(x, y+3) + 2 * v(x, y) - 5 * v(x+1, y) + 4 * v(x+2, y) - v(x+3, y) + 2 * v(x, y)) / 4;
		
		//br
		x = w-1;
		y = h-1;
		Lu(x, y) = (-5 * u(x, y-1) + 4 * u(x, y-2) - u(x, y-3) + 2 * u(x, y) - 5 * u(x-1, y) + 4 * u(x-2, y) - u(x-3, y) + 2 * u(x, y)) / 4;
		Lv(x, y) = (-5 * v(x, y-1) + 4 * v(x, y-2) - v(x, y-3) + 2 * v(x, y) - 5 * v(x-1, y) + 4 * v(x-2, y) - v(x-3, y) + 2 * v(x, y)) / 4;
			
		//bl
		x = 0;
		y = h-1;
		Lu(x, y) = (-5 * u(x, y-1) + 4 * u(x, y-2) - u(x, y-3) + 2 * u(x, y) - 5 * u(x+1, y) + 4 * u(x+2, y) - u(x+3, y) + 2 * u(x, y)) / 4;
		Lv(x, y) = (-5 * v(x, y-1) + 4 * v(x, y-2) - v(x, y-3) + 2 * v(x, y) - 5 * v(x+1, y) + 4 * v(x+2, y) - v(x+3, y) + 2 * v(x, y)) / 4;
		
		//ur
		x = w-1;
		y = 0;
		Lu(x, y) = (-5 * u(x, y+1) + 4 * u(x, y+2) - u(x, y+3) + 2 * u(x, y) - 5 * u(x-1, y) + 4 * u(x-2, y) - u(x-3, y) + 2 * u(x, y)) / 4;
		Lv(x, y) = (-5 * v(x, y+1) + 4 * v(x, y+2) - v(x, y+3) + 2 * v(x, y) - 5 * v(x-1, y) + 4 * v(x-2, y) - v(x-3, y) + 2 * v(x, y)) / 4;
	}

















	//Img mag;
	Mat mag;
	Mat normMag;

	vector<Point> points;

	int N, M;
	Point center;
	double radius;

	int minMoved;


	double alpha_;
	double beta_;
	double gamma_;

	vector<double> alpha;
	vector<double> beta;
	vector<double> gamma;

	double maxCurvature;
	double maxMag;

	int autoAdaptN;
	double minAdapt;
	double maxAdapt;

	int maxIter;


	double mu;
	int iterGVF;
};




// for(int i = 0; i < R; ++i)
// {
// 	u(i, 0) = gx(i, 0) = gx(i, 1);
// 	v(i, 0) = gy(i, 0) = gy(i, 1);

// 	u(i, C-1) = gx(i, C-1) = gx(i, C-2);
// 	v(i, C-1) = gy(i, C-1) = gy(i, C-2);
// }

// for(int i = 0; i < R; ++i)
// {
// 	u(0, i) = gx(0, i) = gx(1, i);
// 	v(0, i) = gy(0, i) = gy(1, i);

// 	u(R-1, i) = gx(R-1, i) = gx(R-2, i);
// 	v(R-1, i) = gy(R-1, i) = gy(R-2, i);
// }



#endif // PI_SNAKE_H