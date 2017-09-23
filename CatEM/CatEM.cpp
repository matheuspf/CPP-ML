#include "Modelo.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen33/Eigen/Dense>


using namespace cv;
using namespace Eigen;



void init (MatrixXd& alphas, VectorXd& lambdas)
{
	// double ls = 0.0;

	// FOR(i, lambdas.rows())
	// 	lambdas(i) = (rand() % lambdas.rows()) + 1, ls += lambdas(i);

	// lambdas /= ls;


	FOR(i, alphas.rows())
	{
		FOR(j, alphas.cols())
			alphas(i, j) = (rand() % 10) + 10;

		alphas.row(i) /= alphas.row(i).sum();
	}

	FOR(i, lambdas.rows())
		lambdas(i) = (rand() % 10) + 10;

	lambdas /= lambdas.sum();

	//alphas = MatrixXd::Constant(alphas.rows(), alphas.cols(), 1.0 / alphas.rows());
	lambdas = VectorXd::Constant(lambdas.rows(), 1.0 / lambdas.rows());


	// alphas = MatrixXd::Constant(alphas.rows(), alphas.cols(), 1.0);


	// FOR(i, 8) alphas(i, 0) += rand() % 3;
	// FORR(i, 8, 16) alphas(i, 1) += rand() % 3;
 

	// alphas.col(0) /= alphas.col(0).sum();
	// alphas.col(1) /= alphas.col(1).sum();
}



auto CatEM (const VectorXi& X, int Q, int K = 2, int maxIter = 10)
{
	Q = Q - 1;

	int M = X.rows();

	MatrixXd R(M, K);

	MatrixXd alphas(Q, K);
	VectorXd lambdas(K);

	init(alphas, lambdas);

	FOR(iter, maxIter)
	{
		DB(alphas(0, 0) << "      " << alphas(0, 1));
		//DB(alphas(alphas.rows()-2, 0) << "      " << alphas(alphas.rows()-2, 1));
		DB("");

		E_Step(X, R, alphas, lambdas);
		M_Step(X, R, alphas, lambdas);
	}


	VectorXi y(M);

	FOR(i, X.rows())
	{
		double best = -1.0;

		FOR(j, K) if(alphas(X(i), j) * lambdas(j) > best)
			best = alphas(X(i), j) * lambdas(j), y(i) = j; //, DB(i << "   " << j << "   " << alphas(X(i), j) * lambdas(j));

		//DB(lambdas(0) << "      " <<  lambdas(1));

		// DB(y(i));
		y(i) *= (256 / K);
	}

	DB(alphas << "\n\n");
	//DB(alphas.array().rowwise() * lambdas.transpose().array() << "\n\n");
	DB(lambdas);

	return y;
}


auto CatEM (Mat_<uchar> img, int D = 16)
{
	VectorXi X(img.rows * img.cols);

	transform(img.data, img.data + img.rows * img.cols, X.data(), [D](auto x){ return (x * D) / 256; });

	return CatEM(X, D);
}






int main ()
{
	srand(time(0));

	Mat_<uchar> img(imread("baboon.png", 0));

	auto res = CatEM(img);

	//FOR(i, 100) DB(res(i));
	//exit(0);


	FOR(i, img.rows) FOR(j, img.cols)
		img(i, j) = res(i * img.cols + j);


	imshow("w", img);
	waitKey(0);



	return 0;
}
