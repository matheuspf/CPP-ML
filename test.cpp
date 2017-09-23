#include "Modelo.h"
#include "Preprocessing/Preprocess.h"
#include "Cap8/LinearRegression/LinearRegression.h"
#include "Cap8/BayesianLR/BayesianLR.h"



int main ()
{
	Mat X = readMat("/home/matheus/Algoritmos/Vision/Data/plane.txt");

	Vec y = Vec(X.col(X.cols()-1));
	X.conservativeResize(Eigen::NoChange, X.cols()-1);

	X = polyExpansion(X, 10);


	auto [Xtrain, ytrain, Xtest, ytest] = trainTestSplit(X, y, 0.3, 0);


	Standardize stand;

	Xtrain = stand.fitTransform(Xtrain);
	Xtest = stand.transform(Xtest);



	//LinearRegression lr;
	BayesianLR lr(1e6);

	lr.learn(Xtrain, ytrain);
	

	//DB(lr.phi.transpose() << "\n\n\n");

	

	DB("Res: \n\n" << (lr(Xtrain)- ytrain).array().abs().sum() / ytrain.rows() << "       " << 
					  (lr(Xtest) - ytest).array().abs().sum() / ytest.rows());


	// double r = 0.0, s = 0.0;

	// FOR(i, Xtrain.rows()) if(r >= 1e-8)
	// 	r += log(lr.infer(Xtrain.row(i), ytrain(i)));

	// FOR(i, Xtest.rows()) if(s >= 1e-8)
	// 	s += log(lr.infer(Xtest.row(i), ytest(i)));


	// FOR(i, Xtrain.rows())
	// 	DB(lr.infer(Xtrain.row(i), ytrain(i)) << "     " << blr.infer(Xtrain.row(i), ytrain(i)));

	// FOR(i, Xtest.rows())
	// 	DB(lr.infer(Xtest.row(i), ytest(i)) << "     " << blr.infer(Xtest.row(i), ytest(i)));



	// DB(r << "     " << s);



	return 0;
}