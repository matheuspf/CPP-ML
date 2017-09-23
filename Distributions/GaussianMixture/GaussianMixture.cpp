#include "GaussianMixture.h"
#include <gnuplot-iostream.h>


int main ()
{
	// 3D Plot

	Vec alphas(3);

	alphas(0) = 0.3;
	alphas(1) = 0.3;
	alphas(2) = 0.4;

	GaussianMixture dist(alphas,
					     Gaussian(Vec::Constant(2, -2.2), Mat::Identity(2, 2)),
					     Gaussian(Vec::Constant(2,  2.2), Mat::Identity(2, 2)),
					     Gaussian(Vec::Constant(2,  0.0), Mat::Identity(2, 2)));



	// Generate and fit

	int pts = 1e5;

	Mat X(pts, 2);

	FOR(i, pts)
		X.row(i) = dist();


	dist.fit(X, alphas.rows());


	DB(dist.mu << "\n\n");
	DB(dist.gaussians[0].mu << "\n\n");
	DB(dist.gaussians[1].mu << "\n\n");
	DB(dist.gaussians[2].mu << "\n\n");




	int it1 = 1e3, it2 = 1e3;
	double l1 = -5.0, l2 = -5.0, u1 = 5.0, u2 = 5.0, d1 = (u1 - l1) / it1, d2 = (u2 - l2) / it2;

	vector<pair<double, pd>> v;

	for(double x = l1; x < u1; x += d1)
		for(double y = l2; y < u2; y += d2)
		{
			Vec z(2); z << x, y;

			v.emplace_back(x, pd(y, dist(z)));
		}


	Gnuplot gp;

	gp << "set pm3d\n";
	gp << "set dgrid3d 30,30\n";
	gp << "splot '-' u 1:2:3 with lines\n";

	gp.send1d(v);

	cin.get();






	
	// 2D plot

	// Vec alphas(2);

	// alphas(0) = 0.3;
	// alphas(1) = 0.7;


	// GaussianMixture dist(alphas,
	// 				     Gaussian(Vec::Constant(1, -2.0), Mat::Constant(1, 1, 1.0)),
	// 				     Gaussian(Vec::Constant(1,  2.0), Mat::Constant(1, 1, 1.0)));




	// int it = 1e4;
	// double l = -5.0, u = 5.0, d = (u - l) / it;

	// vector<pd> v;

	// for(double x = l; x < u; x += d)
	// 	v.pb(pd(x, dist(Vec::Constant(1, x))));


	// Gnuplot gp;

	// gp << "plot '-' with lines\n";

	// gp.send1d(v);

	// cin.get();





	return 0;
}