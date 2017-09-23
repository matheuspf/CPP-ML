#include "Student.h"
#include <gnuplot-iostream.h>


int main ()
{
	// // Gaussian fit

	// Student dist(Vec::Constant(2, 0.0), 3*Mat::Identity(2, 2), 1e3);
	// //Gaussian dist(Vec::Constant(2, 0.0), 2*Mat::Identity(2, 2));


	// int it = 1e5;

	// Mat X(it, 2);

	// FOR(i, it)
	// 	X.row(i) = dist();


	// Gaussian gauss;

	// gauss.fit(X);


	// DB(gauss.mu.transpose() << "\n\n");
	// DB(gauss.sigma * gauss.sigma << "\n");





	// Generate and fit

	int N = 2;

	Vec mu = Vec::Constant(N, -1.0);

	Mat sigma = 3 * Mat::Identity(N, N);

	sigma(0, 1) = sigma(1, 0) = 0.5;

	Student dist(mu, sigma, 1.0);
	//Gaussian dist(Vec::Constant(2, 0.0), 2*Mat::Identity(2, 2));


	int it = 1e3;

	Mat X(it, N);

	FOR(i, it)
		X.row(i) = dist();


	Student stud;

	stud.fit(X);


	DB("\n\n" << stud.mu.transpose() << "\n\n");
	DB(stud.sigma * stud.sigma.transpose() << "\n");
	DB(stud.v << "\n");





	//// 2d generation

	// Student dist(Vec::Constant(1, 0.0), Mat::Constant(1, 1, 1.0), 1e5);

	// int it = 1e3;
	// double l = -5.0, u = 5.0, d = (u - l) / it;

	// vector<pd> v(it);

	// FOR(i, it)
	// {
	// 	Vec x(1); x(0) = l + i * d;

	// 	v[i] = pd(l + i * d, dist(x));
	// }



	// Gnuplot gp;

	// gp << "plot '-' with lines\n";

	// gp.send1d(v);
	// cin.get();






	// // 3d Plot

	// Student dist(Vec::Constant(2, 0.0), Mat::Identity(2, 2), 1.0);

	// int p = 500;
	// double l = -5, u = 5;
	// double d = (u - l) / double(p);

	// vector<tuple<double, double, double>> plot;


	// FOR(i, p) FOR(j, p)
	// {
	// 	Vec x(2); x << l + i * d, l + j * d;

	// 	plot.emplace_back(l + i * d, l + j * d, dist(x));

	// 	//out << l + i * d << " " << l + j * d << " " << dist(x) << "\n";
	// }


	// Gnuplot gp;

	// gp << "set pm3d\n";
	// gp << "set dgrid3d 30,30\n";
	// gp << "splot '-' u 1:2:3 with lines\n";

	// //gp << "plot '-' u 1:2:3 w image\n";

	// gp.send1d(plot);

	// cin.get();


	return 0;
}