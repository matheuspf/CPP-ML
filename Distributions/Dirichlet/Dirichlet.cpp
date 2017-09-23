#include "Dirichlet.h"
#include <gnuplot-iostream.h>


int main ()
{
	// int N = 5;

	// Vec aux(N);

	// iota(aux.data(), aux.data() + N, 2.0);

	// DB(aux.transpose() << "\n");


	// Dirichlet dist(aux);


	// int M = 1e6;

	// Mat X(M, N);


	// for(int i = 0; i < M; ++i)
	// 	X.row(i) = dist();


	// dist.fit(X);


	// DB(dist.alpha.transpose());




	Dirichlet dist(Vec::Constant(2, 10.0));


	vector<tuple<double, double, double>> plot;


	double pts = 1e5;

	for(double i = 1e-5; i < 1.0-1e-5; i += 1.0 / pts)
	{
		Vec x(2);

		x(0) = i, x(1) = 1.0 - i;


		plot.emplace_back(i, 1.0 - i, dist(x));
	}




	Gnuplot gp;

	gp << "set pm3d\n";
	gp << "set dgrid3d 30,30\n";
	gp << "splot '-' u 1:2:3 with lines\n";

	gp.send1d(plot);


	cin.get();



	return 0;
}