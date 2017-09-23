#include "NormalGamma.h"
#include <gnuplot-iostream.h>



int main ()
{
	NormalGamma dist(1.0, 1.0, 1.0, 0.0);



	// Generate and fit

	// int it = 1e6;

	// Vec mu(it), sigma(it);


	// FOR(i, 1e6)
	// {
	// 	auto [m, s] = dist();

	// 	mu(i) = m;
	// 	sigma(i) = s;
	// }

	// dist.fit(mu, sigma);


	// auto [a, b, g, dd] = dist.params();


	// DB(a << "   " << b << "   " << g << "   " << dd);




	// Heat Map

	int l = -4, u = 4.1, p = 500;
	double d = (u - l) / double(p);


	vector<tuple<double, double, double>> plot;



	FOR(i, p) FOR(j, p / 2)
	{
		double mu = l + i * d;
		double sig = j * d + 1e-6;

		//DB(mu << "   " << sig << "          " << dist(mu, sig));

		plot.emplace_back(mu, sig, dist(mu, sig));
	}


	Gnuplot gp;


	//gp << "set cbrange [0:1.0]\n";
	gp << "plot '-' u 1:2:3 w image\n";

	gp.send1d(plot);


	cin.get();


	return 0;
}