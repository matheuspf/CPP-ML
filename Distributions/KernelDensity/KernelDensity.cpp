#include "KernelDensity.h"
#include <gnuplot-iostream.h>

#include "../Gaussian/Gaussian.h"


int main ()
{
	Gnuplot gp;


	vector<tuple<double, double, double>> plot;


	int N = 2;


	Vec v = Vec::Constant(N, 0.0);
	Mat A = Mat::Identity(N, N);

	A(0, 1) = A(1, 0) = 0.3;

	Gaussian gauss(v, A);

	
	Mat X = gauss(1e3);


	KernelDensity dist(0.5);

	dist.fit(X);








	int l = -4, u = 4.1, p = 500;
	double d = (u - l) / double(p);


	FOR(i, p) FOR(j, p)
	{
		Vec x(2); x << l + i * d, l + j * d;

		plot.emplace_back(l + i * d, l + j * d, dist(x));

		//out << l + i * d << " " << l + j * d << " " << dist(x) << "\n";
	}


	gp << "set pm3d\n";
	gp << "set dgrid3d 30,30\n";
	gp << "splot '-' u 1:2:3 with lines\n";

	//gp << "plot '-' u 1:2:3 w image\n";

	gp.send1d(plot);



	// ofstream out("KernelDensity.txt");

	// FOR(i, p)
	// {
	// 	FOR(j, p)
	// 	{
	// 		Vec x(2); x << l + i * d, l + j * d;

	// 		plot.emplace_back(l + i * d, l + j * d, dist(x));

	// 		//out << dist(x) << " ";

	// 		out << l + i * d << " " << l + j * d << " " << dist(x) << "\n";
	// 	}

	// 	out << "\n";
	// }




	// Mat B(2, 2); B << 0, 0, 0, 0;
	// Vec x(2); x << 0.0, 0.0;
	// int it = 1e6;

	// FOR(i, it)
	// {
	// 	Vec y = dist().transpose();

	// 	x += y;

	// 	B += y * y.transpose();
	// }

	// DB((1.0 / it) * x << "\n\n");

	// DB((1.0 / it) * B);





	// int it = 1e6;

	// Mat B(it, 2);


	// FOR(i, it)
	// {
	// 	B.row(i) = dist();
	// }


	// dist.fit(B);
	

	// auto [mu, sigma] = dist.params();


	// DB(mu << "\n\n" << sigma);


	cin.get();

	return 0;
}