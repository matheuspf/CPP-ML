#include "MultinomialBayes.h"
#include <gnuplot-iostream.h>


int main ()
{
	int M = 2e1, N = 5;

	Veci x(M);

	FOR(i, 4) x(i) = 0;
	FORR(i, 4, 8) x(i) = 1;
	FORR(i, 8, 12) x(i) = 2;
	FORR(i, 12, 16) x(i) = 3;
	FORR(i, 16, 20) x(i) = 3;


	MultinomialBayes dist(Vec::Constant(N, 1.0));

	dist.fit(x, N);



	vector<int> v;

	FOR(i, 1e5) v.pb(dist());




	Gnuplot gp;


	gp << "min = 0\n";
	gp << "max = 5\n";
	gp << "n = 5\n";
	gp << "width =(max - min) / n\n";
	gp << "hist(x,width)=width*floor(x/width)+width/2.0\n";
	//gp << "set style fill solid 1.0\n";
	gp << "plot '-' u (hist($1,width)):(1.0) smooth freq w boxes lc rgb\"red\" notitle\n";
	gp.send1d(v);

	cin.get();


	return 0;
}