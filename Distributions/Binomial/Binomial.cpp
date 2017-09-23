#include "Binomial.h"
#include <gnuplot-iostream.h>



int main ()
{
	double mean = 0.25;
	int n = 10;


	Binomial dist(mean, n);


	ofstream out("Binomial.txt");


	vi v;

	FOR(i, n+1)
		v.insert(v.end(), ceil(10000*dist(i)), i);
		//FOR(j, ceil(10000*dist(i))) out << i << "\n";


	// FOR(i, 1e5)
	// 	out << dist() << "\n";


	Gnuplot gp;


	gp << "min = 0\n";
	gp << "max = 10\n";
	gp << "n = 10\n";
	gp << "width =(max - min) / n\n";
	gp << "hist(x,width)=width*floor(x/width)+width/2.0\n";
	//gp << "set style fill solid 1.0\n";
	gp << "plot '-' u (hist($1,width)):(1.0) smooth freq w boxes lc rgb\"red\" notitle\n";
	gp.send1d(v);

	cin.get();

	return 0;
}