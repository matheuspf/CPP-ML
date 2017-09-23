#include "Bernoulli.h"


int main ()
{
	Veci x(10);

	x << 0, 0, 0, 0, 0, 1, 0, 1, -1, 0;

	Bernoulli dist;

	dist.fit(x);

	DB(dist.mean());


	return 0;
}