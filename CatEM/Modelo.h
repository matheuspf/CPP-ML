#ifndef MODELO_H
#define MODELO_H

#include <bits/stdc++.h>
#include <assert.h>
#define FORR(i, I, F) for(auto i = (I); i < (F); ++i)
#define RFORR(i, I, F) for(int i = (I); i >= (F); --i)
#define FOR(i, F) FORR(i, 0, F)
#define RFOR(i, F) RFORR(i, F, 0)
#define RG(v, I, F) begin(v) + (I), begin(v) + (F)
#define ALL(v) begin(v), end(v)
#define DB(...) cout << __VA_ARGS__ << '\n' << flush
#define DEBUG(...) cout << #__VA_ARGS__ << " = " << x << '\n'
#define pb push_back
#define F first
#define S second
#define ctx constexpr
#define fastio ios_base::sync_with_stdio(0); cin.tie(0);
#define unmap unordered_map
#define unset unordered_set
#define INF(T) numeric_limits<T>::max()
#define CIN(T) ({ T t; cin >> t; t; })


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//g++ main.cpp -std=c++14 -O3 -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs -lopencv_highgui

using namespace cv;

using namespace std;


using uchar = unsigned char;
using ll = long long;
using ull = unsigned long long;
using ld = long double;

using vi  = vector<int>;
using pi = pair<int, int>;


ctx double EPS = 1e-9;

ctx int cmpD (ld x, ld y, double e = EPS) { return (x <= y + e) ? (x + e < y) ? -1 : 0 : 1; }


ll power (ll x, ll b)
{
	ll r = 1;

	for(; b; b >>= 1, x *= x)
		if(b & 1) r*= x;

	return r;
}

struct Rand { static std::mt19937 generator; };

std::mt19937 Rand::generator(std::time(0));


struct RandInt: Rand
{
	inline int operator () (int min, int max) const { return std::uniform_int_distribution<>(min, max - 1)(generator); }
};

struct RandDouble : Rand
{
	inline double operator () (double min, double max) const { return std::uniform_real_distribution<>(min, max)(generator); } 
};



namespace std
{
	template <typename T, size_t N>
	struct hash<array<T, N>>
	{
		inline size_t operator () (const array<T, N>& v) const
		{
			return accumulate(v.begin(), v.end(), size_t(0), [](T x, T y){ return x + hash<T>()(y); });
		}
	};
}


template <typename T, size_t N>
decltype(auto) operator << (ostream& out, const array<T, N>& x)
{
	for(auto y : x) out << y << "  ";

	return out;
}


template <class Container, typename T>
inline bool contains (Container&& c, T&& t)
{
	return c.find(t) != c.end();
}


template <typename T>
inline auto bound (const T& x, const T& l, const T& u)
{
	return std::min(u, std::max(l, x));
}



#endif // MODELO_H