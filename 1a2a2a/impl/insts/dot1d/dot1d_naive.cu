#include "dot1d.cuh"

#define BLOQUE_T 32
#define BLOQUE_Y 32

static __global__ void kerd(	//	2 version : 1 stricte et une non stricte
	uint ACTIVATION,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		float s = p[_y*(X+1) + (X+1-1)];
		FOR(0, i, X) s += x[(depart+_t)*X + i] * p[_y*(X+1) + i];
		float a = ACTIV(ACTIVATION, s);
		y[(depart+_t)*Y + _y] = a;
		locd[(depart+_t)*Y + _y] = dACTIV(ACTIVATION, s,a);
	}
};

void nvidia_dot1d_naive(	//	2 version : 1 stricte et une non stricte
	uint ACTIVATION,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd)
{
	kerd<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_T)>>>(
		ACTIVATION,
		X, Y,
		depart, T,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
}

//	============================= Derivation ==============================

static __global__ void kerd_deriv(	//	2 version : 1 stricte et une non stricte
	uint ACTIVATION,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		float _locd = locd[(depart+_t)*Y + _y] * dy[(depart+_t)*Y + _y];
		atomicAdd(&dp[_y*(X+1) + (X+1-1)], _locd);
		FOR(0, i, X) {
			atomicAdd(&dx[(depart+_t)*X + i], _locd * p[_y*(X+1) + i]);
			atomicAdd(&dp[_y*(X+1) + i], _locd * x[(depart+_t)*X + i]);
		}
	}
};

void d_nvidia_dot1d_naive(	//	2 versions : 1x stricte et 1x non stricte
	uint ACTIVATION,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	kerd_deriv<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_T)>>>(
		ACTIVATION,
		X, Y,
		depart, T,
		x, y,
		p,
		locd,
		dy,
		dx,
		dp);
	ATTENDRE_CUDA();
};