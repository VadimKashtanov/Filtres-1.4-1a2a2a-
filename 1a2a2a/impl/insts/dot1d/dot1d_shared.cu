#include "dot1d.cuh"

#define BLOQUE 16

static __global__ void kerd_stricte(	//	2 version : 1 stricte et une non stricte
	uint ACTIVATION,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd)
{
	// <KERD(T, BLOQUE), KERD(Y,BLOQUE)>
	// <         BLOQUE,         BLOQUE>

	__shared__ float __partage__x[BLOQUE][BLOQUE];
	__shared__ float __partage__p[BLOQUE][BLOQUE];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	uint _y = thx + blockIdx.x * blockDim.x;
	uint _t = thy + blockIdx.y * blockDim.y;

	float s = 0;

	FOR(0, d, X/BLOQUE) {
		__partage__x[thy][thx] = x[(depart+_t)*(X) + d*BLOQUE + thx];
		__partage__p[thy][thx] = p[_y*(X+1) + d*BLOQUE + thy];
		__syncthreads();

	//#pragma unroll
		FOR(0, i, BLOQUE) s += __partage__x[thy][i] * __partage__p[i][thx];
		__syncthreads();
	};

#define __partage__b __partage__x[0]

	if (thy == 0) __partage__b[thx] = p[_y*(X+1) + (X+1-1)];
	__syncthreads();

	float a = ACTIV(ACTIVATION, (s + __partage__b[thx]));
	   y[(depart+_t)*Y + _y] = a;
	locd[(depart+_t)*Y + _y] = dACTIV(ACTIVATION, s,a);
};

void nvidia_dot1d_shared(	//	2 version : 1 stricte et une non stricte
	uint ACTIVATION,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd)
{
	if (X%BLOQUE==0 && Y%BLOQUE==0 && T%BLOQUE==0) {
		kerd_stricte<<<dim3(KERD(Y, BLOQUE), KERD(T, BLOQUE)), dim3(BLOQUE, BLOQUE)>>>(
			ACTIVATION,
			X, Y,
			depart, T,
			x, y,
			p,
			locd);
		ATTENDRE_CUDA();
	} else {
		nvidia_dot1d_naive(
			ACTIVATION,
			X, Y,
			depart, T,
			x, y,
			p,
			locd);
	}
}

//	============================= Derivation ==============================

static __global__ void kerd_stricte_deriv(	//	2 version : 1 stricte et une non stricte
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
	// <KERD(T, BLOQUE), KERD(Y,BLOQUE)>
	// <         BLOQUE,         BLOQUE>

	__shared__ float __partage__x[BLOQUE][BLOQUE];
	__shared__ float __partage__p[BLOQUE][BLOQUE];

	__shared__ float __partage__dx[BLOQUE][BLOQUE];
	__shared__ float __partage__dp[BLOQUE][BLOQUE];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	uint _y = thx + blockIdx.x * blockDim.x;
	uint _t = thy + blockIdx.y * blockDim.y;

	float _locd = locd[(depart+_t)*Y + _y] * dy[_t*Y + _y];

	atomicAdd(&dp[_y*(X+1) + (X)], _locd);

	FOR(0, d, X/BLOQUE) {
		__partage__x[thy][thx] = x[(depart+_t)*(X) + d*BLOQUE + thx];
		__partage__p[thy][thx] = p[_y*(X+1) + d*BLOQUE + thy];
		__partage__dx[thy][thx] = 0;
		__partage__dp[thy][thx] = 0;
		__syncthreads();

	//#pragma unroll
		FOR(0, i, BLOQUE) {
			//s += __partage__x[thy][i] * __partage__p[i][thx];
			atomicAdd(&__partage__dx[thy][i], _locd * __partage__p[i][thx]);
			atomicAdd(&__partage__dp[i][thx], _locd * __partage__x[thy][i]);
		}
		__syncthreads();

		atomicAdd(&dx[(depart+_t)*(X) + d*BLOQUE + thx], __partage__dx[thy][thx]); 
		atomicAdd(&dp[_y*(X+1) + d*BLOQUE + thy], __partage__dp[thy][thx]);
	}
};

void d_nvidia_dot1d_shared(	//	2 version : 1 stricte et une non stricte
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
	if (X%BLOQUE==0 && Y%BLOQUE==0 && T%BLOQUE==0) {
		kerd_stricte_deriv<<<dim3(KERD(Y, BLOQUE), KERD(T, BLOQUE)), dim3(BLOQUE, BLOQUE)>>>(
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
	} else {
		d_nvidia_dot1d_naive(
			ACTIVATION,
			X, Y,
			depart, T,
			x, y,
			p,
			locd,
			dy,
			dx,
			dp);
	}
}