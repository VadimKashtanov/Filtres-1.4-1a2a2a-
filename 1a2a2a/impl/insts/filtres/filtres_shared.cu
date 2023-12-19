#include "filtres.cuh"

#define BLOQUE_T  4
#define BLOQUE_B  4
#define BLOQUE_FB N

static __device__ float filtre_device_shared(float * x, float * dif_x, float * f, float * dif_f, float * locd_s, float * locd_d) {
	float s = 0, d = 0;
	FOR(0, i, N-1) {
		s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
		d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
	};
	s += sqrtf(1 + fabs(x[N-1] - f[N-1]));

	s = s/8-1;
	d = d/7-1;

	float y = 2*expf(-s*s -d*d)-1;

	locd_s[0] = -4*y*s;
	locd_d[0] = -4*y*d;

	return y;
};

static __global__ void kerd_filtre_shared(	//	2 version : 1 stricte et une non stricte
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _b = threadIdx.y + blockIdx.y * blockDim.y;
	uint _f = threadIdx.z + blockIdx.z * blockDim.z;

	if (_t < T && _b < bloques && _f < f_par_bloque) {
		y[(depart+_t)*(bloques*f_par_bloque) + _b*f_par_bloque + _f] = filtre_device_shared(
			x + ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR,
			dif_x + ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR,
			f     + _b*f_par_bloque*N     + _f*N,
			dif_f + _b*f_par_bloque*(N-1) + _f*(N-1),
			locd_s+ (depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f,
			locd_d+ (depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f
		);
	}
};

/*
__global__ static
void mdt1_filtre__kerd(
	uint t0,
	uint * EMA_INT_BLOQUE_ligne__d,
	float * normalisee__d,
	float * f, float * dif_f,
	float * y, uint Y_MAX)
{
	//	<<<(T, BLOQUES), (BLOQUE_ST)>>> //BLOQUE_ST*N_FLT==256

	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _b = threadIdx.y + blockIdx.y * blockDim.y;
	uint _f = threadIdx.z + blockIdx.z * blockDim.z;

#define bloque blockIdx.y
#define f_dans_bloque threadIdx.x

	uint t = blockIdx.x;

	//	__partage__
	__shared__ uint ligne;
	if (threadIdx.x == 0)
		ligne = EMA_INT_BLOQUE_ligne__d[bloque];

	//	__partage__
	__shared__ float sh_x[N_FLTR];
	if (f_dans_bloque < N_FLTR)
		sh_x[f_dans_bloque] = normalisee__d[ligne*PRIXS*N_FLTR + (t0+t)*N_FLTR + f_dans_bloque];

	//	fonction de filtre
	float s = 0, d = 0;
	FOR(0, i, N_FLTR-1) {
		s += sqrtf(1 + fabs(     sh_x[i]        -   f[bloque*BLOQUE_ST*N_FLTR + f_dans_bloque*N_FLTR + i]  ));
		d += powf((1 + fabs((sh_x[i+1]-sh_x[i]) - dif_f[bloque*BLOQUE_ST*(N_FLTR-1) + f_dans_bloque*(N_FLTR-1) + i])), 2);
	};
	s += sqrtf(1 + fabs(sh_x[N_FLTR-1] - f[bloque*BLOQUE_ST*N_FLTR + f_dans_bloque*N_FLTR + N_FLTR-1]));

	//	s = a + b + c ...

	s = s/8-1;
	d = d/7-1;

	y[(depart+_t)*(bloques*f_par_bloque) + _b*f_par_bloque + _f] = 2*expf(-s*s -d*d)-1;
};*/

void nvidia_filtres_shared(	//	2 version : 1 stricte et une non stricte
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d)
{
	kerd_filtre_shared<<<dim3(KERD(T, BLOQUE_T), KERD(bloques, BLOQUE_B), KERD(f_par_bloque, BLOQUE_FB)), dim3(BLOQUE_T, BLOQUE_B, BLOQUE_FB)>>>(
		depart, T,
		bloques, f_par_bloque, ligne,
		x, dif_x,
		f, dif_f,
		y,
		locd_s, locd_d);
	ATTENDRE_CUDA();
};

//	=================================================

void d_nvidia_filtres_shared(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df)
{
	TODO()
}