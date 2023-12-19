#include "filtres.cuh"

#define BLOQUE_T  4//8
#define BLOQUE_B  4//8
#define BLOQUE_FB 8//16//32//8

#include "../../../impl_tmpl/tmpl_etc.cu"

static __device__ float filtre_device(float * x, float * dif_x, float * f, float * dif_f, float * locd_s, float * locd_d) {
	float s = 0, d = 0;
	FOR(0, i, N-1) {
		s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
		d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
	};
	s += sqrtf(1 + fabs(x[N-1] - f[N-1]));

	s = s/8-1;
	d = d/7-1;

	float y = expf(-s*s -d*d);

	locd_s[0] = -2*2*s*y;
	locd_d[0] = -2*2*d*y;

	return 2*y-1;
};

static __global__ void kerd_filtre_naive(	//	2 version : 1 stricte et une non stricte
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
		y[(depart+_t)*(bloques*f_par_bloque) + _b*f_par_bloque + _f] = filtre_device(
			x + ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR,
			dif_x + ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR,
			f     + _b*f_par_bloque*N     + _f*N,
			dif_f + _b*f_par_bloque*(N-1) + _f*(N-1),
			locd_s+ (depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f,
			locd_d+ (depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f
		);
	}
};

void nvidia_filtres_naive(	//	2 version : 1 stricte et une non stricte
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d)
{
	kerd_filtre_naive<<<dim3(KERD(T, BLOQUE_T), KERD(bloques, BLOQUE_B), KERD(f_par_bloque, BLOQUE_FB)), dim3(BLOQUE_T, BLOQUE_B, BLOQUE_FB)>>>(
		depart, T,
		bloques, f_par_bloque, ligne,
		x, dif_x,
		f, dif_f,
		y,
		locd_s, locd_d);
	ATTENDRE_CUDA();
}

//	======================================================================================================================================

/*static __device__ float d_filtre_device(float * x, float * dif_x, float * f, float * dif_f, float * locd_s, float * locd_d, float * dy, float * df) {
	float ds = locd_s[0] * dy[0] / 8;
	float dd = locd_d[0] * dy[0] / 7;

	atomicAdd(&df[N-1], ds * 1 / (2*sqrtf(1 + fabs(x[N-1] - f[N-1]))) * (-1) * cuda_signe(x[N-1] - f[N-1]));
	FOR(0, i, N-1)
	{
		//s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
		atomicAdd(&df[i], ds * 1 / (2*sqrtf(1 + fabs(x[i] - f[i]))) * (-1) * cuda_signe(x[i] - f[i]));
		//d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
		atomicAdd(&df[i+1], dd * 2 * (1 + fabs(dif_x[i] - dif_f[i])) * cuda_signe(dif_x[i] - dif_f[i]) * (-1));
		atomicAdd(&df[ i ], dd * 2 * (1 + fabs(dif_x[i] - dif_f[i])) * cuda_signe(dif_x[i] - dif_f[i]) * (+1));
	}
};

static __global__ void d_kerd_filtre_naive(	//	2 version : 1 stricte et une non stricte
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _b = threadIdx.y + blockIdx.y * blockDim.y;
	uint _f = threadIdx.z + blockIdx.z * blockDim.z;

	if (_t < T && _b < bloques && _f < f_par_bloque) {
		d_filtre_device(
			x + ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR,
			dif_x + ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR,
			f     + _b*f_par_bloque*N      + _f*N,
			dif_f + _b*f_par_bloque*(N-1) + _f*(N-1),
			locd_s+ (depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f,
			locd_d+ (depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f,
			dy + (depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f,
			df + _b*f_par_bloque*N + _f*N
		);
	}
};*/

static __global__ void d_kerd_filtre_naive(	//	2 version : 1 stricte et une non stricte
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df)
{
	uint thz = threadIdx.z;

	uint thy = threadIdx.y;	// les 8 qui vont atomicAdd les df en chaque point qui lui est associé

	__shared__ float __x[N];
	__shared__ float __diff_x[N-1];
	//__shared__ float __f[N][N];
	//__shared__ float __diff_f[N][N-1];

	FOR(0, ___t, BLOQUE_T) {
		FOR(0, _b, bloques) {
			uint _t = ___t + blockIdx.x * blockDim.x;
			//uint _b = ___b + blockIdx.y * blockDim.y;
			uint _f = threadIdx.z + blockIdx.z * blockDim.z;

			if (_t < T && _b < bloques && _f < f_par_bloque) {
				//
				if (thy == 0) {
					__x[thz] = x[ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR + thz];
					if (thz != N-1)
						__diff_x[thz] = dif_x[ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR + thz];
				}
				//
				//FOR(0, i, N-1) {
				//	__f[thz][i] = f[_b*f_par_bloque*N      + _f*N + thy];
				//if (thy != N-1)
				//	__diff_f[thz][i] = dif_f[_b*f_par_bloque*(N-1) + _f*(N-1) + thy];
				//}
				//__f[thz][N-1] = f[_b*f_par_bloque*N      + _f*N + N-1];
				//
				__syncthreads();
				float * __df = df + _b*f_par_bloque*N + _f*N;
				float * __f = f + _b*f_par_bloque*N + _f*N;
				float * __diff_f = dif_f + _b*f_par_bloque*(N-1) + _f*(N-1);
				//
				float _dy0 = dy[(depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f];
				float ds = locd_s[(depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f] * _dy0 / 8;
				float dd = locd_d[(depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f] * _dy0 / 7;

				/*atomicAdd(&__df[N-1], ds * 1 / (2*sqrtf(1 + fabs(__x[N-1] - __f[thz][N-1]))) * (-1) * cuda_signe(__x[N-1] - __f[thz][N-1]));
				FOR(0, i, N-1)
				{
					//s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
					atomicAdd(&__df[i], ds * 1 / (2*sqrtf(1 + fabs(__x[i] - __f[thz][i]))) * (-1) * cuda_signe(x[i] - f[i]));
					//d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
					atomicAdd(&__df[i+1], dd * 2 * (1 + fabs(__diff_x[i] - __diff_f[thz][i])) * cuda_signe(__diff_x[i] - __diff_f[thz][i]) * (-1));
					atomicAdd(&__df[ i ], dd * 2 * (1 + fabs(__diff_x[i] - __diff_f[thz][i])) * cuda_signe(__diff_x[i] - __diff_f[thz][i]) * (+1));
				}*/
				//atomicAdd(&__df[N-1], ds * 1 / (2*sqrtf(1 + fabs(__x[N-1] - __f[thz][N-1]))) * (-1) * cuda_signe(__x[N-1] - __f[thz][N-1]));
				atomicAdd(&__df[thy], ds * 1 / (2*sqrtf(1 + fabs(__x[thy] - __f[thy]))) * (-1) * cuda_signe(x[thy] - __f[thy]));
				if (thy != N-1) {
					atomicAdd(&__df[thy+1], dd * 2 * (1 + fabs(__diff_x[thy] - __diff_f[thy])) * cuda_signe(__diff_x[thy] - __diff_f[thy]) * (-1));
					atomicAdd(&__df[ thy ], dd * 2 * (1 + fabs(__diff_x[thy] - __diff_f[thy])) * cuda_signe(__diff_x[thy] - __diff_f[thy]) * (+1));
				}
				__syncthreads();
			}
		}
	}
};

void d_nvidia_filtres_naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df)
{
	d_kerd_filtre_naive<<<dim3(DIV(T, BLOQUE_T), 1, KERD(f_par_bloque, BLOQUE_FB)), dim3(1, N, BLOQUE_FB)>>>(
		depart, T,
		bloques, f_par_bloque, ligne,
		x, dif_x,
		f, dif_f,
		y,
		locd_s, locd_d,
		dy,
		df);
	ATTENDRE_CUDA();
}

/*void d_nvidia_filtres_naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df)
{
	d_kerd_filtre_naive<<<dim3(KERD(T, BLOQUE_T), KERD(bloques, BLOQUE_B), KERD(f_par_bloque, BLOQUE_FB)), dim3(BLOQUE_T, BLOQUE_B, BLOQUE_FB)>>>(
		depart, T,
		bloques, f_par_bloque, ligne,
		x, dif_x,
		f, dif_f,
		y,
		locd_s, locd_d,
		dy,
		df);
	ATTENDRE_CUDA();
}*/