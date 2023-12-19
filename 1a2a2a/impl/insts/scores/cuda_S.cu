#include "S.cuh"

#include "../../../impl_tmpl/tmpl_etc.cu"

//	===============================================================

/*static __global__ void kerd_nvidia_score_powf(
	float * y, uint depart, uint T,
	float * _PRIXS)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;

	if (_t < T) {
		y[depart+_t] = SCORE(
			y[depart+_t], _PRIXS[(depart+_t)+L], _PRIXS[(depart+_t)]
		);
	}
};*/

static __global__ void kerd_nvidia_score_somme(
	float * y, uint depart, uint T,
	float * score, float * _PRIXS)
{
	float s = 0;
	FOR(0, i, T) {
		s += cuda_SCORE(
			y[depart+i], _PRIXS[(depart+i)+L], _PRIXS[(depart+i)]
		);//y[depart+i];
	}
	*score = s / (float)T*100;
};

float nvidia_score(float * y, uint depart, uint T)
{
	float * score__d = cudalloc<float>(1);
	/*kerd_nvidia_score_powf<<<dim3(KERD(T,1024)), dim3(1024)>>>(
		y, depart, T,
		prixs__d
	);
	ATTENDRE_CUDA();*/
	kerd_nvidia_score_somme<<<1,1>>>(
		y, depart, T,
		score__d, prixs__d
	);
	ATTENDRE_CUDA();
	float _score;
	CONTROLE_CUDA(cudaMemcpy(&_score, score__d, sizeof(float)*1, cudaMemcpyDeviceToHost));
	CONTROLE_CUDA(cudaFree(score__d));
	return _score;
};

//	===============================================================

static __global__ void kerd_nvidia_prediction_somme(
	float * y, uint depart, uint T,
	float * pred, float * _PRIXS)
{
	/*
	float _p = 0;
	FOR(0, i, T) {
		if (cuda_signe((y[depart+i])) == cuda_signe((_PRIXS[depart+i+L]/_PRIXS[depart+i]-1)))
			_p += 1.0;
	}
	pred[0] = (_p / (float)T);*/
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thx < T) {
		float p1 = _PRIXS[depart+thx+L];
		float p0 = _PRIXS[depart+thx];
		atomicAdd(
			pred,
			1.0*(uint)(cuda_signe((y[depart+thx])) == cuda_signe((p1/p0-1)))
		);
	};
};

float nvidia_prediction(float * y, uint depart, uint T) {
	float * pred__d = cudalloc<float>(1);
	CONTROLE_CUDA(cudaMemset(pred__d, 0, 1*sizeof(float)));
	kerd_nvidia_prediction_somme<<<dim3(KERD(T,1024)),dim3(1024)>>>(
		y, depart, T,
		pred__d, prixs__d
	);
	ATTENDRE_CUDA();
	float _pred;
	CONTROLE_CUDA(cudaMemcpy(&_pred, pred__d, sizeof(float)*1, cudaMemcpyDeviceToHost));
	cudafree<float>(pred__d);
	return _pred / (float)T;
};

//	===============================================================

static __global__ void kerd_nvidia_score_dpowf(
	float * y, float * dy,
	uint depart, uint T,
	float * _PRIXS)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;

	if (_t < T) {
		dy[depart+_t] = cuda_dSCORE(
			y[depart+_t], _PRIXS[(depart+_t)+L], _PRIXS[(depart+_t)]
		) / (float)T*100;
	}
};

void d_nvidia_score(float * y, float * dy, uint depart, uint T) {
	kerd_nvidia_score_dpowf<<<dim3(KERD(T,1024)), dim3(1024)>>>(
		y, dy,
		depart, T,
		prixs__d
	);
	ATTENDRE_CUDA();
};