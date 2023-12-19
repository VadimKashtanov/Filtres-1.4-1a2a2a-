#include "S.cuh"

#include "../../../impl_tmpl/tmpl_etc.cu"

float  intel_score(float * y, uint depart, uint T) {
	float score = 0;
//#pragma omp parallel
//#pragma omp for
	FOR(0, i, T) {
		score += SCORE(y[depart+i], prixs[depart+i+L], prixs[depart+i]);
	}
	return score / T*100;
};

float intel_prediction(float * y, uint depart, uint T) {
	float pourcent = 0;
//#pragma omp parallel
//#pragma omp for
	FOR(0, i, T) {
		if (signe(y[depart+i]) == signe(prixs[depart+i+L]/prixs[depart+i]-1))
			pourcent += 1.0;
	}
	return pourcent / (float)T;
};

void d_intel_score(float * y, float * dy, uint depart, uint T) {
//#pragma omp parallel
//#pragma omp for
	FOR(0, i, T) {
		dy[depart+i] = dSCORE(y[depart+i], prixs[depart+i+L], prixs[depart+i]) / T*100;
	}
};