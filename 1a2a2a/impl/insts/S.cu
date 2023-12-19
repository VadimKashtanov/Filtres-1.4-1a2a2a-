#include "S.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define EXACTE(assertion) do {			\
	if (assertion) printf("\033[92m");	\
	else printf("\033[91m");			\
} while(0);

PAS_OPTIMISER()
void verifier_S() {
	uint depart = DEPART;
	uint T = 7;
	float  y[depart+T];// = {0.7,0.1,-0.9,0.4};
	y[depart+0]=0.7;
	y[depart+1]=0.1;
	y[depart+2]=-0.9;
	y[depart+3]=+0.4;
	y[depart+4]=-0.4;
	y[depart+5]=-0.1;
	y[depart+6]=+0.5;
	float * y__d = cpu_vers_gpu(y, depart+T);
	float score = (
		SCORE(y[depart+0], prixs[depart+L+0], prixs[depart+0]) + \
		SCORE(y[depart+1], prixs[depart+L+1], prixs[depart+1]) + \
		SCORE(y[depart+2], prixs[depart+L+2], prixs[depart+2]) + \
		SCORE(y[depart+3], prixs[depart+L+3], prixs[depart+3]) + \
		SCORE(y[depart+4], prixs[depart+L+4], prixs[depart+4]) + \
		SCORE(y[depart+5], prixs[depart+L+5], prixs[depart+5]) + \
		SCORE(y[depart+6], prixs[depart+L+6], prixs[depart+6]));// / T;
	float pred = (
		(float)(signe(y[depart+0]) == signe(prixs[depart+L+0]/prixs[depart+0]-1)) + \
		(float)(signe(y[depart+1]) == signe(prixs[depart+L+1]/prixs[depart+1]-1)) + \
		(float)(signe(y[depart+2]) == signe(prixs[depart+L+2]/prixs[depart+2]-1)) + \
		(float)(signe(y[depart+3]) == signe(prixs[depart+L+3]/prixs[depart+3]-1)) + \
		(float)(signe(y[depart+4]) == signe(prixs[depart+L+4]/prixs[depart+4]-1)) + \
		(float)(signe(y[depart+5]) == signe(prixs[depart+L+5]/prixs[depart+5]-1)) + \
		(float)(signe(y[depart+6]) == signe(prixs[depart+L+6]/prixs[depart+6]-1))) / T;
	float dy[depart+T];
	dy[depart+0]=dSCORE(y[depart+0], prixs[depart+L+0], prixs[depart+0]);// / T;
	dy[depart+1]=dSCORE(y[depart+1], prixs[depart+L+1], prixs[depart+1]);// / T; 
	dy[depart+2]=dSCORE(y[depart+2], prixs[depart+L+2], prixs[depart+2]);// / T;
	dy[depart+3]=dSCORE(y[depart+3], prixs[depart+L+3], prixs[depart+3]);// / T;
	dy[depart+4]=dSCORE(y[depart+4], prixs[depart+L+4], prixs[depart+4]);// / T;
	dy[depart+5]=dSCORE(y[depart+5], prixs[depart+L+5], prixs[depart+5]);// / T;
	dy[depart+6]=dSCORE(y[depart+6], prixs[depart+L+6], prixs[depart+6]);// / T;

	titre("Verification SCORES");

	float profondeure = 0.001;

	printf("            manuel  | intel  | nvidia\n");
	float _intel_score = intel_score(y,depart,T);
	float _nvidia_score = nvidia_score(y__d,depart,T);
	EXACTE(fabs(score - _intel_score)<profondeure && fabs(score - _nvidia_score)<profondeure);
	printf("score      %+f %+f %+f\033[0m\n", score, _intel_score, _nvidia_score);
	float _intel_pred = intel_prediction(y,depart,T);
	float _nvidia_pred = nvidia_prediction(y__d,depart,T);
	EXACTE(fabs(pred - _intel_pred)<profondeure && fabs(pred - _nvidia_pred)<profondeure);
	printf("prediction %+f %+f %+f\033[0m\n", pred, _intel_pred, _nvidia_pred);

	printf("------------- dy -----------\n");
	float * _dy = alloc<float>(depart+T);
	float * _dy__d = cudalloc<float>(depart+T);
	d_intel_score(y, _dy, depart, T);
	d_nvidia_score(y__d, _dy__d, depart, T);
	float * _dy__d_cpu = gpu_vers_cpu<float>(_dy__d, depart+T);
	printf("      manuel |  intel   |  nvidia\n");
	FOR(0, i, T) {
		EXACTE(fabs(dy[depart+i]-_dy[depart+i])<profondeure && fabs(dy[depart+i]-_dy__d_cpu[depart+i])<profondeure);
		printf("%2.i| %+f %+f %+f\033[0m\n", i, dy[depart+i], _dy[depart+i], _dy__d_cpu[depart+i]);
	}

	cudafree<float>(_dy__d);
	cudafree<float>(y__d);
	free(_dy);
	free(_dy__d_cpu);
};