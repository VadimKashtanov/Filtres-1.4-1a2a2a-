#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float mdl_score(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	mdl_f(mdl, t0, t1, mode);
	if (mode == 0) return  intel_score(mdl->y[C-1], t0, (t1-t0));
	else           return nvidia_score(mdl->y__d[C-1], t0, (t1-t0));
};

float mdl_pred(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	mdl_f(mdl, t0, t1, mode);
	if (mode == 0) return  intel_prediction(mdl->y[C-1], t0, (t1-t0));
	else           return nvidia_prediction(mdl->y__d[C-1], t0, (t1-t0));
};


float mdl_gain(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	mdl_f(mdl, t0, t1, mode);
	if (mode != 0)
		CONTROLE_CUDA(cudaMemcpy(mdl->y[C-1], mdl->y__d[C-1], sizeof(float)*PRIXS, cudaMemcpyDeviceToHost))
	//
	float u = 100;
	printf("assurance a 0.40\n");
	FOR(t0, t, t1) {
		if (fabs(mdl->y[C-1][t]) > 0.40) u += u*25*mdl->y[C-1][t]*(prixs[t+L]/prixs[t]-1);
		if (u < 0) u = 0;
	};
	return u - 100;
};

void mdl_aller_retour(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	mdl_f(mdl, t0, t1, mode);
	if (mode == 0) mdl_zero_deriv_cpu(mdl);
	else           mdl_zero_deriv_gpu(mdl);
	if (mode == 0) d_intel_score(mdl->y[C-1], mdl->dy[C-1], t0, (t1-t0));
	else           d_nvidia_score(mdl->y__d[C-1], mdl->dy__d[C-1], t0, (t1-t0));
	mdl_df(mdl, t0, t1, mode);
};