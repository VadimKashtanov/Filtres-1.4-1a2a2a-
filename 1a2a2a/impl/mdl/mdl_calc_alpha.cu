#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float mdl_moy_dp(Mdl_t * mdl, uint c) {
	uint POIDS = (c==0 ? mdl->ST[0]*N : mdl->ST[c]*(mdl->ST[c-1]+1));
	//
	float * dp = gpu_vers_cpu<float>(
		(c==0 ? mdl->df : mdl->dp__d[c]),
		POIDS);
	float moy = 0;
	FOR(0, i, POIDS) {
		moy += fabs(dp[i]);
	};
	free(dp);
	return moy / (float)POIDS;
};