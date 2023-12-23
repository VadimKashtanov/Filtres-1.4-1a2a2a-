#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

uint poids_couche(Mdl_t * mdl, uint c) {
	return mdl->ST[c] * (mdl->ST[c-1]+1);
}

uint entrees_couche(Mdl_t * mdl, uint c) {
	return mdl->ST[c-1];
}

PAS_OPTIMISER()
Mdl_t * cree_mdl(uint * ST, uint * activation, uint bloques, uint f_par_bloque, uint * lignes) {
	ASSERT(ST[C-1] == 1);
	
	Mdl_t * mdl = alloc<Mdl_t>(1);

	mdl->activation = copier<uint>(activation, C);

	mdl->ST = copier<uint>(ST, C);
	mdl->ST__d = cpu_vers_gpu<uint>(ST, C);

	mdl->bloques = bloques;
	mdl->f_par_bloque = f_par_bloque;

	mdl->lignes = copier<uint>(lignes, bloques);
	mdl->lignes__d = cpu_vers_gpu<uint>(lignes, bloques);

	ASSERT(ST[0] == bloques * f_par_bloque);

	//	Espace ram intel
	mdl->f = lst_rnd(ST[0] * N, -1, 1);
	mdl->f_locd_s = alloc<float>(ST[0]*PRIXS);
	mdl->f_locd_d = alloc<float>(ST[0]*PRIXS);
	mdl->dif_f = alloc<float>(ST[0]*(N-1));
	FOR(0, i, ST[0]) {
		//	Trouver min, max
		float min = mdl->f[i*N + 0];
		float max = mdl->f[i*N + 0];
		FOR(1, j, N) {
			if (mdl->f[i*N + j] < min) min = mdl->f[i*N + j];
			if (mdl->f[i*N + j] > max) max = mdl->f[i*N + j];
		}

		//	Normaliser (e-min)/(max-min)
		FOR(0, j, N) mdl->f[i*N + j] = (mdl->f[i*N + j]-min)/(max-min);

		FOR(0, j, N-1) {
			mdl->dif_f[i*(N-1) + j] = mdl->f[i*N + j+1]-mdl->f[i*N + j];
		}
	}

	mdl->f__d = cpu_vers_gpu<float>(mdl->f, ST[0] * N);
	mdl->f_locd_s__d = cudalloc<float>(ST[0]*PRIXS);
	mdl->f_locd_d__d = cudalloc<float>(ST[0]*PRIXS);
	mdl->dif_f__d = cpu_vers_gpu<float>(mdl->dif_f, ST[0] * (N-1));

	mdl->df = alloc<float>(ST[0] * N);
	mdl->df__d = cudalloc<float>(ST[0] * N);

	{	//c = 1
		uint c = 0;
		mdl->p[c] = alloc<float>(0);
		mdl->y[c] = alloc<float>(ST[c]*PRIXS);
		mdl->locd[c] = alloc<float>(0);
		mdl->dy[c] = alloc<float>(ST[c]*PRIXS);
		mdl->dp[c] = alloc<float>(0);
		//
		mdl->p__d[c] = alloc<float>(0);
		mdl->y__d[c] = cudalloc<float>(ST[c]*PRIXS);
		mdl->locd__d[c] = cudalloc<float>(0);
		mdl->dy__d[c] = cudalloc<float>(ST[c]*PRIXS);
		mdl->dp__d[c] = alloc<float>(0);
	}

	FOR(1, c, C) {
		float X = (float)ST[c-1];
		float A = -0.05 - 0.7*fabs(1-(X-mdl->ST[C-2])/(mdl->ST[0]-mdl->ST[C-2]));
		float B = +0.05 + 0.7*fabs(1-(X-mdl->ST[C-2])/(mdl->ST[0]-mdl->ST[C-2]));
		mdl->p[c] = lst_rnd((ST[c-1]+1)*ST[c], A, B);
		mdl->y[c] = alloc<float>(ST[c]*PRIXS);
		mdl->locd[c] = alloc<float>(ST[c]*PRIXS);
		mdl->dy[c] = alloc<float>(ST[c]*PRIXS);
		mdl->dp[c] = alloc<float>((ST[c-1]+1)*ST[c]);

		mdl->p__d[c] = cpu_vers_gpu(mdl->p[c], (ST[c-1]+1)*ST[c]);
		mdl->y__d[c] = cudalloc<float>(ST[c]*PRIXS);
		mdl->locd__d[c] = cudalloc<float>(ST[c]*PRIXS);
		mdl->dy__d[c] = cudalloc<float>(ST[c]*PRIXS);
		mdl->dp__d[c] = cudalloc<float>((ST[c-1]+1)*ST[c]);
	}

	mdl_diff_f(mdl);

	return mdl;
};

PAS_OPTIMISER()
void mdl_verif(Mdl_t * mdl) {
	float * r = gpu_vers_cpu<float>(mdl->dif_f__d, mdl->ST[0]*(N-1));
	FOR(0, i, (mdl->ST[0]*(N-1))) ASSERT(fabs(r[i]-mdl->dif_f[i]) < 0.01);
	free(r);
	//
	FOR(1, c, C) {
		float * r = gpu_vers_cpu<float>(mdl->p__d[c], (mdl->ST[c-1]+1)*mdl->ST[c]);
		FOR(0, i, (mdl->ST[c-1]+1)*mdl->ST[c]) ASSERT(fabs(r[i]-mdl->p[c][i]) < 0.01);
		free(r);
	}
};

PAS_OPTIMISER()
void mdl_diff_f(Mdl_t * mdl) {
	FOR(0, i, mdl->ST[0]) {
		FOR(0, j, N-1) {
			mdl->dif_f[i*(N-1) + j] = mdl->f[i*N + j+1]-mdl->f[i*N + j];
		}
	}
	CONTROLE_CUDA(cudaMemcpy(mdl->dif_f__d, mdl->dif_f, sizeof(float)*mdl->ST[0]*(N-1), cudaMemcpyHostToDevice));
};

PAS_OPTIMISER()
void mdl_gpu_vers_cpu(Mdl_t * mdl) {
	CONTROLE_CUDA(cudaMemcpy(mdl->f,     mdl->f__d,     sizeof(float)*mdl->ST[0]*N, cudaMemcpyDeviceToHost));
	CONTROLE_CUDA(cudaMemcpy(mdl->f_locd_s,     mdl->f_locd_s__d,     sizeof(float)*mdl->ST[0]*PRIXS, cudaMemcpyDeviceToHost));
	CONTROLE_CUDA(cudaMemcpy(mdl->f_locd_d,     mdl->f_locd_d__d,     sizeof(float)*mdl->ST[0]*PRIXS, cudaMemcpyDeviceToHost));
	CONTROLE_CUDA(cudaMemcpy(mdl->df,     mdl->df__d,     sizeof(float)*mdl->ST[0]*N, cudaMemcpyDeviceToHost));
	//
	mdl_diff_f(mdl);
	//
	CONTROLE_CUDA(cudaMemcpy(mdl->y[0],  mdl->y__d[0],  sizeof(float)*mdl->ST[0]*PRIXS, cudaMemcpyDeviceToHost));
	CONTROLE_CUDA(cudaMemcpy(mdl->dy[0], mdl->dy__d[0], sizeof(float)*mdl->ST[0]*PRIXS, cudaMemcpyDeviceToHost));
	FOR(1, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p[c],    mdl->p__d[c],    sizeof(float)*(mdl->ST[c-1]+1)*mdl->ST[c], cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->y[c],    mdl->y__d[c],    sizeof(float)*mdl->ST[c]*PRIXS, 		   cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->locd[c], mdl->locd__d[c], sizeof(float)*mdl->ST[c]*PRIXS,  		   cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->dy[c],   mdl->dy__d[c],   sizeof(float)*mdl->ST[c]*PRIXS,  		   cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->dp[c],   mdl->dp__d[c],   sizeof(float)*(mdl->ST[c-1]+1)*mdl->ST[c], cudaMemcpyDeviceToHost));
	}
}

PAS_OPTIMISER()
void mdl_cpu_vers_gpu(Mdl_t * mdl) {
	CONTROLE_CUDA(cudaMemcpy(mdl->f__d,     mdl->f,     sizeof(float)*mdl->ST[0]*N, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->dif_f__d,     mdl->dif_f,     sizeof(float)*mdl->ST[0]*(N-1), cudaMemcpyHostToDevice));
	//
	CONTROLE_CUDA(cudaMemcpy(mdl->f_locd_s__d,     mdl->f_locd_s,     sizeof(float)*mdl->ST[0]*PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->f_locd_d__d,     mdl->f_locd_d,     sizeof(float)*mdl->ST[0]*PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->df__d,     mdl->df,     sizeof(float)*mdl->ST[0]*N, cudaMemcpyHostToDevice));
	//
	CONTROLE_CUDA(cudaMemcpy(mdl->y__d[0],  mdl->y[0],  sizeof(float)*mdl->ST[0]*PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->dy__d[0], mdl->dy[0], sizeof(float)*mdl->ST[0]*PRIXS, cudaMemcpyHostToDevice));
	FOR(1, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p__d[c],    mdl->p[c],    sizeof(float)*(mdl->ST[c-1]+1)*mdl->ST[c], cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->y__d[c],    mdl->y[c],    sizeof(float)*mdl->ST[c]*PRIXS, 				   cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->locd__d[c], mdl->locd[c], sizeof(float)*mdl->ST[c]*PRIXS,  				   cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->dy__d[c],   mdl->dy[c],   sizeof(float)*mdl->ST[c]*PRIXS,  				   cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->dp__d[c],   mdl->dp[c],   sizeof(float)*(mdl->ST[c-1]+1)*mdl->ST[c], cudaMemcpyHostToDevice));
	}
};

PAS_OPTIMISER()
void liberer_mdl(Mdl_t * mdl) {
	CONTROLE_CUDA(cudaFree(mdl->ST__d));
	CONTROLE_CUDA(cudaFree(mdl->lignes__d));
	CONTROLE_CUDA(cudaFree(mdl->f__d));
	CONTROLE_CUDA(cudaFree(mdl->dif_f__d));
	//
	CONTROLE_CUDA(cudaFree(mdl->f_locd_s__d));
	CONTROLE_CUDA(cudaFree(mdl->f_locd_d__d));
	CONTROLE_CUDA(cudaFree(mdl->df__d));
	//
	free(mdl->lignes);
	free(mdl->ST);
	free(mdl->activation);
	free(mdl->f);
	free(mdl->dif_f);
	//
	free(mdl->f_locd_d);
	free(mdl->f_locd_s);
	free(mdl->df);
	//
	{
		uint c = 0;
		free(mdl->y[c]);
		free(mdl->dy[c]);

		CONTROLE_CUDA(cudaFree(mdl->y__d[c]));
		CONTROLE_CUDA(cudaFree(mdl->dy__d[c]));
	}

	FOR(1, c, C) {
		free(mdl->p[c]);
		free(mdl->y[c]);
		free(mdl->locd[c]);
		free(mdl->dy[c]);
		free(mdl->dp[c]);

		CONTROLE_CUDA(cudaFree(mdl->p__d[c]));
		CONTROLE_CUDA(cudaFree(mdl->y__d[c]));
		CONTROLE_CUDA(cudaFree(mdl->locd__d[c]));
		CONTROLE_CUDA(cudaFree(mdl->dy__d[c]));
		CONTROLE_CUDA(cudaFree(mdl->dp__d[c]));
	}
};

PAS_OPTIMISER()
void mdl_zero_deriv_cpu(Mdl_t * mdl) {
	memset(mdl->df, 0, sizeof(float)*mdl->ST[0]*N);
	memset(mdl->dy[0], 0, sizeof(float)*mdl->ST[0]*PRIXS);
	//
	FOR(1, c, C) {
		memset(mdl->dy[c], 0, sizeof(float)*mdl->ST[c]*PRIXS);
		memset(mdl->dp[c], 0, sizeof(float)*((mdl->ST[c-1]+1)*mdl->ST[c]));
	}
};

PAS_OPTIMISER()
void mdl_zero_deriv_gpu(Mdl_t * mdl) {
	CONTROLE_CUDA(cudaMemset(mdl->df__d, 0, sizeof(float)*mdl->ST[0]*N));
	//
	CONTROLE_CUDA(cudaMemset(mdl->dy__d[0], 0, sizeof(float)*mdl->ST[0]*PRIXS));
	//
	FOR(1, c, C) {
		CONTROLE_CUDA(cudaMemset(mdl->dy__d[c], 0, sizeof(float)*mdl->ST[c]*PRIXS));
		CONTROLE_CUDA(cudaMemset(mdl->dp__d[c], 0, sizeof(float)*((mdl->ST[c-1]+1)*mdl->ST[c])));
	}
};