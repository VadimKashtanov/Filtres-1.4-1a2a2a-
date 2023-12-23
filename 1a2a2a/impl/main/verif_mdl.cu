#include "main.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void verif_mdl_1e5() {
	/*ASSERT(C == 3);
	titre("Comparer MODEL 1e-5");
	//
	uint ST[C] = {
		128,
		16,
		1
	};
	uint activations[C] = {
		LOGISTIC,
		TANH,
		TANH
	};
	uint bloques      = 16;
	uint f_par_bloque = 8;
	uint lignes[bloques] = {
		0,0,0,0,0,0,0,
		1, 1,
		2, 2,
		3, 4,
		6,
		8,
		10
	};
	Mdl_t * mdl = cree_mdl(ST, activations, bloques, f_par_bloque, lignes);
	//
	uint plus_T = 16;
	mdl_aller_retour(mdl, DEPART, DEPART+plus_T, 3);
	//mdl_cpu_vers_gpu(mdl);
	float * _grad_mode3 = gpu_vers_cpu<float>(mdl->df__d, mdl->ST[0]*N);
	//float * _f_mode3 = gpu_vers_cpu<float>(mdl->y__d[0], mdl->ST[0]*(DEPART+plus_T));
	//
	float grad__f[mdl->ST[0] * N];
	float _f = mdl_score(mdl, DEPART, DEPART+plus_T, 0);
	titre("mdl1e-5::dfiltres vs 1e-5");
	float _1E5 = 1e-3;
	FOR(0, i, mdl->ST[0] * N)
	{
		mdl->f[i] += _1E5;
		grad__f[i] = (mdl_score(mdl, DEPART, DEPART+plus_T, 0) - _f)/_1E5;
		mdl->f[i] -= _1E5;
		//
		printf("%i| %f === %f\n", i, grad__f[i], _grad_mode3[i]);

		//	t=DEPART+0
		//printf("    (t=DEPART+0) %f = %f\n", mdl->y[0][mdl->ST[0]*DEPART+i], _f_mode3[mdl->ST[0]*DEPART+i]);
		//printf("    (t=DEPART+1) %f = %f\n", mdl->y[0][mdl->ST[0]*(DEPART+1)+i], _f_mode3[mdl->ST[0]*(DEPART+1)+i]);
		//printf("    (t=DEPART+2) %f = %f\n", mdl->y[0][mdl->ST[0]*(DEPART+2)+i], _f_mode3[mdl->ST[0]*(DEPART+2)+i]);

	};
	//
	free(_grad_mode3);
	//free(_f_mode3);
	titre("mdl1e-5::dpoids vs 1e-5");
	//
	float * _grad_mode3_dp[C];
	FOR(1, c, C) {
		uint POIDS = poids_couche(mdl, c);
		_grad_mode3_dp[c] = gpu_vers_cpu<float>(mdl->dp__d[c], POIDS);
	}
	//
	FOR(1, c, C) {
		printf("   c = %i\n", c);
		uint POIDS = poids_couche(mdl, c);
		float grad__p[POIDS];
		//
		FOR(0, i, POIDS) {
			mdl->p[c][i] += _1E5;
			grad__p[i] = (mdl_score(mdl, DEPART, DEPART+plus_T, 0) - _f)/_1E5;
			mdl->p[c][i] -= _1E5;
			//
			printf("%i| %f === %f\n", i, grad__p[i], _grad_mode3_dp[c][i]);
		}
	};

	FOR(1, c, C) {
		free(_grad_mode3_dp[c]);
	}

	//
	liberer_mdl(mdl);*/
};