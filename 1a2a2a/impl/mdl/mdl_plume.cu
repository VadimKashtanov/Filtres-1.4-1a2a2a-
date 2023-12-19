#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void plumer_mdl(Mdl_t * mdl) {
	printf("\033[4m Plume mdl\033[0m\n");
	printf("Filtres : %i\n", mdl->ST[0]);
	printf("Lignes des bloques : ");
	FOR(0, i, mdl->bloques) printf("%i ", mdl->lignes[i]);
	printf("\n");
	uint POIDS = 0;
	FOR(1, c, C) {
		POIDS += poids_couche(mdl, c);
		printf("%2.i | Y=%i (P=%i)\n", c, mdl->ST[c], poids_couche(mdl, c));
	}
	printf("Quantitée poids = %i\n", POIDS);
	printf(" --- fin plume mdl ---\n");
};

void comportement(Mdl_t * mdl, uint t0, uint t1) {
	mdl_f(mdl, t0, t1, 3);
	mdl_gpu_vers_cpu(mdl);
	FOR(0, c, C) {
		printf("###=== c=%i ===###\n", c);
		FOR(0, i, mdl->ST[c]) {
			printf("%3.i| ", i);
			FOR(t0, t, t1) printf("%+f ", mdl->y[c][i+t*mdl->ST[c]]);
			printf("\n");
		}
	}
};

void comportement_c1(Mdl_t * mdl, uint t0, uint t1) {
	mdl_f(mdl, t0, t1, 3);
	mdl_gpu_vers_cpu(mdl);
	uint c = C-1;
	printf("###=== c=%i ===###\n", c);
	FOR(0, i, mdl->ST[c]) {
		printf("%3.i| ", i);
		FOR(t0, t, t1) printf("%+f ", mdl->y[c][i+t*mdl->ST[c]]);
		printf("\n");
	};
};

void cmp_dy_dp(Mdl_t * mdl, uint t0, uint t1) {
	printf(" ########## COMPARER DY #########\n");
	FOR(0, c, C) {
		float * m = gpu_vers_cpu<float>(mdl->dy__d[c], mdl->ST[c]*t1);
		printf("###=== c=%i ===###\n", c);
		FOR(0, i, mdl->ST[c]) {
			printf("%3.i| cpu : ", i);
			FOR(t0, t, t1) printf("%+f ", mdl->dy[c][i+t*mdl->ST[c]]);

			printf(" gpu : ");

			FOR(t0, t, t1) printf("%+f ", m[i+t*mdl->ST[c]]);
			printf("\n");
		}
		free(m);
	}

	printf(" ########## COMPARER DP #########\n");
	FOR(1, c, C) {
		float * m = gpu_vers_cpu<float>(mdl->dp__d[c], (mdl->ST[c-1]+1)*mdl->ST[c]);
		printf("###=== c=%i ===###\n", c);
		FOR(0, i, (mdl->ST[c-1]+1)*mdl->ST[c]) {
			printf("%3.i| cpu : ", i);
			printf("%+f ", mdl->dp[c][i]);

			printf(" gpu : ");

			printf("%+f ", m[i]);
			printf("\n");
		}
		free(m);
	}
};

void mdl_plume_grad(Mdl_t * mdl, uint t0, uint t1) {
	mdl_aller_retour(mdl, t0, t1, 3);
	//mdl_gpu_vers_cpu(mdl);
	//
	printf(" Couche 0 \033[93mX=%i Y=%i\033[0m", mdl->ST[0], mdl->ST[0]);
	uint FILTRES = mdl->ST[0]*N;
	float * _grad = gpu_vers_cpu<float>(mdl->df__d, FILTRES);
	float moyenne = 0;
	float min=_grad[0], max=_grad[0];
	FOR(0, i, FILTRES) {
		if (_grad[i] > max) max = _grad[i];
		if (_grad[i] < min) min = _grad[i];
		moyenne += fabs(_grad[i]);
	}
	printf("[Grad:Min=%+f;Max=%+f;Moyenne=(-/+)%f]\n",
		min, max, moyenne / FILTRES
	);
	free(_grad);
	//
	//
	FOR(1, c, C) {
		printf(" Couche %i \033[93mX=%i Y=%i\033[0m", c, mdl->ST[c-1], mdl->ST[c]);
		uint POIDS = (mdl->ST[c-1]+1)*mdl->ST[c];
		float * _grad = gpu_vers_cpu<float>(mdl->dp__d[c], POIDS);
		float * ____p = gpu_vers_cpu<float>(mdl->p__d[c], POIDS);
		//
		//	Grad
		//
		float moyenne = 0;
		float min=_grad[0], max=_grad[0];
		FOR(0, i, POIDS) {
			if (_grad[i] > max) max = _grad[i];
			if (_grad[i] < min) min = _grad[i];
			moyenne += fabs(_grad[i]);
		}
		printf("[Grad:Min=%+f;Max=%+f;Moyenne=(-/+)%f]",
			min, max, moyenne / POIDS
		);
		free(_grad);
		//
		//	P
		//
		moyenne = 0;
		min=____p[0], max=____p[0];
		FOR(0, i, POIDS) {
			if (____p[i] > max) max = ____p[i];
			if (____p[i] < min) min = ____p[i];
			moyenne += fabs(____p[i]);
		}
		printf("[P:Min=%+f;Max=%+f;Moyenne=(-/+)%f]\n",
			min, max, moyenne / POIDS
		);
		free(____p);
	}
};