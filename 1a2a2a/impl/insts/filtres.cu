#include "filtres.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

//
//	//
//	//	//	2*y-1  ===> y tout court
//	//
//

static float filtre(float * x, float * dif_x, float * f, float * dif_f, float * locd_s, float * locd_d) {
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

void intel_filtres(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d)
{
//#pragma omp parallel
//#pragma omp for
	FOR(0, t, T) {
		FOR(0, b, bloques) {
			FOR(0, _f, f_par_bloque) {
				y[(depart+t)*bloques*f_par_bloque + b*f_par_bloque + _f] = filtre(
						x + ligne[b]*PRIXS*N_FLTR + (depart+t)*N_FLTR,
					dif_x + ligne[b]*PRIXS*N_FLTR + (depart+t)*N_FLTR,
					f     + b*f_par_bloque*N     + _f*N,
					dif_f + b*f_par_bloque*(N-1) + _f*(N-1),
					locd_s+ (depart+t)*bloques*f_par_bloque + b*f_par_bloque + _f,
					locd_d+ (depart+t)*bloques*f_par_bloque + b*f_par_bloque + _f
				);
			}
		}
	}
}

static void d_filtre(float * x, float * dif_x, float * f, float * dif_f, float * locd_s, float * locd_d, float * dy, float * df) {
	float ds = locd_s[0] * dy[0] / 8;
	float dd = locd_d[0] * dy[0] / 7;

	df[N-1] += ds * 1 / (2*sqrtf(1 + fabs(x[N-1] - f[N-1]))) * (-1) * signe(x[N-1] - f[N-1]);
	FOR(0, i, N-1)
	{
		//s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
		df[i] += ds * 1 / (2*sqrtf(1 + fabs(x[i] - f[i]))) * (-1) * signe(x[i] - f[i]);
		//d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
		df[i+1] += dd * 2 * (1 + fabs(dif_x[i] - dif_f[i])) * signe(dif_x[i] - dif_f[i]) * (-1);
		df[ i ] += dd * 2 * (1 + fabs(dif_x[i] - dif_f[i])) * signe(dif_x[i] - dif_f[i]) * (+1);
	}
};

void  d_intel_filtres(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df)
{
	FOR(0, t, T) {
		FOR(0, b, bloques) {
			FOR(0, _f, f_par_bloque) {
				d_filtre(
						x + ligne[b]*PRIXS*N_FLTR + (depart+t)*N_FLTR,
					dif_x + ligne[b]*PRIXS*N_FLTR + (depart+t)*N_FLTR,
					f     + b*f_par_bloque*N     + _f*N,
					dif_f + b*f_par_bloque*(N-1) + _f*(N-1),
					locd_s+ (depart+t)*bloques*f_par_bloque + b*f_par_bloque + _f,
					locd_d+ (depart+t)*bloques*f_par_bloque + b*f_par_bloque + _f,
					dy + (depart+t)*bloques*f_par_bloque + b*f_par_bloque + _f,
					df     + b*f_par_bloque*N     + _f*N
				);
			}
		}
	}
}

void nvidia_filtres(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	uint mode)
{
	if (mode == 0) {
		nvidia_filtres_naive(
			depart, T,
			bloques, f_par_bloque, ligne,
			x, dif_x,
			f, dif_f,
			y,
			locd_s, locd_d);
	} else if (mode == 1) {
		nvidia_filtres_shared(
			depart, T,
			bloques, f_par_bloque, ligne,
			x, dif_x,
			f, dif_f,
			y,
			locd_s, locd_d);
	} else {
		ERR("Pas de mode %i pour mes filtres", mode);
	}
};

void d_nvidia_filtres(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df,
	uint mode)
{
	if (mode == 0) {
		d_nvidia_filtres_naive(
			depart, T,
			bloques, f_par_bloque, ligne,
			x, dif_x,
			f, dif_f,
			y,
			locd_s, locd_d,
			dy,
			df);
	} else if (mode == 1) {
		d_nvidia_filtres_shared(
			depart, T,
			bloques, f_par_bloque, ligne,
			x, dif_x,
			f, dif_f,
			y,
			locd_s, locd_d,
			dy,
			df);
	} else {
		ERR("Pas de mode %i pour mes filtres", mode);
	}
};

static void verifier_nvidia_intel() {
	uint depart = DEPART;
	uint T = 2*16;
	uint bloques = 16;
	uint f_par_bloque = 16;

	uint ligne[bloques] = {1,7,6,8,4,3,9,4,5,6,7,4,5,1,2,3};
	float * f = lst_rnd(bloques*f_par_bloque*N, -1, 1);
	float * diff_f = alloc<float>(bloques*f_par_bloque*(N-1));
	FOR(0, b, bloques*f_par_bloque) {
		FOR(0, i, N-1) {
			diff_f[b*(N-1)+i] = f[b*N + i+1] - f[b*N + i];
		}
	}
	float * y = alloc<float>((depart+T)*f_par_bloque*bloques);

	uint * ligne__d = cpu_vers_gpu<uint>(ligne, bloques);
	float * f__d = cpu_vers_gpu<float>(f, bloques*f_par_bloque*N);
	float * diff_f__d = cpu_vers_gpu<float>(diff_f, bloques*f_par_bloque*(N-1));
	float * y__d = cudalloc<float>((depart+T)*f_par_bloque*bloques);

	float * locd_s = alloc<float>(bloques*f_par_bloque*(DEPART+T));
	float * locd_d = alloc<float>(bloques*f_par_bloque*(DEPART+T));

	float * locd_s__d = cudalloc<float>(bloques*f_par_bloque*(DEPART+T));
	float * locd_d__d = cudalloc<float>(bloques*f_par_bloque*(DEPART+T));

	intel_filtres(
		depart, T,
		bloques, f_par_bloque, ligne,
		normalisee, dif_normalisee,
		f, diff_f,
		y,
		locd_s, locd_d);

	FOR(0, mode, 2) {
		CONTROLE_CUDA(cudaMemset(y__d, 0, sizeof(float) * T*f_par_bloque*bloques));
		//
		nvidia_filtres(
			depart, T,
			bloques, f_par_bloque, ligne__d,
			normalisee__d, dif_normalisee__d,
			f__d, diff_f__d,
			y__d,
			locd_s__d, locd_d__d,
			mode
		);
		printf("########## Verif filtre MODE %i #######\n", mode);

		puts("y__d\n");
		comparer_lst(
			y+depart*f_par_bloque*bloques,
			gpu_vers_cpu<float>(y__d, (depart+T)*f_par_bloque*bloques) + depart*f_par_bloque*bloques,
			T*f_par_bloque*bloques,
			0.001
		);
	}
};

void verif_filtres() {
	MSG("Shared filtres pas encore fait. Juste un copier-coller de la version naive");
	verifier_nvidia_intel();
};