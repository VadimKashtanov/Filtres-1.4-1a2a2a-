#pragma once

#include "marchee.cuh"

#include "dot1d.cuh"
#include "filtres.cuh"
#include "S.cuh"

#define C 20

typedef struct {
	//	Structure
	uint * ST, * ST__d;

	//	  Filtres
	uint bloques, f_par_bloque;
	uint * lignes, * lignes__d;

	//	Espaces ram
	float *        f;
	float * f_locd_s;	//-4*s*__f
	float * f_locd_d;	//-4*d*__f
	float *       df;
	//
	float *    p[C];
	float *    y[C];
	float * locd[C];
	float *   dy[C];
	float *   dp[C];

	//	Espaces vram cuda
	float *        f__d;
	float *       df__d;
	float * f_locd_s__d;
	float * f_locd_d__d;
	//
	float *    p__d[C];
	float *    y__d[C];
	float * locd__d[C];
	float *   dy__d[C];
	float *   dp__d[C];

	//	Optimisation
	float * dif_f, * dif_f__d;
} Mdl_t;

//	Memoire ram & vram
Mdl_t * cree_mdl(uint * ST, uint bloques, uint f_par_bloque, uint * lignes);
void liberer_mdl(Mdl_t * mdl);
void mdl_verif(Mdl_t * mdl);
void mdl_diff_f(Mdl_t * mdl);
void mdl_zero_deriv_cpu(Mdl_t * mdl);
void mdl_zero_deriv_gpu(Mdl_t * mdl);
void mdl_gpu_vers_cpu(Mdl_t * mdl);
void mdl_cpu_vers_gpu(Mdl_t * mdl);

//
uint poids_couche(Mdl_t * mdl, uint c);
uint entrees_couche(Mdl_t * mdl, uint c);

//	I/O
Mdl_t * ouvrire_mdl(char * fichier);
void ecrire_mdl(Mdl_t * mdl, char * fichier);

//	Plume
void   plumer_mdl(Mdl_t * mdl);
void comportement(Mdl_t * mdl, uint t0, uint t1);
void comportement_c1(Mdl_t * mdl, uint t0, uint t1);
void cmp_dy_dp(Mdl_t * mdl, uint t0, uint t1);
//
void mdl_plume_grad(Mdl_t * mdl, uint t0, uint t1);
//
float mdl_moy_dp(Mdl_t * mdl, uint c);

//	F & F'
void  mdl_f(Mdl_t * mdl, uint t0, uint t1, uint mode);
void mdl_df(Mdl_t * mdl, uint t0, uint t1, uint mode);

//	Utilisation
float mdl_score(Mdl_t * mdl, uint t0, uint t1, uint mode);
float mdl_pred(Mdl_t * mdl, uint t0, uint t1, uint mode);
float mdl_gain(Mdl_t * mdl, uint t0, uint t1, uint mode);
void mdl_aller_retour(Mdl_t * mdl, uint t0, uint t1, uint mode);