#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

PAS_OPTIMISER()
void __interne_optimiser(
	Mdl_t * mdl,
	uint t0, uint t1,
	float * alpha, float div,
	uint methode, uint I,
	uint ** masque)
{
	//	Cree les listes pour les `hist` si un opti en a besoin 
	Opti_classe_t opti_classe;
	if      (methode == 0) opti_classe.sgd = (uint)NULL;
	else if (methode == 1) opti_classe.rmsprop = cree_rmsprop(mdl);
	else ERR("Pas de methode %i d'optimisation", methode);
	
	//	Plumer grad pour mieux y voire
	mdl_plume_grad(mdl, t0, t1);
	
	/* ------- Optimisation ----------- */
	FOR(0, i, I) {
		mdl_aller_retour(mdl, t0, t1, 3);
		//
		if (methode == 0) opti_simple(mdl, alpha, div, masque);
		if (methode == 1) opti_rmsprop(mdl, opti_classe.rmsprop, alpha, div, masque);
		mdl_diff_f(mdl);
		//
		if (i % 10 == 0)
			printf("%3.i/%3.i| perf=%+f%% score=%+f\n",
				i, I,
				100*mdl_pred(mdl, t0, t1, 3),
				mdl_score(mdl, t0, t1, 3)
			);
	}

	//	Liberer
	if (methode == 0) opti_classe.sgd = 0;
	else if (methode == 1) liberer_rmsprop(opti_classe.rmsprop);
};

void optimiser(
	Mdl_t * mdl,
	uint t0, uint t1,
	float * alpha, float div,
	uint methode, uint I,
	float * pourcent_masque)
{
	Masque_t * masque = cree_masque(mdl, pourcent_masque);
	//
	__interne_optimiser(
		mdl,
		t0, t1,
		alpha, div,
		methode, I,
		masque->masque);
	//
	sortire_masque(mdl, masque);
};