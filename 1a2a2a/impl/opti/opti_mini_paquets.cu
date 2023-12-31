#include "opti.cuh"

void optimisation_mini_packet(
	Mdl_t * mdl,
	uint t0, uint t1, uint _T_mini_paquet,
	float * alpha, float div,
	uint methode, uint I,
	float * pourcent_masque)
{
	uint _t0 = t0 + (rand()%(t1-_T_mini_paquet-t0));
	//
	optimiser(
		mdl, _t0, _t0+_T_mini_paquet,
		alpha, div,
		RMSPROP, I,
		pourcent_masque);
};