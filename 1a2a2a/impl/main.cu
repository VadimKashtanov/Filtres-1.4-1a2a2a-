#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

float pourcent_masque_nulle[C] = {
	0,
	0,
	0,0,
	0,0,0,
	0,0,0,0,
	0,0,0,
	0,0,0,0,
	0,0,0,0,
	0
};

float pourcent_masque[C] = {
	0.,
	0.20,
	0.10,0.10,
	0.10,0.10,0.10,
	0.10,0.10,0.10,0.10,
	0.10,0.10,0.10,
	0.0,0.0,0.0,0.0,
	0,0,0,0,
	0
};

float * alpha = de_a(1e-3, 1e-3, C);

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	//	-- Init --
	srand(0);
	cudaSetDevice(0);

	titre(" Charger tout ");  charger_tout();
	//titre("Verifier  FILTRES"); verif_filtres();
	//titre("Verifier  DOT1D"); 	 verif_do1d();
	//titre("Verifier  S"); 		 verifier_S();
	//titre("Verifier MDL");      verif_mdl_1e5();

	//===============
	titre("  Programme Generale  ");

	/*uint ST[C] = {
		512,
		256,
		128, 128,
		64, 64, 64,
		32, 32, 32, 32,
		16, 16, 16,
		8, 8, 8, 8,
		4, 4, 4, 4,
		1
	};
	uint activations[C] = {TANH};
	uint bloques      = 64;
	uint f_par_bloque =  8;
	uint lignes[bloques] = {
		0,0,0,0,0,0,0,0,0,0,0,0,
	    1,1,1,1,1,
	    2,2,2,2,2,
	    3,3,3,3,
	    4,4,4,4,4,
	    5,5,5,6,6,7,7,7,7,8,9,9,9,9,10,11,12,13,14,
	    18,19,20,21,21,21,22,23,23,24,25,
	    28,
	    31,31
	};
	Mdl_t * mdl = cree_mdl(ST, activations, bloques, f_par_bloque, lignes);*/

	Mdl_t * mdl = ouvrire_mdl("mdl.bin");

	plumer_mdl(mdl);

	//	================= Initialisation ==============
	uint t0 = DEPART;
	uint t1 = ROND_MODULO(FIN, 16);
	printf("t0=%i t1=%i FIN=%i (t1-t0=%i, %%32=%i)\n", t0, t1, FIN, t1-t0, (t1-t0)%16);
	//
	float ancien = 100*mdl_pred(mdl, t0, t1, 3);
	printf("%%%% pred = %f\n", ancien);
	//
	INIT_CHRONO(chrono)
	//
	DEPART_CHRONO(chrono)
	mdl_aller_retour(mdl, t0, t1, 3);
	float sec_opti = VALEUR_CHRONO(chrono);
	uint OPTIMISATIONS = 150*1500;
	printf("\033[3;92;m1 opti = %+f s, donc %i*%+f = %+f s = %+f mins\033[0m\n",
		sec_opti,
		OPTIMISATIONS, sec_opti,
		OPTIMISATIONS * sec_opti,
		OPTIMISATIONS * sec_opti / 60.0);
	//
	uint REP = 150;
	FOR(0, rep, REP) {
		FOR(0, i, 1) {
			optimisation_mini_packet(
				mdl,
				t0, t1, 16*200,
				alpha, 1.0,
				RMSPROP, 1500,
				pourcent_masque);
			printf("SCORE GENERALE : pred = %f\n", 100*mdl_pred(mdl, t0, t1, 3));
			mdl_gpu_vers_cpu(mdl);
			ecrire_mdl(mdl, "mdl.bin");
		}
		/*FOR(0, i, 5) {
			optimisation_mini_packet(
				mdl,
				t0, t1, 16*800,
				alpha, 1.0,
				RMSPROP, 1500,
				//pourcent_masque_simple);
				pourcent_masque);
			printf("SCORE GENERALE : pred = %f\n", 100*mdl_pred(mdl, t0, t1, 3));
			mdl_gpu_vers_cpu(mdl);
			ecrire_mdl(mdl, "mdl.bin");
		}*/
		//
		/*optimiser(
			mdl,
			t0, t1,
			alpha, 1.0,
			RMSPROP, 2000,
			pourcent_masque_nulle);*/
		//
		mdl_gpu_vers_cpu(mdl);
		ecrire_mdl(mdl, "mdl.bin");
		printf("SCORE GENERALE : pred = %f\n", 100*mdl_pred(mdl, t0, t1, 3));
		printf("===================================================\n");
		printf("==================TERMINE %i/%i=======================\n", rep+1, REP);
		printf("===================================================\n");
	}
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);

	//	-- Fin --
	liberer_tout();
};