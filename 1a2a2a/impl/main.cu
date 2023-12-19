#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

float pourcent_masque_nulle[C] = {0};

float * alpha_moyen__paquets16X500 = de_a(5e-4, 4e-5, C);
float * alpha_initialisation = de_a(1e-3, 1e-4, C);

float alpha_zeropoids[C] = {
	5e-3,
	0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0
};

float * alpha_initialisation__PRIXS = de_a(1e-3, 1e-4, C);

float alpha_zeropoids__PRIXS[C] = {
	3e-2,
	0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0
};


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

	uint ST[C] = {
		512,
		256, 256,
		128, 128,
		64, 64, 64, 64, 64, 64,
		32, 32,
		16, 16,
		8, 8,
		4, 4,
		1
	};
	uint bloques      = 64;
	uint f_par_bloque =  8;
	uint lignes[bloques] = {
	};
	Mdl_t * mdl = cree_mdl(ST, bloques, f_par_bloque, lignes);

	//Mdl_t * mdl = ouvrire_mdl("mdl.bin");

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
	uint OPTIMISATIONS = 10*100;
	printf("\033[3;92;m1 opti = %+f s, donc %i*%+f = %+f s = %+f mins\033[0m\n",
		sec_opti,
		OPTIMISATIONS, sec_opti,
		OPTIMISATIONS * sec_opti,
		OPTIMISATIONS * sec_opti / 60.0);
	//mdl_plume_grad(mdl, t0, t1);
	//
	FOR(0, rep, 100) {
		/*FOR(0, i, 3) {
			optimisation_mini_packet(
				mdl,
				t0, t1, 16*400,
				alpha_initialisation, 1.0,
				RMSPROP, 1000,
				//pourcent_masque_simple);
				pourcent_masque_nulle);
			printf("SCORE GENERALE : pred = %f\n", 100*mdl_pred(mdl, t0, t1, 3));
			mdl_gpu_vers_cpu(mdl);
			ecrire_mdl(mdl, "mdl.bin");
		}
		FOR(0, i, 2) {
			optimisation_mini_packet(
				mdl,
				t0, t1, 16*200,
				alpha_zeropoids, 1.0,
				RMSPROP, 200,
				//pourcent_masque_simple);
				pourcent_masque_nulle);
			printf("SCORE GENERALE : pred = %f\n", 100*mdl_pred(mdl, t0, t1, 3));
			mdl_gpu_vers_cpu(mdl);
			ecrire_mdl(mdl, "mdl.bin");
		}*/
		/*FOR(0, i, 10) {
			optimisation_mini_packet(
				mdl,
				t0, t1, 16*40,
				alpha_zeropoids, 1.0,
				RMSPROP, 30,
				//pourcent_masque_simple);
				pourcent_masque_nulle);
			printf("SCORE GENERALE : pred = %f\n", 100*mdl_pred(mdl, t0, t1, 3));
			mdl_gpu_vers_cpu(mdl);
			ecrire_mdl(mdl, "mdl.bin");
		}
		*/
		//
		optimiser(
			mdl,
			t0, t1,
			alpha_initialisation__PRIXS, 1.0,
			SGD, 300,
			pourcent_masque_nulle);
		/*optimiser(
			mdl,
			t0, t1,
			alpha_zeropoids__PRIXS, 1.0,
			RMSPROP, 100,
			pourcent_masque_nulle);*/
		//
		mdl_gpu_vers_cpu(mdl);
		ecrire_mdl(mdl, "mdl.bin");
		printf("SCORE GENERALE : pred = %f\n", 100*mdl_pred(mdl, t0, t1, 3));
		printf("===================================================\n");
		printf("==================TERMINE %i/%i=======================\n", rep+1, 100);
		printf("===================================================\n");
	}
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);

	//	-- Fin --
	liberer_tout();
};