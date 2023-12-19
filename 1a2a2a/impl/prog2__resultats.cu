#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

/*
Utilisation :
	./prog2__resultats mdl.bin
*/

void __gain(Mdl_t * mdl, uint t0) {
	uint t1 = t0 + 14*24;	//336h % 16 == 0
	printf("gain t0=%i t1=%i [2 semaines] = %f$\n", t0, t1, mdl_gain(mdl, t0, t1, 3));
};

void __pred(Mdl_t * mdl, uint t0, uint t1) {
	printf("pred %i a %i = %f%%\n", t0, t1, 100*mdl_pred(mdl, t0, t1, 3));
}

int main(int argc, char ** argv) {
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");  charger_tout();
	//
	if (argc == 2) {
		Mdl_t * mdl = ouvrire_mdl(argv[1]);
		mdl_verif(mdl);
		//
		uint t0 = DEPART;
		uint t1 = ROND_MODULO(FIN, 16);
		//
		__pred(mdl, t0, t1);
		__pred(mdl, t0, t0+16*1000);
		__pred(mdl, t0+16*1000, t1-16*1000);
		__pred(mdl, t1-16*1000, t1);
		__pred(mdl, t1-16*100, t1);
		//
		titre("Gain en 2 semaines");
		__gain(mdl, t0);
		__gain(mdl, t0+5000);
		__gain(mdl, t0+10000);
		__gain(mdl, t1-2*365*24);
		__gain(mdl, t1-356*24);
		__gain(mdl, t1-3*30*24);
		__gain(mdl, t1-14*24);
		//
		liberer_mdl(mdl);
	} else {
		ERR("./prog2__resultats mdl.bin")
	}
	liberer_tout();
}