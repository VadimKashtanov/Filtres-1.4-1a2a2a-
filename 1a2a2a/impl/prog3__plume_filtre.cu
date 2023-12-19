#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

/*
Utilisation :
	./prog0__plume_filtre mdl.bin bloque f_dans_bloque
*/

int main(int argc, char ** argv) {
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");  charger_tout();
	//
	if (argc == 4) {
		Mdl_t * mdl = ouvrire_mdl(argv[1]);
		//
		char cmd[1000];
		//
		uint depart = N*atoi(argv[2])*mdl->f_par_bloque + atoi(argv[3])*N;
		snprintf(cmd, 1000, "python3 -c \"import matplotlib.pyplot as plt;plt.plot([%f,%f,%f,%f,%f,%f,%f,%f]);plt.show()\"",
			mdl->f[depart + 0],
			mdl->f[depart + 1],
			mdl->f[depart + 2],
			mdl->f[depart + 3],
			mdl->f[depart + 4],
			mdl->f[depart + 5],
			mdl->f[depart + 6],
			mdl->f[depart + 7]
		);
		//
		SYSTEM(cmd);
		liberer_mdl(mdl);
		//
	} else {
		ERR("./prog0__plume_filtre mdl.bin bloque f_dans_bloque")
	}
	liberer_tout();
};