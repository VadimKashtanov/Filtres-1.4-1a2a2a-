#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

PAS_OPTIMISER()
Mdl_t * ouvrire_mdl(char * fichier) {
	FILE * fp = fopen(fichier, "rb");
	uint _C;
	FREAD(&_C, sizeof(uint), 1, fp);
	//
	uint * ST, * lignes, * activation;
	uint bloques, f_par_bloques;
	ST = (uint*)malloc(sizeof(uint) * C);
	FREAD(ST, sizeof(uint), C, fp);
	activation = (uint*)malloc(sizeof(uint) * C);
	FREAD(activation, sizeof(uint), C, fp);
	FREAD(&bloques, sizeof(uint), 1, fp);
	FREAD(&f_par_bloques, sizeof(uint), 1, fp);
	lignes = (uint*)malloc(sizeof(uint) * bloques);
	FREAD(lignes, sizeof(uint), bloques, fp);
	//
	Mdl_t * mdl = cree_mdl(
		ST,
		activation,
		bloques,
		f_par_bloques,
		lignes
	);
	//
	FREAD(mdl->f, sizeof(float), mdl->ST[0]*N, fp);
	//
	FOR(1, c, C) {
		FREAD(mdl->p[c], sizeof(float), (mdl->ST[c-1]+1)*mdl->ST[c], fp);
	}
	//
	mdl_diff_f(mdl);
	//
	mdl_cpu_vers_gpu(mdl);
	//
	fclose(fp);
	OK("Model chargé")
	//
	return mdl;
};

PAS_OPTIMISER()
void ecrire_mdl(Mdl_t * mdl, char * fichier) {
	FILE * fp = fopen(fichier, "wb");
	uint _C = C;
	FWRITE(&_C, sizeof(uint), 1, fp);
	FWRITE(mdl->ST, sizeof(uint), C, fp);
	FWRITE(mdl->activation, sizeof(uint), C, fp);
	FWRITE(&mdl->bloques, sizeof(uint), 1, fp);
	FWRITE(&mdl->f_par_bloque, sizeof(uint), 1, fp);
	FWRITE(mdl->lignes, sizeof(uint), mdl->bloques, fp);
	FWRITE(mdl->f, sizeof(float), mdl->ST[0]*N, fp);
	FOR(1, c, C) {
		FWRITE(mdl->p[c], sizeof(float), (mdl->ST[c-1]+1)*mdl->ST[c], fp);
	}
	fclose(fp);
};