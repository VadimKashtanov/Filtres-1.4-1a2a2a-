#pragma once

#define DEBBUG false

#include "etc.cuh"

#define PRIXS 54901	//53170 * NB_DIFF_EMA * 32-bits = 17014400 bits = 17014.4 Ko = 17 Mo
#define L 1  		//u += u*f*levier*(p[i+L]/p[i]-1)

#define N_FLTR  8
#define N N_FLTR

#define MAX_INTERVALLE 1000

#define DEPART (N_FLTR*MAX_INTERVALLE)
#if (DEBBUG == false)
	#define FIN (PRIXS-L-1)
#else
	#define FIN (DEPART+1)
#endif

typedef struct {
	uint     ligne;
	uint       ema;
	uint    interv;
	float * source;
} ema_int;

#define EMA_INTS (39)

/*	Note : dans `normalisee` et `dif_normalisee`
les intervalles sont deja calculee. Donc tout
ce qui est avant DEPART n'est pas initialisee.
*/

//	Sources
extern float   prixs[PRIXS];	//  prixs.bin
extern float   macds[PRIXS];	//   macd.bin
extern float volumes[PRIXS];	// volume.bin

//	ema des sources
extern float            ema[EMA_INTS * PRIXS];
extern float     normalisee[EMA_INTS * PRIXS * N_FLTR];
extern float dif_normalisee[EMA_INTS * PRIXS * N_FLTR];

//	======================================

//	Sources en GPU
extern float *   prixs__d;	//	nVidia
extern float *   macds__d;	//	nVidia
extern float * volumes__d;	//	nVidia

//	gpu ema des sources
extern float *            ema__d;	//	nVidia
extern float *     normalisee__d;	//	nVidia
extern float * dif_normalisee__d;	//	nVidia

void      charger_les_prixs();
void calculer_ema_norm_diff();
void    charger_vram_nvidia();

void     liberer_cudamalloc();

static ema_int ema_ints[EMA_INTS] = {
//	 id   ema  interv  source
	{ 0,    1,    1,   prixs},
	{ 1,    3,    1,   prixs},
	{ 2,    6,    1,   prixs},
	{ 3,   12,    1,   prixs},
	{ 4,   24,    1,   prixs},
//  --- intervalle = 4 ---
	{ 5,    8,    4,   prixs},
	{ 6,   12,    4,   prixs},
	{ 7,   24,    4,   prixs},
	{ 8,   40,    4,   prixs},
//  --- intervalle = 8 ---
	{ 9,   20,    8,   prixs},
	{10,   40,    8,   prixs},
	{11,   80,    8,   prixs},
//  --- intervalle = 20 ---
	{12,   50,   20,   prixs},
	{13,  100,   20,   prixs},
	{14,  200,   20,   prixs},
//  --- intervalle = 100 ---
	{15,  300,  100,   prixs},
	{16,  600,  100,   prixs},
	{17, 1000,  100,   prixs},
// - intervalles diverses -
	{18,    2,    2,   prixs},
	{19,    4,    4,   prixs},
	{20,    6,    6,   prixs},
	{21,   10,   10,   prixs},
	{22,   20,   20,   prixs},
	{23,   50,   50,   prixs},
	{24,  100,  100,   prixs},
	{25,  200,  200,   prixs},
	{26,  500,  500,   prixs},
    {27, 1000, 1000,   prixs},
// -------  de macd --------
    {28,    1,    1,   macds},
    {29,    2,    2,   macds},
    {30,    4,    4,   macds},
// --------- volume --------
    {31,    1,    1,   volumes},
    {32,    5,    1,   volumes},
    {33,    5,    5,   volumes},
    {34,   20,    5,   volumes},
    {35,   20,   20,   volumes},
    {36,  100,   20,   volumes},
    {37,  100,  100,   volumes},
    {38,  300,  100,   volumes}
};

void charger_tout();
void liberer_tout();