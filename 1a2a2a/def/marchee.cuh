#pragma once

#define DEBBUG false

#include "etc.cuh"

#define PRIXS 54900	//53170 * NB_DIFF_EMA * 32-bits = 17014400 bits = 17014.4 Ko = 17 Mo
#define L 1  		//u += u*f*levier*(p[i+L]/p[i]-1)

#define N_FLTR  8
#define N N_FLTR
#define BLOQUE_ST 16	//cmb de filtres dans un bloque
//	8*32 = 256
//	256*6 = 1536 = max noyaux par multi-processeur

#define MAX_INTERVALLE 1000

#define DEPART (N_FLTR*MAX_INTERVALLE)
#if (DEBBUG == false)
	#define FIN (PRIXS-L-1)
#else
	#define FIN (DEPART+1)
#endif

typedef struct {
	uint  ligne;
	uint    ema;
	uint interv;
} ema_int;

#define EMA_INTS (11 + 3 + 4 + 2 + 2)
/*	     ema,   int    temps
//  --- intervalle = 1 ---
    0|     1      1      8h
    1|     3      1      8h
    2|     6      1      8h
    3|    12      1      8h
    4|    24      1      8h
//  --- intervalle = 4 ---
    5|     8      4     32h
    6|    12      4     32h
    7|    24      4     32h
    8|    40      4     32h
//  --- intervalle = 8 ---
    9|    20      8     2.5j+
   10|    40      8     2.5j+
   11|    80      8     2.5j+
//  --- intervalle = 20 ---
    9|    50     20      6j
   10|   100     20      6j
   11|   200     20      6j
//  --- intervalle = 100 ---
    9|   300    100      1m+
   10|   600    100      1m+
   11|  1000    100      1m+
// - des prixs a intervalles diverses -
   12|     2      2     16h
   13|     4      4      1j+
   14|     6      6      2j
   15|    10     10      3j
   16|    20     20      6j+
   17|    50     50     14j+
   18|   100    100      1m+
   19|   200    200      2m+
   20|   500    500      5m+
   21| 1'000  1'000      1a
// -------  de macd -------- (macd du prixs)
   22|    1       1      8h
   23|    2       2     16h
   24|    4       4      1j+
// --------- volume --------
   25|    1       1      8h
   25|    5       1      8h
   26|    5       5     16h
   26|   20       5     16h
   27|   20      20      6j+
   27|  100      20      6j+
   28|  100     100      1m+
   28|  300     100      1m+
// -------- rsiema1 ---------
   29|    1       1      8h
   30|    4       4     16h
// -------- rsiema5 ---------
   31|    1       1      8h
   32|    4       4     16h
*/
extern ema_int ema_ints[EMA_INTS];

/*	Note : dans `normalisee` et `dif_normalisee`
les intervalles sont deja calculee. Donc tout
ce qui est avant DEPART n'est pas initialisee.
*/

//	Sources
extern float   prixs[PRIXS];
extern float    macd[PRIXS];	//ema12 - ema26
extern float  volume[PRIXS];	//vol_btc*prix - vol_usdt
extern float rsiema1[PRIXS];
extern float rsiema5[PRIXS];

//	ema des sources
extern float            ema[EMA_INTS * PRIXS];
extern float     normalisee[EMA_INTS * PRIXS * N_FLTR];
extern float dif_normalisee[EMA_INTS * PRIXS * N_FLTR];

//	======================================

//	Sources en GPU
extern float *   prixs__d;	//	nVidia
extern float *    macd__d;	//	nVidia
extern float *  volume__d;	//	nVidia
extern float * rsiema1__d;	//	nVidia
extern float * rsiema5__d;	//	nVidia

//	gpu ema des sources
extern float *            ema__d;	//	nVidia
extern float *     normalisee__d;	//	nVidia
extern float * dif_normalisee__d;	//	nVidia

void      charger_les_prixs();
void calculer_ema_norm_diff();
void    charger_vram_nvidia();

void     liberer_cudamalloc();

//	----------------------------------------------	//
//	--------------- Environnement ----------------	//
//	----------------------------------------------	//

typedef struct  {
	//	Chance de muter
	uint f_QUANTITEE[2];	//	Quantitee de filtres a muter
	uint e_QUANTITEE;		//	Quantitee de ema_int a muter

	//	Taux d'heritage paternel
	float f_COEF[2];

	//	+ sng()*rand(PLUS[0], PLUS[1])
	uint   e_EMA_PLUS[2];
	uint e_INTER_PLUS[2];
} Env_t;

void charger_tout();
void liberer_tout();