#pragma once

#include "marchee.cuh"

void  intel_filtres(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d);

void  d_intel_filtres(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df);

//	----------------------------

void nvidia_filtres_naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d);

void nvidia_filtres_shared(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d);

void nvidia_filtres(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	uint mode);

//	----------------------------

void d_nvidia_filtres_naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df);

void d_nvidia_filtres_shared(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df);

void d_nvidia_filtres(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	float * locd_s, float * locd_d,
	float * dy,
	float * df,
	uint mode);

//	----------------------------

void verif_filtres();