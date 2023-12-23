#pragma once

#define TANH 0
#define LOGISTIC 1

#define ACTIV(mode, s) (mode==0 ? tanh(s) : (1/(1+expf(-s))))
#define dACTIV(mode, s,a) (mode == 0 ? (1-a*a) : (a*(a-1)))

#include "marchee.cuh"

void intel_dot1d(
	uint activation,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd);

void d_intel_dot1d(
	uint activation,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

//	============================================

void nvidia_dot1d_naive(	//	2 versions : 1x stricte et 1x non stricte
	uint activation,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_shared(	//	2 versions : 1 stricte et une non stricte
	uint activation,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_shared_2_16(	//	2 versions : 1 stricte et une non stricte
	uint activation,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d(
	uint activation,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	uint mode);

//	============================================

void d_nvidia_dot1d_naive(	//	2 versions : 1x stricte et 1x non stricte
	uint activation,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_dot1d_shared(	//	2 versions : 1 stricte et une non stricte
	uint activation,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_dot1d_shared_2_16(	//	2 versions : 1 stricte et une non stricte
	uint activation,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void  d_nvidia_dot1d(
	uint activation,
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp,
	uint mode);

//	============================================

void verif_do1d();