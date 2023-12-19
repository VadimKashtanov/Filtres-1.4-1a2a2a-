#pragma once

#define ACTIV(s) tanh(s)
#define dACTIV(s,a) (1-a*a)

#include "marchee.cuh"

void intel_dot1d(
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd);

void d_intel_dot1d(
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
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_shared(	//	2 versions : 1 stricte et une non stricte
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_shared_2_16(	//	2 versions : 1 stricte et une non stricte
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d(
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	uint mode);

//	============================================

void d_nvidia_dot1d_naive(	//	2 versions : 1x stricte et 1x non stricte
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_dot1d_shared(	//	2 versions : 1 stricte et une non stricte
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_dot1d_shared_2_16(	//	2 versions : 1 stricte et une non stricte
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void  d_nvidia_dot1d(
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