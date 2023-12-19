#include "dot1d.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void intel_dot1d(
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd)
{
#pragma omp parallel
#pragma omp for
	FOR(0, t, T) {
		FOR(0, _y, Y) {
			float s = p[_y*(X+1)+(X+1-1)];
			FOR(0, k, X) {
				float __x = x[(depart+t)*X+k];
				float __p = p[_y*(X+1)+k];
				s += __x * __p;
			}
			float a = ACTIV(s);
			y[(depart+t)*Y+_y]    = a;
			locd[(depart+t)*Y+_y] = dACTIV(s, a);
		}
	}
}

void d_intel_dot1d(
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
//#pragma omp parallel
//#pragma omp for
/*	FOR(0, t, T) {
		FOR(0, _y, Y) {
			float _locd = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];
			dp[_y*(X+1)+(X+1-1)] += _locd;
			FOR(0, k, X) {
				//s += x[t*X+k] * p[y*(X+1)+k];
				dx[(depart+t)*X+k]      += _locd * p[_y*(X+1)+k];
				dp[_y*(X+1)+k] += _locd * x[(depart+t)*X+k];
			}
		}
	}*/

	//dx = (p @ ((y-_y)*dtanh(x@p)).T).T
#pragma omp parallel
#pragma omp for
	FOR(0, t, T) {
		FOR(0, _x, X) {
			//float _locd = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];
			float s = 0;
			FOR(0, k, Y) {
				float __x = p[k*(X+1)+_x];//x[(depart+t)*X+k];
				float __p = locd[(depart+t)*Y+k] * dy[(depart+t)*Y+k];//p[_y*(X+1)+k];
				s += __x * __p;
			}
			dx[(depart+t)*X+_x]   = s;
		}
	}

	//dp = x.T @ ((y-_y)*dtanh(x@p))
#pragma omp parallel
#pragma omp for
	FOR(0, _y, Y) {
		float dbiais = 0;
		FOR(0, _x, X) {
			float s = 0;
			FOR(0, t, T) {
				float __x = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];//x[(depart+t)*X+k];
				float __p = x[(depart+t)*X+_x];//p[_y*(X+1)+k];
				s += __x * __p;
				if (_x == 0) {	//	Biais
					dbiais += __x;
				}
			}
			dp[_y*(X+1)+_x] = s;
		}
		dp[_y*(X+1)+(X+1-1)] = dbiais;
	}
}

//	=========================================================

void nvidia_dot1d(
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	uint mode)
{
	if (mode == 0) {
		nvidia_dot1d_naive(	//	2 versions : 1x stricte et 1x non stricte
			X, Y,
			depart, T,
			x, y,
			p,
			locd);
	} else if (mode == 1) {
		nvidia_dot1d_shared(	//	2 versions : 1x stricte et 1x non stricte
			X, Y,
			depart, T,
			x, y,
			p,
			locd);
	} else if (mode == 2) {
		nvidia_dot1d_shared_2_16(	//	2 versions : 1x stricte et 1x non stricte
			X, Y,
			depart, T,
			x, y,
			p,
			locd);
	} else {
		ERR("Pas de mode %i pour cuda f(x)", mode);
	}
}

//	----------------------------

void d_nvidia_dot1d(
	uint X, uint Y,
	uint depart, uint T,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp,
	uint mode)
{
	if (mode == 0) {
		d_nvidia_dot1d_naive(	//	2 versions : 1x stricte et 1x non stricte
			X, Y,
			depart, T,
			x, y,
			p,
			locd,
			dy,
			dx,
			dp);
	} else if (mode == 1) {
		d_nvidia_dot1d_shared(	//	2 versions : 1x stricte et 1x non stricte
			X, Y,
			depart, T,
			x, y,
			p,
			locd,
			dy,
			dx,
			dp);
	} else if (mode == 2) {
		d_nvidia_dot1d_shared_2_16(	//	2 versions : 1x stricte et 1x non stricte
			X, Y,
			depart, T,
			x, y,
			p,
			locd,
			dy,
			dx,
			dp);
	} else {
		ERR("Pas de mode %i pour cuda f'(x)", mode);
	}
}

//	----------------------------

static void verif_intel_papier() {	//verifier f(x) avec les valeurs a la main
	uint depart = 0;
	uint X = 3, T = 2, Y = 4; 
	float x[] = {
		//	t = 0
		.1,.2,.3,
		//	t = 1
		.4,.5,.6
	};
	float p[] = {
		//	y = 0
		.1,.2,.3, .4,
		//	y = 1
		.5,.6,.7, .8,
		//	y = 2
		.9,.10,.11, .12,
		//	y = 3
		.13,.14,.15, .16
	};
	float s[] = {
		//	## t = 0 ##
		.1*.1 + .2*.2 + .3*.3 + .4,
		.1*.5 + .2*.6 + .3*.7 + .8,
		.1*.9 + .2*.10 + .3*.11 + .12,
		.1*.13 + .2*.14 + .3*.15 + .16,
		//	## t = 1 ##
		.4*.1 + .5*.2 + .6*.3 + .4,
		.4*.5 + .5*.6 + .6*.7 + .8,
		.4*.9 + .5*.10 + .6*.11 + .12,
		.4*.13 + .5*.14 + .6*.15 + .16
	};
	float y[] = {
		//	## t = 0 ##
		(float)ACTIV(s[0]),
		(float)ACTIV(s[1]),
		(float)ACTIV(s[2]),
		(float)ACTIV(s[3]),
		//	## t = 1 ##
		(float)ACTIV(s[4]),
		(float)ACTIV(s[5]),
		(float)ACTIV(s[6]),
		(float)ACTIV(s[7]),
	};
	float locd[] = {
		dACTIV(s[0], y[0]),
		dACTIV(s[1], y[1]),
		dACTIV(s[2], y[2]),
		dACTIV(s[3], y[3]),

		dACTIV(s[4], y[4]),
		dACTIV(s[5], y[5]),
		dACTIV(s[6], y[6]),
		dACTIV(s[7], y[7])
	};

	float _y[Y*T];
	float _l[Y*T];

	intel_dot1d(
		X, Y,
		depart, T,
		x, _y, p, _l);

	titre("intel dot1d papier");
	puts("paper y - intel y\n");
	comparer_lst(y, _y, Y*T, 0.001);
	puts("paper locd - intel locd\n");
	comparer_lst(locd, _l, Y*T, 0.001);
};

static void verif_intel_1e5() {	//verifier f(x) avec les valeurs a la main
	uint depart = 0;
	uint X = 4, T = 3, Y = 7;
	float * x = lst_rnd(X*T, -1, 1);
	float * y = alloc<float>(T*Y);
	float * p = lst_rnd((X+1)*Y, -1, 1);
	float * l = alloc<float>(T*Y);

	float * sorties = lst_rnd(Y*T, -1, 1);

	float * dp = zero<float>((X+1)*Y);
	float * dx = zero<float>(T*X);
	float * dy = zero<float>(T*Y);

	intel_dot1d(
		X, Y,
		depart, T,
		x, y, p, l);

	//	S
	float S = 0;
	FOR(0, i, Y*T) S += powf((y[i]-sorties[i]), 2) / 2;// / (Y*T);

	//	dS
	FOR(0, i, Y*T) dy[i] = (y[i]-sorties[i]);// / (Y*T);

	d_intel_dot1d(
		X, Y,
		depart, T,
		x, y, p, l,
		dy, dx, dp);

	float * _dp = zero<float>((X+1)*Y);
	float * _dx = zero<float>(T*X);

	float _1E5 = 1e-3;

	//	_dp
	float Sx;
	FOR(0, i, X*Y) {
		p[i] += _1E5;
		//
		intel_dot1d(
			X, Y,
			depart, T,
			x, y, p, l);
		p[i] -= _1E5;
		//
		Sx = 0;
		FOR(0, j, Y*T) Sx += powf((y[j]-sorties[j]), 2) / 2;// / (Y*T);
		//
		_dp[i] = (Sx-S)/_1E5;
	};

	//	_dx
	FOR(0, i, X*T) {
		x[i] += _1E5;
		//
		intel_dot1d(
			X, Y,
			depart, T,
			x, y, p, l);
		x[i] -= _1E5;
		//
		Sx = 0;
		FOR(0, j, Y*T) Sx += powf((y[j]-sorties[j]), 2) / 2;// / (Y*T);
		//
		_dx[i] = (Sx-S)/_1E5;
	};

	titre("intel dot1d 1e-5");
	puts("1e5 dp - intel dp\n");
	comparer_lst(_dp, dp, Y*X, 0.001);
	puts("1e5 dx - intel dx\n");
	comparer_lst(_dx, dx, X*T, 0.001);

	free(x);
	free(y);
	free(p);
	free(l);
	free(sorties);
	free(dp);
	free(dx);
	free(dy);
	free(_dp);
	free(_dx);
};

static void verif_nvidia_intel()
{
	uint depart = 0;
	uint X = 3*16, T = 2*16, Y = 4*16;
	//
	float * x = lst_rnd(X*T, -1, 1);
	float * y = alloc<float>(T*Y);
	float * p = lst_rnd((X+1)*Y, -1, 1);
	float * l = alloc<float>(T*Y);
	float * sorties = lst_rnd(Y*T, -1, 1);
	float * dp = zero<float>((X+1)*Y);
	float * dx = zero<float>(T*X);
	float * dy = zero<float>(T*Y);

	intel_dot1d(
		X, Y,
		depart, T,
		x, y,
		p,
		l);
	//	dS
	FOR(0, i, Y*T) dy[i] = (y[i]-sorties[i]) / (Y*T);
	//
	d_intel_dot1d(
		X, Y,
		depart, T,
		x, y, p, l,
		dy, dx, dp);

	float * x__d = cpu_vers_gpu<float>(x, X*T);
	float * y__d = cpu_vers_gpu<float>(y, Y*T);
	float * p__d = cpu_vers_gpu<float>(p, (X+1)*Y);
	float * l__d = cpu_vers_gpu<float>(l, Y*T);
	float * sorties__d = cpu_vers_gpu<float>(sorties, Y*T);
	float * dp__d = cpu_vers_gpu<float>(dp, (X+1)*Y);
	float * dx__d = cpu_vers_gpu<float>(dx, X*T);
	float * dy__d = cpu_vers_gpu<float>(dy, T*Y);
	//
	FOR(0, mode, 3) {
		CONTROLE_CUDA(cudaMemset(dp__d, 0, sizeof(float) * (X+1)*Y));
		CONTROLE_CUDA(cudaMemset(dx__d, 0, sizeof(float) * X*T));
		CONTROLE_CUDA(cudaMemset(y__d, 0, sizeof(float) * T*Y));
		CONTROLE_CUDA(cudaMemset(l__d, 0, sizeof(float) * Y*T));
		//
		nvidia_dot1d(
			X, Y,
			depart, T,
			x__d, y__d,
			p__d,
			l__d,
			mode);
		d_nvidia_dot1d(
			X, Y,
			depart, T,
			x__d, y__d,
			p__d,
			l__d,
			dy__d,
			dx__d,
			dp__d,
			mode);
		printf("########## Verif dot1d MODE %i #######\n", mode);

		puts("y__d\n");
		comparer_lst(y, gpu_vers_cpu<float>(y__d, (T)*Y), T*Y, 0.001);
		puts("l__d\n");
		comparer_lst(l, gpu_vers_cpu<float>(l__d, T*Y), T*Y, 0.001);
		puts("dp__d\n");
		comparer_lst(dp, gpu_vers_cpu<float>(dp__d, (X+1)*Y), (X+1)*Y, 0.001);
		puts("dx__d\n");
		comparer_lst(dx, gpu_vers_cpu<float>(dx__d, X*T), X*T, 0.001);
	}
};

void verif_do1d() {
	verif_intel_papier();
	verif_intel_1e5();
	verif_nvidia_intel();
};