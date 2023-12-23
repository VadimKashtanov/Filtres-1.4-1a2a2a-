#! /usr/bin/python3

from math import exp, tanh
import struct as st
import numpy as np

N = 8

def lire_uint(I, _bin):
	l = st.unpack('I'*I, _bin[:st.calcsize('I')*I])
	return l, _bin[st.calcsize('I')*I:]

def lire_flotants(I, _bin):
	l = st.unpack('f'*I, _bin[:st.calcsize('f')*I])
	return l, _bin[st.calcsize('f')*I:]

import time
import datetime

import requests

#requette_bitget = lambda de, a: eval(requests.get(f"https://api.bitget.com/api/mix/v1/market/candles?symbol=BTCUSDT_UMCBL&granularity=1H&startTime={de*1000}&endTime={a*1000}").text)
requette_bitget = lambda de, a: eval(requests.get(f"https://api.bitget.com/api/mix/v1/market/history-candles?symbol=BTCUSDT_UMCBL&granularity=1H&startTime={de*1000}&endTime={a*1000}").text)
#donnees = requette_bitget(jour_unix(2023, 11, 28), int(time.time()))
donnees = []
H = 200
la = int(time.time())
for i in range(int(1000*8/H + 1)):
	derniere = requette_bitget(la-(i+1)*H*60*60, la-i*H*60*60)[::-1]
	donnees += derniere
	if i%1 == 0: print(f"%% = {i/int(1000*8/H + 1)*100},   len(derniere)={len(derniere)}")
donnees = donnees[::-1]

norme = lambda arr: [(e-min(arr))/(max(arr)-min(arr)) for e in arr]
e_norme = lambda arr: [2*(e-min(arr))/(max(arr)-min(arr))-1 for e in arr]

def ema(arr, K):
	e = [arr[0]]
	for p in arr[1:]:
		e += [e[-1]*(1-1/(1+K)) + p*1/(1+K)]
	return e

#	id  ema  interv source
prixs   = [float(o) for _,o,_,_,_,vB,vU in donnees]
volumes = [float(o)*float(vB) - float(vU) for _,o,_,_,_,vB,vU in donnees]
ema12 = ema(prixs, K=12)
ema26 = ema(prixs, K=26)
__macd  = [a-b for a,b in zip(ema12, ema26)]
ema9_macd = ema(__macd, K=9)
macds = [a-b for a,b in zip(__macd, ema9_macd)]

PRIXS = len(prixs)

exec("ema_ints = [" + """ 
	{ 0,    1,    1,   prixs},
	{ 1,    3,    1,   prixs},
	{ 2,    6,    1,   prixs},
	{ 3,   12,    1,   prixs},
	{ 4,   24,    1,   prixs},
	{ 5,    8,    4,   prixs},
	{ 6,   12,    4,   prixs},
	{ 7,   24,    4,   prixs},
	{ 8,   40,    4,   prixs},
	{ 9,   20,    8,   prixs},
	{10,   40,    8,   prixs},
	{11,   80,    8,   prixs},
	{12,   50,   20,   prixs},
	{13,  100,   20,   prixs},
	{14,  200,   20,   prixs},
	{15,  300,  100,   prixs},
	{16,  600,  100,   prixs},
	{17, 1000,  100,   prixs},
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
    {28,    1,    1,   macds},
    {29,    2,    2,   macds},
    {30,    4,    4,   macds},
    {31,    1,    1,   volumes},
    {32,    5,    1,   volumes},
    {33,    5,    5,   volumes},
    {34,   20,    5,   volumes},
    {35,   20,   20,   volumes},
    {36,  100,   20,   volumes},
    {37,  100,  100,   volumes},
    {38,  300,  100,   volumes}
]""".replace('{', '[').replace('}', ']'))

prixs_ema = [ema(source, K) for _,K,_,source in ema_ints]

class Model:
	def __init__(self, fichier : str):
		with open(fichier, "rb") as co:
			_bin = co.read()
		#
		(self.C,), _bin = lire_uint(1, _bin)
		self.ST, _bin = lire_uint(self.C, _bin)
		(self.bloques,), _bin = lire_uint(1, _bin)
		(self.f_par_bloque,), _bin = lire_uint(1, _bin)
		self.lignes, _bin = lire_uint(self.bloques, _bin)
		#
		self.f, _bin = lire_flotants(self.ST[0]*N, _bin)
		self.p = [[]]
		for c in range(1, self.C):
			p, _bin = lire_flotants(self.ST[c]*(self.ST[c-1]+1), _bin)
			self.p += [p]

	def filtre(self, ligne, t, f):
		_id, _ema, _intervalle, source = ema_ints[ligne]
		x = norme([prixs_ema[_id][t-i*_intervalle] for i in range(N)])#[::-1] j'avais pas inverser dans le C mais pas gravce, ca change rien si j'oublie pas
		#
		s = (sum((1+abs(x[i]-f[i]))**.5 for i in range(N))) / N - 1
		d = (sum((1+abs(x[i+1]-x[i]-f[i+1]+f[i]))**2 for i in range(N-1))) / (N-1) - 1
		#
		return 2*exp(-s*s -d*d)-1

	def perceptron(self, x, p, y):
		X = len(x)
		for i in range(len(y)):
			y[i] = tanh(sum(p[(X+1)*i + j]*x[j] for j in range(X)) + p[(X+1)*i + (X+1-1)])

	def fonction(self, t):
		y = [[0 for i in range(st)] for st in self.ST]
		for b in range(self.bloques):
			ligne = self.lignes[b]
			for f in range(self.f_par_bloque):
				y[0][b*self.f_par_bloque + f] = self.filtre(
					ligne, t,
					self.f[b*self.f_par_bloque*N + f*N:b*self.f_par_bloque*N + f*N+N]
				)

		for c in range(1, self.C):
			self.perceptron(y[c-1], self.p[c], y[c])

		return y[-1][0]

import matplotlib.pyplot as plt

signe = lambda x: (1 if x >= 0 else -1)

plusde50 = lambda x: ((x) if abs(x) > 0.01 else 0)

if __name__ == "__main__":
	mdl = Model("mdl.bin")

	I = 50

	print(mdl.lignes)

	pred = [mdl.fonction(PRIXS-i-1) for i in range(I)][::-1]

	print(pred)

	plt.plot([2*x-1 for x in norme(prixs[-I:])], label='prixs')
	plt.plot(pred, label='pred')
	plt.plot([0 for _ in pred], label='-')
	for i in range(len(pred)):
		plt.plot([i for _ in pred], e_norme(list(range(len(pred)))), '--')
	plt.legend()
	plt.show()

	u = 50
	usd = []
	T = 3*7*24
	#
	decale = 0
	#
	for i in range(T):
		#print(f"prix = {(prixs[PRIXS-decale-T-1+i+1]/prixs[PRIXS-decale-T-1+i]-1)}")
		u += u * plusde50(mdl.fonction(PRIXS-decale-T-1+i))*(prixs[PRIXS-decale-T-1+i+1]/prixs[PRIXS-decale-T-1+i]-1)*50
		if (u <= 0): u = 0
		print(f"usd = {u}")
		usd += [u]
	plt.plot(usd); plt.show()

	'''
	p = 0
	I = 0
	for i in range(40000, PRIXS-3):
		I += 1
		if signe(mdl.fonction(i)) == signe(prixs[i+2]/prixs[i]-1):
			p += 1
		if i % 1000 == 0: print((i-8000)/(PRIXS-8000)*100)
	print("pred = ", 100*p/I)
	'''