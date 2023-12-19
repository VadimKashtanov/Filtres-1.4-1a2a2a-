#! /usr/bin/python3

from math import exp, tanh
import struct as st

N = 8

def lire_uint(I, _bin):
	l = st.unpack('I'*I, _bin[:st.calcsize('I')*I])
	return l, _bin[st.calcsize('I')*I:]

def lire_flotants(I, _bin):
	l = st.unpack('f'*I, _bin[:st.calcsize('f')*I])
	return l, _bin[st.calcsize('f')*I:]

ema_ints = [
#		   ema,  int
	[ 0,    1,    1],
	[ 1,    2,    2],
	[ 2,    4,    4],
	[ 3,    6,    6],
	[ 4,   10,   10],
	[ 5,   20,   20],
	[ 6,   50,   50],
	[ 7,  100,  100],
	[ 8,  200,  200],
	[ 9,  500,  500],
    [10, 1000, 1000]
]

with open("prixs/prixs.bin", "rb") as co:
	#
	import yfinance as yf
	prixs = yf.download(tickers=['BTC-USD'], period='500d', interval='1h').Close.values
	PRIXS = len(prixs)
	#
	#_bin = co.read()
	#(PRIXS,), _bin = lire_uint(1, _bin)
	#prixs, _bin = lire_flotants(PRIXS, _bin)

norme = lambda arr: [(e-min(arr))/(max(arr)-min(arr)) for e in arr]
e_norme = lambda arr: [2*(e-min(arr))/(max(arr)-min(arr))-1 for e in arr]

def ema(arr, K):
	e = [arr[0]]
	for p in arr[1:]:
		e += [e[-1]*(1-1/(1+K)) + p*1/(1+K)]
	return e

prixs_ema = [ema(prixs, K) for _,K,_ in ema_ints]

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
		_id, _ema, _intervalle = ema_ints[ligne]
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

plusde50 = lambda x: ((x) if abs(x) > 0.20 else 0)

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
	T = 2*7*24
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