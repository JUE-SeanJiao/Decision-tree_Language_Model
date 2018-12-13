#!/usr/unin/env python

import numpy as np
from math import log
import re

print '-------------------- START ----------------------'

with open('textA.txt','r') as f:
	train = f.read()
train1 = ' '+train
noc = len(train)

pt_uni = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
p_uni = np.zeros(27)
for i in range(27):
		patt = re.compile(pt_uni[i])
		ctobj = patt.findall(train)
		p_uni[i] = len(ctobj)
p_uni = p_uni/noc
p_bi = np.zeros([27,27])
for i in range(27):
	for j in range(27):
		patt = re.compile(pt_uni[i]+pt_uni[j])
		ctobj = patt.findall(train1)
		p_bi[i,j] = len(ctobj)
p_bi = p_bi/np.sum(p_bi)
# print p_uni
# print p_bi

f = open('cluster-tree.txt','w')

cl = []
for i in range(27):
	cl.append([i])
# clTree = [cl]
# print clTree

nIter = 0
while 1:
	nIter += 1
	print '* iteration',nIter

	nCl = len(cl)
	I_ij = np.ones([nCl,nCl])*(-float('inf'))
	for i in range(nCl):
		# print '*',i
		# for j in range(i+1):
		# 	I_ij[i,j] = -float('inf')
		for j in range(i+1,nCl):
			# print '$',j

			union = cl[i]+cl[j]
			qt1 = 0
			qt2 = 0
			qt3 = 0
			qt4 = 0

			clrm = []
			for ch in range(nCl):
				if ch!=i and ch!=j:
					clrm.append(cl[ch])

			f_ij = 0
			f_ijij = 0
			for l1 in union:
				f_ij += p_uni[l1]
				for l2 in union:
					f_ijij += p_bi[l1,l2]
			if f_ijij!=0:
				qt4 = f_ijij*log(f_ijij/f_ij/f_ij)


			for kk in clrm:
				f_k = 0
				for l1 in kk:
					f_k += p_uni[l1]
				# print '$'
				# print f_k

				for mm in clrm:
					f_m = 0
					for l2 in mm:
						f_m += p_uni[l2]

					f_km = 0
					for l1 in kk:
						for l2 in mm:
							f_km += p_bi[l1,l2]

					# print '*'
					# print f_m
					# print f_km
					if f_km!=0:
						qt1 += f_km*log(f_km/f_k/f_m)

			for kk in clrm:
				f_k = 0
				f_kij = 0
				for l1 in kk:
					f_k += p_uni[l1]
					for l2 in union:
						f_kij += p_bi[l1,l2]
				if f_kij!=0:
					qt2 += f_kij*log(f_kij/f_k/f_ij)
			# print f_k

			for mm in clrm:
				f_m = 0
				f_ijm = 0
				for l2 in mm:
					f_m += p_uni[l2]
					for l1 in union:
						f_ijm += p_bi[l1,l2]
				if f_ijm!=0:
					qt3 += f_ijm*log(f_ijm/f_ij/f_m)

			# print f_km,f_k,f_m

			I_ij[i,j] = qt1+qt2+qt3+qt4
			# print qt1,qt2,qt3,qt4
			# print I_ij[i,j]

		# 	break
		# break

	# print I_ij
	idx = np.argmax(I_ij)
	print np.amax(I_ij)
	[i_star,j_star] = np.unravel_index(idx,(nCl,nCl))
	# print i_star,j_star
	# print I_ij[i_star,j_star]
	cl[i_star] = cl[i_star]+cl[j_star]
	cl[i_star].sort()
	cl.pop(j_star)
	print cl

	for cc in cl:
		for i in range(len(cc)):
			f.write(str(cc[i]))
			if i!=len(cc)-1:
				f.write(',') 
		f.write(' ')
	f.write('\n')

	# clTree.append(cl)
	# print clTree

	if len(cl)==1:
		break


f.close()

for i in range(27):
	print str(i),pt_uni[i]

# clTree.reverse()
# for tr in clTree:
# 	print tr





