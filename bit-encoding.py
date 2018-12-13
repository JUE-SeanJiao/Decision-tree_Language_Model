#!/usr/unin/env python

import numpy as np
from math import log
from copy import deepcopy

print '------------------------------ START --------------------------------'

pt = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
with open('textA.txt','r') as f:
	train = f.read()
with open('textB.txt','r') as f:
	test = f.read()
noc = len(train)
# hld = train[:noc/5]
# dev = train[noc/5:]
dev = train[:4*noc/5]
hld = train[4*noc/5:]
thrsh = 0.01
# thrsh = 0.1

clTree = []
with open('cluster-tree.txt','r') as f:
	for line in f:
		ll = line.rstrip(' \n').split(' ')
		lev = []
		for l1 in ll:
			l2 = l1.split(',')
			l3 = []
			for lll in l2:
				l3.append(int(lll))
			lev.append(l3)
		clTree.append(lev)
clTree.reverse()
cl = []
for i in range(27):
	cl.append([i])
clTree.append(cl)

############### Bit encoding ###############
bitCode = {}
for ch in pt:
	bitCode[ch] = ''
for lev in range(len(clTree)):
	if lev==0:
		continue
	bit = 0
	for cl in clTree[lev]:
		if cl not in clTree[lev-1]:
			for idx in cl:
				bitCode[pt[idx]] = bitCode[pt[idx]]+str(bit)
			bit += 1
# print 'Bit-Encoding:\n',bitCode
K = 0 ### bit-depth
for ch in bitCode:
	if len(bitCode[ch])>K:
		K = len(bitCode[ch])
# print K 
for ch in bitCode:
	while len(bitCode[ch])<K:
		bitCode[ch] = bitCode[ch]+'0'
print 'Bit-Encoding:\n',bitCode

########## Collect 4-gram statistics ##########
def collect_4gm(data):
	a_4gm = {}
	for i in range(len(data)-3):
		histry = bitCode[data[i+2]]+bitCode[data[i+1]]+bitCode[data[i]]
		dd = histry+data[i+3]
		if dd not in a_4gm:
			a_4gm[dd] = 0
		a_4gm[dd] += 1
	return a_4gm,len(data)-3
[dev_4gm,dev_ct] = collect_4gm(dev)
[hld_4gm,hld_ct] = collect_4gm(hld)
[tes_4gm,tes_ct] = collect_4gm(test)

print '---------------------------------------------------------------------'

########## Build decision tree ##########

# ========================= Functions =========================

def count_data(node):
	count = 0
	for dd,ct in node.items():
		count += ct
	return count

def count_histry(node):
	histt = []
	for dd in node:
		if dd[:3*K] not in histt:
			histt.append(dd[:3*K])
	return len(histt)

def comp_entrpy(node):
	count = count_data(node)
	entrpy = 0
	for ch in pt:
		ctt = 0
		for dd,ct in node.items():
			if dd[3*K] == ch:
				ctt += ct
		if count!=0:
			ctt = ctt/float(count)
		if ctt!=0:
			entrpy += ctt*log(ctt,2)
	entrpy = -entrpy
	return entrpy

def split_node(node,quesBit):
	count = count_data(node)
	subset1 = {}
	subset2 = {}
	ct1 = 0
	ct2 = 0
	for dd,ct in node.items():
		if dd[quesBit] == '0':
			subset1[dd] = ct
			ct1 += ct
		else:
			subset2[dd] = ct
			ct2 += ct
	entr1 = comp_entrpy(subset1)
	entr2 = comp_entrpy(subset2)
	entrpy = (ct1*entr1+ct2*entr2)/float(count)
	return subset1,subset2,ct1,ct2,entr1,entr2,entrpy

def comp_l4frq(node):
	prd_ch = np.zeros(27)
	for i in range(27):
		for dd,ct in node.items():
			if dd[3*K]==pt[i]:
				prd_ch[i] += ct
	prd_ch = prd_ch/np.sum(prd_ch)
	return prd_ch

def smooth(subset,ctt,count,ndPrd_0):
	gamma = ctt/float(count)
	ndPrd = comp_l4frq(subset)*gamma+ndPrd_0*(1-gamma)
	return ndPrd

# ============================================================

dev_node = [deepcopy(dev_4gm)]
hld_node = [deepcopy(hld_4gm)]
tes_node = [deepcopy(tes_4gm)]

prd_trn = [comp_l4frq(hld_4gm)]  ### with smoothing ###
# prd_trn = []  ### without smoothing ###
frq_tes = []
terN = 0

ques = [0,K,2*K]
ques_nd = [ques]
ndQ = [[]]
ndA = ['']
ct_nd = [hld_ct]

entr_hld_0 = comp_entrpy(hld_4gm)
# print 'Held-out entropy:',entr_hld_0
ndEntrpy = [entr_hld_0]

while 1:
	nd = terN
	print '=============================================='
	print 'Number of terminals:',nd
	print 'Number of nodes:',len(dev_node)

	ques = ques_nd[nd][:]
	ndDev = dev_node[nd]
	ndHld = hld_node[nd]
	ndTes = tes_node[nd]
	entr_hld_0 = ndEntrpy[nd]
	count = ct_nd[nd]
	ndPrd_trn = prd_trn[nd]  ### with smoothing ###

	# print count
	# print ques
	# print 'Hld Entropy before partition:',entr_hld_0

	entr_dev = np.ones(3*K)*float('inf')
	for i in ques:
		[_,_,_,_,_,_,entr_dev[i]] = split_node(ndDev,i)
		# print entr_dev[i]
	bit = np.argmin(entr_dev)

	[subset1,subset2,ct1,ct2,entr_hld1,entr_hld2,entr_hld] = split_node(ndHld,bit)
	red_hld = (entr_hld_0-entr_hld)*count/float(hld_ct)
	print 'Question bit:',bit+1
	print 'Hld Entropy reduction:',red_hld

	nHisDev = count_histry(ndDev)
	nHisHld = count_histry(ndHld)
	nHisTes = count_histry(ndTes)
	if red_hld>thrsh:

		print 'Hld counts',count,'->',ct1,'|',ct2
		print 'Hld histries',nHisHld,'->',count_histry(subset1),'|',count_histry(subset2)

		hld_node.append(subset1)
		hld_node.append(subset2)
		hld_node.pop(nd)

		ndEntrpy.append(entr_hld1)
		ndEntrpy.append(entr_hld2)
		ndEntrpy.pop(nd)

		ct_nd.append(ct1)
		ct_nd.append(ct2)
		ct_nd.pop(nd)

		[subset1,subset2,ct1_dev,ct2_dev,_,_,_] = split_node(ndDev,bit)
		print 'Dev histries',nHisDev,'->',count_histry(subset1),'|',count_histry(subset2)

		dev_node.append(subset1)
		dev_node.append(subset2)
		dev_node.pop(nd)

		### with smoothing ###
		prd_trn.append(smooth(subset1,ct1_dev,dev_ct,ndPrd_trn))
		prd_trn.append(smooth(subset2,ct2_dev,dev_ct,ndPrd_trn))
		prd_trn.pop(nd)

		[subset1,subset2,ct1,ct2,_,_,_] = split_node(ndTes,bit)
		print 'Tes histries',nHisTes,'->',count_histry(subset1),'|',count_histry(subset2)

		tes_node.append(subset1)
		tes_node.append(subset2)
		tes_node.pop(nd)

		ques.remove(bit)
		if bit!=(K-1) and bit!=(2*K-1) and bit!=(3*K-1):
			ques.append(bit+1)

		ques_nd.append(ques)
		ques_nd.append(ques)
		ques_nd.pop(nd)

		### =============== >>> ===============
		
		bitsQ = ndQ[nd][:]
		bitsQ.append(bit)
		ans1 = ndA[nd]+'0'
		ans2 = ndA[nd]+'1'

		ndQ.append(bitsQ)
		ndQ.append(bitsQ)
		ndQ.pop(nd)

		ndA.append(ans1)
		ndA.append(ans2)
		ndA.pop(nd)

		### =============== <<< ===============

	else:
		terN += 1

		# prd_trn.append(comp_l4frq(ndDev))  ### without smoothing ###
		frq_tes.append(comp_l4frq(ndTes))

		print 'Hld counts',count
		print 'Hld histries',nHisHld
		print 'Dev histries',nHisDev
		print 'Tes histries',nHisTes

	print 'Number of terminals:',terN
	print 'Number of nodes:',len(dev_node)

	if terN == len(dev_node):
		break


print '----------------------------- RESULTS -------------------------------'

# err = np.zeros(terN)
prplx = np.zeros(terN)
for i in range(terN):
	print '====================================================================='
	print '* Cluster',i+1
	print 'Questions:',len(ndQ[i]),'\n',ndQ[i]
	print 'Answers:','\n$',ndA[i]
	print 'question-answer:'
	for j in range(len(ndQ[i])):
		print str(ndQ[i][j]+1)+'-'+ndA[i][j]
	# err[i] = np.linalg.norm(prd_trn[i]-prd_tes[i])
	crEntr = 0
	for j in range(27):
		if prd_trn[i][j]!=0:
			# crEntr += prd_tes[i][j]*log(prd_trn[i][j],2)
			crEntr += frq_tes[i][j]*log(prd_trn[i][j],2)
	prplx[i] = 2**crEntr	## Computer perplexity by conditional entropy
	# print 'Error:',err[i]
	print 'Perplexity:',prplx[i]
	print 'trn prediction:',np.argmax(prd_trn[i]),'\n',prd_trn[i]
	# print 'tes prediction:',np.argmax(prd_tes[i]),'\n',prd_tes[i]
	print 'tes frequencies:',np.argmax(frq_tes[i]),'\n',frq_tes[i]
print '====================================================================='
# print 'Mean error for each cluster:',np.sum(err)/float(terN)
print 'Average perplexity of each cluster:',np.sum(prplx)/float(terN)
# np.savetxt('perplexity-bit-encoding.csv',prplx)
	
print '------------------------------- END ---------------------------------'


















