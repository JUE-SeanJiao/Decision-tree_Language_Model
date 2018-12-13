#!/usr/unin/env python

import numpy as np
from math import log
from copy import deepcopy
from random import randint

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

########## Collect 4-gram statistics ##########
def collect_4gm(data):
	a_4gm = {}
	for i in range(len(data)-3):
		histry = data[i]+data[i+1]+data[i+2]
		dd = histry+data[i+3]
		if dd not in a_4gm:
			a_4gm[dd] = 0
		a_4gm[dd] += 1
	return a_4gm,len(data)-3
[dev_4gm,dev_ct] = collect_4gm(dev)
[hld_4gm,hld_ct] = collect_4gm(hld)
[tes_4gm,tes_ct] = collect_4gm(test)

########## Build decision tree ##########

# ========================= Functions =========================

def count_data(node):
	count = 0
	for dd,ct in node.items():
		count += ct
	return count

def get_ln(node,n):
	lnL = []
	for dd in node:
		if dd[n-1] not in lnL:
			lnL.append(dd[n-1])
	ln ={}
	for l in lnL:
		ln[l] = 0
		for dd in node:
			if dd[n-1]==l:
				ln[l] += node[dd]
	return ln

def count_histry(node):
	histt = []
	for dd in node:
		if dd[:3] not in histt:
			histt.append(dd[:3])
	return len(histt)

def get_subset(setIdx,lOrder,node,n):
	subset = {}
	for aa in setIdx:
		l = lOrder[aa]
		for dd,ct in node.items():
			if l==dd[n-1]:
				subset[dd] = ct
	ctt = count_data(subset)
	return subset,ctt

def comp_f2(setIdx,lOrder,node,ch,n):
	tmp1 = 0
	tmp2 = 0
	for aa in setIdx:
		ll = lOrder[aa]
		for dd in node:
			if dd[n-1]==ll:
				tmp2 += node[dd]
				if dd[3]==ch:
					tmp1 += node[dd]
	frq = 0
	if tmp2!=0:
		frq = tmp1/float(tmp2)
	return frq

def comp_entrQt(setIdx1,setIdx2,node,l,ct,lOrder,n,f21,f22):
	entr1 = 0
	entr2 = 0
	for ch in pt:  ### sum over w
		idx = pt.index(ch)
		f1 = 0
		for dd in node:
			if dd[n-1]==l and dd[3]==ch:
				f1 += node[dd]
		f1 = f1/float(ct)
		if f1!=0:
			entr1 += f1*log(f1/f21[idx])
			entr2 += f1*log(f1/f22[idx])
	return entr1,entr2

def comp_entrpy(node):
	count = count_data(node)
	entrpy = 0
	for ch in pt:
		ctt = 0
		for dd,ct in node.items():
			if dd[3] == ch:
				ctt += ct
		if count!=0:
			ctt = ctt/float(count)
		if ctt!=0:
			entrpy += ctt*log(ctt,2)
	entrpy = -entrpy
	return entrpy

def prtitn_node(node,n,opt,prt1=[],prt2=[]):
	count = count_data(node)
	ln = get_ln(node,n)
	lOrder = ln.keys()

	if opt=='dev':
		prtitn = {}
		for idx in range(len(ln)):
			prtitn[str(idx)] = randint(0,1)
		A = []
		A_bar = []
		for idx, ptn in prtitn.items():
			if ptn==0:
				A.append(int(idx))
			else:
				A_bar.append(int(idx))
		A.sort()
		A_bar.sort()
		# print A
		# print A_bar

		iterN = 0
		A_0 = []
		while A!=A_0:
			iterN += 1
			# print '* iteration',iterN
			A_0 = A[:]

			f21 = np.zeros(27)
			f22 = np.zeros(27)
			for ch in pt:
				idx = pt.index(ch)
				f21[idx] = comp_f2(A,lOrder,node,ch,n)
				f22[idx] = comp_f2(A_bar,lOrder,node,ch,n)

			for l,ct in ln.items():  ### for each beta...
				bIdx = lOrder.index(l)

				[qt1,qt2] = comp_entrQt(A,A_bar,node,l,ct,lOrder,n,f21,f22)

				if qt1<=qt2 and prtitn[str(bIdx)]==1:
					A_bar.remove(bIdx)
					A.append(bIdx)
					prtitn[str(bIdx)] = 0
				if qt1>qt2 and prtitn[str(bIdx)]==0:
					A.remove(bIdx)
					A_bar.append(bIdx)
					prtitn[str(bIdx)] = 1
			A.sort()
			A_bar.sort()

	elif opt=='hld' or opt=='tes':
		A = []
		A_bar = []
		prtitn = {}

		for l in lOrder:
			idx = lOrder.index(l)
			if l in prt1:
				A.append(idx)
				prtitn[str(idx)] = 0
			elif l in prt2:
				A_bar.append(idx)
				prtitn[str(idx)] = 1
			else:
				prtitn[str(idx)] = randint(0,1)  ### Tie-breaking procedure
				if prtitn[str(idx)]==0:
					A.append(idx)
				else:
					A_bar.append(idx)
		# print A
		# print A_bar
		# print prtitn
	
	[subset1,ct1] = get_subset(A,lOrder,node,n)
	[subset2,ct2] = get_subset(A_bar,lOrder,node,n)

	if opt=='tes':
		return subset1,subset2,ct1,ct2

	elif opt=='dev' or opt=='hld':
		part1 = []
		part2 = []
		for idx in range(len(ln)):
			if prtitn[str(idx)]==0:
				part1.append(lOrder[idx])
			else:
				part2.append(lOrder[idx])
		entr1 = comp_entrpy(subset1)
		entr2 = comp_entrpy(subset2)
		entr = (ct1*entr1+ct2*entr2)/float(count)
		return subset1,subset2,part1,part2,entr1,entr2,entr,ct1,ct2

def chou_split(node):
	subset1 = []
	subset2 = []
	part1 = []
	part2 = []
	entr1 = np.zeros(3)
	entr2 = np.zeros(3)
	entr = np.zeros(3)
	ct1 = np.zeros(3)
	ct2 = np.zeros(3)
	for i in range(3):
		# print '========================='
		# print '$ l'+str(i+1)
		[ss1,ss2,pt1,pt2,entr1[i],entr2[i],entr[i],ct1[i],ct2[i]] = prtitn_node(node,i+1,'dev')
		# print count_histry(node),'->',count_histry(ss1),'|',count_histry(ss2)
		subset1.append(ss1)
		subset2.append(ss2)
		part1.append(pt1)
		part2.append(pt2)
	# print entr
	ques = np.argmin(entr)
	subset1 = subset1.pop(ques)
	subset2 = subset2.pop(ques)
	part1 = part1.pop(ques)
	part2 = part2.pop(ques)
	entr1 = entr1[ques]
	entr2 = entr2[ques]
	ct1 = ct1[ques]
	ct2 = ct2[ques]
	return subset1,subset2,ques,part1,part2,ct1,ct2

def comp_l4frq(node):  ### no smoothing, just based on uni-gram counts
	prd_ch = np.zeros(27)
	for i in range(27):
		for dd,ct in node.items():
			if dd[3]==pt[i]:
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
ndQ = [[]]
ndA = [[]]
ct_nd = [len(hld)-3]

prd_trn = [comp_l4frq(hld_4gm)] ### with smoothing ###
# prd_trn = []  ### without smoothing ###
frq_tes = []
terN = 0

entr_hld_0 = comp_entrpy(hld_4gm)
# print 'Held-out entropy:',entr_hld_0
ndEntr = [entr_hld_0]

while 1:
	nd = terN
	print '====================================================================='
	print 'Number of terminals:',nd
	print 'Number of nodes:',len(dev_node)

	ndDev = dev_node[nd]
	ndHld = hld_node[nd]
	ndTes = tes_node[nd]
	entr_hld_0 = ndEntr[nd]
	count = ct_nd[nd]
	ndPrd_trn = prd_trn[nd]

	# print count
	# print 'Hld entropy before partition:',entr_hld_0

	[sbst1_dev,sbst2_dev,ques,part1,part2,ct1_dev,ct2_dev] = chou_split(ndDev)
	[sbst1_hld,sbst2_hld,pt1_hld,pt2_hld,entr_hld1,entr_hld2,entr_hld,ct1,ct2] = prtitn_node(ndHld,ques+1,'hld',part1,part2)
	[sbst1_tes,sbst2_tes,ct1_tes,ct2_tes] = prtitn_node(ndTes,ques+1,'tes',pt1_hld,pt2_hld)
	
	red_hld = (entr_hld_0-entr_hld)*count/float(len(hld)-3)
	print 'Hld entropy reduction:',red_hld

	nHisDev = count_histry(ndDev)
	nHisHld = count_histry(ndHld)
	nHisTes = count_histry(ndTes)
	if red_hld>thrsh:
		print 'Question: l'+str(ques+1)
		print 'Answer:\n',part1,'\n',part2

		print 'Dev histries',nHisDev,'->',count_histry(sbst1_dev),'|',count_histry(sbst2_dev)
		print 'Hld counts',count,'->',ct1,'|',ct2
		print 'Hld histries',nHisHld,'->',count_histry(sbst1_hld),'|',count_histry(sbst2_hld)
		print 'Tes histries',nHisTes,'->',count_histry(sbst1_tes),'|',count_histry(sbst2_tes)

		dev_node.append(sbst1_dev)
		dev_node.append(sbst2_dev)
		dev_node.pop(nd)

		### with smoothing ###
		prd_trn.append(smooth(sbst1_dev,ct1_dev,dev_ct,ndPrd_trn))
		prd_trn.append(smooth(sbst2_dev,ct2_dev,dev_ct,ndPrd_trn))
		prd_trn.pop(nd)		

		hld_node.append(sbst1_hld)
		hld_node.append(sbst2_hld)
		hld_node.pop(nd)

		ndEntr.append(entr_hld1)
		ndEntr.append(entr_hld2)
		ndEntr.pop(nd)

		ct_nd.append(ct1)
		ct_nd.append(ct2)
		ct_nd.pop(nd)

		tes_node.append(sbst1_tes)
		tes_node.append(sbst2_tes)
		tes_node.pop(nd)

		### =============== >>> ===============

		leQ = ndQ[nd][:]
		leQ.append(ques)
		ans1 = ndA[nd][:]
		ans1.append(part1)
		ans2 = ndA[nd][:]
		ans2.append(part2)

		ndQ.append(leQ)
		ndQ.append(leQ)
		ndQ.pop(nd)

		ndA.append(ans1)
		ndA.append(ans2)
		ndA.pop(nd)

		### =============== <<< ===============

	else:
		terN += 1

		# prd_trn.append(comp_l4frq(ndDev))  ### without smoothing ###
		frq_tes.append(comp_l4frq(ndTes))

		print 'Dev histries',nHisDev
		print 'Hld counts',count
		print 'Hld histries',nHisHld
		print 'Tes histries',nHisTes
		for i in range(3):
			print 'l'+str(i+1),get_ln(ndHld,i+1).keys()

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
		print 'l'+str(ndQ[i][j]+1)+'-'+str(ndA[i][j])
	# err[i] = np.linalg.norm(prd_trn[i]-prd_tes[i])
	crEntr = 0
	for j in range(27):
		if prd_trn[i][j]!=0:
			# crEntr += prd_tes[i][j]*log(prd_trn[i][j],2)
			crEntr += frq_tes[i][j]*log(prd_trn[i][j],2)
	prplx[i] = 2**crEntr	## Compute perplexity by cross entropy
	# print 'Error:',err[i]
	print 'Perplexity:',prplx[i]
	print 'trn prediction:',np.argmax(prd_trn[i]),'\n',prd_trn[i]
	# print 'tes prediction:',np.argmax(prd_tes[i]),'\n',prd_tes[i]
	print 'tes frequencies:',np.argmax(frq_tes[i]),'\n',frq_tes[i]
print '====================================================================='
# print 'Mean error for each cluster:',np.sum(err)/float(terN)
print 'Average perplexity of each cluster:',np.sum(prplx)/float(terN)
# np.savetxt('perplexity-chou-entropy.csv',prplx)

print '------------------------------- END ---------------------------------'

















