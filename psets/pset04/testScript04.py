#!/usr/bin/env python
#
# This script must be run in the same directory as:
#   pset04.py

import numpy as np
import scipy as sp
import pset04


# test data
X1 = sp.random.normal(0,1,(97,4))
y1 = (X1[:,0] + X1[:,1] > 0)*2 - 1
X2 = sp.random.normal(0,1,(97,8))
y2 = (X2[:,0] + X2[:,1] > 0)*2 - 1
X3 = sp.random.normal(0,1,(97,8))
y3 = (X3[:,0] + X3[:,1] > 0)*2 - 1
X4 = sp.random.normal(0,1,(97,10))
y4 = (X4[:,0] + X4[:,1] > 0)*2 - 1


# these are 'fake' answers, that simply have the correct
# format; they will be replaced with real answers when we
# grade them
ans1 = np.ones((4))
ans2 = np.ones((8))
ans3 = np.ones((8))
ans4 = np.ones((10))
ans5 = np.ones((4))
ans6 = np.ones((8))
ans7 = np.ones((8))
ans8 = np.ones((10))


# calculate results from your models
res1 = pset04.my_dual_svm(X1,y1,C=2)
res2 = pset04.my_dual_svm(X2,y2,C=5)
res3 = pset04.my_dual_svm(X3,y3,C=10)
res4 = pset04.my_dual_svm(X4,y4,C=0.5)
res5 = pset04.my_primal_svm(X1,y1,lam=0.5,k=5)
res6 = pset04.my_primal_svm(X2,y2,lam=0.5,k=5)
res7 = pset04.my_primal_svm(X3,y3,lam=0.25,k=5)
res8 = pset04.my_primal_svm(X4,y4,lam=1.0,k=1)


# check results (you should get no errors when testing, but
# won't get 'good' results as these are not 'real' answers)
sum(abs(ans1 - res1))
sum(abs(ans2 - res2))
sum(abs(ans3 - res3))
sum(abs(ans4 - res4))
sum(abs(ans5 - res5))
sum(abs(ans6 - res6))
sum(abs(ans7 - res7))
sum(abs(ans8 - res8))
