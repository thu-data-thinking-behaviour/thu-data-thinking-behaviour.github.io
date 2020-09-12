#!/usr/bin/env Rscript

# This script must be run in the same directory as:
#   pset04.R

e <-new.env()
source("pset04.R",local=e)

# test data
X1 <- matrix(rnorm(4*97), ncol=4)
y1 <- sign(X1[,1] + X1[,2] > 0)*2 - 1
X2 <- matrix(rnorm(8*97), ncol=8)
y2 <- sign(X2[,1] + X2[,2] > 0)*2 - 1
X3 <- matrix(rnorm(8*97), ncol=8)
y3 <- sign(X3[,1] + X3[,2] > 0)*2 - 1
X4 <- matrix(rnorm(10*97), ncol=10)
y4 <- sign(X4[,1] + X4[,2] > 0)*2 - 1

# these are 'fake' answers, that simply have the correct
# format; they will be replaced with real answers when we
# grade them
ans1 <- runif(4)
ans2 <- runif(8)
ans3 <- runif(8)
ans4 <- runif(10)

# check results (you should get no errors when testing, but
# won't get 'good' results as these are not 'real' answers)
max(abs(e$my_dual_svm(X1, y1, C=2) - ans1))
max(abs(e$my_dual_svm(X2, y2, C=5) - ans2))
max(abs(e$my_dual_svm(X3, y3, C=10) - ans3))
max(abs(e$my_dual_svm(X4, y4, C=0.5) - ans4))

max(abs(e$my_primal_svm(X1, y1, lam=0.5, k=5) - ans1))
max(abs(e$my_primal_svm(X2, y2, lam=0.5, k=5) - ans2))
max(abs(e$my_primal_svm(X3, y3, lam=0.25, k=5) - ans3))
max(abs(e$my_primal_svm(X4, y4, lam=1, k=1) - ans4))






