##############
library(e1071)
library(MASS)

n <- 200
x <- rnorm(n,0,0.8)
y <- rnorm(n,0,0.8)
cl <- c(rep(0,n/2),rep(1,n/2))
colVals <- c("#E1B141",  "#72ABC6")
cols <- rep(colVals[1], n)
cols[cl == 1] <-colVals[2]
grid <- expand.grid(seq(0,1,0.01),seq(0,1,0.01))

set.seed(1)
X <- rbind(mvrnorm(n/2,c(0.25,0.75),matrix(c(1,0.5,0.5,1)*0.1,ncol=2,nrow=2)),
           mvrnorm(n/2,c(0.75,0.25),matrix(c(1,0.2,0.2,2)*0.1,ncol=2,nrow=2)))
X[,1] <- (X[,1] - min(X[,1])) / diff(range(X[,1]))
X[,2] <- (X[,2] - min(X[,2])) / diff(range(X[,2]))
X[,1] <- X[,1]*0.97 + 0.015
X[,2] <- X[,2]*0.97 + 0.015
y <- cl*2 - 1

gridDf <- data.frame(x1 = grid[,1], x2 = grid[,2])
df <- data.frame(y = y, cl = cl, x1 = X[,1], x2 = X[,2])

outSvm <- svm(X,  factor(cl), kernel="linear", scale=FALSE, cost=1)
outSvm2 <- svm(X, factor(cl), kernel="polynomial", scale=FALSE, cost=1)

system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1080, 1080, quality=100)

costs <- exp(seq(log(0.0001), log(1e6), length.out=200))
for (i in 1:length(costs)) {
  out <- svm(X, factor(cl), kernel="polynomial", scale=FALSE, cost=costs[i])
  vals <- (as.numeric(predict(out, grid)) - 1L)
  w <- t(out$coefs) %*% out$SV
  b <- -out$rho
  par(mar=c(0,0,0,0))
  plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
  points(grid, pch=19, cex=0.2, col=colVals[vals + 1L])
  points(X, col=cols, cex=1, lwd=3)
  #abline(a = -b/w[1,2], b=-w[1,1]/w[1,2], col="#6E3179", lty="dashed", lwd=1.5)
}

system("rm media/svmCostsRadial.mp4")
system("ffmpeg -r 12 -i ani/foo%04d.jpg -r 24 media/svmCostsLinear.mp4")


##################
##################
library(gbm)
library(randomForest)
library(e1071)
library(glmnet)
library(FNN)

train <- read.csv("../../data/mnist_train.psv", sep="|", header=FALSE)
test  <- read.csv("../../data/mnist_test.psv", sep="|", header=FALSE)
validFlag <- (runif(nrow(train)) > 0.8)
Xtrain <- as.matrix(train[!validFlag,-1])
Xvalid <- as.matrix(train[validFlag,-1])
Xtest <- as.matrix(test[,-1])
ytrain <- train[!validFlag,1]
yvalid <- train[validFlag,1]
ytest <- test[,1]

# Regularized regression:
outLm <- cv.glmnet(Xtrain, ytrain, alpha=0, nfolds=3,
                    family="multinomial")
predLmV <- apply(predict(outLm, Xvalid, s=outLm$lambda.min,
                  type="response"), 1, which.max) - 1L

# Random forest:
outRf <- randomForest(Xtrain,  factor(ytrain), maxnodes=10, do.trace=TRUE)
predRfV <- predict(outRf, Xvalid)

# Gradient boosted trees:
outGbm <- gbm.fit(Xtrain,  factor(ytrain), distribution="multinomial",
                  n.trees=500)
predGbmV <- apply(predict(outGbm, Xvalid, n.trees=outGbm$n.trees),1,which.max) - 1L

# Support vector machines:
outSvm <- svm(Xtrain,  factor(ytrain), kernel="radial", cost=1, probability = TRUE)
predSvmV <- predict(outSvm, Xvalid)

# Compare these:
mean(predLmV != yvalid)
mean(predRfV != yvalid)
mean(predGbmV != yvalid)
mean(predSvmV != yvalid)

# Stacking? How do we do this with predicted classes?
predLmV <- predict(outLm, Xvalid, s=outLm$lambda.1se, type="response")[,,1]
predRfV <- predict(outRf, Xvalid, type="prob")
predGbmV <- predict(outGbm, Xvalid, n.trees=outGbm$n.trees)[,,1]
predSvmV <- attributes(predict(outSvm, Xvalid, probability=TRUE))$probabilities
metaXvalid <- cbind(predLmV, predRfV, predGbmV, predSvmV)

predLmT <- predict(outLm, Xtest, s=outLm$lambda.1se, type="response")[,,1]
predRfT <- predict(outRf, Xtest, type="prob")
predGbmT <- predict(outGbm, Xtest, n.trees=outGbm$n.trees)[,,1]
predSvmT <- attributes(predict(outSvm, Xtest, probability=TRUE))$probabilities
metaXtest <- cbind(predLmT, predRfT, predGbmT, predSvmT)

outStack <- cv.glmnet(metaXvalid, yvalid, alpha=1, nfolds=3,
                    family="multinomial")
predStackT <- apply(predict(outStack, metaXtest, s=outStack$lambda.1se,
                  type="response"), 1, which.max) - 1L

mean(predStackT != ytest)

##############
library(e1071)
library(MASS)

load("../../data/ESL.mixture.rda")
n <- 200
X <- ESL.mixture$x
cl <- ESL.mixture$y
colVals <- c("#E1B141",  "#72ABC6")
cols <- rep(colVals[1], n)
cols[cl == 1] <-colVals[2]
grid <- expand.grid(seq(0,1,0.01),seq(0,1,0.01))

X[,1] <- (X[,1] - min(X[,1])) / diff(range(X[,1]))
X[,2] <- (X[,2] - min(X[,2])) / diff(range(X[,2]))
X[,1] <- X[,1]*0.97 + 0.015
X[,2] <- X[,2]*0.97 + 0.015
y <- cl*2 - 1

gridDf <- data.frame(x1 = grid[,1], x2 = grid[,2])
df <- data.frame(y = y, cl = cl, x1 = X[,1], x2 = X[,2])

outSvm <- svm(X,  factor(cl), kernel="linear", scale=FALSE, cost=1)
outSvm2 <- svm(X, factor(cl), kernel="polynomial", scale=FALSE, cost=1)

system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1080, 1080, quality=100)

costs <- exp(seq(log(0.0001), log(1e6), length.out=100))
for (i in 1:length(costs)) {
  out <- svm(X, factor(cl), kernel="linear", scale=FALSE, cost=costs[i])
  vals <- (as.numeric(predict(out, grid)) - 1L)
  w <- t(out$coefs) %*% out$SV
  b <- -out$rho
  par(mar=c(0,0,0,0))
  plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
  points(grid, pch=19, cex=0.8, col=colVals[vals + 1L])
  points(X, col=cols, cex=2.5, lwd=3)
  #abline(a = -b/w[1,2], b=-w[1,1]/w[1,2], col="#6E3179", lty="dashed", lwd=1.5)
}

system("rm media/svmCostsLinear.mp4")
system("ffmpeg -r 12 -i ani/foo%04d.jpg -r 24 media/svmCostsLinear.mp4")



system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1080, 1080, quality=100)

costs <- exp(seq(log(2), log(1e8), length.out=100))
for (i in 1:length(costs)) {
  out <- svm(X, factor(cl), kernel="radial", scale=FALSE, cost=costs[i])
  vals <- (as.numeric(predict(out, grid)) - 1L)
  w <- t(out$coefs) %*% out$SV
  b <- -out$rho
  par(mar=c(0,0,0,0))
  plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
  points(grid, pch=19, cex=0.8, col=colVals[vals + 1L])
  points(X, col=cols, cex=2.5, lwd=3)
  #abline(a = -b/w[1,2], b=-w[1,1]/w[1,2], col="#6E3179", lty="dashed", lwd=1.5)
}

system("rm media/svmCostsRadial.mp4")
system("ffmpeg -r 12 -i ani/foo%04d.jpg -r 24 media/svmCostsRadial.mp4")


system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1080, 1080, quality=100)

costs <- exp(seq(log(0.001), log(10000), length.out=100))
for (i in 1:length(costs)) {
  out <- svm(X, factor(cl), kernel="polynomial", scale=FALSE,
             cost=costs[i])
  vals <- (as.numeric(predict(out, grid)) - 1L)
  w <- t(out$coefs) %*% out$SV
  b <- -out$rho
  par(mar=c(0,0,0,0))
  plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
  points(grid, pch=19, cex=0.8, col=colVals[vals + 1L])
  points(X, col=cols, cex=2.5, lwd=3)
  #abline(a = -b/w[1,2], b=-w[1,1]/w[1,2], col="#6E3179", lty="dashed", lwd=1.5)
}

system("rm media/svmCostsPoly.mp4")
system("ffmpeg -r 12 -i ani/foo%04d.jpg -r 24 media/svmCostsPoly.mp4")

system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1080, 1080, quality=100)

costs <- exp(seq(log(0.001), log(10000), length.out=100))
for (i in 1:length(costs)) {
  out <- svm(X, factor(cl), kernel="sigmoid", scale=FALSE,
             cost=costs[i])
  vals <- (as.numeric(predict(out, grid)) - 1L)
  w <- t(out$coefs) %*% out$SV
  b <- -out$rho
  par(mar=c(0,0,0,0))
  plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
  points(grid, pch=19, cex=0.8, col=colVals[vals + 1L])
  points(X, col=cols, cex=2.5, lwd=3)
  #abline(a = -b/w[1,2], b=-w[1,1]/w[1,2], col="#6E3179", lty="dashed", lwd=1.5)
}

system("rm media/svmCostsSigmoid.mp4")
system("ffmpeg -r 12 -i ani/foo%04d.jpg -r 24 media/svmCostsSigmoid.mp4")


