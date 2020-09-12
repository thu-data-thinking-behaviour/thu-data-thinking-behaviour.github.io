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

outLm <- lm(y ~ x1 + x2, data=df)
outGlm <- glm(cl ~ x1 + x2, data=df, family="binomial")
outGlm2 <- glm(cl ~ poly(x1,3) + poly(x2,3), data=df, family="binomial")

outSvm <- svm(X,  factor(cl), kernel="linear", scale=FALSE, cost=1)
outSvm2 <- svm(X, factor(cl), kernel="polynomial", scale=FALSE, cost=1)

w <- t(outSvm$coefs) %*% outSvm$SV
b <- -outSvm$rho


pdf("img/fig01.pdf", height=7, width=7)
par(mar=c(0,0,0,0))
plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
points(grid, pch=19, cex=0.33, col=grey(0.2,0.3))
points(X, col=cols, cex=1, lwd=3)
dev.off()

pdf("img/fig02.pdf", height=7, width=7)
par(mar=c(0,0,0,0))
plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
vals <- as.numeric(predict(outLm, gridDf) > 0)
points(grid, pch=19, cex=0.33, col=colVals[vals + 1L])
points(X, col=cols, cex=1, lwd=3)
abline(-1 * outLm$coef[1] / outLm$coef[3], -1 * outLm$coef[2] / outLm$coef[3],
         col="#6E3179", lty="dashed", lwd=1.5)
dev.off()

pdf("img/fig03.pdf", height=7, width=7)
par(mar=c(0,0,0,0))
plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
vals <- as.numeric(predict(outGlm, gridDf) > 0)
points(grid, pch=19, cex=0.33, col=colVals[vals + 1L])
points(X, col=cols, cex=1, lwd=3)
abline(-1 * outLm$coef[1] / outLm$coef[3], -1 * outLm$coef[2] / outLm$coef[3],
         col="#000000", lty="dashed", lwd=1.5)
abline(-1 * outGlm$coef[1] / outGlm$coef[3], -1 * outGlm$coef[2] / outGlm$coef[3],
         col="#6E3179", lty="dashed", lwd=1.5)
dev.off()

pdf("img/fig04.pdf", height=7, width=7)
par(mar=c(0,0,0,0))
plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
vals <- as.numeric(predict(outGlm, gridDf) > 0)
points(grid, pch=19, cex=0.33, col=colVals[vals + 1L])
points(X, col=cols, cex=1, lwd=3)
abline(-1 * outLm$coef[1] / outLm$coef[3], -1 * outLm$coef[2] / outLm$coef[3],
         col="#000000", lty="dashed", lwd=1.5)
abline(-1 * outGlm$coef[1] / outGlm$coef[3], -1 * outGlm$coef[2] / outGlm$coef[3],
         col="#6E3179", lty="dashed", lwd=1.5)
dev.off()

pdf("img/fig04.pdf", height=7, width=7)
par(mar=c(0,0,0,0))
plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
vals <- as.numeric(predict(outGlm2, gridDf) > 0)
points(grid, pch=19, cex=0.33, col=colVals[vals + 1L])
points(X, col=cols, cex=1, lwd=3)
abline(-1 * outLm$coef[1] / outLm$coef[3], -1 * outLm$coef[2] / outLm$coef[3],
         col="#000000", lty="dashed", lwd=1.5)
abline(-1 * outGlm$coef[1] / outGlm$coef[3], -1 * outGlm$coef[2] / outGlm$coef[3],
         col="#000000", lty="dashed", lwd=1.5)
dev.off()

pdf("img/fig05.pdf", height=7, width=7)
par(mar=c(0,0,0,0))
plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
vals <- (as.numeric(predict(outSvm, grid)) - 1L)
points(grid, pch=19, cex=0.33, col=colVals[vals + 1L])
points(X, col=cols, cex=1, lwd=3)
abline(-1 * outLm$coef[1] / outLm$coef[3], -1 * outLm$coef[2] / outLm$coef[3],
         col="#000000", lty="dashed", lwd=1.5)
abline(-1 * outGlm$coef[1] / outGlm$coef[3], -1 * outGlm$coef[2] / outGlm$coef[3],
         col="#000000", lty="dashed", lwd=1.5)
abline(a = -b/w[1,2], b=-w[1,1]/w[1,2], col="#6E3179", lty="dashed", lwd=1.5)
dev.off()

pdf("img/fig06.pdf", height=7, width=7)
par(mar=c(0,0,0,0))
plot(0,0, col="white", axes=FALSE, xlab="", ylab="", main="", ylim=c(0,1), xlim=c(0,1))
vals <- (as.numeric(predict(outSvm2, grid)) - 1L)
points(grid, pch=19, cex=0.33, col=colVals[vals + 1L])
points(X, col=cols, cex=1, lwd=3)
abline(-1 * outLm$coef[1] / outLm$coef[3], -1 * outLm$coef[2] / outLm$coef[3],
         col="#000000", lty="dashed", lwd=1.5)
abline(-1 * outGlm$coef[1] / outGlm$coef[3], -1 * outGlm$coef[2] / outGlm$coef[3],
         col="#000000", lty="dashed", lwd=1.5)
abline(a = -b/w[1,2], b=-w[1,1]/w[1,2], col="#000000", lty="dashed", lwd=1.5)
dev.off()
