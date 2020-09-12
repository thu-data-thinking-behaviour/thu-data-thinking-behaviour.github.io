set.seed(1)
n <- 200
x <- sort(runif(n, 0, 4))
ybar <- sin(x) + x^2*0.3 + sin(x*2) + sin(x*4)
y <- ybar + rnorm(n,sd=0.5)
ybar2 <- sin(x) + x^2*0.3 + sin(x*2) + sin(x*4)
y2 <- ybar2 + rnorm(n,sd=0.5)

###############
pdf("img/scatter.pdf", height=5.5, width=10)
par(mar=c(4,4,0,0))
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
dev.off()

pdf("img/knn1.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(v=1,lwd=2,lty="dashed")
dev.off()

pdf("img/knn2.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(v=1,lwd=2,lty="dashed")
index <- which(rank(abs(x-1)) < 15)
points(x[index], y[index], pch=19, cex=0.7, col=rgb(1,0.64,0,1))
dev.off()

pdf("img/knn3.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(v=1,lwd=2,lty="dashed")
index <- which(rank(abs(x-1)) < 15)
points(x[index], y[index], pch=19, cex=0.7, col=rgb(1,0.64,0,1))
lines(x[range(index)],rep(mean(y[index]),2), col=rgb(1,0.64,0,1), lwd=2)
points(1, mean(y[index]), pch=19, cex=1, col=rgb(1,0.64,0,1))
dev.off()

pdf("img/knn4.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(v=1,lwd=2,lty="dashed")
index <- which(rank(abs(x-1)) < 15)
points(1, mean(y[index]), pch=19, cex=2, col=rgb(1,0.64,0,1))
dev.off()


pdf("img/ksmooth1.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(v=1,lwd=2,lty="dashed")
dev.off()

pdf("img/ksmooth2.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(v=1,lwd=2,lty="dashed")
weights <- dnorm(abs(x-1), sd=0.1)
points(x,weights/2)
dev.off()

pdf("img/ksmooth3.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(v=1,lwd=2,lty="dashed")
points(1,sum(weights*y)/sum(weights),pch=19, cex=2, col=rgb(1,0.64,0,1))
dev.off()

pdf("img/olsSimple.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(lm(y ~ x), col=rgb(1,0.64,0,1), lwd=2)
dev.off()

pdf("img/lowess1.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(v=1,lwd=2,lty="dashed")
dev.off()

pdf("img/lowess2.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(v=1,lwd=2,lty="dashed")
weights <- dnorm(abs(x-1), sd=0.1)
out <- lm(y ~ x,weights=weights)
abline(out, col=rgb(1,0.64,0,1), lwd=2)
dev.off()

pdf("img/lowess3.pdf", height=5.5, width=10)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.5), xlab="", ylab="")
abline(v=1,lwd=2,lty="dashed")
weights <- dnorm(abs(x-1), sd=0.1)
out <- lm(y ~ x,weights=weights)
abline(out, col=rgb(1,0.64,0,1), lwd=2)
points(1,predict(out, data.frame(x=1)),pch=19, cex=2, col=rgb(1,0.64,0,1))
dev.off()


# K nearest neighbors
system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1920, 1080, quality=100)
par(mar=c(8,8,5,5))

kVals <- unique(round(exp(seq(log(1),log(n),length.out=200))))
kVals <- c(kVals,rev(kVals))

for(i in 1:length(kVals)) {
  k <- kVals[i]
  D <- as.matrix(dist(x))
  index <- apply(D, 1, function(v) order(v)[1:k] )
  yhat <- apply(matrix(y[index],ncol=k,byrow=TRUE),1,mean)

  plot(x, y, pch=19, cex=1, col=rgb(0,0,1,0.6))
  lines(x, yhat, col="orange", lwd=4)
  text(0.5,5,sprintf("k = %03d",k), cex=3)
}
dev.off()

system("rm media/knnEst.mp4")
system("ffmpeg -r 12 -i ani/foo%04d.jpg -r 24 media/knnEst.mp4")

# Kernel smoothing
system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1920, 1080, quality=100)
par(mar=c(8,8,5,5))

sigmaVals <- exp(seq(log(0.00001),log(10),length.out=90))
sigmaVals <- c(sigmaVals,rev(sigmaVals))
for (j in 1:length(sigmaVals)) {
  sigma <- sigmaVals[j]
  D <- as.matrix(dist(x))
  weights <- dnorm(D, sd=sigma)
  ymat <- matrix(rep(y,n),n,n)
  yhat <- apply(ymat * weights,2,sum) / apply(weights,2,sum)
  plot(x, y, pch=19, cex=1, col=rgb(0,0,1,0.6))
  lines(x, yhat, col="orange", lwd=4)
  text(0.5,5,sprintf("sigma = %01.05f",sigma),cex=3)
}
dev.off()

system("rm media/ksmoothEst.mp4")
system("ffmpeg -r 12 -i ani/foo%04d.jpg -r 24 media/ksmoothEst.mp4")

# Linear regression
system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1920, 1080, quality=100)
par(mar=c(8,8,5,5))

pVals <- unique(round(exp(seq(log(1),log(n),length.out=200))))
pVals <- c(pVals,rev(pVals))

for(i in 1:length(pVals)) {
  p <- pVals[i]
  X <- t(sapply(split(x,1:n),function(v) c(sin(v*1:p/(2*pi)),cos(v*1:p/(2*pi)))))
  out <- lm(y ~ X)
  yhat <- predict(out)
  plot(x, y, pch=19, cex=1, col=rgb(0,0,1,0.6))
  lines(x, yhat, col="orange", lwd=4)
  text(0.5,5,sprintf("p = %02d",p),cex=3)
}
dev.off()

system("rm media/regEst.mp4")
system("ffmpeg -r 8 -i ani/foo%04d.jpg -r 24 media/regEst.mp4")

# lowess
system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1920, 1080, quality=100)
par(mar=c(8,8,5,5))

fVals <- seq(0.001, 1, length.out=100)
fVals <- c(fVals,rev(fVals))
for (j in 1:length(fVals)) {
  f <- fVals[j]
  plot(x, y, pch=19, cex=1, col=rgb(0,0,1,0.6))
  out <- lowess(x=x,y=y,f=f)
  lines(out$x, out$y, col="orange", lwd=4)
  text(0.5,5,sprintf("f = %01.05f",f),cex=3)
}
dev.off()

system("rm media/loessEst.mp4")
system("ffmpeg -r 12 -i ani/foo%04d.jpg -r 24 media/loessEst.mp4")









p <- 10
out <- lm(y ~ poly(x,p-1,raw=TRUE))
yhat <- predict(out)
plot(x, y, pch=19, cex=0.7, col=rgb(0,0,1,0.3))
lines(x, yhat, col="orange", lwd=4)
text(0.5,5,sprintf("p = %02d",p))
cor(y, yhat)


# Error rate, knn
D <- as.matrix(dist(x))
vals <- rep(NA, 50)
coVals <- rep(NA, 50)
for (j in 1:length(vals)) {
  k <- j
  index <- apply(D, 1, function(v) order(v)[1:k] )
  yhat <- apply(matrix(y[index],ncol=k,byrow=TRUE),1,mean)
  vals[j] <- mean((yhat - y2)^2)
  coVals[j] <- cor(yhat, y)
}
plot(1:length(vals), vals)
plot(1:length(vals), coVals)


# Error rate, kernel smoothing
D <- as.matrix(dist(x))
vals <- rep(NA, 50)
coVals <- rep(NA, 50)
sigmaVals <- exp(seq(log(0.00001),log(1),length.out=length(vals)))
for (j in 1:length(vals)) {
  sigma <- sigmaVals[j]
  weights <- dnorm(D, sd=sigma)
  yhat <- apply(ymat * weights,2,sum) / apply(weights,2,sum)
  vals[j] <- mean((yhat - y2)^2)
  coVals[j] <- cor(yhat, y)
}
plot(sigmaVals, vals, log="x")

# Error rate, regression
D <- as.matrix(dist(x))
vals <- rep(NA, 100)
for (j in 1:length(vals)) {
  p <- j
  out <- lm(y ~ poly(x,p,raw=TRUE))
  yhat <- predict(out)
  vals[j] <- mean((yhat - y)^2)
}
plot(1:length(vals), vals)


p <- 10






# Weights for regression
X <- cbind(1,x,x^2)
hat <- X %*% solve(t(X) %*% X) %*% t(X)

i <- 850
plot(x, hat[i,],pch=19,cex=0.5)
abline(v=x[i])



