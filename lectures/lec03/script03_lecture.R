set.seed(1)
n <- 200
x <- sort(runif(n, 0, 4))
ybar <- sin(x) + x^2*0.3 + sin(x*2) + sin(x*4)
y <- ybar + rnorm(n,sd=0.5)
ybar2 <- sin(x) + x^2*0.3 + sin(x*2) + sin(x*4)
y2 <- ybar2 + rnorm(n,sd=0.5)

# K nearest neighbors
system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1920, 1080, quality=100)
par(mar=c(8,8,5,5))

k <- 25
for(i in 1:25) {
  plot(x, y, pch=19, cex=0.8, col="white", ylim=c(-1,6),axes=FALSE,xlab="",ylab="")
  box()
  text(1,3,sprintf("knn  k=%d",k),cex=4)
}

D <- as.matrix(dist(x))
index <- apply(D, 1, function(v) order(v)[1:k] )
yhat <- apply(matrix(y[index],ncol=k,byrow=TRUE),1,mean)

for(i in 1:n) {
  plot(x, y, pch=19, cex=0.8, col=rgb(0,0,0,0.3), ylim=c(-1,6),axes=FALSE)
  box()
  axis(1)
  axis(2,at=0:6)
  points(x[i], y[i], pch=19, cex=1.2, col=rgb(0,0,1))
  lines(x[1:i],yhat[1:i], cex=1, col=rgb(1,0.64,0,1),lwd=1.6)
  weights <- rep(0,n)
  weights[rank(abs(x-x[i])) < k] <- 1
  points(x,weights/2-1,col=rgb(1,0.64,0,0.7),pch=19,cex=0.8)
}

# Kernel smoothing
sigma <- 0.2
for(i in 1:25) {
  plot(x, y, pch=19, cex=0.8, col="white", ylim=c(-1,6),axes=FALSE,xlab="",ylab="")
  box()
  text(1,3,sprintf("kernel smoother    sigma=%01.02f",sigma),cex=4)
}

D <- as.matrix(dist(x))
weights <- dnorm(D, sd=sigma)
ymat <- matrix(rep(y,n),n,n)
yhat <- apply(ymat * weights,2,sum) / apply(weights,2,sum)

for(i in 1:n) {
  plot(x, y, pch=19, cex=0.8, col=rgb(0,0,0,0.3), ylim=c(-1,6),axes=FALSE)
  box()
  axis(1)
  axis(2,at=0:6)
  points(x[i], y[i], pch=19, cex=1.2, col=rgb(0,0,1))
  lines(x[1:i],yhat[1:i], cex=1, col=rgb(1,0.64,0,1),lwd=1.6)
  points(x,weights[i,]/2-1,col=rgb(1,0.64,0,0.7),pch=19,cex=0.8)
}

# Linear regression
p <- 4
for(i in 1:25) {
  plot(x, y, pch=19, cex=0.8, col="white", ylim=c(-1,6),axes=FALSE,xlab="",ylab="")
  box()
  text(1,3,sprintf("linear regression    p=%02d",p),cex=4)
}

X <- t(sapply(split(x,1:n),function(v) v^(0:p)))
out <- lm(y ~ X - 1)
yhat <- predict(out)
weights <- X %*% solve(crossprod(X)) %*% t(X)

for(i in 1:n) {
  plot(x, y, pch=19, cex=0.8, col=rgb(0,0,0,0.3), ylim=c(-1,6),axes=FALSE)
  box()
  axis(1)
  axis(2,at=0:6)
  points(x[i], y[i], pch=19, cex=1.2, col=rgb(0,0,1))
  lines(x[1:i],yhat[1:i], cex=1, col=rgb(1,0.64,0,1),lwd=1.6)
  points(x,weights[i,]/max(abs(weights[i,]))*0.5-1,col=rgb(1,0.64,0,0.7),pch=19,cex=0.8)
  abline(h=-1,lty="dashed")
}

# lowess
for(i in 1:25) {
  plot(x, y, pch=19, cex=0.8, col="white", ylim=c(-1,6),axes=FALSE,xlab="",ylab="")
  box()
  text(1,3,"lowess",cex=4)
}

D <- as.matrix(dist(x))
weights <- dnorm(D, sd=sigma)
X <- cbind(1,x,x^2,x^3)

yhat <- rep(NA,n)
for(i in 1:n) {
  plot(x, y, pch=19, cex=0.8, col=rgb(0,0,0,0.3), ylim=c(-1,6),axes=FALSE)
  box()
  axis(1)
  axis(2,at=0:6)
  points(x[i], y[i], pch=19, cex=1.2, col=rgb(0,0,1))
  out <- lm(y ~ X - 1, weights=weights[i,])
  yhat[i] <- out$fitted.values[i]
  w <- (X %*% solve(t(X) %*% diag(weights[i,]) %*% X) %*% t(X) %*% diag(weights[i,]))
  lines(x[1:i],yhat[1:i], cex=1, col=rgb(1,0.64,0,1),lwd=1.6)
  points(x,w[i,]/max(abs(w[i,]))*0.5-1,col=rgb(1,0.64,0,0.7),pch=19,cex=0.8)
  abline(h=-1,lty="dashed")
}

dev.off()
system("rm media/linearSmoother.mp4")
system("ffmpeg -r 8 -i ani/foo%04d.jpg -r 24 media/linearSmoother.mp4")

####################
set.seed(1)
n <- 200
x <- sort(runif(n, 0, 4))
ybar <- sin(x) + x^2*0.3 + sin(x*2) + sin(x*4)
y <- ybar + rnorm(n,sd=0.5)
ybar2 <- sin(x) + x^2*0.3 + sin(x*2) + sin(x*4)
y2 <- ybar2 + rnorm(n,sd=0.5)


these <- (runif(n) < 1/3)

k <- 5
D <- as.matrix(dist(x))
D[,these] <- Inf
index <- apply(D, 1, function(v) order(v)[1:k] )
yhat <- apply(matrix(y[index],ncol=k,byrow=TRUE),1,mean)

cols <- rep(rgb(1,0.64,0),n)
cols[these] <- rgb(0.36, 0.54, 0.54)

pdf("img/valid1.pdf", height=5.5, width=10)
par(mar=c(4,4,1,1))
plot(x, y, pch=19, cex=0.8, col=rgb(0,0,0,0.4), ylim=c(0,6),axes=FALSE)
box()
axis(1)
axis(2,at=0:6)
dev.off()

pdf("img/valid2.pdf", height=5.5, width=10)
par(mar=c(4,4,1,1))
plot(x, y, pch=19, cex=0.8, col=cols, ylim=c(0,6),axes=FALSE)
box()
axis(1)
axis(2,at=0:6)
points(x[!these], y[!these], pch=19, cex=0.6, col=rgb(1,0.64,0))
legend(0.5,5,c("training","validation"),col=c(rgb(1,0.64,0),rgb(0.36, 0.54, 0.54)),pch=19)
dev.off()

pdf("img/valid3.pdf", height=5.5, width=10)
cols <- rep(rgb(1,0.64,0),n)
cols[these] <- rgb(0,0,0,0.3)
par(mar=c(4,4,1,1))
plot(x, y, pch=19, cex=0.8, col=cols, ylim=c(0,6),axes=FALSE)
box()
axis(1)
axis(2,at=0:6)
points(x[!these], y[!these], pch=19, cex=0.6, col=rgb(1,0.64,0))
dev.off()

pdf("img/valid4.pdf", height=5.5, width=10)
cols <- rep(rgb(1,0.64,0),n)
cols[these] <- rgb(0,0,0,0.3)
par(mar=c(4,4,1,1))
plot(x, y, pch=19, cex=0.8, col=cols, ylim=c(0,6),axes=FALSE)
box()
axis(1)
axis(2,at=0:6)
points(x[!these], y[!these], pch=19, cex=0.6, col=rgb(1,0.64,0))
lines(x,yhat, cex=1, col=rgb(1,0.64,0,1),lwd=1.6)
dev.off()

pdf("img/valid5.pdf", height=5.5, width=10)
par(mar=c(4,4,1,1))
plot(x, y, pch=19, cex=0.8, col=rgb(0,0,0,0.3), ylim=c(0,6),axes=FALSE)
box()
axis(1)
axis(2,at=0:6)
lines(x,yhat, cex=1, col=rgb(1,0.64,0,1),lwd=1.6)
dev.off()

pdf("img/valid6.pdf", height=5.5, width=10)
cols <- rep(rgb(0,0,0,0.2),n)
cols[these] <-rgb(0.36, 0.54, 0.54)
par(mar=c(4,4,1,1))
plot(x, y, pch=19, cex=0.8, col=cols, ylim=c(0,6),axes=FALSE)
box()
axis(1)
axis(2,at=0:6)
lines(x,yhat, cex=1, col=rgb(1,0.64,0,1),lwd=1.6)
dev.off()

pdf("img/valid7.pdf", height=5.5, width=10)
cols <- rep(rgb(0,0,0,0.2),n)
cols[these] <-rgb(0.36, 0.54, 0.54)
par(mar=c(4,4,1,1))
plot(x, y, pch=19, cex=0.8, col=cols, ylim=c(0,6),axes=FALSE)
box()
axis(1)
axis(2,at=0:6)
lines(x,yhat, cex=1, col=rgb(1,0.64,0,1),lwd=1.6)
segments(x[these], y[these], x[these], yhat[these], col=rgb(0.36, 0.54, 0.54))
dev.off()

####################
system("mkdir -p ani")
system("rm ani/*")
jpeg("ani/foo%04d.jpg", 1920, 1080, quality=100)
par(mar=c(8,8,5,5))

D <- as.matrix(dist(x))
D[,these] <- Inf
cols <- rep(rgb(0,0,0,0.2),n)
cols[these] <-rgb(0.36, 0.54, 0.54)

for (k in 1:sum(!these)) {
  index <- apply(D, 1, function(v) order(v)[1:k] )
  yhat <- apply(matrix(y[index],ncol=k,byrow=TRUE),1,mean)
  err <- mean((yhat - y)[these]^2)
  plot(x, y, pch=19, cex=0.8, col=cols, ylim=c(-1,6),axes=FALSE)
  box()
  axis(1)
  axis(2,at=0:6)
  lines(x,yhat, cex=1, col=rgb(1,0.64,0,1),lwd=1.6)
  segments(x[these], y[these], x[these], yhat[these], col=rgb(0.36, 0.54, 0.54))
  text(0.5,5,sprintf("k   = %03d\n\nmse = %01.04f",k,err), cex=3)
}

dev.off()
system("rm media/validation.mp4")
system("ffmpeg -r 8 -i ani/foo%04d.jpg -r 24 media/validation.mp4")


####################
these <- (runif(n) < 1/3)

D <- as.matrix(dist(x))
D[,these] <- Inf

err <- rep(NA, sum(!these))
errT <- rep(NA, sum(!these))
for (k in length(err):1) {
  index <- apply(D, 1, function(v) order(v)[1:k] )
  yhat <- apply(matrix(y[index],ncol=k,byrow=TRUE),1,mean)
  err[k] <- mean((yhat - y)[these]^2)
  errT[k] <- mean((yhat - y)[!these]^2)
}
kVals <- length(err):1

pdf("img/knnValid.pdf", height=5.5, width=10)
plot(kVals,errT,xlab="complexity",ylab="mse",col=rgb(1,0.64,0),
      ylim=range(c(errT,err)),cex=0.8,pch=19,axes=FALSE)
box()
axis(2)
axis(1,at=c(20,100),label=c("low","high"))
points(kVals,err, col=rgb(0.36, 0.54, 0.54),cex=0.8,pch=19)
legend(20,0.5,c("training","validation"),col=c(rgb(1,0.64,0),rgb(0.36, 0.54, 0.54)),pch=19)
dev.off()

##
fVals <- seq(0.08, 1, length.out=100)
err <- rep(NA, length(fVals))
errT <- rep(NA, length(fVals))

for (j in 1:length(fVals)) {
  f <- fVals[j]
  plot(x, y, pch=19, cex=1, col=rgb(0,0,1,0.6))
  out <- loess(y ~ x,span=f,weights=as.numeric(!these))
  yhat <- out$fitted
  err[j] <- mean((yhat - y)[these]^2)
  errT[j] <- mean((yhat - y)[!these]^2)
}

pdf("img/loessValid.pdf", height=5.5, width=10)
plot(rev(fVals),errT,xlab="complexity",ylab="mse",col=rgb(1,0.64,0),
      ylim=range(c(errT,err)),cex=0.8,pch=19,axes=FALSE)
box()
axis(2)
axis(1,at=c(0.2,0.8),label=c("low","high"))
points(rev(fVals),err, col=rgb(0.36, 0.54, 0.54),cex=0.8,pch=19)
legend(0.2,0.4,c("training","validation"),col=c(rgb(1,0.64,0),rgb(0.36, 0.54, 0.54)),pch=19)
dev.off()

##
fold <- sample(rep(1:5,200/5))

fVals <- seq(0.08, 1, length.out=100)
fVals <- rev(fVals)
err <- matrix(NA, nrow=length(fVals), ncol=5)

for (id in 1:5) {
  these <- (fold == id)
  for (j in 1:length(fVals)) {
    f <- fVals[j]
    out <- loess(y ~ x,span=f,weights=as.numeric(!these))
    yhat <- out$fitted
    err[j,id] <- mean((yhat - y)[these]^2)
  }
}

se <- apply(err,1,sd) / sqrt(5)
mu <- apply(err,1,mean)
minrule <- which.min(mu)
se1rule <- max(which((mu - se < min(mu))))
index <- which((mu - se < min(mu)))

pdf("img/cv1.pdf", height=5.5, width=10)
plot(row(err),err,xlab="complexity",ylab="mse",pch=19,cex=0.7,axes=FALSE)
box()
axis(2)
axis(1,at=c(20,80),label=c("low","high"))
dev.off()

pdf("img/cv2.pdf", height=5.5, width=10)
plot(1:nrow(err),mu,xlab="complexity",ylab="mse",ylim=range(c(mu - se,mu + se)),pch=19,cex=0.7,axes=FALSE)
box()
axis(2)
axis(1,at=c(20,80),label=c("low","high"))
dev.off()

pdf("img/cv3.pdf", height=5.5, width=10)
plot(1:nrow(err),mu,xlab="complexity",ylab="mse",ylim=range(c(mu - se,mu + se)),pch=19,cex=0.7,axes=FALSE)
box()
axis(2)
axis(1,at=c(20,80),label=c("low","high"))
segments(1:nrow(err), mu - se, 1:nrow(err), mu + se, col=rgb(0.36, 0.54, 0.54))
dev.off()

pdf("img/cv4.pdf", height=5.5, width=10)
plot(1:nrow(err),mu,xlab="complexity",ylab="mse",ylim=range(c(mu - se,mu + se)),pch=19,cex=0.7,axes=FALSE)
box()
axis(2)
axis(1,at=c(20,80),label=c("low","high"))
segments(1:nrow(err), mu - se, 1:nrow(err), mu + se, col=rgb(0.36, 0.54, 0.54))
points(index,mu[index],col="red",pch=19,cex=0.7)
abline(h=min(mu), lty="dashed", col="red")
dev.off()


