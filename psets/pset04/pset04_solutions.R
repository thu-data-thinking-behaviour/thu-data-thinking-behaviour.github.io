# Construct a test dataset:
set.seed(1)
n <- 200
X <- rbind(MASS::mvrnorm(n/2,c(0.25,0.75),
            matrix(c(1,0.5,0.5,1)*0.1,ncol=2,nrow=2)),
           MASS::mvrnorm(n/2,c(0.75,0.25),
            matrix(c(1,0.2,0.2,2)*0.1,ncol=2,nrow=2)))
X <- cbind(X, rnorm(n), rnorm(n))
X[,1:2] <- X[,1:2]*0.97 + 0.015
y <- c(rep(1,n/2),rep(-1,n/2))

# dual solution:
my_dual_svm <- function(X, y, C=1L) {
  K <- 0.5 * tcrossprod(X) * tcrossprod(y)
  fn <- function(alpha) {sum(alpha) - alpha %*% crossprod(K, alpha)}
  gr <- function(alpha) {1 - 2 * K %*% alpha}

  alpha <- optim(rep(C/2, n),fn,gr,
                 lower=0,upper=C,method="L-BFGS-B",
                 control=list(fnscale=-1))$par

  apply(out$par * y * X, 2, sum)
}

# primal solution
my_primal_svm <- function(X, y, lam=1, k=5, T=100) {
  w <- runif(ncol(X)) / (1/sqrt(lambda))

  for (t in 1:T) {
    index <- sample(1:n,k)
    Aplus <- which( y[index] * (tcrossprod(w,X[index,])) < 1 )
    eta <- 1 / (lambda * t)
    whalf <- (1 - eta * lambda) * w + (eta / k) * apply(y[index[Aplus]] * X[index[Aplus],,drop=FALSE],2,sum)
    w <- min(c(1, 1/sqrt(lambda) / sqrt(sum(w^2)))) * whalf
  }

  w
}

# run both models (note that the scales are different and will not match)
betaDual <- my_dual_svm(X, y, C=10)
betaPrime <- my_primal_svm(X, y, lam=1/20, k=10, T=500)

# similar predictions, though?
table(sign(X %*% betaDual),sign(X %*% betaPrime))



