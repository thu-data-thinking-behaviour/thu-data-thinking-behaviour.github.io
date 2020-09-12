#' Title: Decision trees
#' Date: 2016-02-13
#' Authors: Taylor Arnold

##############################
# -- Example 1 --
#   www.stat.yale.edu/~tba3/stat665/lectures/lec07/data/CAPA.csv
#   11275 observations, 34 variables
#   economic and housing details aggregated at tract level in CA & PA
#   docs: http://www.stat.cmu.edu/~cshalizi/uADA/16/hw/01/hw-01.pdf

# rmarkdown::render("script08.Rmd")

set.seed(1)
x <- read.csv("data/CAPA.csv", as.is=TRUE)
names(x) <- tolower(names(x))
x <- na.omit(x)
ca <- x[x$statefp==6,] # just take CA data
trainFlag <- (runif(nrow(ca)) < 0.66)

# Fit decision tree (from earlier slide)
library(tree)
tf <- tree(log(median_house_value) ~ longitude + latitude, data = ca)
plot(tf)
text(tf, cex=0.75)

# Grab training response and predictors
X <- ca[, c(6,11:15,33:34)]
y <- log(ca$median_house_value)
Xtrain <- X[trainFlag,]
ytrain <- y[trainFlag]
Xtest <- X[!trainFlag,]
ytest <- y[!trainFlag]

# Random forest
library(randomForest)
rfObj <- randomForest(Xtrain, ytrain, Xtest, ytest,
                      do.trace=TRUE, keep.forest=TRUE,
                      ntree=15L)
rfObj

# Use a lot more trees!
rfObj <- randomForest(Xtrain, ytrain, Xtest, ytest,
                      do.trace=FALSE, keep.forest=TRUE,
                      ntree=500L)
rfObj
rfYhat <- predict(rfObj, Xtest)
mean((rfYhat - ytest)^2)

# Variable importance
importance(rfObj)

# Look at one tree
head(getTree(rfObj,k=1),20)

# Easy to extend
grow(rfObj, how.many=10)

#
library(mgcv)
ca.gam2 <- gam(log(median_house_value)
  ~ s(median_household_income) + s(mean_household_income)
  + s(population) + s(total_units) + s(vacant_units)
  + s(median_rooms) + s(mean_household_size_owners)
  + s(mean_household_size_renters)
  + s(longitude,latitude), data=ca, subset=trainFlag)

# Add to predictors
ca$gamPred <- predict(ca.gam2, ca)
X <- ca[, c(6,11:15,33:35)]
y <- log(ca$median_house_value)
Xtrain <- X[trainFlag,]
ytrain <- y[trainFlag]
Xtest <- X[!trainFlag,]
ytest <- y[!trainFlag]

# Now refit:
rfObj <- randomForest(Xtrain, ytrain, Xtest, ytest,
                      do.trace=FALSE, keep.forest=TRUE)
rfObj

rfYhat <- predict(rfObj, Xtest)
mean((rfYhat - ytest)^2)

importance(rfObj)

# Now, GBM
library(gbm)
gbmObj <- gbm.fit(Xtrain, ytrain, distribution="gaussian",
                  n.trees=100L, shrinkage=0.1)
summary(gbmObj,plotit=FALSE)

# Turn down shrinkage
library(gbm)
gbmObj <- gbm.fit(Xtrain, ytrain, distribution="gaussian",
                  n.trees=100L, shrinkage=0.01)
summary(gbmObj,plotit=FALSE)

# Let it run for a while
gbmObj <- gbm.fit(Xtrain, ytrain, distribution="gaussian",
                  n.trees=1e4, shrinkage=0.03, verbose=FALSE)
ntree <- seq(100,gbmObj$n.trees,length.out=100)
gbmYhat <- predict(gbmObj, Xtest, n.trees=ntree)
r <- gbmYhat - matrix(ytest,ncol=100,nrow=length(ytest),byrow=FALSE)
mse <- apply(r^2,2,mean)
plot(ntree, mse, type="l")
abline(h=min(mse),lty="dashed")
text(ntree[100],min(mse)+0.0004,signif(min(mse),3))

# Let it run for a while with a lower amount of shrinkage
gbmObj <- gbm.fit(Xtrain, ytrain, distribution="gaussian",
                  n.trees=1e4, shrinkage=0.005, verbose=FALSE)
ntree <- seq(500,gbmObj$n.trees,length.out=100)
gbmYhat <- predict(gbmObj, Xtest, n.trees=ntree)
r <- gbmYhat - matrix(ytest,ncol=100,nrow=length(ytest),byrow=FALSE)
mse <- apply(r^2,2,mean)
plot(ntree, mse, type="l")
abline(h=min(mse),lty="dashed")
text(ntree[100],min(mse)+0.0004,signif(min(mse),2))

gbmYhat <- predict(gbmObj, Xtest, n.trees=2500)

mean((rfYhat - ytest)^2)
mean((gbmYhat - ytest)^2)

mean((rfYhat*0.8 + gbmYhat*0.2 - ytest)^2)

gamYhat <- as.numeric(ca$gamPred[!trainFlag])
mean((rfYhat*0.8 + gbmYhat*0.1 + gamYhat*0.1 - ytest)^2)


# Now, let's do classification instead:
set.seed(1)
x <- read.csv("data/CAPA.csv", as.is=TRUE)
names(x) <- tolower(names(x))
x <- na.omit(x)

X <- x[, c(6,11:15,33:34)]
trainFlag <- (runif(nrow(x)) < 0.66)
y <- as.numeric(x$statefp == 6)
Xtrain <- X[trainFlag,]
ytrain <- y[trainFlag]
Xtest <- X[!trainFlag,]
ytest <- y[!trainFlag]

rfObj <- randomForest(Xtrain, factor(ytrain),
                      Xtest, factor(ytest),
                      do.trace=FALSE, keep.forest=TRUE,
                      ntree=500L)
rfYhat <- predict(rfObj, Xtest)
mean(rfYhat == ytest)

# Let it run for a while with a lower amount of shrinkage
gbmObj <- gbm.fit(Xtrain, ytrain, distribution="bernoulli",
                  n.trees=1e4, shrinkage=0.005, verbose=FALSE)
gbmYhat <- as.numeric(predict(gbmObj, Xtest, n.trees=gbmObj$n.trees) > 0)
mean(gbmYhat == ytest)

################
library(mgcv)
x.gam2 <- gam(log(median_house_value)
  ~ s(median_household_income) + s(mean_household_income)
  + s(population) + s(total_units) + s(vacant_units)
  + s(median_rooms) + s(mean_household_size_owners)
  + s(mean_household_size_renters)
  + s(longitude,latitude), data=x, subset=trainFlag)

# Now, let's do classification instead:
x$gamPred <- predict(x.gam2, x)
X <- x[, c(6,11:15,33:35)]
trainFlag <- (runif(nrow(x)) < 0.66)
y <- as.numeric(x$statefp == 6)
Xtrain <- X[trainFlag,]
ytrain <- y[trainFlag]
Xtest <- X[!trainFlag,]
ytest <- y[!trainFlag]

rfObj <- randomForest(Xtrain, factor(ytrain),
                      Xtest, factor(ytest),
                      do.trace=FALSE, keep.forest=TRUE,
                      ntree=500L)
rfYhat <- predict(rfObj, Xtest)
mean(rfYhat == ytest)

table(rfYhat, ytest)

these <- which(rfYhat != ytest)
par(mar=c(0,0,0,0))
plot(x$longitude[!trainFlag][these], x$latitude[!trainFlag][these],
     axes=FALSE, xlab="", ylab="")
snippets::osmap()
points(x$longitude[!trainFlag][these], x$latitude[!trainFlag][these],
       pch=19, cex=0.7)

# Let it run for a while with a lower amount of shrinkage
gbmObj <- gbm.fit(Xtrain, ytrain, distribution="bernoulli",
                  n.trees=25000, shrinkage=0.05, verbose=FALSE)
gbmYhat <- as.numeric(predict(gbmObj, Xtest, n.trees=gbmObj$n.trees) > 0)

mean(rfYhat == ytest)
mean(gbmYhat == ytest)

summary(gbmObj, plotit=FALSE)

######
######
######

library(e1071)

out <- svm(y ~ median_household_income + mean_household_income
  + population + total_units + vacant_units
  + median_rooms + mean_household_size_owners
  + mean_household_size_renters,
  data=x, type="C-classification", subset=trainFlag, kernel="linear")

svmYhat <- predict(out, x[!x$trainFlag,])
mean(x$y[!x$trainFlag] == svmYhat)


out <- svm(y ~ median_household_income + mean_household_income
  + population + total_units + vacant_units
  + median_rooms + mean_household_size_owners
  + mean_household_size_renters,
  data=x, type="C-classification", subset=trainFlag, kernel="radial")

svmYhat <- predict(out, x[!x$trainFlag,])
mean(x$y[!x$trainFlag] == svmYhat)


out <- svm(y ~ median_household_income + mean_household_income
  + population + total_units + vacant_units
  + median_rooms + mean_household_size_owners
  + mean_household_size_renters,
  data=x, type="C-classification", subset=trainFlag, kernel="radial")

svmYhat <- predict(out, x[!x$trainFlag,])
mean(x$y[!x$trainFlag] == svmYhat)



# n <- 1000
# p <- 2
# x <- matrix(rnorm(n*p),ncol=p)
# y <- sample(0:1,n,replace=TRUE)
# x[y == 1,] <- x[y == 1,] + 8

df <- data.frame(lon=x[,1],lat=x[,2],y)
out <- svm(y ~ lon + lat,data=df,type="C-classification", kernel="linear")

out$index
out$coefs



