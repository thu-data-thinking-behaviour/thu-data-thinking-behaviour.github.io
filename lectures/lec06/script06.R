#' Title: Decision trees
#' Date: 2016-02-08
#' Authors: Taylor Arnold

##############################
# -- Example 1 --
#   www.stat.yale.edu/~tba3/stat665/data/CAPA.csv
#   11275 observations, 34 variables
#   economic and housing details aggregated at tract level in CA & PA
#   docs: http://www.stat.cmu.edu/~cshalizi/uADA/16/hw/01/hw-01.pdf

x <- read.csv("../../data/CAPA.csv", as.is=TRUE)
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
X <- ca[, c(6,11:15,31,33:34)]
y <- log(ca$median_house_value)
Xtrain <- X[trainFlag,]
ytrain <- y[trainFlag]
Xtest <- X[!trainFlag,]
ytest <- y[!trainFlag]

# Random forest
library(randomForest)
rfObj <- randomForest(Xtrain, ytrain, Xtest, ytest,
                      do.trace=TRUE, keep.forest=TRUE)
rfObj

# Variable importance
importance(rfObj)

# Look at one tree
head(getTree(rfObj,k=1),100)

# Easy to extend
grow(rfObj, how.many=10)

#
library(mgcv)
ca.gam2 <- gam(log(median_house_value)
  ~ s(median_household_income) + s(mean_household_income)
  + s(population) + s(total_units) + s(vacant_units)
  + s(owners) + s(median_rooms) + s(mean_household_size_owners)
  + s(mean_household_size_renters)
  + s(longitude,latitude), data=ca, subset=trainFlag)

# Add to predictors
ca$gamPred <- predict(ca.gam2, ca)
X <- ca[, c(6,11:15,31,33:35)]
y <- log(ca$median_house_value)
Xtrain <- X[trainFlag,]
ytrain <- y[trainFlag]
Xtest <- X[!trainFlag,]
ytest <- y[!trainFlag]

# Now refit:
rfObj <- randomForest(Xtrain, ytrain, Xtest, ytest,
                      do.trace=TRUE, keep.forest=TRUE)
rfObj

importance(rfObj)




