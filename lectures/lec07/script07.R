#' Title: Exploring PCA via three examples
#' Date: 2016-02-10
#' Authors: Taylor Arnold, Jay Emerson
#' Notes: Much of this material comes directly from
#'  Cosma Shalizi's Advanced Data Analysis textbook,
#'  specifically chapter 16.

##############################
# -- Example 1 --
#   www.stat.yale.edu/~tba3/stat665/data/cars.csv
#   387 observations, 18 variables
#   details of specs and prices of various personal automobiles from 2004
#   docs: http://www.amstat.org/publications/jse/datasets/04cars.txt

cars <- read.csv("../../data/cars.csv", as.is=TRUE)

# basic understanding of the data (ignoring type variables for today)
str(cars)
pairs(cars[, 8:18], gap=0, pch=19, cex=0.4)

# calculate principal components
pcaObj <- prcomp(cars[, 8:18], scale. = TRUE)

# what does the first PC capture?
round(pcaObj$rotation[,1,drop=FALSE], 2)
ord <- order(pcaObj$x[,1])
rownames(cars)[head(ord,10)]
rownames(cars)[tail(ord,10)]

# what does the second PC capture?
round(pcaObj$rotation[,2,drop=FALSE], 2)
ord <- order(pcaObj$x[,2])
rownames(cars)[head(ord,10)]
rownames(cars)[tail(ord,10)]

# use biplot to visualize both of these together:
biplot(pcaObj, cex = 0.4, xlim=c(-0.2,0.15), ylim=c(-0.2,0.11))

# plot eigenvalues to understand contribution of each PC
plot(pcaObj, type = "l", main = "spectrum of XtX")

##############################
# -- Example 2 --
#   www.stat.yale.edu/~tba3/stat665/data/CAPA.csv
#   11275 observations, 34 variables
#   economic and housing details aggregated at tract level in CA & PA
#   docs: http://www.stat.cmu.edu/~cshalizi/uADA/16/hw/01/hw-01.pdf

x <- read.csv("../../data/CAPA.csv", as.is=TRUE)
names(x) <- tolower(names(x))
x <- na.omit(x)
ca <- x[x$statefp==6,] # just take CA data

str(ca)

# principle components of lat/long
pcaObj <- prcomp(ca[,7:8], scale. = TRUE)
bpoints <- quantile(pcaObj$x[,1], seq(0,1,0.05))
bins <- cut(pcaObj$x[,1], bpoints, include.lowest=TRUE, labels=FALSE)

# visualize PC1 over map
library(snippets) # install.packages("snippets",,"http://rforge.net/")
par(mar=c(0,0,0,0))
plot(ca$longitude, ca$latitude, col="white")
osmap(tiles.url="http://c.tile.stamen.com/toner/",alpha=0.5)
points(ca$longitude, ca$latitude, pch=19, cex=1.2,
        col=rev(heat.colors(20,alpha=0.6))[bins])

# let's take a subset of the other variables
X <- ca[, c(6,11:15,31,33:34)]
pcaObj <- prcomp(X, scale. = TRUE)
round(pcaObj$rotation[,1,drop=FALSE], 2)
biplot(pcaObj, xlabs=rep("·", nrow(ca)))

# what does the first PC capture?
round(pcaObj$rotation[,1,drop=FALSE], 2)

# Construct data frame with PCs
caPCA <- data.frame(pcaObj$x)
caPCA$median_house_value <- ca$median_house_value

# Fit a linear model on both raw components and PCs:
lmObjRaw <- lm(log(ca$median_house_value) ~ ., data=X)
lmObjPca <- lm(log(median_house_value) ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 +
              PC7 + PC8 + PC9, data = caPCA)

# How different are the fits
#  (hint: should be same up to numerical errors)
max(abs(lmObjRaw$residuals - lmObjPca$residuals))

# Now fit additive models on both sets:
library(mgcv)
gamObjRaw <- gam(log(median_house_value)
  ~ s(median_household_income) + s(mean_household_income)
  + s(population) + s(total_units) + s(vacant_units)
  + s(owners) + s(median_rooms) + s(mean_household_size_owners)
  + s(mean_household_size_renters), data=ca)
gamObjPca <- gam(log(median_house_value)
  ~ s(PC1) + s(PC2) + s(PC3) + s(PC4) + s(PC5) + s(PC6) + s(PC7)
  + s(PC8) + s(PC9), data=caPCA)

# How different are the fits
#  (hint: similar, but not quite the same)
max(abs(gamObjRaw$residuals - gamObjPca$residuals))
cor(gamObjRaw$residuals, gamObjPca$residuals)

# Which model fits the data better? Why might this be?
mean((gamObjRaw$residuals)^2)
mean((gamObjPca$residuals)^2)

##############################
# -- Example 3 --
#   www.stat.yale.edu/~tba3/stat665/data/nyt_data.csv
#   102 observations, 4432 variables
#   term frequency matrix from 102 articles; first column also has
#     class labels showing whether article is about art or music
#   docs: https://catalog.ldc.upenn.edu/LDC2008T19

nyt <- read.csv("../../data/nyt_data.csv")
dim(nyt)
nyt[1:10,1:25]

# No need to scale here, as each variable already on comparable scales
pcaObj <- prcomp(nyt[, -1])

# What is the first PC capturing?
signif(sort(pcaObj$rotation[,1], decreasing = TRUE)[1:30],2)
signif(sort(pcaObj$rotation[,1], decreasing = FALSE)[1:30],2)

# What about the second PC?
signif(sort(pcaObj$rotation[,2], decreasing = TRUE)[1:30],2)
signif(sort(pcaObj$rotation[,2], decreasing = FALSE)[1:30],2)

# Use PCs to visualize the data, which has too many variables
# to plot using pairs plot. How well do first two components
# separate the classes?
plot(pcaObj$x[,1:2], col="white")
text(pcaObj$x[,1], pcaObj$x[,2], nyt[,1], cex=0.7)








