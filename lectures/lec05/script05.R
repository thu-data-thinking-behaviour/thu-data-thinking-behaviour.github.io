################
options(digits=2)
x <- read.csv("../../data/CAPA.csv", as.is=TRUE)
names(x) <- tolower(names(x))
str(x)

badRows <- (apply(is.na(x),1,sum) != 0)
table(badRows)
tapply(x$median_household_income, badRows, median, na.rm=TRUE)
tapply(x$median_house_value, badRows, median, na.rm=TRUE)
tapply(x$vacant_units, badRows, median, na.rm=TRUE)

x <- na.omit(x)

pdf("img/fig01.pdf", height=6, width=12)
plot(x$longitude, x$latitude, pch=19, cex=0.5)
dev.off()

ca <- x[x$statefp==6,]
pa <- x[x$statefp==42,]

set.seed(1)
testFlag <- (runif(nrow(ca)) > 0.8)
trainFlag <- !testFlag
testFlagPa <- (runif(nrow(pa)) > 0.8)
trainFlagPa <- !testFlagPa

cl <- as.numeric(ca$owners < 50)


# Tuning
X <- cbind(ca$latitude,ca$longitude)[trainFlag,]
y <- cl[trainFlag]

foldId <- sample(1:5,nrow(X),replace=TRUE)

library(FNN)

kvals <- 1:25
res <- matrix(ncol=5, nrow=25)
for (i in 1:5) {
  trainSet <- which(foldId != i)
  validSet <- which(foldId == i)
  for (k in 1:25) {
    pred <- knn(X[trainSet,],X[validSet,],y[trainSet],
      k=kvals[k])
    yhat <- (as.numeric(pred) - 1)
    res[k,i] <- mean((y[validSet] != yhat))
    print(k)
  }
}

head(res)
cvError <- apply(res,1,mean)
cvSe <- apply(res,1,sd) / sqrt(5)

pdf("img/fig02.pdf", height=6, width=12)
plot(1:25, cvError, pch=19, cex=0.8, ylim=c(0.22,0.28))
segments(1:25, cvError - 2*cvSe, 1:25,
         cvError + 2*cvSe, col="red")
dev.off()

Xtest <- cbind(ca$latitude,ca$longitude)[testFlag,]
ytest <- cl[testFlag]
yhat <- (as.numeric(knn(X,Xtest,y,k=4)) - 1)
mean((yhat != ytest))


ca.lm <- lm(log(median_house_value) ~ median_household_income
  + mean_household_income + population + total_units +
  + vacant_units + owners + median_rooms +
  + mean_household_size_owners + mean_household_size_renters
  + latitude + longitude, data = ca, subset=trainFlag)

ca.lm2 <- lm(log(median_house_value) ~ median_household_income
  + mean_household_income + population + total_units +
  + vacant_units + owners + median_rooms +
  + mean_household_size_owners + mean_household_size_renters,
  data = ca, subset=trainFlag)

pa.lm3 <- lm(log(median_house_value) ~ median_household_income
  + mean_household_income + population + total_units +
  + vacant_units + owners + median_rooms +
  + mean_household_size_owners + mean_household_size_renters,
  data = pa, subset=trainFlagPa)

library(mgcv)
ca.gam <- gam(log(median_house_value)
  ~ s(median_household_income) + s(mean_household_income)
  + s(population) + s(total_units) + s(vacant_units)
  + s(owners) + s(median_rooms) + s(mean_household_size_owners)
  + s(mean_household_size_renters) + s(latitude)
  + s(longitude), data=ca, subset=trainFlag)

ca.gam2 <- gam(log(median_house_value)
  ~ s(median_household_income) + s(mean_household_income)
  + s(population) + s(total_units) + s(vacant_units)
  + s(owners) + s(median_rooms) + s(mean_household_size_owners)
  + s(mean_household_size_renters)
  + s(longitude,latitude), data=ca, subset=trainFlag)

ca.gam3 <- gam(log(median_house_value)
  ~ s(median_household_income) + s(mean_household_income)
  + s(population) + s(total_units) + s(vacant_units)
  + s(owners) + s(median_rooms) + s(mean_household_size_owners)
  + s(mean_household_size_renters), data=ca, subset=trainFlag)

pa.gam4 <- gam(log(median_house_value)
  ~ s(median_household_income) + s(mean_household_income)
  + s(population) + s(total_units) + s(vacant_units)
  + s(owners) + s(median_rooms) + s(mean_household_size_owners)
  + s(mean_household_size_renters), data=pa, subset=trainFlagPa)


y <- log(ca$median_house_value)
ca.lm.pred <- predict(ca.lm, ca)
ca.lm2.pred <- predict(ca.lm2, ca)
ca.gam.pred <- predict(ca.gam, ca)
ca.gam2.pred <- predict(ca.gam2, ca)
ca.gam3.pred <- predict(ca.gam3, ca)

tapply((ca.lm.pred - y)^2, trainFlag, mean)
tapply((ca.lm2.pred - y)^2, trainFlag, mean)
tapply((ca.gam.pred - y)^2, trainFlag, mean)
tapply((ca.gam2.pred - y)^2, trainFlag, mean)
tapply((ca.gam3.pred - y)^2, trainFlag, mean)

y.pa <- log(pa$median_house_value)
pa.lm2.pred <- predict(ca.lm2, pa)
pa.gam3.pred <- predict(ca.gam3, pa)
pa.lm3.pred <- predict(pa.lm3, pa)
pa.gam4.pred <- predict(pa.gam4, pa)

tapply((pa.lm2.pred - y.pa)^2,trainFlagPa,mean)
tapply((pa.gam3.pred - y.pa)^2,trainFlagPa,mean)
tapply((pa.lm3.pred - y.pa)^2,trainFlagPa,mean)
tapply((pa.gam4.pred - y.pa)^2,trainFlagPa,mean)


tapply((pa.lm2.pred - y.pa),trainFlagPa,var)
tapply((pa.gam3.pred - y.pa),trainFlagPa,var)
tapply((pa.lm3.pred - y.pa),trainFlagPa,var)
tapply((pa.gam4.pred - y.pa),trainFlagPa,var)

for (i in 1:11) {
  pdf(sprintf("img/gamRug%02d.pdf",i), height=5, width=8)
  plot(ca.gam,select=i,scale=0,se=2,shade=TRUE,resid=FALSE)
  dev.off()
}

plot(ca.gam2,scale=0,se=2,shade=TRUE,resid=FALSE,pages=1)
plot(ca.gam2,scale=0,se=2,shade=TRUE,resid=FALSE,select=10)

library(snippets)
pdf("img/fig03.pdf", height=6, width=5)
par(mar=c(0,0,0,0))
plot(ca.gam2,select=10,se=FALSE,col="white",main="")
osmap(tiles.url="http://c.tile.stamen.com/toner/",alpha=0.5)
par(new=TRUE)
plot(ca.gam2,select=10,se=FALSE,col="orange",main="")
par(new=FALSE)
dev.off()

plot(ca.gam2,select=10,se=FALSE,scheme=2,col=rev(rainbow(50)))




