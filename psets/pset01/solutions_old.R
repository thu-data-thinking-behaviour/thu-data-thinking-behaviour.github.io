#####
library(FNN)
library(glmnet)
train <- read.csv("~/files/class_data/nyc_train.csv", as.is=TRUE)
test <- read.csv("~/files/class_data/nyc_test.csv", as.is=TRUE)
Xtrain <- cbind(train$pickup_longitude, train$pickup_latitude)
Xtest <- cbind(test$pickup_longitude, test$pickup_latitude)
y <- as.numeric(train$dropoff_BoroCode != 1)
trainDf <- data.frame(y=y,hr=substr(train$pickup_datetime, 12, 13),nbh=train$pickup_NTACode)
testDf  <- data.frame(y=1,hr=substr(test$pickup_datetime, 12, 13),nbh=test$pickup_NTACode)
XtrainL2 <- model.matrix(y ~ factor(hr) + factor(nbh) - 1, data=trainDf)
XtestL2  <- model.matrix(y ~ factor(hr) + factor(nbh) - 1, data=testDf)

out1 <- knn.reg(Xtrain, Xtest, y, k=100)
out2 <- lm(y ~ factor(hr) + factor(nbh), data=trainDf)
out3 <- cv.glmnet(XtrainL2, y, alpha=0)

pred1 <- out1$pred
pred2 <- predict(out2, testDf)
pred3 <- predict(out3, XtestL2, lambda=out3$lambda.1se)

write.table(cbind(pred1,pred2,pred3),"~/files/stat665/psets/pset01/solutions01.csv",
            sep=",", quote=FALSE, col.names=FALSE, row.names=FALSE)

#####
import importlib.util
spec = importlib.util.spec_from_file_location("my_knn", "/path/to/file.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
foo.my_knn()

#####
library(FNN)
files <- dir()
index <- regexpr("(", files, fixed=TRUE)
netid <- substr(files,index+1,nchar(files)-1)
files <- paste0(files,"/Submission attachment(s)")

td <- tempdir()
ans <- read.csv("~/files/stat665/psets/pset01/solutions01.csv",header=FALSE)

train <- matrix(rnorm(1000),ncol=10)
y <- as.numeric(rnorm(100) + train[,1] > 0)
outKnn <- knn.reg(train,train,y,k=5)$pred
D <- as.matrix(dist(train)) # the distance matrix
n <- nrow(train)
pred<-rep(NA,n)
for (i in 1:n){
  d<-sort(D[i,])
  l<-as.numeric(names(d))
  N<-y[l]
  w=dnorm(d,0,sd=0.5)
  pred[i]=sum(N*w)/sum(w)
}
outKsmooth <- pred

badIndex <- c()
for (i in 1:length(files)) try({
  system(paste0("rm -rf ", td, "/*"))
  d <- dir(files[i])
  these <- which(d == paste0(netid[i],"_pset01.zip"))
  if (length(these) != 1) {
    badIndex <- append(badIndex, i)
  } else {
    unzip(paste0(files[i], "/", netid[i],"_pset01.zip"),
          exdir=td)
    d <- dir(td,recursive=TRUE,full.names=TRUE)

    if (length(setdiff(c("pset01.csv", "pset01.pdf", "pset01.py"), basename(d))) == 0 ) {
      typeFlag <- 1
    } else if (length(setdiff(c("pset01.csv", "pset01.pdf", "pset01.R"), basename(d))) == 0 ) {
      typeFlag <- 2
    } else {
      typeFlag <- 3
    }

    if (typeFlag == 3) {
      badIndex <- append(badIndex, i)
    }
  }
})

findex <- setdiff(1:length(files),badIndex)
for (i in findex) try({
  system(paste0("rm -rf ", td, "/*"))
  d <- dir(files[i])
  these <- which(d == paste0(netid[i],"_pset01.zip"))


  system(paste0("rm -rf ", td, "/*"))
  d <- dir(files[i])
  these <- which(d == paste0(netid[i],"_pset01.zip"))
  if (length(these) != 1) {
    print(netid[i])
  } else {

    unzip(paste0(files[i], "/", netid[i],"_pset01.zip"),
          exdir=td)
    d <- dir(td,recursive=TRUE,full.names=TRUE)

    if (length(setdiff(c("pset01.csv", "pset01.pdf", "pset01.py"), basename(d))) == 0 ) {
      typeFlag <- 1
    } else if (length(setdiff(c("pset01.csv", "pset01.pdf", "pset01.R"), basename(d))) == 0 ) {
      typeFlag <- 2
    } else {
      typeFlag <- 3
    }

    if (typeFlag == 3) {
      print(netid[i])
    }

    if (typeFlag != 3) {
      x <- read.csv(d[basename(d) == "pset01.csv"], as.is=TRUE, header=FALSE)
      if (nrow(x) == 453249) {
        x <- x[-1,]
      }
      x <- apply(x, 2, as.numeric)
      if (all(dim(x) == c(453248,3)))  {
        if (abs(cor(x[,1], ans[,1], use="complete.obs")) < 0.98) print("bad c1")
        if (abs(cor(x[,2], ans[,2], use="complete.obs")) < 0.99) print("bad c2")
        if (abs(cor(x[,3], ans[,3], use="complete.obs")) < 0.9) print("bad c3")
        if (cor(x[,1], ans[,1]) < 0) print("Negative sign")
      } else {
        print("bad data")
      }
    }

    if (typeFlag == 2) {
      e <-new.env()
      source(d[basename(d) == "pset01.R"],local=e)
      if (!("my_knn" %in% ls(e))) print("no knn")
      if (!("my_ksmooth" %in% ls(e))) print("no knn")
      if (max(abs(e$my_knn(train, y, k=5) - outKnn)) > 0.1) print("bad knn")
      #if (max(abs(e$my_ksmooth(train, y, k=5) - outKnn)) > 0.1) print("bad knn")
    }
    outKnn

  }
})




for (i in 1:length(files)) try({
  system(paste0("rm -rf ", td, "/*"))
  d <- dir(files[i])
  these <- which(d == paste0(netid[i],"_pset01.zip"))
  if (length(these) != 1) {
    print(netid[i])
  } else {

    unzip(paste0(files[i], "/", netid[i],"_pset01.zip"),
          exdir=td)
    d <- dir(td,recursive=TRUE,full.names=TRUE)

    if (length(setdiff(c("pset01.csv", "pset01.pdf", "pset01.py"), basename(d))) == 0 ) {
      typeFlag <- 1
    } else if (length(setdiff(c("pset01.csv", "pset01.pdf", "pset01.R"), basename(d))) == 0 ) {
      typeFlag <- 2
    } else {
      typeFlag <- 3
    }

    if (typeFlag == 3) {
      print(netid[i])
    }

    if (typeFlag != 3) {
      x <- read.csv(d[basename(d) == "pset01.csv"], as.is=TRUE, header=FALSE)
      if (nrow(x) == 453249) {
        x <- x[-1,]
      }
      x <- apply(x, 2, as.numeric)
      if (all(dim(x) == c(453248,3)))  {
        if (abs(cor(x[,1], ans[,1], use="complete.obs")) < 0.98) print("bad c1")
        if (abs(cor(x[,2], ans[,2], use="complete.obs")) < 0.99) print("bad c2")
        if (abs(cor(x[,3], ans[,3], use="complete.obs")) < 0.9) print("bad c3")
        if (cor(x[,1], ans[,1]) < 0) print("Negative sign")
      } else {
        print("bad data")
      }
    }

    if (typeFlag == 2) {
      e <-new.env()
      source(d[basename(d) == "pset01.R"],local=e)
      if (!("my_knn" %in% ls(e))) print("no knn")
      if (!("my_ksmooth" %in% ls(e))) print("no knn")
      if (max(abs(e$my_knn(train, y, k=5) - outKnn)) > 0.1) print("bad knn")
      #if (max(abs(e$my_ksmooth(train, y, k=5) - outKnn)) > 0.1) print("bad knn")
    }
    outKnn

  }
})
