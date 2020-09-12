setwd("/Users/taylor/Downloads/Problem Set 1")
filesRaw <- dir()
index <- regexpr("(", filesRaw, fixed=TRUE)
netid <- substr(filesRaw,index+1,nchar(filesRaw)-1)
files <- paste0(filesRaw,"/Submission attachment(s)")
td <- tempdir()

# Determine who submitted the correct format for pset01
scoreFiles <- rep(1, length(files))
commentFiles <- rep("", length(files))

for (i in 1:length(files)) {
  system(paste0("rm -rf ", td, "/*"))
  flag <- FALSE
  d <- dir(files[i])
  these <- which(d == paste0(netid[i],"_pset01.zip"))
  if (length(these) != 1) {
    commentFiles[i] <- paste0(commentFiles[i], " no zip file found;")
    scoreFiles[i] <- 0
  } else {
    unzip(paste0(files[i], "/", netid[i],"_pset01.zip"),
          exdir=td)
    d <- dir(td,recursive=TRUE,full.names=TRUE)

    if (!("pset01.csv" %in% basename(d))) flag <- TRUE
    if (!("pset01.pdf" %in% basename(d))) flag <- TRUE
    if (!("pset01.py" %in% basename(d)) & !("pset01.r" %in% tolower(basename(d))))
      flag <- TRUE

    if (flag) {
      commentFiles[i] <- paste0(commentFiles[i], " files pset01.pdf, pset01.csv, pset01.py/R not found;")
      scoreFiles[i] <- 0
    }
  }
}

# Try to unzip in place
for (i in 1:length(files)) {
  system(paste0("rm -rf ", td, "/*"))
  d <- dir(files[i])
  these <- which(d == paste0(netid[i],"_pset01.zip"))
  dzip <- dir(files[i], recursive=TRUE, full.names=TRUE, pattern=".zip")
  for (dz in dzip) {
    unzip(dz, exdir=files[i])
  }
}

# Still not correct? (Try to manually fix anything odd here)
for (i in 1:length(files)) {
  d <- dir(files[i],recursive=TRUE,full.names=TRUE)
  b <- basename(d)
  flag <- FALSE
  fextInd <- regexpr(".", b, fixed=TRUE)
  b <- b[fextInd > 0]
  fextInd <- fextInd[fextInd > 0]
  bext <- substr(b, fextInd + 1, nchar(b))

  if (!("csv" %in% bext)) flag <- TRUE
  if (!("pdf" %in% bext)) flag <- TRUE
  if (!("py" %in% bext) & !("R" %in% bext))
    flag <- TRUE

  if (flag) {
    print(filesRaw[i])
    print(b)
    cat("\n\n")
  }
}

# Check csv files
scoreCsv <- rep(3, length(files))
commentCsv <- rep("", length(files))
ans <- read.csv("~/files/stat665/psets/pset01/solutions01.csv",header=FALSE)

findex <- setdiff(1:length(files),c(83,94,147))
scoreCsv[83] <- 2
commentCsv[83] <- "ridge values not correct;"
scoreCsv[94] <- 0
commentCsv[94] <- "empty csv file;"
scoreCsv[147] <- 0
commentCsv[147] <- "unreadable csv file;"
for (i in findex) {
  d <- dir(files[i],recursive=TRUE,full.names=TRUE)
  b <- basename(d)
  flag <- FALSE
  fextInd <- regexpr(".", b, fixed=TRUE)
  b <- b[fextInd > 0]
  d <- d[fextInd > 0]
  fextInd <- fextInd[fextInd > 0]
  bext <- substr(b, fextInd + 1, nchar(b))
  index <- which(bext == "csv")
  if (length(index) == 0) {
    commentCsv[i] <- "no csv file found"
    scoreCsv[i] <- 0
  } else {
    x <- read.csv(d[index[1]], as.is=TRUE, header=FALSE)
    if (nrow(x) == 453249) {
      x <- x[-1,,drop=FALSE]
    }
    if (all(dim(x) == c(453248,3)))  {
      x <- apply(x, 2, as.numeric)
      if (abs(cor(x[,1], ans[,1], use="complete.obs")) < 0.98) {
        commentCsv[i] <- paste0(commentCsv[i], " knn values not correct;")
        scoreCsv[i] <- scoreCsv[i] - 0.5
      }
      if (abs(cor(x[,2], ans[,2], use="complete.obs")) < 0.99) {
        commentCsv[i] <- paste0(commentCsv[i], " lm values not correct;")
        scoreCsv[i] <- scoreCsv[i] - 0.5
      }
      if (abs(cor(x[,3], ans[,3], use="complete.obs")) < 0.9) {
        commentCsv[i] <- paste0(commentCsv[i], " ridge values not correct;")
        scoreCsv[i] <- scoreCsv[i] - 0.5
      }
      if (cor(x[,1], ans[,1]) < 0) {
        commentCsv[i] <- paste0(commentCsv[i], " knn sign not correct;")
        scoreCsv[i] <- scoreCsv[i] - 0.5
      }
    } else {
      commentCsv[i] <- paste0(commentCsv[i], " test value matrix not correct dimensions;")
      scoreCsv[i] <- 1
    }
  }

  print(i)
}

# Check R files
scoreR <- rep(4, length(files))
commentR <- rep("", length(files))

library(FNN)
train <- matrix(rnorm(1000),ncol=10)
y <- as.numeric(rnorm(100) + train[,1] > 0)
outKnn <- knn.reg(train,train,y,k=5)$pred

findex <- setdiff(1:length(files),83)

for (i in findex) {
  d <- dir(files[i],recursive=TRUE,full.names=TRUE)
  b <- basename(d)
  flag <- FALSE
  fextInd <- regexpr(".", b, fixed=TRUE)
  b <- b[fextInd > 0]
  d <- d[fextInd > 0]
  fextInd <- fextInd[fextInd > 0]
  bext <- substr(b, fextInd + 1, nchar(b))
  index <- which(tolower(bext) == "r")
  if (length(index) != 0) {
    e <-new.env()
    try({source(d[index[1]],local=e)},silent=TRUE)
    if (!("my_knn" %in% ls(e))) {
      commentR[i] <- paste0(commentR[i], " no function my_knn found;")
      scoreR[i] <- scoreR[i] - 1
    } else {
      ans <- NULL
      try({ans <- e$my_knn(train, y, k=5)}, silent=TRUE)
      if (is.null(ans) | length(ans) != length(outKnn)) ans <- rep(0, length(outKnn))
      if (!is.numeric(ans)) ans <- rep(0, length(outKnn))
      if (max(abs(ans - outKnn)) > 0.1) {
        #commentR[i] <- paste0(commentR[i], " function my_knn not running properly;")
        #scoreR[i] <- scoreR[i] - 0.5
      }
    }
    if (!("my_ksmooth" %in% ls(e))) {
      commentR[i] <- paste0(commentR[i], " no function my_ksmooth found;")
      scoreR[i] <- scoreR[i] - 1
    } else {
    }
  } else scoreR[i] <- 0

  print(i)
}

# Check python files
scorePy <- rep(4, length(files))
commentPy <- rep("", length(files))

library(FNN)
train <- matrix(rnorm(1000),ncol=10)
y <- as.numeric(rnorm(100) + train[,1] > 0)
outKnn <- knn.reg(train,train,y,k=5)$pred

findex <- setdiff(1:length(files),c(74))
scorePy[74] <- 2
commentPy[74] <- "cannot source python file;"
for (i in findex) {
  d <- dir(files[i],recursive=TRUE,full.names=TRUE)
  b <- basename(d)
  flag <- FALSE
  fextInd <- regexpr(".", b, fixed=TRUE)
  b <- b[fextInd > 0]
  d <- d[fextInd > 0]
  fextInd <- fextInd[fextInd > 0]
  bext <- substr(b, fextInd + 1, nchar(b))
  index <- which(tolower(bext) == "py")
  if (length(index) != 0) {
    m <- system(paste0("python '", d[index[1]],"'"))
    if (m != 0) {
      commentPy[i] <- paste0(commentPy[i], " cannot source python file;")
      scorePy[i] <- scorePy[i] - 2
    } else {
      l <- readLines(d[index[1]])
      if (length(grep("my_knn", l)) == 0) {
        commentPy[i] <- paste0(commentPy[i], " cannot find function my_knn;")
        scorePy[i] <- scorePy[i] - 1
      }
      if (length(grep("my_ksmooth", l)) == 0) {
        commentPy[i] <- paste0(commentPy[i], " cannot find function my_ksmooth;")
        scorePy[i] <- scorePy[i] - 1
      }
    }
  } else scorePy[i] <- 0

  print(i)
}
scorePy[scoreR >= scorePy] <- 0

# Check pdf files
scorePdf <- rep(2, length(files))
commentPdf <- rep("", length(files))

findex <- 1:length(files)
for (i in findex) {
  d <- dir(files[i],recursive=TRUE,full.names=TRUE)
  b <- basename(d)
  flag <- FALSE
  fextInd <- regexpr(".", b, fixed=TRUE)
  b <- b[fextInd > 0]
  d <- d[fextInd > 0]
  fextInd <- fextInd[fextInd > 0]
  bext <- substr(b, fextInd + 1, nchar(b))
  index <- which(tolower(bext) == "pdf")
  if (length(index) == 0) {
    commentPdf[i] <- "no pdf file found"
    scorePdf[i] <- 0
  }
  print(i)
}

score <- scoreFiles + scoreCsv + scoreR + scorePy + scorePdf
cmt <- paste0(commentFiles, commentCsv, commentR, commentPy, commentPdf)

mat <- cbind(filesRaw, netid, score, cmt)
write.csv(mat, "~/Desktop/pset01.csv", row.names=FALSE)


