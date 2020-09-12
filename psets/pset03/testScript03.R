#!/usr/bin/env Rscript

# Usage:
#   testScript03.R pset03.csv

# This script must be run in the same directory as one of:
#   pset03_answers.csv (only the TAs have this)
#   pset03_sample.csv (download from website)

args = commandArgs(trailingOnly=TRUE)
if (length(args) < 1)
  stop("Not enough input arguments")

myAns <- as.numeric(read.csv(args[1], as.is=TRUE, header=FALSE)[,1])

cat("\n")

sampleFlag <- FALSE
if (file.exists("pset03_answers.csv")) {
  ans <- as.numeric(read.csv("pset03_answers.csv", as.is=TRUE, header=FALSE)[,1])
} else if (file.exists("pset03_sample.csv")) {
  ans <- as.numeric(read.csv("pset03_sample.csv", as.is=TRUE, header=FALSE)[,1])
  sampleFlag <- TRUE
}

if (length(ans) != length(myAns)) {
  stop(sprintf("Number of predictions not correct (expected: %d, got %d)",
        length(ans), length(myAns)))
}

if (any(is.na(myAns))) {
  warning(sprintf("There are %d missing values in your solutions!", sum(is.na(myAns))))
}

if (any(!(na.omit(myAns) %in% 1:5 ))) {
  warning("Some predictions are not equal to one of the integers: 1,2,3,4, or 5.")
}

if (sampleFlag) {
  inc <- sum(myAns != ans, na.rm=TRUE)
  num <- sum(!is.na(myAns != ans))
  cat(" --- Running in testing mode --- \n")
  cat(sprintf("Misclassification rate on non-missing values is: %02.6f", inc/num), "\n")
} else {
  index <- ans[is.na(ans)] # the 'free' test set in pset03_sample.csv will be set to NAs
  inc <- sum((myAns != ans)[index], na.rm=TRUE)
  ina <- sum(is.na(myAns != ans)[index])
  cat(sprintf("Misclassification rate is: %02.6f", (inc + ina)/length(ans)), "\n")
  cat(sprintf("Missing predictions on: %d/%d observations", ina,length(ans)), "\n")
}

cat("\n")


