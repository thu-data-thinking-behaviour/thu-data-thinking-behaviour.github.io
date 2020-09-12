#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)
if (length(args) < 2)
  stop("Not enough input arguments")

# You can uncomment this to test the script; don't
#  forget to uncomment before submitting!
# args <- c("train.csv", "test.csv")

# You may find something like this very helpful; note
#   that if x is a vector, you need to wrap it in a call
#   to as.matrix for knn.reg to work without errors
# library(FNN)
# knn.reg(as.matrix(x), as.matrix(x), y, k=20)

# Read in the datasets

# Write your backfitting algorithm here

# Use linear interpolation to predict new values on the test data
# Hint: Use the approxfun function

# Save results to "results.csv"
write.table(yhat, "results.csv", row.names=FALSE, col.names=FALSE, sep=",")

