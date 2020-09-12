# Problem Set 01 starter code
#
# Please make sure your code runs on R version 3.2.2
# Due date: 2016-02-05 13:00

#' Basic k-nearest neighbor functionality
#'
#' k-nearest neighbor regression for a numeric test
#' matrix. Prediction are returned for the same data matrix
#' used for training. For each row of the input, the k
#' closest rows (using the l2 distance) in the training
#' set are identified. The mean of the observations y
#' is used for the predicted value of a new observation.
#'
#' @param X     an n by p numeric matrix; the data
#'                matrix of predictors
#' @param y     a length n numberic vector; the observed
#'                response
#' @param k     integer giving the number of neighbors to
#'                include
#'
#' @return      a vector of predicted responses for each row
#'                of the input matrix X
my_knn <- function(X, y, k=1L) {
  D <- as.matrix(dist(X)) # the distance matrix
}

#' Kernel smoothing function
#'
#' kernel smoother for a numeric test matrix with a Gaussian
#' kernel. Prediction are returned for the same data matrix
#' used for training. For each row of the input, a weighted
#' average of the input y is used for prediction. The weights
#' are given by the density of the normal distribution for
#' the distance of a point to the input.
#'
#' @param X       an n by p numeric matrix; the data
#'                  matrix of predictors
#' @param y       a length n numberic vector; the observed
#'                  response
#' @param sigma   the standard deviation of the normal density
#'                  function used for the weighting scheme
#'
#' @return        a vector of predicted responses for each row
#'                  of the input matrix Xnew
my_ksmooth <- function(X, y, sigma=1.0) {
  D <- as.matrix(dist(X)) # the distance matrix

  dnorm(1, sigma=sigma) # gives the Gaussian density function
}





