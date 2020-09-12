x <- read.csv("http://euler.stat.yale.edu/~tba3/class_data/nyc_test.csv",
              as.is=TRUE, nrow=1e5)
head(x)
dim(x)

par(mar=c(0,0,0,0))
plot(x$pickup_longitude, x$pickup_latitude,
      pch=19, cex=0.5, col=rgb(0,0,0,0.01))

library(snippets)
plot(x$pickup_longitude, x$pickup_latitude,
     pch=19, cex=0.5, col=rgb(0,0,0,0.01))
osmap()
points(x$pickup_longitude, x$pickup_latitude,
       pch=19, cex=0.5, col=rgb(1,0.6,0,0.01))

library(snippets)
plot(x$dropoff_longitude, x$dropoff_latitude,
     pch=19, cex=0.5, col=rgb(0,0,0,0.01))
osmap()
points(x$dropoff_longitude, x$dropoff_latitude,
       pch=19, cex=0.5, col=rgb(1,0.6,0,0.01))


