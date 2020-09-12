library(snippets)

zip <- read.csv("../../data/zip_code.csv", sep="|", as.is=TRUE, header=FALSE)
hp <- read.csv("../../data/zillow.csv", sep=",", )

index <- match(hp$RegionName, zip$V1)
hp <- hp[!is.na(index),]
index <- index[!is.na(index)]

hp$lat <- zip$V2[index]
hp$lon <- zip$V3[index]

val <- hp$X2015.10
temp <- quantile(val,seq(0,1,0.05))
bins <- cut(val, temp, labels=FALSE, include.lowest=TRUE)
colPal <- heat.colors(20)


par(mar=c(0,0,0,0))
index <- (hp$State == "FL")
plot(hp$lon[index], hp$lat[index], pch=19, col=colPal[bins], cex=0.5)
osmap()



