library(sp)
library(maptools)
library(rgeos)
library(rgdal)
library(iotools)

colClasses <- structure(c("character", "character", "character", "integer",
"numeric", "numeric", "numeric", "integer", "character", "numeric",
"numeric", "character", "numeric", "numeric", "numeric", "numeric",
"numeric", "numeric"), .Names = c("vendor_id", "pickup_datetime",
"dropoff_datetime", "passenger_count", "trip_distance", "pickup_longitude",
"pickup_latitude", "rate_code", "store_and_fwd_flag", "dropoff_longitude",
"dropoff_latitude", "payment_type", "fare_amount", "surcharge",
"mta_tax", "tip_amount", "tolls_amount", "total_amount"))

nynta <- readShapePoly("data/nynta_15d/nynta.shp")
poly <- readOGR("data/nynta_15d", "nynta")
nynta <- spTransform(x=poly, CRSobj=CRS(projargs="+proj=longlat"))

r <- readAsRaw("../../../class_data/yellow_tripdata_2014-10.csv")
d <- dstrsplit(r, col_types=colClasses, sep=",")[-c(1,2),]
d <- d[d$tip_amount > 0 & !is.na(d$tip_amount),]
d <- d[ d$pickup_longitude > -180 & d$dropoff_longitude > -180 &
        d$pickup_latitude > -90 & d$dropoff_latitude > -90 &
        d$pickup_longitude < 180 & d$dropoff_longitude < 180 &
        d$pickup_latitude < 90 & d$dropoff_latitude < 90,]
d <- d[sample(1:nrow(d),1e6),]
d <- d[order(d$pickup_datetime),]

pts <- SpatialPoints(d[,c("pickup_longitude","pickup_latitude")])
proj4string(pts) <- CRS(projargs="+proj=longlat")
geoDf <- (pts %over% nynta)
d$pickup_BoroCode <- geoDf$BoroCode
d$pickup_NTACode <- geoDf$NTACode

pts <- SpatialPoints(d[,c("dropoff_longitude","dropoff_latitude")])
proj4string(pts) <- CRS(projargs="+proj=longlat")
geoDf <- (pts %over% nynta)
d$dropoff_BoroCode <- geoDf$BoroCode
d$dropoff_NTACode <- geoDf$NTACode

d <- d[d$pickup_BoroCode == 1,]
d <- d[apply(is.na(d),1,sum) == 0,]
index <- (runif(nrow(d)) > 0.5)
d2 <- d
d2$dropoff_longitude <- d2$dropoff_latitude <- d2$dropoff_BoroCode <- d2$dropoff_NTACode <- NA
write.csv(d[index,], "../../../class_data/nyc_train.csv", quote=FALSE, row.names=FALSE)
write.csv(d2[!index,], "../../../class_data/nyc_test.csv", quote=FALSE, row.names=FALSE)
write.csv(d, "../../../class_private/nyc_all.csv", quote=FALSE, row.names=FALSE)





