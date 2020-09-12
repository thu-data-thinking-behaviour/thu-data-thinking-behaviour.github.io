x <- na.omit(read.csv("../../data/CAPA.csv", as.is=TRUE))
names(x) <- tolower(names(x))
ca <- x[x$statefp == 6L,]

library(tree)

tf <- tree(log(median_house_value) ~ longitude + latitude, data = ca)
pdf("img/simpleTree.pdf", height=7, width=7)
plot(tf)
text(tf, cex=0.75)
dev.off()


pd = quantile(ca$median_house_value, 0:10/10)
bins = cut(ca$median_house_value, pd, include.lowest = TRUE)
pdf("img/simplePartTree.pdf", height=7, width=7)
plot(ca$longitude, ca$latitude, col = grey(10:2/11)[bins], pch = 20,
    xlab = "Longitude", ylab = "Latitude")
partition.tree(tf, ordvars = c("longitude", "latitude"), add = TRUE)
dev.off()

n <- 100
nI <- n/2
nIC <- n/2
y <- runif(n) + 1:n
I <- (1:n > n/2)
sum( (y[I] -  mean(y[I]))^2 )

Ibar <- mean(y[I])
ICbar <- mean(y[!I])

sum( (y[I] - Ibar)^2 ) + sum( (y[!I] - ICbar)^2 )
sum(y[I]^2) - nI*Ibar^2 + sum(y[!I]^2) - nIC*ICbar^2
sum(y^2) - nI*Ibar^2 - nIC*ICbar^2

nI*Ibar^2 + nIC*ICbar^2
sum(y[I])^2


cl <- as.numeric(ca$median_house_value > median(ca$median_house_value))
cols <- rep("#0000FF",nrow(ca))
cols[cl == 1] <- "#FF6600"


plot(ca$median_rooms, ca$mean_household_income, col=cols, cex=0.25,pch=19)

ord <- order(ca$mean_household_income)
clL <- as.numeric(cumsum(cl[ord]) / 1:length(cl) > 0.5)
clR <- as.numeric(cumsum(cl[rev(ord)]) / 1:length(cl) > 0.5)

misClassLeft  <- cumsum(cl[ord] != clL)
misClassRight <- rev(cumsum(cl[rev(ord)] != clR))
misClass <- misClassRight + misClassLeft


out <- rpart(cl ~ ca$mean_household_income, method="class")
val <- out$splits[1,"index"]
abline(h=val, col="black", lwd=3)

index <- ca$mean_household_income > val
out <- rpart(cl ~ ca$median_rooms, method="class",
            subset=index, control=rpart.control(maxdepth=2L, cp=0))
val <- out$splits[1,"index"]
abline(v=val, col="black", lwd=3)

out <- rpart(cl ~ ca$median_rooms, method="class",
            subset=!index, control=rpart.control(maxdepth=2L, cp=0))
val <- out$splits[1,"index"]
abline(v=val, col="black", lwd=3)








