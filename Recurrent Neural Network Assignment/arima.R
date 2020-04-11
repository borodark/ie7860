#0. read f1les
airlaines.file <- 'data/stock_market_data-AAL.csv'
rel.file <- 'data/RELIANCE.csv'

df.aal <- read.csv(airlaines.file, colClasses = c('numeric','Date','numeric','numeric','numeric','numeric'))
df.rel <- read.csv(rel.file, colClasses = c('Date','numeric','numeric','numeric','numeric','numeric','numeric'))

library(dplyr)

df.aal <- df.aal %>% select(Date, Close)
# AAL sorted backwards
df.aal <- df.aal[order(df.aal$Date),]
rownames(df.aal) <- NULL
df.rel <- df.rel %>% select(Date,Close)
df.rel <- df.rel[order(df.rel$Date),]
rownames(df.rel) <- NULL

library(ggplot2)
# 1. Plot Close
ggplot(df.aal, aes(df.aal$Date, y = Close, color = variable)) + 
  geom_line(aes(y = df.aal$Close, col = "Close"))

ggplot(df.rel, aes(df.rel$Date, y = Close, color = variable)) + 
  geom_line(aes(y = df.rel$Close, col = "Close"))

#2. ADF tests forp Close
# https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test
library(tseries)
# ADF test on Close AAL
print(adf.test(df.aal$Close,alternative = "stationary"))
# ADF test on Close REL
print(adf.test(df.rel$Close,alternative = "stationary"))

#3.
#transform
ar_ret <- function(P) {c(NA, P[2:length(P)]/P[1:(length(P)-1)] -1)}
log_ret <- function(P) {c(NA, log(P[2:length(P)]/P[1:(length(P)-1)]))}
# AAL returns
df.aal$ar_ret <- ar_ret(df.aal$Close)
df.aal$log_ret <- log_ret(df.aal$Close)
head(df.aal)
df.aal <- na.omit(df.aal)

# REL returns
df.rel$ar_ret <- ar_ret(df.rel$Close)
df.rel$log_ret <- log_ret(df.rel$Close)
df.rel <- na.omit(df.rel)

#4. Plot Returns
ggplot(df.aal, aes(df.aal$Date, y = Returns, color = variable)) + 
  geom_line(aes(y = df.aal$ar_ret, col = "ar_ret")) +  
  geom_line(aes(y = df.aal$log_ret, col = "log_ret"))
#
# Plot Returns
ggplot(df.rel, aes(df.rel$Date, y = Returns, color = variable)) + 
  geom_line(aes(y = df.rel$ar_ret, col = "ar_ret")) +  
  geom_line(aes(y = df.rel$log_ret, col = "log_ret"))
# Zoom - shoe 100 
df100 <- df.aal[99:130,]
ggplot(df100, aes(df100$Date, y = Returns, color = variable)) + 
  geom_line(aes(y = df100$ar_ret, col = "ar_ret")) +  
  geom_line(aes(y = df100$log_ret, col = "log_ret"))


# ACF-PACF of AR of AAL
acf(df.aal$ar_ret)
pacf(df.aal$ar_ret)
# ACF-PACF of LOGR of AAL
acf(df.aal$log_ret)
pacf(df.aal$log_ret)
# ACF PACF of REL
acf(df.rel$ar_ret)
pacf(df.rel$ar_ret)
acf(df.rel$log_ret)
pacf(df.rel$log_ret)
#5. ADF tests on Ret
# ADF test on ariphmetic return
print(adf.test(df.aal$ar_ret, alternative = "stationary"))
# ADF test on log return
print(adf.test(df.aal$log_ret, alternative = "stationary"))
# ADF test on return
print(adf.test(df.rel$ar_ret, alternative = "stationary"))
print(adf.test(df.rel$log_ret, alternative = "stationary"))


library(stats)
library(forecast)
#6. Fit ARIMA
aal.ar.autoarima <- auto.arima(df.aal[1:3000,]$ar_ret,
                               max.order=300,
                               trace = TRUE)
print(summary(aal.ar.autoarima))
f_ar <- forecast(aal.ar.autoarima, h= 14)
checkresiduals(f_ar)
#
accuracy(f_ar,x=df.aal[3001:3014,]$ar_ret)
#
plot(f_ar, include= 30,
     ylab = 'Ariphmetic Returns', 
     xlab = 'Day', 
     main = 'Arithmetic Returns: Actual vs Predicted')
lines(df.aal$ar_ret)


# Log Returns
aal.ar.autoarima.log <- auto.arima(df.aal[1:3000,]$log_ret,
                               max.order=300,
                               trace = TRUE)
print(summary(aal.ar.autoarima.log))

f_log <- forecast(aal.ar.autoarima.log,h=14)
checkresiduals(f_log)
plot(f_log, include= 30,
     ylab = 'Log Returns', 
     xlab = 'Day', 
     main = 'Log Returns: Actual vs Predicted')
lines(df.aal$log_ret)
accuracy(f_log, x=df.aal[3001:3014,]$log_ret)
#
# Fit REL ar
rel.ar.autoarima <- auto.arima(df.rel[1:5500,]$ar_ret,
                               stepwise=FALSE,
                               parallel = TRUE, 
                               num.cores = 4,
                               max.order=300)
print(rel.ar.autoarima)
f_rel_ar <- forecast(rel.ar.autoarima,h=10)
plot(f_rel_ar,include = 20)
lines(df.rel$ar_ret)
accuracy(f_rel_ar, x=df.rel[5501:5510,]$ar_ret)
# fit REL log
rel.log.autoarima <- auto.arima(df.rel[1:5500,]$log_ret,
                               stepwise=FALSE,
                               parallel = TRUE, 
                               num.cores = 4,
                               max.order=300)
print(rel.log.autoarima)
f_rel_log <- forecast(rel.log.autoarima,h=10)
plot(f_rel_log,include = 20)
lines(df.rel$log_ret)
accuracy(f_rel_log, x=df.rel[5501:5510,]$log_ret)
