## Eliot Ozaki
## CARBON EMISSIONS ON INCOME/OTHER PROJECT
## 7/13/2024

##############################################################################
#Setup
rm(list=ls()) 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(plyr)

#################################
library(tensorflow)
library(keras)
#################################


library(data.table)
library(caret)
library(e1071)
library(DoubleML)
library(mlr3learners)
library(leaps)
library(mgcv)
library(gam)
library(dplyr)

tf$constant("Hello Tensorflow!")

data = fread("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/gdp_merged.csv")
data = data[, !('Unnamed: 0')]

data$Population = gsub(",","",data$Population)

data$Population= as.numeric(data$Population)

########
# Initial research
library(hdm)
library(glmnet)

# Initial (basic) model
x = model.matrix((`Total FFCO2 (tC)`)~ (`Quantity index`)+ (`Thousands of dollars`)^2 + (`Thousands of chained 2012 dollars`) + `Population`, data)
y = data$`Total FFCO2 (tC)`

model = glm(y~x,data = data,family = gaussian())
summary(model)

columns = colnames(data)
emiscols = columns[grepl("\\(tC\\)", columns)]

print(data[is.na(data)])
print(data[complete.cases(data) == FALSE, ])

models = list()


# Making models for every emissions variable
for (i in 1:length(emiscols)) {
  form = as.formula(paste("`",emiscols[i],"` ~ (`Quantity index`)+ (`Thousands of dollars`) + (`Thousands of chained 2012 dollars`) + `Population`", sep = ""))
  models[[i]] = glm(form, data = data, family = gaussian())
}

# Looking for statistically significant models:
for (i in 1:length(models)){
  model_summary = (summary(models[[i]]))
  p_values <- coef(model_summary)[, "Pr(>|t|)"]
  if (any(p_values < 0.05, na.rm = TRUE)) {
    print(paste("Model", emiscols[i], "has significant predictors:"))
    print(model_summary)
  }
}

#Model NRD npt NG (tC) has significant predictors (income)
#Model NRD Total pt FFCO2 (tC) has significant predictors (pop)
#Model NRD Total pt FFCO2 (tC) has significant predictors (pop)

## Same thing, but for logarithmic models
# Making models for every emissions variable
for (i in 1:length(emiscols)) {
  form = as.formula(paste("`",emiscols[i],"` ~ log(`Quantity index`)+ log(`Thousands of dollars`) + log(`Thousands of chained 2012 dollars`) + log(`Population`)", sep = ""))
  models[[i]] = glm(form, data = data, family = gaussian())
}

# Looking for statistically significant models:
for (i in 1:length(models)){
  model_summary = (summary(models[[i]]))
  p_values <- coef(model_summary)[, "Pr(>|t|)"]
  if (any(p_values < 0.05, na.rm = TRUE)) {
    print(paste("Model", emiscols[i], "has significant predictors:"))
    print(model_summary)
  }
}

## Not helpful


#######
# Using a train/test set

set.seed(123)
ntest = floor(nrow(data)/3)  # 1/3 of data is test
testid = sample(1:nrow(data), ntest)  # indices of test obs

testset = data[testid]
trainset = data[-testid]

set.seed(123)
x = model.matrix((`Total FFCO2 (tC)`)~ (`Quantity index`)+ (`Thousands of dollars`) + (`Thousands of chained 2012 dollars`) + `Population`, trainset)
y = trainset$`Total FFCO2 (tC)`

model = glm(y~x,data = trainset, family = gaussian())
summary(model)

models = list()


# Making models for every emissions variable
for (i in 1:length(emiscols)) {
  form = as.formula(paste("`",emiscols[i],"` ~ (`Quantity index`)+ (`Thousands of dollars`) + (`Thousands of chained 2012 dollars`) + `Population`", sep = ""))
  models[[i]] = glm(form, data = trainset, family = gaussian())
}

sigemisvars = list()
sigmodels = list()
# Looking for statistically significant models:
for (i in 1:length(models)) {
  model_summary = (summary(models[[i]]))
  p_values <- coef(model_summary)[, "Pr(>|t|)"]
  significant <- p_values < 0.05 & names(p_values) %in% c("`Quantity index`", "`Thousands of dollars`", "`Thousands of chained 2012 dollars`")
  
  if (any(significant, na.rm = TRUE)) {
    print(paste("Model", emiscols[i], "has significant predictors:"))
    print(model_summary)
    sigemisvars <- append(sigemisvars, emiscols[i])
    sigmodels <- append(sigmodels, models[i])
  }
}

print(sigemisvars)

######
## NOTES:
# [[1]]
# [1] "RES npt Petrol (tC)"
# `Quantity index`                     5.688e+00  9.372e+00   0.607 0.544021    
#`Thousands of dollars`               4.798e-03  6.068e-04   7.908 4.23e-15 ***
# `Thousands of chained 2012 dollars` -5.645e-03  6.962e-04  -8.108 8.73e-16 ***
# [[2]]
# [1] "COM npt Coal (tC)"
# `Quantity index`                    -2.108e-02  5.650e-02  -0.373   0.7092  
# `Thousands of dollars`               7.251e-06  3.658e-06   1.982   0.0476 *
# `Thousands of chained 2012 dollars` -7.453e-06  4.197e-06  -1.776   0.0760 .
# [[3]]
# [1] "COM npt Petrol (tC)"
# `Quantity index`                     4.723e+00  4.755e+00   0.993   0.3207    
# `Thousands of dollars`               2.401e-03  3.078e-04   7.801 9.63e-15 ***
# `Thousands of chained 2012 dollars` -2.623e-03  3.532e-04  -7.426 1.63e-13 ***
# [[4]]
# [1] "COM npt NG (tC)"
# `Quantity index`                     4.377e+00  1.256e+01   0.348   0.7276    
# `Thousands of dollars`               6.034e-03  8.134e-04   7.419 1.71e-13 ***
# `Thousands of chained 2012 dollars` -6.466e-03  9.333e-04  -6.928 5.67e-12 ***
# [[5]]
# [1] "IND npt Petrol (tC)"
# `Quantity index`                    -1.921e+01  2.500e+01  -0.768  0.44229    
# `Thousands of dollars`              -7.995e-03  1.618e-03  -4.940 8.45e-07 ***
# `Thousands of chained 2012 dollars`  1.003e-02  1.857e-03   5.403 7.31e-08 ***
# [[6]]
# [1] "IND npt NG (tC)"
# `Quantity index`                    -9.589e+01  7.305e+01  -1.313    0.189    
# `Thousands of dollars`              -2.552e-02  4.729e-03  -5.396 7.60e-08 ***
# `Thousands of chained 2012 dollars`  3.162e-02  5.427e-03   5.827 6.53e-09 ***
# [[7]]
# [1] "NRD npt Petrol (tC)"
# `Quantity index`                     2.288e+01  7.640e+00   2.994  0.00278 ** 
# `Thousands of dollars`               5.556e-03  4.946e-04  11.233  < 2e-16 ***
# `Thousands of chained 2012 dollars` -6.575e-03  5.675e-04 -11.584  < 2e-16 ***
# [[8]]
# [1] "NRD npt NG (tC)"
# `Quantity index`                    -4.063e+00  7.777e-01  -5.224 1.93e-07 ***
# `Thousands of dollars`              -8.126e-04  5.035e-05 -16.139  < 2e-16 ***
# `Thousands of chained 2012 dollars`  1.013e-03  5.777e-05  17.536  < 2e-16 ***
# [[9]]
# [1] "RRD npt Petrol (tC)"
# `Quantity index`                     2.056e+00  6.154e+00   0.334 0.738395    
# `Thousands of dollars`               7.952e-04  3.984e-04   1.996 0.046080 *  
# `Thousands of chained 2012 dollars` -1.004e-03  4.572e-04  -2.197 0.028113 * 
# [[10]]
# [1] "AIR pt Petrol (tC)"
# `Quantity index`                     1.083e+01  7.409e+00   1.462    0.144    
# `Thousands of dollars`               4.410e-03  4.796e-04   9.194   <2e-16 ***
# `Thousands of chained 2012 dollars` -4.966e-03  5.504e-04  -9.023   <2e-16 ***
# [[11]]
# [1] "COM pt Coal (tC)"
# `Quantity index`                    -7.387e-02  1.781e-01  -0.415   0.6784    
# `Thousands of dollars`               2.264e-05  1.153e-05   1.963   0.0498 *  
# `Thousands of chained 2012 dollars` -2.068e-05  1.323e-05  -1.563   0.1182  
# [[12]]
# [1] "COM pt NG (tC)"
# `Quantity index`                    -1.337e+00  3.734e+00  -0.358 0.720285    
# `Thousands of dollars`              -3.850e-04  2.418e-04  -1.592 0.111447    
# `Thousands of chained 2012 dollars`  6.028e-04  2.774e-04   2.173 0.029900 *
# [[13]]
# [1] "IND pt Petrol (tC)"
# `Quantity index`                     6.499e-01  1.148e+01   0.057  0.95487   
# `Thousands of dollars`              -2.108e-03  7.434e-04  -2.835  0.00462 **
# `Thousands of chained 2012 dollars`  2.426e-03  8.530e-04   2.844  0.00450 **
# [[14]]
# [1] "IND pt NG (tC)"
# `Quantity index`                     4.251e+00  1.985e+01   0.214 0.830421    
# `Thousands of dollars`              -5.198e-03  1.285e-03  -4.045 5.42e-05 ***
# `Thousands of chained 2012 dollars`  5.724e-03  1.474e-03   3.882 0.000107 ***
# [[15]]
# [1] "NRD pt Petrol (tC)"
# `Quantity index`                     1.998e-02  2.953e-02   0.677    0.499    
# `Thousands of dollars`               1.628e-05  1.912e-06   8.514  < 2e-16 ***
# `Thousands of chained 2012 dollars` -1.736e-05  2.193e-06  -7.914 4.02e-15 ***
# [[16]]
# [1] "RRD pt Petrol (tC)"
# `Quantity index`                    -1.715e-01  1.567e+00  -0.109  0.91285   
# `Thousands of dollars`               2.620e-04  1.015e-04   2.582  0.00989 **
# `Thousands of chained 2012 dollars` -2.888e-04  1.164e-04  -2.480  0.01321 * 
# [[17]]
# [1] "ELC Coal (tC)"
# `Quantity index`                    -5.031e+02  2.334e+02  -2.155   0.0312 *  
# `Thousands of dollars`               8.058e-03  1.511e-02   0.533   0.5939    
# `Thousands of chained 2012 dollars` -1.162e-02  1.734e-02  -0.670   0.5030 
# [[18]]
# [1] "ELC NG (tC)"
# `Quantity index`                     1.538e+02  1.029e+02   1.496  0.13493    
# `Thousands of dollars`               1.464e-02  6.659e-03   2.199  0.02798 *  
# `Thousands of chained 2012 dollars` -2.249e-02  7.641e-03  -2.943  0.00328 **
# [[19]]
# [1] "ONR Diesel (tC)"
# `Quantity index`                     4.906e+01  1.674e+01   2.931  0.00342 ** 
# `Thousands of dollars`               6.921e-03  1.084e-03   6.386 2.10e-10 ***
# `Thousands of chained 2012 dollars` -7.923e-03  1.244e-03  -6.371 2.30e-10 ***
# [[20]]
# [1] "ONR NG (tC)"
# `Quantity index`                     1.854e-01  9.570e-01   0.194   0.8464    
# `Thousands of dollars`               3.603e-04  6.195e-05   5.816 6.97e-09 ***
# `Thousands of chained 2012 dollars` -3.950e-04  7.109e-05  -5.557 3.10e-08 ***
# [[21]]
# [1] "CMT (tC)"
# `Quantity index`                     3.147e+01  2.405e+01   1.308 0.190907    
# `Thousands of dollars`               4.624e-03  1.557e-03   2.969 0.003020 ** 
# `Thousands of chained 2012 dollars` -6.145e-03  1.787e-03  -3.439 0.000595 ***
# [[22]]
# [1] "RES Total npt FFCO2 (tC)"
# `Quantity index`                    -1.926e+01  2.647e+01  -0.728  0.46685    
# `Thousands of dollars`               4.137e-03  1.713e-03   2.415  0.01583 *  
# `Thousands of chained 2012 dollars` -4.689e-03  1.966e-03  -2.385  0.01716 *
# [[23]]
# [1] "COM Total npt FFCO2 (tC)"
# `Quantity index`                     9.079e+00  1.527e+01   0.595  0.55210    
# `Thousands of dollars`               8.443e-03  9.883e-04   8.543  < 2e-16 ***
# `Thousands of chained 2012 dollars` -9.096e-03  1.134e-03  -8.021 1.74e-15 ***
# [[24]]
# [1] "IND Total npt FFCO2 (tC)"
# `Quantity index`                    -1.169e+02  9.104e+01  -1.284    0.199    
# `Thousands of dollars`              -3.312e-02  5.894e-03  -5.619 2.18e-08 ***
# `Thousands of chained 2012 dollars`  4.124e-02  6.763e-03   6.098 1.28e-09 ***
# [[25]]
# [1] "RRD Total npt FFCO2 (tC)"
# `Quantity index`                     2.056e+00  6.154e+00   0.334 0.738395    
# `Thousands of dollars`               7.952e-04  3.984e-04   1.996 0.046080 *  
# `Thousands of chained 2012 dollars` -1.004e-03  4.572e-04  -2.197 0.028113 *
# [[26]]
# [1] "NRD Total npt FFCO2 (tC)"
# `Quantity index`                     1.881e+01  7.182e+00   2.620  0.00887 ** 
# `Thousands of dollars`               4.743e-03  4.650e-04  10.201  < 2e-16 ***
# `Thousands of chained 2012 dollars` -5.561e-03  5.335e-04 -10.424  < 2e-16 ***
# [[27]]
# [1] "COM Total pt FFCO2 (tC)"
# `Quantity index`                    -1.867e+00  4.151e+00  -0.450 0.652924    
# `Thousands of dollars`              -4.333e-04  2.688e-04  -1.612 0.107067    
# `Thousands of chained 2012 dollars`  6.756e-04  3.084e-04   2.191 0.028583 *
# [[28]]
# [1] "IND Total pt FFCO2 (tC)"
# `Quantity index`                    -4.907e+00  3.014e+01  -0.163 0.870674    
# `Thousands of dollars`              -7.118e-03  1.951e-03  -3.648 0.000271 ***
# `Thousands of chained 2012 dollars`  7.908e-03  2.239e-03   3.532 0.000421 ***
# [[29]]
# [1] "NRD Total pt FFCO2 (tC)"
# `Quantity index`                     1.998e-02  2.953e-02   0.677    0.499    
# `Thousands of dollars`               1.628e-05  1.912e-06   8.514  < 2e-16 ***
# `Thousands of chained 2012 dollars` -1.736e-05  2.193e-06  -7.914 4.02e-15 ***
# [[30]]
# [1] "RRD Total pt FFCO2 (tC)"
# `Quantity index`                    -1.715e-01  1.567e+00  -0.109  0.91285   
# `Thousands of dollars`               2.620e-04  1.015e-04   2.582  0.00989 **
# `Thousands of chained 2012 dollars` -2.888e-04  1.164e-04  -2.480  0.01321 *
# [[31]]
# [1] "RES Total FFCO2 (tC)"
# `Quantity index`                    -1.926e+01  2.647e+01  -0.728  0.46685    
# `Thousands of dollars`               4.137e-03  1.713e-03   2.415  0.01583 *  
# `Thousands of chained 2012 dollars` -4.689e-03  1.966e-03  -2.385  0.01716 *
# [[32]]
# [1] "COM Total FFCO2 (tC)"
# `Quantity index`                     7.212e+00  1.623e+01   0.444   0.6568    
# `Thousands of dollars`               8.010e-03  1.050e-03   7.625 3.69e-14 ***
# `Thousands of chained 2012 dollars` -8.421e-03  1.205e-03  -6.986 3.79e-12 ***
# [[33]]
# [1] "IND Total FFCO2 (tC)"
# `Quantity index`                    -1.218e+02  1.043e+02  -1.168   0.2431    
# `Thousands of dollars`              -4.024e-02  6.753e-03  -5.958 2.99e-09 ***
# `Thousands of chained 2012 dollars`  4.915e-02  7.749e-03   6.343 2.77e-10 ***
# [[34]]
# [1] "RRD Total FFCO2 (tC)"
# `Quantity index`                     1.884e+00  7.030e+00   0.268  0.78871   
# `Thousands of dollars`               1.057e-03  4.551e-04   2.323  0.02028 * 
# `Thousands of chained 2012 dollars` -1.293e-03  5.222e-04  -2.477  0.01334 *
# [[35]]
# [1] "ONR Total FFCO2 (tC)"
# `Quantity index`                     5.912e+01  4.547e+01   1.300  0.19365    
# `Thousands of dollars`               8.257e-03  2.944e-03   2.805  0.00508 ** 
# `Thousands of chained 2012 dollars` -6.996e-03  3.378e-03  -2.071  0.03846 *
# [[36]]
# [1] "NRD Total FFCO2 (tC)"
# `Quantity index`                     1.883e+01  7.183e+00   2.622  0.00881 ** 
# `Thousands of dollars`               4.760e-03  4.651e-04  10.235  < 2e-16 ***
# `Thousands of chained 2012 dollars` -5.579e-03  5.336e-04 -10.455  < 2e-16 ***
# [[37]]
# [1] "AIR Total FFCO2 (tC)"
# `Quantity index`                     1.083e+01  7.409e+00   1.462    0.144    
# `Thousands of dollars`               4.410e-03  4.796e-04   9.194   <2e-16 ***
# `Thousands of chained 2012 dollars` -4.966e-03  5.504e-04  -9.023   <2e-16 ***
# [[38]]
# [1] "CMT Total CO2 (tC)"
# `Quantity index`                     3.147e+01  2.405e+01   1.308 0.190907    
# `Thousands of dollars`               4.624e-03  1.557e-03   2.969 0.003020 ** 
# `Thousands of chained 2012 dollars` -6.145e-03  1.787e-03  -3.439 0.000595 ***
# 
print(sapply(sigmodels,class))
predictions = list()
mse = list()
for (i in 1:length(sigmodels)) {
  pred = predict(sigmodels[[i]], testset)
  actual <- testset[[sigemisvars[[i]]]]
  mse[[i]] = mean((pred-actual)^2)
  predictions[[i]] = pred
}

print(mse)


######
## Looking at tax data alongside income data



