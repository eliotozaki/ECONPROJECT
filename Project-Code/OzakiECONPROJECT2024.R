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
library(keras3)
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

# Find rows with any missing values
rows_with_na <- rowSums(is.na(data)) > 0

# Print the rows with missing values
print(data[rows_with_na])
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


#############################################################################################################
## Looking at tax data alongside income data

df = fread("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/ALLData-MergedDS.csv")
print(colnames(df))

print(df[, c('Finance and insurance _x', 'Finance and insurance _y')])

## NOTE: _x is Contributions to percent change in real GDP, _y is Gross domestic product (GDP) by county and metropolitan area

column_types <- sapply(df, class)
print(column_types)

colnames(df) <- gsub("_x", " % change in real GDP", colnames(df))
colnames(df) <- gsub("_y", " (GDP)", colnames(df))


# Select only the columns that are numeric
numeric_cols <- sapply(df, is.numeric)
numeric_df <- df[, numeric_cols, with = FALSE]
print(numeric_df)
print(numeric_df$`RES npt Coal (tC)`)
print(numeric_df$N10600)


# Initial (basic) model
x = model.matrix(df$`Total FFCO2 (tC)`~ ., numeric_df)
y = df$`Total FFCO2 (tC)`

model = glm(y~x,data = numeric_df,family = gaussian())
summary(model)

columns = colnames(data)
emiscols = columns[grepl("\\(tC\\)", columns)]

library(glmnet)
# Define a function to perform LASSO for a given response variable
perform_lasso <- function(response_var, df, fulldata) {
  y_numeric <- fulldata[[response_var]]
  
  if (sd(y_numeric) == 0) {
    warning(paste("Response variable", response_var, "is constant; skipping."))
    return(list(relevant_variables = NULL, top10_vars = NULL))
  }
  # Prepare the predictor matrix (X) from df excluding the response variable
  x <- as.matrix(df)
  
  # Set seed for reproducibility
  set.seed(123)
  
  # Fit the Lasso model
  lasso_model <- glmnet(x, y_numeric, family = "gaussian", alpha = 1)
  
  # Cross-validation for Lasso
  set.seed(123)
  cv_model <- cv.glmnet(x, y_numeric, family = "gaussian", alpha = 1)
  plot(cv_model)
  
  # Extract the best lambda and coefficients
  best_lambda <- cv_model$lambda.min
  best_coefs <- coef(cv_model, s = "lambda.min")
  
  # Convert coefficients to a dense matrix and then to a data frame
  best_coefs_dense <- as.matrix(best_coefs)
  rownames(best_coefs_dense) <- make.unique(rownames(best_coefs_dense))  # Ensure unique row names
  best_coefs_df <- data.frame(Coefficient = best_coefs_dense[, 1], row.names = rownames(best_coefs_dense))
  
  # Add absolute coefficient values
  best_coefs_df$AbsoluteCoefficient <- abs(best_coefs_df$Coefficient)
  
  # Filter for relevant variables (non-zero coefficients and not the intercept)
  relevant_variables <- best_coefs_df %>%
    filter(Coefficient != 0 & row.names(best_coefs_df) != "(Intercept)")
  
  # Sort by absolute value of coefficients to identify the most influential variables
  most_influential_vars <- best_coefs_df %>%
    filter(Coefficient != 0 & row.names(best_coefs_df) != "(Intercept)") %>%
    arrange(desc(AbsoluteCoefficient))
  
  # Extract the top 10 most influential variables
  top10_vars <- head(row.names(most_influential_vars), 10)
  
  list(relevant_variables = relevant_variables, top10_vars = top10_vars)
}

# Apply LASSO for each emissions variable

# Select only numeric columns
numeric_cols <- sapply(df, is.numeric)
numeric_df <- df[, numeric_cols, with = FALSE]

# Identify and remove emissions columns
emissions_pattern <- "(tC)"
emissions_vars <- grep(emissions_pattern, colnames(numeric_df), value = TRUE)
numeric_df_no_emissions <- numeric_df[, !colnames(numeric_df) %in% emissions_vars, with = FALSE]
numeric_df_no_emissions = numeric_df_no_emissions[,!colnames(numeric_df_no_emissions) %in% c('Unnamed: 0', 'FIPS', 'STATEFIPS'), with = FALSE]
results <- lapply(emissions_vars, function(var) perform_lasso(var, numeric_df_no_emissions,df))

is.not.null = function(x) !is.null(x)
# Print results for each emissions variable
for (i in 1:length(emissions_vars)) {
  cat("\nResults for", emissions_vars[i], ":\n")
  print(results[[i]]$relevant_variables)
  cat("Top 10 influential variables:\n")
  if (is.not.null(results[[i]]$top10_vars)) { print(results[[i]]$top10_vars)}
}

# Print results for each emissions variable
for (i in 1:length(emissions_vars)) {
  
  
  if (!is.null(results[[i]]$relevant_variables) && nrow(results[[i]]$relevant_variables) > 0) {
    print(results[[i]]$relevant_variables)
  } else {
    cat("No relevant variables found.\n")
  }
  
  cat("Top 10 influential variables:\n")
  if (!is.null(results[[i]]$top10_vars) && length(results[[i]]$top10_vars) > 0) {
    print(results[[i]]$top10_vars)
  } else {
    cat("No top 10 influential variables found.\n")
  }
}

for (i in 1:length(emissions_vars)) {
  if (!is.null(results[[i]]$relevant_variables) && nrow(results[[i]]$relevant_variables) > 0) {
    cat("\nResults for", emissions_vars[i], ":\n")
    print(results[[i]]$relevant_variables)
    print(results[[i]]$top10_vars)
  }
}


########## BUILDING MODELS BASED ON DOUBLE LASSO VARIABLE SELECTION

## Linear Regressions
linear_regression_results <- list()
glm_regression_results <- list()


set.seed(123)
ntest = floor(nrow(df)/3)  # 1/3 of data is test
testid = sample(1:nrow(df), ntest)  # indices of test obs

testset = df[testid]
trainset = df[-testid]
linear_mse = list()
glm_mse = list()

for (i in 1:length(emissions_vars)) {
  response_var <- emissions_vars[i]
  print(response_var)
  relevant_vars <- results[[i]]$relevant_variables
  
  if (!is.null(results[[i]]$relevant_variables) && nrow(results[[i]]$relevant_variables) > 0) {
    # Create a formula for the regression model
    formula <- as.formula(paste("`",response_var, "` ~", paste('`',row.names(relevant_vars),'`', collapse = " + ", sep = ''),sep = ''))
    
    # Fit the regression model
    linear_regression_model <- lm(formula, data = trainset)
    glm_regression_model <- glm(formula,data = trainset)
    
    # Find MSE's
    actual = testset[[paste0(response_var)]]
    linear_preds = predict(linear_regression_model,testset)
    glm_preds = predict(glm_regression_model,testset)
    linear_mse[[response_var]] <- mean((linear_preds - actual)^2)
    glm_mse[[response_var]] <- mean((glm_preds - actual)^2)
    
    # Store the regression model results
    linear_regression_results[[response_var]] <- summary(linear_regression_model)
    glm_regression_results[[response_var]] <- summary(glm_regression_model)
  }
}


for (result in linear_regression_results) {
  print(result)  # Print each regression result
}

for (result in glm_regression_results) {
  print(result)  # Print each regression result
}

# Convert the list to a numeric vector
linear_mse_vector <- unlist(linear_mse)
glm_mse_vector <- unlist(linear_mse)
print(glm_mse)
print(linear_mse)

# Calculate the mean, excluding NA values
mean_linear_mse <- mean(linear_mse_vector, na.rm = TRUE)
mean_glm_mse <- mean(glm_mse_vector, na.rm = TRUE)

# Print the result
print(mean_linear_mse)
print(mean_glm_mse)

###########################
### Ridge/LASSO Regressions

ridge_lambdas = list()
lasso_lambdas = list()
lasso_models = list()
ridge_models = list()
ridge_results = list()
lasso_results = list()
ridge_mse = list()
lasso_mse = list()
ridge_rsquared = list()
lasso_rsquared = list()

library(glmnet)
library(dplyr)


for (i in 1:length(emissions_vars)) {
  response_var <- emissions_vars[i]
  print(response_var)
  relevant_vars <- results[[i]]$relevant_variables
  
  if (!is.null(relevant_vars) && nrow(relevant_vars) > 0) {
    # Define response variable
    y <- trainset[[ response_var]]
    
    # Check if y is a valid vector
    if (!is.vector(y) || length(y) == 0) {
      warning(paste("Response variable", response_var, "is not a valid vector; skipping."))
      next
    }
    
    # Define matrix of predictor variables
    x <- as.matrix(select(trainset, rownames(relevant_vars)))
    
    print(dim(x))
    print(length(y))

    # Check for NA values
    if (any(is.na(y))) {
      warning(paste("Response variable", response_var, "has NA values; skipping."))
      next
    }
    if (any(is.na(x))) {
      warning(paste("Predictor variables for", response_var, "have NA values; skipping."))
      next
    }
    
    # Perform k-fold cross-validation to find optimal lambda value (LASSO and Ridge)
    ridge_model <- cv.glmnet(x, y, alpha = 0)
    lasso_model <- cv.glmnet(x, y, alpha = 1)
    
    # Find the best lambda for each model
    ridge_best_lambda <- ridge_model$lambda.min
    lasso_best_lambda <- lasso_model$lambda.min
    
    # Store best lambdas and models
    ridge_lambdas[[response_var]] <- ridge_best_lambda
    lasso_lambdas[[response_var]] <- lasso_best_lambda
    lasso_models[[response_var]] <- lasso_model
    ridge_models[[response_var]] <- ridge_model
    
    # Model regressions using best lambda
    lasso_best_model <- glmnet(x, y, alpha = 1, lambda = lasso_best_lambda)
    ridge_best_model <- glmnet(x, y, alpha = 0, lambda = ridge_best_lambda)
    
    # Store regression results
    lasso_results[[response_var]] <- coef(lasso_best_model)
    ridge_results[[response_var]] <- coef(ridge_best_model)
    
    # Find MSE's
    actual <- testset[[response_var]]
    lasso_preds <- predict(lasso_best_model, s = lasso_best_lambda, newx = as.matrix(select(testset, rownames(relevant_vars))))
    ridge_preds <- predict(ridge_best_model, s = ridge_best_lambda, newx = as.matrix(select(trainset, rownames(relevant_vars))))
    
    ss_res_lasso <- sum((actual - lasso_preds)^2)
    ss_tot_lasso <- sum((actual - mean(actual))^2)
    lasso_rsquared[[response_var]] <- 1 - (ss_res_lasso / ss_tot_lasso)
    
    
    # Calculate MSE, ensuring no NA values
    ridge_mse[[response_var]] <- mean((ridge_preds - actual)^2, na.rm = TRUE)
    lasso_mse[[response_var]] <- mean((lasso_preds - actual)^2, na.rm = TRUE)
  }
}

for (result in lasso_results) {
  print(result)
}
for (result in ridge_results) {
  print(result)
}

# Convert the list to a numeric vector
ridge_mse_vector <- unlist(ridge_mse)
lasso_mse_vector <- unlist(lasso_mse)
print(ridge_mse)
print(lasso_mse)

# Calculate the mean, excluding NA values
mean_ridge_mse <- mean(ridge_mse_vector, na.rm = TRUE)
mean_lasso_mse <- mean(lasso_mse_vector, na.rm = TRUE)

# Print the result
print(mean_ridge_mse)
print(mean_lasso_mse)
print(mean_linear_mse)
print(mean_glm_mse)

# BEST TO WORST SO FAR:
# 1. LASSO Regression
# 2. GLM Regression
# 3. Linear Regression
# 4. Ridge Regresion


### Trying an NN with different layers

neural_net_mse = list()
neural_net_models = list()

# Define a function to perform neural network modeling
perform_neural_net <- function(response_var, trainset, testset, relevant_vars) {
  set.seed(123)
  
  # Define response variable
  y <- trainset[[response_var]]
  
  # Define matrix of predictor variables
  x <- as.matrix(select(trainset, rownames(relevant_vars)))
  xcolnum = ncol(x)
  
  # Define the neural network model
  set.seed(123)
  model <- keras_model_sequential() %>%
    layer_dense(units=128, activation="relu", input_shape=xcolnum) %>%
    #layer_dropout(rate=0.4) %>%
    layer_dense(units=64, activation="relu") %>%
    #layer_dropout(rate=0.3) %>%
    layer_dense(units=32, activation="relu") %>%
    layer_dense(units=1)
  
  # Compile the model
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = 'adam',
    metrics = c('mean_squared_error')
  )
  
  # Fit the model
  set.seed(123)
  model %>% fit(x, y, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 0)
  
  # Make predictions on the test set
  x_test <- as.matrix(as.matrix(select(testset, rownames(relevant_vars))))
  predictions <- model %>% predict(x_test)
  
  # Calculate MSE
  actual <- testset[[response_var]]
  mse <- mean((predictions - actual)^2, na.rm = TRUE)
  
  return(list(model = model, predictions = predictions, mse = mse))
}

for (i in 1:length(emissions_vars)) {
  response_var <- emissions_vars[i]
  print(response_var)
  relevant_vars <- results[[i]]$relevant_variables
  
  if (!is.null(relevant_vars) && nrow(relevant_vars) > 0) {
    # Call the neural network function
    nn_results <- perform_neural_net(response_var, trainset, testset, relevant_vars)
    
    # Store the MSE
    neural_net_mse[[response_var]] <- nn_results$mse
    
    # Optionally, you can store the model if needed
    neural_net_models[[response_var]] <- nn_results$model
  }
}
print(neural_net_mse)

nn_mse_vector = unlist(neural_net_mse)
mean_nn_mse <- mean(nn_mse_vector, na.rm = TRUE)
print(mean_nn_mse)
print(mean_lasso_mse)

## Better than lasso with more layers
# MSE of [1] 1.334398e+13

print(dim(neural_net_models))


##################################################
########### USING PER CAPITA EMISSIONS DATA


#### Per Capita Linear/GLM Models


# Calculate per capita values for each response variable
# Remove commas from Population column and convert to numeric
df$Population <- as.numeric(gsub(",", "", df$Population))
per_capita_data <- df


# Optional: Check for any warnings or NA values after conversion
if (any(is.na(df$Population))) {
  warning("There are NA values in the Population column after conversion.")
}

for (response_var in emissions_vars) {
  per_capita_data[[paste0(response_var, "_per_capita")]] <- per_capita_data[[response_var]] / df$Population
}


# Apply LASSO for each emissions variable

# Select only numeric columns
pop_numeric_cols <- sapply(df, is.numeric)
pop_numeric_df <- per_capita_data[, pop_numeric_cols, with = FALSE]

# Identify and remove emissions columns
pop_emissions_pattern <- "(tC)"
pop_emissions_vars <- grep(pop_emissions_pattern, colnames(pop_numeric_df), value = TRUE)
pop_numeric_df_no_emissions <- pop_numeric_df[, !colnames(pop_numeric_df) %in% pop_emissions_vars, with = FALSE]
pop_numeric_df_no_emissions = pop_numeric_df_no_emissions[,!colnames(pop_numeric_df_no_emissions) %in% c('Unnamed: 0', 'FIPS', 'STATEFIPS'), with = FALSE]
pop_results <- lapply(pop_emissions_vars, function(var) perform_lasso(var, pop_numeric_df_no_emissions,per_capita_data))

is.not.null = function(x) !is.null(x)
# Print results for each emissions variable
for (i in 1:length(pop_emissions_vars)) {
  cat("\nResults for", pop_emissions_vars[i], ":\n")
  print(pop_results[[i]]$relevant_variables)
  cat("Top 10 influential variables:\n")
  if (is.not.null(pop_results[[i]]$top10_vars)) { print(pop_results[[i]]$top10_vars)}
}

# Print results for each emissions variable
for (i in 1:length(pop_emissions_vars)) {
  
  
  if (!is.null(pop_results[[i]]$relevant_variables) && nrow(pop_results[[i]]$relevant_variables) > 0) {
    print(pop_results[[i]]$relevant_variables)
  } else {
    cat("No relevant variables found.\n")
  }
  
  cat("Top 10 influential variables:\n")
  if (!is.null(results[[i]]$top10_vars) && length(results[[i]]$top10_vars) > 0) {
    print(results[[i]]$top10_vars)
  } else {
    cat("No top 10 influential variables found.\n")
  }
}

for (i in 1:length(emissions_vars)) {
  if (!is.null(pop_results[[i]]$relevant_variables) && nrow(pop_results[[i]]$relevant_variables) > 0) {
    cat("\nResults for", pop_emissions_vars[i], ":\n")
    print(pop_results[[i]]$relevant_variables)
    print(pop_results[[i]]$top10_vars)
  }
}



## Linear Regressions
pop_linear_regression_results <- list()
pop_glm_regression_results <- list()

set.seed(123)
ntest = floor(nrow(per_capita_data) / 4)  # 1/3 of data is test
testid = sample(1:nrow(per_capita_data), ntest)  # indices of test obs

testset = per_capita_data[testid, ]
trainset = per_capita_data[-testid, ]
pop_linear_mse = list()
pop_glm_mse = list()

for (i in 1:length(pop_emissions_vars)) {
  response_var <- pop_emissions_vars[i]
  response_var_per_capita <- paste0(response_var, "_per_capita")
  print(response_var_per_capita)
  relevant_vars <- pop_results[[i]]$relevant_variables
  
  if (!is.null(pop_results[[i]]$relevant_variables) && nrow(pop_results[[i]]$relevant_variables) > 0) {
    # Create a formula for the regression model using per capita values
    formula <- as.formula(paste("`", response_var_per_capita, "` ~", paste('`', row.names(relevant_vars), '`', collapse = " + ", sep = ''), sep = ''))
    
    # Fit the regression model
    pop_linear_regression_model <- lm(formula, data = trainset)
    pop_glm_regression_model <- glm(formula, data = trainset)
    
    # Find MSE's
    actual = testset[[response_var_per_capita]]
    pop_linear_preds = predict(pop_linear_regression_model, testset)
    pop_glm_preds = predict(pop_glm_regression_model, testset)
    pop_linear_mse[[response_var]] <- mean((pop_linear_preds - actual)^2)
    pop_glm_mse[[response_var]] <- mean((pop_glm_preds - actual)^2)
    
    # Store the regression model results
    pop_linear_regression_results[[response_var]] <- summary(pop_linear_regression_model)
    pop_glm_regression_results[[response_var]] <- summary(pop_glm_regression_model)
  }
}

for (result in pop_linear_regression_results) {
  print(result)  # Print each regression result
}

for (result in pop_glm_regression_results) {
  print(result)  # Print each regression result
}

# Convert the list to a numeric vector
pop_linear_mse_vector <- unlist(pop_linear_mse)
pop_glm_mse_vector <- unlist(pop_glm_mse)
print(pop_glm_mse)
print(pop_linear_mse)

# Calculate the mean, excluding NA values
pop_mean_linear_mse <- mean(pop_linear_mse_vector, na.rm = TRUE)
pop_mean_glm_mse <- mean(pop_glm_mse_vector, na.rm = TRUE)

# Print the result
print(pop_mean_linear_mse)
print(pop_mean_glm_mse)

## Much easier to look at

##########

###########################
### Ridge/LASSO Regressions for Per Capita Data

pop_ridge_lambdas = list()
pop_lasso_lambdas = list()
pop_lasso_models = list()
pop_ridge_models = list()
pop_ridge_results = list()
pop_lasso_results = list()
pop_ridge_mse = list()
pop_lasso_mse = list()
pop_lasso_rsquared = list()
pop_ridge_rsquared = list()

library(glmnet)
library(dplyr)

set.seed(123)
ntest = floor(nrow(per_capita_data) / 3)  # 1/3 of data is test
testid = sample(1:nrow(per_capita_data), ntest)  # indices of test obs

pop_testset = per_capita_data[testid, ]
pop_trainset = per_capita_data[-testid, ]
print(colnames(pop_trainset))

for (i in 1:length(pop_emissions_vars)) {
  response_var <- pop_emissions_vars[i]
  response_var_per_capita <- paste0(response_var, "_per_capita")
  print(response_var_per_capita)
  relevant_vars <- pop_results[[i]]$relevant_variables
  
  if (!is.null(relevant_vars) && nrow(relevant_vars) > 0) {
    # Define response variable
    y <- pop_trainset[[response_var_per_capita]]
    
    # Check if y is a valid vector
    if (!is.vector(y) || length(y) == 0) {
      warning(paste("Response variable", response_var_per_capita, "is not a valid vector; skipping."))
      next
    }
    
    # Define matrix of predictor variables
    print('before')
    x <- as.matrix(select(pop_trainset, rownames(relevant_vars)))
    print('after')
    
    print(dim(x))
    print(length(y))
    
    # Check for NA values
    if (any(is.na(y))) {
      warning(paste("Response variable", response_var_per_capita, "has NA values; skipping."))
      next
    }
    if (any(is.na(x))) {
      warning(paste("Predictor variables for", response_var_per_capita, "have NA values; skipping."))
      next
    }
    
    # Perform k-fold cross-validation to find optimal lambda value (LASSO and Ridge)
    pop_ridge_model <- cv.glmnet(x, y, alpha = 0)
    pop_lasso_model <- cv.glmnet(x, y, alpha = 1)
    
    # Find the best lambda for each model
    pop_ridge_best_lambda <- pop_ridge_model$lambda.min
    pop_lasso_best_lambda <- pop_lasso_model$lambda.min
    
    # Store best lambdas and models
    pop_ridge_lambdas[[response_var_per_capita]] <- pop_ridge_best_lambda
    pop_lasso_lambdas[[response_var_per_capita]] <- pop_lasso_best_lambda
    pop_lasso_models[[response_var_per_capita]] <- pop_lasso_model
    pop_ridge_models[[response_var_per_capita]] <- pop_ridge_model
    
    # Model regressions using best lambda
    pop_lasso_best_model <- glmnet(x, y, alpha = 1, lambda = pop_lasso_best_lambda)
    pop_ridge_best_model <- glmnet(x, y, alpha = 0, lambda = pop_ridge_best_lambda)
    
    # Store regression results
    pop_lasso_results[[response_var_per_capita]] <- coef(pop_lasso_best_model)
    pop_ridge_results[[response_var_per_capita]] <- coef(pop_ridge_best_model)
    
    # Find MSE's
    actual <- pop_testset[[response_var_per_capita]]
    pop_lasso_preds <- predict(pop_lasso_best_model, s = pop_lasso_best_lambda, newx = as.matrix(select(testset, rownames(relevant_vars))))
    pop_ridge_preds <- predict(pop_ridge_best_model, s = pop_ridge_best_lambda, newx = as.matrix(select(testset, rownames(relevant_vars))))
    
    # Calculate MSE, ensuring no NA values
    pop_ridge_mse[[response_var_per_capita]] <- mean((pop_ridge_preds - actual)^2, na.rm = TRUE)
    pop_lasso_mse[[response_var_per_capita]] <- mean((pop_lasso_preds - actual)^2, na.rm = TRUE)
    
    ss_res_lasso <- sum((actual - pop_lasso_preds)^2)
    ss_tot_lasso <- sum((actual - mean(actual))^2)
    pop_lasso_rsquared[[response_var_per_capita]] <- 1 - (ss_res_lasso / ss_tot_lasso)
    
    ss_res_ridge <- sum((actual - pop_ridge_preds)^2)
    ss_tot_ridge <- sum((actual - mean(actual))^2)
    pop_ridge_rsquared[[response_var_per_capita]] <- 1 - (ss_res_ridge / ss_tot_ridge)
  }
}

for (result in pop_lasso_results) {
  print(result)
}
for (result in pop_ridge_results) {
  print(result)
}

# Convert the list to a numeric vector
pop_ridge_mse_vector <- unlist(pop_ridge_mse)
pop_lasso_mse_vector <- unlist(pop_lasso_mse)
print(pop_ridge_mse)
print(pop_lasso_mse)

# Calculate the mean, excluding NA values
pop_mean_ridge_mse <- mean(pop_ridge_mse_vector, na.rm = TRUE)
pop_mean_lasso_mse <- mean(pop_lasso_mse_vector, na.rm = TRUE)

# Print the result
print(pop_mean_ridge_mse)
print(pop_mean_lasso_mse)
print(pop_mean_linear_mse)
print(pop_mean_glm_mse)

## Very even between models, with glm and linear having a slight edge


print(pop_ridge_rsquared)
print(pop_lasso_rsquared)

##############################
### Neural Net

pop_neural_net_mse = list()
pop_neural_net_models = list()
pop_nn_rsquared = list()

# Define a function to perform neural network modeling for per capita data
perform_pop_neural_net <- function(response_var, trainset, testset, relevant_vars) {
  set.seed(123)
  
  # Define response variable
  y <- trainset[[response_var]]
  
  # Define matrix of predictor variables
  x <- as.matrix(select(trainset, rownames(relevant_vars)))
  xcolnum = ncol(x)
  
  # Define the neural network model
  modelnn <- keras_model_sequential() %>%
    layer_dense(units=256, activation="relu", input_shape = c(xcolnum)) %>%
    layer_dropout(rate=0.4) %>%
    layer_dense(units=128, activation="relu") %>%
    layer_dropout(rate=0.4) %>%
    layer_dense(units=64, activation="relu") %>%
    layer_dropout(rate=0.3) %>%
    layer_dense(units=32, activation="relu") %>%
    layer_dropout(rate=0.2) %>%
    layer_dense(units=16, activation="relu") %>%
    layer_dropout(rate=0.1) %>%
    layer_dense(units=8, activation="relu") %>%
    layer_dropout(rate=0.1) %>%
    layer_dense(units=1)
  
  # Compile the model
  modelnn %>% compile(
    loss = 'mean_squared_error',
    optimizer = 'adam',
    metrics = c('mean_squared_error')
  )
  
  # Fit the model
  modelnn %>% fit(x, y, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 0)
  
  # Make predictions on the test set
  x_test <- as.matrix(select(testset, rownames(relevant_vars)))
  predictions <- modelnn %>% predict(x_test)
  
  # Calculate MSE
  actual <- testset[[response_var]]
  mse <- mean((predictions - actual)^2, na.rm = TRUE)
  # Assuming `predictions` contains your neural network predictions and `actual` contains the true values
  
  ss_res <- sum((actual - predictions)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  rsquared <- 1 - (ss_res / ss_tot)
  
  return(list(model = modelnn, predictions = predictions, mse = mse , rsquared = rsquared))
}


for (i in 1:length(pop_emissions_vars)) {
  response_var <- pop_emissions_vars[i]
  response_var_per_capita <- paste0(response_var, "_per_capita")
  print(response_var_per_capita)
  relevant_vars <- pop_results[[i]]$relevant_variables
  
  if (!is.null(relevant_vars) && nrow(relevant_vars) > 0) {
    # Call the neural network function
    nn_results <- perform_pop_neural_net(response_var_per_capita, pop_trainset, pop_testset, relevant_vars)
    
    # Store the MSE
    pop_neural_net_mse[[response_var_per_capita]] <- nn_results$mse
    
    # Optionally, you can store the model if needed
    pop_neural_net_models[[response_var_per_capita]] <- nn_results$model
    
    # Assuming `predictions` contains your neural network predictions and `actual` contains the true values

    pop_nn_rsquared[[response_var_per_capita]] <- nn_results$rsquared

  }
}
print(pop_neural_net_mse)

pop_nn_mse_vector = unlist(pop_neural_net_mse)
pop_mean_nn_mse <- mean(pop_nn_mse_vector, na.rm = TRUE)
print(pop_mean_nn_mse)
print(pop_mean_lasso_mse)
print(pop_mean_ridge_mse)
print(pop_mean_linear_mse)
print(pop_mean_glm_mse)

## In order:
# 1. Linear/GLM
# 2. Ridge
# 3. LASSO
# 4. NN

# This could be because with the change from total C02 to average C02 Per Capita, data became more linear
# Going to interperet Linear/GLM model, as well as possibly LASSO

print(pop_glm_regression_results)
print(pop_linear_regression_results)

print(pop_nn_rsquared)

print(pop_lasso_best_model)
print(pop_lasso_results)





##### All R Squareds are Low, so trying more variables


numeric_cols <- sapply(per_capita_data, is.numeric)
numeric_df <- per_capita_data[, numeric_cols, with = FALSE]

# Identify and remove emissions columns
emissions_pattern <- "(tC)"
emissions_vars <- grep(emissions_pattern, colnames(numeric_df), value = TRUE)
numeric_df_no_emissions <- numeric_df[, !colnames(numeric_df) %in% emissions_vars, with = FALSE]
numeric_df_no_emissions <- numeric_df_no_emissions[, !colnames(numeric_df_no_emissions) %in% c('Unnamed: 0', 'FIPS', 'STATEFIPS','Population'), with = FALSE]

print(colnames(numeric_df_no_emissions))

# Store column names as a matrix
numcols <- colnames(numeric_df_no_emissions)

library(glmnet)
library(dplyr)

pop_neural_net_mse = list()
pop_neural_net_models = list()
pop_nn_rsquared = list()

# Define a function to perform neural network modeling for per capita data
perform_pop_neural_net <- function(response_var, trainset, testset, relevant_vars) {
  set.seed(123)
  
  # Define response variable
  y <- trainset[[response_var]]
  
  # Define matrix of predictor variables
  x <- as.matrix(select(trainset, relevant_vars))
  xcolnum = ncol(x)
  
  # Define the neural network model
  modelnn <- keras_model_sequential() %>%
    layer_dense(units=256, activation="relu", input_shape = c(xcolnum)) %>%
    layer_dropout(rate=0.4) %>%
    layer_dense(units=128, activation="relu") %>%
    layer_dropout(rate=0.4) %>%
    layer_dense(units=64, activation="relu") %>%
    layer_dropout(rate=0.3) %>%
    layer_dense(units=32, activation="relu") %>%
    layer_dropout(rate=0.2) %>%
    layer_dense(units=16, activation="relu") %>%
    layer_dropout(rate=0.1) %>%
    layer_dense(units=8, activation="relu") %>%
    layer_dropout(rate=0.1) %>%
    layer_dense(units=1)
  
  # Compile the model
  modelnn %>% compile(
    loss = 'mean_squared_error',
    optimizer = 'adam',
    metrics = c('mean_squared_error')
  )
  
  # Fit the model
  modelnn %>% fit(x, y, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 0)
  
  # Make predictions on the test set
  x_test <- as.matrix(select(testset, relevant_vars))
  predictions <- modelnn %>% predict(x_test)
  
  # Calculate MSE
  actual <- testset[[response_var]]
  mse <- mean((predictions - actual)^2, na.rm = TRUE)
  # Assuming `predictions` contains your neural network predictions and `actual` contains the true values
  
  ss_res <- sum((actual - predictions)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  rsquared <- 1 - (ss_res / ss_tot)
  
  return(list(model = modelnn, predictions = predictions, mse = mse , rsquared = rsquared))
}


for (i in 1:length(pop_emissions_vars)) {
  response_var <- pop_emissions_vars[i]
  response_var_per_capita <- paste0(response_var, "_per_capita")
  print(response_var_per_capita)
  relevant_vars <- emiscols
  
    # Call the neural network function
  nn_results <- perform_pop_neural_net(response_var_per_capita, pop_trainset, pop_testset, relevant_vars)
    
    # Store the MSE
  pop_neural_net_mse[[response_var_per_capita]] <- nn_results$mse
    
    # Optionally, you can store the model if needed
  pop_neural_net_models[[response_var_per_capita]] <- nn_results$model
    
    # Assuming `predictions` contains your neural network predictions and `actual` contains the true values
    
  pop_nn_rsquared[[response_var_per_capita]] <- nn_results$rsquared
    
}
print(pop_neural_net_mse)

pop_nn_mse_vector = unlist(pop_neural_net_mse)
pop_mean_nn_mse <- mean(pop_nn_mse_vector, na.rm = TRUE)
print(pop_mean_nn_mse)
print(pop_mean_lasso_mse)
print(pop_mean_ridge_mse)
print(pop_mean_linear_mse)
print(pop_mean_glm_mse)

print(pop_nn_rsquared)



## Even worse R Squared








## Linear Regressions
pop_linear_regression_results <- list()
pop_glm_regression_results <- list()

set.seed(123)
ntest = floor(nrow(per_capita_data) / 4)  # 1/3 of data is test
testid = sample(1:nrow(per_capita_data), ntest)  # indices of test obs

testset = per_capita_data[testid, ]
trainset = per_capita_data[-testid, ]
pop_linear_mse = list()
pop_glm_mse = list()
library(car)



# Remove leading/trailing whitespace from column names

for (i in 1:length(pop_emissions_vars)) {
  response_var <- pop_emissions_vars[i]
  response_var_per_capita <- paste0(response_var, "_per_capita")
  print(response_var_per_capita)
  
  # Check if response variable exists in trainset
  if (!response_var_per_capita %in% colnames(trainset)) {
    warning(paste("Response variable", response_var_per_capita, "not found in trainset."))
    next
  }

  # Create a formula for the regression model using per capita values
  formula <- as.formula(paste("`", response_var_per_capita, "` ~", paste('`', new_numcols, '`', collapse = " + ",sep = ''), sep = ''))
  
  # Fit the regression model
  pop_linear_regression_model <- lm(formula, data = trainset)
  
  # Check VIF and handle aliased coefficients
  #vif_values <- vif(pop_linear_regression_model)
  #if (any(is.infinite(vif_values))) {
  #  warning("Aliased coefficients found; consider adjusting the model.")
  #  next  # Skip to the next variable
  #}
  
  # Calculate MSE and store results
  actual = testset[[response_var_per_capita]]
  pop_linear_preds = predict(pop_linear_regression_model, testset)
  pop_linear_mse[[response_var]] <- mean((pop_linear_preds - actual)^2)
  
  # Store the regression model results
  pop_linear_regression_results[[response_var]] <- summary(pop_linear_regression_model)
}


for (result in pop_linear_regression_results) {
  print(result$r.squared)  # Print each regression result
}

for (result in pop_glm_regression_results) {
  print(result)  # Print each regression result
}

# Convert the list to a numeric vector
pop_linear_mse_vector <- unlist(pop_linear_mse)
pop_glm_mse_vector <- unlist(pop_glm_mse)
print(pop_glm_mse)
print(pop_linear_mse)

# Calculate the mean, excluding NA values
pop_mean_linear_mse <- mean(pop_linear_mse_vector, na.rm = TRUE)
pop_mean_glm_mse <- mean(pop_glm_mse_vector, na.rm = TRUE)

# Print the result
print(pop_mean_linear_mse)
print(pop_mean_glm_mse)

## Much easier to look at



############## LOOKING AT MULTICOLLINEARITY


# Convert relevant columns to a matrix
numcols_matrix <- as.matrix(trainset[, ..numcols, with = FALSE])

# Calculate the correlation matrix
correlation_matrix <- cor(numcols_matrix, use = "pairwise.complete.obs")
print(correlation_matrix)

threshold <- 0.99  # Set your correlation threshold

# Find indices of correlated pairs
correlated_indices <- which(abs(correlation_matrix) > threshold, arr.ind = TRUE)

# Remove self-correlations (where variable is correlated with itself)
correlated_pairs <- correlated_indices[correlated_indices[, 1] != correlated_indices[, 2], ]

# Get the names of the correlated variable pairs
correlated_var_pairs <- data.frame(
  Variable1 = rownames(correlation_matrix)[correlated_pairs[, 1]],
  Variable2 = colnames(correlation_matrix)[correlated_pairs[, 2]],
  Correlation = correlation_matrix[correlated_pairs]
)

# Print the correlated variable pairs
print(unique(correlated_var_pairs$Variable2))
new_numcols <- numcols[!(numcols %in% unique(correlated_var_pairs$Variable2))]







numcols_matrix <- as.matrix(trainset[, ..new_numcols, with = FALSE])



# Calculate the correlation matrix
correlation_matrix <- cor(numcols_matrix, use = "pairwise.complete.obs")
print(correlation_matrix)

threshold <- 0.9  # Set your correlation threshold

# Find indices of correlated pairs
correlated_indices <- which(abs(correlation_matrix) > threshold, arr.ind = TRUE)

# Remove self-correlations (where variable is correlated with itself)
correlated_pairs <- correlated_indices[correlated_indices[, 1] != correlated_indices[, 2], ]

# Get the names of the correlated variable pairs
correlated_var_pairs <- data.frame(
  Variable1 = rownames(correlation_matrix)[correlated_pairs[, 1]],
  Variable2 = colnames(correlation_matrix)[correlated_pairs[, 2]],
  Correlation = correlation_matrix[correlated_pairs]
)



# Count occurrences of each variable in Variable1
frequency_table <- table(correlated_var_pairs$Variable1)

# Find the variable with the maximum frequency
most_frequent_variable <- names(frequency_table)[which.max(frequency_table)]
most_frequent_count <- max(frequency_table)
# Print the result
cat("Most frequent variable:", most_frequent_variable, "appears", most_frequent_count, "times.\n")

print(length(unique(rownames(frequency_table))))
print(unique(rownames(frequency_table)))

high_freq <- names(frequency_table[frequency_table > 50])
print(high_freq)
new_numcols <- numcols[!(numcols %in% high_freq)]

new_numcols <- new_numcols[!(new_numcols %in% high_freq)]














# Convert relevant columns to a matrix
new_numcols = numcols
thresholds= c(0.99,0.95,0.9,0.85,0.8)
print(length(new_numcols))
for (t in thresholds) {
  
  numcols_matrix <- as.matrix(trainset[, ..new_numcols, with = FALSE])
  
  # Calculate the correlation matrix
  correlation_matrix <- cor(numcols_matrix, use = "pairwise.complete.obs")
  #print(correlation_matrix)
  
  threshold <- t  # Set your correlation threshold
  
  # Find indices of correlated pairs
  correlated_indices <- which(abs(correlation_matrix) > threshold, arr.ind = TRUE)
  
  # Remove self-correlations (where variable is correlated with itself)
  correlated_pairs <- correlated_indices[correlated_indices[, 1] != correlated_indices[, 2], ]
  
  # Get the names of the correlated variable pairs
  correlated_var_pairs <- data.frame(
    Variable1 = rownames(correlation_matrix)[correlated_pairs[, 1]],
    Variable2 = colnames(correlation_matrix)[correlated_pairs[, 2]],
    Correlation = correlation_matrix[correlated_pairs]
  )
  
  # Count occurrences of each variable in Variable1
  frequency_table <- table(correlated_var_pairs$Variable1)
  
  # Find the variable with the maximum frequency
  most_frequent_variable <- names(frequency_table)[which.max(frequency_table)]
  most_frequent_count <- max(frequency_table)
  # Print the result
  cat("Most frequent variable:", most_frequent_variable, "appears", most_frequent_count, "times.\n")
  
  ##print(length(unique(rownames(frequency_table))))
  #print(unique(rownames(frequency_table)))
  
  limit_freq = 110-100*t
  high_freq <- names(frequency_table[frequency_table > limit_freq])
  print(high_freq)

  new_numcols <- new_numcols[!(new_numcols %in% high_freq)]
  print(length(new_numcols))
  
  
}




## After doing this, much better R Squareds.
# Going to do DOUBLE LASSO VARIABLE SELECTION



library(glmnet)
# Define a function to perform LASSO for a given response variable
perform_lasso <- function(response_var, df, fulldata) {
  y_numeric <- fulldata[[response_var]]
  
  if (sd(y_numeric) == 0) {
    warning(paste("Response variable", response_var, "is constant; skipping."))
    return(list(relevant_variables = NULL, top10_vars = NULL))
  }
  # Prepare the predictor matrix (X) from df excluding the response variable
  x <- as.matrix(df)
  
  # Set seed for reproducibility
  set.seed(123)
  
  # Fit the Lasso model
  lasso_model <- glmnet(x, y_numeric, family = "gaussian", alpha = 1)
  
  # Cross-validation for Lasso
  set.seed(123)
  cv_model <- cv.glmnet(x, y_numeric, family = "gaussian", alpha = 1)
  plot(cv_model)
  
  # Extract the best lambda and coefficients
  best_lambda <- cv_model$lambda.min
  best_coefs <- coef(cv_model, s = "lambda.min")
  
  # Convert coefficients to a dense matrix and then to a data frame
  best_coefs_dense <- as.matrix(best_coefs)
  rownames(best_coefs_dense) <- make.unique(rownames(best_coefs_dense))  # Ensure unique row names
  best_coefs_df <- data.frame(Coefficient = best_coefs_dense[, 1], row.names = rownames(best_coefs_dense))
  
  # Add absolute coefficient values
  best_coefs_df$AbsoluteCoefficient <- abs(best_coefs_df$Coefficient)
  
  # Filter for relevant variables (non-zero coefficients and not the intercept)
  relevant_variables <- best_coefs_df %>%
    filter(Coefficient != 0 & row.names(best_coefs_df) != "(Intercept)")
  
  # Sort by absolute value of coefficients to identify the most influential variables
  most_influential_vars <- best_coefs_df %>%
    filter(Coefficient != 0 & row.names(best_coefs_df) != "(Intercept)") %>%
    arrange(desc(AbsoluteCoefficient))
  
  # Extract the top 10 most influential variables
  top10_vars <- head(row.names(most_influential_vars), 10)
  
  list(relevant_variables = relevant_variables, top10_vars = top10_vars)
}

# Apply LASSO for each emissions variable

# Select only numeric columns
numeric_cols <- sapply(per_capita_data, is.numeric)
numeric_df <- per_capita_data[, numeric_cols, with = FALSE]

# Identify and remove emissions columns
emissions_pattern <- "(tC)"
emissions_vars <- grep(emissions_pattern, colnames(numeric_df), value = TRUE)
numeric_df_no_emissions <- numeric_df[, !colnames(numeric_df) %in% emissions_vars, with = FALSE]
numeric_df_no_emissions = numeric_df_no_emissions[,colnames(numeric_df_no_emissions) %in% new_numcols, with = FALSE]
pop_results <- lapply(emissions_vars, function(var) perform_lasso(var, numeric_df_no_emissions,per_capita_data))

is.not.null = function(x) !is.null(x)
# Print results for each emissions variable
for (i in 1:length(emissions_vars)) {
  cat("\nResults for", emissions_vars[i], ":\n")
  print(pop_results[[i]]$relevant_variables)
  cat("Top 10 influential variables:\n")
  if (is.not.null(pop_results[[i]]$top10_vars)) { print(pop_results[[i]]$top10_vars)}
}

# Print results for each emissions variable
for (i in 1:length(emissions_vars)) {
  
  
  if (!is.null(pop_results[[i]]$relevant_variables) && nrow(pop_results[[i]]$relevant_variables) > 0) {
    print(pop_results[[i]]$relevant_variables)
  } else {
    cat("No relevant variables found.\n")
  }
  
  cat("Top 10 influential variables:\n")
  if (!is.null(pop_results[[i]]$top10_vars) && length(pop_results[[i]]$top10_vars) > 0) {
    print(pop_results[[i]]$top10_vars)
  } else {
    cat("No top 10 influential variables found.\n")
  }
}

for (i in 1:length(emissions_vars)) {
  if (!is.null(pop_results[[i]]$relevant_variables) && nrow(pop_results[[i]]$relevant_variables) > 0) {
    cat("\nResults for", emissions_vars[i], ":\n")
    print(pop_results[[i]]$relevant_variables)
    print(pop_results[[i]]$top10_vars)
  }
}





## Linear Regressions
pop_linear_regression_results <- list()
pop_glm_regression_results <- list()

set.seed(123)
ntest = floor(nrow(per_capita_data) / 4)  # 1/4 of data is test
testid = sample(1:nrow(per_capita_data), ntest)  # indices of test obs

testset = per_capita_data[testid, ]
trainset = per_capita_data[-testid, ]
pop_linear_mse = list()
pop_glm_mse = list()

for (i in 1:length(pop_emissions_vars)) {
  response_var <- pop_emissions_vars[i]
  response_var_per_capita <- paste0(response_var, "_per_capita")
  print(response_var_per_capita)
  relevant_vars <- pop_results[[i]]$relevant_variables
  
  if (!is.null(pop_results[[i]]$relevant_variables) && nrow(pop_results[[i]]$relevant_variables) > 0) {
    # Create a formula for the regression model using per capita values
    formula <- as.formula(paste("`", response_var_per_capita, "` ~", paste('`', row.names(relevant_vars), '`', collapse = " + ", sep = ''), sep = ''))
    
    # Fit the regression model
    pop_linear_regression_model <- lm(formula, data = trainset)
    pop_glm_regression_model <- glm(formula, data = trainset)
    
    # Find MSE's
    actual = testset[[response_var_per_capita]]
    pop_linear_preds = predict(pop_linear_regression_model, testset)
    pop_glm_preds = predict(pop_glm_regression_model, testset)
    pop_linear_mse[[response_var]] <- mean((pop_linear_preds - actual)^2)
    pop_glm_mse[[response_var]] <- mean((pop_glm_preds - actual)^2)
    
    # Store the regression model results
    pop_linear_regression_results[[response_var]] <- summary(pop_linear_regression_model)
    pop_glm_regression_results[[response_var]] <- summary(pop_glm_regression_model)
  }
}

for (result in pop_linear_regression_results) {
  print(result)  # Print each regression result
}

for (result in pop_glm_regression_results) {
  print(result)  # Print each regression result
}

# Convert the list to a numeric vector
pop_linear_mse_vector <- unlist(pop_linear_mse)
pop_glm_mse_vector <- unlist(pop_glm_mse)
print(pop_glm_mse)
print(pop_linear_mse)

# Calculate the mean, excluding NA values
pop_mean_linear_mse <- mean(pop_linear_mse_vector, na.rm = TRUE)
pop_mean_glm_mse <- mean(pop_glm_mse_vector, na.rm = TRUE)

# Print the result
print(pop_mean_linear_mse)
print(pop_mean_glm_mse)

## Much easier to look at
## lower MSE and R Squareds compared to all variable Linear Regression





##### All R Squareds are Low, so trying more variables


# Select only numeric columns
numeric_cols <- sapply(per_capita_data, is.numeric)
numeric_df <- per_capita_data[, numeric_cols, with = FALSE]

# Identify and remove emissions columns
emissions_pattern <- "(tC)"
emissions_vars <- grep(emissions_pattern, colnames(numeric_df), value = TRUE)
numeric_df_no_emissions <- numeric_df[, !colnames(numeric_df) %in% emissions_vars, with = FALSE]
numeric_df_no_emissions = numeric_df_no_emissions[,colnames(numeric_df_no_emissions) %in% new_numcols, with = FALSE]

print(colnames(numeric_df_no_emissions))

# Store column names as a matrix
numcols <- colnames(numeric_df_no_emissions)

library(glmnet)
library(dplyr)

pop_neural_net_mse = list()
pop_neural_net_models = list()
pop_nn_rsquared = list()

set.seed(123)
ntest = floor(nrow(per_capita_data) / 4)  # 1/4 of data is test
testid = sample(1:nrow(per_capita_data), ntest)  # indices of test obs

pop_testset = per_capita_data[testid, ]
pop_trainset = per_capita_data[-testid, ]

# Define a function to perform neural network modeling for per capita data
perform_pop_neural_net <- function(response_var, trainset, testset, relevant_vars) {
  set.seed(123)
  
  # Define response variable
  y <- trainset[[response_var]]
  
  # Define matrix of predictor variables
  x <- as.matrix(select(trainset, relevant_vars))
  xcolnum = ncol(x)
  
  # Define the neural network model
  modelnn <- keras_model_sequential() %>%
    layer_dense(units=256, activation="relu", input_shape = c(xcolnum)) %>%
    layer_dropout(rate=0.4) %>%
    layer_dense(units=128, activation="relu") %>%
    layer_dropout(rate=0.4) %>%
    layer_dense(units=64, activation="relu") %>%
    layer_dropout(rate=0.3) %>%
    layer_dense(units=32, activation="relu") %>%
    layer_dropout(rate=0.2) %>%
    layer_dense(units=16, activation="relu") %>%
    layer_dropout(rate=0.1) %>%
    layer_dense(units=8, activation="relu") %>%
    layer_dropout(rate=0.1) %>%
    layer_dense(units=1)
  
  # Compile the model
  modelnn %>% compile(
    loss = 'mean_squared_error',
    optimizer = 'adam',
    metrics = c('mean_squared_error')
  )
  
  # Fit the model
  modelnn %>% fit(x, y, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 0)
  
  # Make predictions on the test set
  x_test <- as.matrix(select(testset, relevant_vars))
  predictions <- modelnn %>% predict(x_test)
  
  # Calculate MSE
  actual <- testset[[response_var]]
  mse <- mean((predictions - actual)^2, na.rm = TRUE)
  # Assuming `predictions` contains your neural network predictions and `actual` contains the true values
  
  ss_res <- sum((actual - predictions)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  rsquared <- 1 - (ss_res / ss_tot)
  
  return(list(model = modelnn, predictions = predictions, mse = mse , rsquared = rsquared))
}


for (i in 1:length(pop_emissions_vars)) {
  response_var <- pop_emissions_vars[i]
  response_var_per_capita <- paste0(response_var, "_per_capita")
  print(response_var_per_capita)
  relevant_vars <- new_numcols
  
  # Call the neural network function
  nn_results <- perform_pop_neural_net(response_var_per_capita, pop_trainset, pop_testset, relevant_vars)
  
  # Store the MSE
  pop_neural_net_mse[[response_var_per_capita]] <- nn_results$mse
  
  # Optionally, you can store the model if needed
  pop_neural_net_models[[response_var_per_capita]] <- nn_results$model
  
  # Assuming `predictions` contains your neural network predictions and `actual` contains the true values
  
  pop_nn_rsquared[[response_var_per_capita]] <- nn_results$rsquared
  
}
print(pop_neural_net_mse)

pop_nn_mse_vector = unlist(pop_neural_net_mse)
pop_mean_nn_mse <- mean(pop_nn_mse_vector, na.rm = TRUE)
print(pop_mean_nn_mse)
print(pop_mean_lasso_mse)
print(pop_mean_ridge_mse)
print(pop_mean_linear_mse)
print(pop_mean_glm_mse)
print(pop_nn_rsquared)







###########################
### Ridge/LASSO Regressions for Per Capita Data

pop_ridge_lambdas = list()
pop_lasso_lambdas = list()
pop_lasso_models = list()
pop_ridge_models = list()
pop_ridge_results = list()
pop_lasso_results = list()
pop_ridge_mse = list()
pop_lasso_mse = list()
pop_lasso_rsquared = list()
pop_ridge_rsquared = list()

library(glmnet)
library(dplyr)

set.seed(123)
ntest = floor(nrow(per_capita_data) / 3)  # 1/3 of data is test
testid = sample(1:nrow(per_capita_data), ntest)  # indices of test obs

pop_testset = per_capita_data[testid, ]
pop_trainset = per_capita_data[-testid, ]
print(colnames(pop_trainset))

numeric_cols <- sapply(per_capita_data, is.numeric)
numeric_df <- per_capita_data[, numeric_cols, with = FALSE]

# Identify and remove emissions columns
emissions_pattern <- "(tC)"
pop_emissions_vars <- grep(emissions_pattern, colnames(numeric_df), value = TRUE)
numeric_df_no_emissions <- numeric_df[, !colnames(numeric_df) %in% emissions_vars, with = FALSE]
numeric_df_no_emissions = numeric_df_no_emissions[,colnames(numeric_df_no_emissions) %in% new_numcols, with = FALSE]


for (i in 1:length(pop_emissions_vars)) {
  response_var <- pop_emissions_vars[i]
  response_var_per_capita <- paste0(response_var, "_per_capita")
  print(response_var_per_capita)
  relevant_vars <- new_numcols
  
  # Define response variable
  y <- pop_trainset[[response_var_per_capita]]
  
  # Check if y is a valid vector
  if (!is.vector(y) || length(y) == 0) {
    warning(paste("Response variable", response_var_per_capita, "is not a valid vector; skipping."))
    next
  }
  
  # Check if y has variability
  if (length(unique(y)) < 2) {
    warning(paste("Response variable", response_var_per_capita, "is constant; skipping."))
    next
  }
  
  # Define matrix of predictor variables
  x <- as.matrix(select(pop_trainset, relevant_vars))

  print(dim(x))
  print(length(y))
  
  # Check for NA values
  if (any(is.na(y))) {
    warning(paste("Response variable", response_var_per_capita, "has NA values; skipping."))
    next
  }
  if (any(is.na(x))) {
    warning(paste("Predictor variables for", response_var_per_capita, "have NA values; skipping."))
    next
  }
  
  # Perform k-fold cross-validation to find optimal lambda value (LASSO and Ridge)
  pop_ridge_model <- cv.glmnet(x, y, alpha = 0)
  pop_lasso_model <- cv.glmnet(x, y, alpha = 1)
  
  # Find the best lambda for each model
  pop_ridge_best_lambda <- pop_ridge_model$lambda.min
  pop_lasso_best_lambda <- pop_lasso_model$lambda.min
  
  # Store best lambdas and models
  pop_ridge_lambdas[[response_var_per_capita]] <- pop_ridge_best_lambda
  pop_lasso_lambdas[[response_var_per_capita]] <- pop_lasso_best_lambda
  pop_lasso_models[[response_var_per_capita]] <- pop_lasso_model
  pop_ridge_models[[response_var_per_capita]] <- pop_ridge_model
  
  # Model regressions using best lambda
  pop_lasso_best_model <- glmnet(x, y, alpha = 1, lambda = pop_lasso_best_lambda)
  pop_ridge_best_model <- glmnet(x, y, alpha = 0, lambda = pop_ridge_best_lambda)
  
  # Store regression results
  pop_lasso_results[[response_var_per_capita]] <- coef(pop_lasso_best_model)
  pop_ridge_results[[response_var_per_capita]] <- coef(pop_ridge_best_model)
  
  # Find MSE's
  actual <- pop_testset[[response_var_per_capita]]
  pop_lasso_preds <- predict(pop_lasso_best_model, s = pop_lasso_best_lambda, newx = as.matrix(select(pop_testset, relevant_vars)))
  pop_ridge_preds <- predict(pop_ridge_best_model, s = pop_ridge_best_lambda, newx = as.matrix(select(pop_testset, relevant_vars)))
  
  # Calculate MSE, ensuring no NA values
  pop_ridge_mse[[response_var_per_capita]] <- mean((pop_ridge_preds - actual)^2, na.rm = TRUE)
  pop_lasso_mse[[response_var_per_capita]] <- mean((pop_lasso_preds - actual)^2, na.rm = TRUE)
  
  ss_res_lasso <- sum((actual - pop_lasso_preds)^2)
  ss_tot_lasso <- sum((actual - mean(actual))^2)
  pop_lasso_rsquared[[response_var_per_capita]] <- 1 - (ss_res_lasso / ss_tot_lasso)
  
  ss_res_ridge <- sum((actual - pop_ridge_preds)^2)
  ss_tot_ridge <- sum((actual - mean(actual))^2)
  pop_ridge_rsquared[[response_var_per_capita]] <- 1 - (ss_res_ridge / ss_tot_ridge)
}



for (result in pop_lasso_results) {
  print(result)
}
for (result in pop_ridge_results) {
  print(result)
}

# Convert the list to a numeric vector
pop_ridge_mse_vector <- unlist(pop_ridge_mse)
pop_lasso_mse_vector <- unlist(pop_lasso_mse)
print(pop_ridge_mse)
print(pop_lasso_mse)

# Calculate the mean, excluding NA values
pop_mean_ridge_mse <- mean(pop_ridge_mse_vector, na.rm = TRUE)
pop_mean_lasso_mse <- mean(pop_lasso_mse_vector, na.rm = TRUE)

# Print the result
print(pop_mean_ridge_mse)
print(pop_mean_lasso_mse)
print(pop_mean_linear_mse)
print(pop_mean_glm_mse)

## Very even between models, with glm and linear having a slight edge


print(pop_ridge_rsquared)
print(pop_lasso_rsquared)






###################################### DATA THAT I WILL ANALYZE

##############################################################################
#Setup
rm(list=ls()) 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(plyr)

#################################

library(tensorflow)
library(keras3)
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


df = fread("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/ALLData-MergedDS.csv")
print(colnames(df))

print(df[, c('Finance and insurance _x', 'Finance and insurance _y')])

## NOTE: _x is Contributions to percent change in real GDP, _y is Gross domestic product (GDP) by county and metropolitan area

column_types <- sapply(df, class)
print(column_types)

colnames(df) <- gsub("_x", " % change in real GDP", colnames(df))
colnames(df) <- gsub("_y", " (GDP)", colnames(df))


# Select only the columns that are numeric
numeric_cols <- sapply(df, is.numeric)
numeric_df <- df[, numeric_cols, with = FALSE]
print(numeric_df)
print(numeric_df$`RES npt Coal (tC)`)
print(numeric_df$N10600)

# Calculate per capita values for each response variable
# Remove commas from Population column and convert to numeric
df$Population <- as.numeric(gsub(",", "", df$Population))
per_capita_data <- df


# Optional: Check for any warnings or NA values after conversion
if (any(is.na(df$Population))) {
  warning("There are NA values in the Population column after conversion.")
}


# Select only numeric columns
pop_numeric_cols <- sapply(per_capita_data, is.numeric)
pop_numeric_df <- per_capita_data[, pop_numeric_cols, with = FALSE]

# Identify and remove emissions columns
pop_emissions_pattern <- "(tC)"
pop_emissions_vars <- grep(pop_emissions_pattern, colnames(pop_numeric_df), value = TRUE)
pop_numeric_df_no_emissions <- pop_numeric_df[, !colnames(pop_numeric_df) %in% pop_emissions_vars, with = FALSE]
pop_numeric_df_no_emissions = pop_numeric_df_no_emissions[,!colnames(pop_numeric_df_no_emissions) %in% c('Unnamed: 0', 'FIPS', 'STATEFIPS'), with = FALSE]

for (response_var in pop_emissions_vars) {
  per_capita_data[[paste0(response_var, "_per_capita")]] <- per_capita_data[[response_var]] / df$Population
}

numeric_cols <- sapply(per_capita_data, is.numeric)
numeric_df <- per_capita_data[, numeric_cols, with = FALSE]

print(colnames(pop_numeric_df_no_emissions))
set.seed(123)
ntest = floor(nrow(per_capita_data) / 4)  # 1/3 of data is test
testid = sample(1:nrow(per_capita_data), ntest)  # indices of test obs

pop_testset = per_capita_data[testid, ]
pop_trainset = per_capita_data[-testid, ]
pop_linear_mse = list()
pop_glm_mse = list()
library(car)
# Store column names as a matrix
numcols <- colnames(pop_numeric_df_no_emissions)

# Convert relevant columns to a matrix
new_numcols = numcols
thresholds= c(0.99,0.95,0.9,0.85,0.8)
print(length(new_numcols))
for (t in thresholds) {
  
  numcols_matrix <- as.matrix(pop_trainset[, ..new_numcols, with = FALSE])
  
  # Calculate the correlation matrix
  correlation_matrix <- cor(numcols_matrix, use = "pairwise.complete.obs")
  #print(correlation_matrix)
  
  threshold <- t  # Set your correlation threshold
  
  # Find indices of correlated pairs
  correlated_indices <- which(abs(correlation_matrix) > threshold, arr.ind = TRUE)
  
  # Remove self-correlations (where variable is correlated with itself)
  correlated_pairs <- correlated_indices[correlated_indices[, 1] != correlated_indices[, 2], ]
  
  # Get the names of the correlated variable pairs
  correlated_var_pairs <- data.frame(
    Variable1 = rownames(correlation_matrix)[correlated_pairs[, 1]],
    Variable2 = colnames(correlation_matrix)[correlated_pairs[, 2]],
    Correlation = correlation_matrix[correlated_pairs]
  )
  
  # Count occurrences of each variable in Variable1
  frequency_table <- table(correlated_var_pairs$Variable1)
  
  # Find the variable with the maximum frequency
  most_frequent_variable <- names(frequency_table)[which.max(frequency_table)]
  most_frequent_count <- max(frequency_table)
  # Print the result
  cat("Most frequent variable:", most_frequent_variable, "appears", most_frequent_count, "times.\n")
  
  ##print(length(unique(rownames(frequency_table))))
  #print(unique(rownames(frequency_table)))
  
  limit_freq = 110-100*t
  high_freq <- names(frequency_table[frequency_table > limit_freq])
  print(high_freq)
  
  new_numcols <- new_numcols[!(new_numcols %in% high_freq)]
  print(length(new_numcols))
  
  
}




## After doing this, much better R Squareds.





## Linear Regressions
pop_linear_regression_results <- list()
pop_glm_regression_results <- list()



# Remove leading/trailing whitespace from column names

for (i in 1:length(pop_emissions_vars)) {
  response_var <- pop_emissions_vars[i]
  response_var_per_capita <- paste0(response_var, "_per_capita")
  print(response_var_per_capita)
  
  # Check if response variable exists in trainset
  if (!response_var_per_capita %in% colnames(pop_trainset)) {
    warning(paste("Response variable", response_var_per_capita, "not found in trainset."))
    next
  }
  
  # Create a formula for the regression model using per capita values
  formula <- as.formula(paste("`", response_var_per_capita, "` ~", paste('`', new_numcols, '`', collapse = " + ",sep = ''), sep = ''))
  
  # Fit the regression model
  pop_linear_regression_model <- lm(formula, data = pop_trainset)
  pop_glm_regression_model <- glm(formula, data = pop_trainset)
  
  # Check VIF and handle aliased coefficients
  #vif_values <- vif(pop_linear_regression_model)
  #if (any(is.infinite(vif_values))) {
  #  warning("Aliased coefficients found; consider adjusting the model.")
  #  next  # Skip to the next variable
  #}
  
  # Calculate MSE and store results
  actual = pop_testset[[response_var_per_capita]]
  pop_linear_preds = predict(pop_linear_regression_model, pop_testset)
  pop_linear_mse[[response_var_per_capita]] <- mean((pop_linear_preds - actual)^2)
  pop_glm_preds = predict(pop_glm_regression_model, pop_testset)
  pop_glm_mse[[response_var_per_capita]] <- mean((pop_glm_preds - actual)^2)
  
  # Store the regression model results
  pop_linear_regression_results[[response_var_per_capita]] <- summary(pop_linear_regression_model)
  pop_glm_regression_results[[response_var_per_capita]] <- summary(pop_glm_regression_model)
}


for (result in pop_linear_regression_results) {
  print(result$r.squared)  # Print each regression result
}

for (result in pop_glm_regression_results) {
  print(result)  # Print each regression result
}

# Convert the list to a numeric vector
pop_linear_mse_vector <- unlist(pop_linear_mse)
pop_glm_mse_vector <- unlist(pop_glm_mse)
print(pop_glm_mse)
print(pop_linear_mse)

# Calculate the mean, excluding NA values
pop_mean_linear_mse <- mean(pop_linear_mse_vector, na.rm = TRUE)
pop_mean_glm_mse <- mean(pop_glm_mse_vector, na.rm = TRUE)

# Print the result
print(pop_mean_linear_mse)
print(pop_mean_glm_mse)

## Much easier to look at


for (result in pop_linear_regression_results) {
  print(result$r.squared)  # Print each regression result
}

for (result in pop_glm_regression_results) {
  print(result)  # Print each regression result
}

# Convert the list to a numeric vector
pop_linear_mse_vector <- unlist(pop_linear_mse)
pop_glm_mse_vector <- unlist(pop_glm_mse)
print(pop_glm_mse)
print(pop_linear_mse)

# Calculate the mean, excluding NA values
pop_mean_linear_mse <- mean(pop_linear_mse_vector, na.rm = TRUE)
pop_mean_glm_mse <- mean(pop_glm_mse_vector, na.rm = TRUE)

# Print the result
print(pop_mean_linear_mse)
print(pop_mean_glm_mse)

# Exclude the last entry
pop_linear_mse_vector_no_last <- pop_linear_mse_vector[-length(pop_linear_mse_vector)]
pop_glm_mse_vector_no_last <- pop_glm_mse_vector[-length(pop_glm_mse_vector)]

# Calculate the mean, excluding NA values
pop_mean_linear_mse <- mean(pop_linear_mse_vector_no_last, na.rm = TRUE)
pop_mean_glm_mse <- mean(pop_glm_mse_vector_no_last, na.rm = TRUE)

# Print the results
print(paste("Mean Linear MSE (excluding last entry):", pop_mean_linear_mse))
print(paste("Mean GLM MSE (excluding last entry):", pop_mean_glm_mse))

## Much easier to look at