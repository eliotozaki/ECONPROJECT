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






set.seed(123)
ntest = floor(nrow(data)/3)  # 1/3 of data is test
testid = sample(1:nrow(data), ntest)  # indices of test obs

testset = data[testid]
trainset = data[-testid]

set.seed(123)
x = model.matrix((`Total FFCO2 (tC)`)~ (`Quantity index`)+ (`Thousands of dollars`) + (`Thousands of chained 2012 dollars`) + `Population`, trainset)
y = trainset$`Total FFCO2 (tC)`

############ DOUBLE LASSO (VARIABLE SELECTION)
set.seed(123)
lasso_model = glmnet(x, y, family = "gaussian", alpha = 1)
set.seed(123)
cv_model = cv.glmnet(x, y_numeric, family = "binomial", alpha = 1)
plot(cv_model)
best_lambda = cv_model$lambda.min
best_coefs = coef(cv_model, s = "lambda.min")
relevant_variables = best_coefs[best_coefs != 0]
print(relevant_variables)
best_coefs_dense = as.matrix(best_coefs)
best_coefs_df = data.frame(Coefficient = best_coefs_dense[,1], row.names = rownames(best_coefs_dense))
best_coefs_df$AbsoluteCoefficient = abs(best_coefs_df$Coefficient)
relevant_variables = best_coefs_df[best_coefs_df$Coefficient != 0 & row.names(best_coefs_df) != "(Intercept)",, drop = FALSE]
print(row.names(relevant_variables))
most_influential_vars = best_coefs_df[order(-best_coefs_df$AbsoluteCoefficient), ]
most_influential_vars = most_influential_vars[most_influential_vars$Coefficient != 0 & row.names(most_influential_vars) != "(Intercept)", ]
print(most_influential_vars)
infcols = row.names(most_influential_vars)
infcols.top10 = infcols[1:10]

#Where Y is RESULTADO
#impx = cbind(smallfsub[,infcols],y_factor)
smallfsubdf = model.matrix(RESULTADO~.,smallfsub)
impx = cbind(smallfsubdf[,infcols],y_numeric)
impxfactor = cbind(smallfsubdf[,infcols],y_factor)
impx.top10 = cbind(smallfsubdf[,infcols[1:10]],y_numeric)

############

x_new = model.matrix(RESULTADO~., testset)[,-1]
y_test = testset$RESULTADO

predictions = predict(lasso_model, newx = x_new, s = best_lambda, type = "response")
predicted_classes = ifelse(predictions > 0.5, 1, 0)
predfactors = as.factor(predicted_classes)
predfactors = revalue(predfactors, c("0" = "NEGATIVE", "1" = "POSITIVE"))


conf_matrix = confusionMatrix(predfactors, as.factor(y_test))
print(conf_matrix)