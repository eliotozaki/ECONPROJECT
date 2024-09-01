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




