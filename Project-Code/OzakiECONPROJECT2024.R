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


# Import data:
DATA = numeric(0)
DATA$Description = "all data"
DATA$EmissionsData = fread("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/Base-Datasets/COUNTY_EMISSIONS2021.csv")
DATA$PopulationData = fread("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/Cleaned-Datasets/POPULATIONDATA-Cleaned.csv")

#Troubleshooting
print(setdiff(DATA$PopulationData$County,DATA$EmissionsData$County))
print(setdiff(DATA$EmissionsData$County,DATA$PopulationData$County))
print(dim(DATA$EmissionsData))
print(dim(DATA$PopulationData))


